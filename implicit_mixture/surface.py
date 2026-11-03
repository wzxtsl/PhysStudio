"""Reconstruct mixture surfaces and nested concentration volumes for Cycles."""
import argparse
import json
from pathlib import Path
import time
import numpy as np
from scipy.ndimage import map_coordinates
from scipy.sparse import coo_matrix
import warp as wp

TEA_LEVELS = [0.0002, 0.0005, 0.0012, 0.003, 0.0075, 0.018, 0.045, 0.11, 0.28]


@wp.kernel(enable_backward=False)
def deposit(
    x: wp.array(dtype=wp.vec3),
    a: wp.array(dtype=wp.vec3),
    field: wp.array3d(dtype=wp.vec3),
    origin: wp.vec3,
    ds: float,
    dx: float,
    sigma: float,
    support: float,
):
    p = wp.tid()
    b = wp.vec3i(
        int(wp.floor((x[p][0] - origin[0]) / ds)),
        int(wp.floor((x[p][1] - origin[1]) / ds)),
        int(wp.floor((x[p][2] - origin[2]) / ds)),
    )
    reach = int(wp.ceil(support * dx / ds))
    for i in range(-reach, reach + 1):
        for j in range(-reach, reach + 1):
            for k in range(-reach, reach + 1):
                cell = b + wp.vec3i(i, j, k)
                if (
                    cell[0] >= 0
                    and cell[1] >= 0
                    and cell[2] >= 0
                    and cell[0] < field.shape[0]
                    and cell[1] < field.shape[1]
                    and cell[2] < field.shape[2]
                ):
                    q = (
                        origin
                        + ds * wp.vec3(float(cell[0]), float(cell[1]), float(cell[2]))
                        - x[p]
                    ) / dx
                    if wp.dot(q, q) < support * support:
                        weight = wp.exp(-wp.dot(q, q) / (2.0 * sigma * sigma)) / (
                            wp.pow(2.0 * 3.141592653589793, 1.5) * sigma * sigma * sigma
                        )
                        wp.atomic_add(field, cell[0], cell[1], cell[2], weight * a[p])


@wp.kernel(enable_backward=False)
def levelset(
    field: wp.array3d(dtype=wp.vec3),
    iso: wp.array3d(dtype=float),
    threshold: float,
    tea: int,
    origin: wp.vec3,
    ds: float,
    dx: float,
    wet_radius: float,
):
    i, j, k = wp.tid()
    a = field[i, j, k]
    total = a[0] + a[1] + a[2]
    value = total - 0.48
    if tea == 1:
        concentration = a[1] / wp.max(total, 1.0e-8)
        value = wp.min(value, (concentration - threshold) * 2.0)
    if wet_radius > 0.0:
        p = origin + ds * wp.vec3(float(i), float(j), float(k))
        value = wp.min(value, (wet_radius - wp.sqrt(p[0] * p[0] + p[1] * p[1])) / dx)
        value = wp.min(value, (p[2] + 0.0002) / dx)
    iso[i, j, k] = value


def reflected_support(positions, fractions, radius, support_distance):
    """Mirror fluid samples at the cup wall and floor for a wetted render boundary."""
    r = np.linalg.norm(positions[:, :2], axis=1)
    mask = r > radius - support_distance
    wall = positions[mask].copy()
    wall[:, :2] *= ((2 * radius - r[mask]) / r[mask])[:, None]
    x = np.concatenate([positions, wall])
    a = np.concatenate([fractions, fractions[mask]])
    mask = x[:, 2] < support_distance
    floor = x[mask].copy()
    floor[:, 2] *= -1
    return np.concatenate([x, floor]), np.concatenate([a, a[mask]])


def surface_volume(vertices, faces):
    if not len(faces):
        return 0.0
    tri = vertices[faces].astype(np.float64)
    return float(
        np.einsum("ij,ij->i", tri[:, 0], np.cross(tri[:, 1], tri[:, 2])).sum() / 6.0
    )


def grid_to_world_vertices(vertices, spacing, origin):
    """Match Warp 1.6 cell-center vertices to integer-grid field samples."""
    return (vertices - 0.5) * spacing + origin


def smooth_surface(vertices, faces, passes):
    """Alternating low-pass steps reduce voxel-scale ripples without mesh shrinkage."""
    if not passes or not len(faces):
        return vertices, dict(max_displacement=0.0, relative_volume_change=0.0)
    edges = np.concatenate([faces[:, [0, 1]], faces[:, [1, 2]], faces[:, [2, 0]]])
    edges = np.concatenate([edges, edges[:, ::-1]])
    adjacency = coo_matrix(
        (np.ones(len(edges)), (edges[:, 0], edges[:, 1])),
        shape=(len(vertices), len(vertices)),
    ).tocsr()
    adjacency.data[:] = 1.0
    degree = np.asarray(adjacency.sum(axis=1)).ravel()
    operator = adjacency.multiply(1.0 / np.maximum(degree, 1.0)[:, None]).tocsr()
    result = vertices.astype(np.float64)
    for _ in range(passes):
        result += 0.50 * (operator @ result - result)
        result -= 0.53 * (operator @ result - result)
    before = surface_volume(vertices, faces)
    after = surface_volume(result, faces)
    stats = dict(
        max_displacement=float(np.linalg.norm(result - vertices, axis=1).max()),
        relative_volume_change=float((after - before) / max(abs(before), 1.0e-12)),
        unsmoothed_volume=before,
        smoothed_volume=after,
    )
    return result.astype(np.float32), stats


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--input", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--frames", type=int, nargs="*")
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--scale", type=float)
    p.add_argument(
        "--kernel-width",
        type=float,
        help="Gaussian standard deviation in particle spacings",
    )
    p.add_argument(
        "--smooth-passes", type=int, help="Alternating mesh smoothing passes"
    )
    p.add_argument(
        "--wet-contact",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Reconstruct liquid contact at the cylindrical cup wall and floor",
    )
    args = p.parse_args()
    cfg = json.loads((args.input / "config.json").read_text())
    if args.scale is None:
        args.scale = 0.4 if cfg["scene"] == "collision" else 0.5
    if args.wet_contact is None:
        args.wet_contact = cfg["scene"] == "tea"
    sigma = (
        args.kernel_width
        if args.kernel_width is not None
        else (0.95 if cfg["scene"] == "collision" else 0.7)
    )
    passes = (
        args.smooth_passes
        if args.smooth_passes is not None
        else (6 if cfg["scene"] == "collision" else 0)
    )
    if sigma <= 0 or args.scale <= 0 or passes < 0:
        raise ValueError("Invalid surface reconstruction settings")
    support = 3 * sigma if sigma > 0.7 else 2.0
    if args.wet_contact and not cfg.get("cylindrical"):
        raise ValueError("Wet contact requires a cylindrical cup")
    wet_radius = cfg["bounds"][1][0] + 0.0002 if args.wet_contact else 0.0
    indices = args.frames if args.frames is not None else list(range(cfg["frames"]))
    args.output.mkdir(parents=True, exist_ok=False)
    dx = cfg["dx"]
    ds = dx * args.scale
    lo, hi = np.array(cfg["bounds"], float)
    origin = lo - 3 * dx
    upper = hi + 3 * dx
    shape = tuple((np.ceil((upper - origin) / ds).astype(int) + 1).tolist())
    wp.init()
    field = wp.zeros(shape, dtype=wp.vec3, device=args.device)
    iso = wp.zeros(shape, dtype=float, device=args.device)
    mc = wp.MarchingCubes(
        *shape, max_verts=1800000, max_tris=3000000, device=args.device
    )
    records = []
    start = time.monotonic()
    nf = cfg["fluid_particles"]
    for index in indices:
        state = np.load(args.input / "states" / f"{index:05d}.npz")
        field.zero_()
        positions = state["positions"][:nf]
        fractions = state["fractions"][:nf]
        if args.wet_contact:
            positions, fractions = reflected_support(
                positions, fractions, cfg["bounds"][1][0], support * dx
            )
        x = wp.array(positions, dtype=wp.vec3, device=args.device)
        a = wp.array(fractions, dtype=wp.vec3, device=args.device)
        wp.launch(
            deposit,
            len(positions),
            inputs=[x, a, field, wp.vec3(*origin), ds, dx, sigma, support],
            device=args.device,
        )
        data = {}
        levels = [("fluid", 0.0)] + (
            [(f"tea_{i}", level) for i, level in enumerate(TEA_LEVELS)]
            if cfg["scene"] == "tea"
            else []
        )
        for name, threshold in levels:
            wp.launch(
                levelset,
                shape,
                inputs=[
                    field,
                    iso,
                    threshold,
                    int(name != "fluid"),
                    wp.vec3(*origin),
                    ds,
                    dx,
                    wet_radius,
                ],
                device=args.device,
            )
            mc.surface(iso, 0.0)
            vertices = (
                grid_to_world_vertices(mc.verts.numpy(), ds, origin)
                if mc.verts.size
                else np.empty((0, 3), np.float32)
            )
            faces = (
                mc.indices.numpy().reshape(-1, 3)[:, [0, 2, 1]]
                if mc.indices.size
                else np.empty((0, 3), np.int32)
            )
            data[name + "_vertices"] = vertices.astype(np.float32)
            data[name + "_faces"] = faces.astype(np.int32)
        sampled = field.numpy()
        vertices = data["fluid_vertices"]
        coords = ((vertices - origin) / ds).T
        vertices, quality = smooth_surface(vertices, data["fluid_faces"], passes)
        data["fluid_vertices"] = vertices
        coords = ((vertices - origin) / ds).T
        fractions = np.stack(
            [
                map_coordinates(sampled[..., i], coords, order=1, mode="nearest")
                for i in range(3)
            ],
            axis=-1,
        )
        fractions /= np.maximum(fractions.sum(1, keepdims=True), 1.0e-8)
        data["fractions"] = fractions.astype(np.float32)
        np.savez_compressed(args.output / f"{index:05d}.npz", **data)
        counts = {name: len(data[name + "_faces"]) for name, _ in levels}
        records.append(dict(frame=index, triangles=counts, smoothing=quality))
        if index % 20 == 0 or index == indices[-1]:
            print(
                json.dumps(
                    dict(
                        frame=index, elapsed=time.monotonic() - start, triangles=counts
                    )
                ),
                flush=True,
            )
    (args.output / "manifest.json").write_text(
        json.dumps(
            dict(
                frames=indices,
                spacing=ds,
                kernel_width=sigma,
                support=support,
                vertex_offset_cells=-0.5,
                smooth_passes=passes,
                wet_contact=args.wet_contact,
                tea_levels=TEA_LEVELS,
                surfaces=records,
            ),
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
