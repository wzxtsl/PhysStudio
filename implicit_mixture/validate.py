"""Check SPH operators, phase-pressure mapping, conservative fluxes and tea release."""
import argparse
import json
from pathlib import Path
import numpy as np
import warp as wp
from .solver import Mixture, W, G, pressure_update, advance
from .scenes import configuration, initial_state, motion, lattice
from .surface import grid_to_world_vertices


@wp.kernel(enable_backward=False)
def evaluate_kernel(
    points: wp.array(dtype=wp.vec3),
    h: float,
    values: wp.array(dtype=float),
    grads: wp.array(dtype=wp.vec3),
):
    i = wp.tid()
    values[i] = W(wp.length(points[i]), h)
    grads[i] = G(points[i], h)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--device", default="cuda:0")
    args = p.parse_args()
    wp.init()
    rng = np.random.default_rng(71)
    result = {}
    device = args.device
    # A known plane catches half-cell offsets at the fluid/glass interface.
    size = 8
    plane = 3.25
    spacing = 0.02
    origin = np.array([0.12, -0.36, 0.18])
    field = (
        np.broadcast_to(
            np.arange(size, dtype=np.float32)[:, None, None], (size, size, size)
        ).copy()
        - plane
    )
    mc = wp.MarchingCubes(
        size, size, size, max_verts=10000, max_tris=10000, device=device
    )
    mc.surface(wp.array(field, dtype=float, device=device), 0.0)
    vertices = grid_to_world_vertices(mc.verts.numpy(), spacing, origin)
    mapping_error = float(np.max(abs(vertices[:, 0] - (origin[0] + plane * spacing))))
    assert mapping_error < 1.0e-6, mapping_error
    result["surface_coordinate_mapping"] = {
        "plane_error": mapping_error,
        "passed": True,
    }
    h = 0.2
    pts = np.array([[0.035, 0.02, 0.04], [0.11, 0.02, 0], [0.21, 0, 0]], np.float32)
    eps = 1.0e-4
    positions = np.concatenate([pts, pts + [eps, 0, 0], pts - [eps, 0, 0]])
    x = wp.array(positions, dtype=wp.vec3, device=device)
    w = wp.zeros(len(x), dtype=float, device=device)
    g = wp.zeros(len(x), dtype=wp.vec3, device=device)
    wp.launch(evaluate_kernel, len(x), inputs=[x, h, w, g], device=device)
    error = np.max(abs((w.numpy()[3:6] - w.numpy()[6:]) / (2 * eps) - g.numpy()[:3, 0]))
    assert error < 0.5, error
    result["kernel_gradient_absolute_error"] = float(error)
    cfg = dict(
        dx=0.06,
        densities=[800.0, 1100.0, 1600.0],
        drag=0.35,
        diffusion=0.025,
        viscosity=0.0,
        gravity=[0, 0, 0],
        bounds=[[-2, -2, -2], [2, 2, 2]],
        pressure_iterations=96,
        pressure_tolerance=1.0e-7,
    )
    points = lattice([-0.21, -0.21, -0.21], [0.21, 0.21, 0.21], cfg["dx"]).astype(
        np.float32
    )
    n = len(points)
    alpha = rng.dirichlet([2, 2, 2], n).astype(np.float32)
    s = Mixture(
        points, alpha, np.zeros_like(points), np.zeros(n, np.int32), points, cfg, device
    )
    s.neighbors()
    before = s.diagnostics()
    s.p = wp.array(
        rng.uniform(0.001, 0.01, n).astype(np.float32), dtype=float, device=device
    )
    s.launch(
        pressure_update,
        n,
        [s.u, s.rho, s.ids, s.grad, s.count, s.p, s.density, s.cd, s.nf],
    )
    s.remix()
    velocity = s.u.numpy()
    m = alpha.astype(float) * np.array(cfg["densities"]) * s.vol
    momentum = (m[..., None] * velocity).sum((0, 1))
    angular = np.cross(points, (m[..., None] * velocity).sum(1)).sum(0)
    assert np.linalg.norm(momentum) < 1.0e-6 and np.linalg.norm(angular) < 1.0e-6, (
        momentum,
        angular,
    )
    rho = alpha @ np.array(cfg["densities"])
    gamma = cfg["drag"] + (1 - cfg["drag"]) * rho[:, None] / np.array(cfg["densities"])
    common = velocity / gamma[..., None]
    mapping_error = np.max(abs(common - common[:, 0:1, :]))
    assert mapping_error < 1.0e-7, mapping_error
    result["pressure_mapping"] = {
        "maximum_error": float(mapping_error),
        "net_impulse": float(np.linalg.norm(momentum)),
        "net_torque": float(np.linalg.norm(angular)),
    }
    phases = rng.normal(0, 0.3, (n, 3, 3)).astype(np.float32)
    s.u = wp.array(phases, dtype=wp.mat33, device=device)
    s.remix()
    s.neighbors()
    before = s.diagnostics()
    s.transfer(0.08)
    after = s.diagnostics()
    mass = np.max(abs(np.array(after["phase_mass"]) - before["phase_mass"]))
    momentum = np.linalg.norm(np.array(after["momentum"]) - before["momentum"])
    assert after["phase_min"] >= -1.0e-7 and after["partition_error"] < 2.0e-6, after
    assert mass < 1.0e-5 and momentum < 1.0e-5, (mass, momentum)
    assert s.limit.numpy().min() < 0.95
    result["limited_transport"] = {
        "mass_error": float(mass),
        "momentum_error": float(momentum),
        "partition_error": after["partition_error"],
        "minimum_fraction": after["phase_min"],
    }
    # The fully coupled limit must preserve identical phase velocities.
    s.cd = 1.0
    s.u = wp.array(
        np.repeat((-0.3 * points)[:, None, :], 3, axis=1), dtype=wp.mat33, device=device
    )
    s.remix()
    s.neighbors()
    s.project(0.01, 0)
    coupled = s.u.numpy()
    assert np.max(abs(coupled - coupled[:, 0:1, :])) < 1.0e-7
    assert s.last_projection["0"]["mean"] < 2.0e-6, s.last_projection
    result["fully_coupled_projection"] = s.last_projection["0"]
    cfg = configuration("tea", 0.06)
    data = initial_state(cfg)
    s = Mixture(*data, cfg, device=device)
    s.set_kinematics(motion(cfg, 0))
    s.neighbors()
    before = s.diagnostics()
    s.transfer(0.01)
    after = s.diagnostics()
    released = after["fluid_phase_mass"][1] - before["fluid_phase_mass"][1]
    assert released > 0, released
    assert np.max(abs(np.array(before["phase_mass"]) - after["phase_mass"])) < 2.0e-5
    result["tea_bag_diffusion"] = {
        "released_tea_mass": released,
        "total_phase_mass_error": float(
            np.max(abs(np.array(before["phase_mass"]) - after["phase_mass"]))
        ),
    }
    # Centerline and end-cap contacts must have a defined escape normal.
    points = np.array(
        [
            [0, 0, 0.5],
            [0.02, 0, 0.695],
            [-0.5, 0, 0.5],
            [0.99, 0, 0.3],
            [0.4, 0, 0.001],
        ],
        np.float32,
    )
    x = wp.array(points, dtype=wp.vec3, device=device)
    u = wp.zeros(5, dtype=wp.mat33, device=device)
    v = wp.zeros(5, dtype=wp.vec3, device=device)
    contact = wp.zeros(1, dtype=float, device=device)
    wp.launch(
        advance,
        5,
        inputs=[
            x,
            u,
            v,
            0.01,
            wp.vec3(-1, -1, 0),
            wp.vec3(1, 1, 1),
            0.1,
            0,
            wp.vec3(-0.5, 0, 0.5),
            wp.vec3(0.12),
            wp.vec3(0),
            wp.vec3(0, 0, 0.5),
            wp.vec3(0),
            0.1,
            0.2,
            1,
            contact,
        ],
        device=device,
    )
    projected = x.numpy()
    assert np.linalg.norm(projected[0, :2]) >= 0.1449
    assert projected[1, 2] >= 0.7449
    assert np.max(abs(projected[2] - points[2])) >= 0.1649
    assert projected[3, 0] <= 0.9551 and projected[4, 2] >= 0.0449
    result["solid_contacts"] = {
        "centerline": projected[0].tolist(),
        "rod_cap": projected[1].tolist(),
        "passed": True,
    }
    result["passed"] = True
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
