"""Numerical invariants and matched environmental controls for plant development."""
import argparse, json
from pathlib import Path
import numpy as np
import warp as wp
from .scenes import scene_config
from .soil import Soil
from .simulate import run
from . import kernels as k


def inspect(path):
    path = Path(path)
    cfg = json.loads((path / "config.json").read_text())
    data = np.load(path / "states" / f"{cfg['steps']:05d}.npz")
    p = data["p"]
    parents = data["parent"]
    root = data["kind"] == 1
    shoot = ~root
    assert np.isfinite(p).all() and np.isfinite(data["radius"]).all()
    assert np.all(parents < np.arange(len(p))) and np.all(parents >= -1)
    assert (data["radius"] >= 0).all()
    assert np.max(p[root, 2]) <= 0.0 and np.min(p[shoot, 2]) >= -0.001
    minimum = 1.0e9
    ids = np.where(root & (parents >= 0))[0]
    for rock in cfg["rocks"]:
        r = np.array(rock["radii"])
        center = np.array(rock["center"])
        for t in np.linspace(0, 1, 17):
            q = p[parents[ids]] * (1 - t) + p[ids] * t
            d = (np.linalg.norm((q - center) / r, axis=1) - 1.0) * min(r)
            clearance = d - data["radius"][ids]
            minimum = min(minimum, float(clearance.min()))
    assert minimum >= -0.002, ("Rock penetration", minimum)
    length = np.zeros(len(p))
    valid = parents >= 0
    length[valid] = np.linalg.norm(p[valid] - p[parents[valid]], axis=1)
    audit = json.loads((path / "audit.json").read_text())
    assert audit["water_min"] >= -1.0e-7 and audit["nutrient_min"] >= -1.0e-7
    assert abs(audit["water_balance_relative_error"]) < 3.0e-4
    assert abs(audit["nutrient_balance_relative_error"]) < 3.0e-4
    records = [
        json.loads(line) for line in (path / "growth.jsonl").read_text().splitlines()
    ]
    for row in records:
        for system in ["shoot", "root"]:
            if system + "_allocated" in row:
                assert (
                    row[system + "_allocated"]
                    <= row[system + "_budget"] * (1.0 + 2.0e-5) + 1.0e-5
                )
    return dict(
        shoot_nodes=int(shoot.sum()),
        root_nodes=int(root.sum()),
        height=float(p[shoot, 2].max()),
        depth=float(-p[root, 2].min()),
        shoot_length=float(length[shoot].sum()),
        root_length=float(length[root].sum()),
        root_width=float(np.ptp(p[root, 0])),
        root_center=p[root].mean(0).tolist(),
        minimum_rock_clearance=None if minimum == 1.0e9 else minimum,
        **audit,
    )


def transport_tests(device):
    cfg = scene_config("oak_taproot")
    cfg.update(grid=[18, 18, 12], gravity=0.0, rain=0.0, patches=[], soil_substeps=5)
    soil = Soil(cfg, device)
    start = soil.water.numpy().copy()
    for _ in range(25):
        soil.transport(rain=0.0, drain=0.0)
    assert np.max(np.abs(soil.water.numpy() - start)) < 1.0e-6
    equilibrium = float(np.max(np.abs(soil.water.numpy() - start)))
    rng = np.random.default_rng(31)
    w = rng.uniform(0.01, 1.2, soil.count).astype(np.float32)
    nu = rng.uniform(0.05, 1.5, soil.count).astype(np.float32)
    wp.copy(soil.water, wp.array(w, device=device))
    wp.copy(soil.nutrients, wp.array(nu, device=device))
    for _ in range(60):
        soil.transport(rain=0.0, drain=0.0)
    ww = soil.water.numpy()
    nn = soil.nutrients.numpy()
    ew = abs(float(ww.sum(dtype=np.float64) - w.sum(dtype=np.float64))) / float(w.sum())
    en = abs(float(nn.sum(dtype=np.float64) - nu.sum(dtype=np.float64))) / float(
        nu.sum()
    )
    assert ew < 2.0e-6 and en < 2.0e-6 and ww.min() >= 0 and nn.min() >= 0
    # A sealed cell retains its contents even under strong neighbor gradients.
    perm = soil.permeability.numpy()
    perm[soil.count // 2] = 0.0
    wp.copy(soil.permeability, wp.array(perm, device=device))
    before = soil.water.numpy()[soil.count // 2]
    for _ in range(10):
        soil.transport(rain=0.0, drain=0.0)
    wall = float(abs(soil.water.numpy()[soil.count // 2] - before))
    assert wall < 1.0e-7
    return dict(
        equilibrium_error=equilibrium,
        closed_water_error=ew,
        closed_nutrient_error=en,
        impermeable_cell_error=wall,
        nonnegative=True,
    )


def nutrient_tropism_check(device):
    cfg = scene_config("oak_taproot")
    soil = Soil(cfg, device)
    n = 256
    array = lambda value, dtype: wp.array(value, dtype=dtype, device=device)
    p = array(np.tile([0.0, 0.0, -1.0], (n, 1)), wp.vec3)
    heading = array(np.tile([0.0, 0.0, -1.0], (n, 1)), wp.vec3)
    one = array(np.ones(n), float)
    ones = array(np.ones(n), int)
    zero = array(np.zeros(n), int)
    parent = array(np.full(n, -1), int)
    optimal = array(np.tile([0.0, 0.0, -1.0], (n, 1)), wp.vec3)
    results = []
    for sign in [1, -1]:
        nu = (1.6 + sign * 0.35 * soil.xyz[..., 0]).astype(np.float32).ravel()
        wp.copy(soil.nutrients, array(nu, float))
        out = wp.empty(n, dtype=wp.vec3, device=device)
        direction = wp.empty_like(out)
        allocated = wp.empty(n, dtype=float, device=device)
        wp.launch(
            k.elongate,
            n,
            [
                p,
                heading,
                ones,
                zero,
                ones,
                parent,
                one,
                one,
                one,
                optimal,
                soil.nutrients,
                soil.resistance,
                soil.sdf,
                *soil.geometry,
                31,
                12,
                1,
                0.047,
                0.058,
                1.0,
                1.0,
                out,
                direction,
                allocated,
            ],
            device=device,
        )
        results.append(out.numpy())
    difference = results[0][:, 0] - results[1][:, 0]
    assert difference.mean() > 0.002 and np.all(difference > 0)
    return dict(
        mean_x_shift_toward_reversed_gradient_m=float(difference.mean()),
        min_x_shift_m=float(difference.min()),
        paired_tips=n,
        all_tips_respond_in_correct_direction=bool(np.all(difference > 0)),
    )


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--input", type=Path, nargs="*", default=[])
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--controls", action="store_true")
    args = p.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    wp.init()
    report = dict(
        transport=transport_tests(args.device),
        nutrient_tropism=nutrient_tropism_check(args.device),
        scenes={},
    )
    for path in args.input:
        report["scenes"][path.name] = inspect(path)
    if args.controls:
        controls = {}
        for name, changes in [
            ("balanced", {}),
            ("dry", dict(water_scale=0.04)),
            ("shade", dict(light_scale=0.10)),
        ]:
            cfg = scene_config("oak_taproot")
            cfg.update(steps=100, **changes)
            path = args.output / name
            run(cfg, path, args.device)
            controls[name] = inspect(path)
        for treatment in ["dry", "shade"]:
            assert (
                controls[treatment]["shoot_length"]
                < 0.85 * controls["balanced"]["shoot_length"]
            )
            assert (
                controls[treatment]["root_length"]
                < 0.90 * controls["balanced"]["root_length"]
            )
        report["controls"] = controls
    (args.output / "validation.json").write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
