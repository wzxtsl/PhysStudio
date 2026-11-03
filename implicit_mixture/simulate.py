"""Cache physical mixture states and conservation diagnostics for rendering."""
import argparse
import hashlib
import json
import shutil
from pathlib import Path
import time
import numpy as np
from .scenes import configuration, initial_state, motion, PRESETS
from .solver import Mixture


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--scene", choices=PRESETS, default="tea")
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--frames", type=int)
    p.add_argument("--dx", type=float)
    p.add_argument("--dt", type=float)
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--config", type=Path)
    args = p.parse_args()
    cfg = configuration(args.scene, args.dx)
    if args.config:
        cfg.update(json.loads(args.config.read_text()))
    if args.dt:
        cfg["dt"] = args.dt
    cfg["frames"] = (
        args.frames
        if args.frames is not None
        else round(cfg["seconds"] * cfg["fps"]) + 1
    )
    if cfg["frames"] < 2:
        raise ValueError("At least two saved frames are required")
    args.output.mkdir(parents=True, exist_ok=False)
    (args.output / "states").mkdir()
    cfg["source_hashes"] = {
        f.name: hashlib.sha256(f.read_bytes()).hexdigest()
        for f in Path(__file__).parent.glob("*.py")
    }
    (args.output / "source").mkdir()
    for f in Path(__file__).parent.glob("*.py"):
        shutil.copy2(f, args.output / "source" / f.name)
    state = initial_state(cfg)
    solver = Mixture(*state, cfg, device=args.device)
    cfg.update(
        fluid_particles=solver.nf,
        bag_particles=solver.na - solver.nf,
        boundary_particles=solver.n - solver.na,
    )
    (args.output / "config.json").write_text(json.dumps(cfg, indent=2))
    np.savez_compressed(
        args.output / "boundary.npz", kinds=state[3], local_positions=state[4]
    )
    t = 0.0
    start = time.monotonic()
    records = []
    initial = None
    for frame in range(cfg["frames"]):
        goal = frame / cfg["fps"]
        while t < goal - 1.0e-9:
            dt = solver.timestep(goal - t)
            solver.step(dt, motion(cfg, t + dt / 2))
            t += dt
        solver.set_kinematics(motion(cfg, t))
        if frame == 0:
            solver.neighbors()
        data = solver.snapshot()
        data.update(
            {
                key: np.asarray(value, dtype=np.float64)
                for key, value in motion(cfg, t).items()
            }
        )
        if not all(np.isfinite(v).all() for v in data.values()):
            raise RuntimeError("Non-finite simulation state")
        np.savez_compressed(args.output / "states" / f"{frame:05d}.npz", **data)
        d = solver.diagnostics()
        d.update(frame=frame, time=t, elapsed=time.monotonic() - start)
        if initial is None:
            initial = np.array(d["phase_mass"])
        error = np.array(d["phase_mass"]) - initial
        d["mass_error"] = error.tolist()
        d["relative_mass_error"] = float(abs(error).max() / max(initial.max(), 1.0e-12))
        records.append(d)
        if (
            d["phase_min"] < -0.00001
            or d["partition_error"] > 0.0005
            or d["relative_mass_error"] > 0.0005
        ):
            (args.output / "failure.json").write_text(json.dumps(d, indent=2))
            raise RuntimeError("Phase conservation check failed")
        (args.output / "diagnostics.json").write_text(json.dumps(records, indent=2))
        if frame % 5 == 0 or frame == cfg["frames"] - 1:
            print(json.dumps(d), flush=True)
    print(str(args.output.resolve()), flush=True)


if __name__ == "__main__":
    main()
