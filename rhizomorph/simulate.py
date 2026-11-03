"""Run the coupled Warp soil and plant model, retaining every development state."""
import argparse, hashlib, json, time
from pathlib import Path
import numpy as np
import warp as wp
from .scenes import scene_config, SCENES
from .soil import Soil
from .growth import Plant


def run(cfg, output, device):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    (output / "states").mkdir()
    (output / "config.json").write_text(json.dumps(cfg, indent=2))
    sources = {
        p.name: hashlib.sha256(p.read_bytes()).hexdigest()
        for p in Path(__file__).parent.glob("*.py")
    }
    (output / "source_hashes.json").write_text(json.dumps(sources, indent=2))
    wp.init()
    soil = Soil(cfg, device)
    plant = Plant(cfg, soil)
    plant.update_radius(0)
    plant.save(output / "states/00000.npz", 0)
    np.savez_compressed(
        output / "soil_initial.npz",
        water=soil.water.numpy().reshape(soil.nz, soil.ny, soil.nx),
        nutrient=soil.nutrients.numpy().reshape(soil.nz, soil.ny, soil.nx),
        **soil.static,
    )
    start = time.monotonic()
    with (output / "growth.jsonl").open("w") as log:
        for step in range(1, cfg["steps"] + 1):
            info = plant.step(step)
            plant.save(output / "states" / f"{step:05d}.npz", step)
            log.write(json.dumps(info) + "\n")
            log.flush()
            if step % 20 == 0:
                print(
                    json.dumps(dict(**info, elapsed=time.monotonic() - start)),
                    flush=True,
                )
    np.savez_compressed(
        output / "soil_final.npz",
        water=soil.water.numpy().reshape(soil.nz, soil.ny, soil.nx),
        nutrient=soil.nutrients.numpy().reshape(soil.nz, soil.ny, soil.nx),
        **soil.static,
    )
    (output / "audit.json").write_text(
        json.dumps(
            dict(
                **soil.audit(),
                elapsed=time.monotonic() - start,
                nodes=len(plant.nodes),
                device=str(device),
            ),
            indent=2,
        )
    )
    return plant


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--scene", choices=SCENES, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--steps", type=int)
    p.add_argument("--settings", type=Path)
    args = p.parse_args()
    cfg = scene_config(args.scene)
    if args.settings:
        cfg.update(json.loads(args.settings.read_text()))
    if args.steps is not None:
        cfg["steps"] = args.steps
    run(cfg, args.output, args.device)


if __name__ == "__main__":
    main()
