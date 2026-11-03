# Tree & Root Growth

[Rhizomorph: The Coordinated Function of Shoots and Roots](https://storage.googleapis.com/pirk.io/projects/rhizomorph/index.html)
couples soil resources, plant signals, and branching to grow roots and crowns
together. Warp handles transport, sensing, resource allocation, and elongation;
Blender renders the developing trees as moving-camera soil cutaways.

## Run

Follow the shared [installation guide](../INSTALL.md), install Blender 4.2.9,
and run from the repository root:

```bash
export BLENDER=/path/to/blender
python -m physstudio run rhizomorph oak_taproot --gpu 0
python -m physstudio run rhizomorph maple_heart --gpu 0
python -m physstudio run rhizomorph birch_lateral --gpu 0
```

All trees, foliage, rocks, and soil are procedural. No asset download is needed.

| Preset | Crown | Root environment |
| :--- | :--- | :--- |
| `oak_taproot` | Lobed oak foliage | Deep primary root and branching laterals |
| `maple_heart` | Broad maple foliage | Impermeable rocks and nutrient pockets |
| `birch_lateral` | Pale bark and light foliage | Shallow laterals, sinkers, and resistant soil |

Each preset runs 180 developmental steps and produces an eight-second,
960 × 1080 video at 30 fps. `--frames` changes the rendered frame count while
retaining the full growth trajectory; `--width` scales the portrait resolution.
Use `--output` to choose a fresh run directory.

```text
outputs/rhizomorph/<scene>/
  simulation/        Growth states, soil fields, scene settings, and resource ledger
  audit/             Numerical checks and trajectory audit
  render/frames/     Rendered PNG sequence
  <scene>.mp4        H.264 video
```

## Controls and replay

For a short end-to-end check:

```bash
python -m rhizomorph.harness --scene oak_taproot --gpu 0 \
  --output outputs/rhizomorph/quick_check \
  --steps 12 --frames 8 --width 480 --samples 12
```

The harness exposes `--stage simulate|audit|render|encode` to run individual
stages in the same output directory. To render saved growth states into a new
directory:

```bash
"$BLENDER" -b --factory-startup --python-exit-code 1 \
  --python rhizomorph/render_blender.py -- \
  --input outputs/rhizomorph/oak_taproot/simulation \
  --output outputs/rhizomorph/oak_replay \
  --frames 240 --width 960 --height 1080 --samples 24
```

Presets are defined in `scenes.py`. Pass `--settings custom.json` to the harness
to override fields such as `water_scale`, `light_scale`, `rain`, `patches`,
`rocks`, and `hardpan`. The root and shoot branching parameters control the
botanical appearance; soil transport and developmental growth use separate clocks.

## Model and checks

The soil uses conservative finite-volume water and nutrient transport, surface
rainfall, bottom drainage, and impermeable rock boundaries. Root uptake and
canopy illumination drive shared vigor, allocated hierarchically to competing
meristems. Local resources, tropism, and swept root/rock contact checks guide
elongation. New branches extend from their parents and leaves unfold from their
recorded birth times.

```bash
CUDA_VISIBLE_DEVICES=0 python -m rhizomorph.validate \
  --input outputs/rhizomorph/oak_taproot/simulation \
  --output outputs/rhizomorph/checks --controls
```

Checks cover water and nutrient conservation, nonnegativity, sealed cells,
nutrient-gradient response, graph connectivity, root/rock clearance, and resource
allocation. Matched dry and shaded controls use the same seed and genotype.
See [METHOD.md](METHOD.md) for equations and numerical adaptations.

## References

- Bosheng Li et al. *Rhizomorph: The Coordinated Function of Shoots and Roots*.
  ACM Transactions on Graphics, 2023. [Paper](https://doi.org/10.1145/3592145).
- [Official implementation](https://github.com/edisonlee0212/rhizomorph),
  revision `a4d2835c8527b9f92895e1cca4a9f30211e97a40`.
  [Reference metadata](references/sources.json) · [Reference license](LICENSE.reference).
