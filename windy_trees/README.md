# Windy Trees

[Windy Trees: Computing Stress Response for Developmental Tree Models](https://storage.googleapis.com/pirk.io/projects/windy_trees/index.html)
couples turbulent wind and elastic trees. Warp simulates SPH air, articulated
branches and individual leaf hinges; Blender renders the trees and fine neutral
filaments following the recorded air particles.

## Run

Follow the shared [installation guide](../INSTALL.md), install Blender 4.2.9,
and run from the repository root:

```bash
export BLENDER=/path/to/blender
python -m physstudio run windy_trees oak_taproot --gpu 0
python -m physstudio run windy_trees maple_heart --gpu 0
python -m physstudio run windy_trees birch_lateral --gpu 0
```

The pipeline generates a mature tree with the packaged [Rhizomorph](../rhizomorph/README.md)
model, simulates ten seconds of wind, validates the result, renders, and encodes
an MP4. Geometry is procedural; no asset download is required.

| Preset | Wind | Nominal speed |
| :--- | :--- | :--- |
| `oak_taproot` | Passing gusts | 8.5 m/s |
| `maple_heart` | Turning wind through a dense crown | 7.0 m/s |
| `birch_lateral` | Turbulent wind and slender branches | 9.0 m/s |

Each full run saves 301 states and produces a 1280 × 720 video at 30 fps.
The simulation uses 15,732 air particles, 240 steps per second, and 60 implicit
iterations per step. Of those particles, 5,244 persistent IDs are recorded for
flow visualization. The default renderer uses 16 samples and a moving camera.

```text
outputs/windy_trees/<scene>/
  growth/            Completed 180-step Rhizomorph source tree
  simulation/        Tree/air trajectories, source geometry, settings, and audits
  validation/        Mechanics, oscillator, flow and crown-feedback checks
  render/frames/     Consecutive PNGs, with render settings in render/render.json
  <scene>.mp4        H.264 video
```

Use `--output` for a fresh destination. A short pipeline check retains the full
source tree and normal physics resolution, while shortening the wind trajectory:

```bash
python -m physstudio run windy_trees oak_taproot --gpu 0 \
  --frames 2 --width 320 --output outputs/windy_trees/oak_check
```

`--frames` includes the initial state and must be at least two. It changes the
wind duration to `(frames - 1) / 30` seconds without changing the time step,
air resolution, or source-tree development. `--width` preserves the 16:9 framing
with an even image height. A two-frame check precedes substantial wind bending;
use the full run to view the interaction.

## Stages and replay

The method harness exposes `--stage grow|simulate|validate|render|encode`.
Run individual stages in the same output directory, or reuse a completed tree:

```bash
python -m windy_trees.harness --scene oak_taproot --gpu 0 \
  --source outputs/rhizomorph/oak_taproot/simulation \
  --output outputs/windy_trees/oak_reuse
```

Render a saved wind trajectory into a fresh folder, then encode it:

```bash
python -m windy_trees.harness --scene oak_taproot --gpu 0 --stage render \
  --simulation outputs/windy_trees/oak_taproot/simulation \
  --output outputs/windy_trees/oak_replay --flow streaks --samples 16
python -m windy_trees.harness --scene oak_taproot --stage encode \
  --simulation outputs/windy_trees/oak_taproot/simulation \
  --output outputs/windy_trees/oak_replay
```

`--flow streaks|pathlines|off` selects neutral filaments, the earlier sparse
speed-colored paths, or an unobstructed tree. `--camera orbit|static`, `--width`,
`--height`, `--samples`, and `--blender` control rendering. `--settings custom.json`
overrides simulation parameters from `scenes.py`; use a fresh run directory.
The Blender script also accepts `--indices` for selected global frame numbers,
retaining the complete trajectory's camera timing and particle history.

## Checks and references

```bash
CUDA_VISIBLE_DEVICES=0 python -m windy_trees.validate \
  --simulations outputs/windy_trees/oak_taproot/simulation \
  --output outputs/windy_trees/checks
python -m windy_trees.validate_flow \
  --simulations outputs/windy_trees/oak_taproot/simulation \
  --output outputs/windy_trees/flow_checks
```

Checks compare the implicit mechanics with independent dense and analytic
references, verify path endpoints and boundary recycling, and compare crown
feedback on and off under matched inflow. Simulation audits check fixed roots,
segment lengths, finite states, and tree/air force balance.

See [METHOD.md](METHOD.md) for equations and adaptations,
[FLOW_VISUALIZATION.md](FLOW_VISUALIZATION.md) for trajectory rendering, and
[reference metadata](references/sources.json) for paper and video sources.
The topology remains fixed during these demos; growth adaptation and fracture
are outside the implemented short-term dynamics.
