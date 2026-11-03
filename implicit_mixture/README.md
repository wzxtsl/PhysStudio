# <img src="../media/project-icons/multi-fluid.svg" width="40" height="40" align="absmiddle" alt=""> Multi-fluid

[An Implicitly Stable Mixture Model for Dynamic Multi-fluid Simulations](https://doi.org/10.1145/3610548.3618215)
(SIGGRAPH Asia 2023) models coupled phase velocities, incompressibility, and
composition transport. The Warp solver drives three 3D scenes, rendered with
Blender Cycles.

| Demo | Scene | Motion | Video |
| :--- | :--- | :--- | :--- |
| Tea brewing | `tea` | Diffusion from a tea bag, lifting, and stirring | [10 seconds](../media/implicit_mixture/tea_brewing.mp4) |
| Density inversion | `separation` | A heavy fluid descends through two lighter phases | [5 seconds](../media/implicit_mixture/density_inversion.mp4) |
| Chromatic collision | `collision` | Two colored liquid blocks collide, splash, and mix | [5 seconds](../media/implicit_mixture/chromatic_collision.mp4) |

## Install and run

Use Python 3.10+, NVIDIA Warp 1.6.2, [Blender 4.2.9](https://download.blender.org/release/Blender4.2/),
and FFmpeg. Run from the repository root:

```bash
python -m pip install -r implicit_mixture/requirements.txt
export BLENDER=/path/to/blender

python -m physstudio run mixture tea --gpu 0
python -m physstudio run mixture separation --gpu 0
python -m physstudio run mixture collision --gpu 0
```

All geometry is procedural. Each run writes numerical checks, simulated states,
phase diagnostics, reconstructed meshes, PNGs, and `render/demo.mp4` under
`outputs/mixture/<scene>/`. The launcher uses `BLENDER`, `FFMPEG`, and `FFPROBE`
from the environment, defaulting to executable names on PATH.

The presets use 24 fps: tea has 241 frames at particle spacing `0.0225`,
separation has 121 frames at `0.032`, and collision has 121 frames at `0.018`.
A short check retains the normal simulation interval and particle resolution:

```bash
python -m physstudio run mixture tea --frames 2 --width 320 \
  --output outputs/mixture/tea_check --gpu 0
```

## Customize and replay

The harness exposes particle spacing, rendering quality, and surface settings:

```bash
python -m implicit_mixture.harness all --scene tea --gpu 0 \
  --output outputs/mixture/custom_tea --dx 0.0225 \
  --width 1280 --samples 48 --tea-absorption 270 --tea-color 0.78 0.50 0.10
```

Use `--config settings.json` for simulation overrides. The stirring control uses
`--config implicit_mixture/configs/tea_no_stir.json`.

Stages can run separately: `validate`, `simulate`, `audit`, `mesh`, `render`,
and `encode`. For example, render and encode an existing simulation and mesh cache:

```bash
python -m implicit_mixture.harness render --scene tea --gpu 0 \
  --output outputs/mixture/custom_tea
python -m implicit_mixture.harness encode --scene tea \
  --output outputs/mixture/custom_tea
```

Add `--resume-render` to continue rendering with the same settings. Use a fresh
output directory for a new configuration. `--surface-scale`, `--kernel-width`,
and `--smooth-passes` control liquid-surface reconstruction. Collision uses a
wider kernel and alternating smoothing; tea uses reflected cup-wall support
and amber volume absorption. The harness verifies PNGs and the encoded video.

## Method and checks

The solver maps pressure impulses to phase velocities, conserves phase mass
during drift and diffusion, and couples the moving tea bag and stirrer to the
liquid. Surface extraction accounts for Warp's cell-center vertex convention.
[METHOD.md](METHOD.md) describes the equations, boundaries, and rendering.

```bash
python -m implicit_mixture.validate --output outputs/mixture/numerics.json
python -m implicit_mixture.audit outputs/mixture/tea/simulation
```

Checks cover kernel derivatives, pressure mapping, impulse and torque balance,
conservative transport, finite tea release, solid contacts, and surface coordinates.
For mixing plots, install Matplotlib and run `python -m implicit_mixture.report --help`.

[Paper manuscript](https://eprints.bournemouth.ac.uk/40107/1/Wb4Nx-papers_671s4-file1.s.pdf)
· [Authors' Taichi implementation](https://github.com/sakamotoyan/TiSPH_multiphase)
· [Pinned references](references/sources.json)
