# Thin-Film Bubbles

[A Moving Eulerian-Lagrangian Particle Method for Thin Film and Foam Simulation](https://yitongdeng.github.io/MELP_Project.github.io/)
uses separate geometric and material particles to combine surface flow with
large deformation. This independent Warp implementation provides three 3D
thin-film scenes, rendered from saved states with Blender Cycles.

| Demo | Preset | Motion | Physical duration |
| --- | --- | --- | --- |
| Giant Bubble Recoil | `giant_recoil` | An elongated film compresses and rebounds under surface tension | 0.6 s |
| Breathing Bubble | `driven_bubble` | A rotating pressure field stretches and releases a bubble | 1.2 s |
| Bubble Collision | `bubble_collision` | Two elongated films compress at contact and rebound off center | 0.6 s |

Each recording contains 241 frames, played at 30 fps for 8.034 seconds. Each
bubble uses 1,622 geometric and 26,000 material particles. Warp runs the
float64 physics and conjugate-gradient solves; the render mesh does not drive
the simulation. Geometry, materials, and lighting are procedural.

The collision preset models intact films with a unilateral contact penalty.
Shared-wall foam, rupture, and cross-region material exchange are not included.
The presets use reduced ambient pressure and effective normal added mass;
[METHOD.md](METHOD.md) describes the formulation and numerical adaptations.

## Install and run

Use the shared [installation](../INSTALL.md), an NVIDIA GPU, Blender 4.2.9
with Cycles/OptiX, and `ffmpeg`/`ffprobe` on `PATH`. The solver and reconstruction
dependencies are in [requirements.txt](requirements.txt); they are already part
of the shared environment. No downloaded assets or author code are required.

```bash
export BLENDER=/path/to/blender
python -m physstudio run thin_film giant_recoil --gpu 0
python -m physstudio run thin_film driven_bubble --gpu 0
python -m physstudio run thin_film bubble_collision --gpu 0
```

Each command simulates, audits the recording, reconstructs and checks the
surface meshes, renders at 1280 x 720 with 48 samples, then encodes and fully
decodes an H.264 video. Choose a fresh output directory for each changed run.

```bash
python -m physstudio run thin_film giant_recoil --gpu 0 \
  --frames 3 --width 320 --output outputs/recoil_check
python -m physstudio run thin_film bubble_collision --dry-run
```

`--frames` includes frame zero and must be at least two. It records a shorter
prefix without changing particle resolution, physics steps, frame intervals,
or the full camera timeline. Recoil/collision use 0.00025-second substeps and
0.0025-second saved intervals; breathing uses 0.0005 and 0.005 seconds.
Playback is approximately 13.33x or 6.67x slow motion, respectively.

Default outputs are `outputs/thin_film/<preset>/`:

```text
simulation/   Config, source hashes, particle states, diagnostics, completion record
audit.json    Full recording checks
mesh/         Reconstructed surfaces and optical film thickness
mesh_audit.json
render/       Numbered PNGs and render completion reports
render_settings.json
<preset>.mp4
<preset>.jpg
<preset>.png
<preset>.storyboard.jpg
<preset>.verification.json
```

Stage logs and `run.json` accompany the pipeline. Bubble Collision reconstructs
optical thickness from conserved fine-particle liquid volume and surface number
density; the other presets use their saved advected thickness.

## Stages and replay

The harness exposes `simulate`, `validate`, `render`, and `encode`; the default
is `all`. The render stage includes reconstruction and mesh validation.

```bash
python -m melp_thin_film.harness --stage simulate --scene bubble_collision \
  --output outputs/collision_run --gpu 0
python -m melp_thin_film.harness --stage validate --scene bubble_collision \
  --output outputs/collision_run
python -m melp_thin_film.harness --stage render --scene bubble_collision \
  --output outputs/collision_run --gpu 0
python -m melp_thin_film.harness --stage encode --scene bubble_collision \
  --output outputs/collision_run
```

Add `--resume` to continue interrupted rendering with the same settings.
`--start` and exclusive `--end` select global frame ranges without changing
camera time. Use a fresh output for visual changes while reusing a recording:

```bash
python -m melp_thin_film.harness --stage render --scene giant_recoil \
  --simulation outputs/recoil_run/simulation --output outputs/recoil_replay \
  --width 1920 --height 1080 --samples 64 --gpu 0
python -m melp_thin_film.harness --stage encode --scene giant_recoil \
  --simulation outputs/recoil_run/simulation --output outputs/recoil_replay
```

Simulation accepts `--settings overrides.json`. The accepted parameter sets
are in [scenes.py](scenes.py). Keep them unchanged to reproduce the showcase.

## Checks and references

```bash
CUDA_VISIBLE_DEVICES=0 python -m melp_thin_film.validate \
  --output outputs/thin_film_checks.json
CUDA_VISIBLE_DEVICES=0 python -m melp_thin_film.validate_contact \
  --output outputs/thin_film_contact_checks.json
CUDA_VISIBLE_DEVICES=0 python -m melp_thin_film.validate_continuous_contact \
  --output outputs/thin_film_continuous_contact_checks.json
python -m melp_thin_film.audit --simulation outputs/recoil_run/simulation \
  --output outputs/recoil_audit.json
```

[Validation results](VALIDATION.md) · [Paper and source metadata](references/sources.json) ·
[Video gallery](../media/index.html#thin-film-bubbles)

Original code is licensed under [Apache 2.0](../LICENSE).
