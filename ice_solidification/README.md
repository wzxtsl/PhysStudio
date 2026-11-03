# <img src="../media/project-icons/ice-solidification.svg" width="40" height="40" align="absmiddle" alt=""> Ice Solidification

[Solidification using smoothed particle hydrodynamics](https://doi.org/10.1016/j.jcp.2004.11.039)
models heat transfer and latent heat to grow ice from cooled boundaries. The
Warp solver couples implicit enthalpy transport to liquid motion, with refractive
ice and water rendered in Blender Cycles.

| Demo | Scene | Motion | Video |
| :--- | :--- | :--- | :--- |
| Cryogenic bloom | `cold_finger` | Ice grows around a copper rod while the outer water moves | [6 seconds](../media/ice_solidification/cold_finger.mp4) |
| Frozen cascade | `cascade` | Water flows down chilled steps and gradually freezes | [8 seconds](../media/ice_solidification/cascade.mp4) |
| Frozen star | `ice_cast` | A bottom-up freezing front fills a transparent star mold | [8 seconds](../media/ice_solidification/ice_cast.mp4) |

## Install and run

Use Python 3.10+, NVIDIA Warp 1.6.2, [Blender 4.2.9](https://download.blender.org/release/Blender4.2/),
and FFmpeg. Run from the repository root:

```bash
python -m pip install -r ice_solidification/requirements.txt
export BLENDER=/path/to/blender

python -m physstudio run freezing cold_finger --gpu 0
python -m physstudio run freezing cascade --gpu 0
python -m physstudio run freezing ice_cast --gpu 0
```

All geometry is procedural. Runs save particle states, thermal and contact
audits, surface meshes, PNGs, and `render/demo.mp4` under
`outputs/freezing/<scene>/`. The launcher reads `BLENDER`, `FFMPEG`, and `FFPROBE`
from the environment and otherwise uses executables on PATH.

A short check keeps the normal simulation interval and particle resolution:

```bash
python -m physstudio run freezing cascade --frames 2 --width 320 \
  --output outputs/freezing/cascade_check --gpu 0
```

The cold-finger preset saves 181 frames; cascade and casting save 241, all played
at 30 fps. Cascade and casting interleave four thermal updates per saved frame
and use continuous phase-dependent roughness, refraction, and volume scattering.

## Customize and replay

```bash
python -m ice_solidification.harness --scene cascade --gpu 0 \
  --output outputs/freezing/custom_cascade \
  --settings ice_solidification/configs/cascade.json \
  --width 1280 --samples 48
```

Individual stages are `simulate`, `audit`, `surface`, `render`, `encode`, and
`validate`. Use `--stage render` when the simulation and surfaces are already
cached and the render directory has not been created. Use a fresh output for
changed settings. `--seconds` overrides the mechanical duration, `--dx` sets
particle spacing, and JSON settings control the coolant temperature, contact
conductance, ramp timing, and thermal clock.

## Method and checks

The enthalpy formulation retains a latent-heat interval at the freezing point.
An implicit SPH heat solve drives the phase fraction, while mushy resistance
slows the liquid and fully frozen particles attach to the cooled fixture.
The surface reconstruction and material field follow the simulated ice/water
volume fractions. [METHOD.md](METHOD.md) describes the equations and rendering.

```bash
python -m ice_solidification.validate --output outputs/freezing/numerics.json
python -m ice_solidification.diagnostics \
  --input outputs/freezing/cascade/simulation --output outputs/freezing/cascade_report
```

Checks cover the latent-temperature plateau, closed heat exchange, an isothermal
state, a Stefan freezing front, particle mass, collider clearance, and frozen
anchors. Install Matplotlib to add plots to the trajectory reports.

The module reuses the particle and rendering helpers in `ice_melting` and
`implicit_mixture`. [Paper references](references/sources.json).
