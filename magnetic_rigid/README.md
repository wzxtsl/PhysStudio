# Magnetic Rigid Bodies

[Strongly Coupled Simulation of Magnetic Rigid Bodies](https://doi.org/10.1111/cgf.15185)
couples dipole fields, induced magnetization, friction, and rigid contact in a
shared implicit solve. This independent Python/Warp implementation provides
three procedural scenes rendered from recorded states with Blender Cycles.

| Scene | Preset | Interaction | Recording |
| :--- | :--- | :--- | :--- |
| Field of Needles | `compass_garden` | 76 pivoted needles follow a rotating magnetic source | 301 frames, 10 s |
| Magnetic Cabinet | `magnetic_globe` | 17 rigid triads assemble against a glass globe, then fall as the field turns off | 361 frames, 12 s |
| Electromagnetic Foundry | `magnetic_crane` | A suspended electromagnet collects, transports, and releases 32 beads | 361 frames, 12 s |

Warp evaluates float64 energies, derivatives, induction, and integration.
NumPy/SciPy solves the assembled projected Newton system on the host. Contact
supports sphere unions, planes, and a spherical enclosure. [METHOD.md](METHOD.md)
explains the paper formulation and numerical adaptations. The authors' STARK
implementation is not used. All geometry and materials are procedural.

## Install and run

Use the shared [installation](../INSTALL.md), with Blender 4.2.9 supporting
Cycles/OptiX, an NVIDIA GPU, and `ffmpeg`/`ffprobe` on `PATH`. The pinned NumPy,
SciPy, Warp, and Pillow versions are in [requirements.txt](requirements.txt).
No asset download or additional Python packages are needed.

From the repository root:

```bash
export BLENDER=/path/to/blender
python -m physstudio run magnetic_rigid compass_garden --gpu 0
python -m physstudio run magnetic_rigid magnetic_globe --gpu 0
python -m physstudio run magnetic_rigid magnetic_crane --gpu 0
```

Each command simulates, audits, renders at 1280 x 720 with 32 samples, and encodes
a 30 fps H.264 MP4. Choose a fresh output for each changed simulation:

```bash
python -m physstudio run magnetic_rigid magnetic_crane --gpu 0 \
  --frames 2 --width 320 --output outputs/crane_check
python -m physstudio run magnetic_rigid magnetic_crane --dry-run
```

`--frames` includes frame zero and must be at least two. Short runs retain the
complete body/sample resolution, 1/120 s base physics step, 30 fps sampling,
and full camera/actuator timeline; they record a shorter prefix. Failed Newton
steps refine time conservatively. Full simulations take substantially longer
than the short pipeline check.

Default outputs are in `outputs/magnetic_rigid/<preset>/`:

```text
simulation/   Config, source hashes, numbered states, Newton diagnostics, completion record
audit.json    State, gap, residual, and full-duration interaction checks
render/       Numbered PNGs, renderer snapshots, per-invocation settings
<preset>.mp4  Encoded video
<preset>.jpg  Video poster
<preset>.storyboard.jpg
<preset>.verification.json
```

Stage logs and `run.json` accompany the pipeline. Encoding checks the entire
frame sequence, verifies the encoded count, and decodes the full video.

## Stages and replay

The method harness exposes `simulate`, `validate`, `render`, and `encode` stages.
The default stage is `all`. For example, simulate once and render saved poses:

```bash
python -m magnetic_rigid.harness --stage simulate --scene magnetic_crane \
  --output outputs/crane_run --gpu 0
python -m magnetic_rigid.harness --stage validate --scene magnetic_crane \
  --output outputs/crane_run
python -m magnetic_rigid.harness --stage render --scene magnetic_crane \
  --output outputs/crane_run --gpu 0
python -m magnetic_rigid.harness --stage encode --scene magnetic_crane \
  --output outputs/crane_run
```

Add `--resume` to the render stage after an interruption. Existing PNGs are
preserved; `--start` and exclusive `--end` select global frame ranges without
changing camera time. Keep the same resolution and sample count when resuming.
Use a fresh output for visual changes and reuse the completed simulation:

```bash
python -m magnetic_rigid.harness --stage render --scene magnetic_crane \
  --simulation outputs/crane_run/simulation --output outputs/crane_replay \
  --width 1920 --height 1080 --samples 64 --gpu 0
python -m magnetic_rigid.harness --stage encode --scene magnetic_crane \
  --simulation outputs/crane_run/simulation --output outputs/crane_replay
```

The simulation stage accepts `--settings path/to/overrides.json` for explicit
parameter overrides. Presets and control trajectories are in [scenes.py](scenes.py).
Retain the accepted defaults to reproduce the showcase.

## Checks and references

```bash
CUDA_VISIBLE_DEVICES=0 OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 \
  python -m magnetic_rigid.validate --output outputs/magnetic_checks.json
python -m magnetic_rigid.audit outputs/crane_run/simulation \
  --require-full-demo --output outputs/crane_audit.json
```

[Validation results](VALIDATION.md) include analytic magnetic derivatives,
induction, strong contact, friction, swept collision, complete scene audits,
and release checks. Short recordings always undergo physical-state checks;
full-duration recordings also require the advertised scene interactions.
`--require-full-demo` rejects incomplete-duration recordings.

[Paper and source metadata](references/sources.json) ·
[Author project page](https://animation.rwth-aachen.de/publication/0590/) ·
[Showcase gallery](../media/index.html#magnetic-rigid-bodies)

Original code is licensed under [Apache 2.0](../LICENSE).
