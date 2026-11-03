# Wind flow visualization

The fine neutral filaments and inlet guide follow the presentation in the
[official Windy Trees video](https://www.youtube.com/watch?v=o6EKTlEV44E) at
[00:15](https://www.youtube.com/watch?v=o6EKTlEV44E&t=15s) and
[02:15](https://www.youtube.com/watch?v=o6EKTlEV44E&t=135s).

The simulation contains 15,732 SPH air particles. The default recording stores
5,244 persistent IDs in `trace_particle_ids.npy`. `WindStreaks` displays a stable
subset, using the previous 0.8 seconds of positions and velocities with cubic
Hermite interpolation at half-frame intervals. These are particle pathlines.

Paths split at domain recycling; no filament joins opposite boundaries.
Departing segments can remain visible until their history fades. Opacity fades
with age and at the display-volume edges. Filaments preserve the measured
curvature, while their motion and length communicate speed.

The outlined plane at `x = -4.72 m` identifies the downstream edge of the inlet
relaxation strip. It is a display guide, without a solid collision surface.
The camera uses each global frame index and the full recording length, so
partial renders preserve the same camera motion and trajectory history.

`--flow pathlines` selects the earlier sparse speed-colored display with a
0.2-second history. `--flow off` hides both flow geometry and the inlet guide.
These options affect rendering only. See [the run guide](README.md) for replay
and [the method notes](METHOD.md) for the air domain and boundary conditions.

The showcase uses 1280 × 720, 30 fps, Cycles/OptiX, 16 samples, a fixed random
seed, and H.264/CRF 19 encoding. Each of the three videos contains 301 frames.
`validate_flow.py` checks trace IDs, measured endpoints, interpolation steps,
recycling cuts, opacity bounds and sparse-display sampling.
