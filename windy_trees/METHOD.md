# Windy Trees: implementation and validation

## Reference and scope

Sören Pirk, Till Niese, Torsten Hädrich, Bedrich Benes, and Oliver Deussen.
**Windy Trees: Computing Stress Response for Developmental Tree Models.**
ACM Transactions on Graphics 33(6), Article 204, 2014.
[Project](https://storage.googleapis.com/pirk.io/projects/windy_trees/index.html) ·
[Paper](https://storage.googleapis.com/pirk.io/papers/Pirk.etal-2014-WindyTrees.pdf) ·
[DOI](https://doi.org/10.1145/2661229.2661252).

This implementation develops the paper's short-term wind interaction on the
three already grown Rhizomorph trees. It implements SPH air, branch/foliage
sensors, two-way coupling, elastic recovery, damping and load propagation.
The topology and rest shape stay fixed during each ten-second run. Long-term
growth adaptation, branch fracture and bud abrasion are outside these demos.

The numerical adaptations are a globally coupled implicit articulated solver,
quadratic aerodynamic drag, and a separate hinge for each leaf blade. The
paper uses its own edge-force model and clustered foliage sensors.

## Tree adapter

`assets.py` reads the final growth state and maps shoot nodes into an
articulated tree. Zero-length lateral-bud edges collapse into their parent;
nonzero segments preserve their original length. Roots retain their original
geometry and stay fixed. The shoot base is pinned in position and rotation.

Wood mass is computed from cylindrical segment volumes at 650 kg/m³; leaf
mass uses the integrated blade area at 0.09 kg/m². These masses are lumped at
shoot nodes. Leaves reuse the original growth renderer's shape, size, location,
and orientation. Material appearance and growth are not animated during wind.
The grown shape is the unloaded equilibrium; self-weight is absorbed into
that rest shape rather than being added as a new transient load.

## Implicit articulated dynamics

Each non-root node owns a local angular displacement `theta` and rate `omega`.
Its quaternion is composed with the parent quaternion. Positions are
reconstructed by accumulating rotated rest edges, so every segment retains
its length exactly up to floating-point error.

At the current configuration the nodal velocity Jacobian maps joint rates to
translation. For a downstream node `i` and ancestor joint `j`,

```text
v_i += cross(R_parent(j) omega_j, p_i - p_parent(j))
M = Jᵀ diag(m_i I₃) J
K_j = E π r_joint,j⁴ / (4 L_j)
```

The radius used at a joint is the smaller of the parent radius and 1.25 times
the child radius. `C` is diagonal rotational damping, with a subtree inertia
estimate setting the damping scale and serving as the PCG preconditioner.
The mass matrix itself includes coupled motion of all downstream nodes.

Each step solves

```text
(M + dt C + dt² K) omega_next = M omega + dt tau - dt K theta
theta_next = theta + dt omega_next
```

The solver evaluates `J` and `Jᵀ` through Warp kernels without assembling `M`.
Subtree forces and moments accumulate from leaves to the base. PCG uses 60
iterations at the default 1/240 s step. The solve freezes the configuration
during the step and omits Coriolis terms; it is an articulated tangent model.
The largest local joint angle in the three completed runs is 0.094 rad.

Individual leaf hinges use the nonlinear elastic torque
`k theta (1 + 0.8 theta²)`. Elastic tangent stiffness and the aerodynamic
velocity derivative enter an implicit hinge update. Branch quaternions and
leaf hinge rotations are both applied to the rendered blade vertices.

## Air and aerodynamic coupling

The air solver is weakly compressible SPH with cubic-spline support, density
summation, symmetric pressure and viscous interactions, and artificial
viscosity for approaching pairs. Default spacing is 0.38 m and kernel support
is 0.80 m. Air density is 1.225 kg/m³; numerical sound speed is 15 m/s. This
coarse visual wind model is intended to resolve crown-scale flow.

A branch samples air at its midpoint. Drag depends on the air velocity
relative to its moving segment and projected transverse area. A leaf samples
at its moving blade center; its relative velocity includes the petiole's
angular motion. The leaf force is

```text
F = 0.5 rho C_d A (|u_n| u_n n + 0.12 |u_t| u_t)
```

Normalized SPH interpolation weights distribute the opposite force back to
air particles. This preserves the linear action/reaction force sum. The
spread is not an angular-momentum-conserving contact discretization.

The domain is `[-6, 7.5] × [-4.4, 4.4] × [0.02, 7.2]` m. Air particles leaving
the volume recycle to the opposite boundary with a newly sampled inlet
velocity. An inlet/side strip relaxes toward the prescribed gust field at
5 s⁻¹; the interior uses 0.25 s⁻¹. The analytic gust perturbation is
divergence-free before its startup ramp and height profile are applied.
Pressure interactions, tree feedback and advection then evolve the flow.
This driven, open domain exchanges momentum with its inlet and boundaries.
The soil surface is rendered below the airflow domain; ground-layer flow and
solid boundary layers are not resolved.

## Airflow visualization

The original runs record a stable set of 404 SPH particle IDs at 30 fps.
`flow.py` draws the last 0.2 seconds of each recorded trajectory with cubic
Hermite interpolation using the measured endpoint velocities. These are
particle **pathlines**, rather than instantaneous streamlines.

The head is the current particle position. Tail opacity and radius fade with
age; color maps speed from teal through pale green to amber. Since every
path covers the same time interval, length also conveys speed. Paths are
cut whenever a particle recycles across a domain boundary. Spatial fading
keeps the background readable without changing any physics.

Blender renders branch/leaf meshes and the path geometry with Cycles/OptiX.
FFmpeg writes H.264/YUV420p videos with CRF 19 and fast-start metadata.

The subsequent [official-video visualization update](FLOW_VISUALIZATION.md)
records 5,244 particle IDs and introduces `WindStreaks`: thin neutral filaments
with a 0.8-second history, a visible inlet guide and a wider camera composition.
The previous colored style remains available as `--flow pathlines`. The
underlying branch and air solvers are unchanged by this display update.

## Showcase presets

The three simulations are `outputs/windy_trees/<scene>/simulation`. Each contains
301 saved states spanning 0–10 seconds. Full parameter dictionaries and
source-state hashes are stored with each simulation.

| Scene | Shoot nodes | Leaf blades | Nominal wind | Young's modulus |
| --- | ---: | ---: | ---: | ---: |
| Oak | 1,419 | 2,674 | 8.5 m/s | 1.0 GPa |
| Maple | 5,237 | 10,764 | 7.0 m/s | 1.3 GPa |
| Birch | 2,372 | 4,520 | 9.0 m/s | 1.1 GPa |

All use 15,732 air particles. Maple's inlet direction turns with a 0.75 rad
amplitude; Birch uses the largest turbulence amplitude. The trees share
their source growth geometry with the earlier growth demos.

## Numerical evidence

The full development trajectories produced the measurements below. Run the
packaged validation commands in [README.md](README.md) for a new trajectory.

| Check | Measured result |
| --- | ---: |
| Matrix-free operator vs. a CPU finite-rotation Jacobian | 1.30 × 10⁻⁷ maximum relative error |
| One implicit step vs. a dense linear solve | 1.12 × 10⁻⁵ relative error |
| Damped branch vs. its analytic small-angle response | 0.572% RMS error, normalized by initial amplitude |
| Branch lengths across all three ten-second runs | ≤ 2.16 × 10⁻⁶ m maximum error |
| Fixed shoot base | Zero measured displacement |
| Normalized tree/air force imbalance | ≤ 1.43 × 10⁻⁷ |
| Displayed path heads vs. recorded particle positions | Zero measured difference in 6,663 checked curves |

The unforced branch oscillator loses energy monotonically over the checked
two seconds. The global solve's maximum preconditioned relative residual is
0.00127 across all scene frames.

A matched control holds the Oak fixed, uses identical gust inputs with zero
turning/turbulence, and compares air feedback on versus off. In the downstream
region `x=[1.5,3.5], y=[-1.25,1.25], z=[1,4]` m, averaging over 4–6 s, the mean
along-wind velocity falls from 6.760 to 6.092 m/s with feedback: a 9.89%
reduction. `wake_comparison.npz` stores the sampled fields for plotting.
This measures crown feedback within this setup; it is not a comparison with
experimental or paper-reported airflow measurements.

## Files

- `assets.py`: mature-tree graph, mass, stiffness and leaf adapter.
- `tree.py`: implicit articulated solver and individual leaf hinges.
- `wind.py`: SPH fluid kernels and equal/opposite aerodynamic exchange.
- `simulate.py`: trajectories, provenance and per-frame audits.
- `flow.py`: pathline extraction, boundary cuts and speed colors.
- `render_blender.py`: branch/leaf binding, scene lighting and flow display.
- `validate.py`: independent numerical references and feedback control.
- `harness.py`: staged simulation, validation, rendering and encoding.
- `validate_flow.py`: dense trajectory endpoints and recycled-boundary checks.
