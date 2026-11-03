# Ice Solidification

## Heat and phase

For specific enthalpy `e`, the liquid fraction is `f = clamp(e/L, 0, 1)`.
Temperature is `e/c_ice` below zero, zero throughout the latent-heat interval
`[0,L]`, and `(e-L)/c_water` above it.

| Property | Value |
| :--- | ---: |
| Latent heat | 333,500 J/kg |
| Ice / water heat capacity | 2,100 / 4,180 J/(kg K) |
| Ice / water conductivity | 2.2 / 0.6 W/(m K) |
| Ice / water density | 917 / 1,000 kg/m³ |

Mass stays constant. Particle volume is `m ((1-f)/917 + f/1000)`, giving 9.05%
expansion at complete solidification. This volume weights both liquid density
constraints and the reconstructed field.

The implicit heat update solves

`m_i (e_i_new-e_i_old) = dt [sum_j G_ij (T_j_new-T_i_new) + b_i-d_i T_i_new]`.

Symmetric conductances use harmonic conductivity and the SPH kernel gradient.
A piecewise local inverse solves each enthalpy update inside Jacobi iteration.
Local sums use double precision; particle state uses single precision. The
solver checks its energy residual and subdivides a step when needed. A separate
ledger accumulates heat exchanged with the environment.

Cooled surfaces provide Robin flux. The coolant follows a smoothstep ramp;
exposed surfaces exchange heat with the air. `thermal_scale` advances heat on
an accelerated clock relative to mechanical time. Cascade and casting interleave
four thermal updates per saved frame with their mechanical substeps.

## Motion and boundaries

Liquid particles use gravity, position-based density constraints, viscosity,
and analytic wall contacts. An implicit mushy resistance
`C (1-f)^2 / (f^3+0.001)` slows partially frozen water. Fully frozen particles
store stationary anchors on the cooled fixture.

The cascade uses twelve under-relaxed pressure iterations per mechanical step.
Its collision geometry matches the two stair blocks and surrounding basin.
The star mold resolves particle clearance against both faces at sharp corners.
The cold-finger scene uses a cylindrical vessel and a cylindrical copper rod.

## Surfaces and optics

Volume-weighted particle deposition reconstructs ice and water fields. Boundary
reflections close the field at plates, steps, vessel walls, and the cold rod.
Warp marching cubes includes its half-cell coordinate correction; Taubin
smoothing reduces grid faceting and records the mesh-volume change.

Cascade and casting reconstruct one outer surface from the total density.
The solid-volume fraction is stored in a padded slice atlas and sampled
trilinearly in Cycles. IOR, roughness, absorption, and bump strength interpolate
between liquid and ice; scattering grows with the square of the local solid
fraction. World-space optical detail stays fixed as the front grows. The
cold-finger preset uses separate reconstructed ice and liquid interfaces.

## Verification

`validate.py` checks the latent plateau, closed heat conservation, an isothermal
fixed point, and a one-dimensional Stefan front embedded in a 3D particle
lattice. `diagnostics.py` checks the saved trajectory for finite state, constant
mass, heat balance, collider clearance, anchor drift, and the constitutive
volume relation. `compare_freezing.py` compares phase increments and optional
consecutive-frame image changes. The encoder decodes each MP4 and verifies its
frame count.
