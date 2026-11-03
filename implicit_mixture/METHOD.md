# Method and implementation

## Phase and mixture states

Each equal-volume SPH particle stores three volume fractions `alpha[k]` and
three phase velocities `u[k]`. The mixture velocity and density follow Eq. 2:
`v = sum(alpha[k] * u[k])`, `rho = sum(alpha[k] * rho[k])`.
Positions follow the mixture velocity. Phase densities stay fixed. A cubic
spline kernel uses support radius `2.3 * dx` and particle volume `dx**3`.

The Warp implementation follows the paper's equations and the authors'
Taichi conventions for discrete viscosity and pressure iteration. Coordinates
and transport coefficients use simulation units.

## Implicit pressure and phase mapping

A volume-based DFSPH-style solver projects divergence and predicted positive
compression. The incompressibility measure is `psi[i] = sum(V[j]*W[i,j])`.
Pressure impulses map to phase velocities through Eq. 16:

```text
gamma[i,k] = Cd + (1-Cd) * rho[i] / rho[k]
delta_u[i,k] = gamma[i,k] * delta_v_pressure[i]
```

Unlike a projection followed by a single phase mapping, each pressure iteration
here includes the mapped mixture mobility `sum(alpha*gamma)/rho`. This keeps
the projected velocity consistent with the phase velocities used to advect
particles. Pairwise central pressure impulses conserve linear and angular
momentum in the absence of boundaries. Nonnegative pressure enforces compression
constraints without attractive surface pressure. Explicit mixture viscosity
follows Eq. 19 with a small kernel-distance regularizer.

## Conservative phase transfer

Drift flux uses Eq. 20 and diffusion uses Eq. 21. A symmetric common edge limiter
restricts all phase fluxes by the available donor volume. Pairwise antisymmetry
preserves each phase's total volume; zero summed edge flux preserves the local
partition of unity. An absent phase remains absent. Local accumulation uses
float64 before storing float32 states.

Transport also carries donor phase momentum. This extends the paper's treatment
of momentum carried by phase transfer, discussed in its conclusion. It is
verified with a closed-system momentum test. Boundary-driven scenes exchange
momentum with prescribed solids and therefore do not have constant fluid
momentum.

## Tea and boundaries

The tea density ratio is 1.05, matching the paper's tea example. The bag consists
of prescribed solid particles with an initial tea fraction of one. Fluid begins
as water. Solid-fluid diffusion transfers tea out and water into the finite bag;
there is no unlimited color source. Drift transport is disabled across this
solid interface. The bag lifts at 3.2–4.6 s; the rod enters at 4.7–5.5 s and then
orbits. Its particles, analytic contact cylinder and rendered geometry use the
same pose and dimensions. Every output state records the moving-solid poses.

The selected tea scene uses `Cd=0.97`, diffusion `0.006`, and viscosity `0.002`.
These parameters produce the diffusion and stirring sequence in the cup
domain. Set `stirring=false` in a JSON configuration for a rod-free control.

Three layers of boundary particles provide SPH support. Analytic cylinder/box,
floor, tea-bag and finite rod contacts prevent penetration and remove inward
relative phase velocities. Rod cap and side contacts use the shallower escape
direction. Prescribed solids receive no dynamic reaction. Viscosity and moving
boundary pressure couple lateral rod motion into the water. There is no
additional animated velocity field.

The timestep respects phase-velocity CFL and viscosity restrictions. A common
flux limiter handles diffusion positivity. The pressure solve uses relaxed
Jacobi updates and reports residuals and iteration counts. The tea model
contains concentration transport, not heat transfer or chemical extraction.

## Rendering

Gaussian particle deposition reconstructs an occupancy field and phase fractions.
Warp marching cubes extracts the free surface. Nested concentration isosurfaces
provide a piecewise constant volume-absorption approximation inside the tea.
Their thresholds and optical density are display settings; they do not alter
the simulation. Water and glass use distinct refractive indices. Other scenes
render concentration-weighted colors on their reconstructed surfaces.

Warp 1.6 marching cubes returns vertex coordinates with a half-cell offset.
Surface extraction subtracts this offset before mapping to world coordinates,
matching the integer-grid sample locations used during particle deposition.
A known planar field verifies the mapping; the tea meshes are checked against
the cylindrical contact radius and floor over the complete sequence.

The revised collision scene uses `dx=0.018`, Gaussian kernel standard deviation
`0.95*dx`, a reconstruction grid of `0.4*dx`, and six alternating mesh smoothing
passes with coefficients `0.50` and `-0.53`. The wider support suppresses visible
particle-scale scalloping; the alternating steps avoid the shrinkage of repeated
one-way Laplacian smoothing. Mesh volume changes and vertex displacements are
recorded in the surface manifest. The particle simulation remains the source of
the splash trajectories.

For tea, reflected render samples provide support at the cup wall and floor.
The reconstructed field is intersected with the cup interior, with a 0.0002-unit
overlap to avoid an artificial air gap. This removes uneven glass/liquid
intersections. These reflected samples are used only during surface extraction.
The optical absorption multiplier is `270`, with linear shader color
`[0.78,0.50,0.10]`. Optical settings are applied to the cached concentration field during rendering.

The rendering and simulation caches remain separate so that lighting, optical
density and cameras can be refined without rerunning physics. Meshes and rendered
frames retain their simulation frame indices. Videos use the simulation's
saved frame rate, without time remapping.

## Validation

`validate.py` checks the spline derivative, phase-pressure mapping, net pressure
impulse and torque, aggressively limited conservative phase and momentum
transport, fully coupled phase velocities, and finite-budget tea release.
`audit.py` reads every saved state, checks finite values, phase bounds, partition
error and per-phase mass conservation, and records phase centers and tea mixing.
`diagnostics.json` also records pressure residuals and maximum contact correction.
Visual review checks glass visibility, bag/rod alignment, concentration transport
and the absence of distracting surface artifacts.
