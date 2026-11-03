# Strongly coupled magnetic rigid bodies in Warp

This is an independent implementation of the formulation in Westhofen,
Fernández-Fernández, Jeske and Bender, *Strongly Coupled Simulation of Magnetic
Rigid Bodies*, Computer Graphics Forum 43(8), 2024, DOI
[10.1111/cgf.15185](https://doi.org/10.1111/cgf.15185).
The paper and official video inform the algorithm and scene design. The authors'
STARK solver is not a dependency; no author implementation is copied or invoked.

## Incremental potential

Unknowns are end-of-step linear and angular velocities, scaled by square roots
of mass and mean principal inertia for conditioning. Backward Euler uses
`x = x_old + h v` and the normalized quaternion update
`q = normalize((h omega / 2, 1) * q_old)` (xyzw storage). The objective adds
translational/rotational inertia (paper Eqs. 3–5), magnetic energy, cubic contact,
regularized Coulomb friction and joint energies. Gravity enters the inertial
predictor. Optional velocity-proportional damping is an explicit scene choice.
Prescribed actuator coordinates and fixed compass pivots are eliminated from
the free variables; reactions on these supports are not integrated.

Each magnetic sample is a volume-weighted dipole rigidly attached to its body.
For distinct-body samples, one unordered pair contributes

```
U_ij = mu0/(4 pi) [ dot(mi,mj)/r^3 - 3 dot(mi,r) dot(mj,r)/r^5 ].
```

This is the sum of the two directed `-1/2 mi dot Bj` terms in paper Eq. 16.
Gradients include both alignment torque and torque from an offset sample's
force. Samples on the same rigid body do not interact. SI units are used:
meters, seconds, kilograms, magnetic moment in A m^2, susceptibility unitless.
The permeability coefficient is `1e-7` in this model.

Permanent samples store local moments. Linear samples update synchronously
from the previous pose and previous moments once per accepted substep:

```
m_i = V_i chi_i/(1 + chi_i/3) sum_j H_j,
H_j = [3 n dot(n,m_j) - m_j] / (4 pi r^3).
```

The paper's Eq. 18 prints B; Eqs. 7–9 require H (equivalently B/mu0) for SI
consistency. This dimensional interpretation is recorded explicitly and
checked against the analytic isolated linear sphere response. New local moments
remain frozen during Newton, while their world directions rotate with the body.
This is lagged, linear magnetization, without saturation, hysteresis or a fully
converged mutual induction solve. Electromagnet current is a prescribed smooth
input; its sample update follows on the next substep.

Near pairs enter the implicit potential. Far pairs use its first-order Taylor
expansion about the old pose: constant generalized force/torque and no magnetic
Hessian. Both branches derive from the same unordered energy, preventing a
factor-of-two discontinuity at the threshold. Scene thresholds are in config.

## Contact, friction and joints

The paper uses the cubic contact penalty `kappa max(dhat - gap, 0)^3` (Eq. 19),
not a logarithmic barrier. Our geometric adapter supports rigid sphere unions,
planes and the inside of a spherical container. It does not implement the
paper's general triangle-mesh point/edge contact system. Rendered interacting
balls/triads and the inner glass wall use the same physical radii. Compass
pointers and coil winding details use visual approximations of their proxies.
They do not alter physical samples or recorded poses.

Contact normals and normal loads are lagged for regularized Coulomb friction.
The dissipation uses tangential connector displacement with the C1 smoothing
`-u^3/(3 eps^2) + u^2/eps + eps/3` below `eps=h*v_slip` and `u` above it.
Contact connectors attach to both rotating bodies. Ball and distance constraints
use quadratic penalty energies; compass pivots eliminate all DOFs except yaw.
The crane's suspension is a stiff distance constraint, not a simulated cable.

A broad phase selects potential contacts with a displacement margin. Accepted
steps must contain every newly activated contact. Line searches and complete
steps use conservative swept-sphere tests: linear center trajectories enlarged
by the bound `|local_offset| * rotation_angle^2 / 8` for rotational curvature.
Invalid or nonconverged steps roll back and halve the time step. Contact
stiffness doubles when the failed iterate has used over 90% of the contact
thickness. Other Newton failures refine time without increasing stiffness:
with our numerical Hessian, unconditional stiffening can create a feedback
loop and inject penalty energy into existing contacts. This is an adaptation
of the paper's unconditional failure rule. Calibrate initial stiffness for
the selected magnetic strength: the two-sphere stress checks use kappa 1e8
for moment 50 and 1e10 for moment 500 (force scales quadratically). Insufficient
stiffness can produce rebound even when nonpenetration is maintained. Kinematic
actuators can do work; energy is not conserved in the driven, damped demos.

## Numerical implementation and scope

Warp executes float64 energy evaluation, reverse-mode gradients, per-element
Hessian assembly, induction and state integration on the GPU. Hessian entries
are central finite differences of the 12-variable element potential; they are
checked independently against differences of exact AD gradients. This differs
from the paper's symbolic analytic Hessian. The small assembled system is copied
to the host for a symmetric eigendecomposition (NumPy/SciPy), projected to the
positive definite cone, and solved in that eigenbasis. Backtracking uses the
original objective and collision checks. Thus this is a Warp physics backend
with a host dense Newton solve, not a claim of a fully GPU sparse implementation.

This adapter targets dozens of bodies, with many sample pairs processed in
parallel. It does not claim the paper's arbitrary-mesh coverage, large-scene
performance, timing numbers or exact scene reproduction. Procedural demos
illustrate the same types of coupling without external mesh assets. Blender
Cycles renders saved rigid states; it does not simulate or modify their motion.

## Evidence

`validate.py` compares magnetic force/torque to independent analytic formulas,
checks the energy Hessian, verifies near/far continuity and induction units,
then exercises strong magnetic contact and frictional rolling. Run artifacts
include configs, source hashes, poses, velocities, moments, Newton residuals,
contact gaps, retries and render settings. [VALIDATION.md](VALIDATION.md) records
measured results and reproduction commands; the [run guide](README.md) describes
the three procedural presets and frozen-state replay.
