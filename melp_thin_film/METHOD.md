# Moving Eulerian-Lagrangian thin films

Reference: Yitong Deng, Mengdi Wang, Xiangxin Kong, Shiying Xiong,
Zangyueyang Xian, and Bo Zhu, *A Moving Eulerian-Lagrangian Particle Method
for Thin Film and Foam Simulation*, ACM TOG 41(4), 2022,
[DOI 10.1145/3528223.3530174](https://doi.org/10.1145/3528223.3530174).
This is an independent Python/Warp implementation. The authors' C++ code is
not copied, installed, or imported. [Source metadata](references/sources.json).

## Geometry, material, and transfers

Sparse E particles describe the three-dimensional moving surface. Dense L
particles carry liquid volume, surfactant amount, momentum, and advected film
thickness. Warp hash grids provide local neighborhoods; quadratic MLS estimates
surface normals, curvature, and material projection. Wendland C2 kernels define
surface support. Normalized L-to-E scatter conserves liquid and surfactant;
centered affine offsets conserve total linear momentum during scatter.
E-to-L transfer fits vector velocity over tangent offsets.

Material thickness used by the physical solve is conserved liquid volume divided
by reconstructed area. Surfactant concentration determines surface tension.
Initial material vortices are recorded velocity inputs. The breathing preset
also applies a rotating quadrupole pressure actuator with zero discrete resultant.

## Implicit dynamics

Warp float64 kernels and GPU conjugate gradient solve the tangential system

`A = diag(a/Gamma) + dt² Gᵀ diag(2 RT a/(rho h)) G`,

where G is the compatible surface gradient. Its exact weighted adjoint makes
the operator symmetric positive definite for positive material state. The
normal system contains a capillary graph and a gas-law rank-one term per closed
region. Gas pressure follows `p V = constant`. A normal-force projection removes
the spurious resultant caused by closed-surface point-cloud quadrature.

E particles move normally with the surface. Tangential density relaxation keeps
support usable. The revised presets evaluate redistribution along an anchored
quadratic MLS surface, excluding the fit's constant offset to avoid repeated-fit
shrinkage. This is a density-relaxation adapter rather than the paper's IISPH
number-density projection. L particles move with material velocity and project
back onto the reconstructed surface. Neighbor support and CG residuals are
checked before accepting a state.

## Contact

Bubble Collision projects each E sample onto the other region's continuous MLS
surface. Normalized positive SPH weights distribute contact impulses over the
guest patch. The implicit contribution has form `dt² Jᵀ diag(c) J`, with its
contact diagonal included in preconditioning. The positive weights approximate
the projection Jacobian rather than exact quadratic MLS shape functions.
The unilateral penalty activates only when signed penetration plus the 0.15 mm
clearance is positive. Its stiffness is 50,000; adhesion and tension sharing
are disabled in this preset.

Optional normal XSPH smoothing acts on the predictor before the implicit solve,
preserving the final gas/contact constraints. Older nearest-sample contact and
experimental shared-tension/multi-region branches remain available in the code,
but the showcase uses intact-film compression and rebound. The implementation
does not provide shared-wall foam topology, stochastic material transfer between
regions, rupture, or reconnection, and has not validated Plateau-angle equilibrium.

## Presets and numerical adaptations

| Preset | Ambient pressure | Added normal mass | Normal damping | Initial geometry |
| --- | ---: | ---: | ---: | --- |
| Giant recoil | 100 Pa | 0.06 kg/m² | 2 s⁻¹ | Ellipsoid axes 1.65 / 0.72 / 0.84 |
| Breathing | 100 Pa | 0.008 kg/m² | 40 s⁻¹ | Sphere with rotating pressure actuator |
| Collision | 100 Pa | 0.06 kg/m² | 1.5 s⁻¹ | Two ellipsoids, axes 1.35 / 0.78 / 0.95 |

The characteristic radius is 0.04 m. Collision centers start at x=±56 mm and
y=±6 mm, with inward velocities of 0.45 m/s. The reduced ambient pressure and
effective added mass are graphics adaptations, not calibrated atmospheric
bubble parameters. Full atmospheric strong deformation remains unvalidated.
Initial discrete gas pressure is calibrated to the sampled geometry; the area
and volume estimators have a separately measured quadrature bias.

## Recorded surfaces and optics

CPU reconstruction subdivides a display mesh and projects it onto the saved
MLS surface. Fixed closed-mesh connectivity is suitable for these intact films.
The collision preset reconstructs optical thickness from each L particle's
conserved liquid volume times its fine surface number density. This avoids
relying on independently advected thickness near contact. Recoil and breathing
retain their saved advected thickness. Mesh audits integrate optical thickness
and compare the resulting liquid volume with the conserved material quantity.

Blender evaluates three-band air/water/air thin-film Fresnel interference from
thickness and view angle. No time-animated color texture is used. Lighting,
cameras, and apparatus are presentation elements; there is no surrounding air
solver or simulated interaction with the exhibition plinth. The revised
collision camera adapts to recorded extent as the films move apart. Rendering
saved states leaves the physical trajectory unchanged.
