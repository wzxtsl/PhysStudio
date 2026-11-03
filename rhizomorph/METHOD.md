# Implementation notes

The implementation follows the soil/plant coupling and signal flow in Li et al.
(2023). The equations below describe the code executed by the
workflow, including its numerical and botanical parameter choices.

## Soil transport

The grid stores water content `w`, nutrient content `nu`, water capacity `c`,
permeability `p`, growth resistance `delta`, and an obstacle distance field.
For each shared face:

```
j = harmonic_mean(p_left, p_right)
    * (-diffusion * grad(w / c) + gravity_vector * upwind(w))
water_transfer = dt * j / face_spacing
nutrient_transfer = water_transfer * upwind(nu)
```

This discretizes the paper's equations 25–30. Each face transfer is shared by
its two cells. The outgoing amount per face is bounded by 0.16 of the donor's
content, so six outgoing faces cannot remove more than the available amount.
Nutrients have a separate donor bound. A second kernel takes the flux divergence.
The boundary pass applies surface rainfall and a three-cell absorbing lower
layer. Side boundaries and rock faces are impermeable. Capacity affects
relative water content; it is not a hard clipping threshold for `w`.

The nutrient transport uses the paper's `j * nu` convention. The preset soil parameters control
resource supply and transport rates.
Both grid concentrations and transfer kernels use FP32. A FP64 ledger tracks
rain, drainage, root uptake, and drained nutrients independently of the fields.

Roots request water from their containing cells. For several roots sharing a
cell, each receives the same fraction of its request if supply is insufficient.
The subtraction and reported uptake therefore agree. Root order and segment
length control the effective absorbing surface. Fine-root effects are represented
by the higher absorption weight of higher-order segments.

## Plant graph and signals

Each node contains position, parent, root/shoot kind, branch identity, order,
birth time, direction, meristem state, and radius. Parent indices precede child
indices. Roots and shoots have separate base nodes at the common collar.

Warp samples five upper-hemisphere directions, attenuated by the nearby crown
segments, to compute light availability and its directional mean. Shoot flux
uses this illumination with an effective foliage weight. Root flux is computed
from actual water uptake. Exponentially decaying seed reserves support the
initial shoot and root before either system can sustain the other.

```
plant_vigor = min(shoot_flux, root_flux)
development = max(0, plant_vigor - maintenance)
```

The minimum follows equation 10. Maintenance is a per-node developmental cost
in these presets. Root/shoot budgets are weighted by the number of active tips
and the relative sizes of the two systems, following the purpose of the official
implementation's auto-balance mode.

Each terminal meristem contributes a competition weight: light for shoots and
nutrient availability divided by resistance for roots. Warp propagates these
weights toward the bases with distance decay. At each branch, the incoming budget
is divided between child subtrees and the node's own active meristem. Apical
continuations receive a species-dependent multiplier. This preserves the budget
even in the step where an apical node activates a lateral bud. The audits check
the sum of allocations at every growth step.

## Development and contact

Warp combines the previous direction, light or nutrient gradients, genetic
tropism, and seeded directional variation. The allocated vigor controls segment
extension. Root elongation decreases with soil resistance. The birch preset can
activate downward sinkers from lateral roots; the oak retains a stronger
downward primary axis.

Near an impermeable obstacle, the proposed direction is projected toward its
tangent plane. Candidate extensions are tested at four points along their
entire length against the sampled distance field and soil bounds. The final
audit independently samples each root segment at 17 points against the analytic
rock ellipsoids, including the segment radius. Rendering puts the rock surface
inside that collision envelope.

Python adds branches and manages the dynamic topology after the Warp kernels
return. The three presets specify branching angles, phyllotaxis, activation
probabilities, and branch-order length budgets. Supporting radii accumulate
downstream pipe weights, with age-dependent secondary thickening.

## Rendering and time

Simulation states are retained for every developmental step. The renderer
interpolates corresponding old nodes, extends new segments from their parents,
and unfolds leaves according to their recorded birth times. The last 10% of
the video holds the completed geometry while the camera continues its orbit.
Soil cutaways are a visibility choice; the transport calculation always uses
the complete 3D soil grid.

The leaf meshes, bark, soil, and rocks are procedural. The scene prefixes
identify procedural botanical genotypes. Soil time, developmental
steps, and playback seconds are recorded separately.

## Relation to the reference implementation

The official C++ project supplies the reference for resource-limited vigor,
auto-balancing, hierarchical allocation, root resistance, and tropism switching.
This implementation uses Warp kernels with a Python topology layer. It replaces
the paper's Lax-Wendroff transport with conservative donor-limited finite volumes,
and the RBV light representation with sampled occlusion.

Reference revision: `a4d2835c8527b9f92895e1cca4a9f30211e97a40`.
