# Validation

The accepted recordings use Warp 1.6.2, NumPy 1.26.4, SciPy 1.14.0 and an
NVIDIA RTX 3090. Blender 4.2.9 Cycles renders saved surfaces; it does not drive
the physics. Shared GPU timings are not a paper performance comparison.

## Numerical checks

| Check | Observed result |
| --- | --- |
| Analytic sphere area / volume quadrature | 1.359% relative bias at 1,622 E samples |
| Sphere MLS curvature | 0.491% relative RMS error |
| Conservative affine scatter momentum | Relative error below 1.2e-14 |
| Constant-velocity transfer round trip | Maximum error below 2.1e-16 |
| Compatible pressure operator vs independent sparse assembly | Relative error below 4.1e-16 |
| Manufactured CG solution | Relative error below 1.2e-9 |
| Unforced sphere after 0.2 s | 0.109% radial RMS change |
| Closed-surface force balance | Residual below 7e-18 |
| Continuous parallel-plane contact projection | Maximum error 1.88e-12 m |
| Distributed contact matrix / force | Maximum errors below 1.2e-23 / 4.7e-23 |
| Contact positive semidefiniteness | Lowest eigenvalue -1.09e-24, within roundoff of zero |
| Separated unilateral contact | Exactly zero force |
| Penetrating unilateral force vs independent assembly | Maximum error below 1e-23 |

The sphere has small sampled-surfactant relaxation, rather than an exactly
motionless discrete equilibrium. The continuous-contact algebra check assembles
`Jᵀ diag(c) J` independently and also checks equal-pressure cancellation in the
experimental multi-region branch. It does not establish full foam accuracy.
The original nearest-sample contact check includes a finite-difference energy
gradient comparison. [Commands](README.md#checks-and-references).

## Complete recordings

| Quantity | Giant recoil | Breathing | Collision |
| --- | ---: | ---: | ---: |
| Saved frames | 241 | 241 | 241 |
| Physical seconds | 0.6 | 1.2 | 0.6 |
| E / L particles | 1622 / 26000 | 1622 / 26000 | 3244 / 52000 |
| Maximum covariance axis ratio | 2.26733 | 1.16705 | 1.71817 |
| Maximum gas-volume change | 0.40597% | 0.0787% | 0.26390% |
| Maximum E displacement per saved frame | 2.408 mm | 1.242 mm | 2.239 mm |
| Maximum MLS projection | 4.442 µm | 1.16 µm | 7.898 µm |
| Maximum local penetration | — | — | 0.02325 mm |

Every state passes finite-value, positive-thickness/surfactant/surface-tension,
timestamp, frame-count, gas-volume, frame-motion and CG checks. Material liquid
and surfactant arrays stay exactly constant; scatter errors remain below 5e-16.
Gas changes are relative to each discrete initial volume, not the analytic
sphere volume. The collision has up to 233 nearby contact-support samples;
proximity does not imply an active unilateral constraint.

All reconstructed surfaces pass finite-coordinate, positive-thickness,
nondegenerate-triangle and outward-orientation checks. Recoil/collision
integrated optical film-volume ratios are 0.99932–1.00664 and 1.00546–1.00709.
Their final cameras keep every recorded E surface inside the image; maximum
normalized horizontal/vertical extents are 0.8897 / 0.8132 against a boundary
of 1.0. [METHOD.md](METHOD.md) describes the contact and optical adapters.

## Media

The three showcase MP4s retain the accepted simulation renders: 1280 x 720,
30 fps, 241 frames / 8.034 seconds, H.264/YUV420p. Every frame decodes, and
encoded storyboards and peak-motion/contact neighborhoods have been reviewed.
The [media manifest](../media/manifest.json) records hashes and dimensions.
GIF and JPG previews are derived from those same videos.

The short pipeline retains full physics resolution and camera timing. It tests
installation and execution; the full recording audits establish the complete
shape changes and collision. Recoil and collision model intact films; these
checks do not claim shared-wall foam, rupture, or calibrated atmospheric bubbles.

## Public interface validation

All three presets completed three-frame pipelines through `python -m physstudio
run thin_film` at the full particle resolution, including recording audits,
mesh reconstruction/checks, 320 x 180 Cycles renders, encoding and full decoding.
The three numerical-check programs and all three full recording audits pass
with the packaged code. Render resume preserves existing PNG hashes and rejects
changed sample settings.

Eleven implementation/check modules match the accepted implementation byte for
byte. Simulation adds public defaults and preserved full camera timing; the
renderer uses a portable root and reads that camera metadata. Regenerated
three-frame prefixes agree within 4.1e-13 m in position and 1.2e-10 m/s in velocity.
Nine frozen frames rendered at 1280 x 720 and 48 samples differ by at most one
8-bit channel level, with mean differences below 0.000054.

The three showcase MP4s are byte-identical to the accepted videos. Every frame
of their 96-frame looping GIF previews decodes. Existing 54 launcher plans,
55 media records and 21 README showcase sections remain unchanged.
