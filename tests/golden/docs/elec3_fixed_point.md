# Affine FixedPointSCF references

`elec3-fixed-point` captures an initial family, separately from LC/LSC and
QEq, and is excluded from `all`. It uses the three clean pinned checkouts in
[the external oracle recipe](elec3.md). The legacy implementation, its version
assertions and all existing golden files remain untouched. CI only needs the
committed JSON and four small canonical safetensors anchors.

The oracle is `FixedPointCore` with `FixedPointSCFRunner`. The historical
`FixedPoint` class is a deserialize-only shell whose forward is not executable.
The fixture uses two MACE layers, two channels, seed 123, and scales its learned
output-response weights by 200 before freezing them. This puts the tiny random
network outside the legacy Fermi-update emergency clamp while keeping the
iteration stable. The same weights are carried to V1's canonical reduced basis.
This is an initial test fixture, not a modification of a published checkpoint.

The four variants are fixed charge with default gated sources; fixed Fermi
with a nonzero Fermi offset; monopoles and dipoles with linear sources, species
scales, two receiver widths, feature normalization, energy self interaction
and quadratic electron energy; and a linear potential embedding with field
self interaction and linear electron energy. Each crosses a molecule, its
rotated copy, a crystal, a slab, and mixed batches under `partial` and
`per_structure`. The latter compares separate physical oracle calls, as V1's
existing boundary contract requires.

The unmodified convergence loop uses up to 1000 steps, mixing 0.25 and a
1e-15 iterate-change criterion. The stronger convergence criterion is needed
for second derivatives: an apparently converged density can still have a
measurable tail in its unrolled force-loss gradient. Capture requires an
actual `converged` status and compares every output and gradient before writing.
No historical tolerance is relaxed. This family uses the existing
`fp64_cpu_reference`, `fp64_accelerated_backend` and `fp32` rows.

Besides all physical outputs, the reference pins the directional derivative
of the squared force/stress loss when scaling the learned field-response
output weights. This direction survives canonical weight reordering and
normalization. The V1 result differentiates the exact affine root, whereas
the oracle differentiates its converged iteration history. Independent package
tests add finite differences and `gradcheck`/`gradgradcheck`.

```bash
# Initial capture or subsequent recapture requires explicit authorization.
MACE_ELEC3_SOURCES=/path/to/checkouts python tests/golden/regenerate.py \
  --target elec3-fixed-point --i-know-what-i-am-doing
python -m pytest tests/parity/test_elec3_fixed_point_goldens.py -m 'not gpu'
MACE_REQUIRE_CAPS=gpu,cueq python -m pytest \
  tests/parity/test_elec3_fixed_point_goldens.py -m gpu
```
