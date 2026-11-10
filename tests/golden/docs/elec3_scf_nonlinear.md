# Nonlinear FixedPointSCF initial references

`elec3-scf-nonlinear` is an initial ELEC-3 family, excluded from `all`. It uses
the same three clean pinned external checkouts as the [affine family](elec3_fixed_point.md).
Existing references, checkpoints, tolerance rows and external source trees are
unchanged. CI reads the committed JSON and 18 canonical safetensors anchors.

The variants cover all four local/many-body updates, the four invariant
activations, scalar tanh, augmented interactions, product mixers and all four
additional electron readouts. Vector readouts have separate nonzero-multipole
fixtures. Every variant crosses molecules, rotated molecules, crystals, slabs,
partial-periodic batches and batches with a distinct profile per structure.

Models use seed 123, two channels and two backbone layers. Most fixtures retain
the affine capture's 200-fold response-readout boost. Scaled updates, the
augmented/product-mixer cases, vector cases and many-body electron readouts use
the ordinary unboosted response weights. The fixed-charge fixture uses charges
[-0.006, 0.004, 0.011] in the mixed batch, one tenth of the affine family's values.
These choices keep the unchanged external convergence loop stable; they are
initial fixture definitions, not edits to trained artifacts or existing goldens.

The oracle uses up to 1000 unrolled steps, mixing 0.25 and a 1e-15 iterate-change
criterion. Capture requires a converged status and compares every result before
writing it. V1 differentiates the root equations. Per-structure batches compare
separate physical oracle evaluations with the same charges/Fermi levels. Stress
comes from differentiating an external strain of the unchanged oracle energy.

Independent block probes use deterministic nonzero descriptor, potential,
charge and edge tensors. They record update/readout values, input gradients and
second directional derivatives, including vector paths that could otherwise be
small in a randomly initialized SCF model. Both model and block references use
the existing CPU/accelerated/float32 tolerance rows.

The new `norm_tanh` vector architecture has no matching legacy function. Its
scalar limit is compared exactly to scalar tanh; covariance, finite differences,
double backward, checkpoint identity and reference/cuEq agreement validate the
vector extension. Componentwise vector tanh is explicitly rejected because it
breaks the required rotation symmetry.

```bash
# Initial capture, or recapture with separate explicit authorization.
MACE_ELEC3_SOURCES=/path/to/checkouts python tests/golden/regenerate.py \
  --target elec3-scf-nonlinear --i-know-what-i-am-doing
python -m pytest tests/parity/test_elec3_scf_nonlinear_goldens.py -m 'not gpu'
MACE_REQUIRE_CAPS=gpu,cueq python -m pytest \
  tests/parity/test_elec3_scf_nonlinear_goldens.py -m gpu
```
