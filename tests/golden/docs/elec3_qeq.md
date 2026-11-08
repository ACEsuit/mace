# MACE-QEq references

The explicit `elec3-qeq` target captures initial QEq references separately from
LC/LSC. It is excluded from `all`. CI reads committed JSON and canonical
safetensors without loading external source checkouts.

The oracle uses the three clean revisions listed in [the LC/LSC recipe](elec3.md),
with pins and subprocess isolation shared through
`targets/_electrostatics_oracle.py`. The code in those checkouts, including their
version assertions, remains unchanged. The test bridge projects backbone
weights into V1's canonical reduced basis and copies both observable heads.
Unused legacy hardness parameters are omitted when hardness is fixed.

Four variants cross six geometries: fixed hardness; learned hardness plus
atomic electronegativity/hardness inputs; the analytic nonperiodic equation;
and disabled energy self interaction with a different Gaussian width. Each
variant covers a molecule, its rotated copy, a crystal, a slab, and mixed
batches under `partial` and `per_structure`. The latter uses separate physical
oracle evaluations with matching input slices, as the existing V1 contract
requires. Total charges differ between structures and external fields are
nonzero.

All 24 cases pin energy, forces, stress, virials, dipole, density coefficients,
electronegativity, hardness and QEq energy. The learned-input cases also pin
gradients of a force/stress loss with respect to both atomic inputs. Stress and
virials are obtained by differentiating the unchanged oracle energy under
external strain: upstream QEq does not itself expose those outputs. Aperiodic
stress is masked; its virial remains physical and unmasked.

Capture compares every output before writing the initial fixtures. Two tiny
canonical anchors cover the fixed and learned readouts; variants differing
only in charge equation or boundary settings load the same weights. CPU uses
the existing `fp64_cpu_reference` and `fp32` tolerance rows. GPU uses
`fp64_accelerated_backend` and `fp32`, including force/stress-loss backward.
Float32 refers to model weights with V1's existing accumulation floor.

```bash
# Initial capture or recapture requires explicit authorization.
MACE_ELEC3_SOURCES=/path/to/checkouts python tests/golden/regenerate.py \
  --target elec3-qeq --i-know-what-i-am-doing
python -m pytest tests/parity/test_elec3_qeq_goldens.py -m 'not gpu'
MACE_REQUIRE_CAPS=gpu,cueq python -m pytest \
  tests/parity/test_elec3_qeq_goldens.py -m gpu
```

The [implementation record](../../../docs/reforge/gate3/2026-11-08/elec3-qeq.md)
describes the two equations, implicit derivatives, solver identity and limits.
Existing golden outputs and tolerance rows were not changed.
