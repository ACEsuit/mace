# Deterministic fp32 reductions (PAR-2)

`references/fp32_reductions.json` freezes a CPU fixture for the reference
backend's `segment_reduce` factory and the edge-to-node accumulation inside
`channelwise_tp_conv`. The latter uses two scalar channels (`0e`), distinct
sender features and an empty receiver node. Inputs, edge order and outputs
are stored together. `fp32_reductions.capture.json` records the exact command,
seed, environment and deterministic settings of the initial capture.

The gate is **maximum absolute error <= `numpy.finfo(numpy.float32).eps`**,
as PAR-2 requires. This is an accumulation-order golden, not a general
cross-platform tolerance for a neural network. Inputs are dyadic numbers and
the scalar coupling is exactly one: platform-dependent transcendental and
matrix-multiplication approximations cannot obscure the reduction order.
An independent sequential NumPy fp32 fold checks both stored outputs; the
frozen legacy CPU scatter also checks the segment reduction. Reversing the
edge order must fail each golden separately. Full-model energy, force and
stress comparisons retain the existing `FP32` and `FP64_CPU_REFERENCE` rows.

Run the check without writing any reference:

```sh
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 python -m pytest tests/parity/test_fp32_bit_golden.py -v
```

The `fp32-reductions` target was used only for the initial capture authorized
by PAR-2. It refuses to overwrite an existing reference or sidecar and is
excluded from `--target all`. Existing goldens and the shared tolerance table
are unchanged. FM-2's use of this reference for converted artifacts is a
separate ticket; this fixture does not claim to verify those artifacts.
