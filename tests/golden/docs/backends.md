# The accelerated-backend goldens

No regeneration target: this family commits no reference of its own.

`test_backend_parity_golden.py` converts the `ScaleShiftMACE` anchor (and
MACE-MP-0 small, where a job allows downloads) to cueq and to oeq, evaluates
it on an accelerator and asserts the committed CPU e3nn references at the
`fp64_accelerated_backend` row. Nothing new is written: the whole point is
that the accelerated numbers are held to the ones the plain CPU path
produced.

That creates a problem no value comparison can solve, and it is the reason
`backend_kernel_audit.py` exists. **Matching the reference is not evidence
that a vendor kernel ran** — the reference *came from* the unaccelerated
path, so the unaccelerated path reproduces it perfectly. Two ways to end up
there without noticing, both silent, both reproduced by CPU tests in that
file:

* `convert_e3nn_cueq.run` sets `conv_fusion=(device == "cuda")`, so
  converting with the default `device="cpu"` and moving the model afterwards
  yields a real cueq model whose conv path is the *unfused*
  `ChannelWiseTensorProduct`. Measured on the anchor: it reproduces the
  committed reference to **1.7e-16**, eleven orders of magnitude inside the
  1e-5 row. No tolerance catches this.
* `cuet.SegmentedPolynomial` falls back to `SegmentedPolynomialNaive` when
  `cuequivariance_ops_torch` cannot be imported — an ops wheel whose CUDA
  major does not match torch's, or a pin below the version that carries the
  fused kernels. It warns, sets `.method = "naive"`, and returns a module
  that computes the right answer with no kernel behind it.

So every backend case audits the module tree (is the accelerated
implementation in place, and is it the fused one?) and counts calls into it
with forward hooks (did it execute?). MACE's own wrapper refuses the naive
downgrade at construction time; the audit deliberately does not rely on that,
because the guard is code under test.

The audit is written so both halves can be exercised **without a GPU**, and
they are: on a host with `[cueq]` but no ops wheel — which is what the
`backends-cpu` CI job has — the degradation is real, not simulated. The
failure it guards against is a *passing* test, which is precisely the failure
that cannot be found by watching a GPU job stay green.

Parity runs on the `ScaleShiftMACE` anchor because the plain-`MACE` one is not
convertible; both converters are pinned to *stop* on the refusal payload
rather than return a model.

## Running

The cases are marked so that each lands in the job that can honour it:

| marks | where it runs |
|---|---|
| `gpu` + `cueq` | the Nvidia GPU job (`-m gpu`) |
| `gpu` + `oeq` | both GPU jobs — AMD selects `gpu and not cueq` |
| `gpu` + `cueq` + `network` | Nvidia only; it lists `network` in `MACE_REQUIRE_CAPS` |
| `cueq` (no `gpu`) | the `backends-cpu` extension job (`cueq and not gpu`) |

Under `MACE_REQUIRE_CAPS` a promised-but-missing capability fails instead of
skipping, so a broken backend install cannot make a vendor job green. The
`network` case is deliberately cueq-only: an oeq case carrying `network`
would also be collected by the AMD job, which promises no network test.

```bash
python -m pytest tests/golden/test_backend_parity_golden.py -m "gpu and cueq"
```

oeq JIT-compiles its kernels, so the host needs a CUDA or HIP toolkit and a
compiler new enough for `-std=c++20`; a stale default `c++` fails the build
rather than the comparison.
