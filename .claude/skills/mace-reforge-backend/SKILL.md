---
name: mace-reforge-backend
description: The MACE Reforge kernel-backend contract - how framework, kernel backend and kernel differ, entry-point registration, torch.library.custom_op requirements, build-time dispatch, the canonical weight layout, the reduced Clebsch-Gordan basis, the conformance suite, and the electrostatics solver analogue. Use when writing or reviewing a kernel, a backend, the dispatch layer, or anything touching weight layout or the CG basis.
---

# mace-reforge-backend

The op layer. This is a frozen spec - implement it, do not redesign it. If your
ticket seems to require a change here, flag it on the ticket.

## Three things that must never be conflated

- **Framework** - torch or jax. The tensor type, the autograd, the compiler. It
  is *which package you installed*, not a runtime choice.
- **Kernel backend** - a *set* of op implementations *for one framework*:
  reference, cueq, oeq, yours.
- **Kernel** - one op implementation: forward, backward, meta.

**A backend is therefore never framework-independent.** Its ops are typed on the
framework's tensor and register through that framework's autograd. Any design
that tries to share a kernel across frameworks is wrong at the first line.

`mace_core` owns only what is *data about* the ops: descriptors, `weight_numel`,
the canonical spec, the Clebsch-Gordan basis, capabilities, and the
tensor-**generic** Protocol (`Generic[TensorT]`, the same trick as `MACEOutput`).
Dtypes there are **names** (`Literal["float64", ...]`), never `torch.dtype`,
because `mace_core` imports no framework.

## Registration and dispatch

- Registration is **per framework**, through entry-point groups
  `mace.kernel_backends.torch` and `mace.kernel_backends.jax` - the same split
  `cuequivariance` makes between `cuequivariance_torch` and `cuequivariance_jax`.
- **The plain-torch reference backend is mandatory and CPU-capable.** Everything
  else is optional acceleration.
- **Dispatch resolves once at model build time** and is frozen into the module
  tree. Nothing is resolved inside `forward`: no dtype check, no device check, no
  `isinstance`, on the hot path.
- Capability negotiation is per op: a backend returning `NotImplemented` falls
  back to the reference **op by op**, not wholesale.

## The hot ops

`linear`, `channelwise_tp_conv`, `symmetric_contraction`, `fully_connected_tp`
are `torch.library.custom_op` with `register_fake`, each with a **differentiable
backward** so double-backward (force training) is correct on the reference and a
hard requirement for any accelerated backend. See `mace-reforge-numerics` for the
`gradgradcheck` gate.

## Weight layout

- **A canonical weight layout means one checkpoint loads into any backend.**
  There are no weight-conversion CLIs, and none should be added.
- Weight conversion happens at the **save/load boundary** and is free.
- **Activation** layout is resolved once for the whole op chain and is never
  converted inside `forward`. Those two conversions are different things and only
  one of them is cheap; conflating them puts a transpose on the hot path.

## The Clebsch-Gordan basis

- **`reduced` is the only value for new training.** A full-basis artifact
  converts to reduced **at load**, exactly, in fp64.
- **The basis and the full<->reduced conversion are computed from first
  principles** - no e3nn, no cuequivariance. Deriving either from a third-party
  library is the bug this replaced.
- The **path order and normalization are pinned, and that is the on-disk weight
  format.** Goldens lock the tensors, not the counts.
- One basis on every device and every backend. The legacy behaviour - a CLI flag
  defaulting to off, an environment variable, and whether `cuequivariance`
  happened to be importable - meant every model defaulted to a roughly 3x
  over-parametrized symmetric contraction and an explicit request was silently
  downgraded. A golden pins the parameter count per `(irreps, correlation)` so
  that coupling cannot come back.

## The conformance suite

Any kernel backend - in-tree or third-party - proves itself against the same
parametrized suite:

- equivariance
- parity against the reference
- `gradcheck` and `gradgradcheck`
- weight round-trip
- compiles with no graph break
- **capability honesty** - what it claims to support is what it supports

A **span factory** (an optional factory that fuses a wider span of the
interaction layer in one kernel, e.g. conv -> linear -> skip_tp ->
symmetric_contraction) is negotiated exactly the same way as any single op, and
runs the same suite.

## The electrostatics solver is the same pattern, one op class over

The long-range/electrostatics solver (k-space or PME, plus an optional SCF fixed
point) dispatches like any other op class, with its **own registry**: a
plain-torch reference solver and accelerated libraries behind it. The op is the
k-space solve, with the SCF loop as an optional span.

**Where the analogy breaks, and it matters:** a bit-parity accelerated solver is
a free backend choice, but a **non-bit-parity solver is model-affecting state**,
not a free choice - it has to be recorded as part of the model, because the
numbers it produces are part of what the model is. Do not treat solver selection
as interchangeable the way you would treat a `linear` kernel.
