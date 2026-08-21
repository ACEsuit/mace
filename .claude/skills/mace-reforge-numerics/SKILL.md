---
name: mace-reforge-numerics
description: Physics and numerics guardrails for MACE Reforge - units, force/stress/virial sign and Voigt conventions, equivariance tests, double-backward requirements for force training, per-op-class precision, determinism, and the capability-marker contract for GPU tests. Use when writing or reviewing anything that computes energies, forces, stress, dipoles, a derivative, or a kernel.
---

# mace-reforge-numerics

The rules that keep the rewrite physically correct. Companion to
`mace-reforge-goldens`, which covers how the numbers are pinned.

## Units and conventions

- **Units are eV and Angstrom.** Align constants with `ase.units`; use the
  package's own units module where one exists rather than a local constant.
- **forces = -dE/d(positions)**
- **stress = (1/V) dE/d(strain)**
- **virials = -stress * V**

Sign and Voigt conventions are pinned by committed tests. **Read those tests'
docstrings before touching any derivative path** - a sign or an ordering error
here reproduces perfectly on your machine and destroys an MD run three months
later.

State units, shapes and conventions in the docstring of every function that
returns a physical quantity. A shape comment is not optional documentation here;
it is the only thing standing between a `[n_atoms, 3]` and a `[3, n_atoms]`.

## Equivariance is a test, not an argument

Every new block or model needs rotation, translation and permutation tests:
rotate the inputs and the outputs must rotate accordingly; energies must be
invariant. A block that is equivariant by construction still gets the test - the
test is what catches the layout bug that construction cannot.

## Double-backward is not optional

Force training backpropagates through the forces (`create_graph=True`), so
**every kernel op - reference or accelerated - needs a differentiable backward.**

- `gradgradcheck` is the gate. An op that does not pass it is not done.
- A backend that cannot support it must declare `supports_double_backward=False`
  and is then **rejected at build time** for force training. It is never silently
  wrong at runtime.
- This applies to the plain reference implementation too. "It is only the
  reference path" is not an exemption; the reference is what everything else is
  compared against.

## Precision

- **No global AMP.** Dtype is assigned per op class through `PrecisionConfig`.
- Linear layers may run in low precision; radial and cutoff ops high;
  **reductions and scatter at the highest precision** - that is an MD-stability
  requirement, not a preference.
- When in doubt about a dtype, it becomes a `PrecisionConfig` decision, not a
  hardcoded cast at the call site.
- **fp32 results depend on the thread count and the reduction order.** Never pin
  an fp32 expectation tightly, and never compare an fp32 number produced on one
  machine against one produced on another outside the `FP32` tolerance row.

## Determinism in tests

Fixed seeds. No dependence on wall-clock time or on the network outside tests
marked `network`. Where a reduction order is not deterministic and the test needs
it to be, make that explicit in the test rather than hoping.

## The capability-marker contract

Markers (`gpu`, `cueq`, `oeq`, `polar`, `les`, `torchsim`, `schedulefree`,
`network`, `bin_lammps`) are enforced by `tests/conftest.py`: a missing
capability **skips** locally, but a CI job that exports `MACE_REQUIRE_CAPS`
**fails** when a capability it guarantees is broken. `--strict-markers` is on, so
a typo in a marker name is an error rather than a silently unmarked test.

Two things about that guard that are easy to get wrong:

- It counts **collected** tests, not **selected** ones - it runs before the mark
  expression deselects. So listing a capability that the job's `-m` expression
  cannot reach is an empty promise the harness will not catch.
- Directory-derived markers are added automatically during collection. Put a
  test in the directory that matches its requirements instead of hand-marking it.

## GPU tests and vendor selection

- **Tests always write `device="cuda"`**, which is valid on ROCm - torch's ROCm
  build keeps the CUDA API names. Do not branch on a vendor inside test code,
  and do not write a device string per vendor.
- **Vendor selection happens in the marker expression**, per runner: the Nvidia
  tier runs `-m gpu`, the AMD tier runs `-m "gpu and not cueq"` because
  cuEquivariance is CUDA-only. OpenEquivariance is the AMD-capable accelerated
  backend.
- A capability is listed for a job only where the marker expression can actually
  reach a test for it.

If your ticket carries the `gpu` label, either run the selection on a GPU host or
say in the PR that you are relying on GPU CI. A CPU-only run is not verification
of a GPU ticket.
