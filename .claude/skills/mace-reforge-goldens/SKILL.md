---
name: mace-reforge-goldens
description: Work with the MACE Reforge frozen oracle - the committed golden references in tests/golden and the in-process parity suite. Use when adding or reading goldens, choosing a tolerance, debugging a golden or parity mismatch, or when asked to regenerate a reference. Also use before touching anything that changes energies, forces or stress.
---

# mace-reforge-goldens

The rewrite is measured against committed numbers. `tests/golden/` holds them;
`tests/parity/` compares the two stacks live, in one process. Nothing in either
is regenerated as a side effect of another change.

## Layout

```
harness.py            fixtures, snapshot schema, tolerance table, comparison
fixtures/             committed .xyz structures + manifest.json
models/               committed anchor checkpoints + build sidecars
references/           committed expected-output JSON
regenerate.py         the only thing allowed to rewrite any of the above
targets/              one module per family: what regenerating that family means
docs/                 one page per family: what its goldens pin and why
surface_scan.py       derives the channel/key schema out of mace/ by AST
feature_inventory.md  the capability ledger, gated by check_inventory.py
```

Two structural properties, both of which are load-bearing:

- **A family of goldens owns exactly two files** - a module in `targets/` and a
  page in `docs/` - and edits no shared list. `regenerate.py` discovers its
  targets rather than naming them. This is a merge property, not filing
  preference: the moment two families have to append to one enumeration, every
  pair of them conflicts. When you add a family, add those two files and touch
  nothing shared.
- **`harness.py` imports no framework** - standard library, numpy and ase only.
  The parity suites that consume it are forbidden from importing the legacy
  tree, so a single convenience `import torch` here would make the shared
  comparison machinery useless to the tests it exists for. Two tests enforce it,
  one by grepping the source and one by importing it in a subprocess with the
  framework blocked on `sys.meta_path`. Anything framework-specific belongs in
  the test files or the build scripts.

## Tolerances

**The table in `harness.py` is the single source of truth.** Rows are used
exactly as `numpy.isclose` does - `|a - b| <= atol + rtol * |b|`, with `b` the
committed reference:

| Row | atol | rtol | For |
|---|---|---|---|
| `FP64_CPU_REFERENCE` | 1e-6 | 0 | fp64 CPU, cross-machine (eV, eV/Angstrom) |
| `FP64_ACCELERATED_BACKEND` | 1e-5 | 0 | cueq/oeq fp64 vs the reference |
| `FP32` | 5e-5 | 1e-3 | anything fp32 |
| `CLOSED_FORM_FP64` | 1e-12 | 1e-12 | closed-form quantities |
| `EXACT` | 0 | 0 | must reproduce bit for bit |

Pick a row by name; never inline a number in a test. **Tolerances are
edit-locked**: changing one is a separate, justified, reviewed PR - never part of
a feature PR, and never the fix for a failing test.

**Bit-exactness holds only within one process, and only with the same gradient
seeds.** Across processes or against a committed file, compare through the
harness table. An `EXACT` expectation written against a file will eventually
fail for reasons that have nothing to do with the change under test.

## When a golden or a parity test fails

**The default assumption is that your change is wrong.** Work through it in this
order:

1. Reproduce the mismatch and read the actual numbers. A mismatch at 1e-16 and
   one at 1e-3 are different bugs.
2. Debug in one process - the legacy package is in the tree and importable
   alongside `packages/`, so no second venv is needed.
3. Suspect shared global state before suspecting the physics. `mace/__init__`
   sets `TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD`, and `e3nn.set_optimization_defaults`
   is process-wide. The parity harness snapshots and restores this state around
   each comparison, or forks a subprocess per engine where in-process isolation
   cannot be guaranteed. **Do not "fix" a parity mismatch by silencing that
   snapshot/restore step** - it is the thing keeping the comparison honest.
4. If the numbers really do disagree and your implementation is right, you have
   found a discrepancy in the oracle. Flag it on the ticket. Do not encode a
   workaround, and do not average the two behaviours.

What is never a fix: widening a tolerance, regenerating the reference, marking
the test xfail, or adding a dtype cast that happens to make the numbers line up.

## The oracle is internally inconsistent about dtype - pin both sides

In `mace/modules/models.py`, total energy is computed in the model's own dtype
while node energy is computed unconditionally in fp64. Total energy and node
energy are therefore pinned as **separate goldens, in both fp32 and fp64, and
separately for `MACE` and `ScaleShiftMACE`** - those two differ in whether the
ZBL/pair-repulsion term falls inside or outside the scale-shift, so their
goldens diverge on the same structure.

Do not resolve this by picking one dtype policy for both quantities. Reproduce
the split. Where `PrecisionConfig` needs a dtype decision here, make it
per-quantity, not a global default.

## Regenerating a reference

Only through `tests/golden/regenerate.py`, only in a **dedicated PR**, with a
physics justification in the description. Never inside a feature PR. The
committed fixtures, checkpoints and reference JSON are the contract the whole
rewrite is measured against; they travel with the repository on purpose (the
`.gitignore` patterns for `*.xyz`, `*.model` and `*.pt` carry explicit negations
for them), and a golden that changes quietly is worse than no golden.
