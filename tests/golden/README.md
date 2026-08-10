# `tests/golden/` — committed numerical references

This directory holds the numbers the rewrite is measured against: a
framework-agnostic comparison harness, a fixed set of structures, two tiny
model checkpoints, and the outputs those checkpoints produce today. Nothing
here is regenerated as a side effect of another change.

```
harness.py            the shared machinery: fixtures, snapshot schema,
                      tolerance table, comparison
fixtures/             committed .xyz structures + manifest.json
models/               committed anchor checkpoints + their build sidecars
references/           committed expected-output JSON
make_fixtures.py      seeded fixture generation
build_mace_anchor.py  the plain-MACE anchor (direct instantiation)
train_anchor.py       the ScaleShiftMACE anchor (training CLI)
regenerate.py         the only thing allowed to rewrite any of the above
```

## The harness never imports the framework

`harness.py` depends on the standard library, numpy and ase, and on nothing
else. That is a structural requirement, not tidiness: the parity suites that
consume it live outside the legacy tree and are forbidden from importing it,
so a single convenience import here would make the shared comparison
machinery unusable to exactly the tests it exists for. Two tests enforce it —
one greps the source, one imports the module in a subprocess with the
framework blocked on `sys.meta_path`.

Anything framework-specific — building a graph, loading a checkpoint — lives
in the test files or in the build scripts, which may import freely.

## The tolerance table

Defined once, in `harness.py`, as three rows:

| row | atol | rtol | what it covers |
|---|---|---|---|
| `fp64_cpu_reference` | 1e-6 | 0 | fp64 e3nn on CPU, cross-machine and cross-Python |
| `fp64_accelerated_backend` | 1e-5 | 0 | cueq/oeq fp64 on an accelerator vs the CPU reference |
| `fp32` | 5e-5 | 1e-3 | fp32 anywhere |

Import the row; never restate a number in a test file. A test enforces that
too, by scanning every other module here for a literal `atol=`/`rtol=`.

The table is reconciled with the two that already exist in the tree rather
than competing with them, and the rationale for each row is in the source
next to it. In short: `tests/backends/backend_parity.py` measures
gradient parity in one process on one device and agrees with the 1e-6 here;
`tests/extensions/polar` measures a committed-JSON regression and is tighter
at fp64 (1e-9) and looser in kind at fp32 — its measured 5e-5 absolute floor
is adopted here rather than re-invented, because that file documents 5e-6
failing in CI.

Tolerances are **edit-locked**. Changing one is a separate, justified,
reviewed change. A test that needs a looser number is a test that found
something.

## The fixtures

Six structures, each the only one reaching a distinct regime of the
neighbour-list layer (`mace/data/neighborhood.py`), which is what decides the
cell a stress is divided by:

| fixture | pbc | regime |
|---|---|---|
| `triclinic_bulk` | T T T | non-orthogonal cell, physical cell returned |
| `water_cluster` | F F F | fully aperiodic — the *extended* search cell is returned |
| `isolated_atom` | F F F | zero edges |
| `dimer_short` | F F F | 0.62 Å C–O separation, deep in the repulsive wall |
| `slab_vacuum` | T T F | mixed pbc with real vacuum, physical cell returned |
| `slab_zero_vacuum` | T T F | mixed pbc with an all-zero vacuum row — the patched-row branch |

Measured on the committed files at `r_max = 3.5` Å, the returned cells are
`(4.30, 4.10, 4.45)`, extended to about `(10.9, 12.2, 9.2)`, `8³`,
`(8, 8, 8.62)`, `(4, 4, 12)` and `(4, 4, 9.465)` respectively — the last pair
differing only in that patched row, which is why their stresses differ by
exactly the volume ratio while their energies are identical.

Species are drawn from {H, C, O} only, so one three-element anchor covers the
whole set. Geometries come from `make_fixtures.py` under a single seed
(`20260810`); `fixtures/tiny_train.xyz` is the seeded synthetic training set
the trainable anchor is fitted on, and is not an evaluation fixture.

The short dimer is C–O rather than H–H on purpose. The repulsion term's
envelope cuts off at the sum of the pair's covalent radii, so two hydrogens
at 0.6 Å would sit at 0.97 of *their* cutoff and contribute almost nothing —
the opposite of what the fixture is for. At C–O (1.42 Å radii sum) the term
is ~10.9 eV and the envelope is far from both ends.

## The two anchors

Both are fp64, seeded, two interaction layers, `16x0e + 16x1o`, three
species, ZBL repulsion on, ~1.07 MB each.

* **`tiny_scaleshift.model`** — a `ScaleShiftMACE` trained by
  `train_anchor.py` on `fixtures/tiny_train.xyz`. This is the class the
  training CLI emits.
* **`tiny_mace.model`** — a plain `MACE`, built by direct instantiation in
  `build_mace_anchor.py` under a fixed seed and committed as initialised.
  It is *not* trained and *cannot* come from the CLI: `--model MACE` returns a
  `ScaleShiftMACE` with `atomic_inter_scale=args.std` and the shift zeroed
  (`mace/tools/model_script_utils.py:279-296`), so a CLI recipe would silently
  anchor the wrong class. A seeded untrained network exercises the plain-`MACE`
  energy assembly exactly as a trained one would.

**Why both carry ZBL.** The pair term enters the two classes differently:
plain `MACE` appends it to `energies` next to `e0` and never scales it
(`mace/modules/models.py:359-361`), while `ScaleShiftMACE` seeds its readout
sum with `[pair_node_energy]` (`:539`) and puts the whole sum through
`scale_shift` (`:579`). On the short-dimer fixture, removing the term moves
the total by 1.000000× the raw pair sum in the plain anchor and by 0.478465×
— exactly the model's scale — in the scale-shift one. `test_tiny_anchors.py`
asserts both ratios, which is what turns that divergence from a comment into
a gate.

**The plain anchor is not convertible.** `extract_config_mace_model`
whitelists `ScaleShiftMACE` and the extension classes and returns an
`{"error": ...}` payload for a plain `MACE`, so any accelerated-backend
parity work runs against the `ScaleShiftMACE` anchor. That refusal is pinned
as a contract here rather than worked around.

Each checkpoint has a `*.build.json` sidecar recording the exact command, the
seed, the dtype and the full configuration that produced it.

## Regenerating

```bash
python tests/golden/regenerate.py --target all --i-know-what-i-am-doing
```

It refuses to run without the flag. Targets are `fixtures`, `anchors`,
`references`, or `all` (in that order). Regenerating a golden discards the
evidence it was collected to provide, so it happens in its own reviewed
change and never inside a feature change — a regenerated reference turns a
failing test into a passing one without anyone having decided the new numbers
are correct.

`references/tiny_scaleshift_training_errors.json` is written by the same run:
it holds the final train/valid error table plus the last evaluation record
(MAE, RMSE and loss), which is what a "the rewrite trains comparably" claim
is checked against.

## Running

```bash
python -m pytest tests/golden -v
```

No GPU, no network, no optional dependency, and no capability marker — these
tests run in the ci-core `unit` job on all four supported Python versions.
An anchor that could skip would be an anchor that could rot.
