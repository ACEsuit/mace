# `tests/golden/` — committed numerical references

This directory holds the numbers the rewrite is measured against: a
framework-agnostic comparison harness, a fixed set of structures, two tiny
model checkpoints, and the outputs those checkpoints produce today. Nothing
here is regenerated as a side effect of another change.

```
harness.py            the shared machinery: fixtures, snapshot schema,
                      tolerance table, comparison
calculator_keys.py    what this repo's calculators call each channel, and
                      where every evaluation reads its inputs
model_keys.py         what the model forwards call each channel
eval_keys.py          what mace_eval_configs writes onto its structures
surface_scan.py       derives all of the above out of mace/ by AST, so the
                      guard checks the schema against the package rather
                      than against a remembered list
fixtures/             committed .xyz structures + manifest.json
models/               committed anchor checkpoints + their build sidecars
references/           committed expected-output JSON
make_fixtures.py      seeded fixture generation
build_mace_anchor.py  the plain-MACE anchor (direct instantiation)
train_anchor.py       the ScaleShiftMACE anchor (training CLI)
paths.py              where this directory sits
regenerate.py         the only thing allowed to rewrite any of the above
targets/              one module per family of goldens: what regenerating
                      that family means
docs/                 one page per family: what its goldens pin and why
```

## A family of goldens owns two files, and edits no shared list

Everything above is shared. Everything specific to one family of goldens — a
model class, a backend, a foundation checkpoint — lives in exactly two places
that no other family touches: a module in `targets/` that knows how to
rewrite it, and a page in `docs/` that says what it pins. Its build scripts
and its test file are named from there rather than added to the listing
above.

That is a merge property, not a filing preference. Families are added
independently and land independently; the moment two of them have to append
to one list, one enumeration in `regenerate.py`, or one section of this file,
every pair of them conflicts. `regenerate.py` therefore discovers its targets
instead of naming them, and this file describes the rule instead of listing
the families.

## One manifest, several periodic tables

**Every consumer selects its own subset by chemistry** —
`load_fixtures(elements=model.atomic_numbers)` — because the families do not
share a periodic table, and handing a three-element H/C/O anchor an iron
structure is a missing z-table entry rather than a tolerance failure. The
no-argument call returns everything, which is a set no single model can
evaluate; an empty selection raises rather than passing quietly.

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

## Nothing is dropped

A snapshot records every key an evaluation returns. A key that is not a
declared channel is an **error**, not a skip, because an output nobody
declared is an output nothing pins — and a reference that quietly recorded
three channels of a family while claiming to pin the family is worse than no
reference at all. Three ways out, in order of preference:

* `register_channel(name, kind, unit)` — a genuinely new quantity, declared
  once with its shape and its unit;
* `register_alias(spelling, channel)` — the same quantity under another
  name. This is not hypothetical: the calculator writes `LES_alphas`,
  `LES_kappas`, `bec` and `MACE_magmoms` where the model's forward returns
  `latent_alphas`, `latent_kappas`, `BEC` and `equilibrated_magmom`;
* `ignore_key(key, reason)` — an explicit, one-at-a-time allowlist entry with
  a written reason. Currently the committee spread statistics and the strain
  `displacement` handle, and only those.

## There are three surfaces, and they disagree

A quantity reaches the harness through one of three doors, and coverage of
one is not coverage of another. Each was added after the previous one turned
out not to be the whole story:

* the **calculator**, an ase `Calculator`'s `results` dict;
* the **model**, a `forward` return dict, reached through `golden_outputs`.
  The first version of the alias map was derived from
  `mace/calculators/mace.py` alone: all 31 calculator keys resolved and 13 of
  the 43 model-forward keys resolved to nothing. That gap is load-bearing —
  `edge_forces` and `hessian` are returned by every energy model and by no
  calculator;
* the **eval CLI**, which returns nothing at all: `mace_eval_configs` writes
  its results back onto the structures, into `info` or `arrays` under a
  caller-chosen prefix, and then writes extxyz. It emits 13 names, of which
  `BO_contributions`, `node_energies` and `descriptors` resolved to nothing —
  so any ticket pinning the eval CLI stopped at authoring time.

So registrations name their surface, and the guard derives its expectation
from the package. That derivation lives in `surface_scan.py` and is not a
regex: it follows how each surface is actually written.

| surface | derived from | forms it has to follow |
|---|---|---|
| calculator | every `class X(Calculator)` in `mace/` | `self.results[k]`, the aliased local (both assignment directions), `.update()` with a literal or a named dict, a whole-dict assignment, and the suffixed committee keys — whose bases come from the `results_store_ensemble` set literal the source guards them with, not from a copy of its members |
| model | every file defining a `forward` that returns a dict | `return {...}`, `out = {...}; return out`, and `out[k] = ...` on a dict from a nested call |
| eval CLI | `mace/cli/eval_configs.py` | `atoms.{info,arrays}[args.info_prefix + "name"]`, prefix stripped |

Two details of that table are the whole reason it exists. The model file list
is **discovered**, because the version that named `modules/models.py` and
`modules/extensions.py` was reading two writers out of four — the other two
are `calculators/lammps_mace.py` and `calculators/mace_torchsim.py`, which is
exactly what a deployment golden evaluates, and `total_energy_local` was the
one key in the whole package that no channel described. And every scan
reports the writes it **could not** resolve, with file and line, because an
extractor that finds nothing reports perfect coverage. The only unresolved
write allowed to stand is one with a written reason in `PASSTHROUGH_WRITES`
(there is one: the torchsim wrapper forwards the wrapped model's dict key by
key, so its key set *is* the model surface's).

Naming the surface buys two things a flat map cannot express:

* **one spelling, two quantities.** `virials` is the graph-level virial in
  every forward and the per-atom virial in the calculator's results, which
  has no key for the graph one at all. A flat map had to pick one and
  mis-shape the other.
* **one quantity, two layouts.** The per-atom stress is `(n_atoms, 3, 3)`
  from the model and Voigt-6 from the calculator. A plain alias is a shape
  failure; a channel each is worse, because both would hold the same physics
  and no comparison would ever put them side by side — the silent split, this
  time arrived at deliberately. Instead the layout is canonicalised on
  ingest: one channel holding the 3x3, and the calculator's Voigt-6 expanded
  as it arrives.

  Whether that is lossless is a measurement, not an assumption — Voigt-6
  cannot carry an asymmetric tensor. It holds because
  `get_atomic_virials_stresses` symmetrises explicitly
  (`mace/modules/utils.py:382`); measured on the `tiny_scaleshift` anchor over
  all six fixtures in float64, the asymmetry, the Voigt round trip and the
  difference between the two routes are all exactly `0.0`. A test re-measures
  it rather than trusting this paragraph.

## Inputs

`inputs` are compared in both directions, with no flag to disable it, and
**exactly** — never at the row the outputs use. A snapshot taken at different
moments, a different total charge or a different field is a different
measurement, not a drift, and an input is read verbatim off the committed
fixture rather than computed, so there is no numerical-agreement argument to
make. At the fp32 row a 2e-3 change to a 2.2 muB moment fits under the bound;
bcc iron is not a contrived magnitude.

Inputs are read from the same place the model reads them — moments from
`atoms.arrays[magmom_key]`, not from ase's initial-moments attribute, which
nothing in the forward pass looks at.

**Which place that is, is derived, not listed.** The first version named two
constructor arguments (`magmom_key`, `charges_key`) and by doing so said
nothing at all about `external_field`, `total_charge` or `total_spin`: a
reference recorded no field, and a snapshot taken at another field compared
clean against it. Adding the missing three would have been the same mistake
with a longer list, so the harness reads the mapping the reader is handed.
Both calculators keep a `KeySpecification`'s two dicts on the instance
(`info_keys`, `arrays_keys`) and `config_from_atoms` iterates them, so
`register_keyspec_source` reads the reader. A constructor argument added
tomorrow lands in one of those dicts and is picked up with no change here;
one that names a property no channel covers **fails the snapshot** and says
which, which is the outcome an enumerated list could not produce.

Four rules follow, and each one closes a way of pinning nothing:

* **an unknown property is an error**, exactly as an unknown output key is.
  `update_keyspec_from_kwargs` will put an arbitrary name in the mapping
  (that is how `embedding_specs` works), so the next input is not
  hypothetical.
* **a label is dismissed in writing.** Training targets travel in the same
  mapping and reach no `forward` — the batch keys the models actually read
  are a much shorter list — so each is declared a non-input with a reason
  rather than silently skipped.
* **an input the evaluation carries itself is recorded too.**
  `MACECalculator(external_field=…)` writes the vector into the batch after
  the graph is built, so it overrides the structure and appears in no array.
  It enters the energy and the BEC force correction.
* **an input under a key nobody reads is refused.** The two live spellings of
  the reference charges disagree by default — the training CLI writes
  `REF_charges`, both calculators read `Qs` — so a structure prepared by one
  and handed to the other has its charges ignored. That is the generalisation
  of the older ase-initial-moments refusal.

One table survives: the fallback spellings in `INPUT_ARRAY_KEYS` /
`INPUT_INFO_KEYS`, used when nothing better is known. It **cannot** be
derived, because deriving it means reading the framework's own key tables and
`harness.py` may not import the framework. So it is literal and a guard test
derives the truth — `DefaultKeys` for one spelling per property, the
`infos`/`arrays` lists in `update_keyspec_from_kwargs` for the store, each
calculator's own `__init__` defaults for the second spelling — and asserts
the table matches. Drift fails there.

## The tolerance table

Defined once, in `harness.py`, as four rows:

| row | atol | rtol | what it covers |
|---|---|---|---|
| `fp64_cpu_reference` | 1e-6 | 0 | fp64 e3nn on CPU, cross-machine and cross-Python |
| `fp64_accelerated_backend` | 1e-5 | 0 | cueq/oeq fp64 on an accelerator vs the CPU reference |
| `fp32` | 5e-5 | 1e-3 | fp32 anywhere |
| `exact` | 0 | 0 | the recorded inputs, always — not selectable for outputs |

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

It refuses to run without the flag. `--help` lists the targets, in the order
`all` runs them, and marks the ones `all` leaves out because they need a
download, an optional dependency or particular hardware; the list is read off
`targets/`, so it is never out of date with what is actually there.
Regenerating a golden discards the evidence it was collected to provide, so it
happens in its own reviewed change and never inside a feature change — a
regenerated reference turns a failing test into a passing one without anyone
having decided the new numbers are correct.

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

A family whose tests do need a marker — a GPU, a download, an optional
package — gives its selection line in its own `docs/` page, next to the
reason it cannot be part of the unmarked set.
