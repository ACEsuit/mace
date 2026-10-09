# The user-observable output surface

"v1 has the same functionality as develop" is a claim about the names a user
can read out, not about the names one function happens to return. A key that
reaches the user only through the ase calculator, or only through
`mace_eval_configs`, is functionality just the same, and a completeness gate
keyed on the model `forward` alone cannot see it.

The surface is therefore the union of three layers.

| layer | read from | owner | keys | new at this layer |
|---|---|---|---|---|
| (a) model forward | `mace/modules/models.py`, `mace/modules/extensions.py` | CORE-1 (#1555) | 43 | 43 |
| (b) ase calculator `results` | `mace/calculators/mace.py` | DEP-1 (#1583) for the energy family, DEP-1a (#1634) for dipole / dielectric / polar / LES / magnetic | 31 | 15 |
| (c) `mace_eval_configs` | `mace/cli/eval_configs.py` | CLI-1 (#1579) | 13 | 3 |
| **union** | | | | **61** |

61, not 43, is the number "the output surface survived the rewrite" is measured
against.

## Deriving it, rather than trusting this table

Every number above is extracted from the frozen tree by
`tests/golden/surface_scan.py`, the same mechanical scan run at three sites,
and `tests/architecture/test_observable_completeness.py` re-derives all four
and fails if this table disagrees. Each owning ticket runs the extraction over
its own layer rather than copying a number from here: a hand-kept list that
looks complete and silently is not is the defect this whole exercise exists to
remove.

Two traps the scan already accounts for, both of which shrink the surface
quietly when missed:

- Layer (a) must follow keys **assigned onto the returned object**, not only
  dict literals. The self-consistent magnetic model assigns three diagnostics
  after building its output, so an extraction that stops at return literals
  stops at 40.
- Layer (b)'s committee keys must be read off the code, not off
  `implemented_properties`. The loop emits both a `_comm` and a `_var` suffix
  for all four members of the ensemble store, while `implemented_properties`
  advertises four of those eight: `forces_var`, `stress_comm` and
  `dipole_comm` are produced and never declared.

## What is new at each layer

**(b) exists only at the calculator**, 15 names: `free_energy` and `energies`
(the aliases and the E0-inclusive per-atom energy), `stresses` (the Voigt
per-atom rename of the model's `atomic_stresses`), `LES_alphas`, `LES_kappas`,
`bec` (the lower-cased mean of the model's `BEC`), `MACE_magmoms`, and the
eight committee keys `{energy,forces,stress,dipole}_{comm,var}`.

**(c) exists only at the evaluation CLI**, 3 names, and each is a *rename* of a
model key, which is exactly why they are easy to lose: `BO_contributions` (model
`contributions`), `descriptors` (model `node_feats`, after invariant extraction
and layer truncation), `node_energies` (model `node_energy`). Only `energy`,
`forces` and `stress` are shared with the calculator; the other ten names that
layer writes reach the user through this CLI alone.

## Layer (a), key by key

CORE-1 owns layer (a) and classifies all 43 in
`tests/architecture/observable_coverage.py`: each key becomes a declared
`ObservableSpec`, a derivative of one under the `d_<q>_d_<x>` rule, or a row
saying explicitly that it is not an observable and naming the mechanism that
owns it instead. The test beside that file fails on a key with no row, so a key
added to a legacy forward cannot pass unclassified.
