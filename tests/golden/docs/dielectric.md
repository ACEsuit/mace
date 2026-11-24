# Dielectric anchor

Target `dielectric` (`targets/dielectric.py`).

## What it pins

`models/tiny_dielectric.model` is a seeded, untrained fp64
`AtomicDielectricMACE`, the class behind MACE-MDP, built by
`build_dielectric_anchor.py`: two layers, correlation 2, H/C/O, and
`4x0e + 4x1o + 4x2e` both as hidden features and as the readout's
`MLP_irreps`. `references/tiny_dielectric_e3nn_cpu_fp64.json` records every
channel its forward returns with `compute_dielectric_derivatives=True`:
`charges`, `dipole`, `atomic_dipoles`, `polarizability`, `polarizability_sh`,
`dmu_dr` and `dalpha_dr`. The fixtures are `dimer_short`, `isolated_atom` and
`water_cluster`.

## Why these

The class reads out a charge, a dipole and a polarizability from every layer,
and its last layer keeps `2e` features. Without `2e` in the hidden features
the anisotropic polarizability is exactly zero, so the anchor carries it, and
`test_the_reference_pins_a_response_and_not_zeros` checks that the water
cluster's anisotropy, dipole and Jacobians are not zero.

The fixtures are the molecular H/C/O set, named rather than selected by tag, so
a fixture added to the shared manifest does not widen this reference without a
regeneration. Periodic fixtures are left out because a dipole under periodic
boundaries depends on the choice of cell.

None of the fixtures carries a total charge, so the reference is taken on
neutral structures. Two tests cover what it cannot: the charges move by
`Q / N` and the dipole by `(Q / N) sum_i r_i` in e*Angstrom when a total
charge `Q` is set, and the Jacobians summed over atoms are `Q` times the
identity for the dipole and zero for the polarizability.

The values are those of the legacy forward, including the dipole baseline in
e*Angstrom where the dipole class converts to Debye
(`test_tiny_dipoles.py::test_the_two_fixed_charge_baselines_do_not_share_a_unit`).
The published MACE-MDP model is pinned separately by `test_mdp_foundation.py`,
which needs a download.

## Regenerating

Legacy only, on CPU, with no download and no optional package. It is part of
`--target all`.

```bash
python tests/golden/regenerate.py --target dielectric --i-know-what-i-am-doing
```

## Running its tests

```bash
python -m pytest tests/golden/test_tiny_dielectric.py
```

No capability marker: the anchor runs on every pull request.
