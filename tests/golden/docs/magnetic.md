# The magnetic goldens

Target: `magnetic`, **not** part of `all` — regenerating needs the `magnetic`
extra (sphericart).

## The fixtures: five structures, {Fe, O}, carrying moments

The magnetic models take a per-node moment as an **input** and return
`magforces` = `dE/dm`, so their fixtures have to carry moments — under
`REF_magmom`, which is the array the models read, never ase's initial
magnetic moments. They are tagged `magmom`, and the magnetic goldens select
on that tag *and* the chemistry: `isolated_atom` is a lone oxygen, so the
element filter alone admits it, and a magnetic forward on a structure with no
moments is a `KeyError` before anything else happens.

| fixture | what only it reaches |
|---|---|
| `mag_fe_atom` | zero edges; the one-body magnetic term alone |
| `mag_fe_dimer_fm` | Fe2 at 2.02 Å, both moments +3.0 µB |
| `mag_fe_dimer_afm` | the *same geometry*, second moment reversed |
| `mag_fe3_canted` | 120° Néel state on a frustrated triangle, canted — the only non-collinear all-Fe case |
| `mag_feo_cluster` | planar Fe2O2 oxo core: two elements with different `m_max` |

The two dimers are the pair that makes the family falsifiable: identical
atoms, identical positions, identical |m|, so every difference between their
two reference entries is the spin state. A rewrite that dropped the moment
from the message passing would make them produce one number while every
non-magnetic golden still passed. The measured splitting on the committed
anchor is 9.5e-2 eV.

Three constraints on the moments, each stated in `make_fixtures.py` next to
the builders and each asserted by a test rather than trusted:

* **nothing saturates the clamp.** The radial magnetic basis is
  `1 - 2·clamp(|m|/m_max, 0, 1)²`, so a moment at or above `m_max` sits on
  the flat side and contributes exactly zero to `dE/dm` through that path — a
  structurally zero derivative with the same shape as a computed one. The set
  spans |m|/m_max = 0.25 … 0.89.
* **two elements, because `m_max` is indexed by species.** With one element a
  transposed lookup is invisible.
* **a zero moment is safe, measured.** The obvious worry — `torch.norm` at
  the origin — does not bite, because the norm enters only *squared*; on the
  committed anchor a site at m = 0 gets a finite `dE/dm` agreeing with a
  central difference. The oxygens carry 0.3 µB because superexchange leaves
  them one, not to dodge a NaN.

## The anchor

`tiny_magnetic.model` is a `MagneticScaleShiftMACE`, built by direct
instantiation (`build_magnetic_anchor.py`), fp64, seeded, two layers, Fe/O,
`use_magmom_one_body=True` so that flag is inside the numbers. It needs the
`magnetic` extra (`sphericart`) both to build **and to load**, so its two
test modules carry an explicit `@pytest.mark.magnetic` — a file in
`tests/golden` gets no directory-derived marker.

Its reference is taken through the **model surface**, not a calculator, and
that is forced: `MagneticMACECalculator` computes `dE/dm` and then keeps only
energy, forces, node_energy and stress, so the quantity this family exists to
pin cannot come out of it. The same reference is then checked against the
calculator and against `mace_eval_configs --return_magforces`, which is the
only way a user gets magnetic forces without writing python.

## The SCF reference, and why it covers three fixtures and not five

`MagneticSCFMACE` relaxes the moments with LBFGS and returns wherever it
stopped. The golden pins that state — `equilibrated_magmom`, the energy and
the forces — and records `scf_steps` and `scf_energy_history` as metadata,
because they describe how the optimiser got there rather than where it is.

Whether "wherever it stopped" *is* a fixed point is a measurement, and it was
made: perturb the initial moments by 1e-9 and see what comes back. On
`mag_fe_atom` and the two dimers the answer moves by ~1e-9 — the relaxation
tracks its input. On `mag_fe3_canted` and `mag_feo_cluster` it moves by 1e-5,
four orders of magnitude of amplification, so two runs of the same code on
two machines land further apart than the tolerance row allows. Those two are
excluded, and a named test re-measures the amplification so the exclusion
cannot quietly become wrong.

Two related facts are pinned rather than fixed, because both are properties
of the current implementation that a rewrite has to reproduce or deliberately
change: `n_scf_step` had to be raised well above the wrapper's default of 10,
since at 10 every structure stops because the budget ran out (measured: the
result is identical at 200 and 500, so the pinned state no longer depends on
the budget); and nothing constrains |m|, so the ferromagnetic dimer relaxes to
4.3 and 6.5 µB against an `m_max` of 4.5 — the magnetic descriptor uses
*solid* harmonics of `m`, which grow as |m|^l, so the energy is unbounded
below in |m| and confinement can only come from training.

## Running

The two magnetic modules are marked `magnetic`: they need `sphericart` to
unpickle their checkpoint at all. They skip in the core job and run in the
ci-extensions `magnetic` job, which guarantees the capability, so there is
exactly one place they can go green by absence and `require-caps` closes it.

```bash
python -m pytest tests/golden -m magnetic -v
```
