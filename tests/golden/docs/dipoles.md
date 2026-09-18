# The dipole and polarizability goldens

Targets: `dipoles` (the tiny anchor, part of `all`) and
`foundation-references` (the two published models, not part of `all`).

These families are snapshotted through the model forward rather than through
a calculator; see [routes.md](routes.md) for why an energy-less model cannot
be driven as an ase calculator at all.

## The tiny dipole anchor

`tiny_dipoles.model` is an `AtomicDipolesMACE`, built by direct instantiation
and committed as initialised: fp64, seeded, two interaction layers,
`16x0e + 16x1o`, three species, 1.37 MB. It carries no ZBL and no E0 table
(the class asserts `atomic_energies is None`), because it is not an energy
model: what it pins is the dipole assembly, which shares no arithmetic with
an energy.

Its reference is taken on the **molecular** fixtures only. A dipole is
defined up to an origin, and under periodic boundaries the choice of cell
moves it, so a slab golden would pin a deterministic number with no physical
meaning.

## The published-model references

Two references pin models nobody in this repository trained. They exist
because the conversion work is gated on them, and because two of the
quantities in them appear nowhere else.

| reference | model | what it pins | marker |
|---|---|---|---|
| `polar_foundation_cpu_fp64.json` | MACE-Polar (`polar-1-s`), `PolarMACE` | energy/forces/stress **and** the electrostatics surface: dipole, charges, spins, the three energy decompositions, the density coefficients, the spin-resolved density, the Fukui functions | `polar`, `network` |
| `mdp_foundation_cpu_fp64.json` | MACE-MDP, `AtomicDielectricMACE` | dipole, atomic dipoles, charges, **polarizability** (Cartesian and spherical) and the two position derivatives `dmu_dr` / `dalpha_dr` | `network` |

**Which class emits what is not interchangeable.** `AtomicDielectricMACE` is
the only class in the tree that emits a `polarizability`. `PolarMACE` emits a
dipole and its electrostatics and no polarizability at all — the word does
not occur in the class, which is asserted rather than remembered. So the
polarizability golden is the MDP one, and a test that "checks the
polarizability of the polar model" would be checking nothing.

**Dtype discipline.** Every loader call spells out `device="cpu"` and
`default_dtype="float64"`. `mace_polar` defaults to **float32** and both
default to CUDA when one is visible, so a golden that took the defaults would
be an fp32 GPU snapshot asserted at the fp64 CPU row. The MACE-Polar weights
are published as float32 and are upcast, which is exact; the arithmetic is
genuinely fp64.

**Headroom.** Re-snapshotting all three new references on one host at 1, 4
and 8 torch threads gives a deviation of exactly `0.0` on every channel, so
the `fp64_cpu_reference` row is carrying the cross-machine term alone — the
same measurement, and the same result, as the energy anchors.

**One caveat these references cannot fix.** The `dipole` channel is declared
in Debye, and the three families do not agree on that. `AtomicDipolesMACE`
converts its fixed-charge baseline to Debye; `AtomicDielectricMACE` has the
same division commented out and `PolarMACE` builds its dipole from charge
times position with no conversion at all. The schema keeps one channel,
because three would mean the families could never be compared, and the 4.8032
discrepancy is pinned as a number in `test_tiny_dipoles.py` so a rewrite that
unifies the two helpers fails there rather than silently rescaling a
committed reference.

**The MDP reference goes through the forward, not the calculator**, because
`dmu_dr` and `dalpha_dr` are in no `results_map` entry and so reach no
calculator at all. The calculator's four shared channels are then asserted
against that same file — one number, two doors — which is how the `mace_mdp`
calculator path is pinned.

## Regenerating

`dipoles` is part of `--target all`. `foundation-references` is not: it
downloads published models and needs `graph_longrange`, and folding it into
the default sequence would make `--target all` fail on a plain development
box that can regenerate everything else. Ask for it by name:

```bash
MACE_CI_ALLOW_NETWORK=1 python tests/golden/regenerate.py \
    --target foundation-references --i-know-what-i-am-doing
```

## Running

The tiny anchor carries no marker and runs in the ci-core `unit` job with
everything else. The two published-model references are downloads, so they
carry markers and have jobs that guarantee those markers, which is what keeps
a skip from reading as a pass:

| where | selection | guarantees |
|---|---|---|
| ci-core `unit` (per PR) | `tests/unit tests/golden` | nothing; the foundation goldens skip |
| ci-extensions `polar` (per PR, paths-filtered on `tests/golden/**` among others) | `tests/extensions/polar tests/golden` | `polar,network` — the only PR-time job with `graph_longrange`, so the MACE-Polar golden is a result here and a skip everywhere else |
| nightly `foundations` | `tests/foundations tests/golden` | `polar,network` |
| nightly `coverage-full` | `tests` | `network,cueq,polar,les,torchsim,schedulefree` |

```bash
MACE_CI_ALLOW_NETWORK=1 python -m pytest tests/golden -v
```
