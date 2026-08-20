# The LES goldens

Target: `les`, **not** part of `all` — it needs the optional `les` package at
the commit `requirements/les.txt` pins.

`tiny_maceles.model` is a third anchor and the only one that skips: it is
`les`-marked, so it runs where the `[les]` extra is installed and fails
rather than skips in the two CI jobs that promise it (ci-extensions `les`,
nightly `coverage-full`). Same backbone as `tiny_scaleshift` — same cutoff,
irreps, species and ZBL — plus the long-range head, and
`keep_last_layer_irreps=True`, which `MACELES` forces on its base class so
the LES readouts can see vector features.

Three things make it different from the other two.

**Its configuration is a committed file, not a constructor call.**
`models/tiny_maceles.les_arguments.yaml` is read through the same `read_yaml`
that backs `--les_arguments`, and `MACELES` passes the same dict on to
`les.Les`, which reads a second, disjoint set of keys from it. Between them
they decide which readouts exist, what shape each latent quantity comes back
in, and how the Ewald sum is evaluated — none of which is recoverable from
the weights. A test asserts the argparse round trip, so the yaml is the
recipe rather than a description of it.

**Its numbers belong to an external library, and the reference says which
one.** Both LES references record the `les` commit in their provenance, and
the first thing the test file asserts is that the installed solver is that
commit — reported as a named mismatch rather than as eight simultaneous
tolerance failures. This is not defensive: the two comparisons in
`tests/extensions/les/test_maceles.py` are `xfail`ed precisely because their
hardcoded energies were generated against an unrecorded `les`, and nothing
can now say whether the model or the solver moved.

**Its latent scales are not the library defaults**, and that is a golden
decision rather than a modelling one. An untrained readout emits O(1e-2); the
default `kappa_scale`/`alpha_scale` of 0.01 take two of the five latents to
1e-4 and the positivity flags then *square* them, landing at 1e-8 and 1e-7 —
below the 1e-6 absolute floor of the row they are asserted at. A reference
whose numbers are smaller than its own tolerance is reproduced by returning
zeros. Unit scales put every latent at 1e-4 or above, and a test asserts that
property rather than trusting the yaml comment that explains it.

Two references, because no single surface carries the family:

| reference | surface | what only it can pin |
|---|---|---|
| `tiny_maceles_e3nn_cpu_fp64.json` | the forward, all six fixtures | `les_energy` and all five latent quantities; the calculator exposes three of them and drops the rest |
| `tiny_maceles_field_cpu_fp64.json` | `MACECalculator`, two fixtures | `external_field`, `eps_infty`, `keep_neutral`, `electric_field_unit` — a Born-charge force correction applied *after* the forward returns |

The field reference is two fixtures and not six because with `eps_infty` set
the calculator divides by `atoms.get_volume()`, and ase refuses a volume for
a cell that is not full rank: both aperiodic fixtures, the dimer and the
zero-vacuum slab raise before any MACE code runs. That refusal is pinned as a
contract so the short fixture list is a documented limit rather than a
trimming nobody recorded.

## Regenerating

Outside `all` deliberately: it needs the optional extra installed at the
pinned commit, and it refuses to write anything when that commit cannot be
established from the install metadata. Folding it into `all` would let a
machine without `les` leave those two references describing an older solver
than their provenance claims.

```bash
python tests/golden/regenerate.py --target les --i-know-what-i-am-doing
```

## Running

`test_tiny_maceles.py` skips cleanly where `[les]` is absent and the rest of
the directory is unaffected; it **fails** where a job promises the
capability, which is why the ci-extensions `les` job runs
`tests/extensions/les tests/golden` and nightly `coverage-full` runs the
whole suite with `les` in `require-caps`.

```bash
python -m pytest tests/golden -m les -v
```
