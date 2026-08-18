# Integration tests

An *integration* is an external runtime that consumes MACE models (LAMMPS
today; OpenMM, GROMACS, ... tomorrow). Each integration lives in its own
directory and is tested in **two tiers**:

1. **Contract tier** — everything testable *without* the external binary:
   export CLIs, artifact loading, and the data-layout contract the runtime
   relies on (e.g. LAMMPS's real-vs-ghost atom partition, reproduced by hand
   in `lammps/_harness.py`). Runs on PRs via `ci-integrations.yaml`
   (paths-filtered) and needs no special capability.
2. **Real tier** — drives the actual binary. Marked with the integration's
   capability marker (`bin_lammps`, ...), which skips locally when the binary
   is absent and *fails* in the CI job that guarantees it
   (`MACE_REQUIRE_CAPS`, see `tests/conftest.py`). Runs in `nightly.yaml`.

What the real tier can cover is bounded by the binary CI can install, and for
LAMMPS that bound is sharp: conda-forge builds every CPU variant with
`PKG_KOKKOS=OFF`, and `forward_exchange` — the ghost node-feature exchange
every interaction layer past the first needs — exists **only** in the KOKKOS
ML-IAP coupling. So the real tier runs a *single-layer* model, the multi-layer
path stays in the contract tier (`lammps/test_mliap_exchange.py`, which also
pins the actionable error the non-KOKKOS case now raises), and no bump of the
`lammps` package will change that.

`forward_exchange` is not the only KOKKOS-only call, and the second one is
worse because it lands *after* the model has run: the writeback in
`lammps_mliap_mace._update_lammps_data` used `data.eatoms` (a getter that
exists only in the KOKKOS coupling — the plain one declares it a
`write_only_property`) and `update_pair_forces_gpu` (KOKKOS-only outright).
A single-layer model on a stock build therefore still died, with
`property 'eatoms' ... has no getter` surfacing as a bare
`mliap_unified.cpp:71 compute_forces failure`. The writeback now branches on
the coupling, and both branches are pinned in `lammps/test_mliap_writeback.py`
with stubs that reproduce each `.pyx`'s property shape — the real tier can only
ever exercise the non-KOKKOS one.

## Adding integration X

1. Create `tests/integrations/<x>/` with contract tests (and a `_harness.py`
   if the data contract deserves one).
2. Register a `bin_<x>` marker in `pyproject.toml` and add its probe to
   `CAPABILITY_PROBES` in `tests/conftest.py` (real import, not `find_spec`:
   a broken install must read as unavailable).
3. Mark real-tier tests `@pytest.mark.bin_<x>`.
4. Add a paths-filtered contract job to `.github/workflows/ci-integrations.yaml`
   and a **blocking** real-tier job to `nightly.yaml`. Not
   `continue-on-error`: a job that cannot turn the run red is indistinguishable
   from a job that does not exist, and LAMMPS proved it by failing every night
   for weeks under a green nightly. Too flaky to block means land it disabled,
   or leave it out until it is not.
5. If tests need a trained model, use the session-scoped
   `trained_tiny_model_path` fixture — never train per-test.
