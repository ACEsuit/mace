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

## Adding integration X

1. Create `tests/integrations/<x>/` with contract tests (and a `_harness.py`
   if the data contract deserves one).
2. Register a `bin_<x>` marker in `pyproject.toml` and add its probe to
   `CAPABILITY_PROBES` in `tests/conftest.py` (real import, not `find_spec`:
   a broken install must read as unavailable).
3. Mark real-tier tests `@pytest.mark.bin_<x>`.
4. Add a paths-filtered contract job to `.github/workflows/ci-integrations.yaml`
   and a real-tier job to `nightly.yaml` (start it with `continue-on-error:
   true`; promote to blocking once it has been green for a while).
5. If tests need a trained model, use the session-scoped
   `trained_tiny_model_path` fixture — never train per-test.
