# Debt book

Every compromise the coexistence window creates, and the ticket whose merge removes it.

The rewrite runs as a coexistence: the new `packages/` stack is built beside a byte-frozen legacy
`mace/`, and legacy is retired capability by capability through deletion-only pull requests. That
window buys parity, and it costs compromises: a tolerance instead of bit-exactness, a capability
whose `--engine` default is still `legacy`, TorchScript decorators live on the path a v1 user hits,
a deployment gap. The known failure mode is not that any of those is wrong. It is that with no
business pressure the retirement never finishes and the dual state becomes permanent.

So each compromise is a row here, and each row has a test that fails until it is burned. There are
**no dates**. A date would be a promise to nobody: `main` stays on v0.3 until v1 ships, and no end
of life for v0.3 is announced. The trigger is the ticket, and the mechanics are in the next
section.

## How a row goes red

Each row names a fitness test in `tests/architecture/test_debt_book.py`. That test asserts the state
of the tree **after** the debt is burned, and it carries `xfail(strict=True)` for as long as its row
is in this file:

| the tree | the row | pytest says | reading |
|---|---|---|---|
| debt still there | open | `xfail` | expected, and green |
| debt burned | open | `XPASS(strict)` → **red** | the burn-step ticket closed and nobody removed the row |
| debt burned | removed | `passed` | done, and the assertion stays as a permanent guard |
| debt still there | removed | `failed` | the row was removed without burning the debt |

The third column is the whole mechanism, and it needs no calendar and no GitHub query: `strict=True`
turns an unexpected pass into a failure, so "the burn-step ticket is closed while the row is still
open" is exactly an `XPASS`. The burn-step pull request therefore deletes the row *and* the
`@open_debt(...)` decorator in the same change; deleting either alone is one of the two red rows
above.

`burn-check` is the prose statement of what the fitness test asserts. It is not a second copy of the
assertion: it says what would have to be true of the tree for the debt to be gone, in the terms a
reviewer of the burn-step pull request can check.

Run the meta-checks:

```bash
python3 tests/architecture/check_debt_book.py
python -m pytest tests/architecture/test_debt_book.py
```

## The ledger

| debt_id | description | burn-step (ticket) | burn-check | fitness-test |
|---|---|---|---|---|
| `DEBT-TOL-PARITY-FP64-CPU` | The parity harness compares v1 against the live legacy model at `fp64_cpu_reference` (atol 1e-6, rtol 0), not bit-for-bit. Two independent implementations of the same physics do not agree to the last bit at fp64, so a tolerance is the only workable contract while both are live. It is still a compromise: any real disagreement smaller than 1e-6 eV is invisible for the length of the window. | RET-6 (#1602) | `tests/parity/` exists and no file under it imports the legacy `mace` package. Once the legacy half is gone there is nothing left to compare in-process, and the test degrades to a frozen golden asserted at the same row. | `test_debt_parity_fp64_cpu_tolerance_is_gone` |
| `DEBT-TOL-PARITY-FP64-ACCEL` | The accelerated-backend comparison runs at `fp64_accelerated_backend` (atol 1e-5, rtol 0), an order looser than the CPU row because it crosses a kernel implementation *and* a device *and*, during the window, the legacy-vs-v1 weight layouts that the five `convert_*` CLIs translate between. | RET-3 (#1599) | None of the five `mace/cli/convert_*.py` CLIs exists. After that the accelerated comparison is v1-internal, backend against backend in one layout, and the cross-layout term the 1e-5 carries is gone. | `test_debt_parity_fp64_accelerated_tolerance_is_gone` |
| `DEBT-ENGINE-DEFAULT-energy` | Energy, forces and stress default to `--engine legacy`. | RET-1 (#1597) | `capabilities.toml` records the `energy` axis at `v1-default` or `retired`. | `test_debt_engine_default_is_v1[energy]` |
| `DEBT-ENGINE-DEFAULT-data` | The data layer defaults to `--engine legacy`. | RET-2 (#1598) | `capabilities.toml` records the `data` axis at `v1-default` or `retired`. | `test_debt_engine_default_is_v1[data]` |
| `DEBT-ENGINE-DEFAULT-backends` | The accelerated backends and their weight conversions default to `--engine legacy`. | RET-3 (#1599) | `capabilities.toml` records the `backends` axis at `v1-default` or `retired`. | `test_debt_engine_default_is_v1[backends]` |
| `DEBT-ENGINE-DEFAULT-dipole` | Dipoles, dielectric response, polar and magnetic models default to `--engine legacy`. | RET-4 (#1600) | `capabilities.toml` records the `dipole` axis at `v1-default` or `retired`. | `test_debt_engine_default_is_v1[dipole]` |
| `DEBT-ENGINE-DEFAULT-lammps` | LAMMPS deployment defaults to `--engine legacy`. | RET-5 (#1601) | `capabilities.toml` records the `lammps` axis at `v1-default` or `retired`. | `test_debt_engine_default_is_v1[lammps]` |
| `DEBT-ENGINE-DEFAULT-training` | The training entry point defaults to `--engine legacy`. | RET-6 (#1602) | `capabilities.toml` records the `training` axis at `v1-default` or `retired`. | `test_debt_engine_default_is_v1[training]` |
| `DEBT-JIT-MODULES` | 52 `@compile_mode("script")` decorators live across seven modules of `mace/modules/`. They are on the path a v1 user reaches, because the frozen tree stays the default engine and the oracle for the whole window, and they constrain how that code may be written: TorchScript rejects PEP 604 unions, needs `torch.jit.annotate` for empty containers, and forces the `torch.jit.is_scripting()` branches in `blocks.py` and `utils.py`. | RET-4 (#1600) | No `@compile_mode` decorator remains anywhere under `mace/modules/`. RET-1 removes most of them with the energy models; the last ones leave with `extensions.py` and `field_blocks.py`. | `test_debt_no_compile_mode_in_legacy_modules` |
| `DEBT-JIT-CALCULATORS` | The two LAMMPS calculators are themselves `@compile_mode("script")` modules, and carry the `lammps_class` / `lammps_natoms` branches that let the same blocks run under LAMMPS's real-versus-ghost atom partitioning. | RET-5 (#1601) | Neither `mace/calculators/lammps_mace.py` nor `mace/calculators/lammps_mliap_mace.py` exists. | `test_debt_no_lammps_calculators` |
| `DEBT-JIT-EXPORT` | `mace_create_lammps_model` scripts the model with `e3nn.util.jit.compile` (`mace/cli/create_lammps_model.py:109`). TorchScript is banned under `packages/`, so this call site cannot be pointed at a v1 model: it is legacy-only by construction. | RET-5 (#1601) | `mace/cli/create_lammps_model.py` does not exist, so no call site in the tree scripts a model for LAMMPS. | `test_debt_no_torchscript_lammps_export` |
| `DEBT-COMPILED-ARTIFACT-BEST-EFFORT` | Both `_compiled.model` emissions in `mace/cli/run_train.py` (`:1190-1211` and `:1212-1229`) sit inside `except Exception`. A scripting failure therefore produces **no artifact and a zero exit status**: the run reports success and the side artifact is simply absent. The handlers do log a warning naming the file, so this row is about the missing artifact and the exit status, not about silence. The coexistence window inherits that contract, and a v1 model is exactly the input that cannot be scripted. | DEP-2 (#1594) | `mace/cli/run_train.py` no longer calls `jit.compile`. v1 emits no side artifact at all: `torch.export` produces the deployment bundle deliberately, in a step that fails when it fails. | `test_debt_no_best_effort_compiled_artifact` |
| `DEBT-V1-NO-DEPLOYMENT` | v1 blocks are born without `@compile_mode`, so the legacy `e3nn.util.jit` export and checkpoint path cannot script them. The `--engine v1` opt-in therefore **excludes LAMMPS export and the compiled checkpoint** for as long as this row is open, and deployment stays legacy-only. This is the one row that is a gap in v1 rather than a leftover of legacy. | DEP-2 (#1594) | `mace_torch` ships a deployment surface: `packages/mace-torch/src/mace_torch/deploy/` exists with an export entry point in it. | `test_debt_v1_has_no_deployment_path` |
