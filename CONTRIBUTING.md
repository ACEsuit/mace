# Contributing to MACE

V1 is an unpublished implementation reference. Reviewed implementation PRs land
on `mace-reforge`; releases will be made from that work. See the
[branch decisions](docs/reforge/reference_branch.md) and
[governance](GOVERNANCE.md) for scope and approval responsibilities.

## Pull request checklist

- [ ] The relevant CI checks pass, including tests for changed public behaviour.
- [ ] Public API, command and configuration changes have documentation.
- [ ] A major feature includes a runnable example or tutorial executed in CI.
- [ ] Any example broken by this change is fixed in this PR.
- [ ] Numerical changes preserve the frozen references and their tolerances.
- [ ] Package tests have unique names: `test_mace_<package>_<topic>.py`.
- [ ] New dependencies and capability markers are declared and tested.

A technical committee reviewer provides final technical sign-off. Review should
explain a concrete defect or tradeoff and cite the relevant test or contract.
Generated suggestions receive the same review as other changes.

## Style and checks

Use descriptive names (`clebsch_gordan`, `spherical_harmonics`) and prefer pure
functions with explicit state. Document units, tensor shapes and sign
conventions at public boundaries. Keep model construction independent of
training, and keep framework objects out of `mace_core`.

The native workspace uses ruff for lint and formatting and ty for types,
through `prek`. `.pre-commit-config.yaml` owns the actual scopes:
`packages/`, `tests/parity/`, `tests/retirement/` and `tests/conftest.py` receive
ruff; ty checks `packages/`. Other tests and documentation are not reformatted.
Legacy and its black/isort/pylint/mypy configuration have been retired from V1.
Historical checkouts keep their original toolchains; never reformat an oracle.

```sh
python -m pip install --group dev -e 'packages/mace-core[matscipy]' -e packages/mace-torch -e packages/mace-backend-example
prek run --all-files
python -m pytest tests/golden tests/retirement tests/architecture
python -m pytest packages/mace-core/tests packages/mace-torch/tests -m 'not slow and not gpu and not network'
```

Install `packages/mace-jax` on Python 3.12 or newer to run its suite. The
`run-tests` composite action inputs are the local reproduction recipe for each
CI job. Missing optional capabilities skip locally; `MACE_REQUIRE_CAPS` makes a
missing capability fail in jobs that promise it. The authority on capability
names is `CAPABILITY_PROBES` in `tests/conftest.py`.

## Examples are part of the interface

A major feature must include a tutorial or example executed at merge time.
Fix a broken existing example in the same PR that changes its behaviour.
Documentation builds and runnable examples belong in the code repository;
text that merely resembles a command is not evidence that the command works.

## Numerical evidence

The original inputs, reference values and tolerance rows remain in
`tests/golden/`. Native converted anchors and coordinate maps in
`tests/retirement/` retain the all-parameter gradient gate after legacy removal.
`historical:` pins in the feature inventory refer to the pre-retirement commit;
they are explicitly not runnable coverage in the current checkout. Keep the
retirement ledger and current numerical gates together when changing them.
