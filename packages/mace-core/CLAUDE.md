# mace-core

Distribution `mace-core`, import `mace_core`.

Public surface: Configuration, observable catalogue, graph schema, neighbour/data backends, kernel protocols and canonical weights.

Imports no torch, jax, e3nn or sibling package. There is no `mace` legacy package or `mace_launcher` on this branch.
The PyTorch package owns native command routing; no `--engine` switch is used.
Never add an import of `tutorials/` to a supported package.

Run `python -m pytest packages/mace-core/tests -m 'not slow and not gpu and not network'`.
Name new tests `test_mace_core_<topic>.py`, with a unique basename across packages.
Run `ruff check packages/mace-core`, `ruff format --check packages/mace-core` and
`ty check packages/mace-core`. See [the root guide](../../CLAUDE.md) for capability
requirements, frozen numerical evidence and the complete installation recipe.
