# mace-jax

Distribution `mace-jax`, import `mace_jax`.

Public surface: Canonical checkpoint inference and the mace-jax evaluation command.

Inference only; imports mace_core, never torch, e3nn or mace_torch. Requires Python 3.12 or newer. There is no `mace` legacy package or `mace_launcher` on this branch.
The PyTorch package owns native command routing; no `--engine` switch is used.
Never add an import of `tutorials/` to a supported package.

Run `python -m pytest packages/mace-jax/tests -m 'not slow and not gpu and not network'`.
Name new tests `test_mace_jax_<topic>.py`, with a unique basename across packages.
Run `ruff check packages/mace-jax`, `ruff format --check packages/mace-jax` and
`ty check packages/mace-jax`. See [the root guide](../../CLAUDE.md) for capability
requirements, frozen numerical evidence and the complete installation recipe.
