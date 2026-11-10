# mace-torch

Distribution `mace-torch`, import `mace_torch`.

Public surface: Build models, derivative engines, staged training, ASE calculators, canonical serialization and deployment bundles.

Depends on mace_core; does not import mace_jax. Inference does not import training. There is no `mace` legacy package or `mace_launcher` on this branch.
The PyTorch package owns native command routing; no `--engine` switch is used.
Never add an import of `tutorials/` to a supported package.

Run `python -m pytest packages/mace-torch/tests -m 'not slow and not gpu and not network'`.
Name new tests `test_mace_torch_<topic>.py`, with a unique basename across packages.
Run `ruff check packages/mace-torch`, `ruff format --check packages/mace-torch` and
`ty check packages/mace-torch`. See [the root guide](../../CLAUDE.md) for capability
requirements, frozen numerical evidence and the complete installation recipe.
