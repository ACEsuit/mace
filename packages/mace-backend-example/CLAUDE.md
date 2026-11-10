# mace-backend-example

Distribution `mace-backend-example`, import `mace_backend_example`.

Public surface: An out-of-tree-style backend registered through entry points.

Core and framework packages never import this example by name. There is no `mace` legacy package or `mace_launcher` on this branch.
The PyTorch package owns native command routing; no `--engine` switch is used.
Never add an import of `tutorials/` to a supported package.

Run `python -m pytest packages/mace-backend-example/tests -m 'not slow and not gpu and not network'`.
Name new tests `test_mace_backend_example_<topic>.py`, with a unique basename across packages.
Run `ruff check packages/mace-backend-example`, `ruff format --check packages/mace-backend-example` and
`ty check packages/mace-backend-example`. See [the root guide](../../CLAUDE.md) for capability
requirements, frozen numerical evidence and the complete installation recipe.
