# mace-launcher

The [v0.3 → V1 migration guide](https://github.com/ACEsuit/mace/blob/mace-reforge/docs/reforge/migration.md) covers commands,
Python callers, supported checkpoint conversion and explicit pending work.
The guide and portable migration skill are prepared in the V1 tree; publication
is a separate release step.

Owns `mace` and every `mace_*` console script and dispatches each one to the legacy or the
v1 engine, chosen by `--engine {legacy,v1}` or `MACE_ENGINE`. The default is
`legacy`, so a freshly installed MACE behaves exactly as it did before.

It is the only place the two stacks meet, and it disappears at the end of the
migration: once the legacy package is gone the legacy branch goes with it and
the scripts point straight at `mace_torch.cli`.
