# mace-launcher

Owns every `mace_*` console script and dispatches each one to the legacy or the
v1 engine, chosen by `--engine {legacy,v1}` or `MACE_ENGINE`. The default is
`legacy`, so a freshly installed MACE behaves exactly as it did before.

It is the only place the two stacks meet, and it disappears at the end of the
migration: once the legacy package is gone the legacy branch goes with it and
the scripts point straight at `mace_torch.cli`.
