---
name: mace-migrate
description: Migrate MACE v0.3 commands, configuration files and Python callers to the implemented V1 API. Use for a migration request; preserves agreed V1 contracts and identifies unsupported checkpoints and pending features.
---

# MACE v0.3 to V1

Use the installed V1 command help and these references to translate the user's
workflow. This skill targets the locally implemented V1 contracts, dated
2026-11-09. Check the target installation before applying the map to a later
release. Preserve the original input files and checkpoints.

## Decide what can migrate

V0.3 pickle checkpoints are not runtime-loadable in V1. The supported one-shot
conversion scope is the published foundation roster in
[foundations.md](references/foundations.md): 24 retained models and ANI-CC
retired in favor of OFF23. Arbitrary user checkpoints have **no automated path**
under this support contract: retain the pinned v0.3 environment or retrain in
V1. A converter accepting a file is not evidence that its architecture is
supported or numerically validated. Preserve the neutral original when a
redundant contraction basis is reduced; source weights cannot be reconstructed
uniquely from the reduced weights.

Published native URLs/DOIs remain pending. Polar medium/large float32 parity
remains open; keep the current precision contract and report the limitation.
EnergyFunctionalSCF awaits an executable upstream reference. Symmetrix 0.3.0 now publishes the general interface; its V1 exporter and
conformance evidence remain pending. Do not describe these as
available, substitute another model, change precision silently, or refresh
goldens to remove a mismatch.

## Commands and configuration

Read [commands.md](references/commands.md) for the command/module map and the
tested migration recipe. Use the native `mace` command; the legacy launcher has been retired. Read [changes.md](references/changes.md) for every DROP/MERGE
inventory row and every legacy training destination, including renamed KEEP
fields. Rows are keyed by argparse destination, so aliases share a row.

Use the existing `mace model migrate --output migrated.yaml -- ...` helper
for training commands or legacy YAML. Inspect its result, then run
`mace config validate --config migrated.yaml`. Validation checks
schema and cross-section constraints; it does not open data or validate a model.
For non-training commands the helper can retain old argument tokens in a
manifest. Translate those arguments against current subcommand help before
running them. The standalone selector is supported by `data select`.

V1 accepts explicit flags, not dotted CLI overrides. Dotted names in the tables
are YAML/JSON/TOML field paths. Unknown options fail. The ten `swa` spellings
remain compatibility inputs to the current adapter; emit `stage_two` names or
the structured schedule. Keep omitted fine-tuning architecture fields absent
so they inherit from the foundation. Do not serialize all schema defaults into
a fine-tuning file. Use flat discriminators such as `optimizer: {kind: adam}`.

Stop the dependent migration when the helper rejects an option; explain its
replacement from the map. Do not silently discard an explicit unsupported
setting. In particular `--model=MACE` has no exact training-default translation;
the native `plain` architecture is a separate explicit choice.

## Python, data and deployment

Read [python-api.md](references/python-api.md) before changing imports or data
access. Imports move to `mace_core`/`mace_torch`; output `energy` becomes
`total_energy`, and `node_energy` becomes `node_energies`. Backend operators use
canonical weights. Device loading requires no backend-weight conversion.

Preserve the twelve label keys listed in the API reference. `head` is set per
file in config. Flat nine-value stress/virial labels must become 3×3 matrices;
conflicting isolated-atom energies must be resolved explicitly. Training accepts
conventional property keys; custom keys use `KeySpecification` at the data API
boundary or the preprocessing key flags. Do not promise arbitrary per-run label-key overrides.

Native deployment uses `torch.export`/AOTInductor. Never introduce TorchScript.
The ML-IAP compatibility callback uses Python; `mace/native` provides the
Python-free LAMMPS loop. Request deployment through `mace export`, separately
from training checkpoints. Preserve solver identity, units, head selection and
precision; a non-bit-parity solver is model state.

## Verify and report

Check the translated config and execute the requested workflow at its existing
test scale. For a supported model conversion compare outputs and gradients
against its references with their existing tolerances. Reuse the project's
parity tests when available. Report the files changed, validated commands and
unresolved inputs or publication dependencies. Migration does not authorize
publishing artifacts, opening PRs, pushing branches or contacting maintainers.

The v0.3 line remains on `main` through V1 release, then at its final release.
There is no maintenance branch or announced end-of-life date. Whether fixes
ship after V1 is still a maintainer decision; this skill makes no promise.
