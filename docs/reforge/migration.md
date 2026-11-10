# Migrating MACE v0.3 workflows to V1

This guide describes the V1 checkout as of 2026-11-09. It is a migration guide
for implemented contracts, not an announcement that V1 or its native foundation
artifacts have been published. The legacy implementation has been retired from this tree.
Installed v0.3 binaries do not announce V1 replacements; this guide is the
migration signpost. Use the native `mace` command; no engine selector is needed. No TorchScript path is supported in V1.

## Start with the checkpoint

V1 stores canonical weights in safetensors with a JSON sidecar. It cannot load a
v0.3 pickle as a runtime model. Keep the original checkpoint and its environment
until the migrated workflow has passed its own checks.

The supported one-shot conversion scope is the published foundation roster:
**24 retained models plus the retired ANI-CC entry**. Arbitrary user-trained
checkpoints have no supported automated migration. Keep a pinned v0.3 environment
or retrain in V1; the generic importer's ability to read a file does not establish
support for its architecture. ANI-CC users should select an appropriate OFF23
model and validate it on their systems, rather than expect identical predictions.

The [foundation map](../../.claude/skills/mace-migrate/references/foundations.md)
lists every model, alias family, canonical name, precision and publication status.
Native artifact URLs and DOIs remain pending. Polar medium/large float32 parity
is also open; their float32 default is unchanged. Float64 evidence does not close
those rows, and migration must not silently change precision.

For a supported trusted source, use the offline FM-1 importer described in the
[conversion recipe](../../.claude/skills/mace-migrate/references/commands.md#model-conversion).
It extracts in a pinned legacy environment and imports through a neutral pair.
Supply the matching reference JSON and fixture directory to request verification.
`mace model convert` has a different purpose: native full-to-reduced basis
conversion. The reduced basis preserves functions and gradients, but redundant
original coefficients cannot be recovered uniquely. Retain the neutral original.
Saving and reloading the resulting V1 checkpoint must preserve its weights exactly.

## Translate commands and configuration

The [command map](../../.claude/skills/mace-migrate/references/commands.md) covers
all thirteen console entry points and every tracked legacy CLI module, including
unregistered converters. The [configuration map](../../.claude/skills/mace-migrate/references/changes.md)
covers every DROP/MERGE decision and every training destination, including
renamed KEEP fields. Those tables are also the portable migration skill's
references, so corrections have one home.

1. Run the documented `mace model migrate` recipe on the old training
   command or YAML file, writing a new output file.
2. Inspect the result and run `mace config validate --config ...`.
   This checks the schema, not the contents of a dataset or checkpoint.
3. Run the migrated workflow on a small representative dataset. Compare labels,
   head selection, predictions, gradients and resumed training with the existing
   references before increasing its scale.

V1 accepts YAML, JSON and TOML and explicit command-line flags. Dotted names in
the map are configuration paths, not accepted CLI overrides. Unknown settings
fail. Flat discriminators select implementations, for example
`optimizer: {kind: adam}`. Omitted fine-tuning architecture fields inherit from
the foundation: do not replace a partial fine-tuning file with all schema defaults.

The ten `swa` spellings still work as compatibility aliases. New commands should
use `stage_two` spellings or the structured schedule; the implementation has not
removed those aliases despite the older release-ticket wording. L-BFGS is a **retained** optimizer kind, including full-batch training; stage-two
losses and learning rates belong to the stage. ASE graph padding and all six
ML-IAP runtime environment controls are also retained; the Python/data map
lists them individually and distinguishes the ML-IAP callback from native export.

Separate log/model/checkpoint/result directories become one `runtime.work_dir`.
The adapter accepts a common directory and rejects conflicting ones.
`downloads_dir` is rejected: use `XDG_CACHE_HOME` for caches independently of
`work_dir`. `foundation_model_kwargs={}` is accepted for compatibility; nonempty
unused options fail and must become explicit device, precision, backend or head
settings. Dataset statistics replace the old mean/std/neighbor override flags.

`migrate` translates training and selection configurations. For other commands
it may only write a command/argument manifest: review the destination's `--help`
and translate its options before execution. In particular, `convert-device`
remains a loading check and writes no rewritten checkpoint. Backend converters
are retired because weights are canonical. Standalone `data select` is retained
alongside integrated fine-tuning selection. The active-learning MD driver is
retired; use ASE dynamics with the calculator's committee mean and variance.

## Data and Python callers

The [Python and data map](../../.claude/skills/mace-migrate/references/python-api.md)
provides verified import targets, examples and all twelve unchanged label keys.
The package split is explicit: `mace_core` owns schemas and physical data,
`mace_torch` owns PyTorch training/inference and `mace_jax` supplies its supported
inference route. Legacy imports are not aliased.

The `head` file key is retired; each source belongs to a configured head.
Stress/virials accept 3×3 or six-component Voigt labels, but not flat arrays of
nine values. Conflicting isolated-atom energies fail. Training keeps the
conventional property keys; custom files can be read using `KeySpecification`
or translated by preprocessing. Prepare one source per head and use separate
replay/pseudolabel sources: there is no extra `*_combined.xyz` artifact.

Checkpoint metadata currently accepts **schema 1.0 exactly**. Unknown versions
fail; no generic older-minor upgrade is promised. User and resolved config,
provenance, citations, head information and parent metadata are recorded. The
canonical E0 tensors remain in the checkpoint. Training checkpoints also carry
the optimizer and schedule state needed by their resume contract; deployment
bundles are separate artifacts, not restart checkpoints.

## Deployment

Use the [integration support table](integration_support.md) to select a target.
`mace export lammps` can produce a neutral ML-IAP compatibility bundle or the
native `torch.export`/AOTInductor target. The ML-IAP compatibility callback uses
Python. The `mace/native` loop and `libmace` execution path are Python-free; their
models are exported ahead of time. There is no TorchScript fallback.

The export contract records head, units, precision, capacity limits, backend and
solver identity. Canonical checkpoints do not depend on a cuEq/OEq weight layout.
A runtime solver substitution is allowed only under the existing parity contract;
a solver that changes results is model state. Preserve the domain/ghost exchange
and neighbor-list behavior verified for the target. The general [Symmetrix 0.3.0 source](https://github.com/wcwitt/symmetrix/tree/fc7e53276baa7728f08bee0ddd865a20f82ff998)
is now available. Its extractor still reads legacy pickles and classes; V1
needs its own exporter and conformance checks before that target is supported.
The standard extraction path explicitly checks two interaction layers and
`l_max=3`; separate nonlinear and polar paths have their own scope. The native
runtime consumes Symmetrix JSON, which must be generated for the same Symmetrix
version. This is a deployment format, not a replacement canonical checkpoint.

## Accepted numerical and workflow differences

Parity remains strict where the two implementations express the same function
and inputs. These agreed differences are not reasons to weaken golden tolerances:

- Canonical reduced coordinates and changed batching can change Adam's trajectory.
  Compare functions and gradients, frozen weights, training and resume; require
  identical trajectories only when weights, coordinates and batches agree.
- V1 honors structure/property weights and masks for missing labels, including
  cases where legacy ignored them. It uses population variance for normalization
  and includes atoms without neighbors in the mean neighbor count.
- The ASE calculator applies dimensional length conversion to dipoles. Nonperiodic
  stress is zero; physical cells are preserved. Edge export uses `+dE/dvectors`.
- Content-valid log filenames are unrestricted. More DDP ranks than structures
  can be valid under the documented sampling/batch rules.
- Inactive Polar controls are rejected. The final density replaces the duplicate
  `charges_history` snapshot. LES `use_atomwise=True` remains representable when
  importing old checkpoints but is rejected for new training where it would be
  inactive. Scalar tanh retains its valid parity scope; the vectorial variant
  is explicitly equivariant and does not reproduce component-wise legacy tanh.

Each feature's numerical evidence remains in its existing tests and GATE-3
records. The migration coverage checks do not certify untested models or targets.

## Use the migration skill

The portable [mace-migrate skill](../../.claude/skills/mace-migrate/SKILL.md)
contains its own relative references. Copy the complete `mace-migrate` directory
into your agent's skill directory. Do not copy only `SKILL.md`. It is prepared in
this checkout; it has not been published or installed globally by this work.
Once the containing revision is published, it can also be discovered by a skill
installer such as `npx skills add ACEsuit/mace --skill mace-migrate` using that
revision. An older remote branch need not contain it yet.

The skill guides translation and verification. It does not extend checkpoint
support, change numerical tolerances, choose a different precision silently, or
authorize publishing changes.

## What is still pending

| Item | Required before claiming completion |
| --- | --- |
| Foundation publication (FM-3) | Native artifact URLs, artifact DOIs and publication checks for retained models. |
| Polar medium/large float32 | Resolve parity with the current precision contract and tolerances; both rows stay open. |
| EnergyFunctionalSCF (ELEC-3) | Obtain an executable upstream oracle before implementing or claiming parity. |
| General Symmetrix adapter | Interface now available in 0.3.0; V1 export, supported-architecture checks and native parity remain to be implemented and validated. |
| Release and integration gates | Team review/sign-off, legacy retirement and release work; local evidence is not a merged or released ticket. |
| Scheduled branch coverage | Dispatcher/default-branch policy still requires an explicit integration decision. |
| v0.3 fixes after V1 | Maintainers must decide whether post-V1 fixes ship; no maintenance promise is made here. |

[ACEsuit/mace-scf](https://github.com/ACEsuit/mace-scf) is the reference for LC-MACE,
LSC-MACE, QEq and FixedPointSCF. Rechecked on 2026-11-09, its `main` remains at
`784a510339aacc4412b61b6926ef1a92bca387c0`; its README still lists
EnergyFunctionalSCF as being prepared. It is not an available implementation.

V0.3 remains on `main` until V1 release and then at its final pinned release.
There is no separate maintenance branch or announced end-of-life deadline.
During the transition use the launcher explicitly to select the desired engine.
The development distribution `mace-torch` is temporary; the intended final
package identity is `mace-torch` after retirement, not a package rename users
should assume has already happened on PyPI.
