# V1 command line

Use `mace --engine v1` or set `MACE_ENGINE=v1`. The launcher retains the legacy
engine as its default. Help discovers commands without importing torch, JAX or
legacy MACE.

```sh
mace --engine v1 train --config run.yaml --device cpu
mace --engine v1 eval --model run.json --configs structures.xyz --output predicted.xyz --compute_stress
mace --engine v1 model select-head run.json --head pbe --output pbe.safetensors
mace --engine v1 model export-config run.json --output resolved.yaml
mace --engine v1 data prepare --train_file train.xyz --output prepared --compute_statistics
mace --engine v1 data select --config finetune.yaml --head replay --output selected.xyz
mace --engine v1 data pseudolabel --config finetune.yaml --head replay --output labels
mace --engine v1 plot training --path results/run.jsonl --output learning.png
mace --engine v1 migrate --output migrated.yaml -- mace_run_train --train_file train.xyz --num_channels 32
```

## Configuration and migration

V1 uses explicit flags and structured YAML, JSON or TOML configuration files.
Unknown options and unknown configuration keys fail. Dotted command-line
overrides remain unsupported. A configuration exported from a checkpoint
resolves to the same settings when read again.

The training parser recognizes the frozen CLI's flag spellings, including
stage-two aliases. Supplied flags are translated through `from_namespace` at
the CLI boundary. Omitted settings keep the V1 schema's defaults. Basic file
flags continue to target the single head already named in a structured file;
a multihead file requires each path under its head.

Migration follows the agreed feature inventory. Separate output directories
become one `runtime.work_dir`, property-label keys keep the data convention,
and checkpoints remain canonical safetensors with a JSON sidecar. A changed
option that cannot be represented raises an explanation. In particular, the
legacy training spelling `--model=MACE` forced a zero-shift scale-shift model;
it is refused rather than mapped to V1's different `plain` architecture.
Choose the V1 model kind in the configuration, or retain the legacy engine
when identical legacy training defaults are required.

`migrate` writes a training configuration for a training command or legacy
training YAML. Runs from scratch record resolved defaults; fine-tunes keep
unspecified architecture settings absent so they inherit the foundation's
architecture when loaded. For the old standalone selector it writes ordinary
head/subselection settings and prints a `data select` command. Other supported
scripts produce a command/argument YAML manifest and an executable command.
Retired device/backend weight conversions and the MD driver explain their
replacement. Convert pickle checkpoints separately with the existing legacy
importer before using these commands.

The source boundary permits legacy model-name strings in the two input
adapters. The architecture checks still forbid legacy imports, identifiers,
classes and attribute access there.

## Evaluation and artifacts

Evaluation preserves the thirteen legacy extxyz names and records their
owners in `OUTPUT_COLUMNS`. Magnetic and polar channels use their existing
model implementations. Requested unsupported derivatives fail. Additional
declared properties use their observable name. Descriptors support per-atom,
mean and per-element mean output. Energies are eV, forces eV/Angstrom and
stress eV/Angstrom cubed; aperiodic stress remains zero.

Head selection keeps the chosen readouts, embedding readout, scale, shift and
E0 table. Model metadata contains required citations, including inherited
citations from foundation models. Train and eval print them. `CITATION.cff`
provides the software and preferred paper references.

```sh
mace --engine v1 model convert full.json --to-basis reduced --output reduced.json \
  --reference reference.json --fixtures fixtures
```

Basis conversion verifies every basis projection. With `--reference` and
`--fixtures`, it also checks energy, forces and stress before writing either
artifact file. Metadata schema 1.2 records the source/target bases, projection
verification and golden-check result. Without reference files that result is
null; it does not claim a golden check. Older metadata remains readable.
Reduced-to-full export is underdetermined and belongs to DEP-3.

Prepared data uses v2 shards and manifests. Isolated-atom reference structures
stay in the training split so the training E0 resolver can read them. Measured
statistics are written only with `--compute_statistics`; `e0s.json` records the
resolved references. Selection and pseudolabelling call the same FT-2 functions
used during training. Training writes per-head JSON-lines metrics; plotting
runs on rank zero and is shared with the standalone command.

Distributed mini-batch training repeats samples as needed to give each process
the same number of samples. Each process must still have enough samples for its
configured batch size. Full-batch L-BFGS partitions the data without repetition;
processes with an empty share contribute zero to the collective loss and
gradient. Evaluation also avoids repetition. Having more processes than
structures is supported when these conditions hold. Legacy's rejection of this
case is explicitly retired for V1.

`mace plot training --path` accepts a file with any name, or a directory of
`*.txt` and `*.jsonl` files. It validates records before writing plots and
reports invalid content with its file and line number. Each nonblank line is a
JSON object with a finite numeric `epoch`, or `null` for initial diagnostics.
Optional `head` is a string or null; optional `mode` is a string and defaults
to `valid`. Requested metrics are numbers or null; missing metrics are skipped.
Nonfinite metrics remain readable for diagnosing divergence. Other fields are
ignored. The CLI selects epochs strictly greater than `--min_epoch` (default
zero); `--min_epoch -1` includes epoch zero. Input basenames do not determine
run or seed metadata. This is an explicitly accepted difference from legacy's
`<run>_run-<seed>_train.txt` requirement.

LAMMPS export remains assigned to DEP-2. The agreed MD-driver removal remains
in force; ASE dynamics can use the calculator's committee mean and variance.

## Verification

The CLI suite exercises training artifacts, configuration precedence, plots,
shards, selection and head extraction. The parity suite enumerates all legacy
training flag spellings and all thirteen evaluation columns, converts both
committed anchors through full and reduced bases, and checks extxyz energy,
forces and stress against the frozen references. A failing reference check is
also tested to leave no converted artifact. Existing golden tolerances and
legacy source files are unchanged.
