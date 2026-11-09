# V1 configuration

The complete file describes a training run. `model`, `precision`, and physical
solver settings also describe inference and are stored in the model checkpoint.
`data`, `loss`, `training`, and fine-tuning settings describe how that model is
trained. Evaluation normally loads the checkpoint; it does not need another
training file.

Files can use YAML, JSON or TOML. Unknown fields and duplicate YAML/JSON keys
are errors. A variant has one string `kind`, alongside its parameters. Nested
`kind: {kind: ...}` sections are rejected. V1 files use the current schema;
there are no aliases or migrations for earlier V1 formats.

Precision overrides are a map keyed by operation name:

```yaml
precision:
  model: float32
  accumulate: float64
  ops:
    linear:
      compute_dtype: float32
      accumulate_dtype: float64
      param_dtype: float32
```

An omitted operation uses the model dtype. Duplicate operation keys are errors,
and the resolved configuration is immutable. JAX currently supports uniform
operation precision and rejects a nonempty `precision.ops` map.

## Train from scratch

```yaml
runtime:
  name: water
  work_dir: runs/water
  device: cpu

data:
  heads:
    pbe:
      train_file: water.xyz
      e0s: {kind: average}
  valid_fraction: 0.1

model:
  num_channels: 64
  r_max: 5.0  # Angstrom

loss:
  kind: huber
  delta: 0.02
  weights: {energy: 1, forces: 100}

training:
  optimizer: {kind: adamw}
  lr: 0.001
  batch_size: 10
  max_num_epochs: 200
  scheduler:
    kind: plateau
    patience: 20
```

`e0s: {kind: average}` fits atomic reference energies from the training data.
Use `isolated_atoms` when the dataset provides isolated-atom reference structures.
Loss weights name requested observables exactly; unknown names are errors.
A missing weight is one, and zero explicitly disables that contribution.

## Fine-tune a V1 checkpoint

```yaml
runtime:
  name: water-finetune
  work_dir: runs/water-finetune
finetune:
  foundation_model: foundation.json

data:
  heads:
    pbe:
      train_file: water.xyz
      e0s: {kind: foundation}

training:
  lr: 0.0001
  max_num_epochs: 100
  ema: {enabled: true, decay: 0.99}
```

Omitted architecture fields inherit from the foundation. For a foundation with
multiple heads, specify `data.heads.pbe.readout_from` and the matching
`data.heads.pbe.e0s.head`.

## Stages

Each stage has a unique name and start epoch. They are ordered by `start_epoch`;
the earliest stage starts at zero. A stage overrides only the settings it names.

```yaml
training:
  stages:
    - name: fit
      start_epoch: 0
      lr: 0.001
    - name: refine
      start_epoch: 100
      lr: 0.0001
      loss_weights: {energy: 100, forces: 10}
```

`training.stage_two` remains a shorthand for two stages, with its overrides in
`loss.stage_two_weights`. Use either the shorthand or the explicit stage list.

## Check before running

```sh
mace --engine v1 config validate --config run.yaml
mace --engine v1 config validate --config run.yaml --show minimal
mace --engine v1 config validate --config run.yaml --show resolved
mace --engine v1 train --config run.yaml
```

Validation reads the configuration and checks the schema and cross-section
constraints. It does not open datasets, download a foundation, or build a model.
Those checks happen when training starts. `--show minimal` prints the supplied
settings and any applied preset defaults. `--show resolved` includes all schema
defaults. Both outputs use the same flat format and can be loaded again.

Explicit CLI flags override the file; omitted flags leave its values intact.
Changing a variant removes the old variant's parameters while preserving shared
settings such as loss weights. For example, `--loss weighted` removes Huber's
`delta`; `--huber_delta 0.03` updates a Huber or universal loss already selected
in the file. A delta with another loss is an error. Use `train --help` for flag
explanations and their configuration paths.

## Import a legacy training file

Legacy files are converted explicitly before training:

```sh
mace --engine v1 migrate --output run.yaml -- legacy.yaml
mace --engine v1 train --config run.yaml
```

The training command reads the structured V1 schema. Existing explicit CLI flags
remain available; their translated values use the same override operation.
