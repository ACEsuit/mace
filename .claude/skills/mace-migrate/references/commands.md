# Command and legacy module map

Use the current V1 checkout or installation. The twelve historical scripts and
the additional `mace` entry point are now owned by `mace-launcher`, rather than
`setup.cfg`. The table also covers every tracked file under `mace/cli/`, including
the package initializer and the four unregistered executable modules.

| Entry point | Legacy module | V1 command | Migration |
| --- | --- | --- | --- |
| `mace` | — | `mace --engine v1` | Hierarchical launcher; legacy remains the default engine. |
| `mace_run_train` | `run_train.py` | `mace --engine v1 train` | Translate flags/config, then validate the file. |
| `mace_eval_configs` | `eval_configs.py` | `mace --engine v1 eval` | Supply a native checkpoint with `--model`, structures with `--configs` and `--output`. |
| `mace_prepare_data` | `preprocess_data.py` | `mace --engine v1 data prepare` | One source/head per preparation; `--heads` is retired. |
| `mace_create_lammps_model` | `create_lammps_model.py` | `mace --engine v1 export lammps` | Positional native checkpoint; neutral ML-IAP bundle, or `--native NODE_SLOTS EDGE_SLOTS` for Python-free execution. |
| `mace_select_head` | `select_head.py` | `mace --engine v1 model select-head` | Native checkpoint, `--head` and `--output`; preserves selected-head predictions. |
| `mace_plot_train` | `plot_train.py` | `mace --engine v1 plot training` | `--path` accepts content-valid logs with arbitrary names, including V1 JSONL. |
| `mace_polar_density_cube` | `polar_density_cube.py` | `mace --engine v1 eval density-cube` | Native Polar model and declared density inputs. |
| `mace_finetuning_select` | `fine_tuning_select.py` | `mace --engine v1 data select` | Selection also belongs to training head config. Standalone: `--config`, `--head`, `--output`. |
| `mace_convert_device` | `convert_device.py` | `mace --engine v1 model convert-device` | `--model` and `--device` verify loading; writes no device-specific checkpoint. Precision is chosen at loading. |
| `mace_e3nn_cueq` | `convert_e3nn_cueq.py` | removed | Canonical weights; choose `model.backend` when building or a supported deployment backend override. |
| `mace_cueq_to_e3nn` | `convert_cueq_e3nn.py` | removed | Same canonical checkpoint for reference and accelerated backends. |
| `mace_active_learning_md` | `active_learning_md.py` | removed | Use ASE dynamics with the V1 calculator's committee mean and variance; the MD driver is outside V1 scope. |
| — | `convert_e3nn_oeq.py` | removed | Backend layouts require no separate checkpoint conversion. |
| — | `convert_oeq_e3nn.py` | removed | Backend layouts require no separate checkpoint conversion. |
| — | `convert_e3nn_hybrid.py` | removed | Canonical checkpoints record their CG basis. Use explicit native basis conversion when needed. |
| — | `visualise_train.py` | removed | Use `mace --engine v1 plot training` for supported learning curves; the introspection plot is retired. |
| — | `__init__.py` | package marker | No executable command; no V1 import alias. |

## Training command recipe

These commands translate and validate configuration; they do not start training
or require `train.xyz` to exist. A real training run needs the named dataset.
The resulting configuration has cutoff 4 Å, 8 channels, float64 precision, an
L-BFGS optimizer and stage-two energy weight 3 starting at epoch 4. The Gaussian
radial basis and legacy `Soft` spelling become `gaussian` and `soft` config kinds.

```sh
mace --engine v1 model migrate --output migrated.yaml -- mace_run_train --train_file train.xyz --r_max 4 --num_channels 8 --distance_transform Soft --radial_type gaussian --default_dtype float64 --lbfgs --stage_two --start_stage_two 4 --stage_two_energy_weight 3 --max_num_epochs 6
mace --engine v1 config validate --config migrated.yaml
```

For an old training YAML, pass its path after `--`. Fine-tunes preserve absent
architecture overrides so the foundation supplies them. For commands other
than training/selection, inspect the generated command manifest and translate
arguments using the destination's `--help`; forwarding an old argument unchanged
is not a guarantee that the new parser accepts it.

## Model conversion

The implemented FM-1 offline importer is a Python module. It uses the pinned
`mace-torch==0.3.16` environment to read the legacy pickle and writes a neutral
pair before importing canonical V1 weights. Use it only for a trusted source
artifact. Optional dependencies must also exist in the extraction environment.

```sh
python -m mace_torch.cli.convert_legacy published.model converted --pinned-environment legacy-converter --reference reference.json --fixtures fixtures
```

The reference JSON and fixture directory must belong to that source model; both
options must be supplied together. Without them, conversion makes no numerical
verification claim. Arbitrary user checkpoints have no supported automated
migration; retrain in V1 or retain the pinned v0.3 installation.

`mace --engine v1 model convert` performs **native full-to-reduced basis
conversion**, not pickle extraction. The reduced representation preserves the
function but cannot recover arbitrary redundant original coefficients. Keep
the original neutral pair. External full-basis weights export is a derived
deployment artifact, not an alternative canonical checkpoint format.
