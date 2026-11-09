# Configuration and retired surfaces

This table covers every DROP/MERGE inventory decision and every training argparse
destination, including renamed KEEP fields. The disposition is the inventory
decision; the migration column describes the implemented V1 boundary. CLI
spellings that share a destination share a row. Field paths are config paths,
not dotted CLI overrides. See [commands.md](commands.md) and
[python-api.md](python-api.md) for executable commands and API examples.

The inventory records architecture decisions, while compatibility adapters may
retain a spelling (for example ASE results or the standalone selector). Such
exceptions are explicit below. Defaults rejected by the adapter are never
silently discarded. A migration test checks coverage; numerical parity remains
the responsibility of each feature's own evidence.

Contents: [training destinations](#training-destinations) ·
[other changed surfaces](#other-changed-surfaces).

## Training destinations

| Inventory ID | Decision | Legacy surface | Migration |
| --- | --- | --- | --- |
| `train.config` | MERGE | `--config` | Use the destination command's `--config` input. YAML, JSON and TOML are loaded before explicit flags; config is not a field inside the document. |
| `train.name` | KEEP | `--name` | Use `runtime.name`. The legacy flag is translated at the input boundary. |
| `train.seed` | KEEP | `--seed` | Use `runtime.seed`. The legacy flag is translated at the input boundary. |
| `train.work_dir` | KEEP | `--work_dir` | Use `runtime.work_dir`. The legacy flag is translated at the input boundary. |
| `train.log_dir` | MERGE | `--log_dir` | Use `runtime.work_dir`. The adapter accepts a single common directory and rejects conflicting output directories. |
| `train.model_dir` | MERGE | `--model_dir` | Use `runtime.work_dir`. The adapter accepts a single common directory and rejects conflicting output directories. |
| `train.checkpoints_dir` | MERGE | `--checkpoints_dir` | Use `runtime.work_dir`. The adapter accepts a single common directory and rejects conflicting output directories. |
| `train.results_dir` | MERGE | `--results_dir` | Use `runtime.work_dir`. The adapter accepts a single common directory and rejects conflicting output directories. |
| `train.downloads_dir` | MERGE | `--downloads_dir` | Explicit values are rejected. Set `XDG_CACHE_HOME` for downloads and `runtime.work_dir` for training outputs; these directories are independent. |
| `train.device` | KEEP | `--device` | Use `runtime.device`. The legacy flag is translated at the input boundary. |
| `train.default_dtype` | MERGE | `--default_dtype` | Use `precision.model`; the compatibility flag still translates float32/float64. Per-operation overrides remain explicit. |
| `train.distributed` | KEEP | `--distributed` | Use `runtime.distributed`. The legacy flag is translated at the input boundary. |
| `train.launcher` | KEEP | `--launcher` | Use `runtime.launcher`. The legacy flag is translated at the input boundary. |
| `train.log_level` | KEEP | `--log_level` | Use `runtime.log_level`. The legacy flag is translated at the input boundary. |
| `train.plot` | KEEP | `--plot` | Use `runtime.plot`. The legacy flag is translated at the input boundary. |
| `train.plot_frequency` | KEEP | `--plot_frequency` | Use `runtime.plot_frequency`. The legacy flag is translated at the input boundary. |
| `train.plot_interaction_e` | DROP | `--plot_interaction_e` | Removed: a legacy plotting mode with no v1 consumer; plotting is the CLI's. Explicit nondefault values fail. |
| `train.error_table` | KEEP | `--error_table` | Use `runtime.error_table`. The legacy flag is translated at the input boundary. |
| `train.model` | MERGE | `--model` | Use `model.model` and `model.observables`; class spellings are compatibility inputs. Explicit `--model=MACE` is rejected because legacy forced a zero shift; choose a native architecture deliberately. |
| `train.r_max` | KEEP | `--r_max` | Use `model.r_max`. The legacy flag is translated at the input boundary. |
| `train.radial_type` | KEEP | `--radial_type` | Use `model.radial_basis.kind`: bessel, gaussian or chebyshev, with basis settings alongside kind. |
| `train.num_radial_basis` | KEEP | `--num_radial_basis` | Use `model.radial_basis.num_basis`. The legacy flag is translated at the input boundary. |
| `train.num_cutoff_basis` | KEEP | `--num_cutoff_basis` | Use `model.cutoff.polynomial_order`. The legacy flag is translated at the input boundary. |
| `train.pair_repulsion` | KEEP | `--pair_repulsion` | Use `model.pair_repulsion`. The legacy flag is translated at the input boundary. |
| `train.distance_transform` | KEEP | `--distance_transform` | Use `model.distance_transform.kind`: None / Agnesi / Soft become none / agnesi / soft. `none` is an explicit kind, not null. |
| `train.apply_cutoff` | KEEP | `--apply_cutoff` | Use `model.apply_cutoff`. The legacy flag is translated at the input boundary. |
| `train.use_last_readout_only` | MERGE | `--use_last_readout_only` | Use `model.readout.last_only`; this selects which layers contribute readouts. |
| `train.use_embedding_readout` | MERGE | `--use_embedding_readout` | Use `model.readout.from_embedding`; the embedding readout remains explicit. |
| `train.interaction` | KEEP | `--interaction` | Use `model.interaction`. The legacy flag is translated at the input boundary. |
| `train.interaction_first` | KEEP | `--interaction_first` | Use `model.interaction_first`. The legacy flag is translated at the input boundary. |
| `train.max_ell` | KEEP | `--max_ell` | Use `model.max_ell`. The legacy flag is translated at the input boundary. |
| `train.correlation` | KEEP | `--correlation` | Use `model.correlation`. The legacy flag is translated at the input boundary. |
| `train.use_reduced_cg` | MERGE | `--use_reduced_cg` | Use `model.clebsch_gordan_basis`: recorded model state rather than a flag, because on the frozen tree it silently follows what is installed. |
| `train.use_so3` | DROP | `--use_so3` | Removed: an SO(3) variant with no trained artifact and no consumer. Explicit nondefault values fail. |
| `train.use_agnostic_product` | KEEP | `--use_agnostic_product` | Use `model.use_agnostic_product`. The legacy flag is translated at the input boundary. |
| `train.num_interactions` | KEEP | `--num_interactions` | Use `model.num_interactions`. The legacy flag is translated at the input boundary. |
| `train.MLP_irreps` | KEEP | `--MLP_irreps` | Use `model.readout.mlp_irreps`. The legacy flag is translated at the input boundary. |
| `train.radial_MLP` | KEEP | `--radial_MLP` | Use `model.radial_mlp`. The legacy flag is translated at the input boundary. |
| `train.hidden_irreps` | KEEP | `--hidden_irreps` | Use `model.hidden_irreps`. The legacy flag is translated at the input boundary. |
| `train.edge_irreps` | KEEP | `--edge_irreps` | Use `model.edge_irreps`. The legacy flag is translated at the input boundary. |
| `train.use_edge_irreps_first` | KEEP | `--use_edge_irreps_first` | Use `model.use_edge_irreps_first`. The legacy flag is translated at the input boundary. |
| `train.num_channels` | KEEP | `--num_channels` | Use `model.num_channels`. The legacy flag is translated at the input boundary. |
| `train.max_L` | KEEP | `--max_L` | Use `model.max_L`. The legacy flag is translated at the input boundary. |
| `train.gate` | KEEP | `--gate` | Use `model.readout.gate`. The legacy flag is translated at the input boundary. |
| `train.kspace_cutoff_factor` | KEEP | `--kspace_cutoff_factor` | Use `electrostatics.kspace_cutoff_factor`. The legacy flag is translated at the input boundary. |
| `train.atomic_multipoles_max_l` | KEEP | `--atomic_multipoles_max_l` | Use `model.polar.multipole_max_l`. The legacy flag is translated at the input boundary. |
| `train.atomic_multipoles_smearing_width` | KEEP | `--atomic_multipoles_smearing_width` | Use `model.polar.multipole_width`. The legacy flag is translated at the input boundary. |
| `train.field_feature_max_l` | KEEP | `--field_feature_max_l` | Use `model.polar.feature_max_l`. The legacy flag is translated at the input boundary. |
| `train.field_feature_widths` | KEEP | `--field_feature_widths` | Use `model.polar.feature_widths`, a list of numbers rather than a Python-literal string. |
| `train.field_feature_norms` | KEEP | `--field_feature_norms` | Use `model.polar.feature_norms`, a list of numbers rather than a Python-literal string. |
| `train.num_recursion_steps` | KEEP | `--num_recursion_steps` | Use `model.polar.num_recursion_steps`. The legacy flag is translated at the input boundary. |
| `train.field_si` | KEEP | `--field_si` | Use `model.polar.feature_self_interaction`. The legacy flag is translated at the input boundary. |
| `train.include_electrostatic_self_interaction` | KEEP | `--include_electrostatic_self_interaction` | Use `model.polar.energy_self_interaction`. The legacy flag is translated at the input boundary. |
| `train.add_local_electron_energy` | KEEP | `--add_local_electron_energy` | Use `model.polar.add_local_electron_energy`. The legacy flag is translated at the input boundary. |
| `train.quadrupole_feature_corrections` | KEEP | `--quadrupole_feature_corrections` | Use `model.polar.quadrupole_feature_corrections`. The legacy flag is translated at the input boundary. |
| `train.return_electrostatic_potentials` | MERGE | `--return_electrostatic_potentials` | Removed: the frozen polar model returns None for it whatever it is set to, so there is no quantity behind it to carry. Explicit nondefault values fail. |
| `train.field_norm_factor` | KEEP | `--field_norm_factor` | Removed: the frozen polar model stores it as a buffer and never reads it: its energy is the same at 1 and at 5. Explicit nondefault values fail. |
| `train.fixedpoint_update_config` | KEEP | `--fixedpoint_update_config` | Validated compatibility input: the implemented Polar update/embedding classes are fixed; unknown types or settings fail. The legacy nonlinearity_cls setting was inactive. |
| `train.field_readout_config` | KEEP | `--field_readout_config` | Validated compatibility input: OneBodyMLPFieldReadout is the supported Polar readout; unknown types or settings fail. |
| `train.scaling` | KEEP | `--scaling` | Use `model.scaling`. The legacy flag is translated at the input boundary. |
| `train.avg_num_neighbors` | MERGE | `--avg_num_neighbors` | Changed legacy values are rejected. Use dataset statistics or `data.statistics_file`; element tables come from the dataset or `finetune.element_table`. V1 uses population variance and includes zero-neighbor atoms. |
| `train.compute_avg_num_neighbors` | MERGE | `--compute_avg_num_neighbors` | Changed legacy values are rejected. Use dataset statistics or `data.statistics_file`; element tables come from the dataset or `finetune.element_table`. V1 uses population variance and includes zero-neighbor atoms. |
| `train.compute_stress` | MERGE | `--compute_stress` | Use `model.observables`: declaring a property is what computes it. |
| `train.compute_forces` | MERGE | `--compute_forces` | Use `model.observables`: declaring a property is what computes it. |
| `train.compute_polarizability` | MERGE | `--compute_polarizability` | Use `model.observables`: declaring a property is what computes it. |
| `train.compute_atomic_dipole` | MERGE | `--compute_atomic_dipole` | Use `model.observables`: declaring a property is what computes it. |
| `train.compute_magforces` | MERGE | `--compute_magforces` | Use `model.observables`: declaring a property is what computes it. |
| `train.train_file` | KEEP | `--train_file` | Use `data.heads.default.train_file`. The legacy flag is translated at the input boundary. |
| `train.valid_file` | KEEP | `--valid_file` | Use `data.heads.default.valid_file`. The legacy flag is translated at the input boundary. |
| `train.test_file` | KEEP | `--test_file` | Use `data.heads.default.test_file`. The legacy flag is translated at the input boundary. |
| `train.test_dir` | KEEP | `--test_dir` | Use `data.test_dir`. The legacy flag is translated at the input boundary. |
| `train.valid_fraction` | KEEP | `--valid_fraction` | Use `data.valid_fraction`. The legacy flag is translated at the input boundary. |
| `train.multi_processed_test` | MERGE | `--multi_processed_test` | Changed legacy values are rejected. Supply a file list, manifest or `test_dir`; the dataset layer resolves multiple files. |
| `train.num_workers` | KEEP | `--num_workers` | Use `data.num_workers`. The legacy flag is translated at the input boundary. |
| `train.pin_memory` | KEEP | `--pin_memory` | Use `data.pin_memory`. The legacy flag is translated at the input boundary. |
| `train.atomic_numbers` | MERGE | `--atomic_numbers` | Changed legacy values are rejected. Use dataset statistics or `data.statistics_file`; element tables come from the dataset or `finetune.element_table`. V1 uses population variance and includes zero-neighbor atoms. |
| `train.mean` | MERGE | `--mean` | Changed legacy values are rejected. Use dataset statistics or `data.statistics_file`; element tables come from the dataset or `finetune.element_table`. V1 uses population variance and includes zero-neighbor atoms. |
| `train.std` | MERGE | `--std` | Changed legacy values are rejected. Use dataset statistics or `data.statistics_file`; element tables come from the dataset or `finetune.element_table`. V1 uses population variance and includes zero-neighbor atoms. |
| `train.statistics_file` | KEEP | `--statistics_file` | Use `data.statistics_file`. The legacy flag is translated at the input boundary. |
| `train.les_arguments` | KEEP | `--les_arguments` | Use `model.les`. The legacy flag is translated at the input boundary. |
| `train.E0s` | KEEP | `--E0s` | Use `data.heads.default.e0s`: one string that names five kinds of E0s becomes the kind it names. |
| `train.keep_isolated_atoms` | KEEP | `--keep_isolated_atoms` | Use `data.heads.default.keep_isolated_atoms`. The legacy flag is translated at the input boundary. |
| `train.config_type_weights` | KEEP | `--config_type_weights` | Use `data.heads.default.config_type_weights`. The legacy flag is translated at the input boundary. |
| `train.energy_key` | MERGE | `--energy_key` | The conventional label key is retained; nondefault CLI values are rejected. For custom files use `KeySpecification` at the data API boundary (see python-api.md). |
| `train.forces_key` | MERGE | `--forces_key` | The conventional label key is retained; nondefault CLI values are rejected. For custom files use `KeySpecification` at the data API boundary (see python-api.md). |
| `train.virials_key` | MERGE | `--virials_key` | The conventional label key is retained; nondefault CLI values are rejected. For custom files use `KeySpecification` at the data API boundary (see python-api.md). |
| `train.stress_key` | MERGE | `--stress_key` | The conventional label key is retained; nondefault CLI values are rejected. For custom files use `KeySpecification` at the data API boundary (see python-api.md). |
| `train.dipole_key` | MERGE | `--dipole_key` | The conventional label key is retained; nondefault CLI values are rejected. For custom files use `KeySpecification` at the data API boundary (see python-api.md). |
| `train.polarizability_key` | MERGE | `--polarizability_key` | The conventional label key is retained; nondefault CLI values are rejected. For custom files use `KeySpecification` at the data API boundary (see python-api.md). |
| `train.charges_key` | MERGE | `--charges_key` | The conventional label key is retained; nondefault CLI values are rejected. For custom files use `KeySpecification` at the data API boundary (see python-api.md). |
| `train.head_key` | DROP | `--head_key` | Removed as a file-label setting. Assign a file to `data.heads.<name>`; the historical file key never selected a head. |
| `train.elec_temp_key` | KEEP | `--elec_temp_key` | Use `data.graph_input_keys.elec_temp`. The legacy flag is translated at the input boundary. |
| `train.total_spin_key` | KEEP | `--total_spin_key` | Use `data.graph_input_keys.total_spin`. The legacy flag is translated at the input boundary. |
| `train.total_charge_key` | KEEP | `--total_charge_key` | Use `data.graph_input_keys.total_charge`. The legacy flag is translated at the input boundary. |
| `train.embedding_specs` | KEEP | `--embedding_specs` | Use `data.embedding_specs`. The legacy flag is translated at the input boundary. |
| `train.skip_evaluate_heads` | KEEP | `--skip_evaluate_heads` | Use `data.skip_evaluate_heads`. The legacy flag is translated at the input boundary. |
| `train.foundation_model` | KEEP | `--foundation_model` | Use `finetune.foundation_model`. The legacy flag is translated at the input boundary. |
| `train.foundation_model_kwargs` | KEEP | `--foundation_model_kwargs` | Removed: foundation loading uses a native safetensors path; only an empty dictionary is accepted for compatibility. Configure runtime.device, precision.model, model.backend and data.heads.<name>.readout_from explicitly.. Explicit nondefault values fail. |
| `train.foundation_model_readout` | KEEP | `--foundation_model_readout` | Use `finetune.transfer_readout`. The legacy flag is translated at the input boundary. |
| `train.multiheads_finetuning` | KEEP | `--multiheads_finetuning` | Use a foundation plus explicit replay and fine-tuning entries in `data.heads`. The legacy adapter requires `pt_train_file`; no implicit download is promised. |
| `train.heads` | KEEP | `--heads` | Use `data.heads.<name>`; the training compatibility adapter accepts a literal mapping. Per-head `test_dir` is rejected: use global `test_dir` or explicit per-head `test_file` lists. |
| `train.foundation_head` | KEEP | `--foundation_head` | Use `data.heads.<name>.readout_from`; each head declares the source readout. |
| `train.weight_pt_head` | KEEP | `--weight_pt_head` | Use `data.heads.pt_head.weight`; requires explicit multihead replay in the legacy adapter. |
| `train.num_samples_pt` | KEEP | `--num_samples_pt` | Use `data.heads.pt_head.subselect.num_samples`; requires explicit multihead replay. |
| `train.real_pt_data_ratio_threshold` | KEEP | `--real_pt_data_ratio_threshold` | Use `data.ratio_guard` with `reference: pt_head` and `threshold`. |
| `train.pt_train_file` | KEEP | `--pt_train_file` | Use `data.heads.pt_head.train_file`; replay sources remain separate by head, with no extra combined XYZ. |
| `train.pt_valid_file` | KEEP | `--pt_valid_file` | Use `data.heads.pt_head.valid_file`; replay validation remains separate by head. |
| `train.subselect_pt` | KEEP | `--subselect_pt` | Use `data.heads.pt_head.subselect.method`. |
| `train.filter_type_pt` | KEEP | `--filter_type_pt` | Use `data.heads.pt_head.subselect.filtering`. |
| `train.allow_random_padding_pt` | KEEP | `--disallow_random_padding_pt` | Use `data.heads.pt_head.subselect.allow_random_padding`. |
| `train.pseudolabel_replay` | KEEP | `--pseudolabel_replay` | Use `finetune.pseudolabels.enabled`. The legacy flag is translated at the input boundary. |
| `train.pseudolabel_replay_compute_stress` | KEEP | `--pseudolabel_replay_compute_stress` | Use `finetune.pseudolabels.stress_if_missing`. The legacy flag is translated at the input boundary. |
| `train.foundation_model_elements` | KEEP | `--foundation_model_elements` | Use `finetune.element_table`: one field says which element table a fine-tune is built over. |
| `train.force_mh_ft_lr` | DROP | `--force_mh_ft_lr` | Removed: a preset is a declared default the user overrides, so there is nothing to force. Explicit nondefault values fail. |
| `train.lora` | KEEP | `--lora` | Use `finetune.lora.enabled`. The legacy flag is translated at the input boundary. |
| `train.lora_rank` | KEEP | `--lora_rank` | Use `finetune.lora.rank`. The legacy flag is translated at the input boundary. |
| `train.lora_alpha` | KEEP | `--lora_alpha` | Use `finetune.lora.alpha`. The legacy flag is translated at the input boundary. |
| `train.freeze` | KEEP | `--freeze` | Use `finetune.freeze`. The legacy flag is translated at the input boundary. |
| `train.finetune_dipoles_polarizabilities` | KEEP | `--finetune_dipoles_polarizabilities` | Nondefault legacy values are rejected. Configure the native dipole/dielectric model, observables and head transfer explicitly; the legacy flag is not a native mode. |
| `train.loss` | MERGE | `--loss` | Use `loss.kind`, `loss.weights` and `loss.use_structure_weights`. The helper translates legacy presets, including their dipole factor and structure-weight policy; keep those settings together when preserving the old objective. |
| `train.energy_weight` | KEEP | `--energy_weight` | Use `loss.weights.energy`. Use the helper when translating a legacy loss preset so its scale and masking policy are preserved. |
| `train.forces_weight` | KEEP | `--forces_weight` | Use `loss.weights.forces`. Use the helper when translating a legacy loss preset so its scale and masking policy are preserved. |
| `train.virials_weight` | KEEP | `--virials_weight` | Use `loss.weights.virials`. Use the helper when translating a legacy loss preset so its scale and masking policy are preserved. |
| `train.stress_weight` | KEEP | `--stress_weight` | Use `loss.weights.stress`. Use the helper when translating a legacy loss preset so its scale and masking policy are preserved. |
| `train.dipole_weight` | KEEP | `--dipole_weight` | Use `loss.weights.dipole`. Use the helper when translating a legacy loss preset so its scale and masking policy are preserved. |
| `train.polarizability_weight` | KEEP | `--polarizability_weight` | Use `loss.weights.polarizability`. Use the helper when translating a legacy loss preset so its scale and masking policy are preserved. |
| `train.magforces_weight` | KEEP | `--magforces_weight` | Use `loss.weights.magforces`. Use the helper when translating a legacy loss preset so its scale and masking policy are preserved. |
| `train.swa_energy_weight` | MERGE | `--swa_energy_weight` `--stage_two_energy_weight` | Use `loss.stage_two_weights.energy`. Only named weights override the first stage; the old swa spelling remains a compatibility alias. |
| `train.swa_forces_weight` | MERGE | `--swa_forces_weight` `--stage_two_forces_weight` | Use `loss.stage_two_weights.forces`. Only named weights override the first stage; the old swa spelling remains a compatibility alias. |
| `train.swa_virials_weight` | MERGE | `--swa_virials_weight` `--stage_two_virials_weight` | Use `loss.stage_two_weights.virials`. Only named weights override the first stage; the old swa spelling remains a compatibility alias. |
| `train.swa_stress_weight` | MERGE | `--swa_stress_weight` `--stage_two_stress_weight` | Use `loss.stage_two_weights.stress`. Only named weights override the first stage; the old swa spelling remains a compatibility alias. |
| `train.swa_dipole_weight` | MERGE | `--swa_dipole_weight` `--stage_two_dipole_weight` | Use `loss.stage_two_weights.dipole`. Only named weights override the first stage; the old swa spelling remains a compatibility alias. |
| `train.swa_polarizability_weight` | MERGE | `--swa_polarizability_weight` `--stage_two_polarizability_weight` | Use `loss.stage_two_weights.polarizability`. Only named weights override the first stage; the old swa spelling remains a compatibility alias. |
| `train.swa_magforces_weight` | MERGE | `--swa_magforces_weight` `--stage_two_magforces_weight` | Use `loss.stage_two_weights.magforces`. Only named weights override the first stage; the old swa spelling remains a compatibility alias. |
| `train.huber_delta` | KEEP | `--huber_delta` | Use `loss.delta` with `loss.kind: huber` or `universal`; there is no delta on a weighted-square loss. |
| `train.optimizer` | KEEP | `--optimizer` | Use `training.optimizer.kind` (adam, adamw, schedulefree or lbfgs) and only the settings of that optimizer. |
| `train.beta` | KEEP | `--beta` | Use `training.optimizer.beta` with a compatible optimizer kind; the helper carries this setting with its optimizer. |
| `train.amsgrad` | KEEP | `--amsgrad` | Use `training.optimizer.amsgrad` with a compatible optimizer kind; the helper carries this setting with its optimizer. |
| `train.weight_decay` | KEEP | `--weight_decay` | Use `training.weight_decay`. The legacy flag is translated at the input boundary. |
| `train.beta1_schedulefree` | KEEP | `--beta1_schedulefree` | Use `training.optimizer.beta1` with a compatible optimizer kind; the helper carries this setting with its optimizer. |
| `train.beta2_schedulefree` | KEEP | `--beta2_schedulefree` | Use `training.optimizer.beta2` with a compatible optimizer kind; the helper carries this setting with its optimizer. |
| `train.warmup_steps_schedulefree` | KEEP | `--warmup_steps_schedulefree` | Use `training.optimizer.warmup_steps` with a compatible optimizer kind; the helper carries this setting with its optimizer. |
| `train.lbfgs` | KEEP | `--lbfgs` | KEEP the full-batch L-BFGS regime with `training.optimizer: {kind: lbfgs}`. The adapter selects a constant scheduler unless one was supplied. |
| `train.batch_size` | KEEP | `--batch_size` | Use `training.batch_size`. The legacy flag is translated at the input boundary. |
| `train.valid_batch_size` | KEEP | `--valid_batch_size` | Use `training.valid_batch_size`. The legacy flag is translated at the input boundary. |
| `train.lr` | KEEP | `--lr` | Use `training.lr`. The legacy flag is translated at the input boundary. |
| `train.lr_factor` | KEEP | `--lr_factor` | Use `training.scheduler.factor` with `training.scheduler.kind: plateau`. |
| `train.scheduler` | KEEP | `--scheduler` | Use `training.scheduler.kind`: ReduceLROnPlateau becomes plateau and ExponentialLR becomes exponential. |
| `train.scheduler_patience` | KEEP | `--scheduler_patience` | Use `training.scheduler.patience` with `training.scheduler.kind: plateau`. |
| `train.lr_scheduler_gamma` | KEEP | `--lr_scheduler_gamma` | Use `training.scheduler.gamma` with `training.scheduler.kind: exponential`. |
| `train.lr_params_factors` | MERGE | `--lr_params_factors` | Use `training.scheduler.group_factors`: typed per-parameter-group factors. |
| `train.swa` | MERGE | `--swa` `--stage_two` | Use `training.stage_two.enabled`; both --swa and --stage_two are still accepted. |
| `train.start_swa` | MERGE | `--start_swa` `--start_stage_two` | Use `training.stage_two.start_epoch`; the helper resolves an omitted start to 75% of max_num_epochs when enabled. |
| `train.swa_lr` | MERGE | `--swa_lr` `--stage_two_lr` | Use `training.stage_two.lr`; --stage_two_lr is the preferred compatibility spelling. |
| `train.ema` | KEEP | `--ema` | Use `training.ema.enabled`. The legacy flag is translated at the input boundary. |
| `train.ema_decay` | KEEP | `--ema_decay` | Use `training.ema.decay`. The legacy flag is translated at the input boundary. |
| `train.max_num_epochs` | KEEP | `--max_num_epochs` | Use `training.max_num_epochs`. The legacy flag is translated at the input boundary. |
| `train.patience` | KEEP | `--patience` | Use `training.patience`. The legacy flag is translated at the input boundary. |
| `train.eval_interval` | KEEP | `--eval_interval` | Use `training.eval_interval`. The legacy flag is translated at the input boundary. |
| `train.clip_grad` | KEEP | `--clip_grad` | Use `training.clip_grad`. The legacy flag is translated at the input boundary. |
| `train.dry_run` | KEEP | `--dry_run` | Use `training.dry_run`. The legacy flag is translated at the input boundary. |
| `train.restart_latest` | KEEP | `--restart_latest` | Use `runtime.restart_latest`. The legacy flag is translated at the input boundary. |
| `train.keep_checkpoints` | KEEP | `--keep_checkpoints` | Use `runtime.keep_checkpoints`. The legacy flag is translated at the input boundary. |
| `train.save_all_checkpoints` | KEEP | `--save_all_checkpoints` | Use `runtime.save_all_checkpoints`. The legacy flag is translated at the input boundary. |
| `train.save_cpu` | DROP | `--save_cpu` | Removed: v1 checkpoints are device-agnostic, so there is nothing to ask. Explicit nondefault values fail. |
| `train.enable_cueq` | MERGE | `--enable_cueq` | Use `model.backend` (`cueq` or `oeq`); the compatibility flags translate at the boundary and conflicting backends fail. Canonical weights need no conversion. |
| `train.enable_oeq` | MERGE | `--enable_oeq` | Use `model.backend` (`cueq` or `oeq`); the compatibility flags translate at the boundary and conflicting backends fail. Canonical weights need no conversion. |
| `train.only_cueq` | MERGE | `--only_cueq` | Use `model.backend` (`cueq` or `oeq`); the compatibility flags translate at the boundary and conflicting backends fail. Canonical weights need no conversion. |
| `train.wandb` | KEEP | `--wandb` | Use `runtime.wandb.enabled`. The legacy flag is translated at the input boundary. |
| `train.wandb_dir` | KEEP | `--wandb_dir` | Use `runtime.wandb.directory`. The legacy flag is translated at the input boundary. |
| `train.wandb_project` | KEEP | `--wandb_project` | Use `runtime.wandb.project`. The legacy flag is translated at the input boundary. |
| `train.wandb_entity` | KEEP | `--wandb_entity` | Use `runtime.wandb.entity`. The legacy flag is translated at the input boundary. |
| `train.wandb_name` | KEEP | `--wandb_name` | Use `runtime.wandb.name`. The legacy flag is translated at the input boundary. |
| `train.wandb_log_hypers` | KEEP | `--wandb_log_hypers` | V1 logs the complete resolved configuration in `runtime.wandb`; selecting a subset is rejected. |
| `train.magmom_key` | MERGE | `--magmom_key` | The conventional label key is retained; nondefault CLI values are rejected. For custom files use `KeySpecification` at the data API boundary (see python-api.md). |
| `train.magforces_key` | MERGE | `--magforces_key` | The conventional label key is retained; nondefault CLI values are rejected. For custom files use `KeySpecification` at the data API boundary (see python-api.md). |
| `train.m_max` | KEEP | `--m_max` | Use `model.magnetic.saturation`: a dict literal, an ordered list or one number in a list of strings. |
| `train.max_m_ell` | KEEP | `--max_m_ell` | Use `model.magnetic.lmax`. The legacy flag is translated at the input boundary. |
| `train.num_mag_radial_basis` | KEEP | `--num_mag_radial_basis` | Use `model.magnetic.num_basis`. The legacy flag is translated at the input boundary. |
| `train.num_mag_radial_basis_one_body` | KEEP | `--num_mag_radial_basis_one_body` | Use `model.magnetic.one_body_basis`. The legacy flag is translated at the input boundary. |
| `train.use_magmom_one_body` | KEEP | `--use_magmom_one_body` | Use `model.magnetic.one_body`. The legacy flag is translated at the input boundary. |
| `train.train_one_body_contribution` | KEEP | `--train_one_body_contribution` | Use `model.magnetic.train_one_body`. The legacy flag is translated at the input boundary. |
| `train.data_aug_magmom` | MERGE | `--data_aug_magmom` | Use `data.augmentations`: a training-data augmentation, which is the data layer's and not the model's. |
| `train.data_aug_magmom_mode` | MERGE | `--data_aug_magmom_mode` | Use `data.augmentations`: a training-data augmentation, which is the data layer's and not the model's. |

## Other changed surfaces

| Inventory ID | Decision | Legacy surface | Migration |
| --- | --- | --- | --- |
| `ep.mace_finetuning_select` | MERGE | `mace.cli.fine_tuning_select:main` | Use integrated head selection or the retained `mace data select` command; see commands.md. |
| `ep.mace_e3nn_cueq` | DROP | `mace.cli.convert_e3nn_cueq:main` | Removed backend-weight conversion. Canonical weights load across supported backends; select backend and device when building/loading. |
| `ep.mace_cueq_to_e3nn` | DROP | `mace.cli.convert_cueq_e3nn:main` | Removed: canonical checkpoints load across supported backends; select the backend when building/loading. |
| `ep.mace_active_learning_md` | DROP | `mace.cli.active_learning_md:main` | The MD driver is retired. Set this behavior in an ASE dynamics workflow using the V1 calculator; committee means and variances remain available. |
| `choice.BOTNet` | DROP | `--model BOTNet` | Removed legacy choice. Its old error recommended MACE / ScaleShiftMACE; use native `plain` / `scale_shift` configuration explicitly. The legacy `--model=MACE` training flag is not equivalent to the native plain architecture. |
| `choice.ScaleShiftBOTNet` | DROP | `--model ScaleShiftBOTNet` | Removed legacy choice. Its old error recommended MACE / ScaleShiftMACE; use native `plain` / `scale_shift` configuration explicitly. The legacy `--model=MACE` training flag is not equivalent to the native plain architecture. |
| `choice.MACE` | MERGE | `--model MACE` | Use `model.model` and `model.observables`; class spellings are compatibility inputs. Explicit `--model=MACE` is rejected because legacy forced a zero shift; choose a native architecture deliberately. |
| `choice.ScaleShiftMACE` | MERGE | `--model ScaleShiftMACE` | Use `model.model: scale_shift` with declared observables and readout configuration; build through the native model factory. |
| `choice.PolarMACE` | MERGE | `--model PolarMACE` | Use `model.model: polar` with declared observables and readout configuration; build through the native model factory. |
| `choice.MACELES` | MERGE | `--model MACELES` | Use `model.model: les` with declared observables and readout configuration; build through the native model factory. |
| `choice.AtomicDipolesMACE` | MERGE | `--model AtomicDipolesMACE` | Use `model.model: dipole` with declared observables and readout configuration; build through the native model factory. |
| `choice.AtomicDielectricMACE` | MERGE | `--model AtomicDielectricMACE` | Use `model.model: dielectric` with declared observables and readout configuration; build through the native model factory. |
| `choice.EnergyDipolesMACE` | MERGE | `--model EnergyDipolesMACE` | Use `model.model: scale_shift` with energy and dipole observables and readout configuration; build through the native model factory. |
| `choice.MagneticScaleShiftMACE` | MERGE | `--model MagneticScaleShiftMACE` | Use `model.model: magnetic` with declared observables and readout configuration; build through the native model factory. |
| `prep.config` | MERGE | `--config` | Use `mace data prepare --config` for one prepared source; see commands.md. Preparation and training share the data conventions. |
| `prep.train_file` | MERGE | `--train_file` | Use `mace data prepare --train_file` for one prepared source; see commands.md. Preparation and training share the data conventions. |
| `prep.valid_file` | MERGE | `--valid_file` | Use `mace data prepare --valid_file` for one prepared source; see commands.md. Preparation and training share the data conventions. |
| `prep.test_file` | MERGE | `--test_file` | Use `mace data prepare --test_file` for one prepared source; see commands.md. Preparation and training share the data conventions. |
| `prep.valid_fraction` | MERGE | `--valid_fraction` | Use `mace data prepare --valid_fraction` for one prepared source; see commands.md. Preparation and training share the data conventions. |
| `prep.work_dir` | MERGE | `--work_dir` | Use `mace data prepare --work_dir` for one prepared source; see commands.md. Preparation and training share the data conventions. |
| `prep.r_max` | MERGE | `--r_max` | Use `mace data prepare --r_max` for one prepared source; see commands.md. Preparation and training share the data conventions. |
| `prep.config_type_weights` | MERGE | `--config_type_weights` | Use `mace data prepare --config_type_weights` for one prepared source; see commands.md. Preparation and training share the data conventions. |
| `prep.energy_key` | MERGE | `--energy_key` | Use the preparation key flag to translate this input label with `KeySpecification`; training then reads the common convention. |
| `prep.forces_key` | MERGE | `--forces_key` | Use the preparation key flag to translate this input label with `KeySpecification`; training then reads the common convention. |
| `prep.virials_key` | MERGE | `--virials_key` | Use the preparation key flag to translate this input label with `KeySpecification`; training then reads the common convention. |
| `prep.stress_key` | MERGE | `--stress_key` | Use the preparation key flag to translate this input label with `KeySpecification`; training then reads the common convention. |
| `prep.dipole_key` | MERGE | `--dipole_key` | Use the preparation key flag to translate this input label with `KeySpecification`; training then reads the common convention. |
| `prep.polarizability_key` | MERGE | `--polarizability_key` | Use the preparation key flag to translate this input label with `KeySpecification`; training then reads the common convention. |
| `prep.charges_key` | MERGE | `--charges_key` | Use the preparation key flag to translate this input label with `KeySpecification`; training then reads the common convention. |
| `prep.head_key` | DROP | `--head_key` | Removed as a file-label setting. Assign a file to `data.heads.<name>`; the historical file key never selected a head. |
| `prep.heads` | MERGE | `--heads` | Removed shortcut: prepare one source/head per invocation, then reference separate sources in `data.heads`. |
| `prep.atomic_numbers` | MERGE | `--atomic_numbers` | Preparation validates its declared element table against the data; training obtains elements from datasets or `finetune.element_table`. |
| `prep.batch_size` | MERGE | `--batch_size` | Use `mace data prepare --batch_size` for one prepared source; see commands.md. Preparation and training share the data conventions. |
| `prep.scaling` | MERGE | `--scaling` | Use `mace data prepare --scaling` for one prepared source; see commands.md. Preparation and training share the data conventions. |
| `prep.E0s` | MERGE | `--E0s` | Use `mace data prepare --E0s` for one prepared source; see commands.md. Preparation and training share the data conventions. |
| `prep.seed` | MERGE | `--seed` | Use `mace data prepare --seed` for one prepared source; see commands.md. Preparation and training share the data conventions. |
| `cli.eval_configs.default_dtype` | MERGE | `--default_dtype` | The evaluation flag remains accepted and sets `PrecisionConfig.model` at loading. |
| `cli.eval_configs.compute_stress` | MERGE | `--compute_stress` | The evaluation flag remains accepted and requests the stress derivative from the model. |
| `cli.eval_configs.enable_cueq` | MERGE | `--enable_cueq` | The evaluation compatibility flag remains accepted and selects `backend="cueq"` at loading; no weight conversion. |
| `cli.eval_configs.magmom_key` | MERGE | `--magmom_key` | The evaluation key override remains accepted at the input boundary; training label flags have a different contract. |
| `cli.eval_configs.return_magforces` | MERGE | `--return_magforces` | The evaluation flag remains accepted and requests the magnetic derivative. |
| `cli.fine_tuning_select.configs_pt` | MERGE | `--configs_pt` | Use `data.heads.selected.train_file`. The migration helper translates selection config for `mace data select`; selection also runs inside training. |
| `cli.fine_tuning_select.configs_ft` | MERGE | `--configs_ft` | Use `data.heads.fine_tune_<index>.train_file`. The migration helper translates selection config for `mace data select`; selection also runs inside training. |
| `cli.fine_tuning_select.num_samples` | MERGE | `--num_samples` | Use `data.heads.selected.subselect.num_samples`. The migration helper translates selection config for `mace data select`; selection also runs inside training. |
| `cli.fine_tuning_select.subselect` | MERGE | `--subselect` | Use `data.heads.selected.subselect.method`. The migration helper translates selection config for `mace data select`; selection also runs inside training. |
| `cli.fine_tuning_select.model` | MERGE | `--model` | Use `finetune.foundation_model`. The migration helper translates selection config for `mace data select`; selection also runs inside training. |
| `cli.fine_tuning_select.output` | MERGE | `--output` | Use `data select --output`. The migration helper translates selection config for `mace data select`; selection also runs inside training. |
| `cli.fine_tuning_select.descriptors` | MERGE | `--descriptors` | Use `data.heads.selected.subselect.structure_descriptors_file`. The migration helper translates selection config for `mace data select`; selection also runs inside training. |
| `cli.fine_tuning_select.device` | MERGE | `--device` | Use `runtime.device`. The migration helper translates selection config for `mace data select`; selection also runs inside training. |
| `cli.fine_tuning_select.head_pt` | MERGE | `--head_pt` | Use `data.heads.selected.readout_from`. The migration helper translates selection config for `mace data select`; selection also runs inside training. |
| `cli.fine_tuning_select.head_ft` | MERGE | `--head_ft` | Use `data.heads.fine_tune_<index>.readout_from`. The migration helper translates selection config for `mace data select`; selection also runs inside training. |
| `cli.fine_tuning_select.filtering_type` | MERGE | `--filtering_type` | Use `data.heads.selected.subselect.filtering`. The migration helper translates selection config for `mace data select`; selection also runs inside training. |
| `cli.fine_tuning_select.weight_ft` | MERGE | `--weight_ft` | Use `data.heads.fine_tune_<index>.weight`. The migration helper translates selection config for `mace data select`; selection also runs inside training. |
| `cli.fine_tuning_select.weight_pt` | MERGE | `--weight_pt` | Use `data.heads.selected.weight`. The migration helper translates selection config for `mace data select`; selection also runs inside training. |
| `cli.fine_tuning_select.filter_atomic_numbers_pt` | MERGE | `--filter_atomic_numbers_pt` | Use `data.heads.selected.subselect.elements`. The migration helper translates selection config for `mace data select`; selection also runs inside training. |
| `cli.fine_tuning_select.allow_random_padding` | MERGE | `--disallow_random_padding` | Use `data.heads.selected.subselect.allow_random_padding`. The migration helper translates selection config for `mace data select`; selection also runs inside training. |
| `cli.fine_tuning_select.seed` | MERGE | `--seed` | Use `runtime.seed`. The migration helper translates selection config for `mace data select`; selection also runs inside training. |
| `cli.fine_tuning_select.default_dtype` | MERGE | `--default_dtype` | Use `precision.model`. The migration helper translates selection config for `mace data select`; selection also runs inside training. |
| `cli.fine_tuning_select.config` | MERGE | `--config` | Use `--config`. The migration helper translates selection config for `mace data select`; selection also runs inside training. |
| `cli.plot_train.start_swa` | MERGE | `--start_stage_two` `--start_swa` | Use `--start_stage_two` (the `--start_swa` alias remains accepted) or V1 stage metadata for plot annotations. |
| `cli.active_learning_md.config` | DROP | `--config` | The MD driver is retired. Set this behavior in an ASE dynamics workflow using the V1 calculator; committee means and variances remain available. |
| `cli.active_learning_md.config_index` | DROP | `--config_index` | The MD driver is retired. Set this behavior in an ASE dynamics workflow using the V1 calculator; committee means and variances remain available. |
| `cli.active_learning_md.error_threshold` | DROP | `--error_threshold` | The MD driver is retired. Set this behavior in an ASE dynamics workflow using the V1 calculator; committee means and variances remain available. |
| `cli.active_learning_md.temperature_K` | DROP | `--temperature_K` | The MD driver is retired. Set this behavior in an ASE dynamics workflow using the V1 calculator; committee means and variances remain available. |
| `cli.active_learning_md.friction` | DROP | `--friction` | The MD driver is retired. Set this behavior in an ASE dynamics workflow using the V1 calculator; committee means and variances remain available. |
| `cli.active_learning_md.timestep` | DROP | `--timestep` | The MD driver is retired. Set this behavior in an ASE dynamics workflow using the V1 calculator; committee means and variances remain available. |
| `cli.active_learning_md.nsteps` | DROP | `--nsteps` | The MD driver is retired. Set this behavior in an ASE dynamics workflow using the V1 calculator; committee means and variances remain available. |
| `cli.active_learning_md.nprint` | DROP | `--nprint` | The MD driver is retired. Set this behavior in an ASE dynamics workflow using the V1 calculator; committee means and variances remain available. |
| `cli.active_learning_md.nsave` | DROP | `--nsave` | The MD driver is retired. Set this behavior in an ASE dynamics workflow using the V1 calculator; committee means and variances remain available. |
| `cli.active_learning_md.ncheckerror` | DROP | `--ncheckerror` | The MD driver is retired. Set this behavior in an ASE dynamics workflow using the V1 calculator; committee means and variances remain available. |
| `cli.active_learning_md.model` | DROP | `--model` | The MD driver is retired. Set this behavior in an ASE dynamics workflow using the V1 calculator; committee means and variances remain available. |
| `cli.active_learning_md.output` | DROP | `--output` | The MD driver is retired. Set this behavior in an ASE dynamics workflow using the V1 calculator; committee means and variances remain available. |
| `cli.active_learning_md.device` | DROP | `--device` | The MD driver is retired. Set this behavior in an ASE dynamics workflow using the V1 calculator; committee means and variances remain available. |
| `cli.active_learning_md.default_dtype` | DROP | `--default_dtype` | The MD driver is retired. Set this behavior in an ASE dynamics workflow using the V1 calculator; committee means and variances remain available. |
| `cli.active_learning_md.compute_stress` | DROP | `--compute_stress` | The MD driver is retired. Set this behavior in an ASE dynamics workflow using the V1 calculator; committee means and variances remain available. |
| `cli.active_learning_md.info_prefix` | DROP | `--info_prefix` | The MD driver is retired. Set this behavior in an ASE dynamics workflow using the V1 calculator; committee means and variances remain available. |
| `cli.polar_density_cube.default_dtype` | MERGE | `--default_dtype` | The density-cube CLI retains this precision input; it controls the loaded model precision. |
| `cli.create_lammps_model.dtype` | MERGE | `--dtype` | Export retains `--dtype` and records the selected float32/float64 precision in its bundle. |
| `cli.create_lammps_model.format` | MERGE | `--format` | The old format selector is removed. `export lammps` writes a neutral ML-IAP bundle; `--native NODE_SLOTS EDGE_SLOTS` requests the Python-free AOTInductor target. Neither uses TorchScript. |
| `cli.convert_e3nn_cueq.input_model` | DROP | `input_model` | Removed backend-weight conversion. Canonical weights load across supported backends; select backend and device when building/loading. |
| `cli.convert_e3nn_cueq.output_model` | DROP | `--output_model` | Removed backend-weight conversion. Canonical weights load across supported backends; select backend and device when building/loading. |
| `cli.convert_e3nn_cueq.device` | DROP | `--device` | Removed backend-weight conversion. Canonical weights load across supported backends; select backend and device when building/loading. |
| `cli.convert_e3nn_cueq.return_model` | DROP | `--return_model` | Removed backend-weight conversion. Canonical weights load across supported backends; select backend and device when building/loading. |
| `cli.convert_cueq_e3nn.input_model` | DROP | `input_model` | Removed backend-weight conversion. Canonical weights load across supported backends; select backend and device when building/loading. |
| `cli.convert_cueq_e3nn.output_model` | DROP | `--output_model` | Removed backend-weight conversion. Canonical weights load across supported backends; select backend and device when building/loading. |
| `cli.convert_cueq_e3nn.device` | DROP | `--device` | Removed backend-weight conversion. Canonical weights load across supported backends; select backend and device when building/loading. |
| `cli.convert_cueq_e3nn.return_model` | DROP | `--return_model` | Removed backend-weight conversion. Canonical weights load across supported backends; select backend and device when building/loading. |
| `cli.convert_e3nn_oeq.input_model` | DROP | `input_model` | Removed backend-weight conversion. Canonical weights load across supported backends; select backend and device when building/loading. |
| `cli.convert_e3nn_oeq.output_model` | DROP | `--output_model` | Removed backend-weight conversion. Canonical weights load across supported backends; select backend and device when building/loading. |
| `cli.convert_e3nn_oeq.device` | DROP | `--device` | Removed backend-weight conversion. Canonical weights load across supported backends; select backend and device when building/loading. |
| `cli.convert_e3nn_oeq.return_model` | DROP | `--return_model` | Removed backend-weight conversion. Canonical weights load across supported backends; select backend and device when building/loading. |
| `cli.convert_oeq_e3nn.input_model` | DROP | `input_model` | Removed backend-weight conversion. Canonical weights load across supported backends; select backend and device when building/loading. |
| `cli.convert_oeq_e3nn.output_model` | DROP | `--output_model` | Removed backend-weight conversion. Canonical weights load across supported backends; select backend and device when building/loading. |
| `cli.convert_oeq_e3nn.device` | DROP | `--device` | Removed backend-weight conversion. Canonical weights load across supported backends; select backend and device when building/loading. |
| `cli.convert_oeq_e3nn.return_model` | DROP | `--return_model` | Removed backend-weight conversion. Canonical weights load across supported backends; select backend and device when building/loading. |
| `cli.convert_e3nn_hybrid.input_model` | DROP | `input_model` | Removed backend-weight conversion. Canonical weights load across supported backends; select backend and device when building/loading. |
| `cli.convert_e3nn_hybrid.output_model` | DROP | `--output_model` | Removed backend-weight conversion. Canonical weights load across supported backends; select backend and device when building/loading. |
| `cli.convert_e3nn_hybrid.device` | DROP | `--device` | Removed backend-weight conversion. Canonical weights load across supported backends; select backend and device when building/loading. |
| `cli.convert_e3nn_hybrid.return_model` | DROP | `--return_model` | Removed backend-weight conversion. Canonical weights load across supported backends; select backend and device when building/loading. |
| `model.MACE` | MERGE | `MACE` | Construct with `build_model` and `ModelConfig(model="plain")`; this is distinct from the historical training flag, which forced a scale-shift model. |
| `model.ScaleShiftMACE` | MERGE | `ScaleShiftMACE` | Use `model.model: scale_shift` with declared observables and readout configuration; build through the native model factory. |
| `model.AtomicDipolesMACE` | MERGE | `AtomicDipolesMACE` | Use `model.model: dipole` with declared observables and readout configuration; build through the native model factory. |
| `model.AtomicDielectricMACE` | MERGE | `AtomicDielectricMACE` | Use `model.model: dielectric` with declared observables and readout configuration; build through the native model factory. |
| `model.EnergyDipolesMACE` | MERGE | `EnergyDipolesMACE` | Use `model.model: scale_shift` with energy and dipole observables and readout configuration; build through the native model factory. |
| `reg.RealAgnosticAttResidualInteractionBlock` | DROP | `RealAgnosticAttResidualInteractionBlock` — interaction_classes | Removed research attention variant: no published model or supported migration. Select a supported interaction explicitly. |
| `reg.LinearDipoleReadoutBlock` | MERGE | `LinearDipoleReadoutBlock` — readout_classes | An observable head declared in the output spec. |
| `reg.NonLinearDipoleReadoutBlock` | MERGE | `NonLinearDipoleReadoutBlock` — readout_classes | Declare the dipole observable and configure its nonlinear readout; no legacy class registry import. |
| `reg.rms_dipoles_scaling` | MERGE | `rms_dipoles_scaling` — scaling_classes | Observable normalization. |
| `block.RealAgnosticInteractionBlock` | MERGE | `RealAgnosticInteractionBlock` | Configure `model.interaction` and `model.interaction_first`; density and residual behavior belong to the declared interaction, not a legacy class import. |
| `block.RealAgnosticDensityInteractionBlock` | MERGE | `RealAgnosticDensityInteractionBlock` | Configure `model.interaction` and `model.interaction_first`; density and residual behavior belong to the declared interaction, not a legacy class import. |
| `block.RealAgnosticDensityResidualInteractionBlock` | MERGE | `RealAgnosticDensityResidualInteractionBlock` | Configure `model.interaction` and `model.interaction_first`; density and residual behavior belong to the declared interaction, not a legacy class import. |
| `block.RealAgnosticAttResidualInteractionBlock` | MERGE | `RealAgnosticAttResidualInteractionBlock` | Removed research attention variant; no numerical-equivalence claim for substituting another interaction. |
| `block.MagneticRealAgnosticResidueSpinOrbitCoupledDensityInteractionBlock` | MERGE | `MagneticRealAgnosticResidueSpinOrbitCoupledDensityInteractionBlock` | Use the magnetic model configuration with the residual SOC interaction; no legacy class alias. |
| `contraction.EmptyParam` | MERGE | `EmptyParam` | Weight bookkeeping for an unreachable order, not a feature. |
| `transform.Random3DRotation` | MERGE | `Random3DRotation` | The `--data_aug_magmom` transform, which travels with the flag. |
| `calc.class.MACELammpsConfig` | MERGE | `MACELammpsConfig` | The ML-IAP wrapper's own config object, an implementation detail. |
| `loss.WeightedEnergyForcesLoss` | MERGE | `WeightedEnergyForcesLoss` | Use `build_loss` with a structured `loss` configuration: a composition preset (the `ef`/`weighted` schemes). No legacy loss-class alias. |
| `loss.WeightedForcesLoss` | MERGE | `WeightedForcesLoss` | Use `build_loss` with a structured `loss` configuration: a composition preset (the `forces_only` scheme). No legacy loss-class alias. |
| `loss.WeightedEnergyForcesStressLoss` | MERGE | `WeightedEnergyForcesStressLoss` | Use `build_loss` with a structured `loss` configuration: a composition preset (the `stress` scheme). No legacy loss-class alias. |
| `loss.WeightedHuberEnergyForcesStressLoss` | MERGE | `WeightedHuberEnergyForcesStressLoss` | Use `build_loss` with a structured `loss` configuration: a composition preset (the `huber` scheme). No legacy loss-class alias. |
| `loss.UniversalLoss` | MERGE | `UniversalLoss` | Use `build_loss` with a structured `loss` configuration: a composition preset (the `universal` scheme). No legacy loss-class alias. |
| `loss.WeightedEnergyForcesVirialsLoss` | MERGE | `WeightedEnergyForcesVirialsLoss` | Use `build_loss` with a structured `loss` configuration: a composition preset (the `virials` scheme). No legacy loss-class alias. |
| `loss.DipoleSingleLoss` | MERGE | `DipoleSingleLoss` | Use `build_loss` with a structured `loss` configuration: a composition preset (the `dipole` scheme). No legacy loss-class alias. |
| `loss.DipolePolarLoss` | MERGE | `DipolePolarLoss` | Use `build_loss` with a structured `loss` configuration: a composition preset (the `dipole_polar` scheme). No legacy loss-class alias. |
| `loss.WeightedEnergyForcesDipoleLoss` | MERGE | `WeightedEnergyForcesDipoleLoss` | Use `build_loss` with a structured `loss` configuration: a composition preset (the `energy_forces_dipole` scheme). No legacy loss-class alias. |
| `loss.WeightedEnergyForcesL1L2Loss` | MERGE | `WeightedEnergyForcesL1L2Loss` | Use `build_loss` with a structured `loss` configuration: a composition preset (the `l1l2energyforces` scheme). No legacy loss-class alias. |
| `calc.param.default_dtype` | MERGE | `default_dtype` — MACECalculator | Prefer `PrecisionConfig` through `precision=`; `default_dtype=` remains a compatibility input. Supplying both fails. |
| `calc.param.model_type` | MERGE | `model_type` — MACECalculator | Auto-detected from model metadata; asking the user to name the model family is asking them to get it wrong. |
| `calc.param.fullgraph` | MERGE | `fullgraph` — MACECalculator | Removed. `compile_mode` uses full-graph compilation; graph breaks fail rather than silently changing execution. |
| `calc.param.enable_cueq` | MERGE | `enable_cueq` — MACECalculator | Load with `load_deployed(..., backend="cueq")` and pass `models=`; enabling this legacy calculator kwarg is rejected. |
| `calc.param.enable_oeq` | MERGE | `enable_oeq` — MACECalculator | Load with `load_deployed(..., backend="oeq")` and pass `models=`; enabling this legacy calculator kwarg is rejected. |
| `calc.param.model_path` | DROP | `model_path` — read from `**kwargs` on MACECalculator | Use `model_paths`. The singular alias now raises instead of warning and forwarding. |
| `calc.export.LAMMPS_MACE` | DROP | `LAMMPS_MACE` | Removed with TorchScript. Use the native torch.export/AOTInductor LAMMPS target or the separate ML-IAP compatibility bundle; see commands.md. |
| `calc.export.mace_anicc` | DROP | `mace_anicc` | A 2023 organic-chemistry model superseded by MACE-OFF, and the only loader with a divergent signature (`model_path` instead of `model`): an API exception for an obsolete artifact. Its tracked checkpoint `mace/calculators/foundations_models/ani500k_large_CC.model` goes with it; the release notes say "use MACE-OFF". |
| `out.model.displacement` | MERGE | `displacement` — first declared by `MACE` | The strain displacement is internal machinery of the derivative engine, not a user-facing output; v1 does not return it. |
| `out.model.charges_history` | MERGE | `charges_history` — first declared by `PolarMACE` | Use `spin_charge_density`. Legacy returned one copy of the final density, not an iteration trace; recover the old shape with `spin_charge_density.detach().unsqueeze(-1)`. |
| `out.model.scf_steps` | MERGE | `scf_steps` — first declared by `MagneticSCFMACE` | Read `output.extras["fixed_point_steps"]` from `FixedPointDriver`; this is solver telemetry, not a declared observable. |
| `out.model.scf_energy_history` | MERGE | `scf_energy_history` — first declared by `MagneticSCFMACE` | Read `output.extras["fixed_point_history"]` from `FixedPointDriver`; this is solver telemetry, not a declared observable. |
| `out.calc.LES_alphas` | MERGE | `LES_alphas` — self.results[...] | Core output is `latent_alphas`; the ASE results mapping retains `LES_alphas` for compatibility. |
| `out.calc.LES_kappas` | MERGE | `LES_kappas` — self.results[...] | Core output is `latent_kappas`; the ASE results mapping retains `LES_kappas` for compatibility. |
| `out.calc.MACE_magmoms` | MERGE | `MACE_magmoms` — self.results[...] | Core output is `converged_magmom`; ASE retains `MACE_magmoms` and the atoms-array writeback for compatibility. |
| `env.MACE_USE_CUEQ_CG` | DROP | `MACE_USE_CUEQ_CG` | The variable goes, not the capability: an environment variable that silently changes model numerics is unreproducible and never lands in the run metadata; it is what makes machine-to-machine differences unexplainable. The CG source becomes a backend decision recorded in the resolved config. |
| `stdenv.TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD` | DROP | `TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD` — set by `mace/__init__.py` so pickled checkpoints load under newer torch defaults | V1 checkpoints are neutral-format (safetensors + manifest), so nothing needs the unsafe-pickle escape hatch; the legacy loader keeps it until the converter is the only reader. |
| `key.HEAD` | DROP | `HEAD` = `"head"` | Removed as a file-label setting. Assign a file to `data.heads.<name>`; the historical file key never selected a head. |
| `kwarg.default_dtype` | MERGE | `default_dtype=` — shared by every loader | Loader compatibility input; prefer `PrecisionConfig`. Registry precision defaults remain family-specific (foundations.md). |
| `kwarg.model_path` | DROP | `mace_anicc(model_path=…)` — the one loader whose first argument is spelled differently from every other | Removed with the ANI-CC loader; choose an OFF23 model using the shared model registry. |
| `lammps.torchscript_wrapper` | DROP | `LAMMPS_MACE` (`@compile_mode("script")`) + the `-lammps.pt` artifact `mace_create_lammps_model` writes by default | User decision 2026-11-07 excludes TorchScript, including the DEP-2 fallback. V1 supports the neutral ML-IAP bundle and the torch.export/AOTInductor native target. |
| `lammps.compiled_side_artifact` | DROP | The `_compiled.model` / `_stagetwo_compiled.model` TorchScript artifacts training emits next to every checkpoint, each inside a bare `except Exception: pass` | A deliberate, recorded removal: v1 checkpoints are neutral-format only and deployment artifacts come solely from `mace export`. |
| `pkg.vendored_torch_geometric` | DROP | The vendored `torch_geometric` copy (`Data`, `Batch`, `DataLoader`, `scatter`) | V1 collates without torch_geometric; the vendored copy is excluded from lint and mypy today, which is the clearest sign it is not maintained code. Complication: `mace/data/augmentation.py` imports the *real* package while the rest of the tree imports the vendored one, and the `[magnetic]` extra declares external `torch-geometric` — so both must go at once. |
| `pkg.extra_property_promotion` | DROP | `AtomicData.from_config` turns every leftover property into a tensor, casts floats to the default dtype and promotes a 1-D value to `(n, 1)`, guessing that it is per atom | An input or label that reaches the graph is declared (an embedding feature with its `per`, or an observable); an undeclared property is carried untouched and never shape-guessed. |
| `pkg.compile_utils` | MERGE | `prepare` / `simplify_if_compile` — utilities that make legacy modules `torch.compile`-able | V1 is compile-first, so the retrofit mechanism has nothing to retrofit. |
| `pkg.public_import_surface` | DROP | The public Python API downstream projects import — `mace.modules.MACE`, `mace.data.AtomicData`, `mace.cli.run_train.run(args)`, … | No legacy import aliases. Use the explicit old-to-new map and examples in python-api.md. |
| `pkg.anicc_checkpoint` | DROP | The tracked MACE-ANI-CC checkpoint available in source/editable installs (never bundled in the legacy wheel) | ANI-CC is retired in favor of OFF23. Its pickle was source/editable-only, never bundled in the legacy wheel. |
| `pkg.bundled_mpa_default` | DROP | Source/editable `mace_mp()` short-circuits to bundled MPA-0 medium | V1 resolves native safetensors/JSON through the shared cache; a fresh checkout downloads once. The legacy wheel never included this pickle. |
| `pkg.bundled_unused_mp` | DROP | Tracked `2023-12-03-mace-mp.model`, with no loader consumer | Unused legacy pickle; v1 ships no foundation artifact and legacy retains its frozen copy until retirement. |
| `pkg.bundled_unused_e0` | DROP | Tracked `mp_vasp_e0.json`, with no consumer | Unused legacy data; v1 reads the checkpoint E0 tensors and ships no standalone foundation E0 table. |
| `pkg.lr_param_groups` | MERGE | Explicit optimizer parameter groups, driven by `--lr_params_factors` and reused by `--freeze` | Typed per-param-group fields of the per-stage optimizer config. |
| `pkg.augmentation` | MERGE | `Random3DRotation` — the training-data augmentation transform behind `--data_aug_magmom` | A registered training-data transform, not a model flag. |
| `pkg.first_block_coercion` | DROP | For `--model MACE`, anything outside that allowlist is **silently rewritten** to `RealAgnosticInteractionBlock` | A config value the tool overwrites without a word is worse than a rejected one: the run trains a different architecture than the user asked for and nothing says so. v1 fails the combination in config validation. |
| `fm.mace_anicc` | DROP | MACE-ANI-CC — the 2023 organic-chemistry model | Superseded by MACE-OFF, not included in the legacy wheel; the release notes say "use MACE-OFF". |
