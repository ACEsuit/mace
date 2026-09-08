"""The MACE v1.0 disposition table: every surface v1 removes or replaces.

Generated from the rewrite's feature inventory, which carries a KEEP / MERGE /
DROP disposition and a reason for every enumerable surface of this package. Only
the non-KEEP rows are reproduced here, because those are the ones a 0.3.x user has
to hear about before the next major version takes them away.

The inventory is tests/golden/feature_inventory.md, in this tree, and
tests/unit/test_deprecation.py asserts the two agree row for row: same ids, same
dispositions, in both directions. So a flag that gains a DROP row there and no
row here fails the build rather than disappearing on a user without warning.
Add the row in the same change, and keep the id, because the id is what makes
the two comparable.

Each row is (id, kind, what, why):

* id    the inventory's stable identifier, namespaced by surface (train., cli.,
        model., out.calc., ...). Emission sites cite it, and the ids are what
        keeps this table checkable against the inventory it came from.
* kind  DROP, the functionality goes; or MERGE, it survives under a more general
        mechanism and only this spelling of it goes.
* what  the option string, class or key as a user writes it.
* why   what replaces it, or why it leaves.

No entry names a v1 command. The last 0.3.x release predates the v1 CLI, so an
imperative "use <new command>" would point at a binary the reader cannot run; the
v1.0 migration guide is the signpost for the new spellings.
"""

from typing import Tuple

DROP = "DROP"
MERGE = "MERGE"

#: (id, kind, what, why), one row per non-KEEP surface of the inventory.
DISPOSITIONS: Tuple[Tuple[str, str, str, str], ...] = (
    (
        "ep.mace_finetuning_select",
        MERGE,
        "mace_finetuning_select",
        "absorbed into the integrated fine-tuning pipeline, driven by a fine-tuning "
        "config rather than a separate selection command",
    ),
    (
        "ep.mace_e3nn_cueq",
        DROP,
        "mace_e3nn_cueq",
        "v1 weights are canonical and backend dispatch is automatic, so there is "
        "nothing left to convert between backends",
    ),
    (
        "ep.mace_cueq_to_e3nn",
        DROP,
        "mace_cueq_to_e3nn",
        "the reverse direction; v1 weights are canonical and backend dispatch is "
        "automatic, so there is nothing left to convert between backends",
    ),
    (
        "ep.mace_active_learning_md",
        DROP,
        "mace_active_learning_md",
        "out of v1.0 scope: an ASE MD loop a user writes in ~30 lines over the "
        "calculator, at the cost of MACE owning thermostat, timestep and trajectory "
        "I/O. The committee variance it consumes stays in calculate; the release "
        "notes document the recipe",
    ),
    (
        "choice.BOTNet",
        DROP,
        "--model BOTNet",
        "no BOTNet class exists anywhere in the tree; the choice reaches only a "
        "deprecation raise. Use MACE instead",
    ),
    (
        "choice.ScaleShiftBOTNet",
        DROP,
        "--model ScaleShiftBOTNet",
        "no ScaleShiftBOTNet class exists anywhere in the tree; the choice reaches "
        "only a deprecation raise. Use MACE instead",
    ),
    (
        "choice.MACE",
        MERGE,
        "--model MACE",
        "model composition is config-driven, not a class name",
    ),
    (
        "choice.ScaleShiftMACE",
        MERGE,
        "--model ScaleShiftMACE",
        "the default energy model becomes the default configuration; model "
        "composition is config-driven, not a class name",
    ),
    (
        "choice.PolarMACE",
        MERGE,
        "--model PolarMACE",
        "selected by declaring the electrostatics observables; model composition is "
        "config-driven, not a class name",
    ),
    (
        "choice.MACELES",
        MERGE,
        "--model MACELES",
        "selected by declaring the LES long-range term; model composition is "
        "config-driven, not a class name",
    ),
    (
        "choice.AtomicDipolesMACE",
        MERGE,
        "--model AtomicDipolesMACE",
        "selected by declaring the dipole observable; model composition is "
        "config-driven, not a class name",
    ),
    (
        "choice.AtomicDielectricMACE",
        MERGE,
        "--model AtomicDielectricMACE",
        "dipole + polarizability observables; model composition is config-driven, "
        "not a class name",
    ),
    (
        "choice.EnergyDipolesMACE",
        MERGE,
        "--model EnergyDipolesMACE",
        "energy + dipole observables; model composition is config-driven, not a "
        "class name",
    ),
    (
        "choice.MagneticScaleShiftMACE",
        MERGE,
        "--model MagneticScaleShiftMACE",
        "the only magnetic entry in the choices; model composition is "
        "config-driven, not a class name",
    ),
    (
        "train.config",
        MERGE,
        "--config",
        "the v1 config system makes this first-class (TOML/YAML/JSON, always "
        "available, resolved config saved as run metadata)",
    ),
    (
        "train.log_dir",
        MERGE,
        "--log_dir",
        "it becomes single work-dir layout convention",
    ),
    (
        "train.model_dir",
        MERGE,
        "--model_dir",
        "it becomes single work-dir layout convention",
    ),
    (
        "train.checkpoints_dir",
        MERGE,
        "--checkpoints_dir",
        "it becomes single work-dir layout convention",
    ),
    (
        "train.results_dir",
        MERGE,
        "--results_dir",
        "it becomes single work-dir layout convention",
    ),
    (
        "train.downloads_dir",
        MERGE,
        "--downloads_dir",
        "it becomes XDG cache-dir convention",
    ),
    (
        "train.default_dtype",
        MERGE,
        "--default_dtype",
        "it becomes PrecisionConfig",
    ),
    (
        "train.plot_interaction_e",
        DROP,
        "--plot_interaction_e",
        "niche diagnostic that drags model introspection into the plotting path",
    ),
    (
        "train.model",
        MERGE,
        "--model",
        "model composition is config-driven (BaseMACE + declared outputs), not a "
        "class name",
    ),
    (
        "train.use_last_readout_only",
        MERGE,
        "--use_last_readout_only",
        "readout policy: once you declare which layers read out, reading only the "
        "last one is configuration, not a boolean",
    ),
    (
        "train.use_embedding_readout",
        MERGE,
        "--use_embedding_readout",
        "readout policy: once you declare which layers read out, also reading the "
        "embedding layer is configuration, not a boolean",
    ),
    (
        "train.use_reduced_cg",
        MERGE,
        "--use_reduced_cg",
        "a CG-representation choice the backend makes, not a modelling decision a "
        "user can judge, and it changes numerics; convert_e3nn_hybrid.py defaults "
        "it to True, so checkpoints carry it and the converter must read it rather "
        "than assume",
    ),
    (
        "train.use_so3",
        DROP,
        "--use_so3",
        "a global parity-convention switch that doubles the irrep-handling surface "
        "in exactly the layer v1 rewrites; no published model sets it",
    ),
    (
        "train.return_electrostatic_potentials",
        MERGE,
        "--return_electrostatic_potentials",
        "an observable declared in the output spec, not a model flag",
    ),
    (
        "train.avg_num_neighbors",
        MERGE,
        "--avg_num_neighbors",
        "it becomes dataset statistics recorded in the model metadata",
    ),
    (
        "train.compute_avg_num_neighbors",
        MERGE,
        "--compute_avg_num_neighbors",
        "it becomes dataset statistics recorded in the model metadata",
    ),
    (
        "train.compute_stress",
        MERGE,
        "--compute_stress",
        "the observable spec drives this: an observable that is declared is "
        "computed, and one that is not is not",
    ),
    (
        "train.compute_forces",
        MERGE,
        "--compute_forces",
        "the observable spec drives this: an observable that is declared is "
        "computed, and one that is not is not",
    ),
    (
        "train.compute_polarizability",
        MERGE,
        "--compute_polarizability",
        "the observable spec drives this: an observable that is declared is "
        "computed, and one that is not is not",
    ),
    (
        "train.compute_atomic_dipole",
        MERGE,
        "--compute_atomic_dipole",
        "the observable spec drives this: an observable that is declared is "
        "computed, and one that is not is not",
    ),
    (
        "train.compute_magforces",
        MERGE,
        "--compute_magforces",
        "the observable spec drives this: dE/dm is a declared derivative exactly "
        "like forces and stress",
    ),
    (
        "train.multi_processed_test",
        MERGE,
        "--multi_processed_test",
        "the data layer infers sharding from the dataset; whether a test set is "
        "split across files is not something the user should have to declare and "
        "get wrong (today it is a bare if in run_train.py)",
    ),
    (
        "train.atomic_numbers",
        MERGE,
        "--atomic_numbers",
        "it becomes statistics / model metadata",
    ),
    (
        "train.mean",
        MERGE,
        "--mean",
        "it becomes statistics override",
    ),
    (
        "train.std",
        MERGE,
        "--std",
        "it becomes statistics override",
    ),
    (
        "train.energy_key",
        MERGE,
        "--energy_key",
        "property-key convention of the observable spec",
    ),
    (
        "train.forces_key",
        MERGE,
        "--forces_key",
        "property-key convention of the observable spec",
    ),
    (
        "train.virials_key",
        MERGE,
        "--virials_key",
        "property-key convention of the observable spec",
    ),
    (
        "train.stress_key",
        MERGE,
        "--stress_key",
        "property-key convention of the observable spec",
    ),
    (
        "train.dipole_key",
        MERGE,
        "--dipole_key",
        "property-key convention of the observable spec",
    ),
    (
        "train.polarizability_key",
        MERGE,
        "--polarizability_key",
        "property-key convention of the observable spec",
    ),
    (
        "train.charges_key",
        MERGE,
        "--charges_key",
        "property-key convention of the observable spec",
    ),
    (
        "train.head_key",
        MERGE,
        "--head_key",
        "property-key convention of the observable spec",
    ),
    (
        "train.force_mh_ft_lr",
        DROP,
        "--force_mh_ft_lr",
        "replay-dependent defaults replace the override; the flag exists only to "
        "defeat a heuristic v1 does not have",
    ),
    (
        "train.loss",
        MERGE,
        "--loss",
        "the 11 named schemes become loss-composition presets over the 10 loss "
        "classes of",
    ),
    (
        "train.swa_energy_weight",
        MERGE,
        "--swa_energy_weight / --stage_two_energy_weight",
        "per-stage schedules; arbitrary stages replace the two-stage special case, "
        "and the swa spelling dies with the namespace",
    ),
    (
        "train.swa_forces_weight",
        MERGE,
        "--swa_forces_weight / --stage_two_forces_weight",
        "per-stage schedules; arbitrary stages replace the two-stage special case, "
        "and the swa spelling dies with the namespace",
    ),
    (
        "train.swa_virials_weight",
        MERGE,
        "--swa_virials_weight / --stage_two_virials_weight",
        "per-stage schedules; arbitrary stages replace the two-stage special case, "
        "and the swa spelling dies with the namespace",
    ),
    (
        "train.swa_stress_weight",
        MERGE,
        "--swa_stress_weight / --stage_two_stress_weight",
        "per-stage schedules; arbitrary stages replace the two-stage special case, "
        "and the swa spelling dies with the namespace",
    ),
    (
        "train.swa_dipole_weight",
        MERGE,
        "--swa_dipole_weight / --stage_two_dipole_weight",
        "per-stage schedules; arbitrary stages replace the two-stage special case, "
        "and the swa spelling dies with the namespace",
    ),
    (
        "train.swa_polarizability_weight",
        MERGE,
        "--swa_polarizability_weight / --stage_two_polarizability_weight",
        "per-stage schedules; arbitrary stages replace the two-stage special case, "
        "and the swa spelling dies with the namespace",
    ),
    (
        "train.swa_magforces_weight",
        MERGE,
        "--swa_magforces_weight / --stage_two_magforces_weight",
        "per-stage schedules; arbitrary stages replace the two-stage special case, "
        "and the swa spelling dies with the namespace",
    ),
    (
        "train.lr_params_factors",
        MERGE,
        "--lr_params_factors",
        "typed per-param-group fields of the per-stage optimizer config; the "
        "capability stays (--freeze reuses it by zeroing factors), the hand-parsed "
        "JSON-in-a-string dies",
    ),
    (
        "train.swa",
        MERGE,
        "--swa / --stage_two",
        "stage two becomes a preset second stage of an arbitrary-stage schedule",
    ),
    (
        "train.start_swa",
        MERGE,
        "--start_swa / --start_stage_two",
        "stage two becomes a preset second stage of an arbitrary-stage schedule",
    ),
    (
        "train.swa_lr",
        MERGE,
        "--swa_lr / --stage_two_lr",
        "stage two becomes a preset second stage of an arbitrary-stage schedule",
    ),
    (
        "train.save_cpu",
        DROP,
        "--save_cpu",
        "safetensors checkpoints are device-agnostic, so there is nothing to choose",
    ),
    (
        "train.enable_cueq",
        MERGE,
        "--enable_cueq",
        "it becomes backend dispatch config",
    ),
    (
        "train.enable_oeq",
        MERGE,
        "--enable_oeq",
        "it becomes backend dispatch config",
    ),
    (
        "train.only_cueq",
        MERGE,
        "--only_cueq",
        "'use cueq for every op, not just the ones that benefit' becomes a dispatch "
        "policy, not a second boolean. Its own row precisely because a group-level "
        "--enable_cueq/--only_cueq/--enable_oeq cell hides it; backend dispatch "
        "config",
    ),
    (
        "train.magmom_key",
        MERGE,
        "--magmom_key",
        "property-key convention; the default REF_magmom extends the on-disk data "
        "contract of",
    ),
    (
        "train.magforces_key",
        MERGE,
        "--magforces_key",
        "default REF_magforces; property-key convention; the default REF_magmom "
        "extends the on-disk data contract of",
    ),
    (
        "train.data_aug_magmom",
        MERGE,
        "--data_aug_magmom",
        "a training-data transform (Random3DRotation), not a model flag",
    ),
    (
        "train.data_aug_magmom_mode",
        MERGE,
        "--data_aug_magmom_mode",
        "selects which spin symmetry the transform draws from (non-soc the full "
        "O(3), soc the sign flip alone), so it travels with data_aug_magmom",
    ),
    (
        "prep.config",
        MERGE,
        "--config",
        "same mechanism and disposition as the training parser's config",
    ),
    (
        "prep.train_file",
        MERGE,
        "--train_file",
        "the v1 config system declares this once for the whole run instead of "
        "redeclaring it on the data-preparation command",
    ),
    (
        "prep.valid_file",
        MERGE,
        "--valid_file",
        "the v1 config system declares this once for the whole run instead of "
        "redeclaring it on the data-preparation command",
    ),
    (
        "prep.test_file",
        MERGE,
        "--test_file",
        "the v1 config system declares this once for the whole run instead of "
        "redeclaring it on the data-preparation command",
    ),
    (
        "prep.valid_fraction",
        MERGE,
        "--valid_fraction",
        "the v1 config system declares this once for the whole run instead of "
        "redeclaring it on the data-preparation command",
    ),
    (
        "prep.work_dir",
        MERGE,
        "--work_dir",
        "the v1 config system declares this once for the whole run instead of "
        "redeclaring it on the data-preparation command",
    ),
    (
        "prep.r_max",
        MERGE,
        "--r_max",
        "the v1 config system declares this once for the whole run instead of "
        "redeclaring it on the data-preparation command",
    ),
    (
        "prep.config_type_weights",
        MERGE,
        "--config_type_weights",
        "the v1 config system declares this once for the whole run instead of "
        "redeclaring it on the data-preparation command",
    ),
    (
        "prep.energy_key",
        MERGE,
        "--energy_key",
        "the v1 config system declares this once for the whole run instead of "
        "redeclaring it on the data-preparation command",
    ),
    (
        "prep.forces_key",
        MERGE,
        "--forces_key",
        "the v1 config system declares this once for the whole run instead of "
        "redeclaring it on the data-preparation command",
    ),
    (
        "prep.virials_key",
        MERGE,
        "--virials_key",
        "the v1 config system declares this once for the whole run instead of "
        "redeclaring it on the data-preparation command",
    ),
    (
        "prep.stress_key",
        MERGE,
        "--stress_key",
        "the v1 config system declares this once for the whole run instead of "
        "redeclaring it on the data-preparation command",
    ),
    (
        "prep.dipole_key",
        MERGE,
        "--dipole_key",
        "the v1 config system declares this once for the whole run instead of "
        "redeclaring it on the data-preparation command",
    ),
    (
        "prep.polarizability_key",
        MERGE,
        "--polarizability_key",
        "the v1 config system declares this once for the whole run instead of "
        "redeclaring it on the data-preparation command",
    ),
    (
        "prep.charges_key",
        MERGE,
        "--charges_key",
        "the v1 config system declares this once for the whole run instead of "
        "redeclaring it on the data-preparation command",
    ),
    (
        "prep.head_key",
        MERGE,
        "--head_key",
        "the v1 config system declares this once for the whole run instead of "
        "redeclaring it on the data-preparation command",
    ),
    (
        "prep.heads",
        MERGE,
        "--heads",
        "the v1 config system declares this once for the whole run instead of "
        "redeclaring it on the data-preparation command",
    ),
    (
        "prep.atomic_numbers",
        MERGE,
        "--atomic_numbers",
        "the v1 config system declares this once for the whole run instead of "
        "redeclaring it on the data-preparation command",
    ),
    (
        "prep.batch_size",
        MERGE,
        "--batch_size",
        "the v1 config system declares this once for the whole run instead of "
        "redeclaring it on the data-preparation command",
    ),
    (
        "prep.scaling",
        MERGE,
        "--scaling",
        "the v1 config system declares this once for the whole run instead of "
        "redeclaring it on the data-preparation command",
    ),
    (
        "prep.E0s",
        MERGE,
        "--E0s",
        "the v1 config system declares this once for the whole run instead of "
        "redeclaring it on the data-preparation command",
    ),
    (
        "prep.seed",
        MERGE,
        "--seed",
        "the v1 config system declares this once for the whole run instead of "
        "redeclaring it on the data-preparation command",
    ),
    (
        "cli.eval_configs.default_dtype",
        MERGE,
        "--default_dtype",
        "it becomes PrecisionConfig",
    ),
    (
        "cli.eval_configs.compute_stress",
        MERGE,
        "--compute_stress",
        "it becomes part of the observable spec: an observable that is declared is "
        "computed",
    ),
    (
        "cli.eval_configs.enable_cueq",
        MERGE,
        "--enable_cueq",
        "it becomes backend dispatch config",
    ),
    (
        "cli.eval_configs.magmom_key",
        MERGE,
        "--magmom_key",
        "it becomes property-key convention",
    ),
    (
        "cli.eval_configs.return_magforces",
        MERGE,
        "--return_magforces",
        "observable spec (dE/dm declared like any other derivative)",
    ),
    (
        "cli.fine_tuning_select.configs_pt",
        MERGE,
        "--configs_pt",
        "absorbed into the integrated fine-tuning pipeline, driven by a fine-tuning "
        "config rather than a separate selection command; selection stops being a "
        "separate CLI over a separate model load",
    ),
    (
        "cli.fine_tuning_select.configs_ft",
        MERGE,
        "--configs_ft",
        "absorbed into the integrated fine-tuning pipeline, driven by a fine-tuning "
        "config rather than a separate selection command; selection stops being a "
        "separate CLI over a separate model load",
    ),
    (
        "cli.fine_tuning_select.num_samples",
        MERGE,
        "--num_samples",
        "absorbed into the integrated fine-tuning pipeline, driven by a fine-tuning "
        "config rather than a separate selection command; selection stops being a "
        "separate CLI over a separate model load",
    ),
    (
        "cli.fine_tuning_select.subselect",
        MERGE,
        "--subselect",
        "absorbed into the integrated fine-tuning pipeline, driven by a fine-tuning "
        "config rather than a separate selection command; selection stops being a "
        "separate CLI over a separate model load",
    ),
    (
        "cli.fine_tuning_select.model",
        MERGE,
        "--model",
        "absorbed into the integrated fine-tuning pipeline, driven by a fine-tuning "
        "config rather than a separate selection command; selection stops being a "
        "separate CLI over a separate model load",
    ),
    (
        "cli.fine_tuning_select.output",
        MERGE,
        "--output",
        "absorbed into the integrated fine-tuning pipeline, driven by a fine-tuning "
        "config rather than a separate selection command; selection stops being a "
        "separate CLI over a separate model load",
    ),
    (
        "cli.fine_tuning_select.descriptors",
        MERGE,
        "--descriptors",
        "absorbed into the integrated fine-tuning pipeline, driven by a fine-tuning "
        "config rather than a separate selection command; selection stops being a "
        "separate CLI over a separate model load",
    ),
    (
        "cli.fine_tuning_select.device",
        MERGE,
        "--device",
        "absorbed into the integrated fine-tuning pipeline, driven by a fine-tuning "
        "config rather than a separate selection command; selection stops being a "
        "separate CLI over a separate model load",
    ),
    (
        "cli.fine_tuning_select.head_pt",
        MERGE,
        "--head_pt",
        "absorbed into the integrated fine-tuning pipeline, driven by a fine-tuning "
        "config rather than a separate selection command; selection stops being a "
        "separate CLI over a separate model load",
    ),
    (
        "cli.fine_tuning_select.head_ft",
        MERGE,
        "--head_ft",
        "absorbed into the integrated fine-tuning pipeline, driven by a fine-tuning "
        "config rather than a separate selection command; selection stops being a "
        "separate CLI over a separate model load",
    ),
    (
        "cli.fine_tuning_select.filtering_type",
        MERGE,
        "--filtering_type",
        "absorbed into the integrated fine-tuning pipeline, driven by a fine-tuning "
        "config rather than a separate selection command; selection stops being a "
        "separate CLI over a separate model load",
    ),
    (
        "cli.fine_tuning_select.weight_ft",
        MERGE,
        "--weight_ft",
        "absorbed into the integrated fine-tuning pipeline, driven by a fine-tuning "
        "config rather than a separate selection command; selection stops being a "
        "separate CLI over a separate model load",
    ),
    (
        "cli.fine_tuning_select.weight_pt",
        MERGE,
        "--weight_pt",
        "absorbed into the integrated fine-tuning pipeline, driven by a fine-tuning "
        "config rather than a separate selection command; selection stops being a "
        "separate CLI over a separate model load",
    ),
    (
        "cli.fine_tuning_select.filter_atomic_numbers_pt",
        MERGE,
        "--filter_atomic_numbers_pt",
        "absorbed into the integrated fine-tuning pipeline, driven by a fine-tuning "
        "config rather than a separate selection command; selection stops being a "
        "separate CLI over a separate model load",
    ),
    (
        "cli.fine_tuning_select.allow_random_padding",
        MERGE,
        "--disallow_random_padding",
        "absorbed into the integrated fine-tuning pipeline, driven by a fine-tuning "
        "config rather than a separate selection command; selection stops being a "
        "separate CLI over a separate model load",
    ),
    (
        "cli.fine_tuning_select.seed",
        MERGE,
        "--seed",
        "absorbed into the integrated fine-tuning pipeline, driven by a fine-tuning "
        "config rather than a separate selection command; selection stops being a "
        "separate CLI over a separate model load",
    ),
    (
        "cli.fine_tuning_select.default_dtype",
        MERGE,
        "--default_dtype",
        "it becomes PrecisionConfig",
    ),
    (
        "cli.fine_tuning_select.config",
        MERGE,
        "--config",
        "it becomes the v1 config system",
    ),
    (
        "cli.plot_train.start_swa",
        MERGE,
        "--start_stage_two / --start_swa",
        "stage boundaries are read from the run's per-stage schedule metadata; once "
        "stages are arbitrary a single 'stage two' marker no longer applies. "
        "Carries both --start_stage_two and the legacy --start_swa spelling",
    ),
    (
        "cli.active_learning_md.config",
        DROP,
        "--config",
        "out of v1.0 scope: an ASE MD loop a user writes in ~30 lines over the "
        "calculator, at the cost of MACE owning thermostat, timestep and trajectory "
        "I/O. What MACE must guarantee is the committee variance in calculate, "
        "which stays",
    ),
    (
        "cli.active_learning_md.config_index",
        DROP,
        "--config_index",
        "out of v1.0 scope: an ASE MD loop a user writes in ~30 lines over the "
        "calculator, at the cost of MACE owning thermostat, timestep and trajectory "
        "I/O. What MACE must guarantee is the committee variance in calculate, "
        "which stays",
    ),
    (
        "cli.active_learning_md.error_threshold",
        DROP,
        "--error_threshold",
        "out of v1.0 scope: an ASE MD loop a user writes in ~30 lines over the "
        "calculator, at the cost of MACE owning thermostat, timestep and trajectory "
        "I/O. What MACE must guarantee is the committee variance in calculate, "
        "which stays",
    ),
    (
        "cli.active_learning_md.temperature_K",
        DROP,
        "--temperature_K",
        "out of v1.0 scope: an ASE MD loop a user writes in ~30 lines over the "
        "calculator, at the cost of MACE owning thermostat, timestep and trajectory "
        "I/O. What MACE must guarantee is the committee variance in calculate, "
        "which stays",
    ),
    (
        "cli.active_learning_md.friction",
        DROP,
        "--friction",
        "out of v1.0 scope: an ASE MD loop a user writes in ~30 lines over the "
        "calculator, at the cost of MACE owning thermostat, timestep and trajectory "
        "I/O. What MACE must guarantee is the committee variance in calculate, "
        "which stays",
    ),
    (
        "cli.active_learning_md.timestep",
        DROP,
        "--timestep",
        "out of v1.0 scope: an ASE MD loop a user writes in ~30 lines over the "
        "calculator, at the cost of MACE owning thermostat, timestep and trajectory "
        "I/O. What MACE must guarantee is the committee variance in calculate, "
        "which stays",
    ),
    (
        "cli.active_learning_md.nsteps",
        DROP,
        "--nsteps",
        "out of v1.0 scope: an ASE MD loop a user writes in ~30 lines over the "
        "calculator, at the cost of MACE owning thermostat, timestep and trajectory "
        "I/O. What MACE must guarantee is the committee variance in calculate, "
        "which stays",
    ),
    (
        "cli.active_learning_md.nprint",
        DROP,
        "--nprint",
        "out of v1.0 scope: an ASE MD loop a user writes in ~30 lines over the "
        "calculator, at the cost of MACE owning thermostat, timestep and trajectory "
        "I/O. What MACE must guarantee is the committee variance in calculate, "
        "which stays",
    ),
    (
        "cli.active_learning_md.nsave",
        DROP,
        "--nsave",
        "out of v1.0 scope: an ASE MD loop a user writes in ~30 lines over the "
        "calculator, at the cost of MACE owning thermostat, timestep and trajectory "
        "I/O. What MACE must guarantee is the committee variance in calculate, "
        "which stays",
    ),
    (
        "cli.active_learning_md.ncheckerror",
        DROP,
        "--ncheckerror",
        "out of v1.0 scope: an ASE MD loop a user writes in ~30 lines over the "
        "calculator, at the cost of MACE owning thermostat, timestep and trajectory "
        "I/O. What MACE must guarantee is the committee variance in calculate, "
        "which stays",
    ),
    (
        "cli.active_learning_md.model",
        DROP,
        "--model",
        "out of v1.0 scope: an ASE MD loop a user writes in ~30 lines over the "
        "calculator, at the cost of MACE owning thermostat, timestep and trajectory "
        "I/O. What MACE must guarantee is the committee variance in calculate, "
        "which stays",
    ),
    (
        "cli.active_learning_md.output",
        DROP,
        "--output",
        "out of v1.0 scope: an ASE MD loop a user writes in ~30 lines over the "
        "calculator, at the cost of MACE owning thermostat, timestep and trajectory "
        "I/O. What MACE must guarantee is the committee variance in calculate, "
        "which stays",
    ),
    (
        "cli.active_learning_md.device",
        DROP,
        "--device",
        "out of v1.0 scope: an ASE MD loop a user writes in ~30 lines over the "
        "calculator, at the cost of MACE owning thermostat, timestep and trajectory "
        "I/O. What MACE must guarantee is the committee variance in calculate, "
        "which stays",
    ),
    (
        "cli.active_learning_md.default_dtype",
        DROP,
        "--default_dtype",
        "out of v1.0 scope: an ASE MD loop a user writes in ~30 lines over the "
        "calculator, at the cost of MACE owning thermostat, timestep and trajectory "
        "I/O. What MACE must guarantee is the committee variance in calculate, "
        "which stays",
    ),
    (
        "cli.active_learning_md.compute_stress",
        DROP,
        "--compute_stress",
        "out of v1.0 scope: an ASE MD loop a user writes in ~30 lines over the "
        "calculator, at the cost of MACE owning thermostat, timestep and trajectory "
        "I/O. What MACE must guarantee is the committee variance in calculate, "
        "which stays",
    ),
    (
        "cli.active_learning_md.info_prefix",
        DROP,
        "--info_prefix",
        "out of v1.0 scope: an ASE MD loop a user writes in ~30 lines over the "
        "calculator, at the cost of MACE owning thermostat, timestep and trajectory "
        "I/O. What MACE must guarantee is the committee variance in calculate, "
        "which stays",
    ),
    (
        "cli.polar_density_cube.default_dtype",
        MERGE,
        "--default_dtype",
        "it becomes PrecisionConfig",
    ),
    (
        "cli.create_lammps_model.dtype",
        MERGE,
        "--dtype",
        "it becomes PrecisionConfig of the export bundle",
    ),
    (
        "cli.create_lammps_model.format",
        MERGE,
        "--format",
        "v1 exports the MLIAP bundle only; the default TorchScript format is "
        "dropped with jit.script, so the choice collapses to one and the flag with "
        "it",
    ),
    (
        "cli.convert_e3nn_cueq.input_model",
        DROP,
        "input_model",
        "v1 weights are canonical and backend dispatch is automatic, so there is "
        "nothing left to convert between backends (positional)",
    ),
    (
        "cli.convert_e3nn_cueq.output_model",
        DROP,
        "--output_model",
        "v1 weights are canonical and backend dispatch is automatic, so there is "
        "nothing left to convert between backends",
    ),
    (
        "cli.convert_e3nn_cueq.device",
        DROP,
        "--device",
        "v1 weights are canonical and backend dispatch is automatic, so there is "
        "nothing left to convert between backends",
    ),
    (
        "cli.convert_e3nn_cueq.return_model",
        DROP,
        "--return_model",
        "v1 weights are canonical and backend dispatch is automatic, so there is "
        "nothing left to convert between backends (library flag: return the "
        "converted model instead of writing it)",
    ),
    (
        "cli.convert_cueq_e3nn.input_model",
        DROP,
        "input_model",
        "v1 weights are canonical and backend dispatch is automatic, so there is "
        "nothing left to convert between backends (positional)",
    ),
    (
        "cli.convert_cueq_e3nn.output_model",
        DROP,
        "--output_model",
        "v1 weights are canonical and backend dispatch is automatic, so there is "
        "nothing left to convert between backends",
    ),
    (
        "cli.convert_cueq_e3nn.device",
        DROP,
        "--device",
        "v1 weights are canonical and backend dispatch is automatic, so there is "
        "nothing left to convert between backends",
    ),
    (
        "cli.convert_cueq_e3nn.return_model",
        DROP,
        "--return_model",
        "v1 weights are canonical and backend dispatch is automatic, so there is "
        "nothing left to convert between backends (library flag: return the "
        "converted model instead of writing it)",
    ),
    (
        "cli.convert_e3nn_oeq.input_model",
        DROP,
        "input_model",
        "v1 weights are canonical and backend dispatch is automatic, so there is "
        "nothing left to convert between backends (positional)",
    ),
    (
        "cli.convert_e3nn_oeq.output_model",
        DROP,
        "--output_model",
        "v1 weights are canonical and backend dispatch is automatic, so there is "
        "nothing left to convert between backends",
    ),
    (
        "cli.convert_e3nn_oeq.device",
        DROP,
        "--device",
        "v1 weights are canonical and backend dispatch is automatic, so there is "
        "nothing left to convert between backends",
    ),
    (
        "cli.convert_e3nn_oeq.return_model",
        DROP,
        "--return_model",
        "v1 weights are canonical and backend dispatch is automatic, so there is "
        "nothing left to convert between backends (library flag: return the "
        "converted model instead of writing it)",
    ),
    (
        "cli.convert_oeq_e3nn.input_model",
        DROP,
        "input_model",
        "v1 weights are canonical and backend dispatch is automatic, so there is "
        "nothing left to convert between backends (positional)",
    ),
    (
        "cli.convert_oeq_e3nn.output_model",
        DROP,
        "--output_model",
        "v1 weights are canonical and backend dispatch is automatic, so there is "
        "nothing left to convert between backends",
    ),
    (
        "cli.convert_oeq_e3nn.device",
        DROP,
        "--device",
        "v1 weights are canonical and backend dispatch is automatic, so there is "
        "nothing left to convert between backends",
    ),
    (
        "cli.convert_oeq_e3nn.return_model",
        DROP,
        "--return_model",
        "v1 weights are canonical and backend dispatch is automatic, so there is "
        "nothing left to convert between backends (library flag: return the "
        "converted model instead of writing it)",
    ),
    (
        "cli.convert_e3nn_hybrid.input_model",
        DROP,
        "input_model",
        "v1 weights are canonical and backend dispatch is automatic, so there is "
        "nothing left to convert between backends (positional)",
    ),
    (
        "cli.convert_e3nn_hybrid.output_model",
        DROP,
        "--output_model",
        "v1 weights are canonical and backend dispatch is automatic, so there is "
        "nothing left to convert between backends",
    ),
    (
        "cli.convert_e3nn_hybrid.device",
        DROP,
        "--device",
        "v1 weights are canonical and backend dispatch is automatic, so there is "
        "nothing left to convert between backends",
    ),
    (
        "cli.convert_e3nn_hybrid.return_model",
        DROP,
        "--return_model",
        "v1 weights are canonical and backend dispatch is automatic, so there is "
        "nothing left to convert between backends (library flag: return the "
        "converted model instead of writing it)",
    ),
    (
        "model.MACE",
        MERGE,
        "MACE",
        "BaseMACE + a declared energy output; the class as a class disappears",
    ),
    (
        "model.ScaleShiftMACE",
        MERGE,
        "ScaleShiftMACE",
        "the default energy model becomes the default configuration; BaseMACE + a "
        "declared energy output; the class as a class disappears",
    ),
    (
        "model.AtomicDipolesMACE",
        MERGE,
        "AtomicDipolesMACE",
        "it becomes the dipole observable",
    ),
    (
        "model.AtomicDielectricMACE",
        MERGE,
        "AtomicDielectricMACE",
        "dipole + polarizability observables. Note this is the MACE-MDP foundation "
        "architecture, so it needs a converter as well as a reimplementation",
    ),
    (
        "model.EnergyDipolesMACE",
        MERGE,
        "EnergyDipolesMACE",
        "it becomes energy + dipole observables",
    ),
    (
        "reg.RealAgnosticAttResidualInteractionBlock",
        DROP,
        "RealAgnosticAttResidualInteractionBlock",
        "unlike the Density blocks it appears in no finetuning_utils branch and no "
        "converter, only in the registry and the parser choices: a research variant "
        "with no published model, no test and no owner",
    ),
    (
        "reg.LinearDipoleReadoutBlock",
        MERGE,
        "LinearDipoleReadoutBlock",
        "an observable head declared in the output spec",
    ),
    (
        "reg.NonLinearDipoleReadoutBlock",
        MERGE,
        "NonLinearDipoleReadoutBlock",
        "an observable head declared in the output spec",
    ),
    (
        "reg.rms_dipoles_scaling",
        MERGE,
        "rms_dipoles_scaling",
        "it becomes observable normalization",
    ),
    (
        "block.RealAgnosticInteractionBlock",
        MERGE,
        "RealAgnosticInteractionBlock",
        "one of five interaction variants that collapse into a configured "
        "convolution",
    ),
    (
        "block.RealAgnosticDensityInteractionBlock",
        MERGE,
        "RealAgnosticDensityInteractionBlock",
        "one of five interaction variants that collapse into a configured "
        "convolution",
    ),
    (
        "block.RealAgnosticDensityResidualInteractionBlock",
        MERGE,
        "RealAgnosticDensityResidualInteractionBlock",
        "one of five interaction variants that collapse into a configured "
        "convolution",
    ),
    (
        "block.RealAgnosticAttResidualInteractionBlock",
        MERGE,
        "RealAgnosticAttResidualInteractionBlock",
        "one of five interaction variants that collapse into a configured "
        "convolution",
    ),
    (
        "block.MagneticRealAgnosticResidueSpinOrbitCoupledDensityInteractionBlock",
        MERGE,
        "MagneticRealAgnosticResidueSpinOrbitCoupledDensityInteractionBlock",
        "the residual spin-orbit-coupled variant collapses into a configured "
        "convolution like the other interaction blocks",
    ),
    (
        "contraction.EmptyParam",
        MERGE,
        "EmptyParam",
        "weight bookkeeping for an unreachable order, not a feature",
    ),
    (
        "transform.Random3DRotation",
        MERGE,
        "Random3DRotation",
        "the --data_aug_magmom transform, which travels with the flag",
    ),
    (
        "calc.class.MACELammpsConfig",
        MERGE,
        "MACELammpsConfig",
        "the ML-IAP wrapper's own config object, an implementation detail",
    ),
    (
        "loss.WeightedEnergyForcesLoss",
        MERGE,
        "WeightedEnergyForcesLoss",
        "a composition preset (the ef/weighted schemes)",
    ),
    (
        "loss.WeightedForcesLoss",
        MERGE,
        "WeightedForcesLoss",
        "it becomes a composition preset (the forces_only scheme)",
    ),
    (
        "loss.WeightedEnergyForcesStressLoss",
        MERGE,
        "WeightedEnergyForcesStressLoss",
        "it becomes a composition preset (the stress scheme)",
    ),
    (
        "loss.WeightedHuberEnergyForcesStressLoss",
        MERGE,
        "WeightedHuberEnergyForcesStressLoss",
        "it becomes a composition preset (the huber scheme)",
    ),
    (
        "loss.UniversalLoss",
        MERGE,
        "UniversalLoss",
        "it becomes a composition preset (the universal scheme)",
    ),
    (
        "loss.WeightedEnergyForcesVirialsLoss",
        MERGE,
        "WeightedEnergyForcesVirialsLoss",
        "it becomes a composition preset (the virials scheme)",
    ),
    (
        "loss.DipoleSingleLoss",
        MERGE,
        "DipoleSingleLoss",
        "it becomes a composition preset (the dipole scheme)",
    ),
    (
        "loss.DipolePolarLoss",
        MERGE,
        "DipolePolarLoss",
        "a composition preset (the dipole_polar scheme)",
    ),
    (
        "loss.WeightedEnergyForcesDipoleLoss",
        MERGE,
        "WeightedEnergyForcesDipoleLoss",
        "a composition preset (the energy_forces_dipole scheme)",
    ),
    (
        "loss.WeightedEnergyForcesL1L2Loss",
        MERGE,
        "WeightedEnergyForcesL1L2Loss",
        "a composition preset (the l1l2energyforces scheme)",
    ),
    (
        "calc.param.default_dtype",
        MERGE,
        "default_dtype",
        "it becomes PrecisionConfig",
    ),
    (
        "calc.param.model_type",
        MERGE,
        "model_type",
        "auto-detected from model metadata; asking the user to name the model "
        "family is asking them to get it wrong",
    ),
    (
        "calc.param.enable_cueq",
        MERGE,
        "enable_cueq",
        "it becomes backend dispatch config",
    ),
    (
        "calc.param.enable_oeq",
        MERGE,
        "enable_oeq",
        "it becomes backend dispatch config",
    ),
    (
        "calc.param.model_path",
        DROP,
        "model_path",
        "deprecated singular alias for model_paths; it warns and forwards, and "
        "refuses when both are given",
    ),
    (
        "calc.export.LAMMPS_MACE",
        DROP,
        "LAMMPS_MACE",
        "the TorchScript wrapper dies with the TorchScript export format; the MLIAP "
        "path replaces it",
    ),
    (
        "calc.export.mace_anicc",
        DROP,
        "mace_anicc",
        "a 2023 organic-chemistry model superseded by MACE-OFF, and the only loader "
        "with a divergent signature (model_path instead of model): an API exception "
        "for an obsolete artifact. Its tracked checkpoint "
        "mace/calculators/foundations_models/ani500k_large_CC.model goes with it; "
        'the release notes say "use MACE-OFF"',
    ),
    (
        "out.model.displacement",
        MERGE,
        "displacement",
        "the strain displacement is internal machinery of the derivative engine, "
        "not a user-facing output; v1 does not return it",
    ),
    (
        "out.model.charges_history",
        MERGE,
        "charges_history",
        "the per-iteration trace of the fixed-point solve; a solver diagnostic, so "
        "it belongs to the solver-dispatch layer rather than the model's outputs",
    ),
    (
        "out.model.scf_steps",
        MERGE,
        "scf_steps",
        "SCF solver diagnostics belong to the model-transform hook, not to the "
        "model's output contract",
    ),
    (
        "out.model.scf_energy_history",
        MERGE,
        "scf_energy_history",
        "SCF solver diagnostics belong to the model-transform hook, not to the "
        "model's output contract",
    ),
    (
        "out.calc.LES_alphas",
        MERGE,
        "LES_alphas",
        "the calculator renames the model's latent_alphas; v1 exposes one name for "
        "one quantity, and a per-surface rename is exactly the kind of thing that "
        "makes a key ungreppable",
    ),
    (
        "out.calc.LES_kappas",
        MERGE,
        "LES_kappas",
        "from latent_kappas; the calculator renames the model's latent_alphas; v1 "
        "exposes one name for one quantity, and a per-surface rename is exactly the "
        "kind of thing that makes a key ungreppable",
    ),
    (
        "out.calc.MACE_magmoms",
        MERGE,
        "MACE_magmoms",
        "the magnetic calculator's spelling of the magnetic-moment observable, also "
        "written back into atoms.arrays; the calculator renames the model's "
        "latent_alphas; v1 exposes one name for one quantity, and a per-surface "
        "rename is exactly the kind of thing that makes a key ungreppable",
    ),
    (
        "env.MACE_USE_CUEQ_CG",
        DROP,
        "MACE_USE_CUEQ_CG",
        "the variable goes, not the capability: an environment variable that "
        "silently changes model numerics is unreproducible and never lands in the "
        "run metadata; it is what makes machine-to-machine differences "
        "unexplainable. The CG source becomes a backend decision recorded in the "
        "resolved config",
    ),
    (
        "stdenv.TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD",
        DROP,
        "TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD",
        "v1 checkpoints are neutral-format (safetensors + manifest), so nothing "
        "needs the unsafe-pickle escape hatch; the legacy loader keeps it until the "
        "converter is the only reader",
    ),
    (
        "kwarg.default_dtype",
        MERGE,
        "default_dtype=",
        "it becomes PrecisionConfig",
    ),
    (
        "kwarg.model_path",
        DROP,
        "mace_anicc(model_path=…)",
        "goes with mace_anicc itself; the signature exception is part of why",
    ),
    (
        "lammps.torchscript_wrapper",
        DROP,
        "the LAMMPS_MACE TorchScript wrapper and the -lammps.pt artifact",
        "v1 blocks are born without @compile_mode, so scripting can never apply to "
        "them; the MLIAP bundle is the one supported artifact",
    ),
    (
        "lammps.compiled_side_artifact",
        DROP,
        "the _compiled.model / _stagetwo_compiled.model side artifacts",
        "a deliberate, recorded removal: v1 checkpoints are neutral-format only and "
        "deployment artifacts come solely from the v1 export command",
    ),
    (
        "pkg.vendored_torch_geometric",
        DROP,
        "the vendored mace.tools.torch_geometric copy",
        "v1 collates without torch_geometric; the vendored copy is excluded from "
        "lint and mypy today, which is the clearest sign it is not maintained code. "
        "Complication: mace/data/augmentation.py imports the *real* package while "
        "the rest of the tree imports the vendored one, and the [magnetic] extra "
        "declares external torch-geometric, so both must go at once",
    ),
    (
        "pkg.compile_utils",
        MERGE,
        "prepare / simplify_if_compile",
        "v1 is compile-first, so the retrofit mechanism has nothing to retrofit",
    ),
    (
        "pkg.public_import_surface",
        DROP,
        "the mace.* public import surface",
        "a deliberate break: v1 defines a new public API, and the release notes "
        "document the old to new equivalences rather than aliasing them",
    ),
    (
        "pkg.anicc_checkpoint",
        DROP,
        "the bundled MACE-ANI-CC checkpoint",
        "goes with mace_anicc; v1 fetches every artifact through the model registry "
        "rather than bundling one",
    ),
    (
        "pkg.lr_param_groups",
        MERGE,
        "the explicit optimizer parameter groups behind --lr_params_factors",
        "typed per-param-group fields of the per-stage optimizer config",
    ),
    (
        "pkg.augmentation",
        MERGE,
        "Random3DRotation",
        "a registered training-data transform, not a model flag",
    ),
    (
        "pkg.first_block_coercion",
        DROP,
        "the silent rewrite of an unsupported --interaction_first to RealAgnosticInteractionBlock",
        "a config value the tool overwrites without a word is worse than a rejected "
        "one: the run trains a different architecture than the user asked for and "
        "nothing says so. v1 fails the combination in config validation",
    ),
    (
        "fm.mace_anicc",
        DROP,
        "MACE-ANI-CC",
        "superseded by MACE-OFF, and the only artifact bundled inside the wheel; "
        'the release notes say "use MACE-OFF"',
    ),
    (
        "ep.convert_e3nn_oeq",
        DROP,
        "the mace.cli.convert_e3nn_oeq command",
        "v1 weights are canonical and backend dispatch is automatic, so there is "
        "nothing left to convert between backends",
    ),
    (
        "ep.convert_oeq_e3nn",
        DROP,
        "the mace.cli.convert_oeq_e3nn command",
        "the reverse direction; v1 weights are canonical and backend dispatch is "
        "automatic, so there is nothing left to convert between backends",
    ),
    (
        "ep.convert_e3nn_hybrid",
        DROP,
        "the mace.cli.convert_e3nn_hybrid command",
        "the mixed e3nn/cueq layout it produces has no counterpart once backend "
        "dispatch is automatic and v1 weights are canonical",
    ),
)
