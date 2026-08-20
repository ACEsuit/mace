# MACE functionality inventory

Every enumerable feature of the tree, with a **KEEP / MERGE / DROP** disposition and the test that
pins its behaviour. It exists so that a feature cannot be dropped in a rewrite without anyone
noticing: the loss is otherwise invisible until a user reports it. A feature the next major version
will not carry gets a `DROP` row with a reason; the *absence* of a row is a bug in this file, not a
decision.

`check_inventory.py`, next to this file, re-derives every source surface by AST and fails when the
two disagree. Run it from anywhere:

```bash
python3 tests/golden/check_inventory.py     # prints "all sources covered"
```

**Row schema** (every table below, in this order):

| column | meaning |
|---|---|
| id | stable identifier, namespaced by source (`train.`, `cli.`, `model.`, `out.calc.`, …). The checker keys on this. |
| feature | the option strings / class / key, plus any gloss |
| source | `file:line` of the declaration |
| disposition | `KEEP` · `MERGE` · `DROP`, optionally followed by ` — reason`. `DROP` **must** carry one |
| pinned by | the test that protects the behaviour, or `⚠️ gap`. Must **resolve**: see below |

**A pin has to resolve, and it has to discriminate.** "Not empty" is not a rule: it accepts the
literal string `TODO`, and the gate then finishes with *all sources covered*. So the `pinned by` cell
must open with one of three things, and every backticked path it names anywhere must exist on disk —
with its `::node_id`, if it carries one, actually declared there. A pin naming a test that was
renamed or never written is worse than a gap marker, because it reads as coverage.

Resolving is not sufficient either, and the two ways it can be hollow are worth naming because both
were live in this file:

* **a path that is true of everything.** `tests/` and `tests/unit` exist, so they pass "must
  resolve" while telling a reader nothing about which behaviour is protected. Directory pins stay
  legitimate — `tests/extensions/magnetic` *is* that family's coverage — so the rule is a floor, not
  a ban: a pinning directory sits at least two levels under `tests/`, which admits every per-family
  directory and rejects exactly the tier-level ones.
* **a file every row in the group could name.** Twelve `marker.*` rows pinned `tests/conftest.py`,
  and the only thing asserted was that the file exists; `marker.anything` would have passed. What
  enforces a capability marker is having an entry in `CAPABILITY_PROBES` — the dict
  `pytest_runtest_setup` iterates, so a marker missing from it is registered, usable, and silently
  outside the `MACE_REQUIRE_CAPS` contract. Each capability row therefore pins its *own* entry,
  `tests/conftest.py::CAPABILITY_PROBES[<name>]`, and the three cost markers must not claim one.

| pin opens with | means |
|---|---|
| a gap marker, `⚠️ gap (…)` | nothing pins this yet, and the cell says where the test belongs |
| a backticked path under `tests/` | a test that runs today: a file, a `::test_name`, a `::TABLE[key]` entry, or a directory at least two levels deep |
| one of two named CI jobs | `the suite itself` and `the lint job itself`, for the two `setup.cfg` extras, where the only thing that can fail is a job installing them. Allowed by name, one entry each, with a written reason |

There is deliberately no fourth form. A pin used to be allowed to name a *planned* test by ticket id,
which meant the cell recorded an intention nobody could resolve: a plan that slipped read exactly
like coverage that existed. Every pin now names something that runs, or admits that nothing does.
A pin naming a *test id* rather than a ticket is also what makes the checker able to verify it, which
is why this file can claim the tests exist rather than asking you to trust it.

**Disposition vocabulary.** `KEEP` = the functionality must survive, possibly renamed or reshaped
(usually as a config field — the flag *as a flag* disappears into the config system). `MERGE` =
subsumed by a named, more general mechanism. `DROP` = intentionally removed; the reason is what the
release notes' migration guide is written from. There is no fourth value: `REVIEW` is not a
disposition the gate accepts, so an undecided row fails the build rather than sitting in the file.

**Maintenance rule.** When a source changes, the checker names the rows to add or delete — re-run it
rather than re-reading the diff. A PR that adds a flag, a class, a registry entry or an output key
adds its row in the same change; that is why the checker runs in the required PR checks rather than
in a nightly.

**Flags are keyed on the argparse dest, never on the option string.** An option string is a spelling;
a dest is a knob. `--swa_lr` and `--stage_two_lr` are one dest, so one row and one disposition, with
both spellings in the feature cell. A dest is counted once **per parser that declares it**, because
defaults and help text differ per parser: 22 dests appear in both training parsers and get a row in
each, and `--device` appears in four CLIs as four knobs. That gives 184 + 26 + 111 = **321 flag rows
over 15 parsers**, against 188 + 74 distinct dests — the deduplicated figures orient, they never
count rows.

**Why per-dest and not per-group.** A group row ("the wandb flags: KEEP") carries its members only by
implication, and nothing fails when one of them is never mentioned. Eight knobs went through exactly
that hole in the previous, prose-grouped version of this file: `only_cueq`, `MLP_irreps`,
`valid_batch_size`, `statistics_file`, `test_file`, and the schedulefree trio (`beta1_schedulefree`,
`beta2_schedulefree`, `warmup_steps_schedulefree`) — the entire tuning surface of an optimizer a
config schema otherwise covers by naming the optimizer alone. Five were folded into a cell listing
several flags under one shared disposition; one was stated in prose between two tables. Neither shape
is a disposition a checker can read.

**What the gate enforces.** Seventeen set comparisons — entry points, the two training parsers, the
thirteen `mace/cli` parsers, the `--model` choices, model classes, registries, losses, calculator
params, calculator exports, extras, the three output-key surfaces, the `MACE_*` environment
variables, the pytest markers, and the default property keys — plus, on every row: a valid
disposition, a reason on every `DROP`, a resolving pin or an explicit `⚠️ gap` on every
`KEEP`/`MERGE`, and no duplicate ids. Four conditions fail a dest: no row, an empty disposition, a
`REVIEW` disposition, and a row for a dest the source no longer declares — so a renamed flag cannot
leave a stale row behind claiming coverage.

**The `⚠️ gap` column is the open work.** Each gap is either a test still to be written or a
conscious downgrade, and the cell says which and where the test would go. The authoritative count is
the checker's `tally:` line, never a number written in prose. Read that count as rows, not as tasks:
because the key is the dest, one missing test marks every dest it would have covered — the sixteen
dests of `mace_finetuning_select` carry one gap between them, not sixteen — so the rows cluster into
far fewer distinct pieces of work than the tally suggests.
---

## 1. CLI entry points (12)

Twelve console scripts in `setup.cfg`. Three further CLIs live in `mace/cli/` with a `main()` and an
argparser but **no entry point** (`convert_e3nn_oeq`, `convert_oeq_e3nn`, `convert_e3nn_hybrid`);
they have no row here because there is nothing registered to keep or drop, but their flags are
inventoried in §5 and they go when `mace_e3nn_cueq` goes.

| id | feature | source | disposition | pinned by |
|---|---|---|---|---|
| `ep.mace_run_train` | `mace.cli.run_train:main` | `setup.cfg` | KEEP — becomes `mace train` | `tests/workflows/test_cli_contracts.py::test_training_reduces_the_validation_loss_and_writes_a_model` |
| `ep.mace_eval_configs` | `mace.cli.eval_configs:main` | `setup.cfg` | KEEP — the capability survives, but the CLI shrinks to a thin `mace eval` adapter over the ASE calculator (read configs → calculator → write XYZ); most of its ~400 lines is argparse duplicating what the calculator already does, and one numerical path is cheaper to pin than two | `tests/workflows/test_cli_contracts.py::test_eval_configs_reproduces_the_committed_anchor_reference` |
| `ep.mace_prepare_data` | `mace.cli.preprocess_data:main` | `setup.cfg` | KEEP — becomes `mace data prepare` | `tests/workflows/test_preprocess.py` |
| `ep.mace_create_lammps_model` | `mace.cli.create_lammps_model:main` | `setup.cfg` | KEEP — becomes `mace export lammps` | `tests/integrations/lammps/test_export_golden.py::test_the_exported_artifact_reproduces_the_committed_numbers` |
| `ep.mace_select_head` | `mace.cli.select_head:main` | `setup.cfg` | KEEP — becomes `mace model select-head` | `tests/workflows/test_cli_contracts.py::test_select_head_lists_the_heads_of_a_multihead_model` + `tests/workflows/test_cli_contracts.py::test_select_head_and_the_multihead_model_agree_on_the_selected_head` |
| `ep.mace_plot_train` | `mace.cli.plot_train:main` | `setup.cfg` | KEEP — reduced: the basic plot subcommand stays (per-head loss curves, cheap over a structured log); `--plot_interaction_e` goes (§3.1) | `tests/workflows/test_plot_train.py::test_a_real_results_log_becomes_a_plot` |
| `ep.mace_polar_density_cube` | `mace.cli.polar_density_cube:main` | `setup.cfg` | KEEP — the electrostatics extra's density-cube tool | `tests/extensions/polar/test_polar_density_cube.py` (port cases) |
| `ep.mace_finetuning_select` | `mace.cli.fine_tuning_select:main` | `setup.cfg` | MERGE — absorbed into the integrated fine-tuning pipeline (`mace train` with a fine-tuning config) | `tests/workflows/test_finetuning_contracts.py::test_subselect_fps_uses_the_model_and_still_returns_the_requested_number` |
| `ep.mace_convert_device` | `mace.cli.convert_device:main` | `setup.cfg` | KEEP — becomes `mace model convert-device`; explicitly **not** one of the five weight converters, because it converts device/dtype and not backend layout | `tests/unit/test_scale_shift_dtype.py::test_the_convert_device_cli_preserves_the_buffers` |
| `ep.mace_e3nn_cueq` | `mace.cli.convert_e3nn_cueq:main` | `setup.cfg` | DROP — v1 weights are canonical and backend dispatch is automatic, so there is nothing left to convert between backends | `tests/golden/test_backend_parity_golden.py::test_converted_model_reproduces_the_committed_cpu_reference` |
| `ep.mace_cueq_to_e3nn` | `mace.cli.convert_cueq_e3nn:main` | `setup.cfg` | DROP — idem, the reverse direction | `tests/backends/backend_parity.py::test_bidirectional_conversion` |
| `ep.mace_active_learning_md` | `mace.cli.active_learning_md:main` | `setup.cfg` | DROP — out of v1.0 scope: an ASE MD loop a user writes in ~30 lines over the calculator, at the cost of MACE owning thermostat, timestep and trajectory I/O. The committee variance it consumes stays in `calculate` (§16); the release notes document the recipe | — |

## 2. `--model` choices (10)

The CLI-selectable model names. A separate set from the model classes of §6 on purpose: **two of the
ten name a class that exists nowhere in the tree.** `BOTNet` and `ScaleShiftBOTNet` reach only
`RuntimeError("... is deprecated, use MACE instead")` in `mace/tools/model_script_utils.py:374-378`.
v1's model enum does not carry them, so an unknown value fails through ordinary config validation
instead of a hand-written runtime raise.

| id | feature | source | disposition | pinned by |
|---|---|---|---|---|
| `choice.BOTNet` | `--model BOTNet` | `mace/tools/arg_parser.py:138` | DROP — no `BOTNet` class exists anywhere in the tree; the choice reaches only a deprecation raise. The release notes name MACE as the replacement | — |
| `choice.ScaleShiftBOTNet` | `--model ScaleShiftBOTNet` | `mace/tools/arg_parser.py:143` | DROP — identical shape: no class, only a deprecation raise | — |
| `choice.MACE` | `--model MACE` | `mace/tools/arg_parser.py:139` | MERGE — model composition is config-driven, not a class name | `tests/golden/anchors.py::ANCHORS[tiny_mace]` + `tests/golden/test_tiny_anchors.py::test_anchor_is_the_class_it_claims_to_be` |
| `choice.ScaleShiftMACE` | `--model ScaleShiftMACE` | `mace/tools/arg_parser.py:140` | MERGE — idem; the default energy model becomes the default configuration | `tests/golden/anchors.py::ANCHORS[tiny_scaleshift]` + `tests/golden/test_tiny_anchors.py::test_anchor_is_the_class_it_claims_to_be` |
| `choice.PolarMACE` | `--model PolarMACE` | `mace/tools/arg_parser.py:141` | MERGE — idem, selected by declaring the electrostatics observables | `tests/golden/test_polar_foundation.py::test_polar_foundation_reproduces_its_reference` |
| `choice.MACELES` | `--model MACELES` | `mace/tools/arg_parser.py:142` | MERGE — idem, selected by declaring the LES long-range term | `tests/golden/test_tiny_maceles.py::test_the_anchor_is_a_maceles_built_from_the_committed_yaml` |
| `choice.AtomicDipolesMACE` | `--model AtomicDipolesMACE` | `mace/tools/arg_parser.py:144` | MERGE — idem, selected by declaring the dipole observable | `tests/golden/test_tiny_dipoles.py::test_anchor_is_the_class_it_claims_to_be` |
| `choice.AtomicDielectricMACE` | `--model AtomicDielectricMACE` | `mace/tools/arg_parser.py:145` | MERGE — idem, dipole + polarizability observables | `tests/golden/test_mdp_foundation.py::test_mdp_foundation_reproduces_its_reference` |
| `choice.EnergyDipolesMACE` | `--model EnergyDipolesMACE` | `mace/tools/arg_parser.py:146` | MERGE — idem, energy + dipole observables | `tests/unit/test_models.py::test_energy_dipole_mace` |
| `choice.MagneticScaleShiftMACE` | `--model MagneticScaleShiftMACE` | `mace/tools/arg_parser.py:147` | MERGE — idem; the only magnetic entry in the choices | `tests/extensions/magnetic` + `tests/golden/test_tiny_magnetic.py::test_anchor_reproduces_its_reference` |

## 3. `mace_run_train` flags — 184 dests

One row per **dest** of `build_default_arg_parser` (`mace/tools/arg_parser.py`), which is what a
knob is; the option strings that spell it are in the feature cell. 184 dests carry 194 option
strings — the 10-string surplus is the `--swa_*`/`--stage_two_*` alias pairs.

### 3.0 Config file (1)

The one dest registered with `parser.add` instead of `add_argument`, and the only optional-dependency flag in the parser: without configargparse installed the option silently does not exist.

| id | feature | source | disposition | pinned by |
|---|---|---|---|---|
| `train.config` | `--config` | `mace/tools/arg_parser.py:23` | MERGE — the v1 config system makes this first-class (TOML/YAML/JSON, always available, resolved config saved as run metadata) | `tests/unit/test_arg_parser.py::test_yaml_config_values_are_applied` (the YAML path at parse level) |

### 3.1 Run and infrastructure (17)

Group default: KEEP as the `runtime` / `output` config sections. The four `*_dir` flags MERGE into one work-dir layout convention rather than four independent paths.

| id | feature | source | disposition | pinned by |
|---|---|---|---|---|
| `train.name` | `--name` | `mace/tools/arg_parser.py:34` | KEEP | `tests/workflows/test_cli_contracts.py::test_training_reduces_the_validation_loss_and_writes_a_model` |
| `train.seed` | `--seed` | `mace/tools/arg_parser.py:35` | KEEP | `tests/golden/test_train_step_gradient.py::test_the_step_reproduces_its_committed_gradient` |
| `train.work_dir` | `--work_dir` | `mace/tools/arg_parser.py:39` | KEEP | `tests/workflows/test_train_work_dir_and_workers.py::test_the_validation_indices_land_in_the_work_dir` |
| `train.log_dir` | `--log_dir` | `mace/tools/arg_parser.py:45` | MERGE — single work-dir layout convention | `tests/workflows/test_cli_contracts.py::test_training_reduces_the_validation_loss_and_writes_a_model` |
| `train.model_dir` | `--model_dir` | `mace/tools/arg_parser.py:48` | MERGE — idem | `tests/workflows/test_cli_contracts.py::test_training_reduces_the_validation_loss_and_writes_a_model` |
| `train.checkpoints_dir` | `--checkpoints_dir` | `mace/tools/arg_parser.py:51` | MERGE — idem | `tests/workflows/test_cli_contracts.py::test_restart_latest_continues_from_the_checkpoint_epoch` |
| `train.results_dir` | `--results_dir` | `mace/tools/arg_parser.py:57` | MERGE — idem | `tests/workflows/test_cli_contracts.py::test_lbfgs_takes_one_step_per_epoch_and_the_other_regime_one_per_batch` |
| `train.downloads_dir` | `--downloads_dir` | `mace/tools/arg_parser.py:60` | MERGE — XDG cache-dir convention | ⚠️ gap (cache-path contract) |
| `train.device` | `--device` | `mace/tools/arg_parser.py:65` | KEEP | `tests/workflows/test_cli_contracts.py::test_select_head_honours_output_file_and_target_device` |
| `train.default_dtype` | `--default_dtype` | `mace/tools/arg_parser.py:72` | MERGE — `PrecisionConfig` | `tests/workflows/test_cli_contracts.py::test_eval_at_float32_reproduces_float64_within_the_fp32_row` + `tests/golden/test_harness.py::test_tolerance_table_rows` |
| `train.distributed` | `--distributed` | `mace/tools/arg_parser.py:79` | KEEP | `tests/workflows/test_distributed.py` |
| `train.launcher` | `--launcher` | `mace/tools/arg_parser.py:85` | KEEP | `tests/workflows/test_distributed.py` |
| `train.log_level` | `--log_level` | `mace/tools/arg_parser.py:90` | KEEP | ⚠️ gap (trivial; conscious downgrade candidate) |
| `train.plot` | `--plot` | `mace/tools/arg_parser.py:93` | KEEP | ⚠️ gap (with the `mace_plot_train` smoke) |
| `train.plot_frequency` | `--plot_frequency` | `mace/tools/arg_parser.py:100` | KEEP | ⚠️ gap (idem) |
| `train.plot_interaction_e` | `--plot_interaction_e` | `mace/tools/arg_parser.py:107` | DROP — niche diagnostic that drags model introspection into the plotting path | — |
| `train.error_table` | `--error_table` | `mace/tools/arg_parser.py:114` | KEEP — the error-table types | `tests/workflows/test_cli_contracts.py::test_the_end_of_training_error_table_is_printed_and_parseable` |

### 3.2 Model architecture (26)

Group default: KEEP as the `model` config section; the defaults are pinned by the committed anchors.

| id | feature | source | disposition | pinned by |
|---|---|---|---|---|
| `train.model` | `--model` | `mace/tools/arg_parser.py:134` | MERGE — model composition is config-driven (`BaseMACE` + declared outputs), not a class name | `tests/golden/test_tiny_anchors.py::test_anchor_is_the_class_it_claims_to_be` |
| `train.r_max` | `--r_max` | `mace/tools/arg_parser.py:151` | KEEP | `tests/unit/test_radial.py::test_polynomial_cutoff_is_exactly_zero_at_and_beyond_r_max` |
| `train.radial_type` | `--radial_type` | `mace/tools/arg_parser.py:154` | KEEP — bessel / gaussian / chebyshev | `tests/unit/test_radial.py::test_bessel_basis_values` + `tests/unit/test_radial.py::test_gaussian_basis_values` + `tests/unit/test_radial.py::test_chebychev_basis_values` |
| `train.num_radial_basis` | `--num_radial_basis` | `mace/tools/arg_parser.py:161` | KEEP | `tests/unit/test_radial.py::test_bessel_shape_and_dtype_contract` |
| `train.num_cutoff_basis` | `--num_cutoff_basis` | `mace/tools/arg_parser.py:167` | KEEP | `tests/unit/test_radial.py::test_polynomial_cutoff_is_one_at_zero_for_every_p` |
| `train.pair_repulsion` | `--pair_repulsion` | `mace/tools/arg_parser.py:173` | KEEP — the ZBL short-range term | `tests/unit/test_radial.py::test_zbl_matches_the_published_formula` + `tests/golden/test_tiny_anchors.py::test_the_repulsion_term_is_scaled_in_one_class_and_raw_in_the_other` |
| `train.distance_transform` | `--distance_transform` | `mace/tools/arg_parser.py:179` | KEEP | `tests/unit/test_radial.py::test_the_basis_sees_the_transformed_lengths` + `tests/unit/test_radial.py::test_the_cutoff_is_computed_before_the_distance_transform` |
| `train.apply_cutoff` | `--apply_cutoff` | `mace/tools/arg_parser.py:185` | KEEP | `tests/unit/test_radial.py::test_apply_cutoff_true_returns_the_product_and_no_envelope` + `tests/unit/test_radial.py::test_apply_cutoff_false_defers_the_envelope_to_the_consumer` |
| `train.use_last_readout_only` | `--use_last_readout_only` | `mace/tools/arg_parser.py:191` | MERGE — readout policy: once you declare which layers read out, 'only the last' is configuration, not a boolean | ⚠️ gap (add a case to `tests/unit/test_models.py`) |
| `train.use_embedding_readout` | `--use_embedding_readout` | `mace/tools/arg_parser.py:197` | MERGE — idem ('also read the embedding layer') | ⚠️ gap (add a case to `tests/unit/test_models.py`) |
| `train.interaction` | `--interaction` | `mace/tools/arg_parser.py:203` | KEEP — see the registry rows in §7 for which classes survive | `tests/golden/test_tiny_anchors.py::test_anchor_reproduces_its_reference` |
| `train.interaction_first` | `--interaction_first` | `mace/tools/arg_parser.py:219` | KEEP | `tests/unit/test_models.py::test_non_linear_first_interaction_block_runs_and_is_equivariant` |
| `train.max_ell` | `--max_ell` | `mace/tools/arg_parser.py:234` | KEEP | `tests/golden/test_tiny_anchors.py::test_anchor_reproduces_its_reference` |
| `train.correlation` | `--correlation` | `mace/tools/arg_parser.py:237` | KEEP | `tests/golden/test_tiny_anchors.py::test_anchor_reproduces_its_reference` |
| `train.use_reduced_cg` | `--use_reduced_cg` | `mace/tools/arg_parser.py:240` | MERGE — a CG-representation choice the backend makes, not a modelling decision a user can judge, and it changes numerics; `convert_e3nn_hybrid.py` defaults it to `True`, so checkpoints carry it and the converter must read it rather than assume | `tests/golden/test_tiny_dipoles.py::test_the_committed_anchor_carries_the_plain_e3nn_basis` (pins the plain basis; the reduced path stays unpinned) |
| `train.use_so3` | `--use_so3` | `mace/tools/arg_parser.py:246` | DROP — a global parity-convention switch that doubles the irrep-handling surface in exactly the layer v1 rewrites; no published model sets it | — |
| `train.use_agnostic_product` | `--use_agnostic_product` | `mace/tools/arg_parser.py:252` | KEEP — MACE-Polar S/M/L set it, so it is foundation-model architecture, not a research knob | `tests/foundations/test_foundations.py::test_polar_extract_config_roundtrip` + `tests/golden/test_polar_foundation.py::test_the_reference_pins_the_electrostatics_and_not_only_the_energy` |
| `train.num_interactions` | `--num_interactions` | `mace/tools/arg_parser.py:258` | KEEP | `tests/golden/test_tiny_anchors.py::test_anchor_reproduces_its_reference` |
| `train.MLP_irreps` | `--MLP_irreps` | `mace/tools/arg_parser.py:261` | KEEP — the non-linear readout's hidden irreps | `tests/golden/test_tiny_anchors.py::test_anchor_reproduces_its_reference` |
| `train.radial_MLP` | `--radial_MLP` | `mace/tools/arg_parser.py:267` | KEEP | `tests/unit/test_radial.py::test_radial_mlp_structure_and_shapes` |
| `train.hidden_irreps` | `--hidden_irreps` | `mace/tools/arg_parser.py:273` | KEEP | `tests/golden/test_tiny_anchors.py::test_anchor_reproduces_its_reference` |
| `train.edge_irreps` | `--edge_irreps` | `mace/tools/arg_parser.py:279` | KEEP — MACE-Polar S/M/L set it (`128x0e` → `128x0e+128x1o+128x2e`) | `tests/foundations/test_foundations.py::test_polar_extract_config_roundtrip` + `tests/golden/test_polar_foundation.py::test_the_reference_pins_the_electrostatics_and_not_only_the_energy` |
| `train.use_edge_irreps_first` | `--use_edge_irreps_first` | `mace/tools/arg_parser.py:285` | KEEP — the first-layer variant of `edge_irreps`; splitting them would leave a half-supported knob, and no published checkpoint stores the attribute | `tests/unit/test_models.py::test_use_edge_irreps_first_narrows_the_first_interaction_block` + `tests/unit/test_models.py::test_use_edge_irreps_first_survives_the_checkpoint_config_round_trip` |
| `train.num_channels` | `--num_channels` | `mace/tools/arg_parser.py:292` | KEEP — shortcut for `hidden_irreps` | `tests/golden/test_tiny_anchors.py::test_anchor_reproduces_its_reference` |
| `train.max_L` | `--max_L` | `mace/tools/arg_parser.py:298` | KEEP — idem | `tests/golden/test_tiny_anchors.py::test_anchor_reproduces_its_reference` |
| `train.gate` | `--gate` | `mace/tools/arg_parser.py:304` | KEEP — see the gate rows in §7 | `tests/unit/test_gate.py::test_forward_matches_e3nn` + `tests/unit/test_gate.py::test_custom_activations` |

### 3.3 PolarMACE architecture (15)

Group default: KEEP as the electrostatics-extra config section, pinned by the polar golden.

| id | feature | source | disposition | pinned by |
|---|---|---|---|---|
| `train.kspace_cutoff_factor` | `--kspace_cutoff_factor` | `mace/tools/arg_parser.py:311` | KEEP | `tests/foundations/test_foundations.py::test_polar_extract_config_roundtrip` + `tests/golden/test_polar_foundation.py::test_the_reference_pins_the_electrostatics_and_not_only_the_energy` |
| `train.atomic_multipoles_max_l` | `--atomic_multipoles_max_l` | `mace/tools/arg_parser.py:317` | KEEP | `tests/foundations/test_foundations.py::test_polar_extract_config_roundtrip` + `tests/golden/test_polar_foundation.py::test_the_reference_pins_the_electrostatics_and_not_only_the_energy` |
| `train.atomic_multipoles_smearing_width` | `--atomic_multipoles_smearing_width` | `mace/tools/arg_parser.py:323` | KEEP | `tests/foundations/test_foundations.py::test_polar_extract_config_roundtrip` + `tests/golden/test_polar_foundation.py::test_the_reference_pins_the_electrostatics_and_not_only_the_energy` |
| `train.field_feature_max_l` | `--field_feature_max_l` | `mace/tools/arg_parser.py:329` | KEEP | `tests/foundations/test_foundations.py::test_polar_extract_config_roundtrip` + `tests/golden/test_polar_foundation.py::test_the_reference_pins_the_electrostatics_and_not_only_the_energy` |
| `train.field_feature_widths` | `--field_feature_widths` | `mace/tools/arg_parser.py:335` | KEEP | `tests/foundations/test_foundations.py::test_polar_extract_config_roundtrip` + `tests/golden/test_polar_foundation.py::test_the_reference_pins_the_electrostatics_and_not_only_the_energy` |
| `train.field_feature_norms` | `--field_feature_norms` | `mace/tools/arg_parser.py:341` | KEEP | `tests/foundations/test_foundations.py::test_polar_extract_config_roundtrip` + `tests/golden/test_polar_foundation.py::test_the_reference_pins_the_electrostatics_and_not_only_the_energy` |
| `train.num_recursion_steps` | `--num_recursion_steps` | `mace/tools/arg_parser.py:347` | KEEP | `tests/foundations/test_foundations.py::test_polar_extract_config_roundtrip` + `tests/golden/test_polar_foundation.py::test_the_reference_pins_the_electrostatics_and_not_only_the_energy` |
| `train.field_si` | `--field_si` | `mace/tools/arg_parser.py:353` | KEEP | `tests/foundations/test_foundations.py::test_polar_extract_config_roundtrip` + `tests/golden/test_polar_foundation.py::test_the_reference_pins_the_electrostatics_and_not_only_the_energy` |
| `train.include_electrostatic_self_interaction` | `--include_electrostatic_self_interaction` | `mace/tools/arg_parser.py:359` | KEEP | `tests/extensions/polar/test_polar_models.py::test_polar_slab_electrostatics_converge_with_vacuum` |
| `train.add_local_electron_energy` | `--add_local_electron_energy` | `mace/tools/arg_parser.py:365` | KEEP | `tests/extensions/polar/test_polar_models.py::test_polar_checkpoint_energy_components` |
| `train.quadrupole_feature_corrections` | `--quadrupole_feature_corrections` | `mace/tools/arg_parser.py:371` | KEEP | `tests/foundations/test_foundations.py::test_polar_extract_config_roundtrip` + `tests/golden/test_polar_foundation.py::test_the_reference_pins_the_electrostatics_and_not_only_the_energy` |
| `train.return_electrostatic_potentials` | `--return_electrostatic_potentials` | `mace/tools/arg_parser.py:377` | MERGE — an observable declared in the output spec, not a model flag | `tests/extensions/polar/test_polar_output_keys.py::test_electrostatic_potentials_are_absent_unless_asked_for` (pins the off state; a case with it on is still to come) |
| `train.field_norm_factor` | `--field_norm_factor` | `mace/tools/arg_parser.py:383` | KEEP | `tests/foundations/test_foundations.py::test_polar_extract_config_roundtrip` + `tests/golden/test_polar_foundation.py::test_the_reference_pins_the_electrostatics_and_not_only_the_energy` |
| `train.fixedpoint_update_config` | `--fixedpoint_update_config` | `mace/tools/arg_parser.py:389` | KEEP — the fixed-point solver settings; the expert electrostatics config section | ⚠️ gap (add a case to `tests/golden/test_polar_foundation.py`) |
| `train.field_readout_config` | `--field_readout_config` | `mace/tools/arg_parser.py:395` | KEEP — idem | ⚠️ gap (add a case to `tests/golden/test_polar_foundation.py`) |

### 3.4 Outputs and scaling (8)

Group default: MERGE into the declarative observable specification — an observable that is declared is computed, so the `--compute_*` booleans stop being independent flags.

| id | feature | source | disposition | pinned by |
|---|---|---|---|---|
| `train.scaling` | `--scaling` | `mace/tools/arg_parser.py:401` | KEEP — see the scaling rows in §7 | `tests/unit/test_e0s_characterization.py::test_the_mean_and_std_scaling_is_the_per_atom_interaction_energy` + `tests/unit/test_e0s_characterization.py::test_no_scaling_overrides_even_an_explicit_std` |
| `train.avg_num_neighbors` | `--avg_num_neighbors` | `mace/tools/arg_parser.py:408` | MERGE — dataset statistics + model metadata | `tests/unit/test_e0s_characterization.py::test_the_average_neighbour_count_skips_atoms_that_have_no_neighbours` |
| `train.compute_avg_num_neighbors` | `--compute_avg_num_neighbors` | `mace/tools/arg_parser.py:414` | MERGE — idem | `tests/unit/test_e0s_characterization.py::test_the_average_neighbour_count_skips_atoms_that_have_no_neighbours` |
| `train.compute_stress` | `--compute_stress` | `mace/tools/arg_parser.py:420` | MERGE — observable spec (stress declared ⇒ computed) | `tests/unit/test_physics_glue.py::test_stress_is_the_strain_derivative_over_the_volume` + `tests/unit/test_physics_glue.py::test_which_derivatives_a_flag_combination_produces` |
| `train.compute_forces` | `--compute_forces` | `mace/tools/arg_parser.py:426` | MERGE — idem | `tests/unit/test_physics_glue.py::test_forces_are_minus_the_energy_gradient` |
| `train.compute_polarizability` | `--compute_polarizability` | `mace/tools/arg_parser.py:432` | MERGE — idem | `tests/golden/test_mdp_foundation.py::test_the_reference_pins_the_polarizability_and_its_derivatives` |
| `train.compute_atomic_dipole` | `--compute_atomic_dipole` | `mace/tools/arg_parser.py:438` | MERGE — idem | `tests/golden/test_tiny_dipoles.py::test_anchor_reproduces_its_reference` |
| `train.compute_magforces` | `--compute_magforces` | `mace/tools/arg_parser.py:444` | MERGE — idem: `dE/dm` is a declared derivative exactly like forces and stress | `tests/golden/test_tiny_magnetic.py::test_compute_magforces_is_only_honoured_alongside_the_forces` |

### 3.5 Data, files and property keys (29)

Group default: the `*_key` flags MERGE into the property-key convention; the file/loading flags KEEP as the `data` config section.

| id | feature | source | disposition | pinned by |
|---|---|---|---|---|
| `train.train_file` | `--train_file` | `mace/tools/arg_parser.py:452` | KEEP | `tests/workflows/test_cli_contracts.py::test_training_reduces_the_validation_loss_and_writes_a_model` |
| `train.valid_file` | `--valid_file` | `mace/tools/arg_parser.py:458` | KEEP | `tests/workflows/test_multifiles.py::test_multifile_training` |
| `train.test_file` | `--test_file` | `mace/tools/arg_parser.py:472` | KEEP | `tests/workflows/test_multifiles.py::test_single_xyz_per_head` |
| `train.test_dir` | `--test_dir` | `mace/tools/arg_parser.py:477` | KEEP | `tests/unit/test_multihead_tools.py::test_prepare_default_head_single_default_head` |
| `train.valid_fraction` | `--valid_fraction` | `mace/tools/arg_parser.py:465` | KEEP | `tests/workflows/test_cli_contracts.py::test_training_reduces_the_validation_loss_and_writes_a_model` |
| `train.multi_processed_test` | `--multi_processed_test` | `mace/tools/arg_parser.py:484` | MERGE — the data layer infers sharding from the dataset; whether a test set is split across files is not something the user should have to declare and get wrong (today it is a bare `if` in `run_train.py`) | ⚠️ gap (add a case to `tests/workflows/test_cli_contracts.py`) |
| `train.num_workers` | `--num_workers` | `mace/tools/arg_parser.py:491` | KEEP | `tests/workflows/test_train_work_dir_and_workers.py::test_every_dataloader_is_given_the_worker_count` (every loader is given it; the run itself is smoked by `test_a_run_with_workers_trains`) |
| `train.pin_memory` | `--pin_memory` | `mace/tools/arg_parser.py:497` | KEEP | ⚠️ gap (perf knob; conscious downgrade candidate) |
| `train.atomic_numbers` | `--atomic_numbers` | `mace/tools/arg_parser.py:503` | MERGE — statistics / model metadata | `tests/foundations/test_foundations.py::test_mace_mh_1_elements_subset_reproduces_energy_forces` |
| `train.mean` | `--mean` | `mace/tools/arg_parser.py:510` | MERGE — statistics override | `tests/unit/test_e0s_characterization.py::test_mean_and_std_together_override_the_dataset_statistics` + `tests/unit/test_e0s_characterization.py::test_mean_without_std_is_accepted_and_then_discarded` |
| `train.std` | `--std` | `mace/tools/arg_parser.py:517` | MERGE — idem | `tests/unit/test_e0s_characterization.py::test_mean_and_std_together_override_the_dataset_statistics` + `tests/unit/test_e0s_characterization.py::test_no_scaling_overrides_even_an_explicit_std` |
| `train.statistics_file` | `--statistics_file` | `mace/tools/arg_parser.py:524` | KEEP | `tests/workflows/test_preprocess.py` |
| `train.les_arguments` | `--les_arguments` | `mace/tools/arg_parser.py:531` | KEEP — the LES extra's solver settings | `tests/extensions/les` + `tests/golden/test_tiny_maceles.py::test_the_model_surface_reproduces_its_reference` |
| `train.E0s` | `--E0s` | `mace/tools/arg_parser.py:538` | KEEP — explicit / average / estimated / foundation | `tests/unit/test_e0s_characterization.py::test_e0s_average_goes_through_the_least_squares_fit` + `tests/unit/test_e0s_characterization.py::test_e0s_from_a_literal_dict_and_from_a_json_file` |
| `train.keep_isolated_atoms` | `--keep_isolated_atoms` | `mace/tools/arg_parser.py:646` | KEEP | `tests/unit/test_e0s_characterization.py::test_keep_isolated_atoms_leaves_them_in_the_training_set` |
| `train.config_type_weights` | `--config_type_weights` | `mace/tools/arg_parser.py:882` | KEEP — per-config-type loss weighting | `tests/unit/test_loss.py::test_a_zero_config_weight_dilutes_rather_than_renormalizing` + `tests/unit/test_data_utils.py::test_config_type_and_per_property_weights_round_trip` |
| `train.energy_key` | `--energy_key` | `mace/tools/arg_parser.py:672` | MERGE — property-key convention of the observable spec | `tests/unit/test_data_utils.py::test_custom_property_keys_round_trip` + `tests/unit/test_data_utils.py::test_keydict_derives_one_cli_argument_per_member` |
| `train.forces_key` | `--forces_key` | `mace/tools/arg_parser.py:678` | MERGE — idem | `tests/unit/test_data_utils.py::test_custom_property_keys_round_trip` + `tests/unit/test_data_utils.py::test_keydict_derives_one_cli_argument_per_member` |
| `train.virials_key` | `--virials_key` | `mace/tools/arg_parser.py:684` | MERGE — idem | `tests/unit/test_data_utils.py::test_custom_property_keys_round_trip` + `tests/unit/test_data_utils.py::test_keydict_derives_one_cli_argument_per_member` |
| `train.stress_key` | `--stress_key` | `mace/tools/arg_parser.py:690` | MERGE — idem | `tests/unit/test_data_utils.py::test_custom_property_keys_round_trip` + `tests/unit/test_data_utils.py::test_keydict_derives_one_cli_argument_per_member` |
| `train.dipole_key` | `--dipole_key` | `mace/tools/arg_parser.py:696` | MERGE — idem | `tests/unit/test_data_utils.py::test_custom_property_keys_round_trip` + `tests/unit/test_data_utils.py::test_keydict_derives_one_cli_argument_per_member` |
| `train.polarizability_key` | `--polarizability_key` | `mace/tools/arg_parser.py:702` | MERGE — idem | `tests/unit/test_data_utils.py::test_custom_property_keys_round_trip` + `tests/unit/test_data_utils.py::test_keydict_derives_one_cli_argument_per_member` |
| `train.charges_key` | `--charges_key` | `mace/tools/arg_parser.py:726` | MERGE — idem | `tests/unit/test_data_utils.py::test_custom_property_keys_round_trip` + `tests/unit/test_data_utils.py::test_keydict_derives_one_cli_argument_per_member` |
| `train.head_key` | `--head_key` | `mace/tools/arg_parser.py:720` | MERGE — idem | `tests/unit/test_data_utils.py::test_custom_property_keys_round_trip` + `tests/unit/test_data_utils.py::test_keydict_derives_one_cli_argument_per_member` |
| `train.elec_temp_key` | `--elec_temp_key` | `mace/tools/arg_parser.py:732` | KEEP — graph-level input feature | `tests/workflows/test_embedding_train.py::test_run_train_with_elec_temp` + `tests/unit/test_lmdb_database.py::test_lmdb_dataset_honors_key_specification` |
| `train.total_spin_key` | `--total_spin_key` | `mace/tools/arg_parser.py:738` | KEEP — idem | `tests/unit/test_lmdb_database.py::test_lmdb_dataset_honors_key_specification` |
| `train.total_charge_key` | `--total_charge_key` | `mace/tools/arg_parser.py:744` | KEEP — idem | `tests/unit/test_lmdb_database.py::test_lmdb_dataset_honors_key_specification` |
| `train.embedding_specs` | `--embedding_specs` | `mace/tools/arg_parser.py:750` | KEEP — categorical / graph-level embeddings | `tests/workflows/test_embedding_train.py::test_run_train_with_atom_embed` + `tests/unit/test_data_utils.py::test_update_keyspec_from_kwargs_embedding_specs` |
| `train.skip_evaluate_heads` | `--skip_evaluate_heads` | `mace/tools/arg_parser.py:773` | KEEP | ⚠️ gap (add a case to `tests/workflows/test_finetuning_contracts.py`) |

### 3.6 Fine-tuning, multihead and foundation models (24)

Group default: KEEP.

| id | feature | source | disposition | pinned by |
|---|---|---|---|---|
| `train.foundation_model` | `--foundation_model` | `mace/tools/arg_parser.py:1019` | KEEP | `tests/workflows/test_finetuning_contracts.py::test_multihead_replay_finetuning_completes_and_carries_both_heads` + `tests/workflows/test_finetuning_contracts.py::test_finetuning_reduces_the_error_on_the_finetuning_set` |
| `train.foundation_model_kwargs` | `--foundation_model_kwargs` | `mace/tools/arg_parser.py:1025` | KEEP | ⚠️ gap (no test passes `--foundation_model_kwargs`) |
| `train.foundation_model_readout` | `--foundation_model_readout` | `mace/tools/arg_parser.py:1031` | KEEP | ⚠️ gap (no test passes `--foundation_model_readout`) |
| `train.multiheads_finetuning` | `--multiheads_finetuning` | `mace/tools/arg_parser.py:573` | KEEP | `tests/workflows/test_finetuning_contracts.py::test_multihead_replay_finetuning_completes_and_carries_both_heads` |
| `train.heads` | `--heads` | `mace/tools/arg_parser.py:566` | KEEP — the heads YAML sub-schema | `tests/workflows/test_multifiles.py::test_multiple_xyz_per_head` |
| `train.foundation_head` | `--foundation_head` | `mace/tools/arg_parser.py:579` | KEEP | `tests/workflows/test_run_train.py::test_run_train_mh_foundation` |
| `train.weight_pt_head` | `--weight_pt_head` | `mace/tools/arg_parser.py:586` | KEEP | ⚠️ gap (no test passes `--weight_pt_head`, so the replay head's loss weight is unprotected) |
| `train.num_samples_pt` | `--num_samples_pt` | `mace/tools/arg_parser.py:598` | KEEP | `tests/workflows/test_finetuning_contracts.py::test_the_replay_selection_flags_exist_on_the_training_cli` |
| `train.real_pt_data_ratio_threshold` | `--real_pt_data_ratio_threshold` | `mace/tools/arg_parser.py:592` | KEEP | `tests/workflows/test_run_train.py::test_run_train_real_pt_data_ratio` |
| `train.pt_train_file` | `--pt_train_file` | `mace/tools/arg_parser.py:628` | KEEP | `tests/workflows/test_run_train.py::test_run_train_multihead_replay_custom_finetuning` |
| `train.pt_valid_file` | `--pt_valid_file` | `mace/tools/arg_parser.py:634` | KEEP | `tests/workflows/test_run_train.py::test_run_train_multihead_replay_custom_finetuning` |
| `train.subselect_pt` | `--subselect_pt` | `mace/tools/arg_parser.py:610` | KEEP | `tests/workflows/test_finetuning_contracts.py::test_subselect_random_returns_exactly_the_requested_number` + `tests/workflows/test_finetuning_contracts.py::test_subselect_fps_uses_the_model_and_still_returns_the_requested_number` |
| `train.filter_type_pt` | `--filter_type_pt` | `mace/tools/arg_parser.py:616` | KEEP | `tests/workflows/test_finetuning_contracts.py::test_filtering_type_restricts_the_pool_to_the_target_elements` |
| `train.allow_random_padding_pt` | `--disallow_random_padding_pt` | `mace/tools/arg_parser.py:622` | KEEP — spelled `--disallow_random_padding_pt`, stored inverted | `tests/workflows/test_finetuning_contracts.py::test_random_padding_tops_up_a_short_pool_and_disallowing_it_does_not` + `tests/workflows/test_finetuning_contracts.py::test_disallow_random_padding_pt_is_a_bare_flag_on_the_training_cli` |
| `train.pseudolabel_replay` | `--pseudolabel_replay` | `mace/tools/arg_parser.py:547` | KEEP | `tests/workflows/test_finetuning_contracts.py::test_pseudolabel_replay_relabels_the_replay_set_from_the_foundation_model` + `tests/workflows/test_finetuning_contracts.py::test_a_failing_pseudolabel_batch_keeps_the_file_labels_and_says_nothing` |
| `train.pseudolabel_replay_compute_stress` | `--pseudolabel_replay_compute_stress` | `mace/tools/arg_parser.py:553` | KEEP | ⚠️ gap (no test passes `--pseudolabel_replay_compute_stress`) |
| `train.foundation_filter_elements` | `--foundation_filter_elements` | `mace/tools/arg_parser.py:559` | KEEP | ⚠️ gap (all-species saving; add a case to `tests/workflows/test_run_train.py`) |
| `train.foundation_model_elements` | `--foundation_model_elements` | `mace/tools/arg_parser.py:640` | KEEP — all-species weight saving is a v1 default | `tests/workflows/test_run_train.py::test_run_train_foundation_elements` + `tests/workflows/test_run_train.py::test_run_train_foundation_elements_multihead` |
| `train.force_mh_ft_lr` | `--force_mh_ft_lr` | `mace/tools/arg_parser.py:604` | DROP — replay-dependent defaults replace the override; the flag exists only to defeat a heuristic v1 does not have | — |
| `train.lora` | `--lora` | `mace/tools/arg_parser.py:652` | KEEP | `tests/unit/test_lora.py` (port cases) |
| `train.lora_rank` | `--lora_rank` | `mace/tools/arg_parser.py:658` | KEEP | `tests/unit/test_lora.py` |
| `train.lora_alpha` | `--lora_alpha` | `mace/tools/arg_parser.py:664` | KEEP | `tests/unit/test_lora.py` |
| `train.freeze` | `--freeze` | `mace/tools/arg_parser.py:949` | KEEP | `tests/workflows/test_freeze.py` (port cases) |
| `train.finetune_dipoles_polarizabilities` | `--finetune_dipoles_polarizabilities` | `mace/tools/arg_parser.py:1037` | KEEP — the MDP fine-tuning path | `tests/workflows/test_mdp_finetune.py::test_mdp_finetune_updates_params` + `tests/workflows/test_mdp_finetune.py::test_mdp_finetune_wrong_model_type_raises` |

### 3.7 Loss (16)

Group default: MERGE into composable per-stage losses; the numerics are pinned by the hand-computed loss cases. **The per-dest re-key changes what the `swa_*_weight` rows say.** `--swa_energy_weight` and `--stage_two_energy_weight` are two spellings of one dest, so there is one row and one disposition: MERGE. The legacy `swa` *spelling* dies with the flag namespace, but that is not a separate disposition — an option-string-keyed inventory that said 'DROP the `--swa_*` aliases' was describing a spelling, not a knob.

| id | feature | source | disposition | pinned by |
|---|---|---|---|---|
| `train.loss` | `--loss` | `mace/tools/arg_parser.py:781` | MERGE — the 11 named schemes become loss-composition presets over the 10 loss classes of §8 | `tests/unit/test_loss.py::test_every_cli_loss_name_reaches_its_class_with_its_weights` |
| `train.energy_weight` | `--energy_weight` | `mace/tools/arg_parser.py:824` | KEEP — per-observable weight | `tests/unit/test_loss.py::test_weighted_energy_forces_loss_global_weights_scale` |
| `train.forces_weight` | `--forces_weight` | `mace/tools/arg_parser.py:799` | KEEP — idem | `tests/unit/test_loss.py::test_weighted_energy_forces_loss_global_weights_scale` |
| `train.virials_weight` | `--virials_weight` | `mace/tools/arg_parser.py:835` | KEEP — idem | `tests/unit/test_loss.py::test_weighted_energy_forces_virials_loss` |
| `train.stress_weight` | `--stress_weight` | `mace/tools/arg_parser.py:846` | KEEP — idem | `tests/unit/test_loss.py::test_weighted_energy_forces_stress_loss_zero_and_hand_value` |
| `train.dipole_weight` | `--dipole_weight` | `mace/tools/arg_parser.py:857` | KEEP — idem | `tests/unit/test_loss.py::test_weighted_energy_forces_dipole_loss` |
| `train.polarizability_weight` | `--polarizability_weight` | `mace/tools/arg_parser.py:876` | KEEP — idem | `tests/unit/test_loss.py::test_weighted_mean_squared_error_polarizability_reshapes_only_the_reference` |
| `train.magforces_weight` | `--magforces_weight` | `mace/tools/arg_parser.py:810` | KEEP — idem | `tests/unit/test_loss.py::test_universal_loss_magforces_hand_value` + `tests/unit/test_loss.py::test_universal_loss_magforces_per_config_weight_multiplies_the_arguments` |
| `train.swa_energy_weight` | `--swa_energy_weight` `--stage_two_energy_weight` | `mace/tools/arg_parser.py:827` | MERGE — per-stage schedules; arbitrary stages replace the two-stage special case, and the `swa` spelling dies with the namespace | `tests/unit/test_stage_two_weights.py::test_the_energy_and_forces_weights_reach_the_stage_two_loss` |
| `train.swa_forces_weight` | `--swa_forces_weight` `--stage_two_forces_weight` | `mace/tools/arg_parser.py:802` | MERGE — idem | `tests/unit/test_stage_two_weights.py::test_the_energy_and_forces_weights_reach_the_stage_two_loss` |
| `train.swa_virials_weight` | `--swa_virials_weight` `--stage_two_virials_weight` | `mace/tools/arg_parser.py:838` | MERGE — idem | `tests/unit/test_stage_two_weights.py::test_the_virials_weight_reaches_the_virials_loss` |
| `train.swa_stress_weight` | `--swa_stress_weight` `--stage_two_stress_weight` | `mace/tools/arg_parser.py:849` | MERGE — idem | `tests/unit/test_stage_two_weights.py::test_the_stress_weight_reaches_the_stress_loss` |
| `train.swa_dipole_weight` | `--swa_dipole_weight` `--stage_two_dipole_weight` | `mace/tools/arg_parser.py:860` | MERGE — idem | ⚠️ gap (no test sets a stage-two loss weight; `--swa_dipole_weight` only takes effect after the swap) |
| `train.swa_polarizability_weight` | `--swa_polarizability_weight` `--stage_two_polarizability_weight` | `mace/tools/arg_parser.py:868` | MERGE — idem | ⚠️ gap (no test sets a stage-two loss weight; `--swa_polarizability_weight` only takes effect after the swap) |
| `train.swa_magforces_weight` | `--swa_magforces_weight` `--stage_two_magforces_weight` | `mace/tools/arg_parser.py:816` | MERGE — idem | ⚠️ gap (add a case to `tests/golden/test_tiny_magnetic.py`) |
| `train.huber_delta` | `--huber_delta` | `mace/tools/arg_parser.py:888` | KEEP | `tests/unit/test_loss.py::test_conditional_huber_forces` + `tests/unit/test_loss.py::test_weighted_huber_energy_forces_stress_loss` |

### 3.8 Optimizer, scheduler and training control (26)

Group default: KEEP as the `optimizer` / `schedule` config sections. `--swa`, `--start_swa` and `--swa_lr` carry the `--stage_two*` spellings on the same dest, so they are one row each.

| id | feature | source | disposition | pinned by |
|---|---|---|---|---|
| `train.optimizer` | `--optimizer` | `mace/tools/arg_parser.py:894` | KEEP — adam / adamw / schedulefree | `tests/workflows/test_cli_contracts.py::test_lbfgs_is_selectable_from_the_command_line_and_reduces_the_loss` + `tests/extensions/schedulefree` |
| `train.beta` | `--beta` | `mace/tools/arg_parser.py:901` | KEEP | `tests/unit/test_optimizer_flags.py::test_beta_is_the_first_adam_moment` |
| `train.amsgrad` | `--amsgrad` | `mace/tools/arg_parser.py:955` | KEEP | `tests/unit/test_optimizer_flags.py::test_amsgrad_reaches_the_optimizer` |
| `train.weight_decay` | `--weight_decay` | `mace/tools/arg_parser.py:940` | KEEP | `tests/unit/test_optimizer_flags.py::test_weight_decay_applies_to_the_interaction_linears_and_the_products` + `tests/unit/test_optimizer_flags.py::test_weight_decay_is_kept_off_the_rest` |
| `train.beta1_schedulefree` | `--beta1_schedulefree` | `mace/tools/arg_parser.py:907` | KEEP — the schedulefree extra's tuning surface | `tests/extensions/schedulefree` |
| `train.beta2_schedulefree` | `--beta2_schedulefree` | `mace/tools/arg_parser.py:913` | KEEP — idem | `tests/extensions/schedulefree` |
| `train.warmup_steps_schedulefree` | `--warmup_steps_schedulefree` | `mace/tools/arg_parser.py:919` | KEEP — idem (linear LR warmup, not an LR scheduler) | `tests/extensions/schedulefree` |
| `train.lbfgs` | `--lbfgs` | `mace/tools/arg_parser.py:992` | KEEP — a second training regime, not an optimizer choice: full-batch gradient assembled in chunks, one `step(closure)` per epoch, ragged tail kept, its own resume fallback | `tests/workflows/test_run_train.py::test_run_train_lbfgs` |
| `train.batch_size` | `--batch_size` | `mace/tools/arg_parser.py:924` | KEEP | `tests/workflows/test_cli_contracts.py::test_lbfgs_keeps_the_last_partial_batch_and_the_default_regime_drops_it` |
| `train.valid_batch_size` | `--valid_batch_size` | `mace/tools/arg_parser.py:926` | KEEP — a separate knob from `--batch_size`, and the one a group-level row loses | `tests/workflows/test_cli_contracts.py::test_lbfgs_keeps_the_last_partial_batch_and_the_default_regime_drops_it` |
| `train.lr` | `--lr` | `mace/tools/arg_parser.py:929` | KEEP | `tests/workflows/test_cli_contracts.py::test_training_reduces_the_validation_loss_and_writes_a_model` |
| `train.lr_factor` | `--lr_factor` | `mace/tools/arg_parser.py:964` | KEEP | `tests/workflows/test_freeze.py::test_run_train_soft_freeze` |
| `train.scheduler` | `--scheduler` | `mace/tools/arg_parser.py:961` | KEEP | `tests/extensions/schedulefree/test_schedulefree.py::test_can_load_checkpoint` (sets `args.scheduler`; nothing passes the flag itself) |
| `train.scheduler_patience` | `--scheduler_patience` | `mace/tools/arg_parser.py:967` | KEEP | `tests/unit/test_optimizer_flags.py::test_scheduler_patience_reaches_the_plateau_scheduler` |
| `train.lr_scheduler_gamma` | `--lr_scheduler_gamma` | `mace/tools/arg_parser.py:970` | KEEP | `tests/extensions/schedulefree/test_schedulefree.py::test_can_load_checkpoint` (sets `args.lr_scheduler_gamma`; nothing passes the flag itself) |
| `train.lr_params_factors` | `--lr_params_factors` | `mace/tools/arg_parser.py:943` | MERGE — typed per-param-group fields of the per-stage optimizer config; the capability stays (`--freeze` reuses it by zeroing factors), the hand-parsed JSON-in-a-string dies | `tests/workflows/test_freeze.py::test_run_train_soft_freeze` |
| `train.swa` | `--swa` `--stage_two` | `mace/tools/arg_parser.py:976` | MERGE — stage two becomes a preset second stage of an arbitrary-stage schedule | `tests/unit/test_arg_parser.py::test_stage_two_alias_maps_to_swa_dest` + `tests/workflows/test_multifiles.py::test_multifile_training` |
| `train.start_swa` | `--start_swa` `--start_stage_two` | `mace/tools/arg_parser.py:984` | MERGE — idem | `tests/unit/test_arg_parser.py::test_stage_two_alias_maps_to_swa_dest` + `tests/workflows/test_multifiles.py::test_multifile_training` |
| `train.swa_lr` | `--swa_lr` `--stage_two_lr` | `mace/tools/arg_parser.py:932` | MERGE — idem | `tests/unit/test_optimizer_flags.py::test_swa_lr_is_the_rate_the_second_stage_anneals_to` |
| `train.ema` | `--ema` | `mace/tools/arg_parser.py:998` | KEEP | `tests/unit/test_optimizer_flags.py::test_the_average_lags_the_parameters` |
| `train.ema_decay` | `--ema_decay` | `mace/tools/arg_parser.py:1004` | KEEP | `tests/unit/test_optimizer_flags.py::test_the_configured_decay_barely_matters_at_the_start` (the cold-start cap) + `tests/unit/test_optimizer_flags.py::test_a_higher_decay_lags_further_once_the_warmup_is_past` |
| `train.max_num_epochs` | `--max_num_epochs` | `mace/tools/arg_parser.py:1010` | KEEP | `tests/workflows/test_cli_contracts.py::test_training_reduces_the_validation_loss_and_writes_a_model` |
| `train.patience` | `--patience` | `mace/tools/arg_parser.py:1013` | KEEP | `tests/workflows/test_multifiles.py::test_multifile_training` |
| `train.eval_interval` | `--eval_interval` | `mace/tools/arg_parser.py:1043` | KEEP | `tests/workflows/test_cli_contracts.py::test_training_reduces_the_validation_loss_and_writes_a_model` |
| `train.clip_grad` | `--clip_grad` | `mace/tools/arg_parser.py:1070` | KEEP | ⚠️ gap (add an assertion to `tests/workflows/test_cli_contracts.py::test_training_reduces_the_validation_loss_and_writes_a_model`) |
| `train.dry_run` | `--dry_run` | `mace/tools/arg_parser.py:1076` | KEEP — cheap and useful | `tests/workflows/test_run_train.py::test_run_train_real_pt_data_ratio` |

### 3.9 Checkpointing (4)

Group default: KEEP.

| id | feature | source | disposition | pinned by |
|---|---|---|---|---|
| `train.restart_latest` | `--restart_latest` | `mace/tools/arg_parser.py:1058` | KEEP | `tests/workflows/test_cli_contracts.py::test_restart_latest_continues_from_the_checkpoint_epoch` |
| `train.keep_checkpoints` | `--keep_checkpoints` | `mace/tools/arg_parser.py:1046` | KEEP | ⚠️ gap (add an assertion to `tests/workflows/test_cli_contracts.py::test_restart_latest_continues_from_the_checkpoint_epoch`) |
| `train.save_all_checkpoints` | `--save_all_checkpoints` | `mace/tools/arg_parser.py:1052` | KEEP | ⚠️ gap (idem) |
| `train.save_cpu` | `--save_cpu` | `mace/tools/arg_parser.py:1064` | DROP — safetensors checkpoints are device-agnostic, so there is nothing to choose | — |

### 3.10 Acceleration (3)

Group default: MERGE into backend-dispatch configuration; the numerics are pinned on GPU CI.

| id | feature | source | disposition | pinned by |
|---|---|---|---|---|
| `train.enable_cueq` | `--enable_cueq` | `mace/tools/arg_parser.py:1083` | MERGE — backend dispatch config | `tests/golden/test_backend_parity_golden.py::test_the_calculators_own_backend_flag_reaches_the_same_kernels` |
| `train.enable_oeq` | `--enable_oeq` | `mace/tools/arg_parser.py:1096` | MERGE — idem | `tests/golden/test_backend_parity_golden.py::test_the_calculators_own_backend_flag_reaches_the_same_kernels` |
| `train.only_cueq` | `--only_cueq` | `mace/tools/arg_parser.py:1089` | MERGE — idem: 'use cueq for every op, not just the ones that benefit' becomes a dispatch policy, not a second boolean. Its own row precisely because a group-level `--enable_cueq/--only_cueq/--enable_oeq` cell hides it | `tests/golden/test_backend_parity_golden.py::test_converting_for_the_cpu_leaves_cueq_unfused_and_the_audit_says_so` |

### 3.11 wandb (6)

Group default: KEEP (the wandb extra).

| id | feature | source | disposition | pinned by |
|---|---|---|---|---|
| `train.wandb` | `--wandb` | `mace/tools/arg_parser.py:1104` | KEEP | ⚠️ gap (offline-mode smoke) |
| `train.wandb_dir` | `--wandb_dir` | `mace/tools/arg_parser.py:1110` | KEEP | ⚠️ gap (idem) |
| `train.wandb_project` | `--wandb_project` | `mace/tools/arg_parser.py:1116` | KEEP | ⚠️ gap (idem) |
| `train.wandb_entity` | `--wandb_entity` | `mace/tools/arg_parser.py:1122` | KEEP | ⚠️ gap (idem) |
| `train.wandb_name` | `--wandb_name` | `mace/tools/arg_parser.py:1128` | KEEP | ⚠️ gap (idem) |
| `train.wandb_log_hypers` | `--wandb_log_hypers` | `mace/tools/arg_parser.py:1134` | KEEP | ⚠️ gap (idem) |

### 3.12 MagneticMACE (9)

Group default: KEEP as the `magnetic`-extra config. Grouped like §3.3, but the subgroups land in different v1 mechanisms: the two `*_key` flags follow the property-key convention and `--compute_magforces` (§3.4) follows the observable spec. The weights are in §3.7.

| id | feature | source | disposition | pinned by |
|---|---|---|---|---|
| `train.magmom_key` | `--magmom_key` | `mace/tools/arg_parser.py:708` | MERGE — property-key convention; the default `REF_magmom` extends the on-disk data contract of §15 | `tests/extensions/magnetic` |
| `train.magforces_key` | `--magforces_key` | `mace/tools/arg_parser.py:714` | MERGE — idem, default `REF_magforces` | `tests/extensions/magnetic` |
| `train.m_max` | `--m_max` | `mace/tools/arg_parser.py:1160` | KEEP — magnetic architecture hyper | `tests/extensions/magnetic` (7 `resolve_m_max` cases) |
| `train.max_m_ell` | `--max_m_ell` | `mace/tools/arg_parser.py:1172` | KEEP — idem | `tests/extensions/magnetic` |
| `train.num_mag_radial_basis` | `--num_mag_radial_basis` | `mace/tools/arg_parser.py:1178` | KEEP — idem | `tests/extensions/magnetic` |
| `train.num_mag_radial_basis_one_body` | `--num_mag_radial_basis_one_body` | `mace/tools/arg_parser.py:1154` | KEEP — idem | `tests/extensions/magnetic` |
| `train.use_magmom_one_body` | `--use_magmom_one_body` | `mace/tools/arg_parser.py:1184` | KEEP — the one-body magmom term | `tests/golden/test_tiny_magnetic.py::test_the_one_body_magnetic_term_is_inside_the_reference` |
| `train.train_one_body_contribution` | `--train_one_body_contribution` | `mace/tools/arg_parser.py:1190` | KEEP — whether the one-body coefficients are optimized | ⚠️ gap (add a case to `tests/golden/test_tiny_magnetic.py`) |
| `train.data_aug_magmom` | `--data_aug_magmom` | `mace/tools/arg_parser.py:1197` | MERGE — a training-data transform (`Random3DRotation`), not a model flag | `tests/extensions/magnetic` (rotation equivariance) |
| `train.data_aug_magmom_mode` | `--data_aug_magmom_mode` | `mace/tools/arg_parser.py:1203` | MERGE — selects which spin symmetry the transform draws from (`non-soc` the full O(3), `soc` the sign flip alone), so it travels with `data_aug_magmom` | `tests/extensions/magnetic/test_magmom_augmentation.py::test_non_soc_mode_samples_the_full_o3` |

## 4. `mace_prepare_data` flags — 26 dests

One row per dest of `build_preprocess_arg_parser`. **22 of the 26 are also declared by the
training parser** and get a row in each section: the defaults and the help text differ per parser,
so the same spelling is two knobs needing two dispositions.

| id | feature | source | disposition | pinned by |
|---|---|---|---|---|
| `prep.config` | `--config` | `mace/tools/arg_parser.py:1215` | MERGE — same mechanism and disposition as the training parser's `config` | `tests/workflows/test_preprocess.py::test_preprocess_config` |
| `prep.train_file` | `--train_file` | `mace/tools/arg_parser.py:1225` | MERGE — `mace data prepare` reads the same config section as `mace train` instead of redeclaring the flag | `tests/workflows/test_preprocess.py` |
| `prep.valid_file` | `--valid_file` | `mace/tools/arg_parser.py:1232` | MERGE — `mace data prepare` reads the same config section as `mace train` instead of redeclaring the flag | `tests/workflows/test_preprocess.py` |
| `prep.test_file` | `--test_file` | `mace/tools/arg_parser.py:1252` | MERGE — `mace data prepare` reads the same config section as `mace train` instead of redeclaring the flag | `tests/workflows/test_preprocess.py` |
| `prep.valid_fraction` | `--valid_fraction` | `mace/tools/arg_parser.py:1245` | MERGE — `mace data prepare` reads the same config section as `mace train` instead of redeclaring the flag | `tests/workflows/test_preprocess.py` |
| `prep.work_dir` | `--work_dir` | `mace/tools/arg_parser.py:1259` | MERGE — `mace data prepare` reads the same config section as `mace train` instead of redeclaring the flag | `tests/workflows/test_preprocess.py` |
| `prep.r_max` | `--r_max` | `mace/tools/arg_parser.py:1271` | MERGE — `mace data prepare` reads the same config section as `mace train` instead of redeclaring the flag | `tests/workflows/test_preprocess.py` |
| `prep.config_type_weights` | `--config_type_weights` | `mace/tools/arg_parser.py:1274` | MERGE — `mace data prepare` reads the same config section as `mace train` instead of redeclaring the flag | `tests/workflows/test_preprocess.py::test_preprocess_data` |
| `prep.energy_key` | `--energy_key` | `mace/tools/arg_parser.py:1280` | MERGE — `mace data prepare` reads the same config section as `mace train` instead of redeclaring the flag | `tests/workflows/test_preprocess.py` |
| `prep.forces_key` | `--forces_key` | `mace/tools/arg_parser.py:1286` | MERGE — `mace data prepare` reads the same config section as `mace train` instead of redeclaring the flag | `tests/workflows/test_preprocess.py` |
| `prep.virials_key` | `--virials_key` | `mace/tools/arg_parser.py:1292` | MERGE — `mace data prepare` reads the same config section as `mace train` instead of redeclaring the flag | `tests/workflows/test_preprocess_keys.py::test_a_property_key_is_read_under_the_name_it_is_given` |
| `prep.stress_key` | `--stress_key` | `mace/tools/arg_parser.py:1298` | MERGE — `mace data prepare` reads the same config section as `mace train` instead of redeclaring the flag | `tests/workflows/test_preprocess.py::test_preprocess_data` |
| `prep.dipole_key` | `--dipole_key` | `mace/tools/arg_parser.py:1304` | MERGE — `mace data prepare` reads the same config section as `mace train` instead of redeclaring the flag | `tests/workflows/test_preprocess_keys.py::test_a_property_key_is_read_under_the_name_it_is_given` |
| `prep.polarizability_key` | `--polarizability_key` | `mace/tools/arg_parser.py:1310` | MERGE — `mace data prepare` reads the same config section as `mace train` instead of redeclaring the flag | ⚠️ gap (preprocess path) |
| `prep.charges_key` | `--charges_key` | `mace/tools/arg_parser.py:1316` | MERGE — `mace data prepare` reads the same config section as `mace train` instead of redeclaring the flag | `tests/workflows/test_preprocess_keys.py::test_a_property_key_is_read_under_the_name_it_is_given` |
| `prep.head_key` | `--head_key` | `mace/tools/arg_parser.py:1368` | MERGE — `mace data prepare` reads the same config section as `mace train` instead of redeclaring the flag | `tests/workflows/test_preprocess.py` |
| `prep.heads` | `--heads` | `mace/tools/arg_parser.py:1374` | MERGE — `mace data prepare` reads the same config section as `mace train` instead of redeclaring the flag | `tests/workflows/test_preprocess.py` |
| `prep.atomic_numbers` | `--atomic_numbers` | `mace/tools/arg_parser.py:1322` | MERGE — `mace data prepare` reads the same config section as `mace train` instead of redeclaring the flag | `tests/workflows/test_preprocess.py` |
| `prep.batch_size` | `--batch_size` | `mace/tools/arg_parser.py:1335` | MERGE — `mace data prepare` reads the same config section as `mace train` instead of redeclaring the flag | `tests/workflows/test_preprocess.py` |
| `prep.scaling` | `--scaling` | `mace/tools/arg_parser.py:1342` | MERGE — `mace data prepare` reads the same config section as `mace train` instead of redeclaring the flag | `tests/workflows/test_preprocess.py` |
| `prep.E0s` | `--E0s` | `mace/tools/arg_parser.py:1349` | MERGE — `mace data prepare` reads the same config section as `mace train` instead of redeclaring the flag | `tests/workflows/test_preprocess.py` |
| `prep.seed` | `--seed` | `mace/tools/arg_parser.py:1362` | MERGE — `mace data prepare` reads the same config section as `mace train` instead of redeclaring the flag | `tests/workflows/test_preprocess.py` |
| `prep.num_process` | `--num_process` | `mace/tools/arg_parser.py:1239` | KEEP — preprocessing parallelism | `tests/workflows/test_preprocess.py` |
| `prep.h5_prefix` | `--h5_prefix` | `mace/tools/arg_parser.py:1265` | KEEP — the shard naming/output prefix | `tests/workflows/test_preprocess.py` |
| `prep.compute_statistics` | `--compute_statistics` | `mace/tools/arg_parser.py:1329` | KEEP — emits `statistics.json` | `tests/workflows/test_preprocess.py` |
| `prep.shuffle` | `--shuffle` | `mace/tools/arg_parser.py:1356` | KEEP | `tests/workflows/test_preprocess.py` |

## 5. Other CLI flags — 111 dests over 13 parsers

The thirteen argparsers under `mace/cli/`: seven user-facing CLIs (88 dests) and the six
`convert_*` weight/device converters (23 dests, 7 distinct). Three of the six have no console
entry point at all, so an extraction driven by `setup.cfg` misses them twice. 74 distinct dests
become 111 rows because a dest is counted once per parser that declares it — `--device` in four
CLIs is four knobs with four defaults.

### `mace_eval_configs` — `mace/cli/eval_configs.py` (19)

| id | feature | source | disposition | pinned by |
|---|---|---|---|---|
| `cli.eval_configs.configs` | `--configs` | `mace/cli/eval_configs.py:28` | KEEP | `tests/workflows/test_cli_contracts.py::test_eval_configs_reproduces_the_committed_anchor_reference` |
| `cli.eval_configs.model` | `--model` | `mace/cli/eval_configs.py:29` | KEEP | `tests/workflows/test_cli_contracts.py::test_eval_configs_reproduces_the_committed_anchor_reference` |
| `cli.eval_configs.output` | `--output` | `mace/cli/eval_configs.py:30` | KEEP | `tests/workflows/test_cli_contracts.py::test_eval_configs_reproduces_the_committed_anchor_reference` |
| `cli.eval_configs.device` | `--device` | `mace/cli/eval_configs.py:32` | KEEP | `tests/workflows/test_cli_contracts.py::test_eval_configs_agrees_with_the_ase_calculator_on_every_fixture` |
| `cli.eval_configs.default_dtype` | `--default_dtype` | `mace/cli/eval_configs.py:45` | MERGE — `PrecisionConfig` | `tests/workflows/test_cli_contracts.py::test_eval_at_float32_reproduces_float64_within_the_fp32_row` + `tests/workflows/test_eval_configs_shapes_and_dtype.py::test_a_dtype_that_disagrees_with_the_checkpoint_is_converted_not_fatal` |
| `cli.eval_configs.batch_size` | `--batch_size` | `mace/cli/eval_configs.py:51` | KEEP | `tests/workflows/test_cli_contracts.py::test_eval_batch_size_does_not_change_the_numbers` |
| `cli.eval_configs.compute_stress` | `--compute_stress` | `mace/cli/eval_configs.py:53` | MERGE — observable spec | `tests/workflows/test_cli_contracts.py::test_eval_compute_stress_writes_a_stress_and_omitting_it_does_not` |
| `cli.eval_configs.info_prefix` | `--info_prefix` | `mace/cli/eval_configs.py:101` | KEEP — prefixes every key written back into the XYZ (§12) | `tests/workflows/test_cli_contracts.py::test_eval_info_prefix_renames_every_written_key` |
| `cli.eval_configs.head` | `--head` | `mace/cli/eval_configs.py:107` | KEEP | `tests/workflows/test_cli_contracts.py::test_eval_head_selects_a_head_and_refuses_one_the_model_does_not_have` |
| `cli.eval_configs.enable_cueq` | `--enable_cueq` | `mace/cli/eval_configs.py:39` | MERGE — backend dispatch config | `tests/golden/test_backend_parity_golden.py::test_the_calculators_own_backend_flag_reaches_the_same_kernels` |
| `cli.eval_configs.return_contributions` | `--return_contributions` | `mace/cli/eval_configs.py:65` | KEEP — typed outputs make this natural | `tests/workflows/test_cli_contracts.py::test_eval_contributions_sum_to_the_total_energy_on_the_plain_model` + `tests/workflows/test_cli_contracts.py::test_eval_contributions_are_refused_for_the_scale_shift_model` |
| `cli.eval_configs.return_node_energies` | `--return_node_energies` | `mace/cli/eval_configs.py:95` | KEEP — idem | `tests/workflows/test_cli_contracts.py::test_eval_node_energies_sum_to_the_total_energy` + `tests/workflows/test_eval_configs_shapes_and_dtype.py::test_node_energies_are_written_per_structure_whatever_its_size` |
| `cli.eval_configs.compute_bec` | `--compute_bec` | `mace/cli/eval_configs.py:59` | KEEP — Born effective charges (IR spectra): a real physical observable of the polar model, and cheap because the derivative already exists | `tests/extensions/les/test_maceles.py::test_run_eval_with_bec` + `tests/extensions/les/test_maceles.py::test_run_eval_fail_with_wrong_model` |
| `cli.eval_configs.return_descriptors` | `--return_descriptors` | `mace/cli/eval_configs.py:71` | KEEP — descriptor extraction is `BaseMACE`'s raison d'être | `tests/workflows/test_cli_contracts.py::test_eval_descriptors_land_per_atom_and_the_aggregations_reduce_them` |
| `cli.eval_configs.descriptor_num_layers` | `--descriptor_num_layers` | `mace/cli/eval_configs.py:77` | KEEP — idem | `tests/workflows/test_cli_contracts.py::test_eval_descriptor_layer_and_invariant_flags_change_the_width` |
| `cli.eval_configs.descriptor_aggregation_method` | `--descriptor_aggregation_method` | `mace/cli/eval_configs.py:83` | KEEP — idem | `tests/workflows/test_cli_contracts.py::test_eval_descriptors_land_per_atom_and_the_aggregations_reduce_them` |
| `cli.eval_configs.descriptor_invariants_only` | `--descriptor_invariants_only` | `mace/cli/eval_configs.py:89` | KEEP — idem | `tests/workflows/test_cli_contracts.py::test_eval_descriptor_layer_and_invariant_flags_change_the_width` |
| `cli.eval_configs.magmom_key` | `--magmom_key` | `mace/cli/eval_configs.py:114` | MERGE — property-key convention | `tests/extensions/magnetic` |
| `cli.eval_configs.return_magforces` | `--return_magforces` | `mace/cli/eval_configs.py:121` | MERGE — observable spec (`dE/dm` declared like any other derivative) | `tests/extensions/magnetic` |

### `mace_finetuning_select` — `mace/cli/fine_tuning_select.py` (18)

| id | feature | source | disposition | pinned by |
|---|---|---|---|---|
| `cli.fine_tuning_select.configs_pt` | `--configs_pt` | `mace/cli/fine_tuning_select.py:94` | MERGE — absorbed into the integrated fine-tuning pipeline (`mace train` with a fine-tuning config); selection stops being a separate CLI over a separate model load | `tests/workflows/test_finetuning_contracts.py::test_filtering_type_restricts_the_pool_to_the_target_elements` |
| `cli.fine_tuning_select.configs_ft` | `--configs_ft` | `mace/cli/fine_tuning_select.py:99` | MERGE — absorbed into the integrated fine-tuning pipeline (`mace train` with a fine-tuning config); selection stops being a separate CLI over a separate model load | `tests/workflows/test_finetuning_select.py::test_select_samples_ft_provided` |
| `cli.fine_tuning_select.num_samples` | `--num_samples` | `mace/cli/fine_tuning_select.py:105` | MERGE — absorbed into the integrated fine-tuning pipeline (`mace train` with a fine-tuning config); selection stops being a separate CLI over a separate model load | `tests/workflows/test_finetuning_contracts.py::test_random_padding_tops_up_a_short_pool_and_disallowing_it_does_not` |
| `cli.fine_tuning_select.subselect` | `--subselect` | `mace/cli/fine_tuning_select.py:112` | MERGE — absorbed into the integrated fine-tuning pipeline (`mace train` with a fine-tuning config); selection stops being a separate CLI over a separate model load | `tests/workflows/test_finetuning_select.py::test_select_samples_random` + `tests/workflows/test_finetuning_contracts.py::test_subselect_fps_uses_the_model_and_still_returns_the_requested_number` |
| `cli.fine_tuning_select.model` | `--model` | `mace/cli/fine_tuning_select.py:119` | MERGE — absorbed into the integrated fine-tuning pipeline (`mace train` with a fine-tuning config); selection stops being a separate CLI over a separate model load | `tests/workflows/test_finetuning_select_cli.py::test_a_local_checkpoint_is_used_for_the_descriptors` |
| `cli.fine_tuning_select.output` | `--output` | `mace/cli/fine_tuning_select.py:121` | MERGE — absorbed into the integrated fine-tuning pipeline (`mace train` with a fine-tuning config); selection stops being a separate CLI over a separate model load | `tests/workflows/test_finetuning_select_cli.py::test_the_output_path_is_where_the_flag_says` |
| `cli.fine_tuning_select.descriptors` | `--descriptors` | `mace/cli/fine_tuning_select.py:123` | MERGE — absorbed into the integrated fine-tuning pipeline (`mace train` with a fine-tuning config); selection stops being a separate CLI over a separate model load | `tests/workflows/test_finetuning_select.py::test_load_descriptors` |
| `cli.fine_tuning_select.device` | `--device` | `mace/cli/fine_tuning_select.py:126` | MERGE — absorbed into the integrated fine-tuning pipeline (`mace train` with a fine-tuning config); selection stops being a separate CLI over a separate model load | `tests/workflows/test_finetuning_select_cli.py::test_a_device_outside_the_choices_is_rejected` |
| `cli.fine_tuning_select.head_pt` | `--head_pt` | `mace/cli/fine_tuning_select.py:140` | MERGE — absorbed into the integrated fine-tuning pipeline (`mace train` with a fine-tuning config); selection stops being a separate CLI over a separate model load | ⚠️ gap (idem) |
| `cli.fine_tuning_select.head_ft` | `--head_ft` | `mace/cli/fine_tuning_select.py:146` | MERGE — absorbed into the integrated fine-tuning pipeline (`mace train` with a fine-tuning config); selection stops being a separate CLI over a separate model load | ⚠️ gap (idem) |
| `cli.fine_tuning_select.filtering_type` | `--filtering_type` | `mace/cli/fine_tuning_select.py:152` | MERGE — absorbed into the integrated fine-tuning pipeline (`mace train` with a fine-tuning config); selection stops being a separate CLI over a separate model load | `tests/workflows/test_finetuning_contracts.py::test_filtering_type_restricts_the_pool_to_the_target_elements` + `tests/workflows/test_finetuning_select.py::test_filter_data` |
| `cli.fine_tuning_select.weight_ft` | `--weight_ft` | `mace/cli/fine_tuning_select.py:159` | MERGE — absorbed into the integrated fine-tuning pipeline (`mace train` with a fine-tuning config); selection stops being a separate CLI over a separate model load | ⚠️ gap (no test passes `--weight_ft` / `--weight_pt`) |
| `cli.fine_tuning_select.weight_pt` | `--weight_pt` | `mace/cli/fine_tuning_select.py:165` | MERGE — absorbed into the integrated fine-tuning pipeline (`mace train` with a fine-tuning config); selection stops being a separate CLI over a separate model load | ⚠️ gap (idem) |
| `cli.fine_tuning_select.filter_atomic_numbers_pt` | `--filter_atomic_numbers_pt` | `mace/cli/fine_tuning_select.py:171` | MERGE — absorbed into the integrated fine-tuning pipeline (`mace train` with a fine-tuning config); selection stops being a separate CLI over a separate model load | ⚠️ gap (idem) |
| `cli.fine_tuning_select.allow_random_padding` | `--disallow_random_padding` | `mace/cli/fine_tuning_select.py:177` | MERGE — absorbed into the integrated fine-tuning pipeline (`mace train` with a fine-tuning config); selection stops being a separate CLI over a separate model load | `tests/workflows/test_finetuning_contracts.py::test_random_padding_tops_up_a_short_pool_and_disallowing_it_does_not` |
| `cli.fine_tuning_select.seed` | `--seed` | `mace/cli/fine_tuning_select.py:182` | MERGE — absorbed into the integrated fine-tuning pipeline (`mace train` with a fine-tuning config); selection stops being a separate CLI over a separate model load | `tests/workflows/test_finetuning_select_cli.py::test_the_same_seed_selects_the_same_configurations` |
| `cli.fine_tuning_select.default_dtype` | `--default_dtype` | `mace/cli/fine_tuning_select.py:133` | MERGE — `PrecisionConfig` | `tests/workflows/test_finetuning_select_cli.py::test_the_descriptors_follow_the_requested_dtype` |
| `cli.fine_tuning_select.config` | `--config` | `mace/cli/fine_tuning_select.py:83` | MERGE — the v1 config system | `tests/workflows/test_finetuning_select_cli.py::test_the_yaml_config_is_read` |

### `mace_plot_train` — `mace/cli/plot_train.py` (8)

| id | feature | source | disposition | pinned by |
|---|---|---|---|---|
| `cli.plot_train.path` | `--path` | `mace/cli/plot_train.py:78` | KEEP | `tests/workflows/test_plot_train.py::test_a_directory_is_searched_for_logs` |
| `cli.plot_train.min_epoch` | `--min_epoch` | `mace/cli/plot_train.py:81` | KEEP | `tests/workflows/test_plot_train.py::test_each_flag_changes_the_plot` |
| `cli.plot_train.linear` | `--linear` | `mace/cli/plot_train.py:93` | KEEP | `tests/workflows/test_plot_train.py::test_each_flag_changes_the_plot` |
| `cli.plot_train.error_bars` | `--error_bars` | `mace/cli/plot_train.py:100` | KEEP | `tests/workflows/test_plot_train.py::test_each_flag_changes_the_plot` |
| `cli.plot_train.keys` | `--keys` | `mace/cli/plot_train.py:107` | KEEP | `tests/workflows/test_plot_train.py::test_each_flag_changes_the_plot` |
| `cli.plot_train.output_format` | `--output_format` | `mace/cli/plot_train.py:115` | KEEP | `tests/workflows/test_plot_train.py::test_the_output_format_is_honoured` |
| `cli.plot_train.heads` | `--heads` | `mace/cli/plot_train.py:123` | KEEP — per-head loss curves | `tests/workflows/test_plot_train.py::test_heads_names_the_plot_after_the_head` |
| `cli.plot_train.start_swa` | `--start_stage_two` `--start_swa` | `mace/cli/plot_train.py:84` | MERGE — stage boundaries are read from the run's per-stage schedule metadata; once stages are arbitrary a single 'stage two' marker no longer applies. Carries both `--start_stage_two` and the legacy `--start_swa` spelling | `tests/workflows/test_plot_train.py::test_start_stage_two_is_the_same_flag_as_start_swa` |

### `mace_active_learning_md` — `mace/cli/active_learning_md.py` (16)

| id | feature | source | disposition | pinned by |
|---|---|---|---|---|
| `cli.active_learning_md.config` | `--config` | `mace/cli/active_learning_md.py:20` | DROP — out of v1.0 scope: an ASE MD loop a user writes in ~30 lines over the calculator, at the cost of MACE owning thermostat, timestep and trajectory I/O. What MACE must guarantee is the committee variance in `calculate`, which stays (§16) | — |
| `cli.active_learning_md.config_index` | `--config_index` | `mace/cli/active_learning_md.py:22` | DROP — out of v1.0 scope: an ASE MD loop a user writes in ~30 lines over the calculator, at the cost of MACE owning thermostat, timestep and trajectory I/O. What MACE must guarantee is the committee variance in `calculate`, which stays (§16) | — |
| `cli.active_learning_md.error_threshold` | `--error_threshold` | `mace/cli/active_learning_md.py:25` | DROP — out of v1.0 scope: an ASE MD loop a user writes in ~30 lines over the calculator, at the cost of MACE owning thermostat, timestep and trajectory I/O. What MACE must guarantee is the committee variance in `calculate`, which stays (§16) | — |
| `cli.active_learning_md.temperature_K` | `--temperature_K` | `mace/cli/active_learning_md.py:27` | DROP — out of v1.0 scope: an ASE MD loop a user writes in ~30 lines over the calculator, at the cost of MACE owning thermostat, timestep and trajectory I/O. What MACE must guarantee is the committee variance in `calculate`, which stays (§16) | — |
| `cli.active_learning_md.friction` | `--friction` | `mace/cli/active_learning_md.py:28` | DROP — out of v1.0 scope: an ASE MD loop a user writes in ~30 lines over the calculator, at the cost of MACE owning thermostat, timestep and trajectory I/O. What MACE must guarantee is the committee variance in `calculate`, which stays (§16) | — |
| `cli.active_learning_md.timestep` | `--timestep` | `mace/cli/active_learning_md.py:29` | DROP — out of v1.0 scope: an ASE MD loop a user writes in ~30 lines over the calculator, at the cost of MACE owning thermostat, timestep and trajectory I/O. What MACE must guarantee is the committee variance in `calculate`, which stays (§16) | — |
| `cli.active_learning_md.nsteps` | `--nsteps` | `mace/cli/active_learning_md.py:30` | DROP — out of v1.0 scope: an ASE MD loop a user writes in ~30 lines over the calculator, at the cost of MACE owning thermostat, timestep and trajectory I/O. What MACE must guarantee is the committee variance in `calculate`, which stays (§16) | — |
| `cli.active_learning_md.nprint` | `--nprint` | `mace/cli/active_learning_md.py:32` | DROP — out of v1.0 scope: an ASE MD loop a user writes in ~30 lines over the calculator, at the cost of MACE owning thermostat, timestep and trajectory I/O. What MACE must guarantee is the committee variance in `calculate`, which stays (§16) | — |
| `cli.active_learning_md.nsave` | `--nsave` | `mace/cli/active_learning_md.py:35` | DROP — out of v1.0 scope: an ASE MD loop a user writes in ~30 lines over the calculator, at the cost of MACE owning thermostat, timestep and trajectory I/O. What MACE must guarantee is the committee variance in `calculate`, which stays (§16) | — |
| `cli.active_learning_md.ncheckerror` | `--ncheckerror` | `mace/cli/active_learning_md.py:38` | DROP — out of v1.0 scope: an ASE MD loop a user writes in ~30 lines over the calculator, at the cost of MACE owning thermostat, timestep and trajectory I/O. What MACE must guarantee is the committee variance in `calculate`, which stays (§16) | — |
| `cli.active_learning_md.model` | `--model` | `mace/cli/active_learning_md.py:42` | DROP — out of v1.0 scope: an ASE MD loop a user writes in ~30 lines over the calculator, at the cost of MACE owning thermostat, timestep and trajectory I/O. What MACE must guarantee is the committee variance in `calculate`, which stays (§16) | — |
| `cli.active_learning_md.output` | `--output` | `mace/cli/active_learning_md.py:47` | DROP — out of v1.0 scope: an ASE MD loop a user writes in ~30 lines over the calculator, at the cost of MACE owning thermostat, timestep and trajectory I/O. What MACE must guarantee is the committee variance in `calculate`, which stays (§16) | — |
| `cli.active_learning_md.device` | `--device` | `mace/cli/active_learning_md.py:49` | DROP — out of v1.0 scope: an ASE MD loop a user writes in ~30 lines over the calculator, at the cost of MACE owning thermostat, timestep and trajectory I/O. What MACE must guarantee is the committee variance in `calculate`, which stays (§16) | — |
| `cli.active_learning_md.default_dtype` | `--default_dtype` | `mace/cli/active_learning_md.py:56` | DROP — out of v1.0 scope: an ASE MD loop a user writes in ~30 lines over the calculator, at the cost of MACE owning thermostat, timestep and trajectory I/O. What MACE must guarantee is the committee variance in `calculate`, which stays (§16) | — |
| `cli.active_learning_md.compute_stress` | `--compute_stress` | `mace/cli/active_learning_md.py:63` | DROP — out of v1.0 scope: an ASE MD loop a user writes in ~30 lines over the calculator, at the cost of MACE owning thermostat, timestep and trajectory I/O. What MACE must guarantee is the committee variance in `calculate`, which stays (§16) | — |
| `cli.active_learning_md.info_prefix` | `--info_prefix` | `mace/cli/active_learning_md.py:69` | DROP — out of v1.0 scope: an ASE MD loop a user writes in ~30 lines over the calculator, at the cost of MACE owning thermostat, timestep and trajectory I/O. What MACE must guarantee is the committee variance in `calculate`, which stays (§16) | — |

### `mace_polar_density_cube` — `mace/cli/polar_density_cube.py` (18)

| id | feature | source | disposition | pinned by |
|---|---|---|---|---|
| `cli.polar_density_cube.configs` | `--configs` | `mace/cli/polar_density_cube.py:511` | KEEP — the electrostatics extra's density-cube tool | `tests/extensions/polar/test_polar_density_cube.py` (port cases) |
| `cli.polar_density_cube.model` | `--model` | `mace/cli/polar_density_cube.py:513` | KEEP — the electrostatics extra's density-cube tool | `tests/extensions/polar/test_polar_density_cube.py` (port cases) |
| `cli.polar_density_cube.output` | `--output` | `mace/cli/polar_density_cube.py:517` | KEEP — the electrostatics extra's density-cube tool | `tests/extensions/polar/test_polar_density_cube.py` (port cases) |
| `cli.polar_density_cube.index` | `--index` | `mace/cli/polar_density_cube.py:518` | KEEP — the electrostatics extra's density-cube tool | `tests/extensions/polar/test_polar_density_cube.py` (port cases) |
| `cli.polar_density_cube.quantity` | `--quantity` | `mace/cli/polar_density_cube.py:520` | KEEP — the electrostatics extra's density-cube tool | `tests/extensions/polar/test_polar_density_cube.py` (port cases) |
| `cli.polar_density_cube.grid` | `--grid` | `mace/cli/polar_density_cube.py:526` | KEEP — the electrostatics extra's density-cube tool | `tests/extensions/polar/test_polar_density_cube.py` (port cases) |
| `cli.polar_density_cube.device` | `--device` | `mace/cli/polar_density_cube.py:533` | KEEP — the electrostatics extra's density-cube tool | `tests/extensions/polar/test_polar_density_cube.py` (port cases) |
| `cli.polar_density_cube.sigma` | `--sigma` | `mace/cli/polar_density_cube.py:537` | KEEP — the electrostatics extra's density-cube tool | `tests/extensions/polar/test_polar_density_cube.py` (port cases) |
| `cli.polar_density_cube.kspace_cutoff` | `--kspace_cutoff` | `mace/cli/polar_density_cube.py:538` | KEEP — the electrostatics extra's density-cube tool | `tests/extensions/polar/test_polar_density_cube.py` (port cases) |
| `cli.polar_density_cube.backend` | `--backend` | `mace/cli/polar_density_cube.py:540` | KEEP — the electrostatics extra's density-cube tool | `tests/extensions/polar/test_polar_density_cube.py` (port cases) |
| `cli.polar_density_cube.realspace_cutoff_factor` | `--realspace_cutoff_factor` | `mace/cli/polar_density_cube.py:546` | KEEP — the electrostatics extra's density-cube tool | `tests/extensions/polar/test_polar_density_cube.py` (port cases) |
| `cli.polar_density_cube.chunk_size` | `--chunk_size` | `mace/cli/polar_density_cube.py:552` | KEEP — the electrostatics extra's density-cube tool | `tests/extensions/polar/test_polar_density_cube.py` (port cases) |
| `cli.polar_density_cube.subtract_total_charge` | `--subtract_total_charge` | `mace/cli/polar_density_cube.py:558` | KEEP — the electrostatics extra's density-cube tool | `tests/extensions/polar/test_polar_density_cube.py` (port cases) |
| `cli.polar_density_cube.external_field` | `--external_field` | `mace/cli/polar_density_cube.py:563` | KEEP — the electrostatics extra's density-cube tool | `tests/extensions/polar/test_polar_density_cube.py` (port cases) |
| `cli.polar_density_cube.fermi_level` | `--fermi_level` | `mace/cli/polar_density_cube.py:570` | KEEP — the electrostatics extra's density-cube tool | `tests/extensions/polar/test_polar_density_cube.py` (port cases) |
| `cli.polar_density_cube.write_potential` | `--write_potential` | `mace/cli/polar_density_cube.py:576` | KEEP — the electrostatics extra's density-cube tool | `tests/extensions/polar/test_polar_density_cube.py` (port cases) |
| `cli.polar_density_cube.quality_report` | `--quality_report` | `mace/cli/polar_density_cube.py:581` | KEEP — the electrostatics extra's density-cube tool | `tests/extensions/polar/test_polar_density_cube.py` (port cases) |
| `cli.polar_density_cube.default_dtype` | `--default_dtype` | `mace/cli/polar_density_cube.py:535` | MERGE — `PrecisionConfig` | `tests/extensions/polar/test_polar_density_cube.py` |

### `mace_create_lammps_model` — `mace/cli/create_lammps_model.py` (4)

| id | feature | source | disposition | pinned by |
|---|---|---|---|---|
| `cli.create_lammps_model.model_path` | `model_path` | `mace/cli/create_lammps_model.py:21` | KEEP — positional; becomes the checkpoint argument of `mace export lammps` | `tests/integrations/lammps/test_export_golden.py::test_the_exported_artifact_reproduces_the_committed_numbers` |
| `cli.create_lammps_model.head` | `--head` | `mace/cli/create_lammps_model.py:26` | KEEP | `tests/workflows/test_cli_contracts.py::test_select_head_writes_a_single_head_model_to_the_default_name` |
| `cli.create_lammps_model.dtype` | `--dtype` | `mace/cli/create_lammps_model.py:33` | MERGE — `PrecisionConfig` of the export bundle | `tests/integrations/lammps/test_export_golden.py::test_the_float32_export_is_single_precision` |
| `cli.create_lammps_model.format` | `--format` | `mace/cli/create_lammps_model.py:40` | MERGE — v1 exports the MLIAP bundle only; the default TorchScript format is dropped with `jit.script`, so the choice collapses to one and the flag with it | `tests/integrations/lammps/test_export_golden.py::test_the_mliap_export_declares_the_committed_interface` + `tests/integrations/lammps/test_export_golden.py::test_the_mliap_export_refuses_a_multilayer_model_without_ghost_exchange` |

### `mace_select_head` — `mace/cli/select_head.py` (5)

| id | feature | source | disposition | pinned by |
|---|---|---|---|---|
| `cli.select_head.model_file` | `model_file` | `mace/cli/select_head.py:33` | KEEP — positional | `tests/workflows/test_cli_contracts.py::test_select_head_writes_a_single_head_model_to_the_default_name` |
| `cli.select_head.head_name` | `--head_name` `-n` | `mace/cli/select_head.py:12` | KEEP | `tests/workflows/test_cli_contracts.py::test_select_head_and_the_multihead_model_agree_on_the_selected_head` |
| `cli.select_head.list_heads` | `--list_heads` `-l` | `mace/cli/select_head.py:18` | KEEP | `tests/workflows/test_cli_contracts.py::test_select_head_lists_the_heads_of_a_multihead_model` |
| `cli.select_head.target_device` | `--target_device` `-d` | `mace/cli/select_head.py:24` | KEEP | `tests/workflows/test_cli_contracts.py::test_select_head_honours_output_file_and_target_device` |
| `cli.select_head.output_file` | `--output_file` `-o` | `mace/cli/select_head.py:29` | KEEP | `tests/workflows/test_cli_contracts.py::test_select_head_honours_output_file_and_target_device` |

### `mace_e3nn_cueq` — `mace/cli/convert_e3nn_cueq.py` (4)

| id | feature | source | disposition | pinned by |
|---|---|---|---|---|
| `cli.convert_e3nn_cueq.input_model` | `input_model` | `mace/cli/convert_e3nn_cueq.py:280` | DROP — v1 weights are canonical and backend dispatch is automatic, so there is nothing left to convert between backends (positional) | `tests/golden/test_backend_parity_golden.py::test_converted_model_reproduces_the_committed_cpu_reference` |
| `cli.convert_e3nn_cueq.output_model` | `--output_model` | `mace/cli/convert_e3nn_cueq.py:282` | DROP — v1 weights are canonical and backend dispatch is automatic, so there is nothing left to convert between backends | `tests/golden/test_backend_parity_golden.py::test_converted_model_reproduces_the_committed_cpu_reference` |
| `cli.convert_e3nn_cueq.device` | `--device` | `mace/cli/convert_e3nn_cueq.py:286` | DROP — v1 weights are canonical and backend dispatch is automatic, so there is nothing left to convert between backends | `tests/golden/test_backend_parity_golden.py::test_converting_for_the_cpu_leaves_cueq_unfused_and_the_audit_says_so` |
| `cli.convert_e3nn_cueq.return_model` | `--return_model` | `mace/cli/convert_e3nn_cueq.py:288` | DROP — v1 weights are canonical and backend dispatch is automatic, so there is nothing left to convert between backends (library flag: return the converted model instead of writing it) | `tests/backends/backend_parity.py::test_bidirectional_conversion` |

### `mace_cueq_to_e3nn` — `mace/cli/convert_cueq_e3nn.py` (4)

| id | feature | source | disposition | pinned by |
|---|---|---|---|---|
| `cli.convert_cueq_e3nn.input_model` | `input_model` | `mace/cli/convert_cueq_e3nn.py:282` | DROP — v1 weights are canonical and backend dispatch is automatic, so there is nothing left to convert between backends (positional) | `tests/backends/backend_parity.py::test_bidirectional_conversion` |
| `cli.convert_cueq_e3nn.output_model` | `--output_model` | `mace/cli/convert_cueq_e3nn.py:284` | DROP — v1 weights are canonical and backend dispatch is automatic, so there is nothing left to convert between backends | `tests/backends/backend_parity.py::test_bidirectional_conversion` |
| `cli.convert_cueq_e3nn.device` | `--device` | `mace/cli/convert_cueq_e3nn.py:286` | DROP — v1 weights are canonical and backend dispatch is automatic, so there is nothing left to convert between backends | `tests/backends/backend_parity.py::test_bidirectional_conversion` |
| `cli.convert_cueq_e3nn.return_model` | `--return_model` | `mace/cli/convert_cueq_e3nn.py:288` | DROP — v1 weights are canonical and backend dispatch is automatic, so there is nothing left to convert between backends (library flag: return the converted model instead of writing it) | `tests/backends/backend_parity.py::test_bidirectional_conversion` |

### `mace/cli/convert_e3nn_oeq.py` — no entry point (4)

| id | feature | source | disposition | pinned by |
|---|---|---|---|---|
| `cli.convert_e3nn_oeq.input_model` | `input_model` | `mace/cli/convert_e3nn_oeq.py:67` | DROP — v1 weights are canonical and backend dispatch is automatic, so there is nothing left to convert between backends (positional) | `tests/golden/test_backend_parity_golden.py::test_the_audit_accepts_a_well_formed_oeq_conversion_and_only_that` |
| `cli.convert_e3nn_oeq.output_model` | `--output_model` | `mace/cli/convert_e3nn_oeq.py:69` | DROP — v1 weights are canonical and backend dispatch is automatic, so there is nothing left to convert between backends | `tests/golden/test_backend_parity_golden.py::test_the_audit_accepts_a_well_formed_oeq_conversion_and_only_that` |
| `cli.convert_e3nn_oeq.device` | `--device` | `mace/cli/convert_e3nn_oeq.py:73` | DROP — v1 weights are canonical and backend dispatch is automatic, so there is nothing left to convert between backends | `tests/golden/test_backend_parity_golden.py::test_the_audit_accepts_a_well_formed_oeq_conversion_and_only_that` |
| `cli.convert_e3nn_oeq.return_model` | `--return_model` | `mace/cli/convert_e3nn_oeq.py:75` | DROP — v1 weights are canonical and backend dispatch is automatic, so there is nothing left to convert between backends (library flag: return the converted model instead of writing it) | `tests/golden/test_backend_parity_golden.py::test_the_audit_accepts_a_well_formed_oeq_conversion_and_only_that` |

### `mace/cli/convert_oeq_e3nn.py` — no entry point (4)

| id | feature | source | disposition | pinned by |
|---|---|---|---|---|
| `cli.convert_oeq_e3nn.input_model` | `input_model` | `mace/cli/convert_oeq_e3nn.py:57` | DROP — v1 weights are canonical and backend dispatch is automatic, so there is nothing left to convert between backends (positional) | `tests/golden/test_backend_parity_golden.py::test_the_audit_fails_a_site_that_is_installed_and_never_called` |
| `cli.convert_oeq_e3nn.output_model` | `--output_model` | `mace/cli/convert_oeq_e3nn.py:59` | DROP — v1 weights are canonical and backend dispatch is automatic, so there is nothing left to convert between backends | `tests/golden/test_backend_parity_golden.py::test_the_audit_fails_a_site_that_is_installed_and_never_called` |
| `cli.convert_oeq_e3nn.device` | `--device` | `mace/cli/convert_oeq_e3nn.py:61` | DROP — v1 weights are canonical and backend dispatch is automatic, so there is nothing left to convert between backends | `tests/golden/test_backend_parity_golden.py::test_the_audit_fails_a_site_that_is_installed_and_never_called` |
| `cli.convert_oeq_e3nn.return_model` | `--return_model` | `mace/cli/convert_oeq_e3nn.py:63` | DROP — v1 weights are canonical and backend dispatch is automatic, so there is nothing left to convert between backends (library flag: return the converted model instead of writing it) | `tests/golden/test_backend_parity_golden.py::test_the_audit_fails_a_site_that_is_installed_and_never_called` |

### `mace/cli/convert_e3nn_hybrid.py` — no entry point (4)

| id | feature | source | disposition | pinned by |
|---|---|---|---|---|
| `cli.convert_e3nn_hybrid.input_model` | `input_model` | `mace/cli/convert_e3nn_hybrid.py:141` | DROP — v1 weights are canonical and backend dispatch is automatic, so there is nothing left to convert between backends (positional) | `tests/golden/test_backend_parity_golden.py::test_the_conversion_whitelist_refuses_the_plain_anchor_and_both_converters_stop` |
| `cli.convert_e3nn_hybrid.output_model` | `--output_model` | `mace/cli/convert_e3nn_hybrid.py:143` | DROP — v1 weights are canonical and backend dispatch is automatic, so there is nothing left to convert between backends | `tests/golden/test_backend_parity_golden.py::test_the_conversion_whitelist_refuses_the_plain_anchor_and_both_converters_stop` |
| `cli.convert_e3nn_hybrid.device` | `--device` | `mace/cli/convert_e3nn_hybrid.py:147` | DROP — v1 weights are canonical and backend dispatch is automatic, so there is nothing left to convert between backends | `tests/golden/test_backend_parity_golden.py::test_the_conversion_whitelist_refuses_the_plain_anchor_and_both_converters_stop` |
| `cli.convert_e3nn_hybrid.return_model` | `--return_model` | `mace/cli/convert_e3nn_hybrid.py:148` | DROP — v1 weights are canonical and backend dispatch is automatic, so there is nothing left to convert between backends (library flag: return the converted model instead of writing it) | `tests/golden/test_backend_parity_golden.py::test_the_conversion_whitelist_refuses_the_plain_anchor_and_both_converters_stop` |

### `mace_convert_device` — `mace/cli/convert_device.py` (3)

| id | feature | source | disposition | pinned by |
|---|---|---|---|---|
| `cli.convert_device.model_file` | `model_file` | `mace/cli/convert_device.py:19` | KEEP — positional | `tests/unit/test_scale_shift_dtype.py::test_the_convert_device_cli_preserves_the_buffers` |
| `cli.convert_device.output_file` | `--output_file` `-o` | `mace/cli/convert_device.py:15` | KEEP | `tests/unit/test_scale_shift_dtype.py::test_the_convert_device_cli_preserves_the_buffers` |
| `cli.convert_device.target_device` | `--target_device` `-t` | `mace/cli/convert_device.py:9` | KEEP — converts device/dtype, not backend layout, which is why this CLI is explicitly not one of the five weight converters above | `tests/workflows/test_cli_contracts.py::test_select_head_honours_output_file_and_target_device` |

## 6. Model-level classes (12)

Every top-level class in `mace/modules/models.py` and `mace/modules/extensions.py`. Two of the
twelve are **not models** — `SHModule` and `ChebyshevBasisGeneral` are blocks that happen to live
in `extensions.py`, which is what the extractor scans; they are listed rather than filtered out so
the set stays mechanical.

| id | feature | source | disposition | pinned by |
|---|---|---|---|---|
| `model.MACE` | `MACE` | `mace/modules/models.py:47` | MERGE — `BaseMACE` + a declared energy output; the class as a class disappears | `tests/golden/test_tiny_anchors.py::test_anchor_is_the_class_it_claims_to_be` + `tests/golden/test_tiny_anchors.py::test_anchor_reproduces_its_reference` |
| `model.ScaleShiftMACE` | `ScaleShiftMACE` | `mace/modules/models.py:444` | MERGE — idem; the default energy model becomes the default configuration | `tests/golden/test_tiny_anchors.py::test_the_repulsion_term_is_scaled_in_one_class_and_raw_in_the_other` |
| `model.AtomicDipolesMACE` | `AtomicDipolesMACE` | `mace/modules/models.py:626` | MERGE — the dipole observable | `tests/golden/test_tiny_dipoles.py::test_anchor_reproduces_its_reference` |
| `model.AtomicDielectricMACE` | `AtomicDielectricMACE` | `mace/modules/models.py:842` | MERGE — dipole + polarizability observables. Note this is the MACE-MDP foundation architecture, so it needs a converter as well as a reimplementation | `tests/golden/test_mdp_foundation.py::test_mdp_foundation_reproduces_its_reference` |
| `model.EnergyDipolesMACE` | `EnergyDipolesMACE` | `mace/modules/models.py:1199` | MERGE — energy + dipole observables | `tests/unit/test_models.py::test_energy_dipole_mace` |
| `model.MACELES` | `MACELES` | `mace/modules/extensions.py:142` | KEEP — the LES extra: latent multipoles, BEC and the external-field path | `tests/extensions/les` + `tests/golden/test_tiny_maceles.py::test_the_model_surface_reproduces_its_reference` |
| `model.PolarMACE` | `PolarMACE` | `mace/modules/extensions.py:663` | KEEP — the electrostatics extra | `tests/golden/test_polar_foundation.py::test_polar_foundation_reproduces_its_reference` + `tests/golden/test_polar_foundation.py::test_polar_mace_emits_no_polarizability` |
| `model.MagneticMACE` | `MagneticMACE` | `mace/modules/extensions.py:1428` | KEEP — the magnetic base class: magmom as an input feature, magnetic-moment observable, `dE/dm` derivative | `tests/extensions/magnetic` (rotation equivariance, inversion parity) + `tests/golden/test_tiny_magnetic.py::test_a_joint_rotation_is_a_symmetry_and_a_spin_only_one_is_not` |
| `model.MagneticScaleShiftMACE` | `MagneticScaleShiftMACE` | `mace/modules/extensions.py:1706` | KEEP — the CLI-reachable magnetic model | `tests/extensions/magnetic` (e2e train, eval, config round-trip) + `tests/golden/test_tiny_magnetic.py::test_the_eval_cli_reproduces_the_reference_including_magforces` |
| `model.MagneticSCFMACE` | `MagneticSCFMACE` | `mace/modules/extensions.py:1968` | KEEP — **not CLI-reachable**: a wrapper applied programmatically over a model (`MagneticSCFMACE(model=…, n_scf_step=2)`). That shape is what a model-transform hook has to support, so it is the in-tree consumer to design one against | `tests/extensions/magnetic::test_run_magnetic_scf` |
| `model.TimeReversalSymmetrizedMACE` | `TimeReversalSymmetrizedMACE` | `mace/modules/extensions.py:2108` | KEEP — **not CLI-reachable**: like `MagneticSCFMACE`, a wrapper applied programmatically, and the second in-tree consumer for a model-transform hook. It averages the wrapped model over `m` and `-m`, so time-reversal symmetry is exact rather than learned from augmentation | `tests/extensions/magnetic/test_time_reversal.py::test_energy_is_invariant_under_global_moment_reversal` |
| `model.SHModule` | `SHModule` | `mace/modules/extensions.py:1351` | KEEP — a spherical-harmonics block wrapping `sphericart.torch.SolidHarmonics`, not a model. Notable as a working in-tree precedent for a non-e3nn spherical-harmonics backend | `tests/extensions/magnetic` (indirect) |
| `model.ChebyshevBasisGeneral` | `ChebyshevBasisGeneral` | `mace/modules/extensions.py:1374` | KEEP — a radial basis living in `extensions.py`, not a model; belongs with the `--radial_type` bases of §3.2 | `tests/unit/test_radial.py::test_chebychev_basis_values` + `tests/unit/test_radial.py::test_chebychev_ignores_r_max_and_diverges_outside_the_unit_interval` |

## 7. String→class registries (21)

The four dicts in `mace/modules/__init__.py` that connect CLI values to implementations. An entry
that is not here is not reachable from the CLI, so this set is exactly the user-selectable block
surface.

| id | feature | source | disposition | pinned by |
|---|---|---|---|---|
| `reg.RealAgnosticResidualInteractionBlock` | `RealAgnosticResidualInteractionBlock` — interaction_classes | `mace/modules/__init__.py:71` | KEEP — the standard interaction block | `tests/golden/test_tiny_anchors.py::test_anchor_reproduces_its_reference` |
| `reg.RealAgnosticInteractionBlock` | `RealAgnosticInteractionBlock` — interaction_classes | `mace/modules/__init__.py:73` | KEEP — the default first layer | `tests/golden/test_tiny_anchors.py::test_anchor_reproduces_its_reference` |
| `reg.RealAgnosticDensityInteractionBlock` | `RealAgnosticDensityInteractionBlock` — interaction_classes | `mace/modules/__init__.py:74` | KEEP — foundation-model architecture, not a research variant: MACE-MP-0b2 S/M/L and MACE-mh-0 use it in the first layer | ⚠️ gap (needs an MP-0b2 or mh-0 case in `tests/golden/test_foundation_goldens.py`) |
| `reg.RealAgnosticDensityResidualInteractionBlock` | `RealAgnosticDensityResidualInteractionBlock` — interaction_classes | `mace/modules/__init__.py:75` | KEEP — idem, used in the remaining layers of the same published models | `tests/integrations/lammps/test_mliap_exchange.py::test_mliap_exchange_density_residual` |
| `reg.RealAgnosticResidualNonLinearInteractionBlock` | `RealAgnosticResidualNonLinearInteractionBlock` — interaction_classes | `mace/modules/__init__.py:76` | KEEP — the interaction block of MACE-Polar S/M/L | `tests/unit/test_models.py::test_non_linear_first_interaction_block_runs_and_is_equivariant` + `tests/unit/test_models.py::test_non_linear_first_interaction_block_cannot_be_torchscripted` |
| `reg.RealAgnosticAttResidualInteractionBlock` | `RealAgnosticAttResidualInteractionBlock` — interaction_classes | `mace/modules/__init__.py:72` | DROP — unlike the Density blocks it appears in no `finetuning_utils` branch and no converter, only in the registry and the parser choices: a research variant with no published model, no test and no owner | — |
| `reg.MagneticRealAgnosticSpinOrbitCoupledDensityInteractionBlock` | `MagneticRealAgnosticSpinOrbitCoupledDensityInteractionBlock` — interaction_classes | `mace/modules/__init__.py:78` | KEEP — the magnetic extra's first layer; a Density variant, so it inherits whatever the rewrite does for that family | `tests/extensions/magnetic` |
| `reg.MagneticRealAgnosticResidueSpinOrbitCoupledDensityInteractionBlock` | `MagneticRealAgnosticResidueSpinOrbitCoupledDensityInteractionBlock` — interaction_classes | `mace/modules/__init__.py:77` | KEEP — idem, the residual variant | `tests/extensions/magnetic` |
| `reg.LinearReadoutBlock` | `LinearReadoutBlock` — readout_classes | `mace/modules/__init__.py:82` | KEEP | `tests/golden/test_tiny_anchors.py::test_anchor_reproduces_its_reference` |
| `reg.NonLinearReadoutBlock` | `NonLinearReadoutBlock` — readout_classes | `mace/modules/__init__.py:85` | KEEP | `tests/golden/test_tiny_anchors.py::test_anchor_reproduces_its_reference` |
| `reg.LinearDipoleReadoutBlock` | `LinearDipoleReadoutBlock` — readout_classes | `mace/modules/__init__.py:83` | MERGE — an observable head declared in the output spec | `tests/golden/test_tiny_dipoles.py::test_anchor_reproduces_its_reference` |
| `reg.NonLinearDipoleReadoutBlock` | `NonLinearDipoleReadoutBlock` — readout_classes | `mace/modules/__init__.py:84` | MERGE — idem | `tests/golden/test_tiny_dipoles.py::test_anchor_reproduces_its_reference` |
| `reg.NonLinearBiasReadoutBlock` | `NonLinearBiasReadoutBlock` — readout_classes | `mace/modules/__init__.py:86` | KEEP — a readout of the published MACE-Polar models, and it backs the fukui source map; not a registry leftover | `tests/golden/test_mdp_foundation.py::test_mdp_foundation_reproduces_its_reference` |
| `reg.GeneralNonLinearBiasReadoutBlock` | `GeneralNonLinearBiasReadoutBlock` — readout_classes | `mace/modules/__init__.py:87` | KEEP — used internally by `field_blocks.py`, load-bearing for the polar model | `tests/golden/test_mdp_foundation.py::test_mdp_foundation_reproduces_its_reference` |
| `reg.std_scaling` | `std_scaling` — scaling_classes | `mace/modules/__init__.py:91` | KEEP | `tests/unit/test_e0s_characterization.py::test_the_mean_and_std_scaling_is_the_per_atom_interaction_energy` |
| `reg.rms_forces_scaling` | `rms_forces_scaling` — scaling_classes | `mace/modules/__init__.py:92` | KEEP — the default | `tests/unit/test_e0s_characterization.py::test_the_rms_forces_scaling_is_the_root_mean_square_force_component` |
| `reg.rms_dipoles_scaling` | `rms_dipoles_scaling` — scaling_classes | `mace/modules/__init__.py:93` | MERGE — observable normalization | `tests/unit/test_e0s_characterization.py::test_the_dipole_scaling_entry_cannot_be_used_the_way_the_other_two_are` |
| `reg.silu` | `silu` — gate_dict | `mace/modules/__init__.py:99` | KEEP — the default gate | `tests/unit/test_gate.py::test_custom_activations` + `tests/unit/test_gate.py::test_forward_matches_e3nn` |
| `reg.tanh` | `tanh` — gate_dict | `mace/modules/__init__.py:98` | KEEP | `tests/unit/test_gate.py::test_custom_activations` |
| `reg.abs` | `abs` — gate_dict | `mace/modules/__init__.py:97` | KEEP | `tests/unit/test_gate_registry.py::test_the_registry_maps_every_cli_value_to_its_callable` |
| `reg.None` | `None` — gate_dict | `mace/modules/__init__.py:100` | KEEP — the string `"None"`, meaning no gate | `tests/unit/test_gate_registry.py::test_a_model_builds_and_runs_with_every_gate` |


## 7b. Blocks, radial bases, contractions, transforms and calculator classes (46)

The classes the registries do not account for. A registry entry is the *string* a
user passes; these are the classes themselves, and one added without a registry
entry is still reachable from Python and from a checkpoint. Adding these five sets
to the checker was prompted by a mutation test: a new class in any of these files
used to pass the gate in silence.

| id | feature | source | disposition | pinned by |
| --- | --- | --- | --- | --- |
| `radial.BesselBasis` | `BesselBasis` | `mace/modules/radial.py:18` | KEEP — the default radial basis | `tests/unit/test_modules.py::test_bessel_basis` |
| `radial.ChebychevBasis` | `ChebychevBasis` | `mace/modules/radial.py:61` | KEEP — `--radial_type chebyshev` | `tests/unit/test_radial.py::test_chebychev_basis_values` |
| `radial.GaussianBasis` | `GaussianBasis` | `mace/modules/radial.py:89` | KEEP — `--radial_type gaussian` | `tests/unit/test_radial.py::test_gaussian_basis_values` |
| `radial.PolynomialCutoff` | `PolynomialCutoff` | `mace/modules/radial.py:113` | KEEP — the envelope every basis is multiplied by | `tests/unit/test_modules.py::test_polynomial_cutoff` |
| `radial.ZBLBasis` | `ZBLBasis` | `mace/modules/radial.py:149` | KEEP — `--pair_repulsion`, a physics term rather than a basis | `tests/unit/test_radial.py::test_zbl_buffers_and_trainability` |
| `radial.AgnesiTransform` | `AgnesiTransform` | `mace/modules/radial.py:225` | KEEP — `--distance_transform agnesi` | `tests/unit/test_radial.py::test_agnesi_transform_values` |
| `radial.SoftTransform` | `SoftTransform` | `mace/modules/radial.py:285` | KEEP — `--distance_transform soft` | `tests/unit/test_radial.py::test_soft_transform_values` |
| `radial.RadialMLP` | `RadialMLP` | `mace/modules/radial.py:361` | KEEP — the MLP over the basis, sized by `--radial_MLP` | `tests/unit/test_radial.py::test_radial_mlp_structure_and_shapes` |
| `block.LinearNodeEmbeddingBlock` | `LinearNodeEmbeddingBlock` | `mace/modules/blocks.py:45` | KEEP — one-hot elements to node features, in every model | `tests/unit/test_models.py::test_mace` |
| `block.RadialEmbeddingBlock` | `RadialEmbeddingBlock` | `mace/modules/blocks.py:395` | KEEP — basis x cutoff, in every model | `tests/unit/test_radial.py::test_the_basis_sees_the_transformed_lengths` |
| `block.AtomicEnergiesBlock` | `AtomicEnergiesBlock` | `mace/modules/blocks.py:364` | KEEP — the E0 term | `tests/unit/test_modules.py::test_atomic_energies` |
| `block.ScaleShiftBlock` | `ScaleShiftBlock` | `mace/modules/blocks.py:1942` | KEEP — the dataset scale and shift | `tests/unit/test_scale_shift_dtype.py::test_the_scale_shift_buffers_are_also_frozen_at_construction` |
| `block.EquivariantProductBasisBlock` | `EquivariantProductBasisBlock` | `mace/modules/blocks.py:440` | KEEP — the many-body product basis | `tests/golden/test_tiny_dipoles.py::test_the_committed_anchor_carries_the_plain_e3nn_basis` |
| `block.InteractionBlock` | `InteractionBlock` | `mace/modules/blocks.py:639` | KEEP — the abstract base every interaction subclasses | `tests/unit/test_models.py::test_mace` |
| `block.GatedEquivariantBlock` | `GatedEquivariantBlock` | `mace/modules/gate.py:39` | KEEP — the gate the nonlinear readouts apply | `tests/unit/test_gate.py::test_forward_matches_e3nn` |
| `block.LinearReadoutBlock` | `LinearReadoutBlock` | `mace/modules/blocks.py:65` | KEEP — the per-layer site-energy readout | `tests/unit/test_models.py::test_mace` |
| `block.NonLinearReadoutBlock` | `NonLinearReadoutBlock` | `mace/modules/blocks.py:87` | KEEP — the last-layer readout when `--MLP_irreps` is set | `tests/unit/test_models.py::test_mace` |
| `block.LinearDipoleReadoutBlock` | `LinearDipoleReadoutBlock` | `mace/modules/blocks.py:163` | KEEP — the dipole family's readout | `tests/unit/test_models.py::test_dipole_mace` |
| `block.NonLinearDipoleReadoutBlock` | `NonLinearDipoleReadoutBlock` | `mace/modules/blocks.py:185` | KEEP — idem, nonlinear | `tests/unit/test_models.py::test_dipole_mace` |
| `block.LinearDipolePolarReadoutBlock` | `LinearDipolePolarReadoutBlock` | `mace/modules/blocks.py:233` | KEEP — dipole and polarizability together | `tests/unit/test_models.py::test_dipole_polar_mace` |
| `block.NonLinearDipolePolarReadoutBlock` | `NonLinearDipolePolarReadoutBlock` | `mace/modules/blocks.py:262` | KEEP — idem, nonlinear | `tests/unit/test_models.py::test_dipole_polar_mace` |
| `block.LinearLesReadoutBlock` | `LinearLesReadoutBlock` | `mace/modules/blocks.py:1974` | KEEP — the LES latent-multipole readout | `tests/extensions/les/test_maceles.py::test_les_readout_equivariance` |
| `block.NonLinearLesReadoutBlock` | `NonLinearLesReadoutBlock` | `mace/modules/blocks.py:2084` | KEEP — idem, nonlinear | `tests/extensions/les/test_maceles.py::test_les_readout_equivariance` |
| `block.NonLinearBiasReadoutBlock` | `NonLinearBiasReadoutBlock` | `mace/modules/blocks.py:123` | KEEP — the biased readout the field models use | ⚠️ gap (registered in `readout_classes` and used from `mace/modules/extensions.py`; no test names it) |
| `block.GeneralNonLinearBiasReadoutBlock` | `GeneralNonLinearBiasReadoutBlock` | `mace/modules/blocks.py:314` | KEEP — idem, for `mace/modules/field_blocks.py` | ⚠️ gap (registered and reachable; no test names it) |
| `block.RealAgnosticInteractionBlock` | `RealAgnosticInteractionBlock` | `mace/modules/blocks.py:835` | MERGE — one of five interaction variants that collapse into a configured convolution | `tests/backends/backend_parity.py::test_bidirectional_conversion` |
| `block.RealAgnosticResidualInteractionBlock` | `RealAgnosticResidualInteractionBlock` | `mace/modules/blocks.py:938` | KEEP — the default | `tests/unit/test_models.py::test_mace` |
| `block.RealAgnosticDensityInteractionBlock` | `RealAgnosticDensityInteractionBlock` | `mace/modules/blocks.py:1041` | MERGE — idem | `tests/backends/backend_parity.py::test_bidirectional_conversion` |
| `block.RealAgnosticDensityResidualInteractionBlock` | `RealAgnosticDensityResidualInteractionBlock` | `mace/modules/blocks.py:1162` | MERGE — idem | `tests/integrations/lammps/test_mliap_exchange.py::test_mliap_exchange_density_residual` |
| `block.RealAgnosticResidualNonLinearInteractionBlock` | `RealAgnosticResidualNonLinearInteractionBlock` | `mace/modules/blocks.py:1412` | KEEP — the PolarMACE interaction | `tests/backends/backend_parity.py::test_bidirectional_conversion` |
| `block.RealAgnosticAttResidualInteractionBlock` | `RealAgnosticAttResidualInteractionBlock` | `mace/modules/blocks.py:1286` | MERGE — idem | ⚠️ gap (registered in `interaction_classes`, so reachable from `--interaction`; nothing builds it) |
| `block.MagneticInteractionBlock` | `MagneticInteractionBlock` | `mace/modules/blocks.py:1602` | KEEP — the magnetic interactions' base class | `tests/extensions/magnetic/test_magmace.py::test_run_train_magnetic_mace` |
| `block.MagneticRealAgnosticSpinOrbitCoupledDensityInteractionBlock` | `MagneticRealAgnosticSpinOrbitCoupledDensityInteractionBlock` | `mace/modules/blocks.py:1632` | KEEP — the SOC magnetic interaction | `tests/extensions/magnetic/test_magmace.py::test_run_magnetic_scf` |
| `block.MagneticRealAgnosticResidueSpinOrbitCoupledDensityInteractionBlock` | `MagneticRealAgnosticResidueSpinOrbitCoupledDensityInteractionBlock` | `mace/modules/blocks.py:1785` | MERGE — the residual SOC variant | ⚠️ gap (registered and reachable from `--interaction`; nothing builds it) |
| `block.EquivariantProductBasisWithSelfMagmomBlock` | `EquivariantProductBasisWithSelfMagmomBlock` | `mace/modules/blocks.py:517` | KEEP — the product basis that carries the site's own moment | ⚠️ gap (used from `mace/modules/extensions.py`; no test names it) |
| `contraction.SymmetricContraction` | `SymmetricContraction` | `mace/modules/symmetric_contraction.py:26` | KEEP — the many-body contraction the accelerated backends replace | `tests/unit/test_modules.py::test_symmetric_contraction` |
| `contraction.Contraction` | `Contraction` | `mace/modules/symmetric_contraction.py:91` | KEEP — one correlation order of it | `tests/unit/test_modules.py::test_symmetric_contraction_zeroes_the_unreachable_correlation_order` |
| `contraction.EmptyParam` | `EmptyParam` | `mace/modules/symmetric_contraction.py:269` | MERGE — weight bookkeeping for an unreachable order, not a feature | `tests/unit/test_modules.py::test_symmetric_contraction_zeroes_the_unreachable_correlation_order` |
| `transform.Random3DRotation` | `Random3DRotation` | `mace/data/augmentation.py:24` | MERGE — the `--data_aug_magmom` transform, which travels with the flag | `tests/extensions/magnetic/test_magmom_augmentation.py::test_non_soc_mode_samples_the_full_o3` |
| `calc.class.MACECalculator` | `MACECalculator` | `mace/calculators/mace.py:77` | KEEP — the ASE calculator | `tests/workflows/test_calculator.py::test_calculator_forces` |
| `calc.class.MagneticMACECalculator` | `MagneticMACECalculator` | `mace/calculators/mace.py:964` | KEEP — the magnetic ASE calculator | `tests/extensions/magnetic/test_magmace.py::test_run_train_magnetic_mace` |
| `calc.class.LAMMPS_MACE` | `LAMMPS_MACE` | `mace/calculators/lammps_mace.py:10` | KEEP — the libtorch deployment wrapper | `tests/integrations/lammps/test_ghost_parity.py::test_virials_path_runs` |
| `calc.class.LAMMPS_MLIAP_MACE` | `LAMMPS_MLIAP_MACE` | `mace/calculators/lammps_mliap_mace.py:125` | KEEP — the ML-IAP deployment wrapper | `tests/integrations/lammps/test_mliap_writeback.py::test_atom_count_mismatch_is_actionable` |
| `calc.class.MACEEdgeForcesWrapper` | `MACEEdgeForcesWrapper` | `mace/calculators/lammps_mliap_mace.py:59` | KEEP — per-pair forces for the ML-IAP path | `tests/integrations/lammps/test_mliap_buffer_dtype.py::test_the_wrapper_buffers_follow_the_model_dtype` |
| `calc.class.MACELammpsConfig` | `MACELammpsConfig` | `mace/calculators/lammps_mliap_mace.py:22` | MERGE — the ML-IAP wrapper's own config object, an implementation detail | `tests/integrations/lammps/test_mliap_writeback.py::test_atom_count_mismatch_is_actionable` |
| `calc.class.MaceTorchSimModel` | `MaceTorchSimModel` | `mace/calculators/mace_torchsim.py:52` | KEEP — the torch-sim backend | `tests/extensions/torchsim/test_torchsim.py::test_mace_torchsim_no_stress` |

## 8. Loss classes (10)

All ten MERGE into composable per-stage losses: each becomes a documented composition
preset producing identical numbers, and the hand-computed cases in `tests/unit/test_loss.py` are
what each one is judged against. The
eleven `--loss` scheme names of §3.7 map onto these ten classes.

| id | feature | source | disposition | pinned by |
|---|---|---|---|---|
| `loss.WeightedEnergyForcesLoss` | `WeightedEnergyForcesLoss` | `mace/modules/loss.py:246` | MERGE — a composition preset (the `ef`/`weighted` schemes) | `tests/unit/test_loss.py::test_weighted_energy_forces_loss_hand_value` |
| `loss.WeightedForcesLoss` | `WeightedForcesLoss` | `mace/modules/loss.py:272` | MERGE — a composition preset (the `forces_only` scheme) | `tests/unit/test_loss.py::test_weighted_forces_loss` |
| `loss.WeightedEnergyForcesStressLoss` | `WeightedEnergyForcesStressLoss` | `mace/modules/loss.py:290` | MERGE — a composition preset (the `stress` scheme) | `tests/unit/test_loss.py::test_weighted_energy_forces_stress_loss_zero_and_hand_value` |
| `loss.WeightedHuberEnergyForcesStressLoss` | `WeightedHuberEnergyForcesStressLoss` | `mace/modules/loss.py:325` | MERGE — a composition preset (the `huber` scheme) | `tests/unit/test_loss.py::test_weighted_huber_energy_forces_stress_loss` |
| `loss.UniversalLoss` | `UniversalLoss` | `mace/modules/loss.py:391` | MERGE — a composition preset (the `universal` scheme) | `tests/unit/test_loss.py::test_universal_loss_full_hand_value_over_all_four_terms` |
| `loss.WeightedEnergyForcesVirialsLoss` | `WeightedEnergyForcesVirialsLoss` | `mace/modules/loss.py:506` | MERGE — a composition preset (the `virials` scheme) | `tests/unit/test_loss.py::test_weighted_energy_forces_virials_loss` |
| `loss.DipoleSingleLoss` | `DipoleSingleLoss` | `mace/modules/loss.py:543` | MERGE — a composition preset (the `dipole` scheme) | `tests/unit/test_loss.py::test_dipole_single_loss` |
| `loss.DipolePolarLoss` | `DipolePolarLoss` | `mace/modules/loss.py:563` | MERGE — a composition preset (the `dipole_polar` scheme) | `tests/unit/test_loss.py::test_dipole_polar_loss` |
| `loss.WeightedEnergyForcesDipoleLoss` | `WeightedEnergyForcesDipoleLoss` | `mace/modules/loss.py:601` | MERGE — a composition preset (the `energy_forces_dipole` scheme) | `tests/unit/test_loss.py::test_weighted_energy_forces_dipole_loss` |
| `loss.WeightedEnergyForcesL1L2Loss` | `WeightedEnergyForcesL1L2Loss` | `mace/modules/loss.py:636` | MERGE — a composition preset (the `l1l2energyforces` scheme) | `tests/unit/test_loss.py::test_weighted_energy_forces_l1l2_loss` |

## 9. Calculator constructor and exports

### 9.1 `__init__` parameters (26)

`MACECalculator.__init__` declares 22 parameters and reads three more out of `**kwargs`
(`head`, `compute_atomic_stresses`, `model_path`); the 26th (`magmom_key`) is added by
`MagneticMACECalculator`, a second `Calculator` subclass rather than a mode of the first, so its
`__init__` is a second public surface. The set is the union of both signatures and both kwargs
bags: a knob that exists on only one of the two calculators is still a knob, and so is one that
only the bag spells.

| id | feature | source | disposition | pinned by |
|---|---|---|---|---|
| `calc.param.model_paths` | `model_paths` — MACECalculator | `mace/calculators/mace.py:105` | KEEP — one path, a glob, or several for a committee | `tests/workflows/test_calculator.py::test_calculator_forces` |
| `calc.param.models` | `models` — MACECalculator | `mace/calculators/mace.py:106` | KEEP — pre-loaded model objects instead of paths | `tests/workflows/test_calculator.py::test_calculator_from_model` |
| `calc.param.device` | `device` — MACECalculator | `mace/calculators/mace.py:107` | KEEP | `tests/workflows/test_cli_contracts.py::test_the_calculator_returns_the_contract_keys_with_the_right_shapes` |
| `calc.param.default_dtype` | `default_dtype` — MACECalculator | `mace/calculators/mace.py:110` | MERGE — `PrecisionConfig` | `tests/unit/test_calculator_dtype_scope.py::test_every_forward_runs_under_the_calculator_dtype` + `tests/workflows/test_calculator.py::test_calculator_dtype_is_instance_local` |
| `calc.param.energy_units_to_eV` | `energy_units_to_eV` — MACECalculator | `mace/calculators/mace.py:108` | KEEP — unit conversion on the way out | `tests/unit/test_calculator_units_and_spread.py::test_the_default_conversions_change_nothing` |
| `calc.param.length_units_to_A` | `length_units_to_A` — MACECalculator | `mace/calculators/mace.py:109` | KEEP — idem | `tests/unit/test_calculator_units_and_spread.py::test_the_default_conversions_change_nothing` |
| `calc.param.charges_key` | `charges_key` — MACECalculator | `mace/calculators/mace.py:111` | KEEP — property-key convention | `tests/unit/test_calculator_charges_key.py::test_charges_key_selects_the_arrays_field` |
| `calc.param.info_keys` | `info_keys` — MACECalculator | `mace/calculators/mace.py:112` | KEEP — which `atoms.info` entries become graph-level inputs | `tests/unit/test_calculator_info_keys_and_state.py::test_info_keys_maps_property_name_to_the_info_key` |
| `calc.param.arrays_keys` | `arrays_keys` — MACECalculator | `mace/calculators/mace.py:113` | KEEP — idem for `atoms.arrays` | `tests/unit/test_calculator_charges_key.py::test_arrays_keys_maps_property_name_to_atoms_key` |
| `calc.param.model_type` | `model_type` — MACECalculator | `mace/calculators/mace.py:114` | MERGE — auto-detected from model metadata; asking the user to name the model family is asking them to get it wrong | `tests/workflows/test_calculator.py::test_calculator_dipole` + `tests/workflows/test_calculator.py::test_calculator_energy_dipole` |
| `calc.param.compile_mode` | `compile_mode` — MACECalculator | `mace/calculators/mace.py:115` | KEEP | `tests/unit/test_compile.py` |
| `calc.param.fullgraph` | `fullgraph` — MACECalculator | `mace/calculators/mace.py:116` | KEEP | `tests/unit/test_compile.py` |
| `calc.param.pad_num_atoms` | `pad_num_atoms` — MACECalculator | `mace/calculators/mace.py:119` | KEEP — graph padding, so a compiled graph is not recaptured per frame | `tests/workflows/test_cli_contracts.py::test_padded_per_atom_arrays_come_back_with_exactly_len_atoms_rows` |
| `calc.param.pad_num_edges` | `pad_num_edges` — MACECalculator | `mace/calculators/mace.py:120` | KEEP — idem | `tests/workflows/test_calculator.py::test_calculator_padding` + `tests/unit/test_padding.py::test_edge_index_within_bounds` |
| `calc.param.warmup` | `warmup` — MACECalculator | `mace/calculators/mace.py:121` | KEEP — one throwaway forward so the first real call is not the compile | ⚠️ gap (no test passes `warmup`, so the throwaway forward is unobserved) |
| `calc.param.enable_cueq` | `enable_cueq` — MACECalculator | `mace/calculators/mace.py:117` | MERGE — backend dispatch config | `tests/golden/test_backend_parity_golden.py::test_the_calculators_own_backend_flag_reaches_the_same_kernels` |
| `calc.param.enable_oeq` | `enable_oeq` — MACECalculator | `mace/calculators/mace.py:118` | MERGE — idem | `tests/golden/test_backend_parity_golden.py::test_the_calculators_own_backend_flag_reaches_the_same_kernels` |
| `calc.param.compute_bec` | `compute_bec` — MACECalculator | `mace/calculators/mace.py:122` | KEEP — Born effective charges from the LES/polar path | `tests/extensions/les/test_maceles.py::test_run_eval_with_bec` + `tests/extensions/les/test_maceles.py::test_run_eval_no_bec` |
| `calc.param.external_field` | `external_field` — MACECalculator | `mace/calculators/mace.py:123` | KEEP — the applied field of the LES/polar path | `tests/golden/test_tiny_maceles.py::test_the_field_force_matches_the_documented_formula` |
| `calc.param.eps_infty` | `eps_infty` — MACECalculator | `mace/calculators/mace.py:124` | KEEP — high-frequency dielectric constant used by the field path | `tests/golden/test_tiny_maceles.py::test_the_field_surface_reproduces_its_reference` |
| `calc.param.electric_field_unit` | `electric_field_unit` — MACECalculator | `mace/calculators/mace.py:125` | KEEP — unit convention for the applied field | `tests/golden/test_tiny_maceles.py::test_the_field_reference_records_the_settings_that_are_not_channels` |
| `calc.param.keep_neutral` | `keep_neutral` — MACECalculator | `mace/calculators/mace.py:126` | KEEP — charge-neutrality enforcement in the field path | `tests/golden/test_tiny_maceles.py::test_keep_neutral_removes_exactly_a_uniform_field_force` + `tests/golden/test_tiny_maceles.py::test_keep_neutral_leaves_the_reported_bec_alone_and_repeats_identically` |
| `calc.param.head` | `head` — read from `**kwargs` on MACECalculator | `mace/calculators/mace.py:297` | KEEP — which head of a multihead model the calculator evaluates | `tests/workflows/test_run_train.py::test_run_train_multihead` |
| `calc.param.compute_atomic_stresses` | `compute_atomic_stresses` — read from `**kwargs` on MACECalculator | `mace/calculators/mace.py:216` | KEEP — decides whether `stresses` and `virials` are implemented properties at all | `tests/golden/test_tiny_anchors.py::test_the_two_per_atom_stress_routes_land_on_one_channel` |
| `calc.param.model_path` | `model_path` — read from `**kwargs` on MACECalculator | `mace/calculators/mace.py:162` | DROP — deprecated singular alias for `model_paths`; it warns and forwards, and refuses when both are given | — |
| `calc.param.magmom_key` | `magmom_key` — MagneticMACECalculator | `mace/calculators/mace.py:993` | KEEP — property-key convention; `MagneticMACECalculator` only | `tests/extensions/magnetic` |

### 9.2 Exports (`mace/calculators/__init__.py`, 9)

| id | feature | source | disposition | pinned by |
|---|---|---|---|---|
| `calc.export.MACECalculator` | `MACECalculator` | `mace/calculators/__init__.py:12` | KEEP — the one ASE calculator | `tests/workflows/test_calculator.py::test_calculator_forces` |
| `calc.export.MagneticMACECalculator` | `MagneticMACECalculator` | `mace/calculators/__init__.py:12` | KEEP — a separate `Calculator` subclass (~530 lines), not a mode of `MACECalculator`. The open v1 question is whether it stays a second class or collapses into the one calculator once magmom is just another input feature | `tests/extensions/magnetic` (eval path) |
| `calc.export.LAMMPS_MACE` | `LAMMPS_MACE` | `mace/calculators/__init__.py:12` | DROP — the TorchScript wrapper dies with the TorchScript export format; the MLIAP path replaces it | `tests/integrations/lammps/test_export_golden.py::test_the_exported_artifact_reproduces_the_committed_numbers` |
| `calc.export.mace_mp` | `mace_mp` | `mace/calculators/__init__.py:12` | KEEP | `tests/golden/test_foundation_goldens.py::test_an_unqualified_mace_mp_is_mpa0_medium_and_reads_no_url` |
| `calc.export.mace_off` | `mace_off` | `mace/calculators/__init__.py:12` | KEEP | `tests/golden/test_foundation_goldens.py::test_foundation_model_reproduces_its_reference` |
| `calc.export.mace_polar` | `mace_polar` | `mace/calculators/__init__.py:12` | KEEP | `tests/golden/test_polar_foundation.py::test_polar_foundation_reproduces_its_reference` |
| `calc.export.mace_mdp` | `mace_mdp` | `mace/calculators/__init__.py:12` | KEEP — a published dipole/polarizability foundation model with released calculator support | `tests/golden/test_mdp_foundation.py::test_mace_mdp_refuses_another_model_type` + `tests/golden/test_mdp_foundation.py::test_mace_mdp_warns_that_it_is_not_an_energy_model` |
| `calc.export.mace_omol` | `mace_omol` | `mace/calculators/__init__.py:12` | KEEP — a recent, large, published multi-head model; converts with heads intact | `tests/foundations/test_foundations.py::test_mace_omol_elements_subset_reproduces_energy_forces` |
| `calc.export.mace_anicc` | `mace_anicc` | `mace/calculators/__init__.py:12` | DROP — a 2023 organic-chemistry model superseded by MACE-OFF, and the only loader with a divergent signature (`model_path` instead of `model`): an API exception for an obsolete artifact. Its tracked checkpoint `mace/calculators/foundations_models/ani500k_large_CC.model` goes with it; the release notes say "use MACE-OFF" | — |

## 10. Optional-dependency extras (12)

From `setup.cfg` `[options.extras_require]`. Two facts here feed design decisions rather than
packaging: `magnetic` pins **`sphericart-torch`**, a shipped dependency on a non-e3nn
spherical-harmonics backend (see `SHModule`, §6), and it also declares **external
`torch-geometric`** while `mace/data/augmentation.py` imports the real package *and* the vendored
copy — so the tree depends on both at once (§19).

| id | feature | source | disposition | pinned by |
|---|---|---|---|---|
| `extra.wandb` | `[wandb]` | `setup.cfg` | KEEP | ⚠️ gap (offline-mode smoke) |
| `extra.fpsample` | `[fpsample]` | `setup.cfg` | KEEP — fast farthest-point sampling for fine-tuning selection | `tests/workflows/test_finetuning_select_cli.py::test_fps_without_fpsample_falls_back_and_says_so` (the fallback, in whichever direction the extra is present) |
| `extra.schedulefree` | `[schedulefree]` | `setup.cfg` | KEEP | `tests/extensions/schedulefree` |
| `extra.torchsim` | `[torchsim]` | `setup.cfg` | KEEP — a first-class deployment path, not a secondary integration; the coupling to torch-sim's still-moving API becomes MACE's problem, so the version is pinned in `requirements/` | `tests/extensions/torchsim` |
| `extra.magnetic` | `[magnetic]` | `setup.cfg` | KEEP — `sphericart-torch` + `torch-geometric` | `tests/extensions/magnetic` |
| `extra.cueq` | `[cueq]` | `setup.cfg` | KEEP — the backend extra naming is worth revisiting | `tests/golden/test_backend_parity_golden.py::test_the_audits_verdict_tracks_whether_the_fused_ops_are_installed` |
| `extra.cueq-cuda-11` | `[cueq-cuda-11]` | `setup.cfg` | KEEP — idem; the ops major must match `torch.version.cuda`, not the newest available | ⚠️ gap (nothing in the suite can assert a `cueq-cuda-11` wheel resolves; only the GPU CI job installing it can) |
| `extra.cueq-cuda-12` | `[cueq-cuda-12]` | `setup.cfg` | KEEP — idem | ⚠️ gap (nothing in the suite can assert a `cueq-cuda-12` wheel resolves; only the GPU CI job installing it can) |
| `extra.cueq-cuda-13` | `[cueq-cuda-13]` | `setup.cfg` | KEEP — idem; cu13 ops start at cuequivariance 0.7.0 | ⚠️ gap (nothing in the suite can assert a `cueq-cuda-13` wheel resolves; only the GPU CI job installing it can) |
| `extra.oeq` | `[oeq]` | `setup.cfg` | KEEP — OpenEquivariance, the AMD-capable accelerated backend | `tests/golden/test_backend_parity_golden.py::test_the_audit_accepts_a_well_formed_oeq_conversion_and_only_that` |
| `extra.dev` | `[dev]` | `setup.cfg` | KEEP — the lint/format toolchain | the lint job itself |
| `extra.test` | `[test]` | `setup.cfg` | KEEP — pytest and its plugins, split out of `dev` so a test job need not install the linters | the suite itself |

## 11. Model output keys (43)

The keys of the dicts the model `forward`s return. This is the contract every consumer reads —
the ASE calculator, `mace_eval_configs`, the LAMMPS runtimes, the training loop — so a renamed
key is a silent breakage in four places at once. v1 replaces the untyped dict with a typed
`MACEOutputs` whose fields are these keys; the disposition column says which name survives that
move and which is absorbed into a declared observable.

| id | feature | source | disposition | pinned by |
|---|---|---|---|---|
| `out.model.energy` | `energy` — first declared by `MACE` | `mace/modules/models.py:428` | KEEP — a field of the typed outputs | `tests/golden/test_tiny_anchors.py::test_anchor_reproduces_its_reference` |
| `out.model.node_energy` | `node_energy` — first declared by `MACE` | `mace/modules/models.py:429` | KEEP — idem | `tests/golden/test_tiny_anchors.py::test_anchor_reproduces_its_reference` |
| `out.model.interaction_energy` | `interaction_energy` — first declared by `ScaleShiftMACE` | `mace/modules/models.py:612` | KEEP — idem (total minus the E0s) | `tests/golden/test_tiny_anchors.py::test_anchor_reproduces_its_reference` |
| `out.model.forces` | `forces` — first declared by `MACE` | `mace/modules/models.py:431` | KEEP — idem | `tests/golden/test_tiny_anchors.py::test_anchor_reproduces_its_reference` |
| `out.model.stress` | `stress` — first declared by `MACE` | `mace/modules/models.py:434` | KEEP — idem | `tests/golden/test_tiny_anchors.py::test_anchor_reproduces_its_reference` |
| `out.model.virials` | `virials` — first declared by `MACE` | `mace/modules/models.py:433` | KEEP — idem | `tests/golden/test_tiny_anchors.py::test_anchor_reproduces_its_reference` |
| `out.model.hessian` | `hessian` — first declared by `MACE` | `mace/modules/models.py:438` | KEEP — idem | `tests/foundations/test_hessian.py` |
| `out.model.edge_forces` | `edge_forces` — first declared by `MACE` | `mace/modules/models.py:432` | KEEP — per-edge forces; a first-class observable because the LAMMPS MLIAP path needs them | `tests/integrations/lammps` |
| `out.model.atomic_virials` | `atomic_virials` — first declared by `MACE` | `mace/modules/models.py:435` | KEEP — per-atom virials | `tests/golden/test_tiny_anchors.py::test_the_two_per_atom_stress_routes_land_on_one_channel` |
| `out.model.atomic_stresses` | `atomic_stresses` — first declared by `MACE` | `mace/modules/models.py:436` | KEEP — per-atom stresses | `tests/golden/test_tiny_anchors.py::test_the_two_per_atom_stress_routes_land_on_one_channel` + `tests/golden/test_tiny_anchors.py::test_the_two_per_atom_stress_routes_snapshot_identically` |
| `out.model.contributions` | `contributions` — first declared by `MACE` | `mace/modules/models.py:430` | KEEP — per-layer energy contributions | `tests/workflows/test_cli_contracts.py::test_eval_contributions_sum_to_the_total_energy_on_the_plain_model` |
| `out.model.node_feats` | `node_feats` — first declared by `MACE` | `mace/modules/models.py:439` | KEEP — the descriptor surface; `BaseMACE` exposes it through the descriptor API | `tests/golden/test_harness.py::test_only_the_per_atom_descriptors_are_pinnable` |
| `out.model.displacement` | `displacement` — first declared by `MACE` | `mace/modules/models.py:437` | MERGE — the strain displacement is internal machinery of the derivative engine, not a user-facing output; v1 does not return it | `tests/unit/test_physics_glue.py::test_prepare_graph_injects_a_differentiable_strain_handle` + `tests/unit/test_physics_glue.py::test_prepare_graph_without_stress_leaves_an_inert_displacement` |
| `out.model.dipole` | `dipole` — first declared by `AtomicDipolesMACE` | `mace/modules/models.py:835` | KEEP — a declared observable | `tests/golden/test_tiny_dipoles.py::test_anchor_reproduces_its_reference` |
| `out.model.atomic_dipoles` | `atomic_dipoles` — first declared by `AtomicDipolesMACE` | `mace/modules/models.py:836` | KEEP — idem | `tests/golden/test_tiny_dipoles.py::test_anchor_reproduces_its_reference` |
| `out.model.charges` | `charges` — first declared by `AtomicDielectricMACE` | `mace/modules/models.py:1187` | KEEP — idem | `tests/golden/test_tiny_dipoles.py::test_the_fixed_charge_baseline_is_live` |
| `out.model.polarizability` | `polarizability` — first declared by `AtomicDielectricMACE` | `mace/modules/models.py:1190` | KEEP — idem | `tests/golden/test_mdp_foundation.py::test_the_reference_pins_the_polarizability_and_its_derivatives` |
| `out.model.polarizability_sh` | `polarizability_sh` — first declared by `AtomicDielectricMACE` | `mace/modules/models.py:1191` | KEEP — the spherical-harmonics form of the polarizability | `tests/golden/test_mdp_foundation.py::test_the_reference_pins_the_polarizability_and_its_derivatives` |
| `out.model.dmu_dr` | `dmu_dr` — first declared by `AtomicDielectricMACE` | `mace/modules/models.py:1192` | KEEP — dipole derivative (the dielectric family's IR path) | `tests/golden/test_mdp_foundation.py::test_the_reference_pins_the_polarizability_and_its_derivatives` |
| `out.model.dalpha_dr` | `dalpha_dr` — first declared by `AtomicDielectricMACE` | `mace/modules/models.py:1193` | KEEP — polarizability derivative (Raman) | `tests/golden/test_mdp_foundation.py::test_the_reference_pins_the_polarizability_and_its_derivatives` |
| `out.model.les_energy` | `les_energy` — first declared by `MACELES` | `mace/modules/extensions.py:648` | KEEP — the LES long-range energy term | `tests/golden/test_tiny_maceles.py::test_the_model_surface_reproduces_its_reference` |
| `out.model.latent_charges` | `latent_charges` — first declared by `MACELES` | `mace/modules/extensions.py:649` | KEEP — LES latent multipoles | `tests/golden/test_tiny_maceles.py::test_every_latent_quantity_is_present_and_above_the_tolerance_floor` |
| `out.model.latent_dipoles` | `latent_dipoles` — first declared by `MACELES` | `mace/modules/extensions.py:650` | KEEP — idem | `tests/golden/test_tiny_maceles.py::test_every_latent_quantity_is_present_and_above_the_tolerance_floor` |
| `out.model.latent_kappas` | `latent_kappas` — first declared by `MACELES` | `mace/modules/extensions.py:651` | KEEP — idem | `tests/golden/test_tiny_maceles.py::test_every_latent_quantity_is_present_and_above_the_tolerance_floor` |
| `out.model.latent_alphas` | `latent_alphas` — first declared by `MACELES` | `mace/modules/extensions.py:652` | KEEP — idem | `tests/golden/test_tiny_maceles.py::test_every_latent_quantity_is_present_and_above_the_tolerance_floor` |
| `out.model.latent_quads` | `latent_quads` — first declared by `MACELES` | `mace/modules/extensions.py:653` | KEEP — idem | `tests/golden/test_tiny_maceles.py::test_every_latent_quantity_is_present_and_above_the_tolerance_floor` |
| `out.model.BEC` | `BEC` — first declared by `MACELES` | `mace/modules/extensions.py:654` | KEEP — Born effective charges | `tests/golden/test_tiny_maceles.py::test_keep_neutral_leaves_the_reported_bec_alone_and_repeats_identically` |
| `out.model.electrostatic_energy` | `electrostatic_energy` — first declared by `PolarMACE` | `mace/modules/extensions.py:1342` | KEEP — polar energy decomposition | `tests/golden/test_polar_foundation.py::test_the_reference_pins_the_electrostatics_and_not_only_the_energy` |
| `out.model.electron_energy` | `electron_energy` — first declared by `PolarMACE` | `mace/modules/extensions.py:1343` | KEEP — idem | `tests/golden/test_polar_foundation.py::test_the_reference_pins_the_electrostatics_and_not_only_the_energy` |
| `out.model.electrostatic_potentials` | `electrostatic_potentials` — first declared by `PolarMACE` | `mace/modules/extensions.py:1344` | KEEP — idem | `tests/extensions/polar/test_polar_output_keys.py::test_electrostatic_potentials_are_absent_unless_asked_for` |
| `out.model.density_coefficients` | `density_coefficients` — first declared by `PolarMACE` | `mace/modules/extensions.py:1331` | KEEP — the polar density expansion, consumed by `mace_polar_density_cube` | `tests/extensions/polar/test_polar_density_cube.py` |
| `out.model.spin_density` | `spin_density` — first declared by `PolarMACE` | `mace/modules/extensions.py:1332` | KEEP — idem | `tests/extensions/polar/test_polar_output_keys.py::test_the_density_keys_keep_their_rank` |
| `out.model.spin_charge_density` | `spin_charge_density` — first declared by `PolarMACE` | `mace/modules/extensions.py:1345` | KEEP — idem | `tests/golden/test_polar_foundation.py::test_the_reference_pins_the_electrostatics_and_not_only_the_energy` |
| `out.model.spins` | `spins` — first declared by `PolarMACE` | `mace/modules/extensions.py:1339` | KEEP — per-atom spin populations | `tests/extensions/polar/test_polar_output_keys.py::test_the_electrostatic_keys_are_present_with_the_right_extent` |
| `out.model.total_charge` | `total_charge` — first declared by `PolarMACE` | `mace/modules/extensions.py:1341` | KEEP — echoed back as an output so a consumer can check what was imposed | `tests/extensions/polar/test_polar_output_keys.py::test_the_electrostatic_keys_are_present_with_the_right_extent` |
| `out.model.fermi_level` | `fermi_level` — first declared by `PolarMACE` | `mace/modules/extensions.py:1336` | KEEP — idem | `tests/extensions/polar/test_polar_output_keys.py::test_the_electrostatic_keys_are_present_with_the_right_extent` |
| `out.model.external_field` | `external_field` — first declared by `PolarMACE` | `mace/modules/extensions.py:1337` | KEEP — idem | `tests/extensions/polar/test_polar_output_keys.py::test_the_electrostatic_keys_are_present_with_the_right_extent` |
| `out.model.fukui_functions` | `fukui_functions` — first declared by `PolarMACE` | `mace/modules/extensions.py:1346` | KEEP — the fukui reactivity output | `tests/extensions/polar/test_polar_models.py::test_polar_calculator_returns_fukui_functions_by_default` |
| `out.model.charges_history` | `charges_history` — first declared by `PolarMACE` | `mace/modules/extensions.py:1333` | MERGE — the per-iteration trace of the fixed-point solve; a solver diagnostic, so it belongs to the solver-dispatch layer rather than the model's outputs | `tests/extensions/polar/test_polar_output_keys.py::test_the_density_keys_keep_their_rank` |
| `out.model.magforces` | `magforces` — first declared by `MagneticScaleShiftMACE` | `mace/modules/extensions.py:1951` | KEEP — `dE/dm`, a declared derivative exactly like forces | `tests/extensions/magnetic` + `tests/golden/test_tiny_magnetic.py::test_anchor_reproduces_its_reference` |
| `out.model.scf_steps` | `scf_steps` — first declared by `MagneticSCFMACE` | `mace/modules/extensions.py:2102` | MERGE — SCF solver diagnostics belong to the model-transform hook, not to the model's output contract | `tests/extensions/magnetic::test_run_magnetic_scf` |
| `out.model.scf_energy_history` | `scf_energy_history` — first declared by `MagneticSCFMACE` | `mace/modules/extensions.py:2099` | MERGE — idem | `tests/extensions/magnetic::test_run_magnetic_scf` |
| `out.model.equilibrated_magmom` | `equilibrated_magmom` — first declared by `MagneticSCFMACE` | `mace/modules/extensions.py:2103` | KEEP — the converged magnetic moments are a result, not a diagnostic | `tests/extensions/magnetic::test_run_magnetic_scf` |

## 12. Calculator and eval output keys (31 + 13)

### 12.1 `Calculator.results` keys (31)

What an ASE user reads back. Four shapes contribute and the extractor covers all four:
`implemented_properties` lists, direct `self.results[...]` assignments, the `results_map` table,
and the committee suffixes derived from `results_store_ensemble` (`_comm` = the per-model stack,
`_var` = its variance). ASE's own vocabulary (`energy`, `free_energy`, `energies`, `forces`,
`stress`, `stresses`) is fixed by ASE, not by MACE, so those names are KEEP by definition.

| id | feature | source | disposition | pinned by |
|---|---|---|---|---|
| `out.calc.energy` | `energy` — results_map | `mace/calculators/mace.py:720` | KEEP — ASE's own property name; the calculator does not get to rename it | `tests/workflows/test_cli_contracts.py::test_the_calculator_returns_the_contract_keys_with_the_right_shapes` |
| `out.calc.free_energy` | `free_energy` — self.results[...] | `mace/calculators/mace.py:785` | KEEP — ASE's own property name; the calculator does not get to rename it (ASE requires it alongside `energy`) | `tests/workflows/test_cli_contracts.py::test_the_calculator_returns_the_contract_keys_with_the_right_shapes` |
| `out.calc.energies` | `energies` — self.results[...] | `mace/calculators/mace.py:787` | KEEP — ASE's own property name; the calculator does not get to rename it (per-atom energies) | `tests/workflows/test_cli_contracts.py::test_the_calculator_returns_the_contract_keys_with_the_right_shapes` |
| `out.calc.forces` | `forces` — results_map | `mace/calculators/mace.py:722` | KEEP — ASE's own property name; the calculator does not get to rename it | `tests/workflows/test_cli_contracts.py::test_the_calculator_returns_the_contract_keys_with_the_right_shapes` |
| `out.calc.stress` | `stress` — results_map | `mace/calculators/mace.py:723` | KEEP — ASE's own property name; the calculator does not get to rename it; note it is converted to Voigt 6-vector on the way out | `tests/workflows/test_cli_contracts.py::test_the_calculator_returns_the_contract_keys_with_the_right_shapes` |
| `out.calc.stresses` | `stresses` — results_map | `mace/calculators/mace.py:724` | KEEP — ASE's own property name; the calculator does not get to rename it (per-atom, Voigt) | `tests/golden/test_tiny_anchors.py::test_the_two_per_atom_stress_routes_snapshot_identically` |
| `out.calc.virials` | `virials` — results_map | `mace/calculators/mace.py:729` | KEEP — not an ASE property but a MACE one, exposed per-atom | `tests/unit/test_physics_glue.py::test_virials_are_minus_the_stress_times_the_volume` |
| `out.calc.node_energy` | `node_energy` — results_map | `mace/calculators/mace.py:721` | KEEP — per-atom energy with the E0s subtracted, which is *not* the same array as `energies`; both are exposed and both must stay distinguishable | `tests/workflows/test_calculator.py::test_calculator_node_energy` |
| `out.calc.dipole` | `dipole` — results_map | `mace/calculators/mace.py:734` | KEEP | `tests/golden/test_mdp_foundation.py::test_the_calculator_surface_is_the_dipole_polarizability_one` |
| `out.calc.charges` | `charges` — results_map | `mace/calculators/mace.py:735` | KEEP | `tests/unit/test_calculator_charges_key.py::test_charges_reach_the_fixed_charge_dipole_baseline` |
| `out.calc.polarizability` | `polarizability` — results_map | `mace/calculators/mace.py:736` | KEEP | `tests/golden/test_mdp_foundation.py::test_the_calculator_surface_is_the_dipole_polarizability_one` |
| `out.calc.polarizability_sh` | `polarizability_sh` — results_map | `mace/calculators/mace.py:737` | KEEP | `tests/golden/test_mdp_foundation.py::test_the_calculator_surface_is_the_dipole_polarizability_one` |
| `out.calc.bec` | `bec` — self.results[...] | `mace/calculators/mace.py:807` | KEEP — Born effective charges | `tests/extensions/les/test_maceles.py::test_keep_neutral_does_not_mutate_stored_bec` |
| `out.calc.interaction_energy` | `interaction_energy` — results_map | `mace/calculators/mace.py:742` | KEEP — polar energy decomposition | `tests/golden/test_polar_foundation.py::test_polar_foundation_reproduces_its_reference` |
| `out.calc.electrostatic_energy` | `electrostatic_energy` — results_map | `mace/calculators/mace.py:747` | KEEP — idem | `tests/golden/test_polar_foundation.py::test_the_reference_pins_the_electrostatics_and_not_only_the_energy` |
| `out.calc.electron_energy` | `electron_energy` — results_map | `mace/calculators/mace.py:752` | KEEP — idem | `tests/golden/test_polar_foundation.py::test_the_reference_pins_the_electrostatics_and_not_only_the_energy` |
| `out.calc.spins` | `spins` — results_map | `mace/calculators/mace.py:753` | KEEP | `tests/extensions/polar/test_polar_density_cube.py::test_select_multipoles_from_calculator_results` |
| `out.calc.density_coefficients` | `density_coefficients` — results_map | `mace/calculators/mace.py:754` | KEEP | `tests/extensions/polar/test_polar_density_cube.py` |
| `out.calc.spin_charge_density` | `spin_charge_density` — results_map | `mace/calculators/mace.py:755` | KEEP | `tests/extensions/polar/test_polar_density_cube.py::test_realspace_spin_channel_integrals_match_coefficients` |
| `out.calc.fukui_functions` | `fukui_functions` — implemented_properties | `mace/calculators/mace.py:227` | KEEP | `tests/extensions/polar/test_polar_models.py::test_polar_calculator_returns_fukui_functions_by_default` |
| `out.calc.LES_alphas` | `LES_alphas` — self.results[...] | `mace/calculators/mace.py:799` | MERGE — the calculator renames the model's `latent_alphas`; v1 exposes one name for one quantity, and a per-surface rename is exactly the kind of thing that makes a key ungreppable | `tests/golden/test_tiny_maceles.py::test_the_isotropic_polarizability_is_squared` |
| `out.calc.LES_kappas` | `LES_kappas` — self.results[...] | `mace/calculators/mace.py:803` | MERGE — idem, from `latent_kappas` | `tests/golden/test_tiny_maceles.py::test_every_latent_quantity_is_present_and_above_the_tolerance_floor` |
| `out.calc.MACE_magmoms` | `MACE_magmoms` — self.results[...] | `mace/calculators/mace.py:1411` | MERGE — idem: the magnetic calculator's spelling of the magnetic-moment observable, also written back into `atoms.arrays` | `tests/extensions/magnetic` |
| `out.calc.energy_comm` | `energy_comm` — committee | `mace/calculators/mace.py:718` | KEEP — committee output (per-model energies). The committee is the part of `mace_active_learning_md` that MACE must keep guaranteeing (§1) | `tests/workflows/test_calculator.py::test_calculator_committee` |
| `out.calc.energy_var` | `energy_var` — committee | `mace/calculators/mace.py:718` | KEEP — committee output (energy variance). The committee is the part of `mace_active_learning_md` that MACE must keep guaranteeing (§1) | `tests/unit/test_calculator_units_and_spread.py::test_a_committee_reports_the_spread_of_the_members_it_averaged` |
| `out.calc.forces_comm` | `forces_comm` — committee | `mace/calculators/mace.py:718` | KEEP — committee output (per-model forces). The committee is the part of `mace_active_learning_md` that MACE must keep guaranteeing (§1) | `tests/unit/test_calculator_units_and_spread.py::test_a_committee_reports_the_spread_of_the_members_it_averaged` |
| `out.calc.forces_var` | `forces_var` — committee | `mace/calculators/mace.py:718` | KEEP — committee output (force variance). The committee is the part of `mace_active_learning_md` that MACE must keep guaranteeing (§1) | `tests/unit/test_calculator_units_and_spread.py::test_the_variance_scales_as_the_square_and_the_members_linearly` |
| `out.calc.stress_comm` | `stress_comm` — committee | `mace/calculators/mace.py:718` | KEEP — committee output (per-model stresses). The committee is the part of `mace_active_learning_md` that MACE must keep guaranteeing (§1) | ⚠️ gap (idem) |
| `out.calc.stress_var` | `stress_var` — committee | `mace/calculators/mace.py:718` | KEEP — committee output (stress variance). The committee is the part of `mace_active_learning_md` that MACE must keep guaranteeing (§1) | ⚠️ gap (idem) |
| `out.calc.dipole_comm` | `dipole_comm` — committee | `mace/calculators/mace.py:718` | KEEP — committee output (per-model dipoles). The committee is the part of `mace_active_learning_md` that MACE must keep guaranteeing (§1) | ⚠️ gap (idem) |
| `out.calc.dipole_var` | `dipole_var` — committee | `mace/calculators/mace.py:718` | KEEP — committee output (dipole variance). The committee is the part of `mace_active_learning_md` that MACE must keep guaranteeing (§1) | ⚠️ gap (idem) |

### 12.2 Keys `mace_eval_configs` writes (13)

Every one is written as `--info_prefix` + the name below, into `atoms.info` or `atoms.arrays`,
and then serialized into the output XYZ — so these names end up in users' files on disk and in
their downstream scripts. The default prefix is `MACE_`.

| id | feature | source | disposition | pinned by |
|---|---|---|---|---|
| `out.eval.energy` | `<info_prefix>energy` → `atoms.info` | `mace/cli/eval_configs.py:397` | KEEP | `tests/workflows/test_cli_contracts.py::test_eval_configs_reproduces_the_committed_anchor_reference` |
| `out.eval.forces` | `<info_prefix>forces` → `atoms.arrays` | `mace/cli/eval_configs.py:398` | KEEP | `tests/workflows/test_cli_contracts.py::test_eval_configs_reproduces_the_committed_anchor_reference` |
| `out.eval.stress` | `<info_prefix>stress` → `atoms.info` | `mace/cli/eval_configs.py:404` | KEEP | `tests/workflows/test_cli_contracts.py::test_eval_compute_stress_writes_a_stress_and_omitting_it_does_not` |
| `out.eval.node_energies` | `<info_prefix>node_energies` → `atoms.arrays` | `mace/cli/eval_configs.py:446` | KEEP — note the plural, which matches neither the model's `node_energy` nor the calculator's `energies`; v1 writes one spelling | `tests/workflows/test_cli_contracts.py::test_eval_node_energies_sum_to_the_total_energy` |
| `out.eval.descriptors` | `<info_prefix>descriptors` → `atoms.info` | `mace/cli/eval_configs.py:440` | KEEP — written to `info` for a single aggregated vector and to `arrays` per atom | `tests/workflows/test_cli_contracts.py::test_eval_descriptors_land_per_atom_and_the_aggregations_reduce_them` + `tests/golden/test_harness.py::test_only_the_per_atom_descriptors_are_pinnable` |
| `out.eval.BO_contributions` | `<info_prefix>BO_contributions` → `atoms.info` | `mace/cli/eval_configs.py:426` | KEEP — the per-layer energy contributions, under a third spelling of the same quantity (`contributions` in the model, `--return_contributions` on the CLI) | `tests/workflows/test_cli_contracts.py::test_eval_contributions_sum_to_the_total_energy_on_the_plain_model` |
| `out.eval.magforces` | `<info_prefix>magforces` → `atoms.arrays` | `mace/cli/eval_configs.py:401` | KEEP | `tests/extensions/magnetic` |
| `out.eval.BEC` | `<info_prefix>BEC` → `atoms.arrays` | `mace/cli/eval_configs.py:407` | KEEP | `tests/golden/test_harness.py::test_the_eval_cli_flattens_the_born_charges_and_the_schema_unflattens_them` |
| `out.eval.latent_charges` | `<info_prefix>latent_charges` → `atoms.arrays` | `mace/cli/eval_configs.py:411` | KEEP | `tests/golden/test_tiny_maceles.py::test_the_eval_cli_lands_on_the_same_numbers_as_the_forward` |
| `out.eval.latent_dipoles` | `<info_prefix>latent_dipoles` → `atoms.arrays` | `mace/cli/eval_configs.py:413` | KEEP | `tests/golden/test_tiny_maceles.py::test_the_eval_cli_lands_on_the_same_numbers_as_the_forward` |
| `out.eval.latent_kappas` | `<info_prefix>latent_kappas` → `atoms.arrays` | `mace/cli/eval_configs.py:415` | KEEP | `tests/golden/test_tiny_maceles.py::test_the_eval_cli_lands_on_the_same_numbers_as_the_forward` |
| `out.eval.latent_alphas` | `<info_prefix>latent_alphas` → `atoms.arrays` | `mace/cli/eval_configs.py:417` | KEEP | `tests/golden/test_tiny_maceles.py::test_the_eval_cli_lands_on_the_same_numbers_as_the_forward` |
| `out.eval.latent_quads` | `<info_prefix>latent_quads` → `atoms.arrays` | `mace/cli/eval_configs.py:421` | KEEP | `tests/golden/test_tiny_maceles.py::test_the_eval_cli_lands_on_the_same_numbers_as_the_forward` |

## 13. Behaviour-affecting environment variables (9 + 3)

Every `MACE_*` literal in the package. They are gated as a set because an environment variable is
the one configuration channel that leaves no trace in the run metadata.

| id | feature | source | disposition | pinned by |
|---|---|---|---|---|
| `env.MACE_TIME` | `MACE_TIME` | `mace/calculators/lammps_mliap_mace.py:25` | KEEP — MLIAP runtime config; behaviour preserved even if it later moves into a deploy manifest (per-step timing) | ⚠️ gap (env-var behaviour untested) |
| `env.MACE_PROFILE` | `MACE_PROFILE` | `mace/calculators/lammps_mliap_mace.py:26` | KEEP — MLIAP runtime config; behaviour preserved even if it later moves into a deploy manifest (torch profiler) | ⚠️ gap (idem) |
| `env.MACE_PROFILE_START` | `MACE_PROFILE_START` | `mace/calculators/lammps_mliap_mace.py:27` | KEEP — MLIAP runtime config; behaviour preserved even if it later moves into a deploy manifest (first profiled step) | ⚠️ gap (idem) |
| `env.MACE_PROFILE_END` | `MACE_PROFILE_END` | `mace/calculators/lammps_mliap_mace.py:28` | KEEP — MLIAP runtime config; behaviour preserved even if it later moves into a deploy manifest (last profiled step) | ⚠️ gap (idem) |
| `env.MACE_ALLOW_CPU` | `MACE_ALLOW_CPU` | `mace/calculators/lammps_mliap_mace.py:29` | KEEP — MLIAP runtime config; behaviour preserved even if it later moves into a deploy manifest (tolerate CPU tensors) | ⚠️ gap (idem) |
| `env.MACE_FORCE_CPU` | `MACE_FORCE_CPU` | `mace/calculators/lammps_mliap_mace.py:30` | KEEP — MLIAP runtime config; behaviour preserved even if it later moves into a deploy manifest (force CPU execution) | ⚠️ gap (idem) |
| `env.MACE_ASE_PAD_NUM_ATOMS` | `MACE_ASE_PAD_NUM_ATOMS` | `mace/calculators/mace.py:411` | KEEP — the calculator's padding override, as an explicit config field in v1 | `tests/workflows/test_cli_contracts.py::test_padding_through_the_two_environment_variables_behaves_identically` |
| `env.MACE_ASE_PAD_NUM_EDGES` | `MACE_ASE_PAD_NUM_EDGES` | `mace/calculators/mace.py:413` | KEEP — idem | `tests/workflows/test_cli_contracts.py::test_padding_through_the_two_environment_variables_behaves_identically` |
| `env.MACE_USE_CUEQ_CG` | `MACE_USE_CUEQ_CG` | `mace/tools/cg.py:23` | DROP — the variable goes, not the capability: an environment variable that silently changes model numerics is unreproducible and never lands in the run metadata; it is what makes machine-to-machine differences unexplainable. The CG source becomes a backend decision recorded in the resolved config | ⚠️ gap (nothing compares the two CG sources against each other) |

Three further variables are read but are not MACE's own namespace, so they are not in the gated
set:

| id | feature | source | disposition | pinned by |
|---|---|---|---|---|
| `stdenv.XDG_CACHE_HOME` | `XDG_CACHE_HOME` — foundation-model cache location | `mace/calculators/foundations_models.py` | KEEP — the standard cache convention | `tests/unit/test_download_urls.py` |
| `stdenv.TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD` | `TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD` — set by `mace/__init__.py` so pickled checkpoints load under newer torch defaults | `mace/__init__.py` | DROP — v1 checkpoints are neutral-format (safetensors + manifest), so nothing needs the unsafe-pickle escape hatch; the legacy loader keeps it until the converter is the only reader | `tests/unit/test_scale_shift_dtype.py::test_a_full_pickle_round_trip_preserves_every_buffer_bit_for_bit` |
| `stdenv.MASTER_PORT` | `MASTER_ADDR` / `MASTER_PORT` / `RANK` / `WORLD_SIZE` / `LOCAL_RANK` / `SLURM_*` / `OMPI_*` — standard DDP and launcher plumbing | `mace/tools/slurm_distributed.py`, `mace/tools/distributed_tools.py` | KEEP — the launcher contract is torch's and SLURM's, not MACE's, so v1 reads the same variables | `tests/workflows/test_distributed.py` |

## 14. Registered pytest markers (13)

The capability model of the suite: locally a test whose capability is missing skips, and in CI a
job that exports `MACE_REQUIRE_CAPS` fails instead. CI generates its capabilities manifest from
this list, which is why the one marker that is **not** a capability has to be named here rather
than left to be inferred.

| id | feature | source | disposition | pinned by |
|---|---|---|---|---|
| `marker.gpu` | `@pytest.mark.gpu` | `pyproject.toml:23` | KEEP — a capability probe in `tests/conftest.py` (any vendor: CUDA or ROCm) | `tests/conftest.py::CAPABILITY_PROBES[gpu]` |
| `marker.cueq` | `@pytest.mark.cueq` | `pyproject.toml:25` | KEEP — a capability probe in `tests/conftest.py` | `tests/conftest.py::CAPABILITY_PROBES[cueq]` |
| `marker.oeq` | `@pytest.mark.oeq` | `pyproject.toml:26` | KEEP — a capability probe in `tests/conftest.py` | `tests/conftest.py::CAPABILITY_PROBES[oeq]` |
| `marker.polar` | `@pytest.mark.polar` | `pyproject.toml:27` | KEEP — a capability probe in `tests/conftest.py` | `tests/conftest.py::CAPABILITY_PROBES[polar]` |
| `marker.les` | `@pytest.mark.les` | `pyproject.toml:28` | KEEP — a capability probe in `tests/conftest.py` | `tests/conftest.py::CAPABILITY_PROBES[les]` |
| `marker.magnetic` | `@pytest.mark.magnetic` | `pyproject.toml:29` | KEEP — a capability probe in `tests/conftest.py` | `tests/conftest.py::CAPABILITY_PROBES[magnetic]` |
| `marker.torchsim` | `@pytest.mark.torchsim` | `pyproject.toml:30` | KEEP — a capability probe in `tests/conftest.py` | `tests/conftest.py::CAPABILITY_PROBES[torchsim]` |
| `marker.schedulefree` | `@pytest.mark.schedulefree` | `pyproject.toml:31` | KEEP — a capability probe in `tests/conftest.py` | `tests/conftest.py::CAPABILITY_PROBES[schedulefree]` |
| `marker.bin_lammps` | `@pytest.mark.bin_lammps` | `pyproject.toml:33` | KEEP — a capability probe in `tests/conftest.py` (an external binary rather than an import) | `tests/conftest.py::CAPABILITY_PROBES[bin_lammps]` |
| `marker.network` | `@pytest.mark.network` | `pyproject.toml:24` | KEEP — a capability probe in `tests/conftest.py`; never autodetected, opt-in via `MACE_CI_ALLOW_NETWORK=1` | `tests/conftest.py::CAPABILITY_PROBES[network]` |
| `marker.slow` | `@pytest.mark.slow` | `pyproject.toml:22` | KEEP — a cost marker, not a capability: applied by directory to `tests/workflows` | `tests/conftest.py::pytest_collection_modifyitems` |
| `marker.benchmark` | `@pytest.mark.benchmark` | `pyproject.toml:32` | KEEP — a cost marker: performance measurement, never part of a correctness gate | `tests/conftest.py::pytest_collection_modifyitems` |
| `marker.timeout` | `@pytest.mark.timeout` | `pyproject.toml:34` | KEEP — test infrastructure, and explicitly **not** a capability: it is registered only so collection works when `pytest-timeout` is absent (the plugin ships in the `test`/`dev` extras). It has no `CAPABILITY_PROBES` entry and must not be absorbed into the capabilities manifest. Three tests use it today (`tests/workflows/test_finetuning_pseudolabels.py:97,133,169`) | `tests/workflows/test_finetuning_pseudolabels.py` |

## 15. Default property keys — the on-disk data contract (13)

`DefaultKeys` (`mace/tools/default_keys.py`) is the name every labelled XYZ in the wild uses.
Silently changing one breaks every existing dataset at once, and the set **grew by two in a single
release** (`REF_magmom`, `REF_magforces`), which is the argument for freezing it explicitly rather
than letting it accrete. All thirteen KEEP their spelling; the mechanism that resolves them moves
to the property-key convention, and any rename is documented with an explicit old→new
map.

| id | feature | source | disposition | pinned by |
|---|---|---|---|---|
| `key.ENERGY` | `ENERGY` = `"REF_energy"` | `mace/tools/default_keys.py:7` | KEEP — the default name is part of the data contract; a rename needs an explicit old→new mapping in the release notes | `tests/unit/test_data_utils.py::test_default_keys_are_exactly_these_thirteen` + `tests/unit/test_data_utils.py::test_custom_property_keys_round_trip` |
| `key.FORCES` | `FORCES` = `"REF_forces"` | `mace/tools/default_keys.py:8` | KEEP — the default name is part of the data contract; a rename needs an explicit old→new mapping in the release notes | `tests/unit/test_data_utils.py::test_default_keys_are_exactly_these_thirteen` + `tests/unit/test_data_utils.py::test_custom_property_keys_round_trip` |
| `key.STRESS` | `STRESS` = `"REF_stress"` | `mace/tools/default_keys.py:9` | KEEP — the default name is part of the data contract; a rename needs an explicit old→new mapping in the release notes | `tests/unit/test_data_utils.py::test_default_keys_are_exactly_these_thirteen` + `tests/unit/test_data_utils.py::test_custom_property_keys_round_trip` |
| `key.VIRIALS` | `VIRIALS` = `"REF_virials"` | `mace/tools/default_keys.py:10` | KEEP — the default name is part of the data contract; a rename needs an explicit old→new mapping in the release notes | `tests/unit/test_data_utils.py::test_default_keys_are_exactly_these_thirteen` + `tests/unit/test_data_utils.py::test_custom_property_keys_round_trip` |
| `key.DIPOLE` | `DIPOLE` = `"dipole"` | `mace/tools/default_keys.py:11` | KEEP — the default name is part of the data contract; a rename needs an explicit old→new mapping in the release notes | `tests/unit/test_data_utils.py::test_default_keys_are_exactly_these_thirteen` + `tests/unit/test_data_utils.py::test_custom_property_keys_round_trip` |
| `key.POLARIZABILITY` | `POLARIZABILITY` = `"polarizability"` | `mace/tools/default_keys.py:12` | KEEP — the default name is part of the data contract; a rename needs an explicit old→new mapping in the release notes | `tests/unit/test_data_utils.py::test_default_keys_are_exactly_these_thirteen` + `tests/unit/test_data_utils.py::test_custom_property_keys_round_trip` |
| `key.HEAD` | `HEAD` = `"head"` | `mace/tools/default_keys.py:13` | KEEP — the default name is part of the data contract; a rename needs an explicit old→new mapping in the release notes | `tests/unit/test_data_utils.py::test_default_keys_are_exactly_these_thirteen` + `tests/unit/test_data_utils.py::test_custom_property_keys_round_trip` |
| `key.CHARGES` | `CHARGES` = `"REF_charges"` | `mace/tools/default_keys.py:14` | KEEP — the default name is part of the data contract; a rename needs an explicit old→new mapping in the release notes | `tests/unit/test_data_utils.py::test_default_keys_are_exactly_these_thirteen` + `tests/unit/test_data_utils.py::test_custom_property_keys_round_trip` |
| `key.TOTAL_CHARGE` | `TOTAL_CHARGE` = `"total_charge"` | `mace/tools/default_keys.py:15` | KEEP — the default name is part of the data contract; a rename needs an explicit old→new mapping in the release notes | `tests/unit/test_data_utils.py::test_default_keys_are_exactly_these_thirteen` + `tests/unit/test_data_utils.py::test_custom_property_keys_round_trip` |
| `key.TOTAL_SPIN` | `TOTAL_SPIN` = `"total_spin"` | `mace/tools/default_keys.py:16` | KEEP — the default name is part of the data contract; a rename needs an explicit old→new mapping in the release notes | `tests/unit/test_data_utils.py::test_default_keys_are_exactly_these_thirteen` + `tests/unit/test_data_utils.py::test_custom_property_keys_round_trip` |
| `key.ELEC_TEMP` | `ELEC_TEMP` = `"elec_temp"` | `mace/tools/default_keys.py:17` | KEEP — the default name is part of the data contract; a rename needs an explicit old→new mapping in the release notes | `tests/unit/test_data_utils.py::test_default_keys_are_exactly_these_thirteen` + `tests/unit/test_data_utils.py::test_custom_property_keys_round_trip` |
| `key.MAGMOM` | `MAGMOM` = `"REF_magmom"` | `mace/tools/default_keys.py:18` | KEEP — the default name is part of the data contract; a rename needs an explicit old→new mapping in the release notes | `tests/unit/test_data_utils.py::test_default_keys_are_exactly_these_thirteen` + `tests/unit/test_data_utils.py::test_custom_property_keys_round_trip` |
| `key.MAGFORCES` | `MAGFORCES` = `"REF_magforces"` | `mace/tools/default_keys.py:19` | KEEP — the default name is part of the data contract; a rename needs an explicit old→new mapping in the release notes | `tests/unit/test_data_utils.py::test_default_keys_are_exactly_these_thirteen` + `tests/unit/test_data_utils.py::test_custom_property_keys_round_trip` |

## 16. Calculator methods, loader keyword arguments and published model names

Not machine-gated (a method set is not a membership set the way a parser's dests are), but inventoried
under the same schema. The loader kwargs are the ones hidden behind `@overload`s in
`foundations_models.py`, which is why an earlier pass missed them.

| id | feature | source | disposition | pinned by |
|---|---|---|---|---|
| `method.calculate` | `MACECalculator.calculate` — E/F/stress, plus committee mean/variance when several models are loaded | `mace/calculators/mace.py` | KEEP | `tests/workflows/test_calculator.py::test_calculator_forces`; committee `tests/unit/test_calculator_units_and_spread.py::test_a_committee_reports_the_spread_of_the_members_it_averaged` |
| `method.check_state` | `MACECalculator.check_state` — ASE's recalculation trigger | `mace/calculators/mace.py` | KEEP — part of the ASE contract | `tests/unit/test_calculator_info_keys_and_state.py::test_a_changed_info_entry_is_a_change` |
| `method.get_hessian` | `MACECalculator.get_hessian` — analytical Hessians, including the polar variant | `mace/calculators/mace.py` | KEEP | `tests/foundations/test_hessian.py` (port cases) |
| `method.get_descriptors` | `MACECalculator.get_descriptors` — per-layer, invariants-only and aggregated node features | `mace/calculators/mace.py` | KEEP — descriptors are `BaseMACE`'s features, so this becomes a first-class API rather than a calculator extra | `tests/workflows/test_calculator.py::test_calculator_descriptor` |
| `method.get_dielectric_derivatives` | `MACECalculator.get_dielectric_derivatives` — `dmu/dr`, `dalpha/dr` | `mace/calculators/mace.py` | KEEP | `tests/golden/test_mdp_foundation.py::test_the_reference_pins_the_polarizability_and_its_derivatives` |
| `kwarg.model` | `model=` — which published artifact/size a loader fetches; shared by every loader | `mace/calculators/foundations_models.py` | KEEP | `tests/golden/test_foundation_goldens.py::test_an_unqualified_mace_mp_is_mpa0_medium_and_reads_no_url` |
| `kwarg.device` | `device=` — shared by every loader | `mace/calculators/foundations_models.py` | KEEP | `tests/golden/test_foundation_goldens.py::test_the_loader_call_forces_cpu_float64` |
| `kwarg.default_dtype` | `default_dtype=` — shared by every loader | `mace/calculators/foundations_models.py` | MERGE — `PrecisionConfig` | `tests/golden/test_foundation_goldens.py::test_the_loader_call_forces_cpu_float64` |
| `kwarg.return_raw_model` | `return_raw_model=` — hand back the `nn.Module` instead of a calculator; shared by every loader | `mace/calculators/foundations_models.py` | KEEP — the library-use path, distinct from the ASE path | `tests/foundations/test_foundations.py::test_polar_extract_config_roundtrip` + `tests/unit/test_calculator_mdp.py::test_extract_config_mace_mdp_local_model` |
| `kwarg.model_path` | `mace_anicc(model_path=…)` — the one loader whose first argument is spelled differently from every other | `mace/calculators/foundations_models.py` | DROP — goes with `mace_anicc` itself (§9.2); the signature exception is part of why | — |
| `kwarg.dispersion` | `mace_mp(dispersion=…)` — D3 dispersion correction via torch-dftd | `mace/calculators/foundations_models.py` | KEEP | `tests/unit/test_foundations_models.py::test_dispersion_damping_is_forwarded` |
| `kwarg.damping` | `mace_mp(damping=…)` — D3 damping function | `mace/calculators/foundations_models.py` | KEEP | `tests/unit/test_foundations_models.py::test_legacy_damping_name_is_accepted` |
| `kwarg.dispersion_xc` | `mace_mp(dispersion_xc=…)` — the functional the D3 parameters are taken from | `mace/calculators/foundations_models.py` | KEEP | ⚠️ gap (idem) |
| `kwarg.dispersion_cutoff` | `mace_mp(dispersion_cutoff=…)` | `mace/calculators/foundations_models.py` | KEEP | ⚠️ gap (idem) |
| `alias.published_names` | The per-loader shortcut names users write in code and docs — `small`/`medium`/`large`, `small-0b`/`medium-0b`/`*-0b2`/`medium-0b3`, `medium-mpa-0`, `small-omat-0`/`medium-omat-0`, plus the OFF/polar/MDP/OMOL sets. This **is** the current model registry | `mace/calculators/foundations_models.py` | KEEP — carried into the model registry, possibly renamed under the new naming scheme with a deprecation mapping per old alias | `tests/unit/test_download_urls.py` |
| `pkg.torchsim_backend` | `mace/calculators/mace_torchsim.py` — the torch-sim backend, including PolarMACE support | `mace/calculators/mace_torchsim.py` | KEEP — a first-class deployment path alongside ASE and LAMMPS | `tests/extensions/torchsim` |

## 17. LAMMPS runtime surface

| id | feature | source | disposition | pinned by |
|---|---|---|---|---|
| `lammps.mliap` | `LAMMPS_MLIAP_MACE` — `compute_forces`, `compute_descriptors`, `compute_gradients`, ghost-atom handling | `mace/calculators/lammps_mliap_mace.py` | KEEP — the primary v1 deployment path | `tests/integrations/lammps` (contract tier + the `bin_lammps` real tier) |
| `lammps.ghost_branches` | `lammps_class` / `lammps_natoms` branches in the interaction blocks — the real-vs-ghost atom partitioning | `mace/modules/blocks.py` | KEEP — must be designed into the rewrite, not bolted on afterwards | `tests/integrations/lammps/test_ghost_parity.py` |
| `lammps.ghost_exchange_check` | `_check_ghost_exchange_support` — refuses a multi-layer model up front when the LAMMPS build cannot exchange ghost node features | `mace/calculators/lammps_mliap_mace.py` | KEEP — the precondition is ported into the v1 unified runtime; without it a multi-layer model dies on a bare `AttributeError` inside layer two | `tests/integrations/lammps` |
| `lammps.torchscript_wrapper` | `LAMMPS_MACE` (`@compile_mode("script")`) + the `-lammps.pt` artifact `mace_create_lammps_model` writes by default | `mace/calculators/lammps_mace.py` | DROP — v1 blocks are born without `@compile_mode`, so scripting can never apply to them; the MLIAP bundle is the one supported artifact | `tests/integrations/lammps/test_export_golden.py::test_the_exported_artifact_reproduces_the_committed_numbers` |
| `lammps.compiled_side_artifact` | The `_compiled.model` / `_stagetwo_compiled.model` TorchScript artifacts training emits next to every checkpoint, each inside a bare `except Exception: pass` | `mace/cli/run_train.py` | DROP — a deliberate, recorded removal: v1 checkpoints are neutral-format only and deployment artifacts come solely from `mace export` | `tests/integrations/lammps/test_export_golden.py::test_the_mliap_export_declares_the_committed_interface` |

## 18. Documentation surface (mace-docs page index)

Each published page is user-promised functionality: the docs have to cover every `KEEP` row a page
maps to. These rows track the *promise*, not a code surface, so nothing in `mace/` corresponds to
them and nothing here is pinned by a test that imports one.

| id | feature | source | disposition | pinned by |
|---|---|---|---|---|
| `doc.quickstart` | Quick Start / Introduction / Installation / Troubleshooting | mace-docs | KEEP — maps to §1 and §10 | ⚠️ gap (docs are not CI-tested today) |
| `doc.training` | Training | mace-docs | KEEP — maps to §3 | ⚠️ gap (idem) |
| `doc.evaluation` | Evaluation | mace-docs | KEEP — maps to §5 eval | ⚠️ gap (idem) |
| `doc.multihead` | Heterogeneous Data Training / Multihead Training | mace-docs | KEEP — maps to §3.6 | ⚠️ gap (idem) |
| `doc.ase` | ASE calculator | mace-docs | KEEP — maps to §9 and §16 | ⚠️ gap (idem) |
| `doc.descriptors` | MACE descriptors | mace-docs | KEEP — maps to `get_descriptors` (§16) and the eval descriptor flags (§5) | ⚠️ gap (idem) |
| `doc.hessians` | Analytical Hessians | mace-docs | KEEP — maps to `get_hessian` (§16) | `tests/foundations/test_hessian.py` |
| `doc.dipoles` | Dipole Moments and Polarizabilities | mace-docs | KEEP — maps to §6 and §11 | ⚠️ gap (docs are not CI-tested today) |
| `doc.cuda` | CUDA Acceleration (cuEquivariance) | mace-docs | KEEP — rewritten: v1 dispatches instead of converting, so the page's central instruction (convert your model) disappears | `tests/golden/test_backend_parity_golden.py::test_the_audits_verdict_tracks_whether_the_fused_ops_are_installed` |
| `doc.openmm` | OpenMM Interface | mace-docs | KEEP | ⚠️ gap (no OpenMM coverage in-tree) |
| `doc.lammps` | MACE in LAMMPS / MACE in LAMMPS with ML-IAP | mace-docs | KEEP — reduced to the MLIAP path; maps to §17 | `tests/integrations/lammps` |
| `doc.foundation_models` | Foundation models | mace-docs | KEEP — maps to §9.2 | `tests/golden/test_foundation_goldens.py::test_the_registry_url_is_still_the_package_url` |
| `doc.electrostatics` | Electrostatic MACE | mace-docs | KEEP — maps to §3.3 and PolarMACE (§6) | `tests/golden/test_polar_foundation.py::test_the_reference_pins_the_electrostatics_and_not_only_the_energy` |
| `doc.finetuning` | Fine-tuning / Multihead Replay / LoRA | mace-docs | KEEP — maps to §3.6 | `tests/workflows/test_finetuning_contracts.py::test_multihead_replay_finetuning_completes_and_carries_both_heads` |
| `doc.preprocessing` | Large Dataset Pre-processing | mace-docs | KEEP — maps to §4 | `tests/workflows/test_preprocess.py` |
| `doc.multigpu` | Multi-GPUs Training | mace-docs | KEEP — maps to `--distributed` (§3.1) | `tests/workflows/test_distributed.py` |
| `doc.examples` | Examples (MD22, ANI-1x, liquid water; NVT with a foundation model) and Tutorials 1–3 | mace-docs | KEEP — end-to-end usage; the theory tutorial is superseded by the marimo notebooks | ⚠️ gap (examples are not CI-tested) |

## 19. Package-level surfaces and second-pass findings

Things that are neither a flag nor a class nor a key, and that an enumeration of the obvious surfaces
misses.

| id | feature | source | disposition | pinned by |
|---|---|---|---|---|
| `pkg.fairchem_lmdb` | FairChem LMDB dataset tools — reading FairChem-format LMDB shards | `mace/tools/fairchem_dataset/lmdb_dataset_tools.py` | KEEP — becomes a dataset backend: OMat24 and OMol25 ship in this format and MACE has models trained on both, so dropping it would mean re-converting terabyte-scale datasets | `tests/unit/test_lmdb_database.py` |
| `pkg.hdf5_dataset` | HDF5 shard reader/writer for on-line loading | `mace/data/hdf5_dataset.py` | KEEP — HDF5 v2 (read + write) plus a legacy-HDF5 read path | `tests/workflows/test_preprocess.py` |
| `pkg.lmdb_dataset` | LMDB shard reader, honouring the CLI key specification | `mace/data/lmdb_dataset.py` | KEEP — read support | `tests/unit/test_lmdb_database.py` |
| `pkg.neighborhood` | Neighbour-list construction via matscipy, including the non-periodic cell rework | `mace/data/neighborhood.py` | KEEP — becomes a pluggable neighbour-list backend | `tests/unit/test_data.py::test_full_directed_edge_set_against_the_oracle` + `tests/unit/test_data.py::test_the_three_returned_cell_regimes_are_actually_three` |
| `pkg.vendored_torch_geometric` | The vendored `torch_geometric` copy (`Data`, `Batch`, `DataLoader`, `scatter`) | `mace/tools/torch_geometric/` | DROP — v1 collates without torch_geometric; the vendored copy is excluded from lint and mypy today, which is the clearest sign it is not maintained code. Complication: `mace/data/augmentation.py` imports the *real* package while the rest of the tree imports the vendored one, and the `[magnetic]` extra declares external `torch-geometric` — so both must go at once | `tests/unit/test_padding.py::test_batch_collation` + `tests/unit/test_padding.py::test_atom_level_keys_are_sliced` |
| `pkg.compile_utils` | `prepare` / `simplify_if_compile` — utilities that make legacy modules `torch.compile`-able | `mace/tools/compile.py` | MERGE — v1 is compile-first, so the retrofit mechanism has nothing to retrofit | `tests/unit/test_compile.py` (compiled == eager) |
| `pkg.visualise_train` | `mace/cli/visualise_train.py` — the plotting support module behind `mace_plot_train`, with no entry point of its own | `mace/cli/visualise_train.py` | KEEP — follows its CLI (§1) | `tests/workflows/test_plot_train.py::test_a_real_results_log_becomes_a_plot` |
| `pkg.public_import_surface` | The public Python API downstream projects import — `mace.modules.MACE`, `mace.data.AtomicData`, `mace.cli.run_train.run(args)`, … | `mace/` | DROP — a deliberate break: v1 defines a new public API, and the release notes document the old→new equivalences rather than aliasing them | — |
| `pkg.anicc_checkpoint` | The tracked MACE-ANI-CC checkpoint shipped inside the wheel | `mace/calculators/foundations_models/ani500k_large_CC.model` | DROP — goes with `mace_anicc` (§9.2); v1 fetches every artifact through the model registry rather than bundling one | — |
| `pkg.statistics_json` | `statistics.json` — the preprocessing side-car carrying E0s, avg neighbours, mean/std and the atomic numbers | `mace/cli/preprocess_data.py` | KEEP — becomes part of the dataset metadata contract | `tests/workflows/test_preprocess.py` |
| `pkg.results_log_format` | The per-epoch results log (`results/*.txt`), which `mace_plot_train` parses and users script against | `mace/tools/train.py` | KEEP — becomes a structured (typed) log; the current line format is not a promise v1 makes | `tests/workflows/test_plot_train.py::test_the_results_log_carries_the_columns_the_plot_reads` |
| `pkg.heads_yaml_schema` | The `--heads` YAML sub-schema (per-head files, weights, E0s and key overrides) | `mace/tools/multihead_tools.py` | KEEP — becomes a typed section of the config schema | `tests/workflows/test_run_train.py::test_run_train_foundation_multihead_json` |
| `pkg.e0_estimation` | The three E0 resolution modes — explicit dict, `average` (least squares) and `estimated` (foundation-corrected) | `mace/tools/scripts_utils.py` | KEEP — E0 specification is a config section with an explicit, tested resolution order | `tests/unit/test_e0s_characterization.py::test_estimated_e0s_are_the_foundation_e0s_plus_a_least_squares_correction` + `tests/unit/test_e0s_characterization.py::test_estimated_e0s_are_deterministic_at_cpu_float64` |
| `pkg.lr_param_groups` | Explicit optimizer parameter groups, driven by `--lr_params_factors` and reused by `--freeze` | `mace/tools/scripts_utils.py` | MERGE — typed per-param-group fields of the per-stage optimizer config | `tests/unit/test_optimizer_param_groups.py::test_trainable_bessel_weights_get_their_own_group` + `tests/workflows/test_freeze.py::test_run_train_freeze` |
| `pkg.augmentation` | `Random3DRotation` — the training-data augmentation transform behind `--data_aug_magmom` | `mace/data/augmentation.py` | MERGE — a registered training-data transform, not a model flag | `tests/extensions/magnetic` (rotation equivariance) |
| `pkg.per_head_reporting` | Per-head validation logging, per-head test error tables and per-head parity plots with labels | `mace/tools/train.py`, `mace/tools/tables_utils.py` | KEEP — multihead is the normal case now, so per-head reporting is not an add-on | ⚠️ gap (multihead reporting; add a case to `tests/workflows/test_finetuning_contracts.py`) |
| `pkg.atomic_download` | Race-free foundation-model downloads: fetch to a temporary file and rename, so a failed or concurrent download cannot leave a truncated checkpoint in the cache | `mace/calculators/foundations_models.py` | KEEP — the property, not the implementation: a partial download must never be readable as a model | `tests/unit/test_download_urls.py` |
| `pkg.first_block_allowlist` | The three first-interaction blocks a `MACE`-type model accepts — plain, Density and, since recently, the **non-linear** residual one | `mace/tools/model_script_utils.py:280-284` | KEEP — a non-linear first block is a valid architecture and the v1 assembly must not re-impose the old restriction | ⚠️ gap (add a case to `tests/unit/test_models.py`) |
| `pkg.first_block_coercion` | For `--model MACE`, anything outside that allowlist is **silently rewritten** to `RealAgnosticInteractionBlock` | `mace/tools/model_script_utils.py:285` | DROP — a config value the tool overwrites without a word is worse than a rejected one: the run trains a different architecture than the user asked for and nothing says so. v1 fails the combination in config validation | ⚠️ gap (add a case to `tests/unit/test_models.py`) |

## 20. External `mace-scf` model families (out-of-tree)

Five charge-aware families built on MACE v0.3.14 + `graph_longrange`, in `ACEsuit/mace-scf`. They are
**architectures, not checkpoints**, so they are reimplemented on the v1 two-layer + model-transform
hook and pinned by parity — not converted. They enter at the lowest support tier, after the core
refactor. Two consequences: the model-transform hook is designed against **three** different SCF
schemes rather than one (validate it with MACE-QEq, the most established), and all five need a v1
golden, which is five new parity artifacts on top of the foundation-model set. Whichever solver they
use inherits the electrostatics solver-dispatch decision, and where an accelerated solver is not
bit-parity the solver identity becomes model state serialized with the checkpoint.

| id | feature | source | disposition | pinned by |
|---|---|---|---|---|
| `scf.lc_mace` | LC-MACE — local charge: multipole moments read out from descriptors; non-SCF | `ACEsuit/mace-scf` | KEEP — reimplement; the cheapest of the five, no implicit-diff hook needed | ⚠️ gap (no golden yet) |
| `scf.lsc_mace` | LSC-MACE — local split charge: charge + multipoles conserving local charge flow; non-SCF | `ACEsuit/mace-scf` | KEEP — reimplement | ⚠️ gap (no golden yet) |
| `scf.qeq` | MACE-QEq — charge equilibration, solved self-consistently with implicit differentiation | `ACEsuit/mace-scf` | KEEP — reimplement; the load-bearing test case for the SCF hook | ⚠️ gap (no golden yet) |
| `scf.fixedpoint` | FixedPointSCF — Kohn–Sham-like SCF cycles for multipole moments | `ACEsuit/mace-scf` | KEEP — reimplement; incremental once the hook is validated by QEq | ⚠️ gap (no golden yet) |
| `scf.energy_functional` | EnergyFunctionalSCF — an alternative functional; upstream marks it experimental | `ACEsuit/mace-scf` | KEEP — reimplement; its golden pins whatever behaviour it has today, experimental or not | ⚠️ gap (no golden yet) |

## 21. Foundation-model migration roster

The published, **trained** artifacts to migrate, in one auditable place. Conversion is one-shot
(legacy pickle → neutral artifact → v1), never a runtime load path; multi-head artifacts convert with
their heads intact, and a single-head export is `mace model select-head`. Model *architectures* are
§6 — a different axis: those are reimplemented, these are converted.

| id | feature | source | disposition | pinned by |
|---|---|---|---|---|
| `fm.mace_mp` | MACE-MP aliases (small/medium/large, 0b/0b2/0b3, MPA-0, OMAT, MATPES) | published artifacts | KEEP — convert | `tests/golden/test_foundation_goldens.py::test_foundation_model_reproduces_its_reference` |
| `fm.mace_mh` | `mh-0` / `mh-1` — the MACE-MP multi-head releases | published artifacts | KEEP — convert with heads intact | ⚠️ gap (needs an mh-0 case in `tests/golden/test_foundation_goldens.py`; it is also the Density-block evidence, §7) |
| `fm.mace_off` | MACE-OFF (OFF23 small/medium/large) | published artifacts | KEEP — convert | `tests/golden/test_foundation_goldens.py::test_foundation_model_reproduces_its_reference` |
| `fm.mace_mdp` | MACE-MDP — the dielectric family (`AtomicDielectricMACE`), dipole and polarizability | published artifacts | KEEP — convert | `tests/golden/test_mdp_foundation.py::test_mdp_foundation_reproduces_its_reference` |
| `fm.mace_omol` | MACE-OMOL — multi-head, `head="omol"` | published artifacts | KEEP — convert with heads intact | `tests/foundations/test_foundations.py::test_mace_omol_elements_subset_reproduces_energy_forces` |
| `fm.mace_polar` | MACE-Polar S/M/L (`PolarMACE`) | published artifacts | KEEP — convert, alongside the electrostatics work | `tests/golden/test_polar_foundation.py::test_polar_foundation_reproduces_its_reference` |
| `fm.mace_anicc` | MACE-ANI-CC — the 2023 organic-chemistry model | `mace/calculators/foundations_models/ani500k_large_CC.model` | DROP — superseded by MACE-OFF, and the only artifact bundled inside the wheel; the release notes say "use MACE-OFF" | — |

## 22. Open items

Not rows: what this inventory turned up that is still owed.

- **The `⚠️ gap` column.** Every gap is either a test still to write or a conscious downgrade, and
  each row says which. The checker's `tally:` line is the authoritative count; a number written in
  prose here would be stale by the next merge.
- **New goldens this inventory implies**, beyond the ones that exist: MP-0b2 or mh-0 (the
  Density interaction blocks turned out to be published-model architecture, not a research variant),
  MACE-OMOL, MACE-MDP, and the five `mace-scf` families.
- **The vendored-`torch_geometric` retirement got harder, not easier.** The `[magnetic]` extra declares
  external `torch-geometric` and `mace/data/augmentation.py` imports the real package and the vendored
  one at once, so deleting the copy means removing both consumers first and then verifying that
  neither survives. A code search over downstream projects belongs in the same change: the vendored
  modules are importable, so somebody outside this repo may be importing them.
- **Three spellings of one quantity.** The per-layer energy contributions are `contributions` in the
  model output, `--return_contributions` on the eval CLI and `BO_contributions` in the written XYZ;
  per-atom energies are `node_energy` in the model, `energies` *and* `node_energy` (with the E0s
  subtracted) in the calculator, and `node_energies` in the XYZ; the LES latent multipoles are
  `latent_alphas` in the model and `LES_alphas` in the calculator. One name per quantity, with the
  old spellings mapped in the release notes, is the only way out of this.

## Tally

Do not gate on any count written in prose. `check_inventory.py` prints a `tally:` line with the live
row count, the KEEP/MERGE/DROP split, the number of `⚠️ gap` rows and the number of `REVIEW`
dispositions (which must be zero) — those figures are authoritative, and they are recomputed from the
file on every run.

| section | rows | keyed on |
|---|---|---|
| 1. CLI entry points | 12 | `setup.cfg` console_scripts |
| 2. `--model` choices | 10 | the `choices=` list |
| 3. `mace_run_train` flags | 184 | argparse dest |
| 4. `mace_prepare_data` flags | 26 | argparse dest |
| 5. Other CLI flags | 111 | argparse dest, per parser |
| 6. Model-level classes | 12 | class name |
| 7. Registries | 21 | registry key |
| 8. Loss classes | 10 | class name |
| 9. Calculator constructor + exports | 23 + 9 | parameter name, `__all__` |
| 10. Optional extras | 12 | extra name |
| 11. Model output keys | 43 | dict key |
| 12. Calculator + eval output keys | 31 + 13 | results key, written key |
| 13. Environment variables | 9 + 3 | variable name |
| 14. Pytest markers | 13 | marker name |
| 15. Default property keys | 13 | enum member |
| 16. Calculator methods, loader kwargs, aliases | 16 | hand-written |
| 17. LAMMPS runtime | 5 | hand-written |
| 18. Documentation surface | 17 | published page |
| 19. Package-level surfaces | 19 | hand-written |
| 20. `mace-scf` families | 5 | hand-written |
| 21. Foundation-model roster | 7 | published artifact |
