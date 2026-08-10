# MACE v0.3.x → v1.0 functionality inventory

Every enumerable feature of the current tree, with a **KEEP / MERGE / DROP** disposition, the Phase 0
test that pins its behaviour, the ticket that carries it into v1, and the ticket that eventually
deletes its legacy implementation. It is the completeness contract of the rewrite: **nothing leaves
silently.** A feature v1 will not carry gets a `DROP` row with a reason; the *absence* of a row is a
bug in this file, not a decision.

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
| pinned by | the Phase 0 test or existing test that protects the behaviour, or `⚠️ gap`. Must **resolve**: see below |
| destination | the v1 ticket that carries the capability |
| retirement | the `RET-*` that deletes the legacy implementation, or `n/a — why` |
| status | `todo` until an implementing ticket lands it, then `done` |

**A pin has to resolve.** "Not empty" is not a rule: it accepts the literal string `TODO`, and the
gate then finishes with *all sources covered*. So the `pinned by` cell must open with one of four
things, and every backticked path it names anywhere must exist on disk — with its `::node_id`, if it
carries one, actually declared there. A pin naming a test that was renamed or never written is worse
than a gap marker, because it reads as coverage.

| pin opens with | means |
|---|---|
| a gap marker, `⚠️ gap (…)` | nothing pins this yet; counted in the tally and owed before the phase gate |
| a backticked path under `tests/` | an existing test, file or directory, optionally `::test_name` |
| a ticket id, `P0-5` / `CORE-4` / … | a test the named ticket will write. Only the families listed in `TICKET_PREFIXES` — and a family the inventory uses but the checker does not know fails too, so the constant cannot rot in either direction |
| one of two named CI jobs | `the suite itself` and `the lint job itself`, for the two `setup.cfg` extras, where the only thing that can fail is a job installing them. Allowed by name, one entry each, with a written reason |

**Disposition vocabulary.** `KEEP` = the functionality must exist in v1, possibly renamed or reshaped
(usually as a config field — the flag *as a flag* disappears into the config system). `MERGE` =
subsumed by a named, more general v1 mechanism. `DROP` = intentionally removed; the reason feeds
REL-1's migration guide. There is no fourth value: `REVIEW` is not a disposition the gate accepts, so
an undecided row fails the build rather than sitting in the file.

**Maintenance rule.** Implementing tickets flip their rows to `done` in the same PR; each phase gate
audits the rows in its scope; the rc1 checklist requires zero unresolved rows. When a source changes,
the checker names the rows to add or delete — re-run it rather than re-reading the diff.

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
`beta2_schedulefree`, `warmup_steps_schedulefree`) — the entire tuning surface of an optimizer the
config schema otherwise covers by naming the optimizer alone. Five were folded into a cell listing
several flags under one shared disposition; one was stated in prose between two tables. Neither shape
is a disposition a checker can read.

**What the gate enforces.** Seventeen set comparisons — entry points, the two training parsers, the
thirteen `mace/cli` parsers, the `--model` choices, model classes, registries, losses, calculator
params, calculator exports, extras, the three output-key surfaces, the `MACE_*` environment
variables, the pytest markers, and the default property keys — plus, on every row: a valid
disposition, a reason on every `DROP`, a pinning test or an explicit `⚠️ gap` on every `KEEP`/`MERGE`
*that resolves to a file, a ticket or a named CI job*, a destination ticket, a retirement ticket, and
no duplicate ids. Four conditions fail a dest: no row,
an empty disposition, a `REVIEW` disposition, and a row for a dest the source no longer declares —
so a renamed flag cannot leave a stale row behind claiming coverage.

**The `⚠️ gap` column is the open work.** Each gap is either a test added by P0-1/2/3a/3b/3c/5/6/7 or
a conscious downgrade, and it must be resolved before the Phase 0 gate closes; each is also carried
into its destination ticket as an explicit acceptance criterion, so closure is visible where the work
happens rather than only here. The authoritative gap count is the checker's `tally:` line, never a
number written in prose. Read that count as rows, not as tasks: because the key is the dest, one
missing test marks every dest it would have covered — the sixteen dests of `mace_finetuning_select`
carry one gap between them, not sixteen — so the rows cluster into roughly sixty distinct pieces of
work.

---

## 1. CLI entry points (12)

Twelve console scripts in `setup.cfg`. Three further CLIs live in `mace/cli/` with a `main()` and an
argparser but **no entry point** (`convert_e3nn_oeq`, `convert_oeq_e3nn`, `convert_e3nn_hybrid`);
they have no row here because there is nothing registered to keep or drop, but their flags are
inventoried in §5 and they die with `mace_e3nn_cueq` under RET-3.

| id | feature | source | disposition | pinned by | destination | retirement | status |
|---|---|---|---|---|---|---|---|
| `ep.mace_run_train` | `mace.cli.run_train:main` | `setup.cfg` | KEEP — becomes `mace train` | P0-5 | TRN-1, CLI-1 | RET-6 | todo |
| `ep.mace_eval_configs` | `mace.cli.eval_configs:main` | `setup.cfg` | KEEP — the capability survives, but the CLI shrinks to a thin `mace eval` adapter over the ASE calculator (read configs → calculator → write XYZ); most of its ~400 lines is argparse duplicating what the calculator already does, and one numerical path is cheaper to pin than two | P0-5 | CLI-1 | RET-6 | todo |
| `ep.mace_prepare_data` | `mace.cli.preprocess_data:main` | `setup.cfg` | KEEP — becomes `mace data prepare` | `tests/workflows/test_preprocess.py` | DATA-1, CLI-1 | RET-6 | todo |
| `ep.mace_create_lammps_model` | `mace.cli.create_lammps_model:main` | `setup.cfg` | KEEP — becomes `mace export lammps` | P0-5 (export golden) | DEP-2 | RET-5 | todo |
| `ep.mace_select_head` | `mace.cli.select_head:main` | `setup.cfg` | KEEP — becomes `mace model select-head` | ⚠️ gap (add to P0-5) | CLI-1 | RET-6 | todo |
| `ep.mace_plot_train` | `mace.cli.plot_train:main` | `setup.cfg` | KEEP — reduced: the basic plot subcommand stays (per-head loss curves, cheap over a structured log); `--plot_interaction_e` goes (§3.1) | ⚠️ gap (smoke: parse a real results log, write a file) | CLI-1 | RET-6 | todo |
| `ep.mace_polar_density_cube` | `mace.cli.polar_density_cube:main` | `setup.cfg` | KEEP — the electrostatics extra's density-cube tool | `tests/extensions/polar/test_polar_density_cube.py` (port cases) | ELEC-2 | RET-4 | todo |
| `ep.mace_finetuning_select` | `mace.cli.fine_tuning_select:main` | `setup.cfg` | MERGE — absorbed into the integrated fine-tuning pipeline (`mace train` with a fine-tuning config) | P0-5 (fine-tuning contract) | FT-1 | RET-6 | todo |
| `ep.mace_convert_device` | `mace.cli.convert_device:main` | `setup.cfg` | KEEP — becomes `mace model convert-device`; explicitly **not** one of the five weight converters, because it converts device/dtype and not backend layout | ⚠️ gap (add to P0-5) | CLI-1 | RET-6 | todo |
| `ep.mace_e3nn_cueq` | `mace.cli.convert_e3nn_cueq:main` | `setup.cfg` | DROP — v1 weights are canonical and backend dispatch is automatic, so there is nothing left to convert between backends | P0-4 pins the backend numerics | REL-1 (doc) | RET-3 | todo |
| `ep.mace_cueq_to_e3nn` | `mace.cli.convert_cueq_e3nn:main` | `setup.cfg` | DROP — idem, the reverse direction | P0-4 | REL-1 (doc) | RET-3 | todo |
| `ep.mace_active_learning_md` | `mace.cli.active_learning_md:main` | `setup.cfg` | DROP — out of v1.0 scope: an ASE MD loop a user writes in ~30 lines over the calculator, at the cost of MACE owning thermostat, timestep and trajectory I/O. The committee variance it consumes stays in `calculate` (§16); REL-1 documents the recipe | — | REL-1 (doc) | RET-6 | todo |

## 2. `--model` choices (10)

The CLI-selectable model names. A separate set from the model classes of §6 on purpose: **two of the
ten name a class that exists nowhere in the tree.** `BOTNet` and `ScaleShiftBOTNet` reach only
`RuntimeError("... is deprecated, use MACE instead")` in `mace/tools/model_script_utils.py:374-378`.
v1's model enum does not carry them, so an unknown value fails through ordinary config validation
instead of a hand-written runtime raise.

| id | feature | source | disposition | pinned by | destination | retirement | status |
|---|---|---|---|---|---|---|---|
| `choice.BOTNet` | `--model BOTNet` | `mace/tools/arg_parser.py:138` | DROP — no `BOTNet` class exists anywhere in the tree; the choice reaches only a deprecation raise. REL-1 names MACE as the replacement | — | REL-1 (doc) | RET-6 | todo |
| `choice.ScaleShiftBOTNet` | `--model ScaleShiftBOTNet` | `mace/tools/arg_parser.py:143` | DROP — identical shape: no class, only a deprecation raise | — | REL-1 (doc) | RET-6 | todo |
| `choice.MACE` | `--model MACE` | `mace/tools/arg_parser.py:139` | MERGE — model composition is config-driven, not a class name | P0-1 | ARCH-3, CFG-1 | RET-1 | todo |
| `choice.ScaleShiftMACE` | `--model ScaleShiftMACE` | `mace/tools/arg_parser.py:140` | MERGE — idem; the default energy model becomes the default configuration | P0-1 | ARCH-3, CFG-1 | RET-1 | todo |
| `choice.PolarMACE` | `--model PolarMACE` | `mace/tools/arg_parser.py:141` | MERGE — idem, selected by declaring the electrostatics observables | P0-3a | ELEC-2 | RET-4 | todo |
| `choice.MACELES` | `--model MACELES` | `mace/tools/arg_parser.py:142` | MERGE — idem, selected by declaring the LES long-range term | P0-3c | ELEC-4 | RET-4 | todo |
| `choice.AtomicDipolesMACE` | `--model AtomicDipolesMACE` | `mace/tools/arg_parser.py:144` | MERGE — idem, selected by declaring the dipole observable | P0-3a | ARCH-3 | RET-4 | todo |
| `choice.AtomicDielectricMACE` | `--model AtomicDielectricMACE` | `mace/tools/arg_parser.py:145` | MERGE — idem, dipole + polarizability observables | ⚠️ gap (MDP golden → P0-3a) | ARCH-3, FT-4 | RET-4 | todo |
| `choice.EnergyDipolesMACE` | `--model EnergyDipolesMACE` | `mace/tools/arg_parser.py:146` | MERGE — idem, energy + dipole observables | ⚠️ gap (add a golden if any published model uses it) | ARCH-3 | RET-4 | todo |
| `choice.MagneticScaleShiftMACE` | `--model MagneticScaleShiftMACE` | `mace/tools/arg_parser.py:147` | MERGE — idem; the only magnetic entry in the choices | `tests/extensions/magnetic` + P0-3b | MAG-1 | RET-4 | todo |

## 3. `mace_run_train` flags — 184 dests

One row per **dest** of `build_default_arg_parser` (`mace/tools/arg_parser.py`), which is what a
knob is; the option strings that spell it are in the feature cell. 184 dests carry 194 option
strings — the 10-string surplus is the `--swa_*`/`--stage_two_*` alias pairs. Retirement for every
row in this section is **RET-6**, which deletes `mace/tools/`.

### 3.0 Config file (1)

The one dest registered with `parser.add` instead of `add_argument`, and the only optional-dependency flag in the parser: without configargparse installed the option silently does not exist.

| id | feature | source | disposition | pinned by | destination | retirement | status |
|---|---|---|---|---|---|---|---|
| `train.config` | `--config` | `mace/tools/arg_parser.py:23` | MERGE — the v1 config system makes this first-class (TOML/YAML/JSON, always available, resolved config saved as run metadata) | ⚠️ gap (no test exercises the YAML path) | CORE-2, CFG-1 | RET-6 | todo |

### 3.1 Run and infrastructure (17)

Group default: KEEP as the `runtime` / `output` config sections (CFG-1). The four `*_dir` flags MERGE into one work-dir layout convention rather than four independent paths.

| id | feature | source | disposition | pinned by | destination | retirement | status |
|---|---|---|---|---|---|---|---|
| `train.name` | `--name` | `mace/tools/arg_parser.py:34` | KEEP | P0-5 | CFG-1 | RET-6 | todo |
| `train.seed` | `--seed` | `mace/tools/arg_parser.py:35` | KEEP | P0-1 (determinism) | CFG-1 | RET-6 | todo |
| `train.work_dir` | `--work_dir` | `mace/tools/arg_parser.py:39` | KEEP | P0-5 | CFG-1 | RET-6 | todo |
| `train.log_dir` | `--log_dir` | `mace/tools/arg_parser.py:45` | MERGE — single work-dir layout convention | P0-5 | CFG-1 | RET-6 | todo |
| `train.model_dir` | `--model_dir` | `mace/tools/arg_parser.py:48` | MERGE — idem | P0-5 | CFG-1 | RET-6 | todo |
| `train.checkpoints_dir` | `--checkpoints_dir` | `mace/tools/arg_parser.py:51` | MERGE — idem | P0-5 | CFG-1 | RET-6 | todo |
| `train.results_dir` | `--results_dir` | `mace/tools/arg_parser.py:57` | MERGE — idem | P0-5 | CFG-1 | RET-6 | todo |
| `train.downloads_dir` | `--downloads_dir` | `mace/tools/arg_parser.py:60` | MERGE — XDG cache-dir convention | ⚠️ gap (cache-path contract) | FM-4 | RET-6 | todo |
| `train.device` | `--device` | `mace/tools/arg_parser.py:65` | KEEP | P0-5 | CFG-1 | RET-6 | todo |
| `train.default_dtype` | `--default_dtype` | `mace/tools/arg_parser.py:72` | MERGE — `PrecisionConfig` | P0-1 (fp64 goldens) | BKD-2 | RET-6 | todo |
| `train.distributed` | `--distributed` | `mace/tools/arg_parser.py:79` | KEEP | `tests/workflows/test_distributed.py` | TRN-4 | RET-6 | todo |
| `train.launcher` | `--launcher` | `mace/tools/arg_parser.py:85` | KEEP | `tests/workflows/test_distributed.py` | TRN-4 | RET-6 | todo |
| `train.log_level` | `--log_level` | `mace/tools/arg_parser.py:90` | KEEP | ⚠️ gap (trivial; conscious downgrade candidate) | TRN-3 | RET-6 | todo |
| `train.plot` | `--plot` | `mace/tools/arg_parser.py:93` | KEEP | ⚠️ gap (with the `mace_plot_train` smoke) | CLI-1 | RET-6 | todo |
| `train.plot_frequency` | `--plot_frequency` | `mace/tools/arg_parser.py:100` | KEEP | ⚠️ gap (idem) | CLI-1 | RET-6 | todo |
| `train.plot_interaction_e` | `--plot_interaction_e` | `mace/tools/arg_parser.py:107` | DROP — niche diagnostic that drags model introspection into the plotting path | — | REL-1 (doc) | RET-6 | todo |
| `train.error_table` | `--error_table` | `mace/tools/arg_parser.py:114` | KEEP — the error-table types | P0-5 (table printed) | TRN-3 | RET-6 | todo |

### 3.2 Model architecture (26)

Group default: KEEP as the `model` config section (CFG-1); the defaults are pinned by the P0-1 anchors.

| id | feature | source | disposition | pinned by | destination | retirement | status |
|---|---|---|---|---|---|---|---|
| `train.model` | `--model` | `mace/tools/arg_parser.py:134` | MERGE — model composition is config-driven (`BaseMACE` + declared outputs), not a class name | P0-1 | ARCH-3, CFG-1 | RET-6 | todo |
| `train.r_max` | `--r_max` | `mace/tools/arg_parser.py:151` | KEEP | P0-1 | CFG-1 | RET-6 | todo |
| `train.radial_type` | `--radial_type` | `mace/tools/arg_parser.py:154` | KEEP — bessel / gaussian / chebyshev | P0-7 | ARCH-1 | RET-6 | todo |
| `train.num_radial_basis` | `--num_radial_basis` | `mace/tools/arg_parser.py:161` | KEEP | P0-7 | ARCH-1 | RET-6 | todo |
| `train.num_cutoff_basis` | `--num_cutoff_basis` | `mace/tools/arg_parser.py:167` | KEEP | P0-7 | ARCH-1 | RET-6 | todo |
| `train.pair_repulsion` | `--pair_repulsion` | `mace/tools/arg_parser.py:173` | KEEP — the ZBL short-range term | P0-1 (anchor has ZBL on) + P0-7 | ARCH-1 | RET-6 | todo |
| `train.distance_transform` | `--distance_transform` | `mace/tools/arg_parser.py:179` | KEEP | P0-7 | ARCH-1 | RET-6 | todo |
| `train.apply_cutoff` | `--apply_cutoff` | `mace/tools/arg_parser.py:185` | KEEP | ⚠️ gap (add case to P0-7) | ARCH-1 | RET-6 | todo |
| `train.use_last_readout_only` | `--use_last_readout_only` | `mace/tools/arg_parser.py:191` | MERGE — readout policy: once you declare which layers read out, 'only the last' is configuration, not a boolean | ⚠️ gap (add case to P0-7) | ARCH-3 | RET-6 | todo |
| `train.use_embedding_readout` | `--use_embedding_readout` | `mace/tools/arg_parser.py:197` | MERGE — idem ('also read the embedding layer') | ⚠️ gap (add case to P0-7) | ARCH-3 | RET-6 | todo |
| `train.interaction` | `--interaction` | `mace/tools/arg_parser.py:203` | KEEP — see the registry rows in §7 for which classes survive | P0-1 (default) | ARCH-2 | RET-6 | todo |
| `train.interaction_first` | `--interaction_first` | `mace/tools/arg_parser.py:219` | KEEP | P0-1 | ARCH-2 | RET-6 | todo |
| `train.max_ell` | `--max_ell` | `mace/tools/arg_parser.py:234` | KEEP | P0-1 | ARCH-2 | RET-6 | todo |
| `train.correlation` | `--correlation` | `mace/tools/arg_parser.py:237` | KEEP | P0-1 | ARCH-2 | RET-6 | todo |
| `train.use_reduced_cg` | `--use_reduced_cg` | `mace/tools/arg_parser.py:240` | MERGE — a CG-representation choice the backend makes, not a modelling decision a user can judge, and it changes numerics; `convert_e3nn_hybrid.py` defaults it to `True`, so checkpoints carry it and the converter must read it rather than assume | ⚠️ gap (CORE-4 triple-oracle basis) | BKD-1, CORE-4 | RET-6 | todo |
| `train.use_so3` | `--use_so3` | `mace/tools/arg_parser.py:246` | DROP — a global parity-convention switch that doubles the irrep-handling surface in exactly the layer v1 rewrites; no published model sets it | — | REL-1 (doc) | RET-6 | todo |
| `train.use_agnostic_product` | `--use_agnostic_product` | `mace/tools/arg_parser.py:252` | KEEP — MACE-Polar S/M/L set it, so it is foundation-model architecture, not a research knob | ⚠️ gap (covered by the P0-3a polar golden) | ARCH-2 | RET-6 | todo |
| `train.num_interactions` | `--num_interactions` | `mace/tools/arg_parser.py:258` | KEEP | P0-1 | CFG-1 | RET-6 | todo |
| `train.MLP_irreps` | `--MLP_irreps` | `mace/tools/arg_parser.py:261` | KEEP — the non-linear readout's hidden irreps | P0-1 | ARCH-3 | RET-6 | todo |
| `train.radial_MLP` | `--radial_MLP` | `mace/tools/arg_parser.py:267` | KEEP | P0-1 | ARCH-2 | RET-6 | todo |
| `train.hidden_irreps` | `--hidden_irreps` | `mace/tools/arg_parser.py:273` | KEEP | P0-1 | CFG-1 | RET-6 | todo |
| `train.edge_irreps` | `--edge_irreps` | `mace/tools/arg_parser.py:279` | KEEP — MACE-Polar S/M/L set it (`128x0e` → `128x0e+128x1o+128x2e`) | ⚠️ gap (covered by the P0-3a polar golden) | ARCH-2 | RET-6 | todo |
| `train.use_edge_irreps_first` | `--use_edge_irreps_first` | `mace/tools/arg_parser.py:285` | KEEP — the first-layer variant of `edge_irreps`; splitting them would leave a half-supported knob, and no published checkpoint stores the attribute | ⚠️ gap (add case to P0-7) | ARCH-2 | RET-6 | todo |
| `train.num_channels` | `--num_channels` | `mace/tools/arg_parser.py:292` | KEEP — shortcut for `hidden_irreps` | P0-1 | CFG-1 | RET-6 | todo |
| `train.max_L` | `--max_L` | `mace/tools/arg_parser.py:298` | KEEP — idem | P0-1 | CFG-1 | RET-6 | todo |
| `train.gate` | `--gate` | `mace/tools/arg_parser.py:304` | KEEP — see the gate rows in §7 | P0-1 | ARCH-3 | RET-6 | todo |

### 3.3 PolarMACE architecture (15)

Group default: KEEP as the electrostatics-extra config section; destination ELEC-2, pinned by the P0-3a polar golden.

| id | feature | source | disposition | pinned by | destination | retirement | status |
|---|---|---|---|---|---|---|---|
| `train.kspace_cutoff_factor` | `--kspace_cutoff_factor` | `mace/tools/arg_parser.py:311` | KEEP | P0-3a | ELEC-2 | RET-6 | todo |
| `train.atomic_multipoles_max_l` | `--atomic_multipoles_max_l` | `mace/tools/arg_parser.py:317` | KEEP | P0-3a | ELEC-2 | RET-6 | todo |
| `train.atomic_multipoles_smearing_width` | `--atomic_multipoles_smearing_width` | `mace/tools/arg_parser.py:323` | KEEP | P0-3a | ELEC-2 | RET-6 | todo |
| `train.field_feature_max_l` | `--field_feature_max_l` | `mace/tools/arg_parser.py:329` | KEEP | P0-3a | ELEC-2 | RET-6 | todo |
| `train.field_feature_widths` | `--field_feature_widths` | `mace/tools/arg_parser.py:335` | KEEP | P0-3a | ELEC-2 | RET-6 | todo |
| `train.field_feature_norms` | `--field_feature_norms` | `mace/tools/arg_parser.py:341` | KEEP | P0-3a | ELEC-2 | RET-6 | todo |
| `train.num_recursion_steps` | `--num_recursion_steps` | `mace/tools/arg_parser.py:347` | KEEP | P0-3a | ELEC-2 | RET-6 | todo |
| `train.field_si` | `--field_si` | `mace/tools/arg_parser.py:353` | KEEP | P0-3a | ELEC-2 | RET-6 | todo |
| `train.include_electrostatic_self_interaction` | `--include_electrostatic_self_interaction` | `mace/tools/arg_parser.py:359` | KEEP | P0-3a | ELEC-2 | RET-6 | todo |
| `train.add_local_electron_energy` | `--add_local_electron_energy` | `mace/tools/arg_parser.py:365` | KEEP | P0-3a | ELEC-2 | RET-6 | todo |
| `train.quadrupole_feature_corrections` | `--quadrupole_feature_corrections` | `mace/tools/arg_parser.py:371` | KEEP | P0-3a | ELEC-2 | RET-6 | todo |
| `train.return_electrostatic_potentials` | `--return_electrostatic_potentials` | `mace/tools/arg_parser.py:377` | MERGE — an observable declared in the output spec, not a model flag | ⚠️ gap (add to the P0-3a polar case) | CORE-1, ELEC-2 | RET-6 | todo |
| `train.field_norm_factor` | `--field_norm_factor` | `mace/tools/arg_parser.py:383` | KEEP | P0-3a | ELEC-2 | RET-6 | todo |
| `train.fixedpoint_update_config` | `--fixedpoint_update_config` | `mace/tools/arg_parser.py:389` | KEEP — the fixed-point solver settings; the expert electrostatics config section | ⚠️ gap (add to the P0-3a polar case) | ELEC-1, ELEC-2 | RET-6 | todo |
| `train.field_readout_config` | `--field_readout_config` | `mace/tools/arg_parser.py:395` | KEEP — idem | ⚠️ gap (add to the P0-3a polar case) | ELEC-1, ELEC-2 | RET-6 | todo |

### 3.4 Outputs and scaling (8)

Group default: MERGE into the declarative observable specification (CORE-1) — an observable that is declared is computed, so the `--compute_*` booleans stop being independent flags.

| id | feature | source | disposition | pinned by | destination | retirement | status |
|---|---|---|---|---|---|---|---|
| `train.scaling` | `--scaling` | `mace/tools/arg_parser.py:401` | KEEP — see the scaling rows in §7 | P0-1 | ARCH-3 | RET-6 | todo |
| `train.avg_num_neighbors` | `--avg_num_neighbors` | `mace/tools/arg_parser.py:408` | MERGE — dataset statistics + model metadata | P0-1 | DATA-1 | RET-6 | todo |
| `train.compute_avg_num_neighbors` | `--compute_avg_num_neighbors` | `mace/tools/arg_parser.py:414` | MERGE — idem | P0-1 | DATA-1 | RET-6 | todo |
| `train.compute_stress` | `--compute_stress` | `mace/tools/arg_parser.py:420` | MERGE — observable spec (stress declared ⇒ computed) | P0-6 | CORE-1 | RET-6 | todo |
| `train.compute_forces` | `--compute_forces` | `mace/tools/arg_parser.py:426` | MERGE — idem | P0-6 | CORE-1 | RET-6 | todo |
| `train.compute_polarizability` | `--compute_polarizability` | `mace/tools/arg_parser.py:432` | MERGE — idem | P0-3a | CORE-1, ELEC-2 | RET-6 | todo |
| `train.compute_atomic_dipole` | `--compute_atomic_dipole` | `mace/tools/arg_parser.py:438` | MERGE — idem | P0-3a | CORE-1 | RET-6 | todo |
| `train.compute_magforces` | `--compute_magforces` | `mace/tools/arg_parser.py:444` | MERGE — idem: `dE/dm` is a declared derivative exactly like forces and stress | P0-3b | CORE-1, MAG-1 | RET-6 | todo |

### 3.5 Data, files and property keys (29)

Group default: the `*_key` flags MERGE into the property-key convention (CORE-3); the file/loading flags KEEP as the `data` config section (DATA-1, CFG-1).

| id | feature | source | disposition | pinned by | destination | retirement | status |
|---|---|---|---|---|---|---|---|
| `train.train_file` | `--train_file` | `mace/tools/arg_parser.py:452` | KEEP | P0-5 | CFG-1, DATA-1 | RET-6 | todo |
| `train.valid_file` | `--valid_file` | `mace/tools/arg_parser.py:458` | KEEP | P0-5 | CFG-1, DATA-1 | RET-6 | todo |
| `train.test_file` | `--test_file` | `mace/tools/arg_parser.py:472` | KEEP | P0-5 | CFG-1, DATA-1 | RET-6 | todo |
| `train.test_dir` | `--test_dir` | `mace/tools/arg_parser.py:477` | KEEP | P0-5 | CFG-1, DATA-1 | RET-6 | todo |
| `train.valid_fraction` | `--valid_fraction` | `mace/tools/arg_parser.py:465` | KEEP | P0-5 | CFG-1 | RET-6 | todo |
| `train.multi_processed_test` | `--multi_processed_test` | `mace/tools/arg_parser.py:484` | MERGE — the data layer infers sharding from the dataset; whether a test set is split across files is not something the user should have to declare and get wrong (today it is a bare `if` in `run_train.py`) | ⚠️ gap (add to P0-5) | DATA-2 | RET-6 | todo |
| `train.num_workers` | `--num_workers` | `mace/tools/arg_parser.py:491` | KEEP | ⚠️ gap (perf knob; conscious downgrade candidate) | DATA-1 | RET-6 | todo |
| `train.pin_memory` | `--pin_memory` | `mace/tools/arg_parser.py:497` | KEEP | ⚠️ gap (perf knob; conscious downgrade candidate) | DATA-1 | RET-6 | todo |
| `train.atomic_numbers` | `--atomic_numbers` | `mace/tools/arg_parser.py:503` | MERGE — statistics / model metadata | P0-7 | DATA-1 | RET-6 | todo |
| `train.mean` | `--mean` | `mace/tools/arg_parser.py:510` | MERGE — statistics override | ⚠️ gap (add to the P0-6 E0/stats cases) | DATA-1 | RET-6 | todo |
| `train.std` | `--std` | `mace/tools/arg_parser.py:517` | MERGE — idem | ⚠️ gap (add to the P0-6 E0/stats cases) | DATA-1 | RET-6 | todo |
| `train.statistics_file` | `--statistics_file` | `mace/tools/arg_parser.py:524` | KEEP | `tests/workflows/test_preprocess.py` | DATA-1 | RET-6 | todo |
| `train.les_arguments` | `--les_arguments` | `mace/tools/arg_parser.py:531` | KEEP — the LES extra's solver settings | `tests/extensions/les` + P0-3c | ELEC-1, ELEC-4 | RET-6 | todo |
| `train.E0s` | `--E0s` | `mace/tools/arg_parser.py:538` | KEEP — explicit / average / estimated / foundation | P0-6 | ARCH-3, DATA-1 | RET-6 | todo |
| `train.keep_isolated_atoms` | `--keep_isolated_atoms` | `mace/tools/arg_parser.py:646` | KEEP | P0-7 | DATA-1 | RET-6 | todo |
| `train.config_type_weights` | `--config_type_weights` | `mace/tools/arg_parser.py:882` | KEEP — per-config-type loss weighting | P0-6 | TRN-2 | RET-6 | todo |
| `train.energy_key` | `--energy_key` | `mace/tools/arg_parser.py:672` | MERGE — property-key convention of the observable spec | P0-7 | CORE-3 | RET-6 | todo |
| `train.forces_key` | `--forces_key` | `mace/tools/arg_parser.py:678` | MERGE — idem | P0-7 | CORE-3 | RET-6 | todo |
| `train.virials_key` | `--virials_key` | `mace/tools/arg_parser.py:684` | MERGE — idem | P0-7 | CORE-3 | RET-6 | todo |
| `train.stress_key` | `--stress_key` | `mace/tools/arg_parser.py:690` | MERGE — idem | P0-7 | CORE-3 | RET-6 | todo |
| `train.dipole_key` | `--dipole_key` | `mace/tools/arg_parser.py:696` | MERGE — idem | P0-7 | CORE-3 | RET-6 | todo |
| `train.polarizability_key` | `--polarizability_key` | `mace/tools/arg_parser.py:702` | MERGE — idem | P0-7 | CORE-3 | RET-6 | todo |
| `train.charges_key` | `--charges_key` | `mace/tools/arg_parser.py:726` | MERGE — idem | P0-7 | CORE-3 | RET-6 | todo |
| `train.head_key` | `--head_key` | `mace/tools/arg_parser.py:720` | MERGE — idem | P0-7 | CORE-3 | RET-6 | todo |
| `train.elec_temp_key` | `--elec_temp_key` | `mace/tools/arg_parser.py:732` | KEEP — graph-level input feature | ⚠️ gap (add parse case to P0-7) | CORE-3, ARCH-3 | RET-6 | todo |
| `train.total_spin_key` | `--total_spin_key` | `mace/tools/arg_parser.py:738` | KEEP — idem | ⚠️ gap (add parse case to P0-7) | CORE-3, ARCH-3 | RET-6 | todo |
| `train.total_charge_key` | `--total_charge_key` | `mace/tools/arg_parser.py:744` | KEEP — idem | ⚠️ gap (add parse case to P0-7) | CORE-3, ARCH-3 | RET-6 | todo |
| `train.embedding_specs` | `--embedding_specs` | `mace/tools/arg_parser.py:750` | KEEP — categorical / graph-level embeddings | ⚠️ gap (port `tests/workflows/test_embedding_train.py` cases) | ARCH-3, ARCH-5 | RET-6 | todo |
| `train.skip_evaluate_heads` | `--skip_evaluate_heads` | `mace/tools/arg_parser.py:773` | KEEP | ⚠️ gap (add to P0-5 multihead case) | TRN-3 | RET-6 | todo |

### 3.6 Fine-tuning, multihead and foundation models (24)

Group default: KEEP, destination FT-1/2/3.

| id | feature | source | disposition | pinned by | destination | retirement | status |
|---|---|---|---|---|---|---|---|
| `train.foundation_model` | `--foundation_model` | `mace/tools/arg_parser.py:1019` | KEEP | P0-5 (fine-tuning contract) | FT-1 | RET-6 | todo |
| `train.foundation_model_kwargs` | `--foundation_model_kwargs` | `mace/tools/arg_parser.py:1025` | KEEP | P0-5 | FT-1 | RET-6 | todo |
| `train.foundation_model_readout` | `--foundation_model_readout` | `mace/tools/arg_parser.py:1031` | KEEP | P0-5 | FT-1 | RET-6 | todo |
| `train.multiheads_finetuning` | `--multiheads_finetuning` | `mace/tools/arg_parser.py:573` | KEEP | P0-5 | FT-1 | RET-6 | todo |
| `train.heads` | `--heads` | `mace/tools/arg_parser.py:566` | KEEP — the heads YAML sub-schema | P0-5 | FT-1, CFG-1 | RET-6 | todo |
| `train.foundation_head` | `--foundation_head` | `mace/tools/arg_parser.py:579` | KEEP | P0-5 | FT-1 | RET-6 | todo |
| `train.weight_pt_head` | `--weight_pt_head` | `mace/tools/arg_parser.py:586` | KEEP | P0-5 | FT-1 | RET-6 | todo |
| `train.num_samples_pt` | `--num_samples_pt` | `mace/tools/arg_parser.py:598` | KEEP | P0-5 | FT-1 | RET-6 | todo |
| `train.real_pt_data_ratio_threshold` | `--real_pt_data_ratio_threshold` | `mace/tools/arg_parser.py:592` | KEEP | P0-5 | FT-1 | RET-6 | todo |
| `train.pt_train_file` | `--pt_train_file` | `mace/tools/arg_parser.py:628` | KEEP | P0-5 | FT-1 | RET-6 | todo |
| `train.pt_valid_file` | `--pt_valid_file` | `mace/tools/arg_parser.py:634` | KEEP | P0-5 | FT-1 | RET-6 | todo |
| `train.subselect_pt` | `--subselect_pt` | `mace/tools/arg_parser.py:610` | KEEP | ⚠️ gap (port `tests/workflows/test_finetuning_select.py` cases into P0-5) | FT-1 | RET-6 | todo |
| `train.filter_type_pt` | `--filter_type_pt` | `mace/tools/arg_parser.py:616` | KEEP | ⚠️ gap (idem) | FT-1 | RET-6 | todo |
| `train.allow_random_padding_pt` | `--disallow_random_padding_pt` | `mace/tools/arg_parser.py:622` | KEEP — spelled `--disallow_random_padding_pt`, stored inverted | ⚠️ gap (idem) | FT-1 | RET-6 | todo |
| `train.pseudolabel_replay` | `--pseudolabel_replay` | `mace/tools/arg_parser.py:547` | KEEP | P0-5 | FT-2 | RET-6 | todo |
| `train.pseudolabel_replay_compute_stress` | `--pseudolabel_replay_compute_stress` | `mace/tools/arg_parser.py:553` | KEEP | P0-5 | FT-2 | RET-6 | todo |
| `train.foundation_filter_elements` | `--foundation_filter_elements` | `mace/tools/arg_parser.py:559` | KEEP | ⚠️ gap (all-species saving; add to P0-5) | FT-3 | RET-6 | todo |
| `train.foundation_model_elements` | `--foundation_model_elements` | `mace/tools/arg_parser.py:640` | KEEP — all-species weight saving is a v1 default | ⚠️ gap (add to P0-5) | FT-3 | RET-6 | todo |
| `train.force_mh_ft_lr` | `--force_mh_ft_lr` | `mace/tools/arg_parser.py:604` | DROP — replay-dependent defaults replace the override; the flag exists only to defeat a heuristic v1 does not have | — | REL-1 (doc) | RET-6 | todo |
| `train.lora` | `--lora` | `mace/tools/arg_parser.py:652` | KEEP | `tests/unit/test_lora.py` (port cases) | FT-1 | RET-6 | todo |
| `train.lora_rank` | `--lora_rank` | `mace/tools/arg_parser.py:658` | KEEP | `tests/unit/test_lora.py` | FT-1 | RET-6 | todo |
| `train.lora_alpha` | `--lora_alpha` | `mace/tools/arg_parser.py:664` | KEEP | `tests/unit/test_lora.py` | FT-1 | RET-6 | todo |
| `train.freeze` | `--freeze` | `mace/tools/arg_parser.py:949` | KEEP | `tests/workflows/test_freeze.py` (port cases) | FT-1 | RET-6 | todo |
| `train.finetune_dipoles_polarizabilities` | `--finetune_dipoles_polarizabilities` | `mace/tools/arg_parser.py:1037` | KEEP — the MDP fine-tuning path | ⚠️ gap (add to P0-5 with the MDP golden) | FT-4 | RET-6 | todo |

### 3.7 Loss (16)

Group default: MERGE into composable per-stage losses (TRN-2); the numerics are pinned by P0-6. **The per-dest re-key changes what the `swa_*_weight` rows say.** `--swa_energy_weight` and `--stage_two_energy_weight` are two spellings of one dest, so there is one row and one disposition: MERGE. The legacy `swa` *spelling* dies with the flag namespace, but that is not a separate disposition — an option-string-keyed inventory that said 'DROP the `--swa_*` aliases' was describing a spelling, not a knob.

| id | feature | source | disposition | pinned by | destination | retirement | status |
|---|---|---|---|---|---|---|---|
| `train.loss` | `--loss` | `mace/tools/arg_parser.py:781` | MERGE — the 11 named schemes become loss-composition presets over the 10 loss classes of §8 | P0-6 | TRN-2 | RET-6 | todo |
| `train.energy_weight` | `--energy_weight` | `mace/tools/arg_parser.py:824` | KEEP — per-observable weight | P0-6 | TRN-2 | RET-6 | todo |
| `train.forces_weight` | `--forces_weight` | `mace/tools/arg_parser.py:799` | KEEP — idem | P0-6 | TRN-2 | RET-6 | todo |
| `train.virials_weight` | `--virials_weight` | `mace/tools/arg_parser.py:835` | KEEP — idem | P0-6 | TRN-2 | RET-6 | todo |
| `train.stress_weight` | `--stress_weight` | `mace/tools/arg_parser.py:846` | KEEP — idem | P0-6 | TRN-2 | RET-6 | todo |
| `train.dipole_weight` | `--dipole_weight` | `mace/tools/arg_parser.py:857` | KEEP — idem | P0-6 | TRN-2 | RET-6 | todo |
| `train.polarizability_weight` | `--polarizability_weight` | `mace/tools/arg_parser.py:876` | KEEP — idem | P0-6 | TRN-2 | RET-6 | todo |
| `train.magforces_weight` | `--magforces_weight` | `mace/tools/arg_parser.py:810` | KEEP — idem | ⚠️ gap (add to P0-3b / P0-6) | TRN-2, MAG-1 | RET-6 | todo |
| `train.swa_energy_weight` | `--swa_energy_weight` `--stage_two_energy_weight` | `mace/tools/arg_parser.py:827` | MERGE — per-stage schedules; arbitrary stages replace the two-stage special case, and the `swa` spelling dies with the namespace | P0-6 | TRN-2 | RET-6 | todo |
| `train.swa_forces_weight` | `--swa_forces_weight` `--stage_two_forces_weight` | `mace/tools/arg_parser.py:802` | MERGE — idem | P0-6 | TRN-2 | RET-6 | todo |
| `train.swa_virials_weight` | `--swa_virials_weight` `--stage_two_virials_weight` | `mace/tools/arg_parser.py:838` | MERGE — idem | P0-6 | TRN-2 | RET-6 | todo |
| `train.swa_stress_weight` | `--swa_stress_weight` `--stage_two_stress_weight` | `mace/tools/arg_parser.py:849` | MERGE — idem | P0-6 | TRN-2 | RET-6 | todo |
| `train.swa_dipole_weight` | `--swa_dipole_weight` `--stage_two_dipole_weight` | `mace/tools/arg_parser.py:860` | MERGE — idem | P0-6 | TRN-2 | RET-6 | todo |
| `train.swa_polarizability_weight` | `--swa_polarizability_weight` `--stage_two_polarizability_weight` | `mace/tools/arg_parser.py:868` | MERGE — idem | P0-6 | TRN-2 | RET-6 | todo |
| `train.swa_magforces_weight` | `--swa_magforces_weight` `--stage_two_magforces_weight` | `mace/tools/arg_parser.py:816` | MERGE — idem | ⚠️ gap (add to P0-3b / P0-6) | TRN-2, MAG-1 | RET-6 | todo |
| `train.huber_delta` | `--huber_delta` | `mace/tools/arg_parser.py:888` | KEEP | P0-6 | TRN-2 | RET-6 | todo |

### 3.8 Optimizer, scheduler and training control (26)

Group default: KEEP as the `optimizer` / `schedule` config sections (CFG-1, TRN-2). `--swa`, `--start_swa` and `--swa_lr` carry the `--stage_two*` spellings on the same dest, so they are one row each.

| id | feature | source | disposition | pinned by | destination | retirement | status |
|---|---|---|---|---|---|---|---|
| `train.optimizer` | `--optimizer` | `mace/tools/arg_parser.py:894` | KEEP — adam / adamw / schedulefree | P0-5 (loss decreases) | TRN-2 | RET-6 | todo |
| `train.beta` | `--beta` | `mace/tools/arg_parser.py:901` | KEEP | ⚠️ gap (add to the P0-5 optimizer case) | TRN-2 | RET-6 | todo |
| `train.amsgrad` | `--amsgrad` | `mace/tools/arg_parser.py:955` | KEEP | ⚠️ gap (add to the P0-5 optimizer case) | TRN-2 | RET-6 | todo |
| `train.weight_decay` | `--weight_decay` | `mace/tools/arg_parser.py:940` | KEEP | ⚠️ gap (add to the P0-5 optimizer case) | TRN-2 | RET-6 | todo |
| `train.beta1_schedulefree` | `--beta1_schedulefree` | `mace/tools/arg_parser.py:907` | KEEP — the schedulefree extra's tuning surface | `tests/extensions/schedulefree` | TRN-2 | RET-6 | todo |
| `train.beta2_schedulefree` | `--beta2_schedulefree` | `mace/tools/arg_parser.py:913` | KEEP — idem | `tests/extensions/schedulefree` | TRN-2 | RET-6 | todo |
| `train.warmup_steps_schedulefree` | `--warmup_steps_schedulefree` | `mace/tools/arg_parser.py:919` | KEEP — idem (linear LR warmup, not an LR scheduler) | `tests/extensions/schedulefree` | TRN-2 | RET-6 | todo |
| `train.lbfgs` | `--lbfgs` | `mace/tools/arg_parser.py:992` | KEEP — a second training regime, not an optimizer choice: full-batch gradient assembled in chunks, one `step(closure)` per epoch, ragged tail kept, its own resume fallback | `tests/workflows/test_run_train.py::test_run_train_lbfgs` | TRN-5 | RET-6 | todo |
| `train.batch_size` | `--batch_size` | `mace/tools/arg_parser.py:924` | KEEP | P0-5 | CFG-1 | RET-6 | todo |
| `train.valid_batch_size` | `--valid_batch_size` | `mace/tools/arg_parser.py:926` | KEEP — a separate knob from `--batch_size`, and the one a group-level row loses | P0-5 | CFG-1 | RET-6 | todo |
| `train.lr` | `--lr` | `mace/tools/arg_parser.py:929` | KEEP | P0-5 | TRN-2 | RET-6 | todo |
| `train.lr_factor` | `--lr_factor` | `mace/tools/arg_parser.py:964` | KEEP | P0-5 | TRN-2 | RET-6 | todo |
| `train.scheduler` | `--scheduler` | `mace/tools/arg_parser.py:961` | KEEP | P0-5 | TRN-2 | RET-6 | todo |
| `train.scheduler_patience` | `--scheduler_patience` | `mace/tools/arg_parser.py:967` | KEEP | P0-5 | TRN-2 | RET-6 | todo |
| `train.lr_scheduler_gamma` | `--lr_scheduler_gamma` | `mace/tools/arg_parser.py:970` | KEEP | P0-5 | TRN-2 | RET-6 | todo |
| `train.lr_params_factors` | `--lr_params_factors` | `mace/tools/arg_parser.py:943` | MERGE — typed per-param-group fields of the per-stage optimizer config; the capability stays (`--freeze` reuses it by zeroing factors), the hand-parsed JSON-in-a-string dies | ⚠️ gap (add to P0-5 with `--freeze`) | TRN-2 | RET-6 | todo |
| `train.swa` | `--swa` `--stage_two` | `mace/tools/arg_parser.py:976` | MERGE — stage two becomes a preset second stage of an arbitrary-stage schedule | P0-5 | TRN-2 | RET-6 | todo |
| `train.start_swa` | `--start_swa` `--start_stage_two` | `mace/tools/arg_parser.py:984` | MERGE — idem | P0-5 | TRN-2 | RET-6 | todo |
| `train.swa_lr` | `--swa_lr` `--stage_two_lr` | `mace/tools/arg_parser.py:932` | MERGE — idem | P0-5 | TRN-2 | RET-6 | todo |
| `train.ema` | `--ema` | `mace/tools/arg_parser.py:998` | KEEP | ⚠️ gap (EMA-affects-eval smoke → P0-5) | TRN-1 | RET-6 | todo |
| `train.ema_decay` | `--ema_decay` | `mace/tools/arg_parser.py:1004` | KEEP | ⚠️ gap (idem) | TRN-1 | RET-6 | todo |
| `train.max_num_epochs` | `--max_num_epochs` | `mace/tools/arg_parser.py:1010` | KEEP | P0-5 | TRN-1 | RET-6 | todo |
| `train.patience` | `--patience` | `mace/tools/arg_parser.py:1013` | KEEP | P0-5 | TRN-1 | RET-6 | todo |
| `train.eval_interval` | `--eval_interval` | `mace/tools/arg_parser.py:1043` | KEEP | P0-5 | TRN-1 | RET-6 | todo |
| `train.clip_grad` | `--clip_grad` | `mace/tools/arg_parser.py:1070` | KEEP | ⚠️ gap (add to the P0-5 training case) | TRN-1 | RET-6 | todo |
| `train.dry_run` | `--dry_run` | `mace/tools/arg_parser.py:1076` | KEEP — cheap and useful | ⚠️ gap (add to P0-5) | TRN-1 | RET-6 | todo |

### 3.9 Checkpointing (4)

Group default: KEEP, destination TRN-4 (safetensors checkpointing and resume).

| id | feature | source | disposition | pinned by | destination | retirement | status |
|---|---|---|---|---|---|---|---|
| `train.restart_latest` | `--restart_latest` | `mace/tools/arg_parser.py:1058` | KEEP | P0-5 (resume contract) | TRN-4 | RET-6 | todo |
| `train.keep_checkpoints` | `--keep_checkpoints` | `mace/tools/arg_parser.py:1046` | KEEP | ⚠️ gap (add to the P0-5 resume case) | TRN-4 | RET-6 | todo |
| `train.save_all_checkpoints` | `--save_all_checkpoints` | `mace/tools/arg_parser.py:1052` | KEEP | ⚠️ gap (idem) | TRN-4 | RET-6 | todo |
| `train.save_cpu` | `--save_cpu` | `mace/tools/arg_parser.py:1064` | DROP — safetensors checkpoints are device-agnostic, so there is nothing to choose | — | REL-1 (doc) | RET-6 | todo |

### 3.10 Acceleration (3)

Group default: MERGE into backend-dispatch configuration (BKD-1); the numerics are pinned by P0-4 on GPU CI.

| id | feature | source | disposition | pinned by | destination | retirement | status |
|---|---|---|---|---|---|---|---|
| `train.enable_cueq` | `--enable_cueq` | `mace/tools/arg_parser.py:1083` | MERGE — backend dispatch config | P0-4 | BKD-1, BKD-3 | RET-6 | todo |
| `train.enable_oeq` | `--enable_oeq` | `mace/tools/arg_parser.py:1096` | MERGE — idem | P0-4 | BKD-1, BKD-3 | RET-6 | todo |
| `train.only_cueq` | `--only_cueq` | `mace/tools/arg_parser.py:1089` | MERGE — idem: 'use cueq for every op, not just the ones that benefit' becomes a dispatch policy, not a second boolean. Its own row precisely because a group-level `--enable_cueq/--only_cueq/--enable_oeq` cell hides it | P0-4 | BKD-1, BKD-3 | RET-6 | todo |

### 3.11 wandb (6)

Group default: KEEP (the wandb extra), destination TRN-3.

| id | feature | source | disposition | pinned by | destination | retirement | status |
|---|---|---|---|---|---|---|---|
| `train.wandb` | `--wandb` | `mace/tools/arg_parser.py:1104` | KEEP | ⚠️ gap (offline-mode smoke) | TRN-3 | RET-6 | todo |
| `train.wandb_dir` | `--wandb_dir` | `mace/tools/arg_parser.py:1110` | KEEP | ⚠️ gap (idem) | TRN-3 | RET-6 | todo |
| `train.wandb_project` | `--wandb_project` | `mace/tools/arg_parser.py:1116` | KEEP | ⚠️ gap (idem) | TRN-3 | RET-6 | todo |
| `train.wandb_entity` | `--wandb_entity` | `mace/tools/arg_parser.py:1122` | KEEP | ⚠️ gap (idem) | TRN-3 | RET-6 | todo |
| `train.wandb_name` | `--wandb_name` | `mace/tools/arg_parser.py:1128` | KEEP | ⚠️ gap (idem) | TRN-3 | RET-6 | todo |
| `train.wandb_log_hypers` | `--wandb_log_hypers` | `mace/tools/arg_parser.py:1134` | KEEP | ⚠️ gap (idem) | TRN-3 | RET-6 | todo |

### 3.12 MagneticMACE (9)

Group default: KEEP as the `magnetic`-extra config; destination MAG-1. Grouped like §3.3, but the subgroups land in different v1 mechanisms: the two `*_key` flags follow the property-key convention and `--compute_magforces` (§3.4) follows the observable spec. The weights are in §3.7.

| id | feature | source | disposition | pinned by | destination | retirement | status |
|---|---|---|---|---|---|---|---|
| `train.magmom_key` | `--magmom_key` | `mace/tools/arg_parser.py:708` | MERGE — property-key convention; the default `REF_magmom` extends the on-disk data contract of §15 | `tests/extensions/magnetic` | CORE-3 | RET-6 | todo |
| `train.magforces_key` | `--magforces_key` | `mace/tools/arg_parser.py:714` | MERGE — idem, default `REF_magforces` | `tests/extensions/magnetic` | CORE-3 | RET-6 | todo |
| `train.m_max` | `--m_max` | `mace/tools/arg_parser.py:1160` | KEEP — magnetic architecture hyper | `tests/extensions/magnetic` (7 `resolve_m_max` cases) | MAG-1, ARCH-2 | RET-6 | todo |
| `train.max_m_ell` | `--max_m_ell` | `mace/tools/arg_parser.py:1172` | KEEP — idem | `tests/extensions/magnetic` | MAG-1, ARCH-2 | RET-6 | todo |
| `train.num_mag_radial_basis` | `--num_mag_radial_basis` | `mace/tools/arg_parser.py:1178` | KEEP — idem | `tests/extensions/magnetic` | MAG-1, ARCH-2 | RET-6 | todo |
| `train.num_mag_radial_basis_one_body` | `--num_mag_radial_basis_one_body` | `mace/tools/arg_parser.py:1154` | KEEP — idem | `tests/extensions/magnetic` | MAG-1, ARCH-2 | RET-6 | todo |
| `train.use_magmom_one_body` | `--use_magmom_one_body` | `mace/tools/arg_parser.py:1184` | KEEP — the one-body magmom term | ⚠️ gap (add to P0-3b) | MAG-1 | RET-6 | todo |
| `train.train_one_body_contribution` | `--train_one_body_contribution` | `mace/tools/arg_parser.py:1190` | KEEP — whether the one-body coefficients are optimized | ⚠️ gap (add to P0-3b) | MAG-1, TRN-2 | RET-6 | todo |
| `train.data_aug_magmom` | `--data_aug_magmom` | `mace/tools/arg_parser.py:1197` | MERGE — a training-data transform (`Random3DRotation`), not a model flag | `tests/extensions/magnetic` (rotation equivariance) | TRN-2 | RET-6 | todo |

## 4. `mace_prepare_data` flags — 26 dests

One row per dest of `build_preprocess_arg_parser`. **22 of the 26 are also declared by the
training parser** and get a row in each section: the defaults and the help text differ per parser,
so the same spelling is two knobs needing two dispositions. Retirement is **RET-6** throughout.

| id | feature | source | disposition | pinned by | destination | retirement | status |
|---|---|---|---|---|---|---|---|
| `prep.config` | `--config` | `mace/tools/arg_parser.py:1215` | MERGE — same mechanism and disposition as the training parser's `config` | ⚠️ gap (no test exercises the YAML path) | CORE-2, CFG-1 | RET-6 | todo |
| `prep.train_file` | `--train_file` | `mace/tools/arg_parser.py:1225` | MERGE — `mace data prepare` reads the same config section as `mace train` instead of redeclaring the flag | `tests/workflows/test_preprocess.py` | CFG-1, DATA-1 | RET-6 | todo |
| `prep.valid_file` | `--valid_file` | `mace/tools/arg_parser.py:1232` | MERGE — `mace data prepare` reads the same config section as `mace train` instead of redeclaring the flag | `tests/workflows/test_preprocess.py` | CFG-1, DATA-1 | RET-6 | todo |
| `prep.test_file` | `--test_file` | `mace/tools/arg_parser.py:1252` | MERGE — `mace data prepare` reads the same config section as `mace train` instead of redeclaring the flag | `tests/workflows/test_preprocess.py` | CFG-1, DATA-1 | RET-6 | todo |
| `prep.valid_fraction` | `--valid_fraction` | `mace/tools/arg_parser.py:1245` | MERGE — `mace data prepare` reads the same config section as `mace train` instead of redeclaring the flag | `tests/workflows/test_preprocess.py` | CFG-1 | RET-6 | todo |
| `prep.work_dir` | `--work_dir` | `mace/tools/arg_parser.py:1259` | MERGE — `mace data prepare` reads the same config section as `mace train` instead of redeclaring the flag | `tests/workflows/test_preprocess.py` | CFG-1 | RET-6 | todo |
| `prep.r_max` | `--r_max` | `mace/tools/arg_parser.py:1271` | MERGE — `mace data prepare` reads the same config section as `mace train` instead of redeclaring the flag | `tests/workflows/test_preprocess.py` | CFG-1 | RET-6 | todo |
| `prep.config_type_weights` | `--config_type_weights` | `mace/tools/arg_parser.py:1274` | MERGE — `mace data prepare` reads the same config section as `mace train` instead of redeclaring the flag | ⚠️ gap (preprocess path) | TRN-2 | RET-6 | todo |
| `prep.energy_key` | `--energy_key` | `mace/tools/arg_parser.py:1280` | MERGE — `mace data prepare` reads the same config section as `mace train` instead of redeclaring the flag | `tests/workflows/test_preprocess.py` | CORE-3 | RET-6 | todo |
| `prep.forces_key` | `--forces_key` | `mace/tools/arg_parser.py:1286` | MERGE — `mace data prepare` reads the same config section as `mace train` instead of redeclaring the flag | `tests/workflows/test_preprocess.py` | CORE-3 | RET-6 | todo |
| `prep.virials_key` | `--virials_key` | `mace/tools/arg_parser.py:1292` | MERGE — `mace data prepare` reads the same config section as `mace train` instead of redeclaring the flag | ⚠️ gap (preprocess path) | CORE-3 | RET-6 | todo |
| `prep.stress_key` | `--stress_key` | `mace/tools/arg_parser.py:1298` | MERGE — `mace data prepare` reads the same config section as `mace train` instead of redeclaring the flag | ⚠️ gap (preprocess path) | CORE-3 | RET-6 | todo |
| `prep.dipole_key` | `--dipole_key` | `mace/tools/arg_parser.py:1304` | MERGE — `mace data prepare` reads the same config section as `mace train` instead of redeclaring the flag | ⚠️ gap (preprocess path) | CORE-3 | RET-6 | todo |
| `prep.polarizability_key` | `--polarizability_key` | `mace/tools/arg_parser.py:1310` | MERGE — `mace data prepare` reads the same config section as `mace train` instead of redeclaring the flag | ⚠️ gap (preprocess path) | CORE-3 | RET-6 | todo |
| `prep.charges_key` | `--charges_key` | `mace/tools/arg_parser.py:1316` | MERGE — `mace data prepare` reads the same config section as `mace train` instead of redeclaring the flag | ⚠️ gap (preprocess path) | CORE-3 | RET-6 | todo |
| `prep.head_key` | `--head_key` | `mace/tools/arg_parser.py:1368` | MERGE — `mace data prepare` reads the same config section as `mace train` instead of redeclaring the flag | `tests/workflows/test_preprocess.py` | CORE-3 | RET-6 | todo |
| `prep.heads` | `--heads` | `mace/tools/arg_parser.py:1374` | MERGE — `mace data prepare` reads the same config section as `mace train` instead of redeclaring the flag | `tests/workflows/test_preprocess.py` | CFG-1 | RET-6 | todo |
| `prep.atomic_numbers` | `--atomic_numbers` | `mace/tools/arg_parser.py:1322` | MERGE — `mace data prepare` reads the same config section as `mace train` instead of redeclaring the flag | `tests/workflows/test_preprocess.py` | DATA-1 | RET-6 | todo |
| `prep.batch_size` | `--batch_size` | `mace/tools/arg_parser.py:1335` | MERGE — `mace data prepare` reads the same config section as `mace train` instead of redeclaring the flag | `tests/workflows/test_preprocess.py` | CFG-1 | RET-6 | todo |
| `prep.scaling` | `--scaling` | `mace/tools/arg_parser.py:1342` | MERGE — `mace data prepare` reads the same config section as `mace train` instead of redeclaring the flag | `tests/workflows/test_preprocess.py` | ARCH-3 | RET-6 | todo |
| `prep.E0s` | `--E0s` | `mace/tools/arg_parser.py:1349` | MERGE — `mace data prepare` reads the same config section as `mace train` instead of redeclaring the flag | `tests/workflows/test_preprocess.py` | DATA-1 | RET-6 | todo |
| `prep.seed` | `--seed` | `mace/tools/arg_parser.py:1362` | MERGE — `mace data prepare` reads the same config section as `mace train` instead of redeclaring the flag | `tests/workflows/test_preprocess.py` | CFG-1 | RET-6 | todo |
| `prep.num_process` | `--num_process` | `mace/tools/arg_parser.py:1239` | KEEP — preprocessing parallelism | `tests/workflows/test_preprocess.py` | DATA-1 | RET-6 | todo |
| `prep.h5_prefix` | `--h5_prefix` | `mace/tools/arg_parser.py:1265` | KEEP — the shard naming/output prefix | `tests/workflows/test_preprocess.py` | DATA-2 | RET-6 | todo |
| `prep.compute_statistics` | `--compute_statistics` | `mace/tools/arg_parser.py:1329` | KEEP — emits `statistics.json` | `tests/workflows/test_preprocess.py` | DATA-1 | RET-6 | todo |
| `prep.shuffle` | `--shuffle` | `mace/tools/arg_parser.py:1356` | KEEP | `tests/workflows/test_preprocess.py` | DATA-1 | RET-6 | todo |

## 5. Other CLI flags — 111 dests over 13 parsers

The thirteen argparsers under `mace/cli/`: seven user-facing CLIs (88 dests) and the six
`convert_*` weight/device converters (23 dests, 7 distinct). Three of the six have no console
entry point at all, so an extraction driven by `setup.cfg` misses them twice. 74 distinct dests
become 111 rows because a dest is counted once per parser that declares it — `--device` in four
CLIs is four knobs with four defaults.

### `mace_eval_configs` — `mace/cli/eval_configs.py` (19)

| id | feature | source | disposition | pinned by | destination | retirement | status |
|---|---|---|---|---|---|---|---|
| `cli.eval_configs.configs` | `--configs` | `mace/cli/eval_configs.py:28` | KEEP | P0-5 | CLI-1 | RET-6 | todo |
| `cli.eval_configs.model` | `--model` | `mace/cli/eval_configs.py:29` | KEEP | P0-5 | CLI-1 | RET-6 | todo |
| `cli.eval_configs.output` | `--output` | `mace/cli/eval_configs.py:30` | KEEP | P0-5 | CLI-1 | RET-6 | todo |
| `cli.eval_configs.device` | `--device` | `mace/cli/eval_configs.py:32` | KEEP | P0-5 | CLI-1 | RET-6 | todo |
| `cli.eval_configs.default_dtype` | `--default_dtype` | `mace/cli/eval_configs.py:45` | MERGE — `PrecisionConfig` | P0-5 | BKD-2 | RET-6 | todo |
| `cli.eval_configs.batch_size` | `--batch_size` | `mace/cli/eval_configs.py:51` | KEEP | P0-5 | CLI-1 | RET-6 | todo |
| `cli.eval_configs.compute_stress` | `--compute_stress` | `mace/cli/eval_configs.py:53` | MERGE — observable spec | P0-5 | CORE-1 | RET-6 | todo |
| `cli.eval_configs.info_prefix` | `--info_prefix` | `mace/cli/eval_configs.py:101` | KEEP — prefixes every key written back into the XYZ (§12) | P0-5 | CLI-1, CORE-3 | RET-6 | todo |
| `cli.eval_configs.head` | `--head` | `mace/cli/eval_configs.py:107` | KEEP | P0-5 | CLI-1 | RET-6 | todo |
| `cli.eval_configs.enable_cueq` | `--enable_cueq` | `mace/cli/eval_configs.py:39` | MERGE — backend dispatch config | P0-4 | BKD-1 | RET-6 | todo |
| `cli.eval_configs.return_contributions` | `--return_contributions` | `mace/cli/eval_configs.py:65` | KEEP — typed outputs make this natural | ⚠️ gap (add assert to P0-5) | CLI-1, CORE-1 | RET-6 | todo |
| `cli.eval_configs.return_node_energies` | `--return_node_energies` | `mace/cli/eval_configs.py:95` | KEEP — idem | ⚠️ gap (add assert to P0-5) | CLI-1, CORE-1 | RET-6 | todo |
| `cli.eval_configs.compute_bec` | `--compute_bec` | `mace/cli/eval_configs.py:59` | KEEP — Born effective charges (IR spectra): a real physical observable of the polar model, and cheap because the derivative already exists | ⚠️ gap (add to P0-3a) | ELEC-2, CORE-1 | RET-6 | todo |
| `cli.eval_configs.return_descriptors` | `--return_descriptors` | `mace/cli/eval_configs.py:71` | KEEP — descriptor extraction is `BaseMACE`'s raison d'être | ⚠️ gap (add to P0-5) | ARCH-2, CLI-1 | RET-6 | todo |
| `cli.eval_configs.descriptor_num_layers` | `--descriptor_num_layers` | `mace/cli/eval_configs.py:77` | KEEP — idem | ⚠️ gap (add to P0-5) | ARCH-2, CLI-1 | RET-6 | todo |
| `cli.eval_configs.descriptor_aggregation_method` | `--descriptor_aggregation_method` | `mace/cli/eval_configs.py:83` | KEEP — idem | ⚠️ gap (add to P0-5) | ARCH-2, CLI-1 | RET-6 | todo |
| `cli.eval_configs.descriptor_invariants_only` | `--descriptor_invariants_only` | `mace/cli/eval_configs.py:89` | KEEP — idem | ⚠️ gap (add to P0-5) | ARCH-2, CLI-1 | RET-6 | todo |
| `cli.eval_configs.magmom_key` | `--magmom_key` | `mace/cli/eval_configs.py:114` | MERGE — property-key convention | `tests/extensions/magnetic` | CORE-3 | RET-6 | todo |
| `cli.eval_configs.return_magforces` | `--return_magforces` | `mace/cli/eval_configs.py:121` | MERGE — observable spec (`dE/dm` declared like any other derivative) | `tests/extensions/magnetic` | CORE-1, MAG-1 | RET-6 | todo |

### `mace_finetuning_select` — `mace/cli/fine_tuning_select.py` (18)

| id | feature | source | disposition | pinned by | destination | retirement | status |
|---|---|---|---|---|---|---|---|
| `cli.fine_tuning_select.configs_pt` | `--configs_pt` | `mace/cli/fine_tuning_select.py:94` | MERGE — absorbed into the integrated fine-tuning pipeline (`mace train` with a fine-tuning config); selection stops being a separate CLI over a separate model load | ⚠️ gap (port `tests/workflows/test_finetuning_select.py` cases into P0-5) | FT-1 | RET-6 | todo |
| `cli.fine_tuning_select.configs_ft` | `--configs_ft` | `mace/cli/fine_tuning_select.py:99` | MERGE — absorbed into the integrated fine-tuning pipeline (`mace train` with a fine-tuning config); selection stops being a separate CLI over a separate model load | ⚠️ gap (idem) | FT-1 | RET-6 | todo |
| `cli.fine_tuning_select.num_samples` | `--num_samples` | `mace/cli/fine_tuning_select.py:105` | MERGE — absorbed into the integrated fine-tuning pipeline (`mace train` with a fine-tuning config); selection stops being a separate CLI over a separate model load | ⚠️ gap (idem) | FT-1 | RET-6 | todo |
| `cli.fine_tuning_select.subselect` | `--subselect` | `mace/cli/fine_tuning_select.py:112` | MERGE — absorbed into the integrated fine-tuning pipeline (`mace train` with a fine-tuning config); selection stops being a separate CLI over a separate model load | ⚠️ gap (idem — fps / random) | FT-1 | RET-6 | todo |
| `cli.fine_tuning_select.model` | `--model` | `mace/cli/fine_tuning_select.py:119` | MERGE — absorbed into the integrated fine-tuning pipeline (`mace train` with a fine-tuning config); selection stops being a separate CLI over a separate model load | ⚠️ gap (idem) | FT-1 | RET-6 | todo |
| `cli.fine_tuning_select.output` | `--output` | `mace/cli/fine_tuning_select.py:121` | MERGE — absorbed into the integrated fine-tuning pipeline (`mace train` with a fine-tuning config); selection stops being a separate CLI over a separate model load | ⚠️ gap (idem) | FT-1 | RET-6 | todo |
| `cli.fine_tuning_select.descriptors` | `--descriptors` | `mace/cli/fine_tuning_select.py:123` | MERGE — absorbed into the integrated fine-tuning pipeline (`mace train` with a fine-tuning config); selection stops being a separate CLI over a separate model load | ⚠️ gap (idem) | FT-1 | RET-6 | todo |
| `cli.fine_tuning_select.device` | `--device` | `mace/cli/fine_tuning_select.py:126` | MERGE — absorbed into the integrated fine-tuning pipeline (`mace train` with a fine-tuning config); selection stops being a separate CLI over a separate model load | ⚠️ gap (idem) | FT-1 | RET-6 | todo |
| `cli.fine_tuning_select.head_pt` | `--head_pt` | `mace/cli/fine_tuning_select.py:140` | MERGE — absorbed into the integrated fine-tuning pipeline (`mace train` with a fine-tuning config); selection stops being a separate CLI over a separate model load | ⚠️ gap (idem) | FT-1 | RET-6 | todo |
| `cli.fine_tuning_select.head_ft` | `--head_ft` | `mace/cli/fine_tuning_select.py:146` | MERGE — absorbed into the integrated fine-tuning pipeline (`mace train` with a fine-tuning config); selection stops being a separate CLI over a separate model load | ⚠️ gap (idem) | FT-1 | RET-6 | todo |
| `cli.fine_tuning_select.filtering_type` | `--filtering_type` | `mace/cli/fine_tuning_select.py:152` | MERGE — absorbed into the integrated fine-tuning pipeline (`mace train` with a fine-tuning config); selection stops being a separate CLI over a separate model load | ⚠️ gap (idem) | FT-1 | RET-6 | todo |
| `cli.fine_tuning_select.weight_ft` | `--weight_ft` | `mace/cli/fine_tuning_select.py:159` | MERGE — absorbed into the integrated fine-tuning pipeline (`mace train` with a fine-tuning config); selection stops being a separate CLI over a separate model load | ⚠️ gap (idem) | FT-1 | RET-6 | todo |
| `cli.fine_tuning_select.weight_pt` | `--weight_pt` | `mace/cli/fine_tuning_select.py:165` | MERGE — absorbed into the integrated fine-tuning pipeline (`mace train` with a fine-tuning config); selection stops being a separate CLI over a separate model load | ⚠️ gap (idem) | FT-1 | RET-6 | todo |
| `cli.fine_tuning_select.filter_atomic_numbers_pt` | `--filter_atomic_numbers_pt` | `mace/cli/fine_tuning_select.py:171` | MERGE — absorbed into the integrated fine-tuning pipeline (`mace train` with a fine-tuning config); selection stops being a separate CLI over a separate model load | ⚠️ gap (idem) | FT-1 | RET-6 | todo |
| `cli.fine_tuning_select.allow_random_padding` | `--disallow_random_padding` | `mace/cli/fine_tuning_select.py:177` | MERGE — absorbed into the integrated fine-tuning pipeline (`mace train` with a fine-tuning config); selection stops being a separate CLI over a separate model load | ⚠️ gap (idem — spelled `--disallow_random_padding`, stored inverted) | FT-1 | RET-6 | todo |
| `cli.fine_tuning_select.seed` | `--seed` | `mace/cli/fine_tuning_select.py:182` | MERGE — absorbed into the integrated fine-tuning pipeline (`mace train` with a fine-tuning config); selection stops being a separate CLI over a separate model load | ⚠️ gap (idem) | FT-1 | RET-6 | todo |
| `cli.fine_tuning_select.default_dtype` | `--default_dtype` | `mace/cli/fine_tuning_select.py:133` | MERGE — `PrecisionConfig` | ⚠️ gap (port `tests/workflows/test_finetuning_select.py`) | BKD-2 | RET-6 | todo |
| `cli.fine_tuning_select.config` | `--config` | `mace/cli/fine_tuning_select.py:83` | MERGE — the v1 config system | ⚠️ gap (no test exercises the YAML path) | CORE-2, CFG-1 | RET-6 | todo |

### `mace_plot_train` — `mace/cli/plot_train.py` (8)

| id | feature | source | disposition | pinned by | destination | retirement | status |
|---|---|---|---|---|---|---|---|
| `cli.plot_train.path` | `--path` | `mace/cli/plot_train.py:78` | KEEP | ⚠️ gap (plot smoke: parse a real results log, write a file) | CLI-1 | RET-6 | todo |
| `cli.plot_train.min_epoch` | `--min_epoch` | `mace/cli/plot_train.py:81` | KEEP | ⚠️ gap (idem) | CLI-1 | RET-6 | todo |
| `cli.plot_train.linear` | `--linear` | `mace/cli/plot_train.py:93` | KEEP | ⚠️ gap (idem) | CLI-1 | RET-6 | todo |
| `cli.plot_train.error_bars` | `--error_bars` | `mace/cli/plot_train.py:100` | KEEP | ⚠️ gap (idem) | CLI-1 | RET-6 | todo |
| `cli.plot_train.keys` | `--keys` | `mace/cli/plot_train.py:107` | KEEP | ⚠️ gap (idem) | CLI-1 | RET-6 | todo |
| `cli.plot_train.output_format` | `--output_format` | `mace/cli/plot_train.py:115` | KEEP | ⚠️ gap (idem) | CLI-1 | RET-6 | todo |
| `cli.plot_train.heads` | `--heads` | `mace/cli/plot_train.py:123` | KEEP — per-head loss curves | ⚠️ gap (idem) | CLI-1 | RET-6 | todo |
| `cli.plot_train.start_swa` | `--start_stage_two` `--start_swa` | `mace/cli/plot_train.py:84` | MERGE — stage boundaries are read from the run's per-stage schedule metadata; once stages are arbitrary a single 'stage two' marker no longer applies. Carries both `--start_stage_two` and the legacy `--start_swa` spelling | ⚠️ gap (idem) | CLI-1, TRN-3 | RET-6 | todo |

### `mace_active_learning_md` — `mace/cli/active_learning_md.py` (16)

| id | feature | source | disposition | pinned by | destination | retirement | status |
|---|---|---|---|---|---|---|---|
| `cli.active_learning_md.config` | `--config` | `mace/cli/active_learning_md.py:20` | DROP — out of v1.0 scope: an ASE MD loop a user writes in ~30 lines over the calculator, at the cost of MACE owning thermostat, timestep and trajectory I/O. What MACE must guarantee is the committee variance in `calculate`, which stays (§16) | — | REL-1 (doc) | RET-6 | todo |
| `cli.active_learning_md.config_index` | `--config_index` | `mace/cli/active_learning_md.py:22` | DROP — out of v1.0 scope: an ASE MD loop a user writes in ~30 lines over the calculator, at the cost of MACE owning thermostat, timestep and trajectory I/O. What MACE must guarantee is the committee variance in `calculate`, which stays (§16) | — | REL-1 (doc) | RET-6 | todo |
| `cli.active_learning_md.error_threshold` | `--error_threshold` | `mace/cli/active_learning_md.py:25` | DROP — out of v1.0 scope: an ASE MD loop a user writes in ~30 lines over the calculator, at the cost of MACE owning thermostat, timestep and trajectory I/O. What MACE must guarantee is the committee variance in `calculate`, which stays (§16) | — | REL-1 (doc) | RET-6 | todo |
| `cli.active_learning_md.temperature_K` | `--temperature_K` | `mace/cli/active_learning_md.py:27` | DROP — out of v1.0 scope: an ASE MD loop a user writes in ~30 lines over the calculator, at the cost of MACE owning thermostat, timestep and trajectory I/O. What MACE must guarantee is the committee variance in `calculate`, which stays (§16) | — | REL-1 (doc) | RET-6 | todo |
| `cli.active_learning_md.friction` | `--friction` | `mace/cli/active_learning_md.py:28` | DROP — out of v1.0 scope: an ASE MD loop a user writes in ~30 lines over the calculator, at the cost of MACE owning thermostat, timestep and trajectory I/O. What MACE must guarantee is the committee variance in `calculate`, which stays (§16) | — | REL-1 (doc) | RET-6 | todo |
| `cli.active_learning_md.timestep` | `--timestep` | `mace/cli/active_learning_md.py:29` | DROP — out of v1.0 scope: an ASE MD loop a user writes in ~30 lines over the calculator, at the cost of MACE owning thermostat, timestep and trajectory I/O. What MACE must guarantee is the committee variance in `calculate`, which stays (§16) | — | REL-1 (doc) | RET-6 | todo |
| `cli.active_learning_md.nsteps` | `--nsteps` | `mace/cli/active_learning_md.py:30` | DROP — out of v1.0 scope: an ASE MD loop a user writes in ~30 lines over the calculator, at the cost of MACE owning thermostat, timestep and trajectory I/O. What MACE must guarantee is the committee variance in `calculate`, which stays (§16) | — | REL-1 (doc) | RET-6 | todo |
| `cli.active_learning_md.nprint` | `--nprint` | `mace/cli/active_learning_md.py:32` | DROP — out of v1.0 scope: an ASE MD loop a user writes in ~30 lines over the calculator, at the cost of MACE owning thermostat, timestep and trajectory I/O. What MACE must guarantee is the committee variance in `calculate`, which stays (§16) | — | REL-1 (doc) | RET-6 | todo |
| `cli.active_learning_md.nsave` | `--nsave` | `mace/cli/active_learning_md.py:35` | DROP — out of v1.0 scope: an ASE MD loop a user writes in ~30 lines over the calculator, at the cost of MACE owning thermostat, timestep and trajectory I/O. What MACE must guarantee is the committee variance in `calculate`, which stays (§16) | — | REL-1 (doc) | RET-6 | todo |
| `cli.active_learning_md.ncheckerror` | `--ncheckerror` | `mace/cli/active_learning_md.py:38` | DROP — out of v1.0 scope: an ASE MD loop a user writes in ~30 lines over the calculator, at the cost of MACE owning thermostat, timestep and trajectory I/O. What MACE must guarantee is the committee variance in `calculate`, which stays (§16) | — | REL-1 (doc) | RET-6 | todo |
| `cli.active_learning_md.model` | `--model` | `mace/cli/active_learning_md.py:42` | DROP — out of v1.0 scope: an ASE MD loop a user writes in ~30 lines over the calculator, at the cost of MACE owning thermostat, timestep and trajectory I/O. What MACE must guarantee is the committee variance in `calculate`, which stays (§16) | — | REL-1 (doc) | RET-6 | todo |
| `cli.active_learning_md.output` | `--output` | `mace/cli/active_learning_md.py:47` | DROP — out of v1.0 scope: an ASE MD loop a user writes in ~30 lines over the calculator, at the cost of MACE owning thermostat, timestep and trajectory I/O. What MACE must guarantee is the committee variance in `calculate`, which stays (§16) | — | REL-1 (doc) | RET-6 | todo |
| `cli.active_learning_md.device` | `--device` | `mace/cli/active_learning_md.py:49` | DROP — out of v1.0 scope: an ASE MD loop a user writes in ~30 lines over the calculator, at the cost of MACE owning thermostat, timestep and trajectory I/O. What MACE must guarantee is the committee variance in `calculate`, which stays (§16) | — | REL-1 (doc) | RET-6 | todo |
| `cli.active_learning_md.default_dtype` | `--default_dtype` | `mace/cli/active_learning_md.py:56` | DROP — out of v1.0 scope: an ASE MD loop a user writes in ~30 lines over the calculator, at the cost of MACE owning thermostat, timestep and trajectory I/O. What MACE must guarantee is the committee variance in `calculate`, which stays (§16) | — | REL-1 (doc) | RET-6 | todo |
| `cli.active_learning_md.compute_stress` | `--compute_stress` | `mace/cli/active_learning_md.py:63` | DROP — out of v1.0 scope: an ASE MD loop a user writes in ~30 lines over the calculator, at the cost of MACE owning thermostat, timestep and trajectory I/O. What MACE must guarantee is the committee variance in `calculate`, which stays (§16) | — | REL-1 (doc) | RET-6 | todo |
| `cli.active_learning_md.info_prefix` | `--info_prefix` | `mace/cli/active_learning_md.py:69` | DROP — out of v1.0 scope: an ASE MD loop a user writes in ~30 lines over the calculator, at the cost of MACE owning thermostat, timestep and trajectory I/O. What MACE must guarantee is the committee variance in `calculate`, which stays (§16) | — | REL-1 (doc) | RET-6 | todo |

### `mace_polar_density_cube` — `mace/cli/polar_density_cube.py` (18)

| id | feature | source | disposition | pinned by | destination | retirement | status |
|---|---|---|---|---|---|---|---|
| `cli.polar_density_cube.configs` | `--configs` | `mace/cli/polar_density_cube.py:511` | KEEP — the electrostatics extra's density-cube tool | `tests/extensions/polar/test_polar_density_cube.py` (port cases) | ELEC-2 | RET-4 | todo |
| `cli.polar_density_cube.model` | `--model` | `mace/cli/polar_density_cube.py:513` | KEEP — the electrostatics extra's density-cube tool | `tests/extensions/polar/test_polar_density_cube.py` (port cases) | ELEC-2 | RET-4 | todo |
| `cli.polar_density_cube.output` | `--output` | `mace/cli/polar_density_cube.py:517` | KEEP — the electrostatics extra's density-cube tool | `tests/extensions/polar/test_polar_density_cube.py` (port cases) | ELEC-2 | RET-4 | todo |
| `cli.polar_density_cube.index` | `--index` | `mace/cli/polar_density_cube.py:518` | KEEP — the electrostatics extra's density-cube tool | `tests/extensions/polar/test_polar_density_cube.py` (port cases) | ELEC-2 | RET-4 | todo |
| `cli.polar_density_cube.quantity` | `--quantity` | `mace/cli/polar_density_cube.py:520` | KEEP — the electrostatics extra's density-cube tool | `tests/extensions/polar/test_polar_density_cube.py` (port cases) | ELEC-2 | RET-4 | todo |
| `cli.polar_density_cube.grid` | `--grid` | `mace/cli/polar_density_cube.py:526` | KEEP — the electrostatics extra's density-cube tool | `tests/extensions/polar/test_polar_density_cube.py` (port cases) | ELEC-2 | RET-4 | todo |
| `cli.polar_density_cube.device` | `--device` | `mace/cli/polar_density_cube.py:533` | KEEP — the electrostatics extra's density-cube tool | `tests/extensions/polar/test_polar_density_cube.py` (port cases) | ELEC-2 | RET-4 | todo |
| `cli.polar_density_cube.sigma` | `--sigma` | `mace/cli/polar_density_cube.py:537` | KEEP — the electrostatics extra's density-cube tool | `tests/extensions/polar/test_polar_density_cube.py` (port cases) | ELEC-2 | RET-4 | todo |
| `cli.polar_density_cube.kspace_cutoff` | `--kspace_cutoff` | `mace/cli/polar_density_cube.py:538` | KEEP — the electrostatics extra's density-cube tool | `tests/extensions/polar/test_polar_density_cube.py` (port cases) | ELEC-2 | RET-4 | todo |
| `cli.polar_density_cube.backend` | `--backend` | `mace/cli/polar_density_cube.py:540` | KEEP — the electrostatics extra's density-cube tool | `tests/extensions/polar/test_polar_density_cube.py` (port cases) | ELEC-2 | RET-4 | todo |
| `cli.polar_density_cube.realspace_cutoff_factor` | `--realspace_cutoff_factor` | `mace/cli/polar_density_cube.py:546` | KEEP — the electrostatics extra's density-cube tool | `tests/extensions/polar/test_polar_density_cube.py` (port cases) | ELEC-2 | RET-4 | todo |
| `cli.polar_density_cube.chunk_size` | `--chunk_size` | `mace/cli/polar_density_cube.py:552` | KEEP — the electrostatics extra's density-cube tool | `tests/extensions/polar/test_polar_density_cube.py` (port cases) | ELEC-2 | RET-4 | todo |
| `cli.polar_density_cube.subtract_total_charge` | `--subtract_total_charge` | `mace/cli/polar_density_cube.py:558` | KEEP — the electrostatics extra's density-cube tool | `tests/extensions/polar/test_polar_density_cube.py` (port cases) | ELEC-2 | RET-4 | todo |
| `cli.polar_density_cube.external_field` | `--external_field` | `mace/cli/polar_density_cube.py:563` | KEEP — the electrostatics extra's density-cube tool | `tests/extensions/polar/test_polar_density_cube.py` (port cases) | ELEC-2 | RET-4 | todo |
| `cli.polar_density_cube.fermi_level` | `--fermi_level` | `mace/cli/polar_density_cube.py:570` | KEEP — the electrostatics extra's density-cube tool | `tests/extensions/polar/test_polar_density_cube.py` (port cases) | ELEC-2 | RET-4 | todo |
| `cli.polar_density_cube.write_potential` | `--write_potential` | `mace/cli/polar_density_cube.py:576` | KEEP — the electrostatics extra's density-cube tool | `tests/extensions/polar/test_polar_density_cube.py` (port cases) | ELEC-2 | RET-4 | todo |
| `cli.polar_density_cube.quality_report` | `--quality_report` | `mace/cli/polar_density_cube.py:581` | KEEP — the electrostatics extra's density-cube tool | `tests/extensions/polar/test_polar_density_cube.py` (port cases) | ELEC-2 | RET-4 | todo |
| `cli.polar_density_cube.default_dtype` | `--default_dtype` | `mace/cli/polar_density_cube.py:535` | MERGE — `PrecisionConfig` | `tests/extensions/polar/test_polar_density_cube.py` | BKD-2 | RET-4 | todo |

### `mace_create_lammps_model` — `mace/cli/create_lammps_model.py` (4)

| id | feature | source | disposition | pinned by | destination | retirement | status |
|---|---|---|---|---|---|---|---|
| `cli.create_lammps_model.model_path` | `model_path` | `mace/cli/create_lammps_model.py:21` | KEEP — positional; becomes the checkpoint argument of `mace export lammps` | P0-5 (export golden) | DEP-2 | RET-5 | todo |
| `cli.create_lammps_model.head` | `--head` | `mace/cli/create_lammps_model.py:26` | KEEP | P0-5 (export golden) | DEP-2 | RET-5 | todo |
| `cli.create_lammps_model.dtype` | `--dtype` | `mace/cli/create_lammps_model.py:33` | MERGE — `PrecisionConfig` of the export bundle | P0-5 (export golden) | BKD-2, DEP-2 | RET-5 | todo |
| `cli.create_lammps_model.format` | `--format` | `mace/cli/create_lammps_model.py:40` | MERGE — v1 exports the MLIAP bundle only; the default TorchScript format is dropped with `jit.script`, so the choice collapses to one and the flag with it | P0-5 (export golden) | DEP-2 | RET-5 | todo |

### `mace_select_head` — `mace/cli/select_head.py` (5)

| id | feature | source | disposition | pinned by | destination | retirement | status |
|---|---|---|---|---|---|---|---|
| `cli.select_head.model_file` | `model_file` | `mace/cli/select_head.py:33` | KEEP — positional | ⚠️ gap (add `mace model select-head` case to P0-5) | CLI-1 | RET-6 | todo |
| `cli.select_head.head_name` | `--head_name` `-n` | `mace/cli/select_head.py:12` | KEEP | ⚠️ gap (idem) | CLI-1 | RET-6 | todo |
| `cli.select_head.list_heads` | `--list_heads` `-l` | `mace/cli/select_head.py:18` | KEEP | ⚠️ gap (idem) | CLI-1 | RET-6 | todo |
| `cli.select_head.target_device` | `--target_device` `-d` | `mace/cli/select_head.py:24` | KEEP | ⚠️ gap (idem) | CLI-1 | RET-6 | todo |
| `cli.select_head.output_file` | `--output_file` `-o` | `mace/cli/select_head.py:29` | KEEP | ⚠️ gap (idem) | CLI-1 | RET-6 | todo |

### `mace_e3nn_cueq` — `mace/cli/convert_e3nn_cueq.py` (4)

| id | feature | source | disposition | pinned by | destination | retirement | status |
|---|---|---|---|---|---|---|---|
| `cli.convert_e3nn_cueq.input_model` | `input_model` | `mace/cli/convert_e3nn_cueq.py:280` | DROP — v1 weights are canonical and backend dispatch is automatic, so there is nothing left to convert between backends (positional) | P0-4 pins the backend numerics | REL-1 (doc) | RET-3 | todo |
| `cli.convert_e3nn_cueq.output_model` | `--output_model` | `mace/cli/convert_e3nn_cueq.py:282` | DROP — v1 weights are canonical and backend dispatch is automatic, so there is nothing left to convert between backends | P0-4 pins the backend numerics | REL-1 (doc) | RET-3 | todo |
| `cli.convert_e3nn_cueq.device` | `--device` | `mace/cli/convert_e3nn_cueq.py:286` | DROP — v1 weights are canonical and backend dispatch is automatic, so there is nothing left to convert between backends | P0-4 pins the backend numerics | REL-1 (doc) | RET-3 | todo |
| `cli.convert_e3nn_cueq.return_model` | `--return_model` | `mace/cli/convert_e3nn_cueq.py:288` | DROP — v1 weights are canonical and backend dispatch is automatic, so there is nothing left to convert between backends (library flag: return the converted model instead of writing it) | P0-4 pins the backend numerics | REL-1 (doc) | RET-3 | todo |

### `mace_cueq_to_e3nn` — `mace/cli/convert_cueq_e3nn.py` (4)

| id | feature | source | disposition | pinned by | destination | retirement | status |
|---|---|---|---|---|---|---|---|
| `cli.convert_cueq_e3nn.input_model` | `input_model` | `mace/cli/convert_cueq_e3nn.py:282` | DROP — v1 weights are canonical and backend dispatch is automatic, so there is nothing left to convert between backends (positional) | P0-4 pins the backend numerics | REL-1 (doc) | RET-3 | todo |
| `cli.convert_cueq_e3nn.output_model` | `--output_model` | `mace/cli/convert_cueq_e3nn.py:284` | DROP — v1 weights are canonical and backend dispatch is automatic, so there is nothing left to convert between backends | P0-4 pins the backend numerics | REL-1 (doc) | RET-3 | todo |
| `cli.convert_cueq_e3nn.device` | `--device` | `mace/cli/convert_cueq_e3nn.py:286` | DROP — v1 weights are canonical and backend dispatch is automatic, so there is nothing left to convert between backends | P0-4 pins the backend numerics | REL-1 (doc) | RET-3 | todo |
| `cli.convert_cueq_e3nn.return_model` | `--return_model` | `mace/cli/convert_cueq_e3nn.py:288` | DROP — v1 weights are canonical and backend dispatch is automatic, so there is nothing left to convert between backends (library flag: return the converted model instead of writing it) | P0-4 pins the backend numerics | REL-1 (doc) | RET-3 | todo |

### `mace/cli/convert_e3nn_oeq.py` — no entry point (4)

| id | feature | source | disposition | pinned by | destination | retirement | status |
|---|---|---|---|---|---|---|---|
| `cli.convert_e3nn_oeq.input_model` | `input_model` | `mace/cli/convert_e3nn_oeq.py:67` | DROP — v1 weights are canonical and backend dispatch is automatic, so there is nothing left to convert between backends (positional) | P0-4 pins the backend numerics | REL-1 (doc) | RET-3 | todo |
| `cli.convert_e3nn_oeq.output_model` | `--output_model` | `mace/cli/convert_e3nn_oeq.py:69` | DROP — v1 weights are canonical and backend dispatch is automatic, so there is nothing left to convert between backends | P0-4 pins the backend numerics | REL-1 (doc) | RET-3 | todo |
| `cli.convert_e3nn_oeq.device` | `--device` | `mace/cli/convert_e3nn_oeq.py:73` | DROP — v1 weights are canonical and backend dispatch is automatic, so there is nothing left to convert between backends | P0-4 pins the backend numerics | REL-1 (doc) | RET-3 | todo |
| `cli.convert_e3nn_oeq.return_model` | `--return_model` | `mace/cli/convert_e3nn_oeq.py:75` | DROP — v1 weights are canonical and backend dispatch is automatic, so there is nothing left to convert between backends (library flag: return the converted model instead of writing it) | P0-4 pins the backend numerics | REL-1 (doc) | RET-3 | todo |

### `mace/cli/convert_oeq_e3nn.py` — no entry point (4)

| id | feature | source | disposition | pinned by | destination | retirement | status |
|---|---|---|---|---|---|---|---|
| `cli.convert_oeq_e3nn.input_model` | `input_model` | `mace/cli/convert_oeq_e3nn.py:57` | DROP — v1 weights are canonical and backend dispatch is automatic, so there is nothing left to convert between backends (positional) | P0-4 pins the backend numerics | REL-1 (doc) | RET-3 | todo |
| `cli.convert_oeq_e3nn.output_model` | `--output_model` | `mace/cli/convert_oeq_e3nn.py:59` | DROP — v1 weights are canonical and backend dispatch is automatic, so there is nothing left to convert between backends | P0-4 pins the backend numerics | REL-1 (doc) | RET-3 | todo |
| `cli.convert_oeq_e3nn.device` | `--device` | `mace/cli/convert_oeq_e3nn.py:61` | DROP — v1 weights are canonical and backend dispatch is automatic, so there is nothing left to convert between backends | P0-4 pins the backend numerics | REL-1 (doc) | RET-3 | todo |
| `cli.convert_oeq_e3nn.return_model` | `--return_model` | `mace/cli/convert_oeq_e3nn.py:63` | DROP — v1 weights are canonical and backend dispatch is automatic, so there is nothing left to convert between backends (library flag: return the converted model instead of writing it) | P0-4 pins the backend numerics | REL-1 (doc) | RET-3 | todo |

### `mace/cli/convert_e3nn_hybrid.py` — no entry point (4)

| id | feature | source | disposition | pinned by | destination | retirement | status |
|---|---|---|---|---|---|---|---|
| `cli.convert_e3nn_hybrid.input_model` | `input_model` | `mace/cli/convert_e3nn_hybrid.py:141` | DROP — v1 weights are canonical and backend dispatch is automatic, so there is nothing left to convert between backends (positional) | P0-4 pins the backend numerics | REL-1 (doc) | RET-3 | todo |
| `cli.convert_e3nn_hybrid.output_model` | `--output_model` | `mace/cli/convert_e3nn_hybrid.py:143` | DROP — v1 weights are canonical and backend dispatch is automatic, so there is nothing left to convert between backends | P0-4 pins the backend numerics | REL-1 (doc) | RET-3 | todo |
| `cli.convert_e3nn_hybrid.device` | `--device` | `mace/cli/convert_e3nn_hybrid.py:147` | DROP — v1 weights are canonical and backend dispatch is automatic, so there is nothing left to convert between backends | P0-4 pins the backend numerics | REL-1 (doc) | RET-3 | todo |
| `cli.convert_e3nn_hybrid.return_model` | `--return_model` | `mace/cli/convert_e3nn_hybrid.py:148` | DROP — v1 weights are canonical and backend dispatch is automatic, so there is nothing left to convert between backends (library flag: return the converted model instead of writing it) | P0-4 pins the backend numerics | REL-1 (doc) | RET-3 | todo |

### `mace_convert_device` — `mace/cli/convert_device.py` (3)

| id | feature | source | disposition | pinned by | destination | retirement | status |
|---|---|---|---|---|---|---|---|
| `cli.convert_device.model_file` | `model_file` | `mace/cli/convert_device.py:19` | KEEP — positional | ⚠️ gap (add `mace model convert-device` case to P0-5) | CLI-1 | RET-6 | todo |
| `cli.convert_device.output_file` | `--output_file` `-o` | `mace/cli/convert_device.py:15` | KEEP | ⚠️ gap (idem) | CLI-1 | RET-6 | todo |
| `cli.convert_device.target_device` | `--target_device` `-t` | `mace/cli/convert_device.py:9` | KEEP — converts device/dtype, not backend layout, which is why this CLI is explicitly not one of the five weight converters above | ⚠️ gap (idem) | CLI-1 | RET-6 | todo |

## 6. Model-level classes (12)

Every top-level class in `mace/modules/models.py` and `mace/modules/extensions.py`. Two of the
twelve are **not models** — `SHModule` and `ChebyshevBasisGeneral` are blocks that happen to live
in `extensions.py`, which is what the extractor scans; they are listed rather than filtered out so
the set stays mechanical.

| id | feature | source | disposition | pinned by | destination | retirement | status |
|---|---|---|---|---|---|---|---|
| `model.MACE` | `MACE` | `mace/modules/models.py:47` | MERGE — `BaseMACE` + a declared energy output; the class as a class disappears | P0-1 | ARCH-2, ARCH-3 | RET-1 | todo |
| `model.ScaleShiftMACE` | `ScaleShiftMACE` | `mace/modules/models.py:444` | MERGE — idem; the default energy model becomes the default configuration | P0-1 | ARCH-3 | RET-1 | todo |
| `model.AtomicDipolesMACE` | `AtomicDipolesMACE` | `mace/modules/models.py:626` | MERGE — the dipole observable | P0-3a | ARCH-3 | RET-4 | todo |
| `model.AtomicDielectricMACE` | `AtomicDielectricMACE` | `mace/modules/models.py:842` | MERGE — dipole + polarizability observables. Note this is the MACE-MDP foundation architecture, so it needs a converter as well as a reimplementation | ⚠️ gap (MDP golden → P0-3a) | ARCH-3, FM-2, FT-4 | RET-4 | todo |
| `model.EnergyDipolesMACE` | `EnergyDipolesMACE` | `mace/modules/models.py:1199` | MERGE — energy + dipole observables | ⚠️ gap (add a golden if any published model uses it) | ARCH-3 | RET-4 | todo |
| `model.MACELES` | `MACELES` | `mace/modules/extensions.py:142` | KEEP — the LES extra: latent multipoles, BEC and the external-field path | `tests/extensions/les` + P0-3c | ELEC-4 | RET-4 | todo |
| `model.PolarMACE` | `PolarMACE` | `mace/modules/extensions.py:663` | KEEP — the electrostatics extra | P0-3a | ELEC-2 | RET-4 | todo |
| `model.MagneticMACE` | `MagneticMACE` | `mace/modules/extensions.py:1428` | KEEP — the magnetic base class: magmom as an input feature, magnetic-moment observable, `dE/dm` derivative | `tests/extensions/magnetic` (rotation equivariance, inversion parity) + P0-3b | MAG-1 | RET-4 | todo |
| `model.MagneticScaleShiftMACE` | `MagneticScaleShiftMACE` | `mace/modules/extensions.py:1706` | KEEP — the CLI-reachable magnetic model | `tests/extensions/magnetic` (e2e train, eval, config round-trip) + P0-3b | MAG-1 | RET-4 | todo |
| `model.MagneticSCFMACE` | `MagneticSCFMACE` | `mace/modules/extensions.py:1968` | KEEP — **not CLI-reachable**: a wrapper applied programmatically over a model (`MagneticSCFMACE(model=…, n_scf_step=2)`). That shape is the TRN-2 model-transform hook, so it is an in-tree consumer to design the hook against | `tests/extensions/magnetic::test_run_magnetic_scf` | MAG-1, TRN-2 | RET-4 | todo |
| `model.SHModule` | `SHModule` | `mace/modules/extensions.py:1351` | KEEP — a spherical-harmonics block wrapping `sphericart.torch.SolidHarmonics`, not a model. Notable as a working in-tree precedent for a non-e3nn spherical-harmonics backend | `tests/extensions/magnetic` (indirect) | ARCH-1, ARCH-2 | RET-4 | todo |
| `model.ChebyshevBasisGeneral` | `ChebyshevBasisGeneral` | `mace/modules/extensions.py:1374` | KEEP — a radial basis living in `extensions.py`, not a model; belongs with the `--radial_type` bases of §3.2 | ⚠️ gap (add case to P0-7) | ARCH-1 | RET-4 | todo |

## 7. String→class registries (21)

The four dicts in `mace/modules/__init__.py` that connect CLI values to implementations. An entry
that is not here is not reachable from the CLI, so this set is exactly the user-selectable block
surface.

| id | feature | source | disposition | pinned by | destination | retirement | status |
|---|---|---|---|---|---|---|---|
| `reg.RealAgnosticResidualInteractionBlock` | `RealAgnosticResidualInteractionBlock` — interaction_classes | `mace/modules/__init__.py:71` | KEEP — the standard interaction block | P0-1 | ARCH-2 | RET-1 | todo |
| `reg.RealAgnosticInteractionBlock` | `RealAgnosticInteractionBlock` — interaction_classes | `mace/modules/__init__.py:73` | KEEP — the default first layer | P0-1 | ARCH-2 | RET-1 | todo |
| `reg.RealAgnosticDensityInteractionBlock` | `RealAgnosticDensityInteractionBlock` — interaction_classes | `mace/modules/__init__.py:74` | KEEP — foundation-model architecture, not a research variant: MACE-MP-0b2 S/M/L and MACE-mh-0 use it in the first layer | ⚠️ gap (needs an MP-0b2 or mh-0 golden → P0-2) | ARCH-2 | RET-1 | todo |
| `reg.RealAgnosticDensityResidualInteractionBlock` | `RealAgnosticDensityResidualInteractionBlock` — interaction_classes | `mace/modules/__init__.py:75` | KEEP — idem, used in the remaining layers of the same published models | ⚠️ gap (idem → P0-2) | ARCH-2 | RET-1 | todo |
| `reg.RealAgnosticResidualNonLinearInteractionBlock` | `RealAgnosticResidualNonLinearInteractionBlock` — interaction_classes | `mace/modules/__init__.py:76` | KEEP — the interaction block of MACE-Polar S/M/L | P0-3a | ARCH-2, ELEC-2 | RET-4 | todo |
| `reg.RealAgnosticAttResidualInteractionBlock` | `RealAgnosticAttResidualInteractionBlock` — interaction_classes | `mace/modules/__init__.py:72` | DROP — unlike the Density blocks it appears in no `finetuning_utils` branch and no converter, only in the registry and the parser choices: a research variant with no published model, no test and no owner | — | REL-1 (doc) | RET-1 | todo |
| `reg.MagneticRealAgnosticSpinOrbitCoupledDensityInteractionBlock` | `MagneticRealAgnosticSpinOrbitCoupledDensityInteractionBlock` — interaction_classes | `mace/modules/__init__.py:78` | KEEP — the magnetic extra's first layer; a Density variant, so it inherits whatever ARCH-2 does for that family | `tests/extensions/magnetic` | MAG-1, ARCH-2 | RET-4 | todo |
| `reg.MagneticRealAgnosticResidueSpinOrbitCoupledDensityInteractionBlock` | `MagneticRealAgnosticResidueSpinOrbitCoupledDensityInteractionBlock` — interaction_classes | `mace/modules/__init__.py:77` | KEEP — idem, the residual variant | `tests/extensions/magnetic` | MAG-1, ARCH-2 | RET-4 | todo |
| `reg.LinearReadoutBlock` | `LinearReadoutBlock` — readout_classes | `mace/modules/__init__.py:82` | KEEP | P0-1 | ARCH-3 | RET-1 | todo |
| `reg.NonLinearReadoutBlock` | `NonLinearReadoutBlock` — readout_classes | `mace/modules/__init__.py:85` | KEEP | P0-1 | ARCH-3 | RET-1 | todo |
| `reg.LinearDipoleReadoutBlock` | `LinearDipoleReadoutBlock` — readout_classes | `mace/modules/__init__.py:83` | MERGE — an observable head declared in the output spec | P0-3a | ARCH-3, CORE-1 | RET-4 | todo |
| `reg.NonLinearDipoleReadoutBlock` | `NonLinearDipoleReadoutBlock` — readout_classes | `mace/modules/__init__.py:84` | MERGE — idem | P0-3a | ARCH-3, CORE-1 | RET-4 | todo |
| `reg.NonLinearBiasReadoutBlock` | `NonLinearBiasReadoutBlock` — readout_classes | `mace/modules/__init__.py:86` | KEEP — a readout of the published MACE-Polar models, and it backs the fukui source map; not a registry leftover | P0-3a | ARCH-3, ELEC-2 | RET-4 | todo |
| `reg.GeneralNonLinearBiasReadoutBlock` | `GeneralNonLinearBiasReadoutBlock` — readout_classes | `mace/modules/__init__.py:87` | KEEP — used internally by `field_blocks.py`, load-bearing for the polar model | P0-3a | ARCH-3, ELEC-2 | RET-4 | todo |
| `reg.std_scaling` | `std_scaling` — scaling_classes | `mace/modules/__init__.py:91` | KEEP | P0-1 | ARCH-3 | RET-1 | todo |
| `reg.rms_forces_scaling` | `rms_forces_scaling` — scaling_classes | `mace/modules/__init__.py:92` | KEEP — the default | P0-1 | ARCH-3 | RET-1 | todo |
| `reg.rms_dipoles_scaling` | `rms_dipoles_scaling` — scaling_classes | `mace/modules/__init__.py:93` | MERGE — observable normalization | P0-3a | ARCH-3, CORE-1 | RET-4 | todo |
| `reg.silu` | `silu` — gate_dict | `mace/modules/__init__.py:99` | KEEP — the default gate | P0-1 | ARCH-3 | RET-1 | todo |
| `reg.tanh` | `tanh` — gate_dict | `mace/modules/__init__.py:98` | KEEP | ⚠️ gap (add case to P0-7) | ARCH-3 | RET-1 | todo |
| `reg.abs` | `abs` — gate_dict | `mace/modules/__init__.py:97` | KEEP | ⚠️ gap (add case to P0-7) | ARCH-3 | RET-1 | todo |
| `reg.None` | `None` — gate_dict | `mace/modules/__init__.py:100` | KEEP — the string `"None"`, meaning no gate | ⚠️ gap (add case to P0-7) | ARCH-3 | RET-1 | todo |

## 8. Loss classes (10)

All ten MERGE into composable per-stage losses (TRN-2): each becomes a documented composition
preset producing identical numbers, and P0-6's hand-computed cases are the acceptance test. The
eleven `--loss` scheme names of §3.7 map onto these ten classes.

| id | feature | source | disposition | pinned by | destination | retirement | status |
|---|---|---|---|---|---|---|---|
| `loss.WeightedEnergyForcesLoss` | `WeightedEnergyForcesLoss` | `mace/modules/loss.py:246` | MERGE — a composition preset (the `ef`/`weighted` schemes) | P0-6 | TRN-2 | RET-6 | todo |
| `loss.WeightedForcesLoss` | `WeightedForcesLoss` | `mace/modules/loss.py:272` | MERGE — a composition preset (the `forces_only` scheme) | P0-6 | TRN-2 | RET-6 | todo |
| `loss.WeightedEnergyForcesStressLoss` | `WeightedEnergyForcesStressLoss` | `mace/modules/loss.py:290` | MERGE — a composition preset (the `stress` scheme) | P0-6 | TRN-2 | RET-6 | todo |
| `loss.WeightedHuberEnergyForcesStressLoss` | `WeightedHuberEnergyForcesStressLoss` | `mace/modules/loss.py:325` | MERGE — a composition preset (the `huber` scheme) | P0-6 | TRN-2 | RET-6 | todo |
| `loss.UniversalLoss` | `UniversalLoss` | `mace/modules/loss.py:391` | MERGE — a composition preset (the `universal` scheme) | P0-6 | TRN-2 | RET-6 | todo |
| `loss.WeightedEnergyForcesVirialsLoss` | `WeightedEnergyForcesVirialsLoss` | `mace/modules/loss.py:506` | MERGE — a composition preset (the `virials` scheme) | P0-6 | TRN-2 | RET-6 | todo |
| `loss.DipoleSingleLoss` | `DipoleSingleLoss` | `mace/modules/loss.py:543` | MERGE — a composition preset (the `dipole` scheme) | P0-6 | TRN-2 | RET-6 | todo |
| `loss.DipolePolarLoss` | `DipolePolarLoss` | `mace/modules/loss.py:563` | MERGE — a composition preset (the `dipole_polar` scheme) | P0-6 | TRN-2 | RET-6 | todo |
| `loss.WeightedEnergyForcesDipoleLoss` | `WeightedEnergyForcesDipoleLoss` | `mace/modules/loss.py:601` | MERGE — a composition preset (the `energy_forces_dipole` scheme) | P0-6 | TRN-2 | RET-6 | todo |
| `loss.WeightedEnergyForcesL1L2Loss` | `WeightedEnergyForcesL1L2Loss` | `mace/modules/loss.py:636` | MERGE — a composition preset (the `l1l2energyforces` scheme) | P0-6 | TRN-2 | RET-6 | todo |

## 9. Calculator constructor and exports

### 9.1 `__init__` parameters (23)

`MACECalculator.__init__` carries 22; the 23rd (`magmom_key`) is added by
`MagneticMACECalculator`, a second `Calculator` subclass rather than a mode of the first, so its
`__init__` is a second public surface. The set is the union: a knob that exists on only one of the
two calculators is still a knob.

| id | feature | source | disposition | pinned by | destination | retirement | status |
|---|---|---|---|---|---|---|---|
| `calc.param.model_paths` | `model_paths` — MACECalculator | `mace/calculators/mace.py:105` | KEEP — one path, a glob, or several for a committee | P0-5 | DEP-1 | RET-6 | todo |
| `calc.param.models` | `models` — MACECalculator | `mace/calculators/mace.py:106` | KEEP — pre-loaded model objects instead of paths | P0-5 | DEP-1 | RET-6 | todo |
| `calc.param.device` | `device` — MACECalculator | `mace/calculators/mace.py:107` | KEEP | P0-5 | DEP-1 | RET-6 | todo |
| `calc.param.default_dtype` | `default_dtype` — MACECalculator | `mace/calculators/mace.py:110` | MERGE — `PrecisionConfig` | P0-5 | BKD-2, DEP-1 | RET-6 | todo |
| `calc.param.energy_units_to_eV` | `energy_units_to_eV` — MACECalculator | `mace/calculators/mace.py:108` | KEEP — unit conversion on the way out | ⚠️ gap (unit-conversion case → P0-5) | DEP-1, CORE-3 | RET-6 | todo |
| `calc.param.length_units_to_A` | `length_units_to_A` — MACECalculator | `mace/calculators/mace.py:109` | KEEP — idem | ⚠️ gap (idem) | DEP-1, CORE-3 | RET-6 | todo |
| `calc.param.charges_key` | `charges_key` — MACECalculator | `mace/calculators/mace.py:111` | KEEP — property-key convention | P0-5 | CORE-3, DEP-1 | RET-6 | todo |
| `calc.param.info_keys` | `info_keys` — MACECalculator | `mace/calculators/mace.py:112` | KEEP — which `atoms.info` entries become graph-level inputs | ⚠️ gap (key pass-through → P0-5) | CORE-3, DEP-1 | RET-6 | todo |
| `calc.param.arrays_keys` | `arrays_keys` — MACECalculator | `mace/calculators/mace.py:113` | KEEP — idem for `atoms.arrays` | ⚠️ gap (idem) | CORE-3, DEP-1 | RET-6 | todo |
| `calc.param.model_type` | `model_type` — MACECalculator | `mace/calculators/mace.py:114` | MERGE — auto-detected from model metadata; asking the user to name the model family is asking them to get it wrong | P0-5 | CORE-2, DEP-1 | RET-6 | todo |
| `calc.param.compile_mode` | `compile_mode` — MACECalculator | `mace/calculators/mace.py:115` | KEEP | `tests/unit/test_compile.py` | BKD-2, DEP-1 | RET-6 | todo |
| `calc.param.fullgraph` | `fullgraph` — MACECalculator | `mace/calculators/mace.py:116` | KEEP | `tests/unit/test_compile.py` | BKD-2, DEP-1 | RET-6 | todo |
| `calc.param.pad_num_atoms` | `pad_num_atoms` — MACECalculator | `mace/calculators/mace.py:119` | KEEP — graph padding, so a compiled graph is not recaptured per frame | ⚠️ gap (padding contract → P0-5) | BKD-2, DEP-1 | RET-6 | todo |
| `calc.param.pad_num_edges` | `pad_num_edges` — MACECalculator | `mace/calculators/mace.py:120` | KEEP — idem | ⚠️ gap (idem) | BKD-2, DEP-1 | RET-6 | todo |
| `calc.param.warmup` | `warmup` — MACECalculator | `mace/calculators/mace.py:121` | KEEP — one throwaway forward so the first real call is not the compile | ⚠️ gap (idem) | BKD-2, DEP-1 | RET-6 | todo |
| `calc.param.enable_cueq` | `enable_cueq` — MACECalculator | `mace/calculators/mace.py:117` | MERGE — backend dispatch config | P0-4 | BKD-1, BKD-3 | RET-6 | todo |
| `calc.param.enable_oeq` | `enable_oeq` — MACECalculator | `mace/calculators/mace.py:118` | MERGE — idem | P0-4 | BKD-1, BKD-3 | RET-6 | todo |
| `calc.param.compute_bec` | `compute_bec` — MACECalculator | `mace/calculators/mace.py:122` | KEEP — Born effective charges from the LES/polar path | ⚠️ gap (add to P0-3a/P0-3c) | ELEC-2, ELEC-4, DEP-1a | RET-6 | todo |
| `calc.param.external_field` | `external_field` — MACECalculator | `mace/calculators/mace.py:123` | KEEP — the applied field of the LES/polar path | ⚠️ gap (add to P0-3c) | ELEC-4, DEP-1a | RET-6 | todo |
| `calc.param.eps_infty` | `eps_infty` — MACECalculator | `mace/calculators/mace.py:124` | KEEP — high-frequency dielectric constant used by the field path | ⚠️ gap (add to P0-3c) | ELEC-4, DEP-1a | RET-6 | todo |
| `calc.param.electric_field_unit` | `electric_field_unit` — MACECalculator | `mace/calculators/mace.py:125` | KEEP — unit convention for the applied field | ⚠️ gap (add to P0-3c) | ELEC-4, CORE-3 | RET-6 | todo |
| `calc.param.keep_neutral` | `keep_neutral` — MACECalculator | `mace/calculators/mace.py:126` | KEEP — charge-neutrality enforcement in the field path | ⚠️ gap (add to P0-3c) | ELEC-4, DEP-1a | RET-6 | todo |
| `calc.param.magmom_key` | `magmom_key` — MagneticMACECalculator | `mace/calculators/mace.py:993` | KEEP — property-key convention; `MagneticMACECalculator` only | `tests/extensions/magnetic` | CORE-3, MAG-1, DEP-1a | RET-6 | todo |

### 9.2 Exports (`mace/calculators/__init__.py`, 9)

| id | feature | source | disposition | pinned by | destination | retirement | status |
|---|---|---|---|---|---|---|---|
| `calc.export.MACECalculator` | `MACECalculator` | `mace/calculators/__init__.py:12` | KEEP — the one ASE calculator | P0-5 | DEP-1 | RET-6 | todo |
| `calc.export.MagneticMACECalculator` | `MagneticMACECalculator` | `mace/calculators/__init__.py:12` | KEEP — a separate `Calculator` subclass (~530 lines), not a mode of `MACECalculator`. The v1 question DEP-1a answers is whether it stays a second class or collapses into the one calculator once magmom is just another input feature | `tests/extensions/magnetic` (eval path) | MAG-1, DEP-1a | RET-6 | todo |
| `calc.export.LAMMPS_MACE` | `LAMMPS_MACE` | `mace/calculators/__init__.py:12` | DROP — the TorchScript wrapper dies with the TorchScript export format; the MLIAP path replaces it | P0-5 (export golden pins the numerics) | DEP-2 | RET-5 | todo |
| `calc.export.mace_mp` | `mace_mp` | `mace/calculators/__init__.py:12` | KEEP | P0-2 | FM-2, FM-3 | RET-6 | todo |
| `calc.export.mace_off` | `mace_off` | `mace/calculators/__init__.py:12` | KEEP | P0-2 | FM-2, FM-3 | RET-6 | todo |
| `calc.export.mace_polar` | `mace_polar` | `mace/calculators/__init__.py:12` | KEEP | P0-3a | ELEC-2, FM-3 | RET-6 | todo |
| `calc.export.mace_mdp` | `mace_mdp` | `mace/calculators/__init__.py:12` | KEEP — a published dipole/polarizability foundation model with released calculator support | ⚠️ gap (MDP golden → P0-3a) | FM-2, FT-4 | RET-6 | todo |
| `calc.export.mace_omol` | `mace_omol` | `mace/calculators/__init__.py:12` | KEEP — a recent, large, published multi-head model; converts with heads intact | ⚠️ gap (OMOL golden → P0-2) | FM-2, FM-3 | RET-6 | todo |
| `calc.export.mace_anicc` | `mace_anicc` | `mace/calculators/__init__.py:12` | DROP — a 2023 organic-chemistry model superseded by MACE-OFF, and the only loader with a divergent signature (`model_path` instead of `model`): an API exception for an obsolete artifact. Its tracked checkpoint `mace/calculators/foundations_models/ani500k_large_CC.model` goes with it; REL-1 says "use MACE-OFF" | — | REL-1 (doc) | RET-6 | todo |

## 10. Optional-dependency extras (12)

From `setup.cfg` `[options.extras_require]`. Two facts here feed design decisions rather than
packaging: `magnetic` pins **`sphericart-torch`**, a shipped dependency on a non-e3nn
spherical-harmonics backend (see `SHModule`, §6), and it also declares **external
`torch-geometric`** while `mace/data/augmentation.py` imports the real package *and* the vendored
copy — so the tree depends on both at once (§19).

| id | feature | source | disposition | pinned by | destination | retirement | status |
|---|---|---|---|---|---|---|---|
| `extra.wandb` | `[wandb]` | `setup.cfg` | KEEP | ⚠️ gap (offline-mode smoke) | TRN-3 | RET-6 | todo |
| `extra.fpsample` | `[fpsample]` | `setup.cfg` | KEEP — fast farthest-point sampling for fine-tuning selection | ⚠️ gap (port `tests/workflows/test_finetuning_select.py`) | FT-1 | RET-6 | todo |
| `extra.schedulefree` | `[schedulefree]` | `setup.cfg` | KEEP | `tests/extensions/schedulefree` | TRN-2 | RET-6 | todo |
| `extra.torchsim` | `[torchsim]` | `setup.cfg` | KEEP — a first-class deployment path, not a secondary integration; the coupling to torch-sim's still-moving API becomes MACE's problem, so the version is pinned in `requirements/` | `tests/extensions/torchsim` | DEP-4 | RET-6 | todo |
| `extra.magnetic` | `[magnetic]` | `setup.cfg` | KEEP — `sphericart-torch` + `torch-geometric` | `tests/extensions/magnetic` | MAG-1, INF-1 | RET-6 | todo |
| `extra.cueq` | `[cueq]` | `setup.cfg` | KEEP — the backend extra naming is revisited at INF-1 | P0-4 | BKD-3 | RET-3 | todo |
| `extra.cueq-cuda-11` | `[cueq-cuda-11]` | `setup.cfg` | KEEP — idem; the ops major must match `torch.version.cuda`, not the newest available | P0-4 | BKD-3 | RET-3 | todo |
| `extra.cueq-cuda-12` | `[cueq-cuda-12]` | `setup.cfg` | KEEP — idem | P0-4 | BKD-3 | RET-3 | todo |
| `extra.cueq-cuda-13` | `[cueq-cuda-13]` | `setup.cfg` | KEEP — idem; cu13 ops start at cuequivariance 0.7.0 | P0-4 | BKD-3 | RET-3 | todo |
| `extra.oeq` | `[oeq]` | `setup.cfg` | KEEP — OpenEquivariance, the AMD-capable accelerated backend | P0-4 | BKD-3 | RET-3 | todo |
| `extra.dev` | `[dev]` | `setup.cfg` | KEEP — the lint/format toolchain, retooled at INF-1 and path-scoped at INF-4 | the lint job itself | INF-1, INF-4 | n/a — tooling metadata, not legacy code | todo |
| `extra.test` | `[test]` | `setup.cfg` | KEEP — pytest and its plugins, split out of `dev` so a test job need not install the linters | the suite itself | INF-1 | n/a — tooling metadata, not legacy code | todo |

## 11. Model output keys (43)

The keys of the dicts the model `forward`s return. This is the contract every consumer reads —
the ASE calculator, `mace_eval_configs`, the LAMMPS runtimes, the training loop — so a renamed
key is a silent breakage in four places at once. CORE-1 replaces the untyped dict with a typed
`MACEOutputs` whose fields are these keys; the disposition column says which name survives that
move and which is absorbed into a declared observable.

| id | feature | source | disposition | pinned by | destination | retirement | status |
|---|---|---|---|---|---|---|---|
| `out.model.energy` | `energy` — first declared by `MACE` | `mace/modules/models.py:428` | KEEP — a field of the typed outputs | P0-1 | CORE-1, ARCH-4 | RET-1 | todo |
| `out.model.node_energy` | `node_energy` — first declared by `MACE` | `mace/modules/models.py:429` | KEEP — idem | P0-1 | CORE-1, ARCH-3 | RET-1 | todo |
| `out.model.interaction_energy` | `interaction_energy` — first declared by `ScaleShiftMACE` | `mace/modules/models.py:612` | KEEP — idem (total minus the E0s) | P0-1 | CORE-1, ARCH-3 | RET-1 | todo |
| `out.model.forces` | `forces` — first declared by `MACE` | `mace/modules/models.py:431` | KEEP — idem | P0-1 | CORE-1, ARCH-4 | RET-1 | todo |
| `out.model.stress` | `stress` — first declared by `MACE` | `mace/modules/models.py:434` | KEEP — idem | P0-1 | CORE-1, ARCH-4 | RET-1 | todo |
| `out.model.virials` | `virials` — first declared by `MACE` | `mace/modules/models.py:433` | KEEP — idem | P0-1 | CORE-1, ARCH-4 | RET-1 | todo |
| `out.model.hessian` | `hessian` — first declared by `MACE` | `mace/modules/models.py:438` | KEEP — idem | `tests/foundations/test_hessian.py` | CORE-1, ARCH-4, DEP-1 | RET-1 | todo |
| `out.model.edge_forces` | `edge_forces` — first declared by `MACE` | `mace/modules/models.py:432` | KEEP — per-edge forces; a first-class observable because the LAMMPS MLIAP path needs them | `tests/integrations/lammps` | CORE-1, DEP-2 | RET-1 | todo |
| `out.model.atomic_virials` | `atomic_virials` — first declared by `MACE` | `mace/modules/models.py:435` | KEEP — per-atom virials | ⚠️ gap (add to P0-6) | CORE-1, ARCH-4 | RET-1 | todo |
| `out.model.atomic_stresses` | `atomic_stresses` — first declared by `MACE` | `mace/modules/models.py:436` | KEEP — per-atom stresses | ⚠️ gap (add to P0-6) | CORE-1, ARCH-4 | RET-1 | todo |
| `out.model.contributions` | `contributions` — first declared by `MACE` | `mace/modules/models.py:430` | KEEP — per-layer energy contributions | ⚠️ gap (add assert to P0-5 eval) | CORE-1, ARCH-3 | RET-1 | todo |
| `out.model.node_feats` | `node_feats` — first declared by `MACE` | `mace/modules/models.py:439` | KEEP — the descriptor surface; `BaseMACE` exposes it through the descriptor API | ⚠️ gap (descriptor contract → P0-5) | ARCH-2 | RET-1 | todo |
| `out.model.displacement` | `displacement` — first declared by `MACE` | `mace/modules/models.py:437` | MERGE — the strain displacement is internal machinery of the derivative engine, not a user-facing output; v1 does not return it | P0-6 (stress via strain) | ARCH-4 | RET-1 | todo |
| `out.model.dipole` | `dipole` — first declared by `AtomicDipolesMACE` | `mace/modules/models.py:835` | KEEP — a declared observable | P0-3a | CORE-1, ARCH-3 | RET-4 | todo |
| `out.model.atomic_dipoles` | `atomic_dipoles` — first declared by `AtomicDipolesMACE` | `mace/modules/models.py:836` | KEEP — idem | P0-3a | CORE-1, ARCH-3 | RET-4 | todo |
| `out.model.charges` | `charges` — first declared by `AtomicDielectricMACE` | `mace/modules/models.py:1187` | KEEP — idem | P0-3a | CORE-1, ARCH-3 | RET-4 | todo |
| `out.model.polarizability` | `polarizability` — first declared by `AtomicDielectricMACE` | `mace/modules/models.py:1190` | KEEP — idem | P0-3a | CORE-1, ELEC-2 | RET-4 | todo |
| `out.model.polarizability_sh` | `polarizability_sh` — first declared by `AtomicDielectricMACE` | `mace/modules/models.py:1191` | KEEP — the spherical-harmonics form of the polarizability | P0-3a | CORE-1, ELEC-2 | RET-4 | todo |
| `out.model.dmu_dr` | `dmu_dr` — first declared by `AtomicDielectricMACE` | `mace/modules/models.py:1192` | KEEP — dipole derivative (the dielectric family's IR path) | ⚠️ gap (add to the P0-3a MDP case) | CORE-1, FT-4 | RET-4 | todo |
| `out.model.dalpha_dr` | `dalpha_dr` — first declared by `AtomicDielectricMACE` | `mace/modules/models.py:1193` | KEEP — polarizability derivative (Raman) | ⚠️ gap (idem) | CORE-1, FT-4 | RET-4 | todo |
| `out.model.les_energy` | `les_energy` — first declared by `MACELES` | `mace/modules/extensions.py:648` | KEEP — the LES long-range energy term | P0-3c | ELEC-4 | RET-4 | todo |
| `out.model.latent_charges` | `latent_charges` — first declared by `MACELES` | `mace/modules/extensions.py:649` | KEEP — LES latent multipoles | P0-3c | ELEC-4 | RET-4 | todo |
| `out.model.latent_dipoles` | `latent_dipoles` — first declared by `MACELES` | `mace/modules/extensions.py:650` | KEEP — idem | P0-3c | ELEC-4 | RET-4 | todo |
| `out.model.latent_kappas` | `latent_kappas` — first declared by `MACELES` | `mace/modules/extensions.py:651` | KEEP — idem | P0-3c | ELEC-4 | RET-4 | todo |
| `out.model.latent_alphas` | `latent_alphas` — first declared by `MACELES` | `mace/modules/extensions.py:652` | KEEP — idem | P0-3c | ELEC-4 | RET-4 | todo |
| `out.model.latent_quads` | `latent_quads` — first declared by `MACELES` | `mace/modules/extensions.py:653` | KEEP — idem | P0-3c | ELEC-4 | RET-4 | todo |
| `out.model.BEC` | `BEC` — first declared by `MACELES` | `mace/modules/extensions.py:654` | KEEP — Born effective charges | ⚠️ gap (add to P0-3c) | ELEC-4, CORE-1 | RET-4 | todo |
| `out.model.electrostatic_energy` | `electrostatic_energy` — first declared by `PolarMACE` | `mace/modules/extensions.py:1342` | KEEP — polar energy decomposition | P0-3a | ELEC-2 | RET-4 | todo |
| `out.model.electron_energy` | `electron_energy` — first declared by `PolarMACE` | `mace/modules/extensions.py:1343` | KEEP — idem | P0-3a | ELEC-2 | RET-4 | todo |
| `out.model.electrostatic_potentials` | `electrostatic_potentials` — first declared by `PolarMACE` | `mace/modules/extensions.py:1344` | KEEP — idem | ⚠️ gap (add to the P0-3a polar case) | ELEC-2 | RET-4 | todo |
| `out.model.density_coefficients` | `density_coefficients` — first declared by `PolarMACE` | `mace/modules/extensions.py:1331` | KEEP — the polar density expansion, consumed by `mace_polar_density_cube` | `tests/extensions/polar/test_polar_density_cube.py` | ELEC-2 | RET-4 | todo |
| `out.model.spin_density` | `spin_density` — first declared by `PolarMACE` | `mace/modules/extensions.py:1332` | KEEP — idem | ⚠️ gap (add to the P0-3a polar case) | ELEC-2 | RET-4 | todo |
| `out.model.spin_charge_density` | `spin_charge_density` — first declared by `PolarMACE` | `mace/modules/extensions.py:1345` | KEEP — idem | ⚠️ gap (idem) | ELEC-2 | RET-4 | todo |
| `out.model.spins` | `spins` — first declared by `PolarMACE` | `mace/modules/extensions.py:1339` | KEEP — per-atom spin populations | ⚠️ gap (idem) | ELEC-2 | RET-4 | todo |
| `out.model.total_charge` | `total_charge` — first declared by `PolarMACE` | `mace/modules/extensions.py:1341` | KEEP — echoed back as an output so a consumer can check what was imposed | ⚠️ gap (idem) | ELEC-2, CORE-3 | RET-4 | todo |
| `out.model.fermi_level` | `fermi_level` — first declared by `PolarMACE` | `mace/modules/extensions.py:1336` | KEEP — idem | ⚠️ gap (idem) | ELEC-2 | RET-4 | todo |
| `out.model.external_field` | `external_field` — first declared by `PolarMACE` | `mace/modules/extensions.py:1337` | KEEP — idem | ⚠️ gap (idem) | ELEC-2, ELEC-4 | RET-4 | todo |
| `out.model.fukui_functions` | `fukui_functions` — first declared by `PolarMACE` | `mace/modules/extensions.py:1346` | KEEP — the fukui reactivity output | ⚠️ gap (add to the P0-3a polar case) | ELEC-2, DEP-1a | RET-4 | todo |
| `out.model.charges_history` | `charges_history` — first declared by `PolarMACE` | `mace/modules/extensions.py:1333` | MERGE — the per-iteration trace of the fixed-point solve; a solver diagnostic, so it belongs to the solver-dispatch layer rather than the model's outputs | ⚠️ gap (fixed-point convergence → P0-3a) | ELEC-1 | RET-4 | todo |
| `out.model.magforces` | `magforces` — first declared by `MagneticScaleShiftMACE` | `mace/modules/extensions.py:1951` | KEEP — `dE/dm`, a declared derivative exactly like forces | `tests/extensions/magnetic` + P0-3b | CORE-1, MAG-1 | RET-4 | todo |
| `out.model.scf_steps` | `scf_steps` — first declared by `MagneticSCFMACE` | `mace/modules/extensions.py:2102` | MERGE — SCF solver diagnostics belong to the TRN-2 model-transform hook, not to the model's output contract | `tests/extensions/magnetic::test_run_magnetic_scf` | TRN-2, MAG-1 | RET-4 | todo |
| `out.model.scf_energy_history` | `scf_energy_history` — first declared by `MagneticSCFMACE` | `mace/modules/extensions.py:2099` | MERGE — idem | `tests/extensions/magnetic::test_run_magnetic_scf` | TRN-2, MAG-1 | RET-4 | todo |
| `out.model.equilibrated_magmom` | `equilibrated_magmom` — first declared by `MagneticSCFMACE` | `mace/modules/extensions.py:2103` | KEEP — the converged magnetic moments are a result, not a diagnostic | `tests/extensions/magnetic::test_run_magnetic_scf` | MAG-1, CORE-1 | RET-4 | todo |

## 12. Calculator and eval output keys (31 + 13)

### 12.1 `Calculator.results` keys (31)

What an ASE user reads back. Four shapes contribute and the extractor covers all four:
`implemented_properties` lists, direct `self.results[...]` assignments, the `results_map` table,
and the committee suffixes derived from `results_store_ensemble` (`_comm` = the per-model stack,
`_var` = its variance). ASE's own vocabulary (`energy`, `free_energy`, `energies`, `forces`,
`stress`, `stresses`) is fixed by ASE, not by MACE, so those names are KEEP by definition.

| id | feature | source | disposition | pinned by | destination | retirement | status |
|---|---|---|---|---|---|---|---|
| `out.calc.energy` | `energy` — results_map | `mace/calculators/mace.py:720` | KEEP — ASE's own property name; the calculator does not get to rename it | P0-5 | DEP-1 | RET-6 | todo |
| `out.calc.free_energy` | `free_energy` — self.results[...] | `mace/calculators/mace.py:785` | KEEP — ASE's own property name; the calculator does not get to rename it (ASE requires it alongside `energy`) | P0-5 | DEP-1 | RET-6 | todo |
| `out.calc.energies` | `energies` — self.results[...] | `mace/calculators/mace.py:787` | KEEP — ASE's own property name; the calculator does not get to rename it (per-atom energies) | P0-5 | DEP-1 | RET-6 | todo |
| `out.calc.forces` | `forces` — results_map | `mace/calculators/mace.py:722` | KEEP — ASE's own property name; the calculator does not get to rename it | P0-5 | DEP-1 | RET-6 | todo |
| `out.calc.stress` | `stress` — results_map | `mace/calculators/mace.py:723` | KEEP — ASE's own property name; the calculator does not get to rename it; note it is converted to Voigt 6-vector on the way out | P0-5 | DEP-1 | RET-6 | todo |
| `out.calc.stresses` | `stresses` — results_map | `mace/calculators/mace.py:724` | KEEP — ASE's own property name; the calculator does not get to rename it (per-atom, Voigt) | ⚠️ gap (add to P0-5) | DEP-1 | RET-6 | todo |
| `out.calc.virials` | `virials` — results_map | `mace/calculators/mace.py:729` | KEEP — not an ASE property but a MACE one, exposed per-atom | ⚠️ gap (add to P0-5) | DEP-1 | RET-6 | todo |
| `out.calc.node_energy` | `node_energy` — results_map | `mace/calculators/mace.py:721` | KEEP — per-atom energy with the E0s subtracted, which is *not* the same array as `energies`; both are exposed and both must stay distinguishable | ⚠️ gap (add to P0-5) | DEP-1, ARCH-3 | RET-6 | todo |
| `out.calc.dipole` | `dipole` — results_map | `mace/calculators/mace.py:734` | KEEP | P0-3a | DEP-1a | RET-6 | todo |
| `out.calc.charges` | `charges` — results_map | `mace/calculators/mace.py:735` | KEEP | P0-3a | DEP-1a | RET-6 | todo |
| `out.calc.polarizability` | `polarizability` — results_map | `mace/calculators/mace.py:736` | KEEP | P0-3a | DEP-1a | RET-6 | todo |
| `out.calc.polarizability_sh` | `polarizability_sh` — results_map | `mace/calculators/mace.py:737` | KEEP | P0-3a | DEP-1a | RET-6 | todo |
| `out.calc.bec` | `bec` — self.results[...] | `mace/calculators/mace.py:807` | KEEP — Born effective charges | ⚠️ gap (add to P0-3a/P0-3c) | ELEC-2, ELEC-4, DEP-1a | RET-6 | todo |
| `out.calc.interaction_energy` | `interaction_energy` — results_map | `mace/calculators/mace.py:742` | KEEP — polar energy decomposition | P0-3a | ELEC-2, DEP-1a | RET-6 | todo |
| `out.calc.electrostatic_energy` | `electrostatic_energy` — results_map | `mace/calculators/mace.py:747` | KEEP — idem | P0-3a | ELEC-2, DEP-1a | RET-6 | todo |
| `out.calc.electron_energy` | `electron_energy` — results_map | `mace/calculators/mace.py:752` | KEEP — idem | P0-3a | ELEC-2, DEP-1a | RET-6 | todo |
| `out.calc.spins` | `spins` — results_map | `mace/calculators/mace.py:753` | KEEP | ⚠️ gap (add to the P0-3a polar case) | ELEC-2, DEP-1a | RET-6 | todo |
| `out.calc.density_coefficients` | `density_coefficients` — results_map | `mace/calculators/mace.py:754` | KEEP | `tests/extensions/polar/test_polar_density_cube.py` | ELEC-2, DEP-1a | RET-6 | todo |
| `out.calc.spin_charge_density` | `spin_charge_density` — results_map | `mace/calculators/mace.py:755` | KEEP | ⚠️ gap (add to the P0-3a polar case) | ELEC-2, DEP-1a | RET-6 | todo |
| `out.calc.fukui_functions` | `fukui_functions` — implemented_properties | `mace/calculators/mace.py:227` | KEEP | ⚠️ gap (add to the P0-3a polar case) | ELEC-2, DEP-1a | RET-6 | todo |
| `out.calc.LES_alphas` | `LES_alphas` — self.results[...] | `mace/calculators/mace.py:799` | MERGE — the calculator renames the model's `latent_alphas`; v1 exposes one name for one quantity, and a per-surface rename is exactly the kind of thing that makes a key ungreppable | ⚠️ gap (add to P0-3c) | ELEC-4, DEP-1a | RET-6 | todo |
| `out.calc.LES_kappas` | `LES_kappas` — self.results[...] | `mace/calculators/mace.py:803` | MERGE — idem, from `latent_kappas` | ⚠️ gap (add to P0-3c) | ELEC-4, DEP-1a | RET-6 | todo |
| `out.calc.MACE_magmoms` | `MACE_magmoms` — self.results[...] | `mace/calculators/mace.py:1411` | MERGE — idem: the magnetic calculator's spelling of the magnetic-moment observable, also written back into `atoms.arrays` | `tests/extensions/magnetic` | MAG-1, DEP-1a | RET-6 | todo |
| `out.calc.energy_comm` | `energy_comm` — committee | `mace/calculators/mace.py:718` | KEEP — committee output (per-model energies). The committee is the part of `mace_active_learning_md` that MACE must keep guaranteeing (§1) | ⚠️ gap (committee contract → P0-5) | DEP-1 | RET-6 | todo |
| `out.calc.energy_var` | `energy_var` — committee | `mace/calculators/mace.py:718` | KEEP — committee output (energy variance). The committee is the part of `mace_active_learning_md` that MACE must keep guaranteeing (§1) | ⚠️ gap (idem) | DEP-1 | RET-6 | todo |
| `out.calc.forces_comm` | `forces_comm` — committee | `mace/calculators/mace.py:718` | KEEP — committee output (per-model forces). The committee is the part of `mace_active_learning_md` that MACE must keep guaranteeing (§1) | ⚠️ gap (idem) | DEP-1 | RET-6 | todo |
| `out.calc.forces_var` | `forces_var` — committee | `mace/calculators/mace.py:718` | KEEP — committee output (force variance). The committee is the part of `mace_active_learning_md` that MACE must keep guaranteeing (§1) | ⚠️ gap (idem) | DEP-1 | RET-6 | todo |
| `out.calc.stress_comm` | `stress_comm` — committee | `mace/calculators/mace.py:718` | KEEP — committee output (per-model stresses). The committee is the part of `mace_active_learning_md` that MACE must keep guaranteeing (§1) | ⚠️ gap (idem) | DEP-1 | RET-6 | todo |
| `out.calc.stress_var` | `stress_var` — committee | `mace/calculators/mace.py:718` | KEEP — committee output (stress variance). The committee is the part of `mace_active_learning_md` that MACE must keep guaranteeing (§1) | ⚠️ gap (idem) | DEP-1 | RET-6 | todo |
| `out.calc.dipole_comm` | `dipole_comm` — committee | `mace/calculators/mace.py:718` | KEEP — committee output (per-model dipoles). The committee is the part of `mace_active_learning_md` that MACE must keep guaranteeing (§1) | ⚠️ gap (idem) | DEP-1a | RET-6 | todo |
| `out.calc.dipole_var` | `dipole_var` — committee | `mace/calculators/mace.py:718` | KEEP — committee output (dipole variance). The committee is the part of `mace_active_learning_md` that MACE must keep guaranteeing (§1) | ⚠️ gap (idem) | DEP-1a | RET-6 | todo |

### 12.2 Keys `mace_eval_configs` writes (13)

Every one is written as `--info_prefix` + the name below, into `atoms.info` or `atoms.arrays`,
and then serialized into the output XYZ — so these names end up in users' files on disk and in
their downstream scripts. The default prefix is `MACE_`.

| id | feature | source | disposition | pinned by | destination | retirement | status |
|---|---|---|---|---|---|---|---|
| `out.eval.energy` | `<info_prefix>energy` → `atoms.info` | `mace/cli/eval_configs.py:397` | KEEP | P0-5 | CLI-1 | RET-6 | todo |
| `out.eval.forces` | `<info_prefix>forces` → `atoms.arrays` | `mace/cli/eval_configs.py:398` | KEEP | P0-5 | CLI-1 | RET-6 | todo |
| `out.eval.stress` | `<info_prefix>stress` → `atoms.info` | `mace/cli/eval_configs.py:404` | KEEP | P0-5 | CLI-1 | RET-6 | todo |
| `out.eval.node_energies` | `<info_prefix>node_energies` → `atoms.arrays` | `mace/cli/eval_configs.py:446` | KEEP — note the plural, which matches neither the model's `node_energy` nor the calculator's `energies`; v1 writes one spelling | ⚠️ gap (add assert to P0-5) | CLI-1, CORE-3 | RET-6 | todo |
| `out.eval.descriptors` | `<info_prefix>descriptors` → `atoms.info` | `mace/cli/eval_configs.py:440` | KEEP — written to `info` for a single aggregated vector and to `arrays` per atom | ⚠️ gap (descriptor contract → P0-5) | ARCH-2, CLI-1 | RET-6 | todo |
| `out.eval.BO_contributions` | `<info_prefix>BO_contributions` → `atoms.info` | `mace/cli/eval_configs.py:426` | KEEP — the per-layer energy contributions, under a third spelling of the same quantity (`contributions` in the model, `--return_contributions` on the CLI) | ⚠️ gap (add assert to P0-5) | CLI-1, CORE-1 | RET-6 | todo |
| `out.eval.magforces` | `<info_prefix>magforces` → `atoms.arrays` | `mace/cli/eval_configs.py:401` | KEEP | `tests/extensions/magnetic` | MAG-1, CLI-1 | RET-6 | todo |
| `out.eval.BEC` | `<info_prefix>BEC` → `atoms.arrays` | `mace/cli/eval_configs.py:407` | KEEP | ⚠️ gap (add to P0-3c) | ELEC-4, CLI-1 | RET-6 | todo |
| `out.eval.latent_charges` | `<info_prefix>latent_charges` → `atoms.arrays` | `mace/cli/eval_configs.py:411` | KEEP | ⚠️ gap (add to P0-3c) | ELEC-4, CLI-1 | RET-6 | todo |
| `out.eval.latent_dipoles` | `<info_prefix>latent_dipoles` → `atoms.arrays` | `mace/cli/eval_configs.py:413` | KEEP | ⚠️ gap (idem) | ELEC-4, CLI-1 | RET-6 | todo |
| `out.eval.latent_kappas` | `<info_prefix>latent_kappas` → `atoms.arrays` | `mace/cli/eval_configs.py:415` | KEEP | ⚠️ gap (idem) | ELEC-4, CLI-1 | RET-6 | todo |
| `out.eval.latent_alphas` | `<info_prefix>latent_alphas` → `atoms.arrays` | `mace/cli/eval_configs.py:417` | KEEP | ⚠️ gap (idem) | ELEC-4, CLI-1 | RET-6 | todo |
| `out.eval.latent_quads` | `<info_prefix>latent_quads` → `atoms.arrays` | `mace/cli/eval_configs.py:421` | KEEP | ⚠️ gap (idem) | ELEC-4, CLI-1 | RET-6 | todo |

## 13. Behaviour-affecting environment variables (9 + 3)

Every `MACE_*` literal in the package. They are gated as a set because an environment variable is
the one configuration channel that leaves no trace in the run metadata.

| id | feature | source | disposition | pinned by | destination | retirement | status |
|---|---|---|---|---|---|---|---|
| `env.MACE_TIME` | `MACE_TIME` | `mace/calculators/lammps_mliap_mace.py:25` | KEEP — MLIAP runtime config; behaviour preserved even if DEP-2 moves it into the deploy manifest (per-step timing) | ⚠️ gap (env-var behaviour untested) | DEP-2 | RET-5 | todo |
| `env.MACE_PROFILE` | `MACE_PROFILE` | `mace/calculators/lammps_mliap_mace.py:26` | KEEP — MLIAP runtime config; behaviour preserved even if DEP-2 moves it into the deploy manifest (torch profiler) | ⚠️ gap (idem) | DEP-2 | RET-5 | todo |
| `env.MACE_PROFILE_START` | `MACE_PROFILE_START` | `mace/calculators/lammps_mliap_mace.py:27` | KEEP — MLIAP runtime config; behaviour preserved even if DEP-2 moves it into the deploy manifest (first profiled step) | ⚠️ gap (idem) | DEP-2 | RET-5 | todo |
| `env.MACE_PROFILE_END` | `MACE_PROFILE_END` | `mace/calculators/lammps_mliap_mace.py:28` | KEEP — MLIAP runtime config; behaviour preserved even if DEP-2 moves it into the deploy manifest (last profiled step) | ⚠️ gap (idem) | DEP-2 | RET-5 | todo |
| `env.MACE_ALLOW_CPU` | `MACE_ALLOW_CPU` | `mace/calculators/lammps_mliap_mace.py:29` | KEEP — MLIAP runtime config; behaviour preserved even if DEP-2 moves it into the deploy manifest (tolerate CPU tensors) | ⚠️ gap (idem) | DEP-2 | RET-5 | todo |
| `env.MACE_FORCE_CPU` | `MACE_FORCE_CPU` | `mace/calculators/lammps_mliap_mace.py:30` | KEEP — MLIAP runtime config; behaviour preserved even if DEP-2 moves it into the deploy manifest (force CPU execution) | ⚠️ gap (idem) | DEP-2 | RET-5 | todo |
| `env.MACE_ASE_PAD_NUM_ATOMS` | `MACE_ASE_PAD_NUM_ATOMS` | `mace/calculators/mace.py:411` | KEEP — the calculator's padding override, as an explicit config field in v1 | ⚠️ gap (padding contract → P0-5) | BKD-2, DEP-1 | RET-6 | todo |
| `env.MACE_ASE_PAD_NUM_EDGES` | `MACE_ASE_PAD_NUM_EDGES` | `mace/calculators/mace.py:413` | KEEP — idem | ⚠️ gap (idem) | BKD-2, DEP-1 | RET-6 | todo |
| `env.MACE_USE_CUEQ_CG` | `MACE_USE_CUEQ_CG` | `mace/tools/cg.py:23` | DROP — the variable goes, not the capability: an environment variable that silently changes model numerics is unreproducible and never lands in the run metadata; it is what makes machine-to-machine differences unexplainable. The CG source becomes a backend decision recorded in the resolved config | ⚠️ gap (CORE-4 pins the two CG sources against each other) | BKD-1, CORE-4 | RET-6 | todo |

Three further variables are read but are not MACE's own namespace, so they are not in the gated
set:

| id | feature | source | disposition | pinned by | destination | retirement | status |
|---|---|---|---|---|---|---|---|
| `stdenv.XDG_CACHE_HOME` | `XDG_CACHE_HOME` — foundation-model cache location | `mace/calculators/foundations_models.py` | KEEP — the standard cache convention | `tests/unit/test_download_urls.py` | FM-4 | RET-6 | todo |
| `stdenv.TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD` | `TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD` — set by `mace/__init__.py` so pickled checkpoints load under newer torch defaults | `mace/__init__.py` | DROP — v1 checkpoints are neutral-format (safetensors + manifest), so nothing needs the unsafe-pickle escape hatch; the legacy loader keeps it until the converter is the only reader | P0-5 (converter contract) | FM-1, TRN-4 | RET-6 | todo |
| `stdenv.MASTER_PORT` | `MASTER_ADDR` / `MASTER_PORT` / `RANK` / `WORLD_SIZE` / `LOCAL_RANK` / `SLURM_*` / `OMPI_*` — standard DDP and launcher plumbing | `mace/tools/slurm_distributed.py`, `mace/tools/distributed_tools.py` | KEEP — the launcher contract is torch's and SLURM's, not MACE's, so v1 reads the same variables | `tests/workflows/test_distributed.py` | TRN-4 | RET-6 | todo |

## 14. Registered pytest markers (13)

The capability model of the suite: locally a test whose capability is missing skips, and in CI a
job that exports `MACE_REQUIRE_CAPS` fails instead. INF-5 generates its capabilities manifest from
this list, which is why the one marker that is **not** a capability has to be named here rather
than left to be inferred.

| id | feature | source | disposition | pinned by | destination | retirement | status |
|---|---|---|---|---|---|---|---|
| `marker.gpu` | `@pytest.mark.gpu` | `pyproject.toml:23` | KEEP — a capability probe in `tests/conftest.py`; INF-5 carries it into the capabilities manifest (any vendor: CUDA or ROCm) | `tests/conftest.py` | INF-5, P0-8 | n/a — test infrastructure, not legacy code | todo |
| `marker.cueq` | `@pytest.mark.cueq` | `pyproject.toml:25` | KEEP — a capability probe in `tests/conftest.py`; INF-5 carries it into the capabilities manifest | `tests/conftest.py` | INF-5, P0-4 | n/a — test infrastructure, not legacy code | todo |
| `marker.oeq` | `@pytest.mark.oeq` | `pyproject.toml:26` | KEEP — a capability probe in `tests/conftest.py`; INF-5 carries it into the capabilities manifest | `tests/conftest.py` | INF-5, P0-4 | n/a — test infrastructure, not legacy code | todo |
| `marker.polar` | `@pytest.mark.polar` | `pyproject.toml:27` | KEEP — a capability probe in `tests/conftest.py`; INF-5 carries it into the capabilities manifest | `tests/conftest.py` | INF-5, P0-3a | n/a — test infrastructure, not legacy code | todo |
| `marker.les` | `@pytest.mark.les` | `pyproject.toml:28` | KEEP — a capability probe in `tests/conftest.py`; INF-5 carries it into the capabilities manifest | `tests/conftest.py` | INF-5, P0-3c | n/a — test infrastructure, not legacy code | todo |
| `marker.magnetic` | `@pytest.mark.magnetic` | `pyproject.toml:29` | KEEP — a capability probe in `tests/conftest.py`; INF-5 carries it into the capabilities manifest | `tests/conftest.py` | INF-5, P0-3b | n/a — test infrastructure, not legacy code | todo |
| `marker.torchsim` | `@pytest.mark.torchsim` | `pyproject.toml:30` | KEEP — a capability probe in `tests/conftest.py`; INF-5 carries it into the capabilities manifest | `tests/conftest.py` | INF-5, DEP-4 | n/a — test infrastructure, not legacy code | todo |
| `marker.schedulefree` | `@pytest.mark.schedulefree` | `pyproject.toml:31` | KEEP — a capability probe in `tests/conftest.py`; INF-5 carries it into the capabilities manifest | `tests/conftest.py` | INF-5, TRN-2 | n/a — test infrastructure, not legacy code | todo |
| `marker.bin_lammps` | `@pytest.mark.bin_lammps` | `pyproject.toml:33` | KEEP — a capability probe in `tests/conftest.py`; INF-5 carries it into the capabilities manifest (an external binary rather than an import) | `tests/conftest.py` | INF-5, DEP-2 | n/a — test infrastructure, not legacy code | todo |
| `marker.network` | `@pytest.mark.network` | `pyproject.toml:24` | KEEP — a capability probe in `tests/conftest.py`; INF-5 carries it into the capabilities manifest; never autodetected, opt-in via `MACE_CI_ALLOW_NETWORK=1` | `tests/conftest.py` | INF-5, P0-2 | n/a — test infrastructure, not legacy code | todo |
| `marker.slow` | `@pytest.mark.slow` | `pyproject.toml:22` | KEEP — a cost marker, not a capability: applied by directory to `tests/workflows` | `tests/conftest.py` | INF-5, P0-8 | n/a — test infrastructure, not legacy code | todo |
| `marker.benchmark` | `@pytest.mark.benchmark` | `pyproject.toml:32` | KEEP — a cost marker: performance measurement, never part of a correctness gate | `tests/conftest.py` | INF-5, P0-8 | n/a — test infrastructure, not legacy code | todo |
| `marker.timeout` | `@pytest.mark.timeout` | `pyproject.toml:34` | KEEP — test infrastructure, and explicitly **not** a capability: it is registered only so collection works when `pytest-timeout` is absent (the plugin ships in the `test`/`dev` extras). It has no `CAPABILITY_PROBES` entry and must not be absorbed into INF-5's manifest. Three tests use it today (`tests/workflows/test_finetuning_pseudolabels.py:97,133,169`) | `tests/workflows/test_finetuning_pseudolabels.py` | INF-5 | n/a — test infrastructure, not legacy code | todo |

## 15. Default property keys — the on-disk data contract (13)

`DefaultKeys` (`mace/tools/default_keys.py`) is the name every labelled XYZ in the wild uses.
Silently changing one breaks every existing dataset at once, and the set **grew by two in a single
release** (`REF_magmom`, `REF_magforces`), which is the argument for freezing it explicitly rather
than letting it accrete. All thirteen KEEP their spelling; the mechanism that resolves them moves
to the CORE-3 property-key convention, and any rename REL-1 documents with an explicit old→new
map.

| id | feature | source | disposition | pinned by | destination | retirement | status |
|---|---|---|---|---|---|---|---|
| `key.ENERGY` | `ENERGY` = `"REF_energy"` | `mace/tools/default_keys.py:7` | KEEP — the default name is part of the data contract; a rename needs an explicit REL-1 mapping | P0-7 (key-variant parsing; the default names explicitly) | CORE-3 | RET-6 | todo |
| `key.FORCES` | `FORCES` = `"REF_forces"` | `mace/tools/default_keys.py:8` | KEEP — the default name is part of the data contract; a rename needs an explicit REL-1 mapping | P0-7 (key-variant parsing; the default names explicitly) | CORE-3 | RET-6 | todo |
| `key.STRESS` | `STRESS` = `"REF_stress"` | `mace/tools/default_keys.py:9` | KEEP — the default name is part of the data contract; a rename needs an explicit REL-1 mapping | P0-7 (key-variant parsing; the default names explicitly) | CORE-3 | RET-6 | todo |
| `key.VIRIALS` | `VIRIALS` = `"REF_virials"` | `mace/tools/default_keys.py:10` | KEEP — the default name is part of the data contract; a rename needs an explicit REL-1 mapping | P0-7 (key-variant parsing; the default names explicitly) | CORE-3 | RET-6 | todo |
| `key.DIPOLE` | `DIPOLE` = `"dipole"` | `mace/tools/default_keys.py:11` | KEEP — the default name is part of the data contract; a rename needs an explicit REL-1 mapping | P0-7 (key-variant parsing; the default names explicitly) | CORE-3 | RET-6 | todo |
| `key.POLARIZABILITY` | `POLARIZABILITY` = `"polarizability"` | `mace/tools/default_keys.py:12` | KEEP — the default name is part of the data contract; a rename needs an explicit REL-1 mapping | P0-7 (key-variant parsing; the default names explicitly) | CORE-3 | RET-6 | todo |
| `key.HEAD` | `HEAD` = `"head"` | `mace/tools/default_keys.py:13` | KEEP — the default name is part of the data contract; a rename needs an explicit REL-1 mapping | P0-7 (key-variant parsing; the default names explicitly) | CORE-3 | RET-6 | todo |
| `key.CHARGES` | `CHARGES` = `"REF_charges"` | `mace/tools/default_keys.py:14` | KEEP — the default name is part of the data contract; a rename needs an explicit REL-1 mapping | P0-7 (key-variant parsing; the default names explicitly) | CORE-3 | RET-6 | todo |
| `key.TOTAL_CHARGE` | `TOTAL_CHARGE` = `"total_charge"` | `mace/tools/default_keys.py:15` | KEEP — the default name is part of the data contract; a rename needs an explicit REL-1 mapping | P0-7 (key-variant parsing; the default names explicitly) | CORE-3 | RET-6 | todo |
| `key.TOTAL_SPIN` | `TOTAL_SPIN` = `"total_spin"` | `mace/tools/default_keys.py:16` | KEEP — the default name is part of the data contract; a rename needs an explicit REL-1 mapping | P0-7 (key-variant parsing; the default names explicitly) | CORE-3 | RET-6 | todo |
| `key.ELEC_TEMP` | `ELEC_TEMP` = `"elec_temp"` | `mace/tools/default_keys.py:17` | KEEP — the default name is part of the data contract; a rename needs an explicit REL-1 mapping | P0-7 (key-variant parsing; the default names explicitly) | CORE-3 | RET-6 | todo |
| `key.MAGMOM` | `MAGMOM` = `"REF_magmom"` | `mace/tools/default_keys.py:18` | KEEP — the default name is part of the data contract; a rename needs an explicit REL-1 mapping | P0-7 (key-variant parsing; the default names explicitly) | CORE-3 | RET-6 | todo |
| `key.MAGFORCES` | `MAGFORCES` = `"REF_magforces"` | `mace/tools/default_keys.py:19` | KEEP — the default name is part of the data contract; a rename needs an explicit REL-1 mapping | P0-7 (key-variant parsing; the default names explicitly) | CORE-3 | RET-6 | todo |

## 16. Calculator methods, loader keyword arguments and published model names

Not machine-gated (a method set is not a membership set the way a parser's dests are), but inventoried
under the same schema. The loader kwargs are the ones hidden behind `@overload`s in
`foundations_models.py`, which is why an earlier pass missed them.

| id | feature | source | disposition | pinned by | destination | retirement | status |
|---|---|---|---|---|---|---|---|
| `method.calculate` | `MACECalculator.calculate` — E/F/stress, plus committee mean/variance when several models are loaded | `mace/calculators/mace.py` | KEEP | P0-5; committee ⚠️ gap (§12.1) | DEP-1 | RET-6 | todo |
| `method.check_state` | `MACECalculator.check_state` — ASE's recalculation trigger | `mace/calculators/mace.py` | KEEP — part of the ASE contract | P0-5 | DEP-1 | RET-6 | todo |
| `method.get_hessian` | `MACECalculator.get_hessian` — analytical Hessians, including the polar variant | `mace/calculators/mace.py` | KEEP | `tests/foundations/test_hessian.py` (port cases) | DEP-1 | RET-6 | todo |
| `method.get_descriptors` | `MACECalculator.get_descriptors` — per-layer, invariants-only and aggregated node features | `mace/calculators/mace.py` | KEEP — descriptors are `BaseMACE`'s features, so this becomes a first-class API rather than a calculator extra | ⚠️ gap (descriptor contract → P0-5) | ARCH-2, DEP-1 | RET-6 | todo |
| `method.get_dielectric_derivatives` | `MACECalculator.get_dielectric_derivatives` — `dmu/dr`, `dalpha/dr` | `mace/calculators/mace.py` | KEEP | ⚠️ gap (add to the P0-3a MDP case) | ELEC-2, FT-4, DEP-1a | RET-6 | todo |
| `kwarg.model` | `model=` — which published artifact/size a loader fetches; shared by every loader | `mace/calculators/foundations_models.py` | KEEP | P0-2 | FM-3, FM-4 | RET-6 | todo |
| `kwarg.device` | `device=` — shared by every loader | `mace/calculators/foundations_models.py` | KEEP | P0-2 | FM-3 | RET-6 | todo |
| `kwarg.default_dtype` | `default_dtype=` — shared by every loader | `mace/calculators/foundations_models.py` | MERGE — `PrecisionConfig` | P0-2 | BKD-2, FM-3 | RET-6 | todo |
| `kwarg.return_raw_model` | `return_raw_model=` — hand back the `nn.Module` instead of a calculator; shared by every loader | `mace/calculators/foundations_models.py` | KEEP — the library-use path, distinct from the ASE path | ⚠️ gap (add to P0-2) | FM-3 | RET-6 | todo |
| `kwarg.model_path` | `mace_anicc(model_path=…)` — the one loader whose first argument is spelled differently from every other | `mace/calculators/foundations_models.py` | DROP — goes with `mace_anicc` itself (§9.2); the signature exception is part of why | — | REL-1 (doc) | RET-6 | todo |
| `kwarg.dispersion` | `mace_mp(dispersion=…)` — D3 dispersion correction via torch-dftd | `mace/calculators/foundations_models.py` | KEEP | ⚠️ gap (no dispersion test anywhere) | FM-3, DEP-1 | RET-6 | todo |
| `kwarg.damping` | `mace_mp(damping=…)` — D3 damping function | `mace/calculators/foundations_models.py` | KEEP | ⚠️ gap (idem) | FM-3, DEP-1 | RET-6 | todo |
| `kwarg.dispersion_xc` | `mace_mp(dispersion_xc=…)` — the functional the D3 parameters are taken from | `mace/calculators/foundations_models.py` | KEEP | ⚠️ gap (idem) | FM-3, DEP-1 | RET-6 | todo |
| `kwarg.dispersion_cutoff` | `mace_mp(dispersion_cutoff=…)` | `mace/calculators/foundations_models.py` | KEEP | ⚠️ gap (idem) | FM-3, DEP-1 | RET-6 | todo |
| `alias.published_names` | The per-loader shortcut names users write in code and docs — `small`/`medium`/`large`, `small-0b`/`medium-0b`/`*-0b2`/`medium-0b3`, `medium-mpa-0`, `small-omat-0`/`medium-omat-0`, plus the OFF/polar/MDP/OMOL sets. This **is** the current model registry | `mace/calculators/foundations_models.py` | KEEP — carried into the FM-3 registry, possibly renamed under the new naming scheme with a deprecation mapping per old alias | `tests/unit/test_download_urls.py` | FM-2, FM-3 | RET-6 | todo |
| `pkg.torchsim_backend` | `mace/calculators/mace_torchsim.py` — the torch-sim backend, including PolarMACE support | `mace/calculators/mace_torchsim.py` | KEEP — a first-class deployment path alongside ASE and LAMMPS | `tests/extensions/torchsim` | DEP-4 | RET-6 | todo |

## 17. LAMMPS runtime surface

| id | feature | source | disposition | pinned by | destination | retirement | status |
|---|---|---|---|---|---|---|---|
| `lammps.mliap` | `LAMMPS_MLIAP_MACE` — `compute_forces`, `compute_descriptors`, `compute_gradients`, ghost-atom handling | `mace/calculators/lammps_mliap_mace.py` | KEEP — the primary v1 deployment path | `tests/integrations/lammps` (contract tier + the `bin_lammps` real tier) | DEP-2 | RET-5 | todo |
| `lammps.ghost_branches` | `lammps_class` / `lammps_natoms` branches in the interaction blocks — the real-vs-ghost atom partitioning | `mace/modules/blocks.py` | KEEP — must be designed into ARCH-2, not bolted on afterwards | `tests/integrations/lammps/test_ghost_parity.py` | ARCH-2, DEP-2 | RET-1 | todo |
| `lammps.ghost_exchange_check` | `_check_ghost_exchange_support` — refuses a multi-layer model up front when the LAMMPS build cannot exchange ghost node features | `mace/calculators/lammps_mliap_mace.py` | KEEP — the precondition is ported into the v1 unified runtime; without it a multi-layer model dies on a bare `AttributeError` inside layer two | `tests/integrations/lammps` | DEP-2 | RET-5 | todo |
| `lammps.torchscript_wrapper` | `LAMMPS_MACE` (`@compile_mode("script")`) + the `-lammps.pt` artifact `mace_create_lammps_model` writes by default | `mace/calculators/lammps_mace.py` | DROP — v1 blocks are born without `@compile_mode`, so scripting can never apply to them; the MLIAP bundle is the one supported artifact | P0-5 (export golden pins the numerics of the replacement) | DEP-2 | RET-5 | todo |
| `lammps.compiled_side_artifact` | The `_compiled.model` / `_stagetwo_compiled.model` TorchScript artifacts training emits next to every checkpoint, each inside a bare `except Exception: pass` | `mace/cli/run_train.py` | DROP — a deliberate, recorded removal: v1 checkpoints are neutral-format only and deployment artifacts come solely from `mace export` | P0-5 (checkpoint contract) | TRN-4, DEP-2 | RET-5 | todo |

## 18. Documentation surface (mace-docs page index)

Each published page is user-promised functionality: v1 docs (REL-3) must cover every `KEEP` row it
maps to, and the CI-tested-tutorials rule (GOV-1) applies. These rows track the *promise*, not a code
surface, so their retirement is the docs consolidation rather than a `RET-*`.

| id | feature | source | disposition | pinned by | destination | retirement | status |
|---|---|---|---|---|---|---|---|
| `doc.quickstart` | Quick Start / Introduction / Installation / Troubleshooting | mace-docs | KEEP — maps to §1 and §10 | ⚠️ gap (docs are not CI-tested today) | REL-3, GOV-1 | n/a — documentation, superseded by REL-3 | todo |
| `doc.training` | Training | mace-docs | KEEP — maps to §3 | ⚠️ gap (idem) | REL-3 | n/a — documentation, superseded by REL-3 | todo |
| `doc.evaluation` | Evaluation | mace-docs | KEEP — maps to §5 eval | ⚠️ gap (idem) | REL-3 | n/a — documentation, superseded by REL-3 | todo |
| `doc.multihead` | Heterogeneous Data Training / Multihead Training | mace-docs | KEEP — maps to §3.6 | ⚠️ gap (idem) | REL-3, FT-2 | n/a — documentation, superseded by REL-3 | todo |
| `doc.ase` | ASE calculator | mace-docs | KEEP — maps to §9 and §16 | ⚠️ gap (idem) | REL-3 | n/a — documentation, superseded by REL-3 | todo |
| `doc.descriptors` | MACE descriptors | mace-docs | KEEP — maps to `get_descriptors` (§16) and the eval descriptor flags (§5) | ⚠️ gap (idem) | REL-3 | n/a — documentation, superseded by REL-3 | todo |
| `doc.hessians` | Analytical Hessians | mace-docs | KEEP — maps to `get_hessian` (§16) | `tests/foundations/test_hessian.py` | REL-3 | n/a — documentation, superseded by REL-3 | todo |
| `doc.dipoles` | Dipole Moments and Polarizabilities | mace-docs | KEEP — maps to §6 and §11 | ⚠️ gap (idem) | REL-3 | n/a — documentation, superseded by REL-3 | todo |
| `doc.cuda` | CUDA Acceleration (cuEquivariance) | mace-docs | KEEP — rewritten: v1 dispatches instead of converting, so the page's central instruction (convert your model) disappears | P0-4 | REL-3 | n/a — documentation, superseded by REL-3 | todo |
| `doc.openmm` | OpenMM Interface | mace-docs | KEEP | ⚠️ gap (no OpenMM coverage in-tree) | DEP-3, REL-3 | n/a — documentation, superseded by REL-3 | todo |
| `doc.lammps` | MACE in LAMMPS / MACE in LAMMPS with ML-IAP | mace-docs | KEEP — reduced to the MLIAP path; maps to §17 | `tests/integrations/lammps` | DEP-2, REL-3 | n/a — documentation, superseded by REL-3 | todo |
| `doc.foundation_models` | Foundation models | mace-docs | KEEP — maps to §9.2 | P0-2 | FM-3, REL-3 | n/a — documentation, superseded by REL-3 | todo |
| `doc.electrostatics` | Electrostatic MACE | mace-docs | KEEP — maps to §3.3 and PolarMACE (§6) | P0-3a | ELEC-2, REL-3 | n/a — documentation, superseded by REL-3 | todo |
| `doc.finetuning` | Fine-tuning / Multihead Replay / LoRA | mace-docs | KEEP — maps to §3.6 | P0-5 | FT-2, REL-3 | n/a — documentation, superseded by REL-3 | todo |
| `doc.preprocessing` | Large Dataset Pre-processing | mace-docs | KEEP — maps to §4 | `tests/workflows/test_preprocess.py` | DATA-2, REL-3 | n/a — documentation, superseded by REL-3 | todo |
| `doc.multigpu` | Multi-GPUs Training | mace-docs | KEEP — maps to `--distributed` (§3.1) | `tests/workflows/test_distributed.py` | TRN-4, REL-3 | n/a — documentation, superseded by REL-3 | todo |
| `doc.examples` | Examples (MD22, ANI-1x, liquid water; NVT with a foundation model) and Tutorials 1–3 | mace-docs | KEEP — end-to-end usage; the theory tutorial is superseded by the marimo notebooks | ⚠️ gap (examples are not CI-tested) | REL-3, EDU-2 | n/a — documentation, superseded by REL-3 | todo |

## 19. Package-level surfaces and second-pass findings

Things that are neither a flag nor a class nor a key, and that an enumeration of the obvious surfaces
misses.

| id | feature | source | disposition | pinned by | destination | retirement | status |
|---|---|---|---|---|---|---|---|
| `pkg.fairchem_lmdb` | FairChem LMDB dataset tools — reading FairChem-format LMDB shards | `mace/tools/fairchem_dataset/lmdb_dataset_tools.py` | KEEP — becomes a DATA-1 backend: OMat24 and OMol25 ship in this format and MACE has models trained on both, so dropping it would mean re-converting terabyte-scale datasets | `tests/unit/test_lmdb_database.py` | DATA-2 | RET-2 | todo |
| `pkg.hdf5_dataset` | HDF5 shard reader/writer for on-line loading | `mace/data/hdf5_dataset.py` | KEEP — HDF5 v2 (read + write) plus a legacy-HDF5 read path | `tests/workflows/test_preprocess.py` | DATA-2 | RET-2 | todo |
| `pkg.lmdb_dataset` | LMDB shard reader, honouring the CLI key specification | `mace/data/lmdb_dataset.py` | KEEP — read support | `tests/unit/test_lmdb_database.py` | DATA-2 | RET-2 | todo |
| `pkg.neighborhood` | Neighbour-list construction via matscipy, including the non-periodic cell rework | `mace/data/neighborhood.py` | KEEP — becomes a pluggable neighbour-list backend | P0-7 | DATA-3 | RET-2 | todo |
| `pkg.vendored_torch_geometric` | The vendored `torch_geometric` copy (`Data`, `Batch`, `DataLoader`, `scatter`) | `mace/tools/torch_geometric/` | DROP — v1 collates without torch_geometric; the vendored copy is excluded from lint and mypy today, which is the clearest sign it is not maintained code. Complication: `mace/data/augmentation.py` imports the *real* package while the rest of the tree imports the vendored one, and the `[magnetic]` extra declares external `torch-geometric` — so both must go at once | P0-7 (batching semantics) | DATA-3 | RET-2 | todo |
| `pkg.compile_utils` | `prepare` / `simplify_if_compile` — utilities that make legacy modules `torch.compile`-able | `mace/tools/compile.py` | MERGE — v1 is compile-first, so the retrofit mechanism has nothing to retrofit | `tests/unit/test_compile.py` (compiled == eager) | BKD-2 | RET-6 | todo |
| `pkg.visualise_train` | `mace/cli/visualise_train.py` — the plotting support module behind `mace_plot_train`, with no entry point of its own | `mace/cli/visualise_train.py` | KEEP — follows its CLI (§1) | ⚠️ gap (with the plot smoke) | CLI-1 | RET-6 | todo |
| `pkg.public_import_surface` | The public Python API downstream projects import — `mace.modules.MACE`, `mace.data.AtomicData`, `mace.cli.run_train.run(args)`, … | `mace/` | DROP — a deliberate break: v1 defines a new public API, and REL-1 documents the old→new equivalences rather than aliasing them | — (deliberate break) | REL-1 | RET-6 | todo |
| `pkg.anicc_checkpoint` | The tracked MACE-ANI-CC checkpoint shipped inside the wheel | `mace/calculators/foundations_models/ani500k_large_CC.model` | DROP — goes with `mace_anicc` (§9.2); v1 fetches every artifact through the FM-3 registry rather than bundling one | — | REL-1 (doc), FM-3 | RET-6 | todo |
| `pkg.statistics_json` | `statistics.json` — the preprocessing side-car carrying E0s, avg neighbours, mean/std and the atomic numbers | `mace/cli/preprocess_data.py` | KEEP — becomes part of the dataset metadata contract | `tests/workflows/test_preprocess.py` | DATA-1 | RET-2 | todo |
| `pkg.results_log_format` | The per-epoch results log (`results/*.txt`), which `mace_plot_train` parses and users script against | `mace/tools/train.py` | KEEP — becomes a structured (typed) log; the current line format is not a promise v1 makes | ⚠️ gap (resolved together with the plot smoke) | TRN-3 | RET-6 | todo |
| `pkg.heads_yaml_schema` | The `--heads` YAML sub-schema (per-head files, weights, E0s and key overrides) | `mace/tools/multihead_tools.py` | KEEP — becomes a typed section of the config schema | P0-5 (multihead case) | CFG-1, FT-1 | RET-6 | todo |
| `pkg.e0_estimation` | The three E0 resolution modes — explicit dict, `average` (least squares) and `estimated` (foundation-corrected) | `mace/tools/scripts_utils.py` | KEEP — E0 specification is a config section with an explicit, tested resolution order | P0-6 | ARCH-3, CFG-1 | RET-6 | todo |
| `pkg.lr_param_groups` | Explicit optimizer parameter groups, driven by `--lr_params_factors` and reused by `--freeze` | `mace/tools/scripts_utils.py` | MERGE — typed per-param-group fields of the per-stage optimizer config | ⚠️ gap (add to P0-5 with `--freeze`) | TRN-2 | RET-6 | todo |
| `pkg.augmentation` | `Random3DRotation` — the training-data augmentation transform behind `--data_aug_magmom` | `mace/data/augmentation.py` | MERGE — a registered training-data transform, not a model flag | `tests/extensions/magnetic` (rotation equivariance) | TRN-2 | RET-2 | todo |
| `pkg.per_head_reporting` | Per-head validation logging, per-head test error tables and per-head parity plots with labels | `mace/tools/train.py`, `mace/tools/tables_utils.py` | KEEP — multihead is the normal case now, so per-head reporting is not an add-on | ⚠️ gap (multihead reporting → P0-5) | TRN-3 | RET-6 | todo |
| `pkg.atomic_download` | Race-free foundation-model downloads: fetch to a temporary file and rename, so a failed or concurrent download cannot leave a truncated checkpoint in the cache | `mace/calculators/foundations_models.py` | KEEP — the property, not the implementation: a partial download must never be readable as a model | `tests/unit/test_download_urls.py` | FM-4 | RET-6 | todo |
| `pkg.first_block_allowlist` | The three first-interaction blocks a `MACE`-type model accepts — plain, Density and, since recently, the **non-linear** residual one | `mace/tools/model_script_utils.py:280-284` | KEEP — a non-linear first block is a valid architecture and the v1 assembly must not re-impose the old restriction | ⚠️ gap (add case to P0-1/P0-7) | ARCH-2, CFG-1 | RET-1 | todo |
| `pkg.first_block_coercion` | For `--model MACE`, anything outside that allowlist is **silently rewritten** to `RealAgnosticInteractionBlock` | `mace/tools/model_script_utils.py:285` | DROP — a config value the tool overwrites without a word is worse than a rejected one: the run trains a different architecture than the user asked for and nothing says so. v1 fails the combination in config validation | ⚠️ gap (add case to P0-1/P0-7) | CFG-1 | RET-1 | todo |

## 20. External `mace-scf` model families (out-of-tree)

Five charge-aware families built on MACE v0.3.14 + `graph_longrange`, in `ACEsuit/mace-scf`. They are
**architectures, not checkpoints**, so they are reimplemented on the v1 two-layer + model-transform
hook and pinned by parity — not converted. They enter at the lowest support tier, after the core
refactor. Two consequences: the TRN-2 model-transform hook is designed against **three** different SCF
schemes rather than one (validate it with MACE-QEq, the most established), and all five need a v1
golden, which is five new parity artifacts on top of the foundation-model set. Whichever solver they
use inherits the electrostatics solver-dispatch decision, and where an accelerated solver is not
bit-parity the solver identity becomes model state serialized with the checkpoint.

| id | feature | source | disposition | pinned by | destination | retirement | status |
|---|---|---|---|---|---|---|---|
| `scf.lc_mace` | LC-MACE — local charge: multipole moments read out from descriptors; non-SCF | `ACEsuit/mace-scf` | KEEP — reimplement; the cheapest of the five, no implicit-diff hook needed | ⚠️ gap (no v1 golden yet) | ELEC-3 | n/a — out-of-tree, nothing in `mace/` to delete | todo |
| `scf.lsc_mace` | LSC-MACE — local split charge: charge + multipoles conserving local charge flow; non-SCF | `ACEsuit/mace-scf` | KEEP — reimplement | ⚠️ gap (no v1 golden yet) | ELEC-3 | n/a — out-of-tree, nothing in `mace/` to delete | todo |
| `scf.qeq` | MACE-QEq — charge equilibration, solved self-consistently with implicit differentiation | `ACEsuit/mace-scf` | KEEP — reimplement; the load-bearing test case for the TRN-2 SCF hook | ⚠️ gap (no v1 golden yet) | ELEC-3, TRN-2 | n/a — out-of-tree, nothing in `mace/` to delete | todo |
| `scf.fixedpoint` | FixedPointSCF — Kohn–Sham-like SCF cycles for multipole moments | `ACEsuit/mace-scf` | KEEP — reimplement; incremental once the hook is validated by QEq | ⚠️ gap (no v1 golden yet) | ELEC-3, TRN-2 | n/a — out-of-tree, nothing in `mace/` to delete | todo |
| `scf.energy_functional` | EnergyFunctionalSCF — an alternative functional; upstream marks it experimental | `ACEsuit/mace-scf` | KEEP — reimplement; its golden pins whatever behaviour it has today, experimental or not | ⚠️ gap (no v1 golden yet) | ELEC-3 | n/a — out-of-tree, nothing in `mace/` to delete | todo |

## 21. Foundation-model migration roster

The published, **trained** artifacts to migrate, in one auditable place. Conversion is one-shot
(legacy pickle → neutral artifact → v1), never a runtime load path; multi-head artifacts convert with
their heads intact, and a single-head export is `mace model select-head`. Model *architectures* are
§6 — a different axis: those are reimplemented, these are converted.

| id | feature | source | disposition | pinned by | destination | retirement | status |
|---|---|---|---|---|---|---|---|
| `fm.mace_mp` | MACE-MP aliases (small/medium/large, 0b/0b2/0b3, MPA-0, OMAT, MATPES) | published artifacts | KEEP — convert | P0-2 (MP-small) | FM-2 | n/a — published artifacts, not in-tree code | todo |
| `fm.mace_mh` | `mh-0` / `mh-1` — the MACE-MP multi-head releases | published artifacts | KEEP — convert with heads intact | ⚠️ gap (mh-0 golden → P0-2; it is also the Density-block evidence, §7) | FM-2 | n/a — published artifacts, not in-tree code | todo |
| `fm.mace_off` | MACE-OFF (OFF23 small/medium/large) | published artifacts | KEEP — convert | P0-2 (OFF-small) | FM-2 | n/a — published artifacts, not in-tree code | todo |
| `fm.mace_mdp` | MACE-MDP — the dielectric family (`AtomicDielectricMACE`), dipole and polarizability | published artifacts | KEEP — convert | ⚠️ gap (MDP golden → P0-3a) | FM-2, FT-4 | n/a — published artifacts, not in-tree code | todo |
| `fm.mace_omol` | MACE-OMOL — multi-head, `head="omol"` | published artifacts | KEEP — convert with heads intact | ⚠️ gap (OMOL golden → P0-2) | FM-2 | n/a — published artifacts, not in-tree code | todo |
| `fm.mace_polar` | MACE-Polar S/M/L (`PolarMACE`) | published artifacts | KEEP — convert, in Phase 5 with the electrostatics work | P0-3a | ELEC-2, FM-2 | n/a — published artifacts, not in-tree code | todo |
| `fm.mace_anicc` | MACE-ANI-CC — the 2023 organic-chemistry model | `mace/calculators/foundations_models/ani500k_large_CC.model` | DROP — superseded by MACE-OFF, and the only artifact bundled inside the wheel; REL-1 says "use MACE-OFF" | — | REL-1 (doc) | RET-6 | todo |

## 22. Open items

Not rows: the work this inventory hands to other tickets.

- **The `⚠️ gap` column.** Every gap is either a Phase 0 test addition (P0-1/2/3a/3b/3c/5/6/7) or a
  conscious downgrade recorded in the row, and all of them must be resolved before the Phase 0 gate
  closes. Each is also carried into its destination ticket as an explicit acceptance criterion, so
  closure is visible where the work happens rather than only here. The checker's `tally:` line is the
  authoritative count.
- **New goldens this inventory implies**, beyond the ones already scheduled: MP-0b2 or mh-0 (the
  Density interaction blocks turned out to be published-model architecture, not a research variant),
  MACE-OMOL, MACE-MDP, and the five `mace-scf` families.
- **The vendored-`torch_geometric` retirement got harder, not easier.** The `[magnetic]` extra declares
  external `torch-geometric` and `mace/data/augmentation.py` imports the real package and the vendored
  one at once, so RET-2 cannot simply delete the copy: MAG-1 removes both consumers first, and RET-2
  verifies neither survives. The external-dependents sweep (a code search over downstream projects
  before deletion) is an acceptance criterion of RET-2.
- **Three spellings of one quantity.** The per-layer energy contributions are `contributions` in the
  model output, `--return_contributions` on the eval CLI and `BO_contributions` in the written XYZ;
  per-atom energies are `node_energy` in the model, `energies` *and* `node_energy` (with the E0s
  subtracted) in the calculator, and `node_energies` in the XYZ; the LES latent multipoles are
  `latent_alphas` in the model and `LES_alphas` in the calculator. CORE-1 picks one name per quantity
  and REL-1 maps the old ones.

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
