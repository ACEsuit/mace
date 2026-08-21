# Rewriting `run_train`: from a 1,090-line `run(args)` to a staged pipeline

**Status:** design note · **Expands:** the Phase 3 tickets **CFG-1** (config schema), **TRN-1**
(staged pipeline + explicit loop), **TRN-2/3/4** (losses, dataloaders, checkpoint/DDP) and
**CLI-1** (`mace train`) for the training driver specifically. **Companions:**
[`plan.md`](plan.md) §4 (Phase 3), the [ticket board](https://github.com/orgs/ACEsuit/projects/2), roadmap Ch. 6/9.

This is a design note, not a new decision surface. The staged-pipeline direction is already
decided (roadmap Ch. 6, plan Phase 3); this doc pins *how* it lands for `run_train`, block by
block, so TRN-1/CFG-1 are writable without re-deriving the mapping. Behavior is pinned by **P0-5**
(E2E CLI contracts) and **PAR-1** (in-process parity vs frozen legacy); the rewrite reproduces
numbers, not structure.

## 1. The mess, concretely

`mace/cli/run_train.py` is **1,181 lines**; `run(args)` is a single function spanning lines
**88→1181** (~1,090 lines — the roadmap's headline number). Measured properties of that function,
all on frozen `mace/` (read in place, never edited):

- **The argparse `Namespace` is the data model.** `run()` reads **71 distinct `args.*`
  attributes** and **mutates `args` in place in ~25 sites** — the namespace is both input and
  scratch state. Examples (absolute line numbers):
  - `args.heads = prepare_default_head(args)` (`:237`), `args.r_max = model_foundation.r_max...`
    (`:187`), `args.loss = "universal"` (`:383`), `args.enable_oeq = False` (`:797`),
    `args.avg_num_neighbors = get_avg_num_neighbors(...)` (`:752`).
  - Foundation-model presets silently overwrite user inputs: `args.lr = 0.0001`,
    `args.ema = True`, `args.ema_decay = 0.99999` (`:208-210`).
- **Model identity is a string that rewrites the observable set.** `:536-573` branches on
  `args.model == "AtomicDipolesMACE" | "AtomicDielectricMACE" | "EnergyDipolesMACE" | "PolarMACE"`
  and sets six `args.compute_*` booleans by hand per class. Adding an output means editing this
  ladder.
- **Everything is interleaved in one scope.** Foundation-model resolution → head prep → data
  loading (xyz / h5 / aselmdb branching, ASE-vs-non-ASE file splitting) → z-table → atomic
  energies / E0s (multiple branches incl. foundation, estimated, padding) → pseudolabels →
  dataloaders → `configure_model` → cueq/oeq conversion → optimizer/scheduler → SWA/stage-two →
  EMA → checkpoint resume → `tools.train(...)`. No stage boundary; a change in E0 handling sits
  next to a change in the optimizer.
- **The parser matches.** `mace/tools/arg_parser.py` is **1,316 lines** of untyped flags, config
  lives only as an optional `configargparse` YAML path.

The failure mode this produces: no place to validate before running, no typed contract between
"what data do we have" and "what model do we build", and foundation-model/fine-tuning logic
smeared across the whole function instead of isolated.

## 2. Target: three stages with typed boundaries

Roadmap Ch. 6 / TRN-1. `run()` becomes an **orchestrator of ~40 lines** that wires three stages;
the argparse namespace never crosses a stage boundary. Each stage takes and returns a typed object
(Pydantic / dataclass, defined in `mace_core`):

```
ResolvedConfig ──▶ DataStage ──▶ DataBundle ──▶ ModelStage ──▶ BuiltModel ──▶ TrainStage ──▶ TrainedModel
   (CFG-1)          (TRN-1)      (typed)        (TRN-1)        (typed)         (TRN-1)         (+ metadata)
```

- **`ResolvedConfig` (CFG-1, CORE-2).** One Pydantic object with `data / model / loss / optimizer /
  schedule / runtime / finetune` sections; config-file-first (TOML/YAML/JSON) with dotted CLI
  overrides; unknown keys are hard errors. All defaults filled and **saved into model metadata**
  (mandatory for every model, plan flag #10). A `from_namespace(args)` shim maps the legacy flag
  surface onto it so the legacy CLI keeps working during coexistence.
- **`DataStage` → `DataBundle`.** Owns everything from §1's data block: backends yield
  `Configuration`s (RFC-05), graph building/collation once above the backend (RFC-03), z-table,
  **E0 resolution as a typed `E0Spec` union** (RFC-07, replacing the six stringly-typed `--E0s`
  paths and their silent fallbacks), statistics (avg neighbors, mean/std) via the single
  `mace_core` implementation. Pseudolabel generation is a **callable stage** here (RFC-06), not
  `requires_grad` juggling welded into training.
- **`ModelStage` → `BuiltModel`.** The successor to `configure_model`: `ResolvedConfig` +
  statistics → a two-layer `BaseMACE` + `MACEOutputs`. **Observables are declared, not
  hand-set** — the `args.compute_*` ladder (§1) disappears; declaring `energy, forces, dipole`
  in config creates the heads (ARCH-3, CORE-1). Backend dispatch resolves here, once, at build
  time (RFC-01); the in-place cueq/oeq conversion goes away.
- **`TrainStage` → `TrainedModel`.** An **explicit train/validate/checkpoint/EMA/stage-two loop**
  (a few hundred readable lines, no Lightning, reproducible by a power user), losses composed from
  observable specs (TRN-2), multi-dataloader balancing (TRN-3), checkpoint/resume on
  **safetensors** + DDP (TRN-4). Fine-tuning is not a special case: replay vs real head differ
  only in their data source (FT-1).

## 3. Responsibility map (current `run()` block → v1 destination)

| Current `run()` responsibility (frozen `mace/`) | v1 home | Ticket |
|---|---|---|
| Parse + mutate `args` namespace | `ResolvedConfig` (immutable, validated) + `from_namespace` shim | CFG-1, CORE-2 |
| Foundation-model resolution + preset overrides (`:187-210`) | `finetune` config section + `ModelStage`; presets are declared defaults, never silent mutations | CFG-1, FT-1 |
| Head prep / multihead / `pt_head` (`:226-237`) | typed head configs; replay head = same interface, different source | FT-1 |
| Data loading, xyz/h5/aselmdb + ASE-file splitting (`:248-660`) | `DataStage`; backends yield `Configuration` (RFC-05), graph build once (RFC-03) | DATA-1/2/3 |
| z-table + atomic energies + E0s + padding (`:329-495`) | `E0Spec` typed union + resolver; silent fallbacks become hard errors | RFC-07, CFG-1 |
| Pseudolabels (`:595`) | callable `generate_pseudolabels` stage, rank-0, persisted artifact | RFC-06, FT-2 |
| `args.compute_*` set by model-class string (`:536-573`) | declarative observables → automatic heads | CORE-1, ARCH-3 |
| `configure_model` (`:755`) | `ModelStage` (`ResolvedConfig` + stats → model) | TRN-1, ARCH-2/3 |
| cueq/oeq in-place conversion (`:806`) | backend resolved once at build time; no conversion CLIs | RFC-01, BKD-1/3 |
| optimizer / LRScheduler / SWA / EMA / resume (`:780-890`) | `TrainStage` setup, per-stage schedules | TRN-1/2 |
| `tools.train(...)` + checkpoint + save (`:942…`) | explicit loop, safetensors checkpoint, resolved-config metadata | TRN-1/4 |

## 4. Coexistence and how it ships

- **Launcher (INF-2).** `mace_run_train` is owned by `mace-launcher`; `--engine legacy` (default)
  calls the frozen `mace.cli.run_train:run` **byte-identically**; `--engine v1` calls
  `mace_torch.cli.run_train`. Day-one behavior is unchanged.
- **CLI-1** exposes it as `mace train` with a faithful port of the pinned legacy flag surface, so
  the 65 black-box workflow tests re-run under `--engine v1` unchanged (one indirection in
  `tests/helpers.py`).
- **Gate (GATE-3).** `mace train --engine v1` on the tiny task reaches legacy-comparable
  loss/errors (`tiny_scaleshift_training_errors.json`), reinforced by PAR-1's single-step gradient
  parity. The default flips to `v1` only when parity is green; legacy stays as the oracle until
  RET-1 deletes it.

## 5. Out of scope / open

- **Not this doc:** the full `arg_parser` → config field mapping (CFG-1 owns it, driven by the P0-0
  inventory), the derivative engine seam (ARCH-4), the loss algebra (P0-6/TRN-2).
- **CLI surface reduction** (eval/plot/active-learning, marked REVIEW in the inventory) is decided
  in the broader CLI redesign, not here — this note only covers the training driver.
- **Open:** exact stage-boundary object schemas (TRN-1 first PR), and whether `DataStage` and
  `ModelStage` share a single `Statistics` object or recompute — settled when TRN-1 is written.
