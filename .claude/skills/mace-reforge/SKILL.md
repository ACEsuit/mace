---
name: mace-reforge
description: Shared working context for the MACE Reforge rewrite (MACE v1.0). Use whenever the task touches the rewrite - any ticket from the reforge board, anything under packages/, the frozen legacy mace/ package, the golden or parity test suites, or a design question about the new architecture. Load this before writing code for a reforge ticket.
---

# mace-reforge

The MACE v1.0 rewrite. This skill carries the rules every reforge change must
follow, so that work done by different people converges. Load it first; the
companion skills below carry the procedures.

| Companion skill | Use it for |
|---|---|
| `mace-reforge-ticket` | Taking a board ticket to a merged PR: branch, scope, verification, review gate |
| `mace-reforge-goldens` | The frozen oracle: golden/parity tests, tolerances, debugging a mismatch |
| `mace-reforge-numerics` | Units, sign conventions, equivariance, double-backward, precision |
| `mace-reforge-backend` | Kernel backends, dispatch, canonical weights, Clebsch-Gordan basis |

## Where the work comes from

Tickets live on the **reforge board**: <https://github.com/orgs/ACEsuit/projects/2>
(68 items, GitHub issues in `ACEsuit/mace`). Fields are `Status`
(Todo -> In Progress -> PR Open -> Closed), `Phase`, `Size`, `Area`, plus flag
labels (`gpu`, `network`, `numerics-critical`, `critical path`).

**Ticket bodies are self-contained by design** - a ticket states its context,
its acceptance criteria and its out-of-scope boundary, and is implementable
without opening anything else. Work only on tickets whose dependencies are
merged; the ticket names them.

Two planning documents are in the repository, for when you want the reasoning
behind a rule rather than the rule itself:

- **`docs/reforge/plan.md`** - the execution plan: the transition strategy, the
  five hard rules, the phases and their exit gates, the backend/kernel layer,
  the risk register, and a ticket index mapping every ID to its issue.
- **`docs/reforge/target_layout.md`** - the destination tree: where each piece
  ends up, module by module. Read this before deciding where a new file goes.

`docs/reforge/extending_mace.md` and `extending_plugin.md` cover the extension
surface, and `run_train_rewrite.md` the training-entry-point rewrite.

The **RFCs** are not in the repository: they are internal decision records, and
their decisions are inlined into the tickets they block. So a ticket is the
authority on what to build, and its RFC is not something to go looking for. If a
ticket genuinely does not determine what to build, say so on the ticket rather
than improvising a design.

## The repository holds two stacks at once

The rewrite is developed on the **`mace-reforge`** branch, cut from `develop`.
There is one state of that branch, not a before and after:

`mace/` is frozen **here**, not on `develop`. `develop` carries on taking v0.3
bug fixes and self-contained features, and whatever is new there is migrated
into v1 before the branches merge. So a fix you see on `develop` is not a fix
this branch is missing; it is a fix the migration owes v1.

```
mace/              Legacy v0.3.x. BYTE-FROZEN on this branch. The live differential
                   oracle. Never edited here; never imported by packages/.
packages/
  mace-core/       Framework-agnostic: types, config, observables, kernel Protocol,
                   Clebsch-Gordan, neighbours, data spec. Imports no torch, no jax.
  mace-torch/      The new PyTorch implementation. Depends on mace-core.
  mace-jax/        Inference-only JAX. Depends on mace-core.
  mace-launcher/   ~50 LOC. Owns the console entry points; dispatches on --engine {legacy,v1}.
tests/
  golden/          Committed fixtures, reference JSON, tiny checkpoints. Edit-locked.
  parity/          In-process legacy-vs-v1 comparison. Double-import allowlisted.
  architecture/    Fitness functions and the import-direction contract.
docs/reforge/      The execution plan and the target layout.
```

Check the tree before relying on a path: `tests/golden/` landed with Phase 0,
while `packages/`, `tests/parity/` and `tests/architecture/` are created by
their own Phase 1 tickets. This skill deliberately records **no** phase or
ticket status - that state lives on the board, which is always current.

## The five hard rules

- **R1 - Import direction.** `packages/` never imports `mace/`; `mace/` is never
  edited to import `packages/`. Enforced statically by import-linter and at
  runtime by an audit hook that catches dynamic `importlib` reach-in. Exactly two
  places are allowlisted for double import: the launcher's dispatch module and
  `tests/parity/`.
- **R2 - Legacy is spec, not source.** The frozen `mace/` answers "what must the
  numbers be", never "how should the code look". Do not copy legacy code into
  `packages/`. A meta-lint asserts no file under `packages/` names a legacy class
  (`ScaleShiftMACE`, `interaction_classes`, `AtomicData`, ...).
- **R3 - `mace_core` purity.** It imports no torch, no jax, no e3nn, and no
  sibling package. A test asserts it. Dtypes there are **names**
  (`Literal["float64", ...]`), never `torch.dtype`.
- **R4 - Spec-first.** Implement a frozen spec. If your ticket's design is not
  settled, stop and flag it on the ticket. Do not improvise, not even
  provisionally.
- **R5 - The oracle never moves.** `mace/` is byte-frozen and the goldens are
  edit-locked. A tolerance change is its own PR, never bundled with a feature.

## Rebuild, do not port

Start from the pinned behaviour and write the simplest implementation that
reproduces it. If you find yourself translating legacy code line by line -
abstractions, class hierarchy, parameter plumbing and all - you are doing it
wrong. When a legacy mechanism and a simpler one produce the same pinned
numbers, the simpler one wins.

Corollary: **breaking changes are fine, silent behaviour changes are not.**
Code and checkpoint compatibility are explicitly not preserved; a v0.3
checkpoint enters the new architecture through an explicit converter. But if
your implementation produces different numbers than the golden reference,
either it is wrong or you found a real discrepancy worth flagging. Never widen
a tolerance to make a test pass.

## Conventions for every line written under `packages/`

- **Descriptive names, no math shorthand.** `sender_node_features`, not `x_s`;
  `edge_radial_embedding`, not `R_ij`. Put the paper symbol in the docstring.
- **Typed, structured I/O.** Dataclass/Pydantic objects across public
  boundaries; never a raw dict. Models return `MACEOutput`, with `extras` as the
  only escape hatch.
- **Pure functions; pass the model explicitly.** No closures that capture and
  hide the model.
- **Do not over-abstract.** Three similar concrete functions beat one
  configurable god-function; duplicating a pipeline is acceptable when it avoids
  a framework. An abstraction added "for the future" should be deleted.
- **TorchScript is banned under `packages/`.** No `@torch.jit.script`, no
  `torch.jit.annotate`, no scripting-driven type contortions. `torch.compile`
  first, with eager as the always-working reference path.
- **JAX is inference-only.** Do not add training code to `mace_jax`.
- **Docstrings state units, shapes and conventions.** Error messages name the
  offending key or value and the fix.
- **Comments state constraints the code cannot** - units, conventions, why a
  seam exists - not narration.

**Path-scoped dual toolchain.** `mace/**` keeps black + isort + pylint + mypy
(`pre-commit run --all-files`) and is **never reformatted**, even where a diff
would look cleaner - that toolchain is permanent for the frozen tree, not
transitional. `packages/**` uses ruff (lint + format) + `ty`, run through
`prek`. A meta-lint asserts 1:1 file ownership, so no file falls between them.

## Target architecture, in one screen

- **Two layers.** `BaseMACE` is the equivariant backbone: a learned descriptor,
  node features in, node features out, no readouts and no gradient calls.
  `MACEOutputs` holds everything learnable beyond it - heads, readouts,
  normalization including readout-only E0s, typed output construction. New
  observables and heads are added there, driven by config.
- **Two-phase forward.** Forces and stress are computed by a derivative engine
  *around* the model call, never inside a module forward, so `torch.compile`
  sees no `autograd.grad` graph break. The strain displacement still has to be
  injected before edge vectors are computed; that seam is designed in its own
  ticket.
- **Declarative observables.** Any atomic or total spherical-tensor property is
  declared in config (name, irreps, per-atom flag, units, normalization) and
  becomes trainable with no new code: spec -> automatic head -> automatic loss
  term. Derivatives are auto-named (`d_<q>_d_pos`; `energy` special-cases to
  `forces`/`stress`). Declaring a property absent from the data is a hard error.
- **Staged training.** `DataStage -> ModelStage -> TrainStage` with typed
  boundary objects; an argparse namespace never crosses a stage boundary. The
  loop is explicit and short - no Lightning, no callback framework.
- **Config.** Pydantic(-settings), TOML/YAML/JSON, config-file-first with dotted
  CLI overrides, unknown keys are errors. The fully resolved config is saved into
  model metadata. CLI is hierarchical: `mace train / eval / model / data / export`.
- **Precision.** No global AMP. A `PrecisionConfig` assigns dtypes per op class.
- **Data.** Format-agnostic protocol in `mace_core` with pluggable backends
  (XYZ, HDF5, LMDB); padded/static-shape neighbour lists for compiled paths. No
  vendored `torch_geometric`.
- **e3nn and e3nn-jax are removed entirely** - not a dependency, not even an
  optional backend. Reference backends are plain torch and plain jax.

## Vocabulary used by the tickets

| Term | Meaning |
|---|---|
| Frozen oracle | The in-tree `mace/`: byte-frozen, the source of truth for what the numbers must be |
| Live parity harness | `tests/parity/` - in-process legacy-vs-v1 comparison, run continuously |
| Launcher / `--engine` | `mace_launcher`, owns the console entry points; `--engine {legacy,v1}` picks the stack |
| Deletion-only PR (`RET-*`) | Removes a piece of `mace/` once its v1 counterpart has parity. Zero new v1 logic |
| Debt book | Dated ledger of every legacy commitment still owed a v1 replacement |
| Span factory | Optional backend factory fusing a wider span of the interaction layer in one kernel |
| Conformance suite | The parametrized suite any kernel backend runs to prove itself |
| Parity anchor | The committed tiny `ScaleShiftMACE` checkpoint the new architecture must reproduce |
| Keystone gate | The exit test that the tiny checkpoint converts and matches its goldens at fp64, and matches the live legacy model in-process |
| Capability marker | pytest marker (`gpu`, `cueq`, `network`, ...): skip locally, fail in CI jobs that guarantee it |
| `prek` / `ty` | Pre-commit-style runner / type checker, used by the new stack only |
| Merge back | When the rewrite finishes, what is new on `develop` is migrated into v1, then `mace-reforge` merges into `develop` and `develop` into `main` for the v1 release |
