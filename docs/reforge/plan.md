# MACE Reforge — Execution Plan

**Status:** living document · **Source of truth for scope:** the MACE Reforge workshop roadmap, an internal design document. It is cited below by chapter ("roadmap Ch. 4") where it is the authority for a decision; this plan is otherwise self-contained.
**Companions:** the [ticket board](https://github.com/orgs/ACEsuit/projects/2) (the backlog — one GitHub issue per ticket) · [`target_layout.md`](target_layout.md) (the destination tree) · §10 (ticket ID → issue)

This document turns the workshop roadmap into an executable plan: how the migration from v0.3.x to the rewrite works, what the phases and their exit gates are, how the work is organised, and where the roadmap is contradictory or under-specified. Dates are deliberately absent — phases are gated by exit criteria, not by the calendar. The roadmap's June–August window is treated as aspirational, not binding.

## 0. Binding decisions

These are fixed and are not re-litigated in tickets:

| # | Decision |
|---|----------|
| D1 | **Full roadmap scope**: `mace-core` + `mace-torch` + `mace-jax` (inference-only) + an educational reference implementation. |
| D2 | **Compatibility**: user checkpoints from v0.3.x are not runtime-loadable in the new architecture. Only published foundation models get a **one-shot converter** — the explicit artifact list (amended in review, 2026-07: the earlier family shorthand was ambiguous about the multi-head releases): the MACE-MP aliases **including `mh-0`/`mh-1`**, MACE-OFF, MACE-Polar, MACE-MDP, and **MACE-OMOL** (multi-head artifacts convert with their heads intact; single-head exports are `mace model select-head`, a separate step). MACE-ANI-CC: **dropped** (decided 2026-07-29) — superseded by MACE-OFF for organic chemistry, and its loader was the only one with a divergent signature; REL-1 points those users at MACE-OFF. "Loadable" in roadmap Principle 4 means *one-shot convertible*, not runtime-loadable. |
| D3 | **Test prerequisite bar** (roadmap Ch. 11 "hard prerequisite"): golden characterization tests of physics behaviour + E2E CLI contracts + line-coverage floors **only** on modules whose behaviour ports to the rewrite. No global coverage percentage target. |
| D4 | **Workflow**: tickets are self-contained work packages — context, steps, acceptance criteria and verify commands — consolidated so that one ticket is one PR. The limiting factors are review bandwidth per PR and parallelism, not implementation time. The backlog is the [ticket board](https://github.com/orgs/ACEsuit/projects/2). |
| D5 | **Backlog location**: the [**GitHub Project**](https://github.com/orgs/ACEsuit/projects/2) is the source of truth — one issue per ticket in `ACEsuit/mace`, with dependency fields, progress tracking and native PR linking. Ticket IDs are stable and resolve to issue numbers through §10. **Board shape (decided 2026-08-03):** an org-owned Project v2 under ACEsuit mirroring **61 tickets** — Phase 1 through 5 plus the independent tracks. Two groups are deliberately not exported: the **5 RFC-track** tickets (closed work, decisions already inlined) and the **9 Phase 0** tickets (prerequisite characterization of the frozen legacy, done locally before the plan starts — they depend on local artifacts nobody else can act on). `Status` has exactly four options — **TODO → IN PROGRESS → PR OPEN → CLOSED** — plus fields `Phase` and `Size` and the flag labels (`gpu`, `network`, `numerics-critical`, `critical path`). Deliberately flat: no `Type`, no `Outcome`, no per-type lanes. **All 61 import as `TODO`** — the board starts clean and its state is maintained there from then on, never back-filled from local branch state. Note that a Projects `Status` field is a single-select and enforces no transitions: the column a ticket sits in is a statement of where it is, not a gate. |
| D6 | **Progressive migration.** The new implementation grows alongside the frozen legacy package, which serves as the differential oracle. Legacy is reference material for *behaviour* only, never for structure; it is never edited and never imported by the new stack. It is deleted capability-by-capability once each capability reaches full parity. |

## 1. Working model

Tickets are picked from the backlog, not pre-assigned. Two kinds of work are distinguished by the tickets themselves, not by who does them:

- **Numerics-critical tickets** (model architecture, derivative engine, losses, electrostatics, backends) require physics judgment in both implementation and review — they carry explicit design notes and golden/finite-difference validation so correctness is checkable, not assumed.
- **Specification-driven tickets** (ports pinned by characterization tests, data backends, CLI subcommands, docs, pattern-following work) have tight, executable acceptance criteria so anyone can land them and review stays cheap.

Rules of the working model:

- Every session starts with the ticket, then the files the ticket names.
- Tickets never contain open design decisions — those are resolved in an RFC or a parent ticket before the ticket is picked up.
- **Numerics expertise is the capacity bottleneck** (about half the tickets touch numerics). Mitigation: inside bundled tickets, the first instance of each pattern (first backend, first foundation-model conversion) is the numerics-critical half; the pattern-following half is a separate specification-driven hand-off.
- Every PR is reviewed before merge; numerics-critical changes need a reviewer with physics expertise.
- Branch naming: `reforge/<TICKET-ID>-<slug>`. PRs target `mace-reforge`, which is installable and useful at every point.

## 2. Transition strategy: coexistence with a live differential oracle

The new stack (`packages/{mace-core,mace-torch,mace-jax}`) is built alongside a **byte-frozen** legacy `mace/` package. Frozen legacy is a **live, in-process differential oracle**: the parity harness loads both stacks in the same process (or in a forked subprocess) and compares them numerically on every migrated capability. Development proceeds as end-to-end **vertical slices** (config → kernel → observable → model → loss), selectable by an `--engine {legacy,v1}` flag on a thin launcher. Legacy is retired capability-by-capability at the end, through **deletion-only** PRs. `mace-reforge` is installable, releasable, and useful at every point; a hard release gate blocks the 1.0 tag until `git ls-files mace/ == 0`.

### 2.1 The two stacks and the double-import allowlist

The two stacks **never import each other**. The only exceptions on the double-import allowlist are:

- **`mace_launcher.dispatch`** — the launcher (`packages/mace-launcher`, ~50 LOC) owns **all** console entry points and dispatches via `--engine {legacy,v1}` / `MACE_ENGINE` to `mace.cli.*` or the new `mace_torch.cli.*`. It **defaults to `legacy`**, so day-one behaviour is unchanged.
- **`tests/parity/`** — the harness that imports both stacks to compare them numerically.

Everything else is one-directional (`packages ⊥ mace`). New code is organised per [`target_layout.md`](target_layout.md):

- **`mace_core`** (no torch/jax/e3nn): typed `MACEOutputs`, Pydantic `ModelConfig`/`PrecisionConfig`, declarative observables, the flat-dict graph contract, numpy/matscipy neighbours, Clebsch–Gordan, the kernel Protocol + entry-point registry, and `get_outputs` as pure physics — all **reimplemented**, never imported from legacy.
- **`mace_torch`**: two-layer `BaseMACE` models with blocks born **without `@compile_mode("script")`**, the staged training pipeline, the ASE calculator, the new `cli/`, and the kernel backends registered via `entry_points("mace.kernel_backends.torch")`. Hot ops are `torch.library.custom_op` + `register_fake` with a reference double-backward autograd path; the canonical weight layout runs `to_canonical`/`from_canonical` only at the backend boundary.
- **`mace_jax`**: the inference-only destination, validated against the neutral safetensors+JSON format and the goldens.

### 2.2 The five rules that guarantee design purity

Purity is not a review aspiration; it is enforced in CI.

| # | Rule | CI enforcement |
|---|------|----------------|
| R1 | **One-way import direction** (`packages ⊥ mace`): a legacy class is physically unreachable from the new zone — it cannot be subclassed, re-exported, or imported. | `import-linter` (static) **plus** a runtime `sys.addaudithook` that closes the dynamic `importlib` reach-in. |
| R2 | **Spec-first**: RFC-A..D are Accepted before the keystone slice is written; v1 implements a frozen spec, not a port. | Process gate at the Phase 0 exit. |
| R3 | **No structural copying**: no `packages/**` file names a legacy class (`ScaleShiftMACE`, `interaction_classes`, `AtomicData`, …). | Meta-lint over the new tree. |
| R4 | **Debt book**: every temporary compromise is a dated countdown. The dates are **internal** — there is no public EOL promise; `main` stays on v0.3 until v1 is released, and no end date for v0.3 is announced. | Fitness functions go **red** when a date expires. |
| R5 | **The oracle never moves**: `mace/` is byte-frozen and the goldens are edit-locked. The freeze is **on `mace-reforge`**, not on `develop`: the rewrite is built on that branch and never edits `mace/` there, while `develop` carries on taking 0.3.x bugfixes and self-contained features. Nothing closes the v0.3 line; `main` holds it until v1 ships. A tolerance change is its own PR. | Branch rule + edit-lock check; a tolerance change lands only as a standalone, reviewed PR. |

### 2.3 Strangling order and the retirement criterion

Migration is bottom-up. `mace-core` foundations first; then the kernel Protocol + plain-torch reference backend (compile-first from day one); then the keystone vertical slice — `ScaleShiftMACE` energy+forces, reference backend, in-memory XYZ producing the flat-dict contract, single head, float64 — which front-loads every hard bet. On top of the keystone come the physics/observables axis (stress/virials, ZBL, per-model E0/scale-shift semantics), the backends axis (cueq/oeq behind the single Protocol, plus an out-of-tree example backend as a gate), the data axis (HDF5/LMDB, `mace_prepare_data`, DDP), the extra readouts and foundation-model conversion, and finally deployment.

Each capability reaches **full fp64 parity** in v1 while its legacy counterpart runs as the oracle under `--engine=legacy`. After it has been default-`v1` through a deprecation window with green parity, its legacy modules are **physically deleted** in a deletion-only PR (zero new v1 logic), and its parity test degrades to the frozen golden. `git ls-files mace/` decreases monotonically. **Done = `git ls-files mace/ == 0` with the 1.0 release gate satisfied.**

### 2.4 Alternatives considered

- **In-tree coexistence with adapters/shims** is rejected: it requires editing the legacy package to expose adapters, which *moves* the numerical oracle and opens a structural-recontamination path where new code inherits legacy structure through a shim. Freezing legacy and forbidding all imports across the boundary is what makes the oracle trustworthy.
- **A long-lived rewrite branch** was initially rejected, and is now the approach: the rewrite lives on `mace-reforge`, with `mace/` frozen there and `develop` continuing to ship v0.3. Two of the three original objections do not apply to that shape. `develop` is neither stale nor broken, because it keeps taking bug fixes and self-contained features and releases through `main`; and parity stays continuous, because the frozen oracle travels **on the branch**, so the harness compares against it every day rather than at milestones. The third objection stands and is accepted as a cost: the final merge is large. It is mitigated by keeping every PR into `mace-reforge` ticket-sized, so the work is reviewed continuously and only the merge itself is bulk, and by an explicit migration pass that carries whatever landed on `develop` into v1 before the branches meet.
- **A new repository** is rejected: it loses history, issues, and the GPU-runner setup, and contradicts the workshop's unanimous monorepo decision (roadmap Ch. 3).

## 3. Prerequisites (before the Phase 0 gate closes)

1. **CI branches are merged into `develop`** — already done. The capability-marker test layout (`tests/{unit,workflows,backends,extensions,foundations,integrations,benchmarks}`), the `.github/actions/run-tests` composite action (a job's `with:` block maps 1:1 to pytest flags and is its local reproduction recipe), and the `ci-core`/`ci-extensions`/`ci-integrations`/`ci-gpu-mpcdf`/`nightly`/`release` workflows are all on `develop`.
2. **GPU coverage — closed (2026-08-03).** Both gaps this section used to track are gone, and the mechanism that tracked them is gone with them: the self-hosted single-host fleet (`.github/gpu-fleet.json` + `ci-gpu.yaml`) was replaced by `ci-gpu-mpcdf.yaml`, which drives MPCDF's shared GitLab runners from GitHub and surfaces **one independent check per vendor**. There is no fleet file and no `enabled` flag any more; a vendor is a job in `.github/gitlab/ci.yml`.
   - **Nvidia** (A30 MIG): `-m gpu`, caps `gpu,cueq,oeq,les,schedulefree,network` — **49 passed / 1 xfailed**, first green run driven from GitHub 2026-08-03 (PR #1546).
   - **AMD** (MI210): `-m "gpu and not cueq"`, caps `gpu,oeq,schedulefree` — **38 passed / 0 skipped**. AMD had never run before.
   - What the change actually bought: **`oeq` now runs at all** (the vendor jobs override the base image with a real toolkit — `nvidia/cuda:*-devel`, `rocm/dev-ubuntu-*` — because OpenEquivariance JIT-compiles its kernels; the retired host had only a driver). The 24 `oeq` tests had been dead everywhere until then. Network is enabled too (`MACE_CI_ALLOW_NETWORK=1`, `network` in the required caps).
   - Backend-parity gating (P0-4, BKD-3) therefore covers **cueq on Nvidia and oeq on both vendors** from the outset — it is no longer waiting on a human to provision anything.
   - Caveat for any cross-vendor claim: the two jobs are **not on the same torch** (`2.13.0+cu130` vs `2.9.1+rocm6.3`, because the `rocm6.3` wheel index tops out at 2.9.1). A vendor comparison that omits this is measuring the wheel, not the vendor.

## 4. Phases and exit gates

Dependency-ordered. "Phase N+1 starts" means its critical-path tickets start; independent tracks (EDU, GOV, RFCs) run whenever someone is free. Every phase leaves `mace-reforge` installable and useful; `--engine` defaults to `legacy` until a capability reaches full parity, at which point it can flip to `v1` with opt-out. Full ticket definitions are in [`tickets.md`](https://github.com/orgs/ACEsuit/projects/2).

### Phase 0 — Behavioural safety net and frozen design

Golden characterization of v0.3.x physics behaviour + targeted coverage of port-worthy modules + the **functionality inventory** — this is what makes the rewrite falsifiable and complete. Tickets **P0-0..P0-8** and **RFC-A..D**.

- **P0-0** enumerates every v0.3.16 feature mechanically from the code (all `arg_parser` flags, entry points, module registries, calculator options, extras, documented tutorials) with a KEEP/MERGE/DROP disposition and its pinning test.
- **P0-1** builds the framework-agnostic golden harness, the per-dtype tolerance table, and a tiny `ScaleShiftMACE` checkpoint trained once and committed as the parity anchor. **P0-2/P0-3** add goldens for MACE-MP/OFF and dipole/polar/MDP models; **P0-4** adds cueq/oeq parity goldens on GPU CI.
- **P0-5** pins E2E CLI contracts (train smoke with loss-decrease, resume, eval, ASE calculator, fine-tuning replay, LAMMPS export). **P0-6/P0-7** characterize loss math, the forces/stress-via-autograd glue, E0s, radial bases, parsing, neighbour lists, and batching. **P0-8** wires per-module coverage floors and performance baselines.

**Exit gate:**
- [ ] Functionality inventory complete (count cross-check passes); every KEEP/MERGE row pinned; DROP list reviewed.
- [ ] Golden harness merged; references committed (e3nn CPU fp64) for the tiny anchor, MACE-MP-small, MACE-OFF-small, tiny dipole, MACE-Polar, MACE-MDP; the **virial sign convention is pinned**.
- [ ] The anchor fixture set includes **ZBL/`--pair_repulsion` enabled for BOTH `MACE` and `ScaleShiftMACE`** (the two differ in whether the ZBL term is inside or outside the scale-shift), and **explicit E0/scale-shift goldens in fp32 AND fp64, per-model and per-quantity** (total energy vs node energy — see §7).
- [ ] Backend golden parity green on `ci-gpu-mpcdf.yaml` for **both vendor checks** (cueq+oeq on Nvidia, oeq on AMD); E2E CLI contracts green on `ci-core`.
- [ ] **Training numerics pinned, not just behaviour**: beyond the P0-5 loss-decrease smoke and the committed error table, the **single-step `d(loss)/d(params)` gradient golden** (P0-6, fixed seed + committed weights, fp64, both anchors, exercising `create_graph=True`) is committed, and `gradgradcheck` is green on the tiny fixture — this closes the §7 training-numerics gap that a JSON-of-errors cannot see.
- [ ] Per-module coverage floors enforced for `mace/modules/{loss,radial,utils}.py` and `mace/data/{utils,neighborhood,atomic_data}.py`.
- [ ] **RFC-A..D Accepted** (R2). RFC-A fixes the final shape of the kernel Protocol, `custom_op` contract, and canonical layout.

### RFC track (roadmap Appendix A)

Eight RFC **documents** — internal decision records whose decisions are inlined into the tickets they block — delivered by four bundled tickets pairing RFCs that share context: **RFC-A** (backend dispatch + Clebsch–Gordan), **RFC-B** (neighbour lists + canonical data format), **RFC-C** (E0 specification + pseudolabel caching), **RFC-D** (JAX scope + deployment). Acceptance = both records at status `Accepted` and the blocked tickets writable against the interface sketches.

A ninth RFC surfaced during implementation, outside the roadmap's Appendix-A eight: **RFC-E** (**RFC-09** — electrostatics / long-range solver backend dispatch), motivated by [`ACEsuit/mace#1524`](https://github.com/ACEsuit/mace/pull/1524). It generalizes RFC-01's dispatch pattern to the electrostatics solver op class (k-space/PME + SCF fixed point) so the solver is swappable between a plain-torch reference (`graph_longrange` successor) and accelerated libraries (NVIDIA `nvalchemiops`), and pins the one place the analogy breaks — a non-bit-parity accelerated solver is model-affecting state, not a free backend choice. It **blocked ELEC-1/ELEC-2/ELEC-3** and is **Accepted (2026-07-29)** — 10 decisions closed, mirroring RFC-01 (separate registry, bit-parity→free / non-parity→model-state, build-time selection, k-space op + optional SCF span, reference = `graph_longrange` successor, v1.0 = reference + dispatch, `nvalchemiops` follow-on); the electrostatics tickets are unblocked as of that acceptance. It **is** on the critical path: ELEC-1/ELEC-2 are Phase 3 and GATE-3 covers them, so this RFC has to be accepted before Phase 3 opens; only ELEC-3 waits for Phase 5.

### Phase 1 — Coexistence scaffold and mace-core foundations

Tickets **INF-1..INF-5, CORE-1..CORE-3, CORE-4**. INF-1 is a **non-destructive scaffold** ticket: it creates `packages/` and installs the three packages + `mace-launcher` alongside a frozen `mace/`; it deletes nothing. INF-2 is the launcher; INF-3 the import guard (import-linter + audit-hook + anti-structural meta-lint); INF-4 the path-scoped dual toolchain; INF-5 the debt book. CORE-1..3 build typed outputs + observable spec, the config/metadata skeleton, and the framework-agnostic `Configuration` type; CORE-4 vendors the reduced Clebsch-Gordan basis and a pure-torch `SegmentedPolynomial` evaluator (see §6).

**Exit gate:**
- [ ] The three packages + `mace-launcher` install editable **alongside** `mace/`; `import mace, mace_core, mace_torch, mace_jax` succeed in one process.
- [ ] `git ls-files mace/` unchanged; the legacy suite is green; `--engine=legacy` is byte-identical to calling `mace.cli.*`.
- [ ] **Exactly one installed distribution provides each `mace_*` entry point**: `console_scripts` are removed from the root `setup.cfg`; `mace-launcher` is the sole owner (asserted in CI).
- [ ] `import-linter` green; the runtime audit-hook operational; the dual path-scoped toolchain green; the anti-structural meta-lint green.
- [ ] `MACEOutput`, the observable spec, the config round-trip, `Configuration`, and **CORE-4** merged — the reduced basis golden-locked against an **independent** CPU Clebsch-Gordan derivation (never against itself), with a parameter-count golden per `(irreps, correlation)` so the basis can no longer vary with `CUET_AVAILABLE`, and no `cuequivariance` import left in `mace_core`.
- [ ] A dummy backend registered via an entry point is discovered.
- [ ] The branch model is recorded in `CONTRIBUTING.md`: `main` carries releases and stays on v0.3 until v1 ships, `develop` keeps taking 0.3.x work, `mace-reforge` carries the rewrite with `mace/` frozen. There is no maintenance branch, and **no announced EOL date**; the debt book (INF-5) carries the rewrite's own internal countdowns.

### Phase 2 — Keystone slice: architecture, backends, data

Tickets **ARCH-1..4, BKD-1..4, DATA-1..3, FM-00, PAR-1, PAR-2**. ARCH-1 ports radial/embedding/spherical-harmonics blocks; BKD-1 builds the reference backend + Clebsch-Gordan module per RFC-A; ARCH-2 assembles the backbone (interaction + symmetric contraction on the dispatch interface) with equivariance tests; ARCH-3 the output layer (`MACEOutputs`, readout-only E0s, dipole/polarizability as observables); ARCH-4 the two-phase forward + derivative engine (strain injected before edge vectors). FM-00 is the in-process keystone converter; BKD-2 adds the compile path + `PrecisionConfig`; BKD-3/4 the cueq/oeq backends and the out-of-tree example backend; DATA-1..3 the dataset API, HDF5/LMDB, and neighbour lists/batching without vendored `torch_geometric`. PAR-1/PAR-2 are the continuous parity harness.

**Exit gate (the keystone):**
- [ ] **FM-00**: the committed anchor converts **in-process** (importing the in-tree frozen `mace/`, no separate venv) and reproduces the Phase 0 E/F/stress goldens at fp64, **and** matches the live legacy model loaded in the same process — with the full→reduced conversion pinned to an **independent** Clebsch-Gordan derivation, never to CORE-4's own basis.
- [ ] **PAR-1**: in-process legacy-vs-v1 parity green on tiny fixtures, with **global process-state isolation** between the two runs (snapshot/restore of env vars, default dtype, e3nn optimization defaults, and RNG, or a forked subprocess).
- [ ] **Single-step, init-agnostic gradient parity test** (per-PR): byte-identical weights in both stacks, one forward+backward on an E+F loss, `d(loss)/d(params)` allclose at fp64 — this exercises the `create_graph=True` second-derivative path.
- [ ] Forces/stress validated by finite differences against v1; equivariance passes; compiled == eager with no graph break; cueq/oeq parity (fwd + bwd + **double-bwd**) green on GPU; **BKD-4** passes with zero core edits.
- [ ] Data axis: **zero `torch_geometric` at the model interface** (active fitness function); a golden verifies the v1 collater reproduces `ptr`/`batch` bit-exact.
- [ ] The energy+forces+stress capability can flip its default to `v1` with opt-out; legacy retained as oracle. Opt-in **explicitly excludes LAMMPS export/checkpoint until DEP-2** (documented in the debt book).

### Phase 3 — Training pipeline, config/CLI, electrostatics core, fine-tuning

Tickets **CFG-1, TRN-1..4, CLI-1, FT-1..3, ELEC-1/2, DEP-1, MAG-1, GATE-3**. Full training config schema (CFG-1); the staged pipeline with typed stage contracts replacing the >1,100-line `run()` and an explicit training loop, no Lightning (TRN-1); composable losses from the observable spec (TRN-2); multi-dataloader balancing + metrics/logging (TRN-3); safetensors checkpointing + DDP (TRN-4); the hierarchical `mace train/eval/model/data/export` CLI via the launcher (CLI-1); heads-as-modules fine-tuning, pseudolabels, all-species weights (FT-1..3); the **electrostatics core** — graph_longrange behind the `[electrostatics]` extra with its solver-dispatch layer and its own CI job (ELEC-1) and PolarMACE on the new architecture with golden parity and the polar user surface (ELEC-2); the ASE calculator (DEP-1, which starts once ARCH-4 lands); the MagneticMACE family re-landed declaratively as magmom-input-feature + magnetic-moment observable + `dE/dm` derivative, rather than as the forked class family it is in legacy (MAG-1, added to this list 2026-07-29 — the ticket existed but the plan did not name it); the phase gate (GATE-3).

ELEC-1/2 sit here rather than with the rest of electrostatics because Phase 3 consumes them: the polar and LES calculator surfaces are a Phase 3 deliverable, so is MDP fine-tuning, and GATE-3 gates both — leaving the two ports in Phase 5 made the Phase 3 gate wait on most of Phase 5. Their MACE-Polar **artifact conversion** does not come with them; it runs on the FM-1 tool and therefore stays in Phase 4 with the rest of the roster (FM-2). ELEC-3 (split-charge/SCF) and ELEC-4 (MACELES) stay in Phase 5.

**Exit gate:**
- [ ] `mace train --engine v1` trains the tiny task end-to-end: loss decreases, checkpoint + resolved-config metadata written, resume works, errors comparable to legacy (`tiny_scaleshift_training_errors.json`) — reinforced by the single-step gradient test.
- [ ] `mace eval` and the new ASE calculator agree numerically; multihead replay and CPU-DDP smokes pass.
- [ ] The must-have example works: a new spherical-tensor observable declared purely in config, trained and evaluated (GATE-3).
- [ ] Electrostatics installable via the `[electrostatics]` extra with its own CI job green; the reference solver resolves once at build time with nothing dispatching in `forward`; the ported PolarMACE reproduces the P0-3a polar golden at fp64 and its user surface (density-cube CLI, the seven polar result keys, Fukui output, the polar Hessian) is reachable under `--engine v1`.
- [ ] The **65 black-box tests** (54 in `tests/workflows` + 11 in `tests/integrations`) pass with `--engine v1` for the migrated capabilities.
- [ ] From Phase 3 onward, CI requires `git ls-files mace/` to be **strictly non-increasing on `mace-reforge`** on the schedule set in the debt book. It says nothing about `develop`, where `mace/` keeps growing until the merge back.

### Phase 4 — Foundation models and JAX

Tickets **FM-1..3, JAX-1..2**. The production two-step converter generalizing FM-00 (in-process for development; a pinned `mace-torch==0.3.16` venv only at the end-user packaging boundary, for old pickles) with golden verification (FM-1); conversion of the MACE-MP/OFF/MDP/Polar families with parity vs Phase 0 goldens (FM-2); the model registry with naming, deprecation warnings, DOIs, loaders, and a nightly validation badge matrix (FM-3); the mace-jax skeleton + weight import + inference forward with torch parity (JAX-1); the jitted static-shape pipeline + minimal JAX ASE calculator (JAX-2). MACE-Polar conversion stays here with the rest of the roster: the model it converts onto is ELEC-2's, delivered in Phase 3, but the conversion itself runs on FM-1's two-step tool, so FM-2 adds the `PolarMACE` source path and gates the three artifacts on the P0-3a polar golden.

**Exit gate:**
- [ ] MACE-MP + MACE-OFF + MACE-MDP + MACE-Polar converted, matching Phase 0 goldens on CPU and GPU **and** the live legacy in-process; an **fp32 bit-reproduction golden** with a deterministic `segment_reduce` is matched at fp32 machine epsilon (see §7).
- [ ] The model registry is live with metadata and deprecation mechanics; mace-jax reproduces the torch goldens for the RFC-D model list.

### Phase 5 — Electrostatics extensions, deployment, legacy retirement, release

Tickets **ELEC-3/4, DEP-2/3/4, RET-1..6, REL-1..3**. The electrostatics *core* — the reference solver, its dispatch layer and PolarMACE — moved to Phase 3 (ELEC-1/2); what remains here builds on it: the split-charge / SCF charge-aware families from `ACEsuit/mace-scf` reimplemented on the two-layer + model-transform-hook design (ELEC-3) and MACELES with its `[les]` extra, BEC, latent multipoles and external-field path (ELEC-4). LAMMPS via `torch.export`/MLIAP with an emergency exit (DEP-2); OpenMM/Symmetrix/libmace (DEP-3); the torch-sim backend on the new engine, promoted to a first-class deployment path by the P0-0 disposition pass (DEP-4); the capability-by-capability retirement in deletion-only PRs (RET-1..6, in the order energy models → data layer → the five `convert_*` CLIs → dipole/polar/`extensions.py` → LAMMPS jit → the `mace/tools` god-module + the launcher's legacy branch); the migration guide (REL-1, no release); the package release pipeline and the v1.0.0 tag + DOI (REL-2); docs consolidation (REL-3). REL-1 publishes the guide and ships no release.

**Exit gate:**
- [ ] The `[les]` extra installs and its BEC / latent-multipole / external-field surface is green against the P0-3c golden, and every `mace-scf` family has a recorded disposition (reimplemented or version-tagged) — the `[electrostatics]` extra itself was gated at Phase 3; LAMMPS MLIAP numerics match the ASE calculator; no `jit.*` in the live v1 path (except an isolated export adapter if the emergency exit applies).
- [ ] RET-1..6 merged; `git ls-files mace/ == 0`; each parity test degraded to its frozen golden; `import-linter` trivially satisfied; a single toolchain; coverage floors over `packages/` only.
- [ ] The migration guide published with the DROP list; mace-docs tutorials executed against the v1.0.0 build. (REL-1 ships **no release**, and no deprecation warning will ever name a v1 command — the last 0.3.x binary predates v1's CLI — so the guide is the only signpost and must be linked from the README, the docs landing page and the PyPI description.)
- [ ] **Hard release gate**: the 1.0 tag is blocked until `git ls-files mace/ == 0`.
- [ ] v1.0.0 tagged and published with DOI (no release candidate).

### Independent tracks (no phase gate)

- **EDU** — educational JAX implementation in `tutorials/` (top-level, a sibling of `packages/`; **not** the "reference backend", which is the plain-torch/jax kernel oracle inside the packages): a pure-function forward, **evaluation/inspection-only** — for chemists/physicists to read the model blocks, not a training path (there is no JAX training anywhere) nor a mirror of the latest architecture (EDU-1); marimo notebooks with CI execution and a numerics cross-check vs mace-torch (EDU-2, only the cross-check depends on the new stack).
- **GOV** — CONTRIBUTING.md + governance doc (GOV-1); agent-friendly repo layout + issue automation + a PR-review bot (GOV-2).

## 5. Critical path and de-scope line

**Critical path:**
```
Phase 0 → RFC-A → CORE-4 → BKD-1 → ARCH-2 → ARCH-3 → ARCH-4
       → FM-00 → PAR-1 → TRN-1/2 → FT-1 → FM-1/2 → RET-1..RET-6 → REL-1/2
```
Any slip here slips v1.0.0. **CORE-4** (computing/vendoring `reduced_symmetric_tensor_product_basis` from first principles, a bounded numpy job) is on the path: it is what lets `mace_core` fix the basis for every device with **no e3nn and no cuequivariance** (§6.5). It pins the canonical path order and per-path normalization (that is the on-disk weight format, §6.5) and also owns the native `full ↔ reduced` conversion used at load; **BKD-1a** implements the canonical serializer against it before FM-00. There is no separate `SegmentedPolynomial` evaluator ticket — the reference keeps its Horner cascade.

**De-scope line** — if capacity runs out, scope is cut in a recorded order rather than silently: several items are roadmap must-haves, so invoking a cut is an explicit, recorded deviation. **The critical path is never cut**, and **RET-6 (`git ls-files mace/ == 0`) is non-negotiable for 1.0** — the retirement is de-scopable but never silent, because the hard release gate blocks the tag.

## 6. The backend / kernel layer

The layer that lets a third party bring a different library or a custom GPU kernel (CUDA/HIP/SYCL/Triton) **without touching the core, without breaking `torch.compile`, and with correct force-training (double-backward)**. It subsumes and simplifies what `mace/modules/wrapper_ops.py` does today, where the backend leaks into the model in at least eight places: `blocks.py:687,790,909,1033` branch on `hasattr(self, "conv_fusion")`, and `wrapper_ops.py:130,157,205,227` monkeypatch `.forward` with `types.MethodType`. RFC-01 (RFC-01) is the full design; the summary below is normative.

### 6.1 The ops contract

A **small, closed** set of primitives — exactly the ops that differ between backends:

- **`ChannelwiseTPConv`** — always node-level: it returns `[n_nodes, …]`. Whether it fuses the scatter into one kernel is a backend-internal choice the model never sees; this deletes the `hasattr(conv_fusion)` branching and the monkeypatch. `num_nodes` is passed explicitly, never inferred from `edge_index.max()`.
- **`SymmetricContractionOp`**, **`LinearOp`** (bias is first-class, because `cuet.Linear` has none, which is why the legacy bias readouts hardcode `o3.Linear(biases=True)` outside the wrapper), **`FullyConnectedTPOp`** (skip_tp), and **`SegmentReduceOp`** (the one op with no irreps semantics, reused for the energy and pair-repulsion reductions).
- **`SphericalHarmonicsOp`** and **`RadialBasisOp`** are **reference-only**: closed-form, cheap, never dispatched today; a backend *may* override them but is not required to. Their authoritative reference implementations live in `mace_torch/backends/reference` (the torch-free *spec* — descriptors, Clebsch-Gordan data, tolerances — lives in `mace_core`), which is what removes the CPU-only e3nn dependency from the hot path. **`sphericart` is the concrete candidate for an *accelerated* `SphericalHarmonicsOp`** (fast, CPU+GPU, differentiable to second order) — evaluate it in ARCH-1/BKD-1; it is already an in-tree dependency via the `[magnetic]` extra (#1244), so adopting it as an optional SH override is cheap. The native plain-torch SH stays the reference.

An optional **span factory** `make_interaction_layer(descriptors) -> InteractionLayerOp | NotImplemented` lets a backend fuse the whole interaction layer (`linear_up → conv → linear → symmetric_contraction → linear_out`, with the residual `skip_tp` reading the *pre-conv* input and added after the contraction) into a single kernel (the direction `cue.SegmentedPolynomial` is headed). The model prefers it when present and falls back to the op-by-op path otherwise, using the same `NotImplemented` negotiation as the per-op fallback.

### 6.2 Capability negotiation and dispatch

The authoritative predicate receives the **full descriptor**: `supports(descriptor) -> bool`. Real constraints depend on the irreps' multiplicity structure, not on `(op, device, dtype, lmax)` — cueq's fused conv fails at runtime on non-uniform-multiplicity irreps (`mace_torchsim.py:126-135` catches this and falls back to the hybrid path), which a coarse predicate cannot express. Coarse `ops`/`dtypes`/`max_lmax` is kept only as a cheap first filter. **`vendor` is not a capability**: a ROCm/HIP backend still declares `device="cuda"`.

Backends register via setuptools entry points, per-framework groups **`mace.kernel_backends.torch`** / **`mace.kernel_backends.jax`** — a third party ships a wheel and needs zero repo edits; an import failure (a missing CUDA library) is recorded, not raised. Dispatch is resolved **once at model build time** and frozen into the module tree; nothing resolves in `forward` (no dtype/device/isinstance checks on the hot path — a requirement for clean compiled graphs). A `CompositeBackend` falls back op-by-op to the reference for any op/dtype/device/irreps the primary declines. The one non-fallback rule: a backend lacking double-backward is **hard-rejected at build time for force/stress training** — a loud error, never silently wrong forces.

### 6.3 Compile-safety and custom kernels

Every hot op is a `torch.library.custom_op` with a `register_fake` (meta) impl, so Dynamo propagates shapes without launching the kernel and the op is a single opaque node — no graph break, no `allow_in_graph` gymnastics. `num_nodes` enters as a **symbolic dimension** (a `SymInt`/tensor dim, never a Python `int` and never `edge_index.max()`), and `register_fake` builds the output at the symbolic size, so dynamic shapes do not trigger a recompile. **Data-dependent host work inside an op body is forbidden by contract** — it silently breaks CUDA-graph capture under `reduce-overhead` — and is caught by `test_cudagraph_safe` (capture + replay with the same shapes and different values). A hand-written kernel that declares `supports_double_backward=True` must supply forward, backward, and backward-of-backward (each itself a differentiable op); a first-order-only kernel declares `supports_double_backward=False` and is a first-class **inference** path. The reference backend publishes the analytic second-derivative descriptors so writing that third kernel is transcription, not research.

### 6.4 Two kinds of layout conversion, and only one is free

**Weights** convert once, at the save/load boundary, and are cached in the op: `to_canonical` / `from_canonical` never run in `forward`. Runtime cost, zero.

**Activations** are a different matter. A `mul_ir ↔ ir_mul` permute of node features is bandwidth-bound and happens *every layer, every step*: measured on CPU with 20 000 nodes, 128 channels and `irreps = 0e+1o+2e`, it costs 4.7 ms and moves 369 MB at fp64 — **30–48 % of the equivariant `Linear` over the same tensor**. MACE defaults to fp64. The legacy hybrid path pays exactly this: `wrapper_ops.py:192-202` runs `layout_transpose_in`/`layout_transpose_out` inside `forward`.

So the activation layout is resolved **once for the whole op chain**, not per op, and features carry it end to end. The accelerated backend's native layout wins; the reference backend adapts, because for plain torch a layout is only an einsum index order and costs nothing — the inverse of today's arrangement. Op-level fallback in `CompositeBackend` is numerically sound but can create a seam between two layouts; where it does, the transpose is explicit, counted, and reported at build time, never inserted silently. `test_no_layout_transpose_in_forward` in the conformance suite enforces it. See RFC-01 §3.4.1.

### 6.5 Canonical weights kill the conversion CLIs

A checkpoint stores **canonical weights + the descriptor spec** (safetensors + a JSON sidecar), never a backend-specific `state_dict` — which also resolves the `TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD` hazard and decouples weights from the LAMMPS jit. `to_canonical`/`from_canonical` run only at the save/load boundary. One checkpoint loads into any backend, so the five cross-backend converters in `mace/cli/` (two installed console scripts + three importable modules) disappear; the layout conversion becomes an internal, build-time step inside whichever backend needs it.

**The canonical parametrization of the symmetric contraction** (RFC-01 §2.4). The `a1 @ pinv(a2)` map in `cg_cueq_tools.py:109` is a change of *parametrization* over the same basis (`poly1`, which "replicates the behavior of the original MACE implementation" at `cg_cueq_tools.py:75`, ↔ cueq's fused `poly2`), and it applies precisely on the reduced-basis path. **The basis is decided — reduced** — and the on-disk form follows from it with no further parametrization choice:

**The defect it must also repair.** Which Clebsch-Gordan basis a model uses is decided today by three independent inputs — the `--use_reduced_cg` flag (`arg_parser.py:235`, default **`False`**), the `MACE_USE_CUEQ_CG` environment variable (`cg.py:23`, consulted when the argument is `None`), and whether `cuequivariance` happens to be importable (`wrapper_ops.py:385`: `use_reduced_cg = use_reduced_cg and CUET_AVAILABLE`) — with signature defaults that disagree along the chain (`models.py:66` → `True`, `symmetric_contraction.py:34` → `False`, `blocks.py:446` → `None`). Two consequences: **by default every MACE model, on every device, is built over the full, redundant basis**; and an explicit `--use_reduced_cg True` on a host without `cuequivariance` is **silently downgraded** to `False` — the user asked for one architecture and trained another. Measured: the symmetric contraction carries 29 parameters on the reduced basis versus 86 on the full one at `lmax=3`, and checkpoints from the two are not interchangeable without a rectangular map. v1 makes the basis explicit model state, with **one** value for every device and every backend.

**Where the backend/model boundary falls.** Layout (`mul_ir` ↔ `ir_mul`) and parametrization (`poly1` ↔ `poly2`) are *bijections*: same dimension, exact both ways, so a backend may hold whichever it likes and convert at the `to_canonical`/`load_canonical` boundary. The Clebsch-Gordan basis is a *surjection* — it changes the trainable parameter count — so it is **model state, not a backend choice**: `ModelConfig.clebsch_gordan_basis`, serialized with the checkpoint. A backend declares which bases it supports and the build fails loudly on a mismatch; it never selects one. The current code already treats it this way — `use_reduced_cg` is passed to both implementations, and cueq accepts either basis via `original_mace` (`wrapper_ops.py:399`) — which is why line 385 is a defect rather than a design. See RFC-01 §2.4.3.

**Decided: the reduced basis.** `ModelConfig.clebsch_gordan_basis = "reduced"` is the only value for new training. Both the reduced basis and the `full ↔ reduced` conversion are computed **from first principles (Clebsch–Gordan / Wigner) in pure numpy — no e3nn, no cuequivariance** (elementary group theory; neither library is required, and the caveat that the serializer must store the zeroed paths — `symmetric_contraction.py:218-233` — or record the mask in the descriptor stands regardless). The reduced basis is 2.05× faster forward, 1.80× with backward, carries 54–66 % fewer contraction parameters, and gives identical predictions.

**A model may be stored in full basis** — a v0.3 foundation model, or an external artifact. Loading it performs the exact `full → reduced` conversion **at load time** (explicit, function-preserving, fp64), so runtime is always reduced; the persisted v1 checkpoint is reduced. `reduced → full` exists only as an explicit export for external full-basis consumers (Symmetrix/libmace, DEP-3), never at load. **CORE-4** stays ahead of BKD-1 because it **pins the path order and the per-path normalization — that is the on-disk weight format** — and golden-locks the tensors, not the path counts.

The **on-disk layout is not a decision**: measured across the production grid, the `poly1↔poly2` map is a **permutation times a diagonal** — one non-zero per row and column, scales that are square roots of small integers — and the nested tensors are contiguous slices of a flat `[Z, A, mul]` tensor (`convert_e3nn_cueq.py:117-127` concatenates before projecting). The canonical form is that flat tensor. The reference `cat`s and `split`s and keeps its Horner cascade; the cueq backend permutes and rescales once at build time. `mace_core` needs no `SegmentedTensorProduct` machinery and no `SegmentedPolynomial` evaluator. If cueq changes its internal segmentation, only its index buffer changes: **MACE owns its format**. See RFC-01 §2.4.1 and §2.4.5.

### 6.6 Conformance suite

`run_backend_conformance(backend_name, …)` is a parametrized suite any backend runs over itself with one flag: `test_equivariance`, `test_parity_vs_reference`, `test_gradcheck`/`test_gradgradcheck` (the double-backward gate), `test_weight_roundtrip` (cross-backend, the executable proof the converters are unnecessary), `test_compile_no_graph_break` under `torch.compile(fullgraph=True)`, `test_cudagraph_safe`, `test_no_layout_transpose_in_forward`, and `test_capabilities_honest`. Its `ToleranceTable` is edit-locked. The **out-of-tree example backend (BKD-4)** runs the full suite in CI as a **permanent extensibility gate** that exposes exactly what cueq/oeq mask: correct double-backward and a fullgraph-clean compile with zero core edits.

## 7. Risk register

| Risk | Mitigation |
|------|------------|
| **`develop` and `mace-reforge` diverge faster than the migration can absorb.** Every fix and self-contained feature that lands on `develop` is work the migration pass owes v1, and the debt grows for the length of the rewrite. | `CONTRIBUTING.md` restricts what `develop` accepts to exactly what can be carried: bug fixes, and self-contained features whose tests pin numbers a port can be checked against. Core and shared changes are refused against v0.3 and go straight to v1, so the divergence stays in additive, portable pieces. A fix that changes numbers also changes the goldens it is pinned by, so the migration regenerates those in its own PR. |
| **The retirement stalls and the dual toolchain becomes permanent.** | Three gates, none of which depends on anyone remembering: a fitness test that goes **red** when a debt-book date expires; a CI check requiring `git ls-files mace/` to be strictly non-increasing from Phase 3 onward; and the hard 1.0 release gate on `git ls-files mace/ == 0`. |
| **Path-scoped dual toolchain/CI during the overlap** (`mace/**` → black/isort/pylint/mypy; `packages/**` → ruff/ty/prek). | Accepted; the only recurring, non-self-inflicted cost. Held in check by 1:1 entry-point ownership and the anti-structural meta-lint. |
| **Global process-state contaminates the in-process oracle** — `mace/__init__` sets `TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD`; `e3nn.set_optimization_defaults` is process-wide; default dtype/RNG are global. | The parity harness snapshots and restores **all** global state between runs (env vars, default dtype, e3nn optimization defaults, torch/np RNG), or forks a subprocess and diffs serialized outputs. |
| **fp32 GPU non-determinism invisible to an fp64 CPU net** — `scatter_add_` is order-nondeterministic on CUDA, so ~1e-4 fp32 differences hide under fp64 tolerance. | A frozen **fp32 bit-reproduction golden** with a deterministic `segment_reduce`; FM release is gated on max-abs-error vs that golden at fp32 machine epsilon, not a loose tolerance. |
| **Training-numerics gap** — a comparison against a JSON cannot see a gradient bug the size of init noise. | A **one-step, init-agnostic gradient parity test** (per-PR): byte-identical weights in both stacks, one forward+backward on an E+F loss, `d(loss)/d(params)` allclose at fp64 — exercising the `create_graph=True` path — plus a model-level gradgrad/finite-diff of the force+stress loss on the tiny fixture. |
| **Internally inconsistent legacy energy** — `models.py:581` computes `total_energy = e0 + inter_e` in the model dtype but `node_energy` via an unconditional `.double()`; no single accumulate dtype matches both. | Both quantities are pinned as **separate goldens** (total in model dtype, node in fp64), in **fp32 and fp64**, for both `MACE` and `ScaleShiftMACE` (which further differ in whether the ZBL term is inside or outside the scale-shift). v1 replicates precision per-quantity via `PrecisionConfig.reduction.accumulate_dtype`, not a single global policy. |
| **Console-script ownership collision** — two distributions declaring the same `console_scripts` is undefined in pip. | Exactly one distribution provides each `mace_*` entry point: `console_scripts` are removed from the root `setup.cfg`, `mace-launcher` is the sole owner, and `packages/mace-torch` ships under a distinct distribution name while keeping the `mace_torch` import name. Asserted in CI. |
| **No v1 deployment path until DEP-2 lands** — v1 blocks are born without `@compile_mode`, so `jit.compile`/`jit.save` (`create_lammps_model.py:109`, `run_train.py:1145,1160`) cannot script them. | The `--engine=v1` opt-in for the energy capability **explicitly excludes** LAMMPS export/checkpoint until DEP-2 (`torch.export`); deployment stays legacy-only during the window; recorded as a dated burn-step in the debt book. |
| **Flat-dict couples `torch_geometric` semantics** — several forwards read `num_graphs = data['ptr'].numel()-1`; `ptr`/`batch` are `Batch` collation artifacts. | `ptr`, `batch`, `shifts`, `unit_shifts` are explicit fields of RFC-B's flat-dict contract with a schema validator, plus a golden verifying the v1 collater reproduces `ptr`/`batch` bit-exact. |
| **CORE-4 is the central mathematical risk** and sits on the critical path. | It is bounded (numpy, self-contained; compute Clebsch–Gordan / Wigner and the reduced basis from first principles — e3nn's MIT routine is one porting reference, but the result depends on **no** runtime library). Golden-locked against an **independent** CPU Clebsch-Gordan derivation — never against itself — reviewed by someone with group-theory expertise, with the cueq round-trip as best-effort nightly confirmation only. `mace_core` depends on **neither e3nn nor cuequivariance** at any point (no interim dependency). |
| **CORE-4 pins the on-disk weight format.** Not a vendor-algorithm port — `e3nn.o3.ReducedTensorProducts` computes the reduced basis (path counts verified in 21 cases; span verified including `l>0`; `correlation=4` verified). The residual risk is that the **path order and per-path normalization**, which MACE must now invent, are got wrong. | Golden-lock the **tensors**, not the path counts. Review by someone with group-theory/e3nn expertise. Bounded: correcting it later is a `CanonicalSpec.schema_version` bump with an in-repo migration, possible because `full ↔ reduced` is exact in both directions. `wrapper_ops.py:385` and `cg.py:119-126`, which silently substitute the full basis, are bugs to be removed. |
| **The Clebsch-Gordan basis is decided by three uncoordinated inputs** — a CLI flag defaulting to `False`, an environment variable, and whether `cuequivariance` is importable (`wrapper_ops.py:385`) — so every model defaults to a ~3× over-parametrized symmetric contraction, and an explicit `--use_reduced_cg True` is silently downgraded on a host without cueq. | The canonical-basis decision (§6.5) removes the branch: one basis on every device and backend. A Phase 0 golden pins the parameter count per `(irreps, correlation)` so the coupling cannot reappear. |
| **The parity harness is a substantial, permanent-until-retirement deliverable** with hand-tuned fp64 tolerances. | Tolerances edit-locked (R5); property/fuzz inputs run nightly (PAR-2), not just tiny goldens. |
| **torch.compile instability upstream.** | The eager reference path is authoritative; the compiled path is an optimization guarded by a parity smoke, never the only path. |

## 8. Open questions from the roadmap, and how this plan resolves them

1. **"Remove e3nn" vs the CPU-required test matrix** (Ch. 11). cueq is CUDA-only. Resolution (team decision, tightening Ch. 5's "remove **or reduce**" to **remove entirely**): **e3nn and e3nn-jax are removed from the whole project — not used anywhere, not even as a backend.** The plain-PyTorch reference backend is mandatory and native (TP, linear, **gated nonlinearity**, symmetric contraction, spherical harmonics, radial — every e3nn-provided block reimplemented in plain torch, differentiable to second order); `mace_core` computes the Clebsch-Gordan/reduced basis from first principles (numpy, no e3nn, no cueq); mace-jax mirrors torch (native jax reference backend, `cuequivariance_jax` an optional backend). cueq/oeq stay as optional **acceleration** backends and a best-effort nightly oracle — never a requirement for the basis. No transitional e3nn backend.
2. **`BaseMACE` "no gradients" vs stress-via-strain.** The strain displacement is injected *before* edge vectors are computed (`prepare_graph`), so the derivative engine is not purely downstream of the backbone. ARCH-4 carries a design note reviewed before implementation.
3. **Principle 4 "foundation models remain loadable" vs D2.** Loadable = one-shot convertible; REL-1 states this explicitly.
4. **JAX CI burden.** Ch. 11 marks JAX "Required" on both vendors while Ch. 3 makes mace-jax minimal. RFC-D sets the model list and CI tier (recommendation: required NVIDIA, best-effort AMD for v1.0.0).
5. **"Electrostatics" conflates two things.** PolarMACE lives in-repo (`mace/modules/extensions.py`) and is golden-gated (ELEC-2); graph_longrange is external and can slip to beta (ELEC-1).
6. **E0s "readout-only" vs current ScaleShiftMACE semantics** — a real numerical-equivalence question for the converter. RFC-C resolves it before ARCH-3; FM-00 enforces it.
7. **Tooling schism.** Ch. 11 mandates ruff+ty+prek; the legacy tree uses black/isort/pylint/mypy. Resolved by the path-scoped dual toolchain during the overlap (R1/INF-4): `mace/**` keeps the frozen legacy toolchain, `packages/**` runs ruff+ty+prek; no mixed state on one path. The schism ends when RET-* removes `mace/`.
8. **External dependencies with no code owner:** fine-tuning data hosting, DOI repository choice, MPI GPU-node security hardening, steering-committee funding, and the GPU-fleet gaps (§3). Tracked here as external dependencies needing named human owners before FT-2, FM-3, and GOV work can finish.
9. **Metadata scope ambiguity** ("at minimum foundation models, possibly all"). Decided: mandatory for **all** models — nearly free once CORE-2/CLI-1 exist.
10. **v1 naming.** The import names `mace_core`/`mace_torch`/`mace_jax` never collide with `mace` by construction; the distribution name for the new torch package is distinct (§7, console-script row). Decided and not revisited.

## 9. Traceability

| Roadmap chapter/axis | Plan section / tickets |
|---|---|
| Ch. 3 package structure | §2, [`target_layout.md`](target_layout.md), INF-* |
| Ch. 4 model architecture (Axis A) | ARCH-*, CORE-1 |
| Ch. 5 backend/perf (B, I, K) | §6, BKD-*, RFC-A |
| Ch. 6 training/loss (C, G) | TRN-*, CFG-1 |
| Ch. 7 electrostatics (L) | ELEC-1/2 |
| Ch. 8 data pipeline (E) | DATA-*, RFC-B |
| Ch. 9 config/CLI/fine-tuning/FM usability (D, F, O) | CORE-2, CFG-1, CLI-1, FT-*, FM-3, RFC-C |
| Ch. 10 deployment (H) | DEP-*, RFC-D |
| Ch. 11 testing/CI (M) | Phase 0, P0-*, PAR-*, FM-3 (badge matrix) |
| Ch. 12 educational reference | EDU-1/2 |
| Ch. 13 community/governance (N) | GOV-1/2 |
| Ch. 14–15 timeline/dependencies | §4–§5 |
| Ch. 16 risks | §7 |
| Appendix A open questions | RFC docs 01…08 via tickets RFC-A…D |
| Appendix B CLI migration map | CLI-1, REL-1 |

## 10. Ticket index

Every ticket ID in this document is a GitHub issue in `ACEsuit/mace`, tracked on the [board](https://github.com/orgs/ACEsuit/projects/2). Phase 0 and the RFC-track tickets are not on the board: Phase 0 characterized the frozen legacy before the plan opened, and the RFC decisions are inlined into the tickets they block.

| Family | Tickets |
|---|---|
| **ARCH** | [ARCH-1](https://github.com/ACEsuit/mace/issues/1559) · [ARCH-2](https://github.com/ACEsuit/mace/issues/1561) · [ARCH-3](https://github.com/ACEsuit/mace/issues/1562) · [ARCH-4](https://github.com/ACEsuit/mace/issues/1563) · [ARCH-5](https://github.com/ACEsuit/mace/issues/1633) |
| **BKD** | [BKD-1](https://github.com/ACEsuit/mace/issues/1560) · [BKD-1a](https://github.com/ACEsuit/mace/issues/1564) · [BKD-2](https://github.com/ACEsuit/mace/issues/1566) · [BKD-3](https://github.com/ACEsuit/mace/issues/1567) · [BKD-4](https://github.com/ACEsuit/mace/issues/1568) |
| **CFG** | [CFG-1](https://github.com/ACEsuit/mace/issues/1574) |
| **CLI** | [CLI-1](https://github.com/ACEsuit/mace/issues/1579) |
| **CORE** | [CORE-1](https://github.com/ACEsuit/mace/issues/1555) · [CORE-2](https://github.com/ACEsuit/mace/issues/1556) · [CORE-3](https://github.com/ACEsuit/mace/issues/1557) · [CORE-4](https://github.com/ACEsuit/mace/issues/1558) |
| **DATA** | [DATA-1](https://github.com/ACEsuit/mace/issues/1569) · [DATA-2](https://github.com/ACEsuit/mace/issues/1570) · [DATA-3](https://github.com/ACEsuit/mace/issues/1571) |
| **DEP** | [DEP-1](https://github.com/ACEsuit/mace/issues/1583) · [DEP-1a](https://github.com/ACEsuit/mace/issues/1634) · [DEP-2](https://github.com/ACEsuit/mace/issues/1594) · [DEP-3](https://github.com/ACEsuit/mace/issues/1595) · [DEP-4](https://github.com/ACEsuit/mace/issues/1596) |
| **EDU** | [EDU-1](https://github.com/ACEsuit/mace/issues/1607) · [EDU-2](https://github.com/ACEsuit/mace/issues/1608) |
| **ELEC** | [ELEC-1](https://github.com/ACEsuit/mace/issues/1591) · [ELEC-2](https://github.com/ACEsuit/mace/issues/1592) · [ELEC-3](https://github.com/ACEsuit/mace/issues/1593) · [ELEC-4](https://github.com/ACEsuit/mace/issues/1635) |
| **FM** | [FM-00](https://github.com/ACEsuit/mace/issues/1565) · [FM-1](https://github.com/ACEsuit/mace/issues/1586) · [FM-2](https://github.com/ACEsuit/mace/issues/1587) · [FM-3](https://github.com/ACEsuit/mace/issues/1588) · [FM-4](https://github.com/ACEsuit/mace/issues/1636) |
| **FT** | [FT-1](https://github.com/ACEsuit/mace/issues/1580) · [FT-2](https://github.com/ACEsuit/mace/issues/1581) · [FT-3](https://github.com/ACEsuit/mace/issues/1582) · [FT-4](https://github.com/ACEsuit/mace/issues/1637) |
| **GATE** | [GATE-3](https://github.com/ACEsuit/mace/issues/1585) |
| **GOV** | [GOV-1](https://github.com/ACEsuit/mace/issues/1609) · [GOV-2a](https://github.com/ACEsuit/mace/issues/1638) · [GOV-2b](https://github.com/ACEsuit/mace/issues/1610) |
| **INF** | [INF-1](https://github.com/ACEsuit/mace/issues/1550) · [INF-2](https://github.com/ACEsuit/mace/issues/1551) · [INF-3](https://github.com/ACEsuit/mace/issues/1552) · [INF-4](https://github.com/ACEsuit/mace/issues/1553) · [INF-5](https://github.com/ACEsuit/mace/issues/1554) |
| **JAX** | [JAX-1](https://github.com/ACEsuit/mace/issues/1589) · [JAX-2](https://github.com/ACEsuit/mace/issues/1590) |
| **MAG** | [MAG-1](https://github.com/ACEsuit/mace/issues/1584) |
| **PAR** | [PAR-1](https://github.com/ACEsuit/mace/issues/1572) · [PAR-2](https://github.com/ACEsuit/mace/issues/1573) |
| **REL** | [REL-1](https://github.com/ACEsuit/mace/issues/1603) · [REL-2](https://github.com/ACEsuit/mace/issues/1604) · [REL-3](https://github.com/ACEsuit/mace/issues/1605) · [REL-4](https://github.com/ACEsuit/mace/issues/1606) |
| **RET** | [RET-1](https://github.com/ACEsuit/mace/issues/1597) · [RET-2](https://github.com/ACEsuit/mace/issues/1598) · [RET-3](https://github.com/ACEsuit/mace/issues/1599) · [RET-4](https://github.com/ACEsuit/mace/issues/1600) · [RET-5](https://github.com/ACEsuit/mace/issues/1601) · [RET-6](https://github.com/ACEsuit/mace/issues/1602) |
| **TRN** | [TRN-1](https://github.com/ACEsuit/mace/issues/1575) · [TRN-2](https://github.com/ACEsuit/mace/issues/1576) · [TRN-3](https://github.com/ACEsuit/mace/issues/1577) · [TRN-4](https://github.com/ACEsuit/mace/issues/1578) · [TRN-5](https://github.com/ACEsuit/mace/issues/1639) |
