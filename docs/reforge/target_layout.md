# MACE v1 final layout: three packages, one-way dependency, registry-based extensibility

## 0. Scope and guiding principle

This is the **destination** (not the path). It covers everything `mace/` does today (75 modules) regrouped into three packages with a **strict one-way dependency direction**. The rule that shapes the entire tree:

```
mace_jax ─┐
          ├──▶ mace_core  ◀── mace_torch
          (mace_core does not import torch, jax, e3nn, or matscipy-with-torch)
```

`mace_core` is the shared contract and pure math (numpy). `mace_torch` and `mace_jax` are *implementations* of the same contract. Neither imports the other. Legacy `mace/` is unreachable from all three. The glue lives in `mace-launcher` (entry points) and `tests/parity/` (oracle), both on the double-import allowlist.

---

## 1. Final directory tree (file-level)

> Paths below use the `packages/` container — the name in force through the entire migration
> (Milestones A–B), while the frozen legacy `mace/` still lives at the root. The optional final
> step **REL-4** renames `packages/ → mace/` once `git ls-files mace/ == 0` (§5, Milestone C);
> import names (`mace_core`/`mace_torch`/`mace_jax`) are unaffected either way.

### 1.1 `packages/mace-core/` — contract + pure math, **no torch/jax/e3nn**

```
packages/mace-core/
├── pyproject.toml                      # import name `mace_core`; deps: numpy, pydantic, matscipy; NO torch
├── src/mace_core/
│   ├── __init__.py                     # re-exports public types/config/registries; does NOT import heavy submodules
│   ├── _version.py                     # contract version (semver of the weights format/spec)
│   │
│   ├── types.py                        # MACEOutputs (typed dataclass, replaces the get_outputs dict); GRAPH_SCHEMA + GraphInfo/GraphView (rfc-03 flat-dict contract)
│   ├── graph.py                        # flat-dict contract {node_attrs,edge_index,positions,batch,cell,shifts,...} + shape/dtype validation
│   │
│   ├── config/
│   │   ├── __init__.py                 # ModelConfig.from_namespace(args) shim from legacy argparse → Pydantic
│   │   ├── model.py                    # ModelConfig Pydantic (replaces the ~100 args.* of configure_model)
│   │   ├── precision.py                # PrecisionConfig (default_dtype, float64 accumulation, tf32) — replaces the scattered .double()
│   │   ├── training.py                 # TrainConfig, StageTwoConfig (ex-SWA), EMAConfig, OptimizerConfig, SchedulerConfig
│   │   ├── data.py                     # DataConfig (E0s: average/estimated/dict), heads, cutoff, num_workers
│   │   ├── loss.py                     # LossConfig (energy/forces/stress/virials/dipole weights)
│   │   └── backend.py                  # BackendConfig (kernel backend name + options; replaces CuEquivarianceConfig/OEQConfig)
│   │
│   ├── elements/
│   │   ├── number_table.py             # AtomicNumberTable (reimplemented; mirror of tools/utils.py, without dragging in train.py)
│   │   └── default_keys.py             # DefaultKeys (reimplemented; mirror of tools/default_keys.py)
│   │
│   ├── observables/
│   │   ├── __init__.py                 # OBSERVABLE_REGISTRY (declarative: name → spec)
│   │   ├── base.py                     # Observable protocol: output irreps, how it's derived (readout | autograd | grad-strain)
│   │   ├── energy.py                   # Energy, SiteEnergy (readout+scatter_sum)
│   │   ├── forces.py                   # Forces (=-dE/dx via autograd) — spec only, the physics is executed by mace_torch/mace_jax
│   │   ├── stress.py                   # Stress, Virials (grad w.r.t. strain; sign convention PINNED here)
│   │   ├── dipole.py                   # Dipole, AtomicDipole
│   │   ├── polarizability.py           # Polarizability (dielectric/polar)
│   │   └── hessian.py                  # Hessian (second derivative)
│   │
│   ├── kernels/
│   │   ├── protocol.py                 # KernelBackend Protocol, generic over TensorT: make_* factories + capabilities (§3.1)
│   │   ├── precision.py                # Precision = Literal["float64","float32",…] — dtype NAMES, never torch.dtype
│   │   ├── descriptors.py              # IrrepsDescriptor, TPDescriptor, ContractionDescriptor (irreps+instructions, no weights)
│   │   ├── registry.py                 # get_backend(name, framework) via entry_points('mace.kernel_backends.{torch,jax}')
│   │   └── canonical.py               # canonical weight layout + to_canonical/load_canonical spec (kills the 5 convert_* modules)
│   │
│   ├── clebsch_gordan/
│   │   ├── coefficients.py             # Clebsch-Gordan coefficients / Wigner (mirror of tools/cg.py: U_matrix_real, pure numpy)
│   │   ├── reduced_basis.py            # reduced symmetric TP basis + the PINNED path order and normalization (the file format, RFC-01 §2.4.1)
│   │   └── irreps.py                   # e3nn-independent irreps utilities (parse, dim, mul_ir/ir_mul layout)
│   │
│   ├── neighbors/
│   │   └── neighborhood.py             # matscipy neighbor construction (mirror of data/neighborhood.py; numpy, no torch)
│   │
│   ├── data_spec/
│   │   ├── configuration.py            # Configuration (raw parsed structure; mirror of data/utils.py without torch)
│   │   ├── xyz.py                       # XYZ / extxyz parser → Configuration (E0s, energy/forces/stress keys)
│   │   ├── statistics.py               # avg_num_neighbors, energy-forces mean/std; statistics.json format
│   │   └── shard_format.py             # HDF5/LMDB shard spec (schema, not torch I/O)
│   │
│   ├── weights/
│   │   ├── neutral_format.py           # neutral safetensors + JSON format (stable spec; basis for the FM converter)
│   │   └── extract_spec.py             # declarative description of which weights each model has (replaces extract_config_mace_model via reflection)
│   │
│   └── registries.py                   # string→spec registries: MODEL_REGISTRY, READOUT_REGISTRY, INTERACTION_REGISTRY, LOSS_REGISTRY, TRANSFORM_REGISTRY, DATASET_REGISTRY, SCALING_REGISTRY (specs/names only; torch classes are registered in mace_torch)
│
└── tests/                              # pure core tests (no torch) — see §4.1
    ├── test_clebsch_gordan.py          # golden Clebsch-Gordan vs hand-computed values (P0-6 fragments)
    ├── test_reduced_basis.py           # golden-locks the basis TENSORS, order and scales (not just path counts)
    ├── test_neighborhood.py            # P0-7 neighbors/batching
    ├── test_config_from_namespace.py   # ModelConfig.from_namespace round-trip of all legacy args
    ├── test_observable_registry.py     # each observable declares consistent irreps/derivation
    └── test_canonical_layout.py        # to_canonical/load_canonical round-trip (golden)
```

`clebsch_gordan/reduced_basis.py` computes the reduced symmetric tensor-product basis and — the numerics-critical part — **pins its path order and per-path normalization**, which *is* the on-disk weight format (RFC-01 §2.4.1). The basis is computed **from first principles (Clebsch–Gordan / Wigner) in pure numpy — no e3nn, no cuequivariance** (both are elementary group theory; neither library is required). The `full ↔ reduced` conversion is likewise native (full→reduced exact, reduced→full min-norm), so **loading a full-basis artifact converts it to reduced at load time** with no accelerated dependency. No `SegmentedPolynomial` evaluator is needed: the reference keeps a Horner cascade, and the cueq backend applies its own scaled-permutation at its boundary (§1.2, `backends/cueq/canonical.py`).

### 1.2 `packages/mace-torch/` — models, training, backends, deployment

```
packages/mace-torch/
├── pyproject.toml                      # import name `mace_torch`; deps: torch, ase, mace_core; NO e3nn (removed entirely); CLI and kernel_backends entry_points
├── src/mace_torch/
│   ├── __init__.py                     # public API: MACE model factory, Calculator; does NOT set TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD globally
│   │
│   ├── graph/
│   │   ├── batch.py                    # torch tensorization of the rfc-03 flat graph dict; replaces AtomicData+torch_geometric in the model INTERFACE
│   │   └── collate.py                  # collation of configurations → flat graph dict (no torch_geometric.data.Data)
│   │
│   ├── kernels/
│   │   ├── dtypes.py                   # binds mace_core's Precision names to torch.dtype — the framework seam
│   │   ├── ops.py                      # the 3 hot ops as torch.library.custom_op: channelwise_tp_conv, symmetric_contraction, segment_reduce
│   │   ├── fake.py                     # register_fake (shape-only meta impls) → torch.compile with no graph-break
│   │   └── autograd_ref.py             # reference autograd.Function (double-backward) for the 3 hot ops
│   │
│   ├── backends/
│   │   ├── reference/
│   │   │   ├── backend.py              # ReferenceBackend: implements KernelBackend in plain torch — NO e3nn (native TP/linear/contraction; the correctness oracle, CPU-ok)
│   │   │   └── sph_harm.py             # spherical harmonics + reference radial basis (non-hot)
│   │   ├── cueq/
│   │   │   ├── backend.py              # CuEqBackend: wraps cuequivariance behind the Protocol (kills the types.MethodType monkeypatch)
│   │   │   └── canonical.py            # to_canonical/load_canonical ONLY at the boundary (mul_ir↔ir_mul reshape; the canonical→fused-basis map is a scaled permutation applied once at build time, rfc-01 §2.4.1)
│   │   ├── oeq/
│   │   │   ├── backend.py              # OEQBackend: wraps OpenEquivariance behind the Protocol
│   │   │   └── canonical.py            # oeq layout boundary
│   │   └── example/
│   │       └── backend.py             # EXAMPLE backend, out-of-tree-shaped (permanent extensibility gate; zero core edits)
│   │
│   ├── nn/                             # blocks WITHOUT @compile_mode('script') from day 1
│   │   ├── embedding.py                # LinearNodeEmbeddingBlock, RadialEmbeddingBlock
│   │   ├── radial.py                   # BesselBasis, ChebychevBasis, GaussianBasis, PolynomialCutoff, ZBLBasis, AgnesiTransform, SoftTransform, RadialMLP
│   │   ├── interaction.py              # the 6 RealAgnostic*InteractionBlock (fusion-agnostic forward: calls channelwise_tp_conv)
│   │   ├── product_basis.py            # EquivariantProductBasisBlock (uses the symmetric_contraction op; no layout torch.transpose)
│   │   ├── symmetric_contraction.py    # SymmetricContraction as a native op (no fx.symbolic_trace, no CodeGenMixin)
│   │   ├── readout.py                  # LinearReadoutBlock, NonLinearReadoutBlock, NonLinearBiasReadoutBlock, GeneralNonLinearBiasReadoutBlock, LinearDipoleReadoutBlock, NonLinearDipoleReadoutBlock
│   │   ├── scale_shift.py              # ScaleShiftBlock (E0-after-shift semantics PINNED per-model)
│   │   ├── gate.py                     # equivariant gates (mirror of modules/gate.py)
│   │   ├── field.py                    # field_blocks (LES/external field)
│   │   ├── irreps_layout.py            # reshape_irreps/transpose layout (only called by the backend boundary, not the forward)
│   │   └── lora.py                     # LoRA adapters for fine-tuning
│   │
│   ├── models/
│   │   ├── base.py                     # two-layer BaseMACE: forward→MACEOutputs; built from ModelConfig; declarative observables
│   │   ├── energy.py                   # MACE, ScaleShiftMACE
│   │   ├── dipole.py                   # AtomicDipolesMACE, EnergyDipolesMACE
│   │   ├── dielectric.py               # AtomicDielectricMACE
│   │   ├── electrostatics.py           # PolarMACE (ex-extensions.py; electrostatics/LES)
│   │   └── build.py                    # configure_model equivalent: ModelConfig + statistics → model instance
│   │
│   ├── physics/
│   │   └── outputs.py                  # get_outputs torch: forces via autograd, virial/stress via strain; displacement injection (prepare_graph)
│   │
│   ├── data/
│   │   ├── dataset_xyz.py              # in-memory dataset from XYZ → flat graph dicts
│   │   ├── hdf5_dataset.py             # on-line HDF5 shards
│   │   ├── lmdb_dataset.py             # on-line LMDB shards
│   │   ├── fairchem_dataset.py         # fairchem-style loader (ex-tools/fairchem_dataset)
│   │   ├── padding.py                  # padding_tools (fixed-size batches for compile/LAMMPS)
│   │   ├── dataloader.py               # DataLoader (own collation; no vendored tools/torch_geometric)
│   │   └── distributed_sampler.py      # multi-GPU DDP sampler
│   │
│   ├── train/
│   │   ├── loop.py                     # train()/evaluate() by stage (stage-two ex-SWA)
│   │   ├── ema.py                      # EMA
│   │   ├── metrics.py                  # MACELoss torchmetric
│   │   ├── loss.py                     # WeightedEnergyForces(Stress/Virials/Dipole/Huber/L1L2)Loss, UniversalLoss, DipolePolarLoss (registered in LOSS_REGISTRY)
│   │   ├── checkpoint.py               # checkpoint I/O (canonical tensors: safetensors + JSON sidecar per rfc-01 §2.1 — no pickle, NO jit.save)
│   │   ├── schedulers.py               # LR schedulers + schedulefree
│   │   ├── stage_two.py                # stage-two switch (start_stage_two)
│   │   ├── ddp.py                      # distributed_tools (CPU/GPU DDP setup)
│   │   └── slurm.py                    # slurm_distributed
│   │
│   ├── finetune/
│   │   ├── foundations.py              # load_foundations / load_foundations_elements (copies compatible weights)
│   │   ├── multihead.py                # HeadConfig, prepare_default_head, prepare_pt_head, multihead_tools
│   │   ├── replay.py                   # assemble_replay_data, pt_head replay
│   │   ├── pseudolabels.py             # generate_pseudolabels_for_configs
│   │   └── select_subset.py            # fine_tuning_select (farthest-point sampling / fpsample)
│   │
│   ├── calculators/
│   │   ├── ase_calculator.py           # ASE Calculator (ex-calculators/mace.py)
│   │   ├── foundations.py              # mace_mp / mace_off / mace_polar (download+cache; ~/.cache/mace)
│   │   ├── torchsim.py                 # backend torchsim
│   │   ├── lammps.py                   # LAMMPS runtime adapter: translates LAMMPS real/ghost partitioning into the rfc-03 graph contract (LocalityInfo) — NO lammps_class/lammps_natoms branches in the model (rfc-01 §6b.1)
│   │   └── lammps_mliap.py             # ML-IAP runtime adapter (the Python object LAMMPS calls; boundary-only)
│   │
│   ├── deploy/
│   │   ├── export.py                   # torch.export/AOTInductor for LAMMPS (default + mliap); replaces jit.compile
│   │   ├── export_adapter.py           # EMERGENCY ESCAPE HATCH: the one scriptable module isolated in case torch.export slips
│   │   ├── select_head.py              # select_head for multihead export
│   │   └── neutral_io.py               # save/load to mace_core.weights' neutral safetensors+JSON format
│   │
│   └── cli/
│       ├── run_train.py                # training main (thin adapter: parse args → ModelConfig → train.loop)
│       ├── eval_configs.py             # eval over XYZ
│       ├── preprocess_data.py          # mace_prepare_data (shards + statistics.json)
│       ├── create_lammps_model.py      # LAMMPS export (uses deploy/export.py)
│       ├── select_head.py              # mace_select_head
│       ├── convert_device.py           # mace_convert_device (still useful: cpu↔cuda)
│       ├── fine_tuning_select.py       # mace_finetuning_select
│       ├── active_learning_md.py       # active learning MD
│       ├── plot_train.py               # training plots (ex-plot_train/visualise_train)
│       └── arg_parser.py               # argparse parser (source of flags; delegates to ModelConfig/TrainConfig)
│
└── tests/                              # see §4 (backend conformance, parity, equivariance, E2E)
    ├── unit/                           # value-tests ported from the ~18 legacy white-box tests
    ├── backends/                       # KernelBackend conformance + backend-vs-reference parity
    ├── workflows/                      # E2E CLI (subprocess), auto-slow
    ├── extensions/{polar,les,torchsim,schedulefree}/
    ├── foundations/                    # network-marked
    └── integrations/lammps/            # contract + bin_lammps
```

### 1.3 `packages/mace-jax/` — parallel destination, off the critical path

```
packages/mace-jax/
├── pyproject.toml                      # import name `mace_jax`; deps: jax, mace_core; NO e3nn-jax (removed entirely); cuequivariance_jax is an OPTIONAL backend, never a base dep; NEVER imports mace_torch
├── src/mace_jax/
│   ├── __init__.py
│   ├── graph.py                        # flat-dict graph contract bound to jax.Array (same schema as mace_core.graph, rfc-03)
│   ├── kernels/
│   │   ├── ops.py                      # jax primitives (jax.lax scatter/segment_sum, NATIVE tp/contraction over mace_core's irreps/Clebsch-Gordan — no e3nn-jax)
│   │   └── reference.py               # jax reference backend
│   ├── nn/                             # equivalent blocks in flax/haiku-like
│   │   ├── interaction.py
│   │   ├── product_basis.py
│   │   ├── symmetric_contraction.py
│   │   ├── readout.py
│   │   └── radial.py
│   ├── models/
│   │   ├── base.py                     # BaseMACE jax → MACEOutputs
│   │   └── energy.py                   # MACE / ScaleShiftMACE jax
│   ├── physics/outputs.py              # forces/stress via jax.grad (first derivatives only — inference-only per D1)
│   ├── data/dataset.py                 # jax loader
│   ├── deploy/neutral_io.py            # loads from the neutral safetensors+JSON format
│   └── cli/eval.py                     # CLI jax, inference-only (validated against goldens + the neutral format, NOT against the legacy E2E harness). NO training loop / run_train here: mace-jax is inference-only (plan D1, rfc-02 §1.1). There is no JAX training anywhere — the educational `tutorials/` (EDU-1) is evaluation/inspection-only
└── tests/
    ├── test_parity_neutral.py          # v1-jax loaded from the neutral format == fp64 goldens
    └── test_equivariance.py            # jax equivariance
```

### 1.4 `packages/mace-launcher/` — the glue (double-import allowlist)

```
packages/mace-launcher/
├── pyproject.toml                      # OWNS all console entry points (mace_run_train, mace_eval_configs, mace_prepare_data, mace_create_lammps_model, mace_select_head, mace_convert_device, mace_finetuning_select)
└── src/mace_launcher/
    ├── __init__.py
    ├── dispatch.py                     # ~50 LOC: --engine {legacy,v1} / MACE_ENGINE → mace.cli.* | mace_torch.cli.*
    └── audit.py                        # sys.addaudithook: fails if it detects an importlib/entry-point reach-in from packages→mace
```

**What it does.** During coexistence two full stacks are installed side by side — the frozen legacy
`mace/` and the new `mace_torch`/`mace_core` — and the hard rule is that they **never import each
other** (`packages ⊥ mace`). The launcher is the *single operational meeting point*, one of only two
modules on the double-import allowlist (the other is `tests/parity/`). It is deliberately tiny
(~50 LOC):

- **Owns every console script.** The root `setup.cfg` `console_scripts` are removed and re-pointed
  at `mace_launcher:*`, so exactly one installed distribution provides each `mace_*` command (two
  distributions declaring the same script is undefined in pip). Asserted in CI.
- **Dispatches on `--engine {legacy,v1}` / `MACE_ENGINE`**, default **`legacy`** → day-one behaviour
  is byte-identical to calling `mace.cli.*` directly. The flag is consumed *before* the target's
  argparse; `legacy` calls `mace.cli.<mod>:main`, `v1` calls `mace_torch.cli.<mod>:main`. `--engine v1`
  on a not-yet-migrated capability raises a clear "capability X not yet available on the v1 engine".
- **Closes the dynamic reach-in.** `audit.py` registers a `sys.addaudithook` that fails if anything
  under `packages/` dynamically imports `mace.*` in v1 mode — the runtime backstop behind
  import-linter's static check (INF-3).

**It is temporary.** The launcher exists *only* for the coexistence window: its whole job is to pick
between two engines. As each capability reaches parity its `--engine` default flips to `v1` (opt-out);
at **RET-6**, once the legacy `mace/` is deleted (`git ls-files mace/ == 0`), there is no second engine
to choose — the legacy dispatch branch is removed and the launcher collapses to a **trivial shim**
(entry points → `mace_torch.cli.*` directly), at which point it can be folded into `mace-torch`
entirely. In the Milestone C end-state it carries no logic. The `--engine` flag and this whole package
are migration scaffolding, not part of the v1 architecture.

### 1.5 `tests/` (repo root) — cross-package parity and the fitness suite

```
tests/
├── parity/                             # double-import allowlist: legacy-vs-v1 oracle comparison
│   ├── conftest.py                     # tiny XYZ fixtures, --engine legacy/v1 harness
│   ├── test_energy_forces_stress.py    # fp64 tolerance, legacy vs v1
│   ├── test_equivariance.py            # rotation/translation/parity of MACEOutputs
│   └── test_reduced_basis_oracle.py    # cross-checks clebsch_gordan.reduced_basis vs an independent Clebsch-Gordan derivation and (best-effort, nightly) vs cueq — never against e3nn (removed)
└── architecture/                       # fitness suite (fast CPU, gates every PR)
    ├── test_import_contracts.py        # import-linter contract checks
    ├── test_model_shape.py             # "forward returns MACEOutputs", "models built via ModelConfig"
    └── test_no_legacy_leak.py          # no v1 file references legacy class names
```

### 1.6 `tutorials/` (repo root) — the educational implementation

An **independent track** (EDU-1/EDU-2, no phase gate), a top-level sibling of `packages/` and the
frozen `mace/` — **not** a package and **not** the "reference backend" (that is the plain-torch/jax
kernel oracle inside the packages, `mace_torch/backends/reference/` and `mace_jax/kernels/reference.py`).
Pure-function JAX, readable over fast, **evaluation-only**: it exists so chemists/physicists can read
and inspect the model blocks, not as a training path nor a mirror of the latest architecture. There is
**no JAX training anywhere** — the `mace_jax` package is inference-only (plan D1) and the tutorials are
inspection/eval-only. It may cross-check its forward numerics against `mace_torch` on a golden fixture,
but nothing in `packages/` depends on it.

```
tutorials/                              # top-level, independent track (roadmap Ch. 12)
├── mace_forward.py                     # pure-function JAX MACE forward (energies), nanoGPT-style, no closures — eval/inspection only (EDU-1)
├── notebooks/                          # marimo notebooks: SH/Clebsch-Gordan/message-passing/many-body walkthrough + a forward/eval demo (EDU-2)
└── tests/                              # CI executes the scripts/notebooks + numerics cross-check vs mace-torch (EDU-2)
```

---

## 2. Layer diagram and dependency direction

### 2.1 Layers (bottom-up; one arrow = "can import")

```
        ┌───────────────────────────────────────────────────────────┐
  L0    │  mace_core   (numpy, pydantic, matscipy)  — no torch/jax    │
        │  types, config, observables, kernels.protocol+registry,     │
        │  clebsch_gordan.reduced_basis, neighbors, weights.neutral_format         │
        └───────────────▲───────────────────────────▲─────────────────┘
                        │                           │
        ┌───────────────┴───────┐       ┌───────────┴───────────────┐
  L1    │  mace_torch.kernels    │       │  mace_jax.kernels          │
        │  + backends (custom_op)│       │  (jax primitives)          │
        └───────────▲───────────┘       └───────────▲───────────────┘
                    │                               │
  L2    │  mace_torch.nn / models / physics │  mace_jax.nn / models  │
                    │                               │
  L3    │  mace_torch.{data,train,finetune,calculators,deploy} │  mace_jax.{data,deploy} │
                    │                               │
  L4    │  mace_torch.cli │                 │  mace_jax.cli │
                    └───────────┬───────────────────┘
                                │  (only from the launcher/tests-parity)
  L5    │  mace-launcher (entry points, --engine)  +  tests/parity/  │
        │           — ONLY ONES with double-import legacy/v1 —        │
                                │
  Lx    │  mace/ (legacy)  — FROZEN, does not import packages, nobody imports it except launcher+parity │
```

### 2.2 Import rules (invariants)

1. `mace_core` **does not import** torch, jax, e3nn, `mace_torch`, `mace_jax`, or `mace`.
2. `mace_torch` imports `mace_core`; **never** `mace_jax` or `mace`.
3. `mace_jax` imports `mace_core`; **never** `mace_torch` or `mace`.
4. Within `mace_torch`: L(n) only imports L(≤n). `nn`/`models` do not import `train`/`cli`; `kernels` does not import `nn`. (kills the god-module: there is no `__init__` that eagerly imports `train`).
5. `packages/**` **never** imports `mace` (legacy). `mace` (legacy) is **never** edited to import `packages/**`.
6. Double-import exception: **only** `mace_launcher.dispatch` and `tests/parity/**`.

### 2.3 CI verification (four gates)

- **import-linter** (`.importlinter`, whole tree): `forbidden`/`layers` contracts:
  - `mace_core` ⊥ {torch, jax, e3nn, mace_torch, mace_jax, mace}
  - `packages` ⊥ `mace` (with allowlist `mace_launcher.dispatch`, `tests.parity`)
  - `mace_jax` ⊥ `mace_torch`
  - internal layers of `mace_torch` (`layers: cli > train,data > models > nn > kernels > mace_core`)
- **runtime audit-hook** (`mace_launcher.audit`, active in `tests/parity`): closes the static false-negative — a dynamic `importlib.import_module("mace...")` from packages fails the test.
- **fitness suite** (`tests/architecture/`, fast CPU job, gating every PR): always-green asserts about the target shape — "forward returns `MACEOutputs`", "models are built via `ModelConfig`", "zero `torch_geometric` in the model interface", "zero `jit.*`/`@compile_mode('script')` in the live v1 path", "no v1 file references legacy class names".
- **toolchain meta-lint**: no file matches two toolchains (1:1 ownership: `mace/**`→black+isort+pylint+mypy; `packages/**`→ruff+ty+prek).

---

## 3. Extension points

**The extension spectrum — most of it is config, not code.** Extending MACE runs a ladder from a
plain config knob to a genuinely new model. The common cases touch **zero code**; only new physics
in the forward is a real code change:

| You want to… | How | New code? |
|---|---|---|
| **Tune a parameter** — a loss weight, a Huber `delta`, a cutoff, a schedule, any hyperparameter | set a field in config | **none** |
| **Train a new property** — any well-defined spherical-tensor observable (a dipole, a rank-2 tensor, spectra, a magnetic moment) | add a **row to the observable table** (`ObservableSpec` in config, canonical defaults in `defaults/observables.yaml`) — it auto-creates the head, the loss term, and the derivative names | **none** |
| **A new loss** — a non-standard reduction, or a data/relative-energy/mask transform | `@register_loss` / `@register_transform` + select it in config | a small module |
| **A new readout / head** | `@register_readout` + config | a small module |
| **A new backend** — kernel, data format, neighbour list, electrostatics solver | ship a wheel with one entry-point line (`mace.kernel_backends.torch`, `mace.data_backends`, `mace.neighbor_backends`, `mace.electrostatics_backends.torch`) | a backend module, **zero core edits** |
| **A new model** — new physics in the forward (electrostatics k-space, an SCF loop, magnetic-with-new-physics) | a `BaseMACE` subclass + `@register_model` (+ a model-transform hook) | a real model — **the one non-free case** |

Principle: **extending the system touches ZERO core files** — everything enters via a config field, a
registry decorator, or an entry point + a test that inherits a parametrized harness. Two clarifications
that come up a lot:

- **Tuning an existing loss is config, not a new loss.** A loss carries its own parameters — the
  per-observable weights *and* its own hyperparameters (e.g. Huber `delta`) — set from config
  (`LossConfig`, §3.3). Adding a knob to an existing loss is a field, not a `@register_loss`. Losses
  are also **generated from the observable table**, so a new spherical-tensor property needs no loss
  code at all — its loss term appears automatically with a default weight you can override.
- **A new property is a table row, not an add-on.** The framework is property-agnostic by
  construction: the observable table maps a property name to its mathematical structure (irreps,
  per-atom vs total, units, normalization), and everything downstream (head, loss, derivatives) is
  derived from that row.

The sections below give the worked examples for each row of the ladder; for a single **end-to-end
example of a complex feature** (a new input + observable + derivative + augmentation + loss + an SCF
model, change by change), see [`extending_mace.md`](extending_mace.md).

### 3.1 A new kernel backend (e.g. a custom Triton/SYCL one)

- **Contract** (`mace_core.kernels.protocol.KernelBackend`, authoritative definition in `rfcs/rfc-01-backend-dispatch.md` §3.2/§3.3): a closed set of `make_*` factories, resolved once at model build time, each returning an op instance; `capabilities()` returns a `BackendCapabilities` record — coarse fields (`ops`, `devices`, `dtypes`, `max_lmax`, `layouts`, `bases`, `supports_double_backward`) as a cheap first filter, plus the authoritative `supports(descriptor)` predicate that sees the full op descriptor, so a backend can accept or decline down to the exact shape being built:

  ```python
  class KernelBackend(Protocol):
      name: str
      def capabilities(self) -> BackendCapabilities: ...   # .supports(OpDescriptor) -> bool is authoritative;
                                                           # .supports_double_backward / .bases drive hard-reject rules
      # factories: descriptor -> op instance, resolved ONCE at model build time
      def make_linear(self, d: LinearDescriptor) -> LinearOp: ...
      def make_channelwise_tp_conv(self, d: ChannelwiseTPConvDescriptor) -> ChannelwiseTPConv: ...  # ALWAYS node-level; there is no per-edge variant
      def make_symmetric_contraction(self, d: SymmetricContractionDescriptor) -> SymmetricContractionOp: ...
      def make_fully_connected_tp(self, d: FullyConnectedTPDescriptor) -> FullyConnectedTPOp: ...
      def make_segment_reduce(self, d: SegmentReduceDescriptor) -> SegmentReduceOp: ...
      def make_spherical_harmonics(self, d: SphericalHarmonicsDescriptor) -> SphericalHarmonicsOp: ...  # reference-only, may decline
      def make_radial_basis(self, d: RadialBasisDescriptor) -> RadialBasisOp: ...                        # reference-only, may decline
      # optional span factory: a backend MAY fuse a whole interaction layer instead of composing it op-by-op
      def make_interaction_layer(self, descriptors: "InteractionLayerDescriptors") -> "InteractionLayerOp | NotImplemented": ...
  ```

  Internal-weight ops (`linear`, `symmetric_contraction`, `fully_connected_tp`) additionally implement `to_canonical()` / `load_canonical()`, applied only at the checkpoint save/load boundary, never on the forward path.
- **Extender touches:** your own package (or `mace_torch/backends/mybackend/backend.py`) + one line in its `pyproject.toml`:
  ```toml
  [project.entry-points."mace.kernel_backends.torch"]
  mybackend = "my_pkg.backend:MyBackend"
  ```
- **Core touched:** **zero**. It is discovered via `mace_core.kernels.registry.get_backend("mybackend")`.
- **Test covering it:** inherits the parametrized `tests/backends/conftest.py::BackendConformance` — forward + backward + **double-backward** == `ReferenceBackend` at fp64, and **torch.compile with no graph-break** (forces `register_fake`). The **example** backend already runs this harness as a permanent gate, so a third party only needs to register and the harness picks it up.

**Worked example — define, install, select.** A minimal third-party backend that only accelerates the
symmetric contraction and lets everything else fall back to the reference:

```python
# my_pkg/backend.py
from mace_core.kernels import BackendCapabilities   # data-about-ops only; no torch types here

class MyBackend:
    name = "mybackend"

    def capabilities(self) -> BackendCapabilities:
        return BackendCapabilities(
            ops={"symmetric_contraction"},          # coarse first filter: only this op
            devices={"cuda"}, dtypes={"float32", "float64"},
            supports_double_backward=True,          # REQUIRED for force training, else hard-rejected at build
            supports=lambda d: d.op == "symmetric_contraction" and d.uniform_multiplicity,
        )

    def make_symmetric_contraction(self, d):        # called once at build time, only for ops `supports` accepted
        return MySymmetricContractionOp(d)          # a torch.library.custom_op + register_fake + register_autograd
    # every other make_* may raise NotImplementedError: `supports` already declined them,
    # and the CompositeBackend routes those ops to the reference backend automatically.
```

```toml
# my_pkg/pyproject.toml — the ONLY registration step; zero edits to mace
[project.entry-points."mace.kernel_backends.torch"]
mybackend = "my_pkg.backend:MyBackend"
```

```bash
pip install my-backend-pkg        # ships the entry point; MACE discovers it, no repo edits
```

Selecting and **switching** between backends is a build-time config choice, resolved once and frozen
into the module tree (nothing dispatches in `forward`). Because weights are canonical, **the same
checkpoint loads into any backend** — switching never reconverts weights:

```python
from mace_torch import build_model
build_model(ModelConfig(..., backend="reference"))   # default: plain torch, CPU-ok, the oracle
build_model(ModelConfig(..., backend="cueq"))         # NVIDIA cueq where it applies, reference elsewhere
build_model(ModelConfig(..., backend="mybackend"))    # your wheel, picked up via the entry point
```

```bash
mace train --model.backend cueq  ...      # same choice from the CLI (dotted override)
```

Any op/dtype/device/irreps a backend declines is served by the reference backend through a
`CompositeBackend` — so a partial backend (one op, one dtype) is a first-class citizen. A backend that
lacks double-backward is **hard-rejected at build time for force/stress training** (loud error, never
silently wrong forces); it stays usable for inference (`supports_double_backward=False`).

### 3.2 A new observable (config only)

- **Extender touches:** a `mace_core/observables/myobs.py` file with an `Observable` (declares output irreps and derivation mode: `readout` | `autograd(energy, wrt=positions)` | `grad_strain`) + `@register_observable("myobs")`.
- **Core touched:** zero existing files (only the new module is added). The model exposes it automatically because `BaseMACE` iterates over `config.observables`; `MACEOutputs` is a dataclass with optional fields populated by name.
- **Enabling it:** `ModelConfig(observables=["energy","forces","myobs"])`.
- **Test:** `mace_core/tests/test_observable_registry.py` validates irreps/derivation consistency (pure); if it is autograd-derived, `tests/parity` verifies finite-diff.

### 3.3 A new loss / transform (plugin registry)

- **Tuning an existing loss is config, not a new loss.** `LossConfig` carries the per-observable **weights** *and* the loss's own **parameters** (e.g. `params={"huber_delta": 0.02}`) — this preserves the legacy `--energy_weight`/`--forces_weight`/`--huber_delta` knobs as config fields. Changing a coefficient never needs a `@register_loss`. And a well-defined spherical-tensor observable needs **no** loss code at all — its term is generated from the observable table with `default_loss_weight`.
- **A genuinely new loss:** `mace_torch/train/loss.py` (or an external package) with `@register_loss("myloss")` on a `torch.nn.Module`; select via `LossConfig(name="myloss", weights=..., params=...)`.
- **Data transform:** `@register_transform("mytransform")` in `mace_torch/data/`; chained via `DataConfig(transforms=[...])`.
- **Core touched:** the registries (`LOSS_REGISTRY`, `TRANSFORM_REGISTRY`) live in `mace_core.registries` as specs; **adding one does not edit the registry**, only the decorator populates it at import time. Zero core edits.
- **Test:** `tests/unit/test_loss_registry.py` (value test: the loss produces the expected scalar over a fixed batch).

**Worked example — define and register.** The registry pattern is the same as observables; a loss
registers itself with a decorator at import time, and is then selected in config via
`LossConfig(name=...)` (the bullet above):

```python
# mace_torch/train/loss.py
import torch
from mace_torch.train import register_loss

@register_loss("my_huber_ef")                       # the decorator IS the registration — no registry edit
class MyHuberEnergyForces(torch.nn.Module):
    def __init__(self, delta: float = 0.01):
        super().__init__()
        self.delta = delta

    def forward(self, pred, ref, weights):          # typed batch of MACEOutputs; returns a scalar
        e = torch.nn.functional.huber_loss(pred.total_energy, ref.total_energy, delta=self.delta)
        f = torch.nn.functional.huber_loss(pred.forces, ref.forces, delta=self.delta)
        return weights.energy * e + weights.forces * f
```

A well-defined spherical-tensor observable needs **no** new loss at all — TRN-2 generates the loss term
from the observable spec; a custom `@register_loss` is only for non-standard reductions/transforms.

### 3.4 A new data format

- **Extender touches:** a `DataBackend` implementation yielding `Configuration`s (rfc-05 protocol) registered via the `mace.data_backends` entry-point group; optionally a shard-writer in `mace_core.data_spec.shard_format`.
- **Core touched:** zero (the model only consumes the flat-dict; no `torch_geometric` in the interface). Selected with `DataConfig(format="myfmt")`.
- **Test:** `tests/workflows/test_preprocess_myfmt.py` (E2E: preprocess→train→eval) + a value test asserting that the produced graph dict passes the `GRAPH_SCHEMA` validator (rfc-03).

**Worked example — define, install, select.** A backend yields framework-free `Configuration`s;
graph building (neighbours, one-hot, collation) happens once *above* every backend, so the same
backend feeds torch and jax:

```python
# my_pkg/backend.py
from mace_core.data import Configuration, DataBackendError

class ZarrBackend:                                  # map-style: lazy, picklable, exact __len__
    name = "zarr"

    @staticmethod
    def sniff(path) -> bool:                        # format="auto" runs each backend's sniff() in priority order
        return str(path).endswith(".zarr")

    def open(self, path): self._store = ...; return self
    def __len__(self) -> int: ...
    def __getitem__(self, i: int) -> Configuration:  # RAISES DataBackendError on a corrupt row (never returns None)
        ...
    def iter_range(self, start, stop): ...           # rank-contiguous reads for DDP (sampler's concern, not the backend's)
    def statistics(self): ...                        # or None → mace_core computes them
```

```toml
# my_pkg/pyproject.toml — one line, discovered like a kernel backend (RFC-05 mirrors RFC-01 §3.1)
[project.entry-points."mace.data_backends"]
zarr = "my_pkg.backend:ZarrBackend"
```

```bash
pip install mace-zarr
mace train --data.format zarr  --data.train path/to/set.zarr  ...   # or format="auto" sniffs it
```

`mace data prepare` still writes exactly **one** blessed format (HDF5 v2 shards); a custom backend is a
first-class *read* path for external corpora, not a new write schema.

### 3.5 A new head / readout

- **Extender touches:** `mace_torch/nn/readout.py` (or its own module) with `@register_readout("MyReadout")`; for a head, `HeadConfig(readout="MyReadout", ...)` in `mace_torch/finetune/multihead.py`.
- **Core touched:** zero — `READOUT_REGISTRY` is populated by decorator; `BaseMACE` builds readouts by name from `ModelConfig`. Bias readouts force `mul_ir`; the `linear_irreps(..., bias=)` contract covers this explicitly instead of the legacy's hardcoded `o3.Linear(biases=True)`.
- **Test:** `tests/unit/test_readout_registry.py` (value) + equivariance (`tests/parity/test_equivariance.py` includes the new head if it is in the registry).

### 3.6 A new model (e.g. electrostatics)

- **Extender touches:** `mace_torch/models/mymodel.py` subclassing `BaseMACE` (block composition + `observables`), + `@register_model("MyModel")`. Real example: `PolarMACE` lives in `models/electrostatics.py` as just another entry, not as an `extensions.py` bolt-on.
- **Core touched:** zero — `MODEL_REGISTRY` by decorator; `build.py` instantiates via `ModelConfig(model="MyModel")`.
- **Test:** per-model harness in `tests/parity` (fp64 parity vs legacy if a legacy counterpart exists; if brand-new, finite-diff + equivariance + committed goldens).

**Worked example — define and register.** Unlike the first five rows, a new model is **not free** (see
the honest-boundary note below): it subclasses the backbone and may add a model-transform hook. It is
still registered, not patched:

```python
# mace_torch/models/electrostatics.py
from mace_torch.models import BaseMACE, register_model

@register_model("MyElectrostatic")                  # decorator registers it; build.py finds it by name
class MyElectrostatic(BaseMACE):
    def forward(self, graph):
        features = super().forward(graph)           # the equivariant backbone (node features)
        # add a k-space / SCF term that depends on positions and cell and couples into the
        # derivative engine — this is why a model is more than a readout (RFC-01 §6b.2)
        return features
```

```python
build_model(ModelConfig(model="MyElectrostatic", observables=["energy", "forces"]))
```

No entry-point/pip step is shown because a model ships **inside** `mace-torch` (or a package that
depends on it), not as a drop-in kernel/data plugin; the electrostatics solver *inside* it is the part
that is backend-swappable, and that goes through RFC-09, not here.

**Summary of "core files touched by extension": 0 in all 6 cases.** Everything enters via entry_point or registry decorator; the core only defines Protocols/registries.

**Honest boundary.** This holds for the first five rows. A **new model** is different: `MACELES` and `PolarMACE` subclass `ScaleShiftMACE` and override its forward (`mace/modules/extensions.py:78,307`), and `PolarMACE` adds k-space electrostatic energies that depend on positions and cell and couple into the derivative engine. A model is therefore a `BaseMACE` subclass plus a `MACEOutputs` plus, for electrostatics, a model-transform hook — registered rather than patched, but not free. See RFC-01 §6b.2.

---

## 4. Test strategy by layer

| Layer | What it tests | Depends on | Speed |
|------|-----------|-----------|-----------|
| **Pure unit (core)** | `mace_core/tests`: Clebsch-Gordan, reduced basis, neighbors, `ModelConfig.from_namespace`, observable/canonical registries | numpy only | **fast** (no torch, no GPU) |
| **Backend conformance** | `tests/backends`: every `KernelBackend` satisfies the Protocol; fwd+bwd+**double-bwd** == reference; compile with no graph-break | torch + backend | fast (CPU for reference/example) / slow on GPU (cueq/oeq) |
| **Numerical parity** | `tests/parity`: same tiny XYZ via `--engine legacy` and `--engine v1` → energy/forces/stress at fp64 tolerance; oracle = frozen legacy in-process | torch + legacy | **fast** on PR (tiny), **slow** on nightly (property/fuzz: PBC, mixed float32/64, high L, empty neighborhood) |
| **Equivariance** | `tests/parity/test_equivariance.py`: rotation/translation/parity of MACEOutputs; energy invariant, forces covariant | torch/jax | fast |
| **CLI E2E contracts** | `tests/workflows` (65 black-box functions: 54 in `tests/workflows` + 11 in `tests/integrations`) + `integrations/lammps`: subprocess to `mace_run_train` with `--engine v1`, returncode + loss-decrease + ASE `get_potential_energy` | subprocess, PYTHONPATH=REPO_ROOT | **slow** (auto-slow) |
| **Goldens** | P0-2/P0-3 (external FM snapshots, network), P0-6 (finite-diff, virial sign PINNED), P0-7 (neighbors/radial/batching) | edit-locked | fast; foundations = network (opt-in) |

Fast/slow split:
- **Every PR (fast, CPU):** unit core + reference/example conformance + tiny parity + equivariance + fitness suite + goldens P0-6/P0-7.
- **Nightly (slow):** property/fuzz parity, cueq/oeq on the GPU vendor jobs (`ci-gpu-mpcdf.yaml`: `-m gpu` on Nvidia, `-m "gpu and not cueq"` on AMD, skip-o-fail `MACE_REQUIRE_CAPS`), full-matrix workflows, `bin_lammps` real tier, network foundations.

Reuse of the existing machinery: the live harness is the **65 black-box functions in `tests/workflows` + `integrations/`** run with `--engine v1` via the `PYTHONPATH=REPO_ROOT` harness, without writing a new E2E test or touching `tests/helpers.py`. The capability contract in `tests/conftest.py` (`CAPABILITY_PROBES`, auto-marked by directory, zero-collection guard) is inherited as-is; a new backend enters as a probe+marker.

Multiple oracles for the reduced basis and its pinned order/normalization: `test_reduced_basis.py` compares it against (1) live legacy in-process, (2) cueq round-trip on GPU (alive for one release), (3) committed goldens — before any backend relies on it. This oracle sits on the critical path (RFC-A → CORE-4 → BKD-1): the pinned path order and per-path normalization *are* the on-disk weight format (rfc-01 §2.4.1).

---

## 5. The tree at three migration milestones

The legacy `mace/` package stays frozen as the behavioral oracle through Milestones A and B. It is retired capability-by-capability in Milestone C, gated by the hard release requirement `git ls-files mace/ == 0` before the 1.0 tag.

### Milestone A — packages scaffolded, legacy the only live path

```
mace/                         ← FROZEN byte-for-byte, legacy suite green (oracle)
packages/
  mace-core/                  ← installable scaffold, nearly empty (types.py, kernels/registry.py, config/ stubs)
  mace-torch/                 ← installable scaffold, nearly empty
  mace-jax/                   ← empty scaffold
  mace-launcher/              ← OWNS all entry points, default --engine=legacy (zero-change behavior)
tests/
  parity/                     ← in-process legacy-vs-v1 harness (still trivial)
  architecture/               ← first fitness functions + import-linter contract
  (unit,workflows,backends,extensions,foundations,integrations,benchmarks)  ← legacy, unchanged
.importlinter, .pre-commit dual-path, audit hook
```
`git ls-files mace/` = maximum. Everything goes through `mace.cli.*`. import-linter is green because `packages/` is empty.

### Milestone B — data, kernels, and models complete in `mace_torch`; deployment remains legacy

```
mace/                         ← frozen and complete; remains the default for non-migrated capabilities
packages/
  mace-core/                  ← complete: clebsch_gordan.reduced_basis with pinned order, observables, config Pydantic, kernels.protocol, neighbors
  mace-torch/
    kernels/ (custom_op) ✓    backends/{reference,cueq,oeq,example} ✓
    nn/ models/ physics/ ✓    data/ (hdf5/lmdb/xyz, no torch_geometric) ✓
    train/ calculators/ ✓     finetune/ (partial)   deploy/ (LAMMPS export remains in legacy)
  mace-jax/                   ← energy+forces parallel, validated against the neutral format
  mace-launcher/              ← default --engine=v1 for energy/forces/stress/data (opt-out); legacy for dipole/polar/LAMMPS
tests/
  parity/                     ← dense: energy+forces+stress+data at fp64, property/fuzz in nightly
  backends/                   ← conformance + double-backward gate (permanent example backend)
  workflows/                  ← running with --engine v1 for what's migrated
```
`git ls-files mace/` = maximum: the legacy package is untouched and continues to serve as the oracle. Dual toolchain active. The 5 `convert_*` CLIs remain alive only for legacy (redundant for v1: single canonical layout).

### Milestone C — capability-by-capability retirement complete, release gate satisfied

```
packages/
  mace-core/   mace-torch/   mace-jax/      ← only live paths; deploy/ with torch.export (or isolated export_adapter)
  mace-launcher/                            ← flat shim: entry points → mace_torch.cli.* directly (legacy branch deleted)
tests/
  unit/ backends/ workflows/ extensions/ foundations/ integrations/ parity/ architecture/
    parity/ degraded to frozen goldens (legacy counterparts no longer exist)
  (SINGLE toolchain: ruff+ty+prek; per-package --cov floors)
mace/                                        ← REMOVED (git ls-files mace/ == 0)
```
Deletion order (delete-only PRs): energy models → data layer → `convert_*` CLIs → dipole/polar → LAMMPS jit → `mace/tools` god-module + launcher legacy branch. The hard release gate blocks the 1.0 tag until `git ls-files mace/ == 0`. The extensibility layer (kernels) is native from the start — `custom_op + register_fake` — and was never deformed by TorchScript.

Once `git ls-files mace/ == 0`, the freed `mace` name no longer shadows anything, so the container `packages/` may be renamed to `mace/` as a final cosmetic step (**REL-4**) — container directory only; import names stay `mace_core`/`mace_torch`/`mace_jax`. Optional: keeping `packages/` is equally valid (LangChain `libs/`, uv `packages/`).
