# The MACE v1 packages

Three packages, one direction of dependency:

```
mace-core            contract + pure math; no torch, no jax, no e3nn
   ^        ^
   |        |
mace-torch  mace-jax
```

`mace_torch` and `mace_jax` are two implementations of one contract. Neither
imports the other, and neither imports the legacy `mace` package, which stays
in the tree as a frozen numerical oracle until it is retired.

`mace_core` holds what both frameworks agree on: the configuration schema, the
data layer, the declared observables, the Clebsch-Gordan basis, the kernel and
solver contracts, and the metadata a model carries. `mace_torch` is the full
implementation: models, training, deployment. `mace_jax` is inference only and
is still empty. [Navigating the tree](#navigating-the-tree) says what is in
every file.

## Names

| Directory | Distribution | Import name | Tag prefix |
|---|---|---|---|
| `packages/mace-core` | `mace-core` | `mace_core` | `mace-core-v*` |
| `packages/mace-torch` | `mace-torch-v1` | `mace_torch` | `mace-torch-v*` |
| `packages/mace-jax` | `mace-jax` | `mace_jax` | `mace-jax-v*` |
| `packages/mace-launcher` | `mace-launcher` | `mace_launcher` | `mace-launcher-v*` |

The import names never collide with the legacy import name `mace`, so both
stacks live in one process.

The distribution names cannot all match their import names. The legacy package
holds the `mace-torch` distribution name, and two distributions cannot both
hold one name in a single environment, so the v1 PyTorch package ships as
**`mace-torch-v1`** while keeping the `mace_torch` import name.

**`mace-torch-v1` is never published to PyPI.** It exists so the two stacks can
be installed side by side while both are in the tree. `mace-torch` on PyPI is
frozen at its final v0.3.x release and stays dormant for the whole rewrite, so
the name is free the moment legacy leaves the tree: RET-6 renames the
distribution to `mace-torch`, and the v1.0.0 release publishes under that name.
PyPI has no rename operation, so publishing `mace-torch-v1` even once would
turn that switch from a no-op into a permanent user migration.

Tag prefixes are keyed on the directory rather than the distribution name, so
each package versions independently from git tags and the RET-6 rename touches
only the `name` field.

`mace-launcher` owns every `mace_*` console script and none of the other three
declares one, because two distributions declaring the same script name is
undefined behaviour in pip. That is not a theoretical hazard: uninstalling one
of two such distributions deletes the script files the other one had
overwritten, and the surviving distribution is left with none.

The rule is one owner per script name, not one owner overall. `mace-jax` will
declare a single script of its own, `mace-jax`, when its CLI lands: it does not
depend on `mace-torch` and has to work with no torch installed, so it cannot
inherit a script from either the launcher or `mace-torch`. The hyphen keeps the
name outside the `mace_*` set, so the two owners never overlap.

The launcher is also the only place the two stacks meet. It picks one with
`--engine {legacy,v1}` or `MACE_ENGINE`, defaulting to `legacy`, and installs a
runtime guard that fails if a v1 module imports the frozen legacy package.

**`mace-launcher` is never published either, and unlike `mace-torch-v1` it does
not survive under another name.** It exists to choose between two engines, so it
has nothing left to do once the legacy package is deleted. RET-6 removes the
package and moves its `[project.scripts]` table into `mace-torch`, pointed at
`mace_torch.cli.*`. Single ownership survives that move: exactly one installed
distribution still declares each `mace_*` script, and from then on it is
`mace-torch`. The v1.0.0 release therefore publishes three distributions, not
four.

## Installing alongside the legacy package

Both stacks in one environment, from a fresh venv at the repository root:

```bash
python -m venv .venv-v1 && source .venv-v1/bin/activate
pip install -e packages/mace-core -e packages/mace-torch \
            -e packages/mace-jax -e packages/mace-launcher
pip install -e .
```

One `pip install` for the packages, not one each: `mace-torch-v1` and
`mace-jax` require `mace-core`, which is not on PyPI, so pip has to see it as a
local requirement in the same resolution.

Then, in one process:

```bash
python -c "import mace, mace_core, mace_torch, mace_jax; print('coexist ok')"
```

The legacy install is the same `pip install -e .` it has always been. Nothing
about it changes, and the scaffold cannot enter the legacy wheel: its build
discovers packages with `setuptools.find_packages()`, which prunes any path
component that is not a Python identifier, and every directory here is
hyphenated. `tests/architecture/test_packaging_isolation.py` asserts that
rather than trusting it.

## Test file names

Test files are collected from several packages in one pytest run, under one
rootdir, and none of the `tests/` directories is an importable package. Two
files sharing a basename therefore collide on import. Prefix each test module
with its package: `test_mace_core_scaffold.py`, not `test_scaffold.py`.

## Dependencies

`mace-core` depends on numpy, ase, pydantic, pyyaml, tomli, matscipy and
safetensors, and on no array framework. `mace-torch-v1` adds torch and scipy,
with `wandb` as its one extra. `mace-jax` requires only `mace-core` until its
inference code lands. `mace-launcher` has no dependency of its own; it runs
whichever engine is installed.

## Navigating the tree

What is in each directory, where to start reading, and one line for every
source file. The file tables are checked against the tree by
`tests/architecture/test_packages_readme.py`: a source file that is added,
moved or deleted without its row fails there, so the guide cannot go stale.

```
packages/
  mace-core/src/mace_core/       framework-free contract and pure math
    clebsch_gordan/              irreps, Wigner 3j, the reduced symmetric basis
    config/                      the pydantic configuration, section by section
    data/                        structures, backends, E0s, statistics, splits
      backends/                  in-memory and XYZ data backends
    defaults/                    the default observable catalogue, as YAML
    electrostatics/              long-range solve descriptors, solver registry
    elements/                    element tables, default file keys, E0 values
    kernels/                     op descriptors, backend protocol and registry
    observables/                 declared observables and their derivatives
    weights/                     the neutral weights format
  mace-torch/src/mace_torch/     the PyTorch implementation
    backends/reference/          the plain-torch kernel backend
    calculators/                 the ASE calculator and static-shape padding
    cli/                         console entry points
    data/                        graphs, batches, loaders, transforms
    deploy/                      loading checkpoints, converting legacy ones
    electrostatics/              solver dispatch and the long-range ops
      reference/                 the in-tree reference solver (MIT, see LICENSE)
    finetune/                    foundation models, replay, LoRA, freezing
    kernels/                     custom ops and weight initialization
    legacy-extractor/            a script run by the legacy venv, not a module
    models/                      the model and its output layer
    nn/                          equivariant building blocks
    physics/                     forces, stress and fixed points, around the model
    train/                       the three stages and the training loop
  mace-jax/src/mace_jax/         inference only, not yet implemented
  mace-launcher/src/mace_launcher/  console scripts, engine dispatch, import audit
```

### Where to start

**A training run**, in the order it happens:
`mace_torch/cli/run_train.py` parses the command line into
`mace_core/config/resolved.py`; `mace_torch/train/data_stage.py` reads the data
and resolves the E0s and statistics; `mace_torch/train/model_stage.py` builds
the model; `mace_torch/train/loop.py` trains it and
`mace_torch/train/checkpoint.py` writes it down.

**A forward pass**, from graph to forces: `mace_torch/data/graphs.py` builds
the graph `mace_core/graph.py` declares; `mace_torch/physics/outputs.py` wraps
the model, applies the strain and takes the derivatives;
`mace_torch/models/base.py` runs `mace_torch/nn/backbone.py` and then
`mace_torch/models/outputs.py`, which returns `mace_core/outputs.py`'s typed
output.

**A new kernel backend** implements `mace_core/kernels/protocol.py`, reads its
ops from `mace_core/kernels/descriptors.py` and registers an entry point; the
reference in `mace_torch/backends/reference/backend.py` is the one to copy.
**A new long-range solver** does the same against
`mace_core/electrostatics/capabilities.py` and `mace_torch/electrostatics/solver.py`.

### `mace-core`

| File | What is in it |
|---|---|
| `mace_core/__init__.py` | The package version; the public surface is imported from the submodules. |
| `mace_core/clebsch_gordan/__init__.py` | The public surface of the basis: path order and normalization conventions. |
| `mace_core/clebsch_gordan/coefficients.py` | Complex Clebsch-Gordan and Wigner 3j coefficients from the Racah formula. |
| `mace_core/clebsch_gordan/conversion.py` | Converting weights between the full and reduced bases, and between layouts. |
| `mace_core/clebsch_gordan/irreps.py` | O(3) irreps: parsing, ordering, dimensions, selection rules. |
| `mace_core/clebsch_gordan/real_basis.py` | The Wigner 3j table in the real spherical-harmonic basis the models use. |
| `mace_core/clebsch_gordan/reduced_basis.py` | The reduced symmetric tensor-product basis and its path order. |
| `mace_core/config/__init__.py` | The configuration schemas, re-exported. |
| `mace_core/config/base.py` | The base schema: file loading, dotted CLI overrides, unknown keys as errors. |
| `mace_core/config/data.py` | Datasets, heads, splits, transforms and the graph-input file keys. |
| `mace_core/config/e0s.py` | The ways a head's isolated-atom energies can be given or estimated. |
| `mace_core/config/electrostatics.py` | Which long-range solver a model uses, and for which systems. |
| `mace_core/config/fixed_point.py` | Describing a self-consistent relaxation loop. |
| `mace_core/config/legacy.py` | Every legacy training flag and what became of it, plus the translator. |
| `mace_core/config/loss.py` | Loss kinds and per-observable weights. |
| `mace_core/config/model.py` | Architecture, declared observables, readouts, and the polar section. |
| `mace_core/config/provenance.py` | Turning what a run asked for into what its model records. |
| `mace_core/config/resolved.py` | The whole configuration and the rules that span its sections. |
| `mace_core/config/runtime.py` | Run name, output directory, seed, device, distribution, logging, plots. |
| `mace_core/config/section.py` | A section that is immutable once validated. |
| `mace_core/config/tracking.py` | Experiment tracking settings. |
| `mace_core/config/training.py` | Optimizers, learning-rate schedules and the training stages. |
| `mace_core/data/__init__.py` | The data layer's boundary objects, re-exported. |
| `mace_core/data/backend.py` | The protocol every data backend implements, and the statistics type. |
| `mace_core/data/backends/__init__.py` | The shipped backends and their registration. |
| `mace_core/data/backends/memory.py` | Structures already in memory, presented as a backend. |
| `mace_core/data/backends/xyz.py` | The XYZ backend over anything ase can read. |
| `mace_core/data/configuration.py` | `Configuration`: one parsed, labelled structure. |
| `mace_core/data/conformance.py` | The checks every data backend has to pass. |
| `mace_core/data/e0_resolution.py` | Turning an E0 declaration into one energy per element. |
| `mace_core/data/keys.py` | Which file key each property is read from. |
| `mace_core/data/registry.py` | Finding a data backend by name, with no fallback. |
| `mace_core/data/splitting.py` | Train and validation splits, and grouping for reports. |
| `mace_core/data/statistics.py` | Average neighbours, energy shift and scale, over any backend. |
| `mace_core/data/xyz.py` | Reading labelled structure files into configurations. |
| `mace_core/defaults/observables.yaml` | The default observable catalogue: energy with forces and stress. |
| `mace_core/electrostatics/__init__.py` | The electrostatics contract, re-exported. |
| `mace_core/electrostatics/capabilities.py` | What a solver declares it can do, and the refusal when it cannot. |
| `mace_core/electrostatics/descriptor.py` | One long-range solve as data: profile, multipoles, projection, method. |
| `mace_core/electrostatics/registry.py` | Solver discovery by entry point, and which solver a model loads with. |
| `mace_core/elements/__init__.py` | Element bookkeeping, re-exported. |
| `mace_core/elements/default_keys.py` | The default file keys of a labelled structure file. |
| `mace_core/elements/e0s.py` | Resolved isolated-atom energies, per head and element. |
| `mace_core/elements/number_table.py` | The ordered element table a model was fitted for. |
| `mace_core/graph.py` | The graph schema: every field, its shape, dtype and padding, and collation. |
| `mace_core/kernels/__init__.py` | The kernel contract, re-exported. |
| `mace_core/kernels/canonical.py` | The canonical weight layout that makes checkpoints backend independent. |
| `mace_core/kernels/capabilities.py` | What a kernel backend declares it can build. |
| `mace_core/kernels/descriptors.py` | Every dispatched op, described before a backend builds it. |
| `mace_core/kernels/paths.py` | The tensor-product paths of the convolution and their order. |
| `mace_core/kernels/precision.py` | Dtype names and the per-op precision configuration. |
| `mace_core/kernels/protocol.py` | The interface a kernel backend implements. |
| `mace_core/kernels/registry.py` | Finding kernel backends by entry point. |
| `mace_core/metadata.py` | `ModelMetadata`: the versioned record every checkpoint carries. |
| `mace_core/neighbors.py` | The neighbour list as a pure function, with the cell regimes stated. |
| `mace_core/observables/__init__.py` | Declared observables, re-exported. |
| `mace_core/observables/defaults.py` | Loading an observable catalogue from its YAML file. |
| `mace_core/observables/derivatives.py` | How a derivative of a declared quantity is named. |
| `mace_core/observables/grammar.py` | The irreps string grammar. |
| `mace_core/observables/request.py` | Turning configured names into observables and derivatives. |
| `mace_core/observables/spec.py` | `ObservableSpec`, `InputSpec` and the catalogue. |
| `mace_core/outputs.py` | `MACEOutput`, the typed result every model returns. |
| `mace_core/stages.py` | The objects passed between the data, model and training stages. |
| `mace_core/tables.py` | The error table a finished run prints. |
| `mace_core/units.py` | Units and sign conventions for the whole stack. |
| `mace_core/weights/__init__.py` | The neutral weights format, re-exported. |
| `mace_core/weights/neutral_format.py` | safetensors for the numbers and JSON for their meaning. |

### `mace-torch`

| File | What is in it |
|---|---|
| `mace_torch/__init__.py` | The package version. |
| `mace_torch/backends/__init__.py` | The shipped kernel backends. |
| `mace_torch/backends/reference/__init__.py` | The reference backend, re-exported. |
| `mace_torch/backends/reference/backend.py` | Plain-torch linear, convolution, contraction, radial basis and reductions. |
| `mace_torch/backends/reference/spherical_harmonics.py` | Real spherical harmonics in the legacy convention. |
| `mace_torch/calculators/__init__.py` | The calculators, re-exported. |
| `mace_torch/calculators/ase_calculator.py` | `MACECalculator` for ASE: results, committees, Hessian, units. |
| `mace_torch/calculators/padding.py` | Padding a batch to a fixed size for compiled GPU runs, and cutting it back. |
| `mace_torch/cli/__init__.py` | Console entry points only. |
| `mace_torch/cli/convert_legacy.py` | `mace_convert_legacy`: a legacy checkpoint into a v1 one. |
| `mace_torch/cli/run_train.py` | `mace_run_train --engine v1`: training from a configuration file. |
| `mace_torch/data/__init__.py` | Graph building and batching, re-exported. |
| `mace_torch/data/batch.py` | `TrainingBatch`, collation, the lazy graph dataset and its loader. |
| `mace_torch/data/distributed_sampler.py` | Splitting an epoch across the processes of a distributed run. |
| `mace_torch/data/graphs.py` | One structure into a graph and its training targets. |
| `mace_torch/data/transforms.py` | Registered transforms of what a run trains against. |
| `mace_torch/deploy/__init__.py` | Loading and converting models, re-exported. |
| `mace_torch/deploy/legacy.py` | Legacy conversion: extract in the legacy venv, import here. |
| `mace_torch/deploy/loader.py` | `load_deployed`: a checkpoint rebuilt for evaluation. |
| `mace_torch/deploy/neutral_io.py` | A neutral artifact into a v1 model through the checkpoint path. |
| `mace_torch/deploy/reference.py` | Checking a converted model against committed reference values. |
| `mace_torch/electrostatics/__init__.py` | The long-range ops and their dispatch, re-exported. |
| `mace_torch/electrostatics/reference/LICENSE.md` | The MIT license of the solver code taken from graph_electrostatics. |
| `mace_torch/electrostatics/reference/__init__.py` | The reference solver package. |
| `mace_torch/electrostatics/reference/energy.py` | `GTOElectrostaticEnergy`: the long-range energy for each periodicity mode. |
| `mace_torch/electrostatics/reference/features.py` | `GTOElectrostaticFeatures`: the potential projected onto each atom. |
| `mace_torch/electrostatics/reference/gto_utils.py` | Gaussian basis, normalizations, self-interaction and applied-field blocks. |
| `mace_torch/electrostatics/reference/kspace.py` | Reciprocal-space vectors and Fourier series. |
| `mace_torch/electrostatics/reference/realspace_electrostatics.py` | The open-system sum by displaced charges, over complete pair lists. |
| `mace_torch/electrostatics/reference/slabs.py` | The slab dipole correction and the molecule-in-box corrections. |
| `mace_torch/electrostatics/reference/utils.py` | Scatter and dense-batch helpers. |
| `mace_torch/electrostatics/solver.py` | The reference solver, the shared geometry, and resolving a solver once. |
| `mace_torch/finetune/__init__.py` | Fine-tuning, re-exported. |
| `mace_torch/finetune/foundation.py` | What a run reads from its foundation model. |
| `mace_torch/finetune/freeze.py` | Freezing parameter groups up to a level. |
| `mace_torch/finetune/lora.py` | Low-rank adapters in place of the weights they adapt. |
| `mace_torch/finetune/ratio.py` | Repeating the other heads when one head dominates. |
| `mace_torch/finetune/replay.py` | The published replay datasets. |
| `mace_torch/finetune/subselect.py` | Keeping a representative subset of a dataset. |
| `mace_torch/finetune/transfer.py` | Copying a foundation model's weights into a fine-tune. |
| `mace_torch/graph.py` | Tensorizing a collated graph. |
| `mace_torch/kernels/__init__.py` | The torch side of the kernel contract, re-exported. |
| `mace_torch/kernels/initialization.py` | Seeded per-op weight initialization. |
| `mace_torch/kernels/ops.py` | The dispatched ops as torch custom operators. |
| `mace_torch/legacy-extractor/extract_legacy.py` | Run by the pinned legacy venv to extract a pickle into the neutral format. |
| `mace_torch/models/__init__.py` | The model and its output layer, re-exported. |
| `mace_torch/models/base.py` | `MACEModel`: backbone, repulsion and output layer; the class models subclass. |
| `mace_torch/models/energy.py` | The energy head: E0s, scale and shift, and the two reductions. |
| `mace_torch/models/heads.py` | One readout head per declared observable, per head of theory. |
| `mace_torch/models/outputs.py` | `MACEOutputs`: every declared observable read out and typed. |
| `mace_torch/nn/__init__.py` | The building blocks, re-exported. |
| `mace_torch/nn/backbone.py` | `MACEBackbone`: embeddings, interactions and products, features out. |
| `mace_torch/nn/embedding.py` | Element and radial embeddings. |
| `mace_torch/nn/graph_features.py` | Graph-level inputs embedded into the node features. |
| `mace_torch/nn/interaction.py` | The interaction blocks: convolution, linear, skip. |
| `mace_torch/nn/layout.py` | Index maps between channel-major and irrep-grouped layouts. |
| `mace_torch/nn/node_inputs.py` | Declared per-node inputs mixed into the features. |
| `mace_torch/nn/product_basis.py` | The many-body product on the backend's contraction. |
| `mace_torch/nn/radial.py` | Radial bases, cutoffs, distance transforms and the ZBL repulsion. |
| `mace_torch/nn/radial_mlp.py` | The radial MLP that produces the convolution's path weights. |
| `mace_torch/physics/__init__.py` | The derivative engine, re-exported. |
| `mace_torch/physics/fixed_point.py` | Relaxing a declared input to a fixed point around the engine. |
| `mace_torch/physics/outputs.py` | `DerivativeEngine`: strain, forces, stress, virials, Hessian. |
| `mace_torch/serialization.py` | Checkpoints as canonical tensors and a JSON sidecar, no pickle. |
| `mace_torch/train/__init__.py` | The stages and the loop, re-exported. |
| `mace_torch/train/checkpoint.py` | Run checkpoints for resuming, with optimizer and schedule state. |
| `mace_torch/train/contracts.py` | The stage objects with their torch types bound. |
| `mace_torch/train/data_stage.py` | Stage one: configuration in, resolved data and loaders out. |
| `mace_torch/train/ddp.py` | Distributed data parallel setup and communication. |
| `mace_torch/train/ema.py` | The exponential moving average of the weights. |
| `mace_torch/train/full_batch.py` | The full-batch L-BFGS step. |
| `mace_torch/train/loaders.py` | Balancing several heads into one epoch of batches. |
| `mace_torch/train/logs.py` | Logging setup, once per run. |
| `mace_torch/train/loop.py` | The training loop and stage schedule. |
| `mace_torch/train/loss.py` | The loss generated from the declared observables. |
| `mace_torch/train/metrics.py` | Per-head, per-quantity error metrics. |
| `mace_torch/train/model_stage.py` | Stage two: configuration and data in, model out. |
| `mace_torch/train/optimizers.py` | Optimizers, parameter groups and schedules. |
| `mace_torch/train/slurm.py` | Reading a process's place in a SLURM job. |
| `mace_torch/train/tracking.py` | Reporting to an experiment tracker. |

### `mace-jax` and `mace-launcher`

| File | What is in it |
|---|---|
| `mace_jax/__init__.py` | The package version; inference code arrives with its tickets. |
| `mace_launcher/__init__.py` | The `mace_*` console scripts, dispatching on `--engine` or `MACE_ENGINE`. |
| `mace_launcher/audit.py` | The runtime guard that fails when a v1 module imports the legacy package. |

### Tests

Each package's `tests/` directory tests that package alone; `mace-core`'s run
with no torch installed. Three directories at the repository root test across
packages: `tests/architecture` holds the fitness functions (import direction,
packaging isolation, legacy-flag coverage, this guide), `tests/parity`
compares v1 with the frozen legacy model in one process, and `tests/golden`
holds the committed reference values both stacks are measured against.
