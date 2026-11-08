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
covers three foundation architectures. [Navigating the tree](#navigating-the-tree) says what is in
every file.

## Names

| Directory | Distribution | Import name | Tag prefix |
|---|---|---|---|
| `packages/mace-core` | `mace-core` | `mace_core` | `mace-core-v*` |
| `packages/mace-torch` | `mace-torch-v1` | `mace_torch` | `mace-torch-v*` |
| `packages/mace-jax` | `mace-jax` | `mace_jax` | `mace-jax-v*` |
| `packages/mace-launcher` | `mace-launcher` | `mace_launcher` | `mace-launcher-v*` |
| `packages/mace-backend-example` | `mace-backend-example` | `mace_backend_example` | `mace-backend-example-v*` |

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

The launcher owns `mace` and the `mace_*` console scripts. The JAX distribution
owns `mace-jax` for XYZ inference and installs independently. Each script name
has a single owner so uninstalling one distribution preserves the other's
commands.

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

**`mace-backend-example` is never published.** It is a kernel backend shaped
like a third party's wheel, depending on `mace-core` and torch alone and
registered only through its entry point, and it exists to be checked: CI runs
the whole conformance suite over it on every change, and nothing in the other
four packages may name it. Its README is the template for writing a backend.

## Checkpoints across the agreed backbone integration

The integrated V1 tree writes checkpoint format **3**, with raw linear and skip
weights and the reviewed component identities. Pre-integration V1 checkpoints
in format 2 are refused explicitly; their folded normalizations cannot be read
as raw weights. Keep the old V1 environment for those checkpoints. Legacy pickle
models can be extracted and converted again with the current converter.

Model settings now use `radial_basis.num_basis` and `cutoff.polynomial_order`;
interactions are named `plain`, `residual`, `density`, `density_residual` or
`nonlinear`. Legacy CLI flags still pass through the configuration translator.

Models take `ModelConfig` as their architecture description; element tables,
head counts and measured neighbour statistics are runtime inputs. Feature
tensors use the agreed `mul_ir` layout throughout. The calculator collates a
structure once and uses the same tensor padding as training.

Custom properties are declared in the optional `catalogue` section of the
training configuration, using `ObservableCatalogue` and `ObservableSpec`.
List the desired names in `model.observables` and their weights in `loss.weights`.
The data reader reads each custom label under its declared name, from the
per-atom or per-graph store the spec selects. Energy is optional for standard
models; repulsion, polar and magnetic energy models still require it. The
catalogue travels with the checkpoint, so loading needs no separate declaration.

Model and block classes bind stable checkpoint roles to their children through
`canonical_components()`. Plain containers retain their registered child names;
an empty role makes an implementation wrapper transparent. This preserves the
format-3 tensor names and values while allowing Python attributes to be renamed.
An undeclared component with canonical state is rejected when saving or loading.

## Source attribution

When code adapts or is strongly inspired by another project, add a nearby
comment naming the project and the relevant function or module, with a source
link (prefer a tag or commit). State whether the relationship is an adaptation,
API inspiration, or compatibility with a convention. Preserve upstream authorship
and license notices. A shared mathematical formula alone does not establish that
an implementation was copied.

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
with optional extensions. `mace-jax` uses JAX, Flax, safetensors and the core
matscipy extra. The launcher runs whichever engine is installed.

## Navigating the tree

What is in each directory, where to start reading, and one line for every
source file. The file tables are checked against the tree by
`tests/architecture/test_packages_readme.py`: a source file that is added,
moved or deleted without its row fails there, so the guide cannot go stale.

```
packages/
  mace-core/src/mace_core/       framework-free contract and pure math
    clebsch_gordan/              irreps, Wigner 3j, the reduced symmetric basis
    cli/                         the mace program; commands come from entry points
    config/                      the pydantic configuration, section by section
    data/                        structures, backends, E0s, statistics, splits
      backends/                  in-memory, XYZ, HDF5 and LMDB data backends
    data_spec/                   the v2 shard format and its writer
    electrostatics/              long-range solve descriptors, solver registry
    elements/                    element tables, default file keys, E0 values
    kernels/                     op descriptors, backend protocol and registry
    observables/                 declared observables and their derivatives
    weights/                     the neutral weights format
  mace-torch/src/mace_torch/     the PyTorch implementation
    backends/                    kernel backends, harmonics and radial functions
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
  mace-jax/src/mace_jax/         inference only; native JAX kernels and canonical checkpoint loading
  mace-launcher/src/mace_launcher/  console scripts, engine dispatch, import audit
```

### Where to start

**A training run**, in the order it happens:
`mace train` (`mace_core/cli/`, with the command in
`mace_torch/cli/commands.py`) reads the file and flags into
`mace_core/config/resolved.py`; `mace_torch/train/data_stage.py` reads the data
and resolves the E0s and statistics; `mace_torch/train/model_stage.py` builds
the model; `mace_torch/train/loop.py` trains it and
`mace_torch/train/checkpoint.py` writes it down.

**A forward pass**, from graph to forces: `mace_torch/data/graphs.py` builds
the graph `mace_core/graph/` declares; `mace_torch/physics/outputs.py` wraps
the model, applies the strain and takes the derivatives;
`mace_torch/models/base.py` runs `mace_torch/nn/backbone.py` and then
`mace_torch/models/outputs.py`, which returns `mace_core/outputs.py`'s typed
output.

**A new kernel backend** implements `mace_core/kernels/protocol.py`, reads its
ops from `mace_core/kernels/descriptors.py` and registers an entry point; the
reference in `mace_torch/backends/reference.py` is the one to copy.
**A new long-range solver** does the same against
`mace_core/electrostatics/capabilities.py` and `mace_torch/electrostatics/solver.py`.

**Neighbour providers and edge order:** `resolve_neighbor_provider` in
`mace_torch.neighbors` accepts names from `mace.neighbor_backends` and
`mace.neighbor_backends.torch`. Pass the kernel's `edge_features` as
`required_features`; the returned object records the provider and its edge
plan. `build_batch(..., neighbor_backend=provider)` uses optional batched
search and joins device edges after host collation. After padding, call
`apply_edge_plan(batch, provider.edge_plan)`. Stateful providers belong to one
system and cannot offer batched search. DATA-4 supplies this boundary API;
the training, calculator and deployment tickets own calling it in their flows.

**A fused edge convolution** matches `EdgeConvolutionSpan` from
`mace_core.kernels.spans` in `make_interaction_layer`. Declare kernel spec
`1.2` in the backend capabilities and return a stateless module implementing
`InteractionSpan` from `mace_torch.backends.spans`. That public module also
exports `SpanCapabilities`, `EdgeInputs` and `EdgeRuns` and documents the live
canonical weight tuple. The model selects the span only when its declared
weight-gradient and double-backward support meets the call's needs; otherwise
it uses the same model-owned members op by op. Checkpoints are independent of
that choice. `run_span_conformance` in `mace_torch.backends.span_conformance`
checks a registered backend on CPU or CUDA, including optional full-graph
compilation and CUDA replay. No accelerated span ships with this contract.

### `mace-core`

| File | What is in it |
|---|---|
| `mace_core/__init__.py` | The package version; the public surface is imported from the submodules. |
| `mace_core/clebsch_gordan/__init__.py` | The public surface of the basis: path order and normalization conventions. |
| `mace_core/clebsch_gordan/coefficients.py` | Complex Clebsch-Gordan and Wigner 3j coefficients from the Racah formula. |
| `mace_core/clebsch_gordan/conversion.py` | Converting weights between the full and reduced bases, and between layouts. |
| `mace_core/_entry_points.py` | Shared entry-point collision checks and import diagnostics. |
| `mace_core/_irreps.py` | Shared irreps syntax; callers retain their spelling and error contracts. |
| `mace_core/edges.py` | Edge-order vocabulary, provider capabilities and minimal-cost adapter plans. |
| `mace_core/clebsch_gordan/irreps.py` | O(3) irreps: parsing, ordering, dimensions, selection rules. |
| `mace_core/clebsch_gordan/real_basis.py` | The Wigner 3j table in the real spherical-harmonic basis the models use. |
| `mace_core/clebsch_gordan/reduced_basis.py` | The reduced symmetric tensor-product basis and its path order. |
| `mace_core/cli/__init__.py` | The `mace` program: the commands installed packages declare, parsed and run. |
| `mace_core/cli/command.py` | A command, its explicit flags, and how they write into the configuration file. |
| `mace_core/cli/registry.py` | Finding commands in the `mace.commands` entry point group, refusing conflicts. |
| `mace_core/config/__init__.py` | The configuration schemas, re-exported. |
| `mace_core/config/base.py` | The base schema: one file, one validation, unknown keys as errors. |
| `mace_core/config/data.py` | Datasets, heads, splits, transforms and the graph-input file keys. |
| `mace_core/config/e0s.py` | The ways a head's isolated-atom energies can be given or estimated. |
| `mace_core/config/electrostatics.py` | Which long-range solver a model uses, and for which systems. |
| `mace_core/config/fixed_point.py` | Describing a self-consistent relaxation loop. |
| `mace_core/config/legacy.py` | Every legacy training flag and what became of it, plus the translator. |
| `mace_core/config/loss.py` | Loss kinds and per-observable weights. |
| `mace_core/config/local_sources.py` | LC/LSC local multipoles and the exclusive formal-charge input/table contract. |
| `mace_core/config/qeq.py` | Quadratic charge equilibration and optional atomic input contracts. |
| `mace_core/config/model.py` | Architecture, declared observables, readouts, and the polar section. |
| `mace_core/config/precision.py` | Which dtype each op kind computes and accumulates in, the presets, and degradation on a device without float64. |
| `mace_core/config/provenance.py` | Turning what a run asked for into what its model records. |
| `mace_core/config/pseudolabels.py` | Replay labels from the foundation model: their settings and the record of what they were made from. |
| `mace_core/config/resolved.py` | The whole configuration and the rules that span its sections. |
| `mace_core/config/runtime.py` | Run name, output directory, seed, device, distribution, logging, plots. |
| `mace_core/config/section.py` | A section that is immutable once validated. |
| `mace_core/config/tracking.py` | Experiment tracking settings. |
| `mace_core/config/training.py` | Optimizers, learning-rate schedules and the training stages. |
| `mace_core/data/__init__.py` | The data layer's boundary objects, re-exported. |
| `mace_core/data/backend.py` | The protocol every data backend implements, and the statistics type. |
| `mace_core/data/backends/__init__.py` | The shipped backends and their registration. |
| `mace_core/data/backends/hdf5.py` | Prepared datasets of v2 shards, read lazily. |
| `mace_core/data/backends/hdf5_legacy.py` | The frozen tree's HDF5 files and statistics, read-only. |
| `mace_core/data/backends/lmdb.py` | Fairchem-style ASE LMDB databases, read-only. |
| `mace_core/data/backends/memory.py` | Structures already in memory, presented as a backend. |
| `mace_core/data/backends/xyz.py` | The XYZ backend over anything ase can read. |
| `mace_core/data/configuration.py` | `Configuration`: one parsed, labelled structure. |
| `mace_core/data/conformance.py` | The checks every data backend has to pass. |
| `mace_core/data/e0_resolution.py` | Turning an E0 declaration into one energy per element. |
| `mace_core/data/keys.py` | Which file key each property is read from. |
| `mace_core/data/registry.py` | Finding a data backend by name, with no fallback. |
| `mace_core/data/splitting.py` | Train and validation splits, and grouping for reports. |
| `mace_core/data/statistics.py` | Average neighbours, energy shift and scale, over any backend. |
| `mace_core/data/normalization.py` | Per-observable, per-head scales and invariant shifts from training labels. |
| `mace_core/data/xyz.py` | Reading labelled structure files into configurations. |
| `mace_core/data_spec/__init__.py` | The on-disk dataset formats, re-exported. |
| `mace_core/data_spec/shard_format.py` | The v2 shard layout, the manifest check and the shard writer. |
| `mace_core/electrostatics/__init__.py` | The electrostatics contract, re-exported. |
| `mace_core/electrostatics/capabilities.py` | What a solver declares it can do, and the refusal when it cannot. |
| `mace_core/electrostatics/descriptor.py` | One long-range solve as data: profile, multipoles, projection, method. |
| `mace_core/electrostatics/registry.py` | Solver discovery by entry point, and which solver a model loads with. |
| `mace_core/elements/__init__.py` | Element bookkeeping, re-exported. |
| `mace_core/elements/default_keys.py` | The default file keys of a labelled structure file. |
| `mace_core/elements/e0s.py` | Resolved isolated-atom energies, per head and element. |
| `mace_core/elements/number_table.py` | The ordered element table a model was fitted for. |
| `mace_core/foundation_roster.py` | The published foundation models, what each converts onto, and the one dropped. |
| `mace_core/kernels/__init__.py` | The kernel contract, re-exported. |
| `mace_core/kernels/canonical.py` | The canonical weight layout that makes checkpoints backend independent. |
| `mace_core/kernels/capabilities.py` | What a kernel backend declares it can build. |
| `mace_core/kernels/descriptors.py` | Every dispatched op, described before a backend builds it. |
| `mace_core/kernels/spans.py` | Closed edge-convolution span shape and kernel-version compatibility. |
| `mace_core/kernels/paths.py` | The tensor-product paths of the convolution and their order. |
| `mace_core/kernels/precision.py` | Dtype names, and which is wider. |
| `mace_core/kernels/protocol.py` | The interface a kernel backend implements. |
| `mace_core/kernels/reorder.py` | The map from canonical contraction weights to a backend's own path order, derived from two bases and split into its independent blocks. |
| `mace_core/kernels/registry.py` | Finding kernel backends by entry point. |
| `mace_core/metadata.py` | `ModelMetadata`: the versioned record every checkpoint carries. |
| `mace_core/citations.py` | Required method citations and deduplication across model metadata. |
| `mace_core/observables/__init__.py` | Declared observables, re-exported. |
| `mace_core/observables/defaults.py` | The default observable catalogue: energy with forces and stress. |
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

| `mace_core/artifacts.py` | Artifact directories, sidecars and versioned filenames. |
| `mace_core/model_registry.py` | Foundation identities, alias warnings, metadata queries and docs export, without frameworks. |
| `mace_core/model_registry.json` | Canonical roster, licenses, scope, hyperparameters and pending native publication references. |
| `mace_core/config/cli.py` | Typed command-line overrides for the nested configuration. |
| `mace_core/graph/__init__.py` | Public graph schema, construction, collation and padding API. |
| `mace_core/graph/build.py` | Build graph samples from configurations and a resolved neighbour backend. |
| `mace_core/graph/collate.py` | Batch numpy graph samples and preserve host metadata. |
| `mace_core/graph/padding.py` | Padding policies, budgets and padding metadata. |
| `mace_core/graph/schema.py` | Graph field shapes, storage levels, dtypes and padding values. |
| `mace_core/graph/validation.py` | Validate graph samples and batches at their boundaries. |
| `mace_core/neighbors/__init__.py` | Public neighbour-list protocol and registry. |
| `mace_core/neighbors/all_pairs.py` | The reference all-pairs neighbour-list implementation. |
| `mace_core/neighbors/backends.py` | Neighbour backend discovery, capabilities and resolution. |
| `mace_core/neighbors/geometry.py` | Periodic-cell geometry shared by neighbour backends. |
| `mace_core/neighbors/matscipy_backend.py` | The optional matscipy neighbour-list adapter. |
| `mace_core/neighbors/neighborhood.py` | Neighbourhood records and construction helpers. |

### `mace-torch`

| File | What is in it |
|---|---|
| `mace_torch/__init__.py` | The package version. |
| `mace_torch/backends/__init__.py` | The shipped kernel backends. |
| `mace_torch/backends/composite.py` | A chosen backend with the reference behind it, op by op, and how a name resolves to one. |
| `mace_torch/backends/conformance.py` | The checks every kernel backend passes against the reference. |
| `mace_torch/backends/spans.py` | Public stateless span call, capability declaration and build-time validation. |
| `mace_torch/backends/span_conformance.py` | Span values, derivatives, symmetry, edge order, padding and compiler conformance. |
| `mace_torch/backends/layout.py` | Views of feature tensors in the agreed canonical `mul_ir` layout. |
| `mace_torch/backends/cueq.py` | The cuEquivariance backend: linear, convolution, contraction and skip on NVIDIA's kernels. |
| `mace_torch/backends/oeq.py` | The OpenEquivariance backend: the fused convolution. |
| `mace_torch/backends/precision.py` | A backend whose ops are built at the precision configured for each, with casts at the boundary of those that differ. |
| `mace_torch/calculators/__init__.py` | The calculators, re-exported. |
| `mace_torch/calculators/ase_calculator.py` | `MACECalculator` for ASE: results, committees, Hessian, units. |
| `mace_torch/calculators/foundations.py` | Five native foundation loaders, shared registry resolution and MP dispersion options. |
| `mace_torch/calculators/padding.py` | Padding a batch to a fixed size for compiled GPU runs, and cutting it back. |
| `mace_torch/cli/__init__.py` | Console entry points only. |
| `mace_torch/cli/commands.py` | `mace train` and `mace model export-config`, declared for the `mace` program. |
| `mace_torch/train/plots.py` | Plotting shared by training and the standalone command. |
| `mace_torch/cli/migrate.py` | Migration commands and configuration files for legacy invocations. |
| `mace_torch/cli/reporting.py` | Print the citations recorded with an artifact. |
| `mace_torch/cli/plot_train.py` | Training-curve command over recorded metrics. |
| `mace_torch/cli/preprocess_data.py` | Prepare versioned shards through the shared data reader and statistics. |
| `mace_torch/cli/model.py` | Head selection and verified full-to-reduced basis conversion. |
| `mace_torch/cli/training_config.py` | Translate explicit flags and legacy configuration syntax into the V1 schema. |
| `mace_torch/cli/legacy_flags.py` | Legacy flag spellings and parser grammar at the input boundary. |
| `mace_torch/cli/eval_configs.py` | Batch extxyz evaluation, descriptors and the thirteen legacy output names. |
| `mace_torch/cli/convert_legacy.py` | `mace_convert_legacy`: a legacy checkpoint into a v1 one. |
| `mace_torch/cli/polar_density_cube.py` | `mace_polar_density_cube --engine v1`: a charge-aware model's density as cube files. |
| `mace_torch/cli/run_train.py` | `mace_run_train --engine v1`: training from a configuration file. |
| `mace_torch/data/__init__.py` | Graph building and batching, re-exported. |
| `mace_torch/data/augmentation.py` | Registered training-data augmentations, drawn on every read, and the magnetic-moment one. |
| `mace_torch/data/batch.py` | `TrainingBatch`, collation, the lazy graph dataset and its loader. |
| `mace_torch/data/distributed_sampler.py` | Splitting an epoch across the processes of a distributed run. |
| `mace_torch/data/graphs.py` | One structure into a graph and its training targets. |
| `mace_torch/data/transforms.py` | Registered transforms of what a run trains against. |
| `mace_torch/deploy/__init__.py` | Loading and converting models, re-exported. |
| `mace_torch/deploy/legacy.py` | Legacy conversion: extract in the legacy venv, import here. |
| `mace_torch/deploy/bundle.py` | Atomic export of a deployment directory from a canonical checkpoint pair. |
| `mace_torch/deploy/full_basis.py` | Export-only minimum-norm full-basis contraction tensors, with named paths and normalization. |
| `mace_torch/deploy/openmm.py` | OpenMM direct PyTorch callback, shared loader, neighbour provider and explicit unit conversion. |
| `mace_torch/deploy/torchsim.py` | TorchSim ModelInterface, declared state inputs/outputs, shared graph padding and static compile buckets. |
| `mace_torch/deploy/grouped_skip.py` | Lower scalar residual connections to fixed-shape per-species matrix products for native export. |
| `mace_torch/deploy/native.py` | Compile owned site energies and edge gradients for native LAMMPS and libmace, with explicit halo exchanges. |
| `mace_torch/deploy/manifest.py` | Deployment schema, selected head and halo exchange contract. |
| `mace_torch/deploy/mliap.py` | ML-IAP callback, first-order ghost exchange and owned-site edge derivatives. |
| `mace_torch/deploy/loader.py` | `load_deployed`: a checkpoint rebuilt for evaluation. |
| `mace_torch/deploy/neutral_io.py` | A neutral artifact into a v1 model through the checkpoint path. |
| `mace_torch/deploy/reference.py` | Checking a converted model against committed reference values. |
| `mace_torch/electrostatics/__init__.py` | The long-range ops and their dispatch, re-exported. |
| `mace_torch/electrostatics/density_cube.py` | Sampling a model's density on a grid, in reciprocal or real space, with quality metrics. |
| `mace_torch/electrostatics/reference/LICENSE.md` | The MIT license of the solver code taken from graph_electrostatics. |
| `mace_torch/electrostatics/reference/__init__.py` | The reference solver package. |
| `mace_torch/electrostatics/reference/energy.py` | `GTOElectrostaticEnergy`: the long-range energy for each periodicity mode. |
| `mace_torch/electrostatics/reference/features.py` | `GTOElectrostaticFeatures`: the potential projected onto each atom. |
| `mace_torch/electrostatics/reference/gto_utils.py` | Gaussian basis, normalizations, self-interaction and applied-field blocks. |
| `mace_torch/electrostatics/reference/kspace.py` | Reciprocal-space vectors and Fourier series. |
| `mace_torch/electrostatics/reference/qeq.py` | The analytic Gaussian potential for the alternate nonperiodic QEq equation. |
| `mace_torch/electrostatics/reference/realspace_electrostatics.py` | The open-system sum by displaced charges, over complete pair lists. |
| `mace_torch/electrostatics/reference/slabs.py` | The slab dipole correction and the molecule-in-box corrections. |
| `mace_torch/electrostatics/reference/utils.py` | Scatter and dense-batch helpers. |
| `mace_torch/electrostatics/solver.py` | The reference solver, the shared geometry, and resolving a solver once. |
| `mace_torch/finetune/__init__.py` | Fine-tuning, re-exported. |
| `mace_torch/finetune/extend.py` | Adding elements to a trained model: rows carried, new rows from a written spec. |
| `mace_torch/finetune/readouts.py` | Add initialized readouts while preserving every compatible trained operator. |
| `mace_torch/finetune/foundation.py` | What a run reads from its foundation model. |
| `mace_torch/finetune/freeze.py` | Freezing parameter groups up to a level. |
| `mace_torch/finetune/lora.py` | Low-rank adapters in place of the weights they adapt. |
| `mace_torch/finetune/pseudolabels.py` | Generating replay labels once per run, writing them exactly, and checking a reuse. |
| `mace_torch/finetune/ratio.py` | Repeating the other heads when one head dominates. |
| `mace_torch/finetune/replay.py` | The published replay datasets. |
| `mace_torch/finetune/subselect.py` | Keeping a representative subset of a dataset. |
| `mace_torch/finetune/stages.py` | A fine-tune as separate steps: select, relabel, extend, build, train. |
| `mace_torch/finetune/transfer.py` | Copying a foundation model's weights into a fine-tune. |
| `mace_torch/kernels/__init__.py` | The torch side of the kernel contract, re-exported. |
| `mace_torch/kernels/initialization.py` | Seeded per-op weight initialization. |
| `mace_torch/kernels/ops.py` | The dispatched ops as torch custom operators. |
| `mace_torch/legacy-extractor/extract_legacy.py` | Run by the pinned legacy venv to extract a pickle into the neutral format. |
| `mace_torch/models/__init__.py` | The model and its output layer, re-exported. |
| `mace_torch/models/base.py` | `MACEModel`: backbone, repulsion and output layer; the class models subclass. |
| `mace_torch/models/dipoles.py` | `DipoleModel`: a dipole, or a dipole and a polarizability, read off per-atom responses. |
| `mace_torch/models/electrostatics.py` | Polar density refinement and LC/LSC local sources, sharing solver dispatch and physical derivatives. |
| `mace_torch/models/registry.py` | Model registration used by data declarations and assembly, outside the forward path. |
| `mace_torch/models/energy.py` | The energy head: E0s, scale and shift, and the two reductions. |
| `mace_torch/models/heads.py` | One readout head per declared observable, per head of theory. |
| `mace_torch/models/les.py` | Long-range LES coupling, latent quantities, Born charges and their applied-field force. |
| `mace_torch/models/magnetic.py` | `MagneticModel`: an energy model reading a moment on every atom, with its one-body term. |
| `mace_torch/models/outputs.py` | `MACEOutputs`: every declared observable read out and typed. |
| `mace_torch/models/normalization.py` | Equivariant output normalization and its canonical checkpoint buffers. |
| `mace_torch/nn/__init__.py` | The building blocks, re-exported. |
| `mace_torch/nn/backbone.py` | `MACEBackbone`: embeddings, interactions and products, features out. |
| `mace_torch/nn/embedding.py` | Element and radial embeddings. |
| `mace_torch/nn/les.py` | Linear and nonlinear Cartesian tensor readouts for LES multipoles. |
| `mace_torch/nn/charge_transfer.py` | Canonical equivariant bond transfers and higher-multipole sources for LSC-MACE. |
| `mace_torch/nn/field.py` | The charge-aware blocks: density update, electron-energy readout, channel-pair products. |
| `mace_torch/nn/graph_features.py` | Declared graph or atomic inputs embedded into scalar node channels. |
| `mace_torch/nn/interaction.py` | The interaction blocks: convolution, linear, skip. |
| `mace_torch/nn/magnetic.py` | The moment-reading blocks: moment features, interaction and product couplings with the moment. |
| `mace_torch/nn/node_inputs.py` | Declared per-node inputs mixed into the features. |
| `mace_torch/nn/product_basis.py` | The many-body product on the backend's contraction. |
| `mace_torch/nn/radial.py` | Radial bases, cutoffs, distance transforms and the ZBL repulsion. |
| `mace_torch/nn/radial_mlp.py` | The radial MLP that produces the convolution's path weights. |
| `mace_torch/physics/__init__.py` | The derivative engine, re-exported. |
| `mace_torch/physics/charge_equilibration.py` | Constrained charge solving with implicit linear derivatives and a dispatched potential. |
| `mace_torch/physics/fixed_point.py` | Relaxing a declared input to a fixed point around the engine. |
| `mace_torch/physics/outputs.py` | `DerivativeEngine`: strain, forces, stress, virials, Hessian. |
| `mace_torch/serialization.py` | Checkpoints as canonical tensors and a JSON sidecar, no pickle. |
| `mace_torch/canonical.py` | Stable logical component identities for canonical checkpoint state. |
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

| `mace_torch/backends/harmonics.py` | Real spherical harmonics in the legacy convention. |
| `mace_torch/backends/radial.py` | Radial bases and the polynomial cutoff envelope. |
| `mace_torch/backends/reference.py` | Plain-torch kernels with raw canonical weights and forward normalization. |
| `mace_torch/graph/__init__.py` | Public torch graph collation and padding API. |
| `mace_torch/graph/collate.py` | Tensorize and batch graph samples at an explicit dtype. |
| `mace_torch/graph/edges.py` | Apply edge plans on device and expose stable node runs to backends. |
| `mace_torch/graph/padding.py` | Pad batches using the shared schema and isolated dummy graph. |
| `mace_torch/kernels/contraction.py` | Fused monomial contraction with differentiable backward and canonical paths. |
| `mace_torch/neighbors/__init__.py` | Resolve neighbour backends for PyTorch data construction. |
| `mace_torch/neighbors/providers.py` | Torch provider protocols, discovery, batch search and edge-plan resolution. |
| `mace_torch/neighbors/vesin_backend.py` | The vesin neighbour-list adapter. |
| `mace_torch/nn/edges.py` | Edge geometry and envelope preparation. |
| `mace_torch/nn/gate.py` | Scalar gates for grouped equivariant features. |

### `mace-jax` and `mace-launcher`

| File | What is in it |
|---|---|
| `mace_jax/__init__.py` | The installed package version; the JAX inference distribution. |
| `mace_launcher/__init__.py` | The `mace_*` console scripts, dispatching on `--engine` or `MACE_ENGINE`. |
| `mace_launcher/audit.py` | The runtime guard that fails when a v1 module imports the legacy package. |

| `mace_jax/neighbors.py` | Resolve framework-neutral neighbour backends for JAX. |
| `mace_jax/cli/__init__.py` | Inference command namespace. |
| `mace_jax/cli/eval.py` | The mace-jax XYZ inference command. |
| `mace_jax/deploy/__init__.py` | Canonical deployment namespace. |
| `mace_jax/deploy/neutral_io.py` | Strict safetensors/JSON loader and pure NNX apply boundary. |
| `mace_jax/graph.py` | Natural JAX batches and schema-defined inert padding. |
| `mace_jax/inference.py` | Jitted energy, geometry derivatives and reusable graph capacity. |
| `mace_jax/calculators.py` | Compiled ASE inference for a neutral single-head checkpoint. |
| `mace_jax/kernels/__init__.py` | Native kernel namespace. |
| `mace_jax/kernels/ops.py` | Canonical linear, convolution, reduced contraction and radial MLP operators. |
| `mace_jax/kernels/reference.py` | Reference backend factories, capabilities, harmonics and reductions. |
| `mace_jax/models/__init__.py` | Inference model namespace. |
| `mace_jax/models/energy.py` | Energy readouts, E0 assembly and backbone/output composition. |
| `mace_jax/models/registry.py` | The three supported foundation architectures and pending publication references. |
| `mace_jax/nn/__init__.py` | Neural building-block namespace. |
| `mace_jax/nn/backbone.py` | Four interaction variants and the descriptor-built product basis. |
| `mace_jax/nn/radial.py` | Bessel basis, polynomial envelope, Agnesi transform and ZBL repulsion. |
| `mace_jax/physics/__init__.py` | Derivative-output namespace. |
| `mace_jax/physics/outputs.py` | Forces, stress and virials from position and symmetric-strain gradients. |

### `mace-backend-example`

| File | What is in it |
|---|---|
| `mace_backend_example/__init__.py` | The backend the entry point names, and the package version. |
| `mace_backend_example/backend.py` | Every dispatched op in plain torch, its capabilities, and the canonical weights. |
| `mace_backend_example/ops.py` | Its own convolution, contraction and reduction kernels, each with a fake and a differentiable backward. |

### Tests

Each package's `tests/` directory tests that package alone; `mace-core`'s run
with no torch installed. Three directories at the repository root test across
packages: `tests/architecture` holds the fitness functions (import direction,
packaging isolation, legacy-flag coverage, this guide), `tests/parity`
compares v1 with the frozen legacy model in one process, and `tests/golden`
holds the committed reference values both stacks are measured against.
