# Deployment integration support

V1 runtimes load canonical safetensors/JSON through `load_deployed`; native
runtimes execute AOTInductor artifacts compiled from those same models.
The full-basis weights export is a derived artifact for external contraction
implementations. It is not another model checkpoint.

| Integration | Verified scope | Remaining work |
| --- | --- | --- |
| ASE | Energy, response and magnetic models, units, padding and compiled execution | See the per-family parity records |
| LAMMPS native | Python-free torch.export/AOTInductor, reference/cuEq, changing neighbours and MPI migration on one/two GPUs | Performance results are recorded separately from correctness |
| ML-IAP | Direct callback, reference/cuEq, eager/compiled regions, one/two MPI ranks | Python callback is required by this compatibility interface |
| OpenMM | Direct PyTorch callback on CPU, molecular/periodic trajectories, changing triclinic cells, energy/force parity with ASE | GPU execution and host-transfer performance are not yet measured |
| Symmetrix | Public implementations inspected; target is the forthcoming general implementation | Concrete converter and benchmark sweep await its published interface |
| libmace | Native AOTInductor C/C++, Fortran on two MPI ranks, changing geometry, legacy E/F/virial parity; installed clients with cuEq float32/float64 on Blackwell | Full-structure root evaluation; distributed GPU scaling and MD throughput are not claimed |
| GROMACS native | Pinned 2026.3 NNPot adapter over libmace; CPU and B200 cuEq float32/float64 E/F/virial parity, changing cells/neighbours and two MPI domains | Whole-system model on the main rank; hybrid embedding, throughput and model scaling across GPUs are not claimed |
| torch-sim | CPU trajectory and legacy E/F/polar parity; static compile reuse; cuEq on H200 and RTX PRO 6000 Blackwell in float32/float64 | Host graph assembly remains; end-to-end throughput is not yet measured |

## OpenMM direct PyTorch interface

Verified versions: OpenMM 8.6.1, torch 2.11, OpenMM-Torch commit
`b6532900ad825cb3d12193817a1e34a7f46ae6a3`, built from source. The PyPI
`openmmtorch==1.5` wheel inspected during this work does not expose
`PythonTorchForce`; a matching version string alone is not sufficient.
The pinned build and capability-enforced trajectory tests are in
`.github/workflows/nightly.yaml`. These workflow changes do not schedule runs
on the rewrite branch before publication.

```python
from mace_torch.deploy.openmm import OpenMMEvaluator

callback = OpenMMEvaluator(
    "model.mace-deploy", atomic_numbers=[8, 1, 1], periodic=True,
)
system.addForce(callback.as_force())
```

The species sequence must match OpenMM's particle order, or the selected
particle subset if the caller uses `force.setParticles(...)`. The callback
reads the current box from State, rebuilds neighbours, and evaluates the
shared model. OpenMM supplies nanometres; the adapter converts them to
Angstrom. It returns kJ/mol and kJ/mol/nm. Inputs such as charge and fields
remain in MACE units. `callback.evaluate(positions, cell, inputs=...)` also
allows direct calls with updated declared inputs.

This path executes Python. It uses neither a scripted model nor a serialized
Python model. OpenMM's optional callback pickling is refused: after restoring
simulation state, recreate the callback from the same neutral bundle and
attach the force. The native LAMMPS runtime remains the Python-free route.
The current graph builder prepares host metadata and transfers positions for
neighbour preparation; this implementation makes no claim of zero host-copy
overhead on GPU.

## Full-basis weights export

```bash
mace export full-basis-weights model.mace-deploy --output model-full-basis.json
```

The command writes a JSON sidecar and a safetensors file with format identity
`mace-full-basis-weights`, schema `1.0`. Each `blocks.<index>` tensor has shape
`[Z, A, mul]`. `blocks` records its stable model component, input/output irreps,
correlation, channel count, dtype and species-row mapping. `groups` partitions
`A` by increasing body order, then output declaration order. Every group records
its offset, count, coupling-tree labels, full-basis Frobenius norms and forward
factors. The sidecar states the real-basis change and Wigner phase, including
signs that need not match another implementation's Wigner tables.

For each element and channel, `reduced_to_full` computes the minimum-norm
representative in float64 and writes the source dtype. The forward applies no
additional path factor. A shared-species contraction has one row explicitly
marked `shared`; otherwise rows follow `species`. Bundle runtime precision and
backend pins do not cast or alter the stored weights being exported.

Correctness is checked by full-basis model predictions against both committed
tiny-anchor E/F/stress references, plus full-to-reduced reconstruction under
the existing numerical bounds. Projection is function-preserving; floating
point weight reconstruction is not a bit-exact identity. The source files,
goldens and tolerances remain unchanged. Checkpoint readers reject the exported
format, and `mace model convert --to-basis full` remains refused.

The schema is implemented for review. External maintainer agreement on names
and ordering, requested by DEP-3, has not taken place; this work has not sent
messages or published artifacts. No consumer is declared compatible merely
because the weight export exists.

## External consumer findings

The intended Symmetrix target is the forthcoming general implementation
identified by the user on 2026-11-07, with
[`wcwitt/symmetrix-polar`](https://github.com/wcwitt/symmetrix-polar/tree/f8a23ea0f15dd036375dbaeae2e8542d85d964fc)
as a reference, rather than a Polar-only converter or Symmetrix-XL. The latter
is a separate project and was inspected before this clarification. No target
schema or compatibility promise is inferred for the unpublished general version.
V1 retains its canonical checkpoint and the component/full-basis export above;
the concrete consumer adapter and performance sweep remain pending the general
implementation's published interface. No Polar-specific architecture restrictions
are added to the V1 export contract. Symmetrix remains an optional external
runtime; MACE owns the export and parity checks.

The inspected Polar branch identifies itself as 0.1.0, with model format 1.
Its LAMMPS implementation supports fully periodic cells, charge and spin,
direct sums or a particle-mesh solver, with a specialized CUDA model shape.
Its own correctness report excludes stress, and per-atom virials are explicitly
refused. These are observations about that branch, not requirements for the
forthcoming general interface or evidence of V1 compatibility.

The earlier inspection below concerns the original public implementation:

Symmetrix revision `2aae0987ef63cc775f99f75b4fb45d8d0035a3ff` consumes an entire
model JSON: radial splines, embeddings, contractions, residuals and readouts.
Its supplied extractor reads a legacy model and imposes architecture limits,
including two interactions and l_max=3. Matching coupling-tree labels alone is
not a whole-model converter. The planned performance sweep must first verify
that translation, including spline approximation, on the same model.

libmace revision `e3413ae658530654894b6bac6a62a764a74c5115` loads
`torch::jit::Module` in `src/mace.cpp`; its Fortran interface wraps that runtime.
This differs from DEP-3's description of an independent full-basis contraction
reader. V1 now supplies a compatible C/C++ runtime in
`packages/mace-torch/libmace`, consuming the existing `native.pt2` artifact.
Its compiled weights retain the canonical reduced basis. The upstream scripted
reader is not linked, and the published upstream binary is not claimed to load V1.

The public C ABI and Fortran geometry/units convention are preserved; C++
applications rebuild against the framework-free header. The pinned neighbour
builder remains unchanged. The Fortran module retains its API and root-only
evaluation with MPI broadcasts. Its derived build source corrects an uninitialized
non-root pointer and a factor-of-1000 error in the barrier's sleep interval.
These change neither the model nor numerical results. MPI here does not mean
spatial domain decomposition or GPU scaling.

Build/export instructions are in
[the libmace integration guide](../../tests/integrations/libmace/README.md),
with measured scope in [the native runtime evidence](gate3/2026-11-07/dep3-libmace.md).
Installed clients use `find_package(mace CONFIG REQUIRED)` with `mace::mace`
or `mace::fortran`. The runtime needs matching libtorch/backend shared libraries,
but neither Python nor a scripted model. Fixed capacities are checked on every
call; changed counts and neighbours within those budgets reuse the exported graph.

Sources: [OpenMM-Torch](https://github.com/openmm/openmm-torch/tree/b6532900ad825cb3d12193817a1e34a7f46ae6a3),
[Symmetrix](https://github.com/wcwitt/symmetrix/tree/2aae0987ef63cc775f99f75b4fb45d8d0035a3ff),
[libmace](https://github.com/stenczelt/libmace/tree/e3413ae658530654894b6bac6a62a764a74c5115).


## TorchSim deployment

The V1 optional extra and `requirements/torchsim.txt` pin official TorchSim
revision `d6d8e9fc30129e01a05b551c2d71912821c9b68b` (package version 0.6.2).
The released 0.6.2 wheel still scripts two helpers during import; the pinned
revision removes them and fixes triclinic geometry. A fresh-process test blocks
`torch.jit.script`, `trace`, `load` and `save` before importing TorchSim, then
loads a V1 model and runs an NVE step. It also verifies that legacy `mace` was
not imported. No local dependency patch is applied.

```python
from mace_torch.deploy.torchsim import MaceTorchSimModel

model = MaceTorchSimModel(
    "model-bundle", device="cuda", backend="cueq", compile_mode="reduce-overhead"
)
result = model(state)
```

Import the adapter from `mace_torch.deploy.torchsim`: TorchSim's historical
`torch_sim.models.mace` wrapper still imports the legacy package. The adapter
uses the shared neutral loader, selects the head once, and resolves the kernel
and neighbour providers at construction. Runtime precision can be supplied as
V1's `PrecisionConfig`; a bundle's explicit pins still apply.

State cells are transposed from TorchSim's column vectors to MACE's row vectors.
Input features use their declarations and V1 names, including `total_charge`,
`total_spin` and `external_field`; `input_names` can map a declared name to an
older state attribute. An optional `displacement` is passed through shared
padding unchanged. When absent, only the derivative engine creates strain.
All outputs are detached and unpadded from their core or observable declarations.
Model diagnostics have their own declarations, so an output that echoes an input
can coexist with derivatives of that input without changing the catalogue's
naming rules.

Compilation uses the existing whole-model `dynamic=False` path. The default
compiled policy selects power-of-two atom, edge and system capacities, with a
dummy atom and system. `FixedPadding` can set explicit budgets; it follows the
shared overflow policy. Edge requirements are materialized by the engine after
padding. Changing a real count within the same capacities reuses the graph;
changing a capacity or another compiler guard can compile another graph.

The adapter currently uses the common host `Configuration`/graph builder.
GPU evaluation therefore does not imply an entirely device-resident MD step:
state metadata/positions are copied to the host during graph assembly. End-to-end
TorchSim throughput has not yet been measured. The CPU tests cover a real NVE
trajectory against ASE, both committed energy anchors, and eight polar comparisons
with the frozen model (molecules, a charged field-driven system, crystal and slab,
one and two systems). Polar stress uses V1 ASE's established strain convention;
the previously reconciled frozen legacy stress bug is not reintroduced.

The cuEq path passes eager-versus-Inductor checks on Dais H200 and RTX PRO 6000
Blackwell GPUs, for energy and Polar models in both float32 and float64 while
the real atom count changes. These are correctness checks, not throughput
measurements. See [DEP-4 evidence](gate3/2026-11-07/dep4-torchsim.md).

## GROMACS native integration

The user-authorized GROMACS work is additional to the original DEP-3 scope.
It uses GROMACS 2026.3 at `121090014570a53a17ea391bcddae45e5ea05eb4`, with a
small source patch selecting a `MACE` NNPot implementation and linking the
existing V1 native libmace runtime. The stock GROMACS TorchScript reader and
its Python dependency are excluded from this build path. No checkpoint,
weight layout or model implementation is added.

[Build and input instructions](../../tests/integrations/gromacs/README.md)
define the supported whole-system scope, units and restrictions. GROMACS
collects the full model input on its main rank and distributes forces to the
owning domains. The direct virial is included once with GROMACS's `-W/2`
convention. CPU and cuEq float32/float64 MD and rerun cases compare against
the frozen legacy model. The full Dais B200 selection passed all 17 tests
using external MPI; macOS additionally verifies thread-MPI. There is no
existing in-tree GROMACS legacy wrapper. See the [measured results and
scope](gate3/2026-11-07/gromacs-native.md).
