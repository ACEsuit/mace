# Deployment integration support

V1 runtimes load canonical safetensors/JSON through `load_deployed`; native
runtimes execute AOTInductor artifacts compiled from those same models.
The full-basis weights export is a derived artifact for external contraction
implementations. It is not another model checkpoint.

| Integration | Interface | Limits |
| --- | --- | --- |
| ASE | Energy, response and magnetic models; padding and compiled execution | Family and precision restrictions apply |
| LAMMPS native | AOTInductor artifacts; reference/cuEq; neighbour and MPI adapters | Native Polar uses a fixed cell and one rank |
| ML-IAP | Python callback with reference/cuEq and eager/compiled regions | Requires compatible feature-exchange support for multiple layers |
| OpenMM | Native C++/AOTInductor and a direct Python callback | Native whole-System models; local-energy packages use host arrays; periodic Polar supports CUDA buffers |
| Symmetrix | No V1 adapter | Integration remains unimplemented |
| libmace | Native C/C++ and Fortran interfaces over AOTInductor | Full-structure root evaluation; no spatial model decomposition |
| GROMACS native | Pinned 2026.3 NNPot adapter over libmace | Whole-system main-rank evaluation; upstream Verlet rejects `pbc=no` |
| TorchSim | Shared model loader with eager or compiled execution | Graph assembly still uses host data |

Build instructions and executable contract tests accompany each adapter.
Performance depends on the model, neighbour construction, hardware and runtime;
use `tests/benchmarks/deployment` for complete-step measurements.

## Native Polar at fixed cell

The native exporter accepts Polar through a complete-system interface carrying
positions, the periodic cell and integer lattice shifts for local neighbours. The reciprocal-space
selection is prepared at export. Its cell derivatives remain in the graph,
so the shared V1 derivative engine produces both electrostatic forces and the full
virial. Edge vectors are formed inside the model in its selected precision.
The same artifact serves native LAMMPS, libmace, OpenMM and GROMACS.

```sh
mace export lammps polar-medium.json --native 96 16384 \
  --structure water-96.xyz --no-halo-exchange \
  --device cuda --backend cueq --dtype float32 --output polar-96
```

For Polar, the first capacity is the exact atom count, with no padding atom.
The edge capacity includes unused edges. The XYZ must be fully periodic with
a right-handed cell; its `total_charge`, `total_spin` and `external_field`
metadata fix those inputs for the artifact. Defaults are zero charge, spin
multiplicity one and zero field. The electrostatics solver is `reference`;
`cueq` selects the equivariant kernels independently.

Every runtime checks the cell and atom count before evaluation. Changing either
requires another export; this path supports fixed-cell dynamics, including NVE
and NVT. NPT, partial periodicity and runtime charge/field updates remain outside
this native Polar interface. The ASE and direct OpenMM Python interfaces retain
their existing dynamic inputs.

LAMMPS runs Polar with one MPI rank and atom IDs, using its periodic ghost
images only to build local edges. It rejects per-atom energy and virial requests
because Polar has no complete per-atom electrostatic partition. libmace's
`node_energy` retains V1's local readout; callers must use `total_energy` for
the energy including electrostatics. Global forces and virials include all terms.
The ML-IAP callback continues to accept only the local energy architectures.

For CUDA neighbour construction, build LAMMPS with `PKG_KOKKOS=ON`,
`Kokkos_ENABLE_CUDA=ON`, `KOKKOS_PREC=double` and `KOKKOS_LAYOUT=legacy`.
Build the plugin with `MACE_WITH_KOKKOS=ON`, pointing `CMAKE_PREFIX_PATH` at
that LAMMPS installation so it finds the matching Kokkos package. Select the
GPU architecture for both builds. The adapter is compiled as a separate CUDA
translation unit; the PyTorch runtime remains ordinary C++. Keep
`mace_native.so` and its companion `libmace_kokkos_neighbors.so` together when
copying the plugin. Both use the Kokkos runtime already owned by LAMMPS.

Run `lmp -k on g 1 -sf kk -pk kokkos neigh full newton off -in run.in`, with
`pair_style mace/native/kk /path/to/native.pt2`. The log must report a
`kokkos_device` neighbour list. This route currently accepts periodic CUDA
packages on one rank. The adapter consumes Kokkos's device neighbour list,
maps ghost images to their real owners on the GPU and writes forces back to
Kokkos's device array. Only requested global energy and virial values are
copied to the host. Complete-step timings include synchronization. Merely enabling `-k on` with
the ordinary `mace/native` style does not select GPU neighbours.

`tests/parity/test_native_polar.py` checks float32/float64 values and derivatives,
strict export capture, real C/C++ calls and rejection of changed cells.
`tests/benchmarks/deployment/polar_engines.py` runs the four-engine correctness
gates and complete-step measurements on repeated periodic water cells. All four
timed paths execute the same native AOT artifact. ASE uses
`NativeMACECalculator`; eager inference is used only for numerical checks.

## Native OpenMM

The C++ `mace::MaceForce` uses OpenMM 8.6.1's `CustomCPPForceImpl` and the
same libmace `native.pt2` artifact as GROMACS. Its simulation process loads
neither Python nor TorchScript. The native force accepts all System particles,
reads the current Context box, and converts OpenMM nm/kJ/mol to the existing
MACE units. Both isolated molecules and periodic triclinic cells are verified.
OpenMM handles force groups and platform-specific accumulation. CUDA/cuEq
float32 and float64 use the precision selected at export.

[Build and CMake client instructions](../../tests/integrations/openmm/native.md)
cover installation, the required CUDA plugin and the whole-System scope.
With `MACE_OPENMM_CUDA=ON`, periodic Polar packages exchange positions and
forces directly with OpenMM's CUDA buffers and use the shared CUDA cell list.
CPU and local-energy packages use the CustomCPPForce host-array path. Timings
include the complete MD step for the selected route. Additional charge/field inputs remain available
through the direct callback below.

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
attach the force. The native force above provides the Python-free energy route.
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

## Symmetrix

V1 has no Symmetrix adapter. The canonical checkpoint and full-basis weight
export do not establish compatibility with an external consumer. A runtime or
kernel-backend integration requires its own architecture checks and numerical
validation.

## Native libmace

`packages/mace-torch/libmace` provides a C/C++ runtime for the `native.pt2`
artifact. Its compiled weights retain the canonical reduced basis. Applications
must build against this runtime; the legacy TorchScript reader cannot load V1
artifacts.

The public C ABI and Fortran geometry/units convention are preserved; C++
applications rebuild against the framework-free header. The pinned neighbour
builder remains unchanged for CPU calls. Periodic CUDA packages use the shared
CUDA cell list, including calls through the C/Fortran API. The Fortran module
retains its API and root-only
evaluation with MPI broadcasts. Its derived build source corrects an uninitialized
non-root pointer and a factor-of-1000 error in the barrier's sleep interval.
These change neither the model nor numerical results. MPI here does not mean
spatial domain decomposition or GPU scaling.

Build/export instructions are in
[the libmace integration guide](../../tests/integrations/libmace/README.md).
Installed clients use `find_package(mace CONFIG REQUIRED)` with `mace::mace`
or `mace::fortran`. The runtime needs matching libtorch/backend shared libraries,
but neither Python nor a scripted model. Fixed capacities are checked on every
call; changed counts and neighbours within those budgets reuse the exported graph.

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

The adapter uses the common host `Configuration`/graph builder. Metadata and
positions are copied to the host during graph assembly, so GPU evaluation still
incurs host work. Measure complete MD steps separately from compiled evaluation.
The integration tests compare real NVE trajectories with ASE and frozen energy
and Polar references. Polar stress follows V1's strain convention.

## GROMACS native integration

The adapter uses GROMACS 2026.3 at `121090014570a53a17ea391bcddae45e5ea05eb4`, with a
small source patch selecting a `MACE` NNPot implementation and linking the
existing V1 native libmace runtime. The stock GROMACS TorchScript reader and
its Python dependency are excluded from this build path. No checkpoint,
weight layout or model implementation is added.

[Build and input instructions](../../tests/integrations/gromacs/README.md)
define the supported whole-system scope, units and restrictions. GROMACS
collects the full model input on its main rank and distributes forces to the
owning domains. This upstream NNPot interface exposes host arrays; the MACE
CUDA runtime builds its neighbours on the GPU after uploading positions. The
direct virial is included once with GROMACS's `-W/2`
convention. The integration tests compare CPU and cuEq float32/float64 MD and
rerun cases against frozen references, including changing cells, neighbours and
MPI ownership. See the integration guide for executable capability requirements.
