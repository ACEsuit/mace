# mace-torch-v1

The PyTorch stack for MACE v1: models, training, kernel backends and
deployment. It depends on `mace_core` and torch, and is the primary training
and research backend.

Distribution `mace-torch-v1`, import name `mace_torch`. The distribution name
carries a suffix only because the frozen legacy package holds `mace-torch`; it
is never published and loses the suffix at RET-6.

## Optional LES solver

Install the external latent Ewald solver from a V1 checkout with

```bash
python -m pip install ./packages/mace-core './packages/mace-torch[les]'
```

The `les` extra selects the solver commit used by the frozen LES references.
Set `model.model` to `les` and configure the multipoles under `model.les`.
The training CLI also accepts the historical model name `MACELES` and a
`--les_arguments` YAML file. Request Born effective charges with
`mace eval --compute_bec` or `MACECalculator(..., compute_bec=True)`.

MACE readouts supply the charges to the solver. A new training configuration
rejects `use_atomwise=True`. Conversion records that setting when present
in an old checkpoint and uses the charge readouts that produced its predictions.

## Native LAMMPS with MPI

Build the `mace/native` pair plugin against the same LAMMPS source revision,
shared library, MPI implementation and libtorch ABI as the target installation.
LAMMPS needs the PLUGIN package. Python and the V1 packages are needed on the
export/build host. The MD executable loads libtorch and the compiled model
directly.

```bash
cmake -S packages/mace-torch/lammps -B build/mace-native \
  -DLAMMPS_SOURCE_DIR=/path/to/lammps-source \
  -DLAMMPS_LIBRARY=/path/to/liblammps.so \
  -DPython3_EXECUTABLE="$(command -v python)"
cmake --build build/mace-native
mace --engine v1 export lammps model.safetensors --native 512 16000
```

The two capacities are per MPI rank. The node count includes owned atoms,
ghosts and one padding slot. The edge count includes directed neighbors
inside the model cutoff. Choose capacities for the largest local domain in
the trajectory. Exceeding either stops the run and reports which capacity
to increase. The neutral export without `--native` retains its three-file
format. Native export adds `native.pt2` and records target `lammps-native`
in version 1.0 of the deployment manifest.

For scalar element-conditioned skips, the native exporter groups node slots
by their one-hot species and applies the canonical channel matrix for each
group. Counts and row order are recomputed from each step's input. Whole tiles
keep the compiled shapes fixed across atom migration. Vector or mixed-precision
skips use the original backend operation. The optional
`mace_grouped_scalar_skips` field in the AOT package metadata records how many
layers use the grouped implementation.

```text
units metal
atom_style atomic
newton on
# Define the simulation box and atoms before setting the pair coefficients.
plugin load /path/to/build/mace-native/mace_native.so
pair_style mace/native /path/to/model.mace-deploy/native.pt2
pair_coeff * * H C O
```

List one element symbol per LAMMPS atom type in type order. The pair style
requests a full neighbor list, differentiates only the site energies owned
by this rank and returns the force on each end of each edge. For models
with multiple interaction layers, leave `--halo-exchange` enabled. LAMMPS
fills ghost features between layers and sends their adjoints back to the
owners. Periodic images, atom reordering and migration use LAMMPS's current
communication lists.

For NVIDIA cuEquivariance, install the V1 CUDA extra matching the torch build
and the native ops wheel at version 0.10.0. Set `CUEQ_OPS_DIR` at plugin
build time to the installed `cuequivariance_ops` directory (the directory
containing `lib/libcue_ops.so`). Export on a GPU of the target compute capability:

```bash
cmake -S packages/mace-torch/lammps -B build/mace-native \
  -DLAMMPS_SOURCE_DIR=/path/to/lammps-source -DLAMMPS_LIBRARY=/path/to/liblammps.so \
  -DPython3_EXECUTABLE="$(command -v python)" \
  -DCUEQ_OPS_DIR="$(python -c 'import cuequivariance_ops as c; from pathlib import Path; print(Path(c.__file__).parent)')"
cmake --build build/mace-native
mace --engine v1 export lammps model.safetensors --native 512 16000 \
  --backend cueq --device cuda --output model-cuda.mace-deploy
```

cuEq and libtorch must be visible to the dynamic loader on the compute nodes.
Ranks select devices by their node-local MPI rank modulo the number of visible
GPUs. The current plugin uses host buffers for LAMMPS neighbor data, feature
exchange and force writeback. The wheel includes the plugin sources in
`mace_torch/deploy/lammps/` for builds outside a source checkout.

## ML-IAP compatibility

The ML-IAP adapter loads the three-file neutral bundle and runs through
LAMMPS's Python callback interface. The native `mace/native` deployment above
provides the Python-free MD loop. Both targets use the canonical checkpoint
and the exported head and precision. V1 excludes TorchScript exports.

```bash
mace --engine v1 export lammps model.safetensors --halo-exchange --backend cueq
```

Activate the coupling and register the object before setting the pair style.
The CUDA KOKKOS coupling also needs CuPy to expose its device arrays. Install
the CuPy distribution matching the CUDA major used by LAMMPS and PyTorch.
Multi-rank CUDA execution requires a CUDA-aware MPI build with this LAMMPS
revision.
For a CUDA KOKKOS build, the setup in a Python driver is

```python
from lammps import lammps
from lammps.mliap import activate_mliappy_kokkos, load_unified_kokkos
from mace_torch.deploy.mliap import MACEMLIAPUnified

lmp = lammps(
    cmdargs=(
        "-k on g 1 -sf kk -pk kokkos newton on neigh half comm device gpu/aware on"
    ).split()
)
activate_mliappy_kokkos(lmp)
load_unified_kokkos(MACEMLIAPUnified("model.mace-deploy"))
# Create the box and atoms, then configure the interactions.
lmp.command("pair_style mliap/kk unified EXISTS 0")
lmp.command("pair_coeff * * H C O")
```

ML-IAP requests its full neighbor list internally. The KOKKOS package setting
`neigh half` permits the required `newton on` force reduction.
After the simulation, a standalone driver that owns MPI closes LAMMPS with
`lmp.close()`, then calls `lmp.lib.lammps_kokkos_finalize()` and
`lmp.lib.lammps_mpi_finalize()`. Python retains ownership of its interpreter.
The element names follow LAMMPS atom-type order. For a plain CPU ML-IAP build,
use `activate_mliappy`, `load_unified` and `pair_style mliap unified EXISTS 0`.
Single-layer models work with that interface. Multiple layers with
`halo_exchange=True` require KOKKOS forward and reverse ghost-feature exchange.
The adapter reports a missing exchange interface before model evaluation.

`--no-halo-exchange` selects evaluation over the complete receptive field.
Use `pair_style mliap unified EXISTS 1` so LAMMPS builds neighbors for ghosts.
The adapter requests the cutoff `r_max * num_interactions`. This mode needs
a coupling that accumulates forces for every supplied pair, including pairs
centered on ghosts. Validation uses LAMMPS `stable_22Jul2025_update4`, commit
`611ca3b7f8525ba802373b04f9e3d632d515f7e3`. An older local 22Jul2025 library
discarded those contributions and produced incorrect multi-layer forces.

The default `compile_regions=True` compiles interaction and product forwards
while communication stays between regions. `--no-compile-regions` selects
eager execution. The manifest's optional padding budget is applied before
the backend edge-order plan, and pair forces return in LAMMPS's original order.

Runtime controls retain their legacy names. `MACE_TIME` logs step duration.
`MACE_PROFILE` starts profiling at `MACE_PROFILE_START` (default 5) and stops
and exits at `MACE_PROFILE_END` (default 10). `MACE_ALLOW_CPU` permits a KOKKOS
callback carrying CPU tensors. `MACE_FORCE_CPU` selects CPU model evaluation.
Boolean values accept `true`, `1`, `t` and `yes`, case-insensitively.
