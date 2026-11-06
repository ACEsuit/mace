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
in version 1.1 of the deployment manifest.

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
