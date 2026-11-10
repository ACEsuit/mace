# Native libmace integration

V1 supplies an AOTInductor implementation of libmace's C++ class and `cmace_*`
C ABI. It compiles the neighbour-list source and MPI Fortran module
from `stenczelt/libmace` revision `e3413ae658530654894b6bac6a62a764a74c5115`.
The upstream scripted reader is not compiled or linked. This is a MACE-provided
runtime for that interface, not a claim that the upstream binary loads V1.

```sh
git clone https://github.com/stenczelt/libmace.git libmace-source
git -C libmace-source checkout e3413ae658530654894b6bac6a62a764a74c5115
cmake -S packages/mace-torch/libmace -B build-libmace \
  -D LIBMACE_SOURCE_DIR="$PWD/libmace-source" \
  -D Python3_EXECUTABLE="$(command -v python)" \
  -D MACE_TEST_DRIVER_DIR="$PWD/tests/integrations/libmace" \
  -D CMAKE_BUILD_TYPE=Release
cmake --build build-libmace -j 2
cmake --install build-libmace --prefix "$PWD/libmace-install"
```

The build needs libtorch from the exporting PyTorch installation, a C++20
compiler, and Fortran/MPI. Python locates the build dependencies; the resulting
library and application do not link Python. `MACE_WITH_FORTRAN=OFF` builds only
the C/C++ interface. A cuEq build also supplies `CUEQ_OPS_DIR` for the existing
0.10.0 native operator bindings and a matching CUDA toolkit.
Installed consumers use `find_package(mace CONFIG REQUIRED)` and link
`mace::mace` or `mace::fortran`. Keep the matching libtorch/cuEq shared libraries
available at runtime; the native package is specific to their ABI and device.
Build the Fortran module with the application's compiler and MPI installation.

Export with the existing native deployment command:

```sh
mace export lammps model.json --native 1024 65536 --halo-exchange --output model.mace-deploy
```

Use `MaceModel("model.mace-deploy/native.pt2", MPI_COMM_WORLD)` in Fortran, or
`MACE("model.mace-deploy/native.pt2")` in C++. CUDA exports add `--device cuda`
and optionally `--backend cueq`. Compile on the intended GPU architecture.
The native package owns dtype/device/backend/head selection. Atomic numbers
are read from its metadata. Node capacity includes one padding atom. Overflow and
unknown elements fail explicitly; there is no Python or scripted fallback.

For Polar, add `--structure water.xyz --no-halo-exchange` and give the exact
atom count as the node capacity. The structure fixes the fully periodic cell,
charge, spin multiplicity and field; changing the cell or count requires a new
export. `total_energy`, forces and virial include electrostatics, while
`node_energy` is the local readout. See the
[native Polar contract](../../../docs/reforge/integration_support.md#native-polar-at-fixed-cell).

C arrays use row-vector cells and `[n_atoms,3]` positions; Fortran uses
column-vector cells and `(3,n_atoms)`. Units are Å, eV and eV/Å. Virials are in
eV, ordered `xx, yy, zz, xy, xz, yz`, with `virial = -V * stress`. A disabled
virial is six zeros. A zero-volume slab still has a valid virial; choosing a
volume to report its stress is a caller decision. The C++ source API and C ABI
are preserved, but applications must rebuild against the new C++ header.
C++ validation throws; the established status-less C ABI reports the error and
aborts rather than unwinding an exception through Fortran.

Fortran MPI retains libmace's root-only full-structure evaluation and broadcasts
all results. This is not spatial domain decomposition or multi-GPU scaling.
The build applies two corrections to the pinned module: its unused C pointer
on non-root ranks starts as null, and its barrier polls at the intended 10 ms
interval instead of multiplying microseconds by another 1000. The upstream
checkout remains unchanged; the corrected module is included in the installation.
The exported halo operations are identity operations for this all-owned graph.
LAMMPS continues using its own distributed implementation of those operations.

## Verification

```sh
pytest tests/integrations/libmace -m 'not bin_libmace'
export MACE_LIBMACE_DRIVER="$PWD/build-libmace/mace_libmace_driver"
export MACE_LIBMACE_FORTRAN="$PWD/build-libmace/mace_libmace_fortran"
MACE_REQUIRE_CAPS=bin_libmace pytest tests/integrations/libmace tests/parity/test_libmace.py -m 'not gpu'
```

The real tier checks both committed energy anchors, C++ and C calls, compatible
Fortran on two MPI ranks, changing atom counts/neighbours/cells, reload,
per-atom energy, forces, physical virials and stress for nondegenerate periodic
cells. The library-load trace must contain libtorch and must not contain Python.
GPU cases add cuEq in float32 and float64 against frozen legacy. Reference
files and tolerances are unchanged. The real nightly is blocking; it is not
scheduled on the unpublished rewrite branch.

The ticket assumed libmace read full-basis contraction weights. Its published
reader actually executed a scripted model. The no-TorchScript replacement
therefore consumes V1's existing `native.pt2`, compiled from canonical reduced
weights. It needs neither another checkpoint format nor a full-basis expansion.
The separate full-basis export remains available for external implementations.

## Periodic CUDA neighbours and device input

A CUDA/cuEq build also installs `libmace_cuda_neighbors.so`. Periodic native
packages use its deterministic cell list with double precision distance tests,
including float32 models. CPU calls retain the pinned upstream neighbour code.
Both paths enforce the same strict squared-distance cutoff.

C++ GPU clients can call `MACE::calculate_cuda(cell, device_positions, numbers)`
with a contiguous `[N,3]` float64 CUDA buffer in Angstrom. The returned
`DeviceCalculation` owns its contiguous CUDA force buffer; `forces()` does not
copy it to the host. `energy()` transfers a scalar, and `to_host()` explicitly
materializes a normal `CalculationResult`. Device work uses the current ATen
stream on visible GPU 0. Clients using a separate CUDA context must synchronize
before handing over buffers and before consuming results.

ASE can use the same cell list through the registered `native_cuda` provider:

```python
from mace_core.neighbors import NeighborListConfig
neighbors = NeighborListConfig(
    kind="native_cuda",
    settings={"library": "/path/to/install/lib64/libmace_cuda_neighbors.so"},
)
calculator = MACECalculator(models=model, device="cuda", neighbor_list=neighbors)
```

This provider requires full periodicity. It returns graph arrays on CUDA and
reads back only the edge count for an exact allocation. ASE's integrator and
atom storage remain host based. The provider is explicit; it does not replace
the portable default on installations without the native CUDA library.

To execute the complete compiled Polar graph from ASE, use the same artifact
as the native engines:

```python
from mace_torch.calculators import NativeMACECalculator
atoms.calc = NativeMACECalculator(
    "polar-96/native.pt2",
    neighbor_library="/path/to/install/lib64/libmace_cuda_neighbors.so",
)
```

This calculator executes AOT-compiled energy, forces and virial, with a fixed
edge capacity and CUDA neighbour construction. It does not rebuild the model
or invoke eager autograd. The package fixes the fully periodic cell, atom count,
charge, spin and field. Changing these requires a new export; conflicting ASE
`info` inputs are rejected even when only cached properties are requested.
ASE positions, integration and returned arrays remain on the host.

`tests/parity/test_native_cuda_neighbors.py` compares exact neighbour sets with
ASE in orthogonal, triclinic, rotated, unwrapped and multiple-image geometries,
checks the strict cutoff boundary, reproducibility and capacity failure, and
exercises the torch provider including empty edge sets.
