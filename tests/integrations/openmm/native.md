# Native OpenMM

`mace::MaceForce` connects OpenMM's public `CustomCPPForceImpl` interface to the
V1 libmace SDK. It evaluates the same AOTInductor `native.pt2` used by the native
LAMMPS and GROMACS integrations. Model execution and simulation require no
Python interpreter and no TorchScript. Python remains an export/build tool.

The verified SDK version is OpenMM 8.6.1. Its native C++ force interface handles
force groups, platform-specific force accumulation and transfers. MACE supplies
energy and forces through libmace. There is no additional checkpoint format.

## Build and use

Build and install [libmace](../libmace/README.md) first. OpenMM's wheel includes
the C++ SDK under `OpenMM.libs`; a source or conda SDK can also provide the same
headers and library. CUDA uses the matching `OpenMM-CUDA-13==8.6.1` package
alongside `openmm==8.6.1` and libmace built with cuEq support.

```sh
cmake -S packages/mace-torch/openmm -B openmm-native \
  -D CMAKE_BUILD_TYPE=Release \
  -D CMAKE_PREFIX_PATH="/path/to/libmace-install;/path/to/OpenMM.libs" \
  -D CMAKE_INSTALL_PREFIX="$PWD/openmm-native-install"
cmake --build openmm-native -j 4
cmake --install openmm-native
```

An installed CMake client uses `find_package(mace-openmm CONFIG REQUIRED)` and
links `mace::openmm`. Add the force to an existing OpenMM `System`:

```cpp
#include <MaceForce.h>
system.addForce(new mace::MaceForce("model.mace-deploy/native.pt2", {8, 1, 1}, true));
```

The atomic-number sequence must match all System particles in order. The last
argument selects full periodicity or no periodicity. The current Context box
is read on every evaluation, including after triclinic box updates. The native
export fixes the model backend, dtype, device and capacities. CUDA packages
must target the executing architecture. Use the matching OpenMM plugin directory
through `OPENMM_PLUGIN_DIR` in a standalone C++ application.

The force converts nm to Angstrom and eV to kJ/mol using the same ASE CODATA
2014 conversion as V1's existing OpenMM callback. It supports energy models
accepted by the native exporter. Additional charge/field inputs and hybrid
particle subsets are not introduced by this interface. The Python callback
remains available for its broader declared-input contract.

For periodic Polar CUDA packages, build with `-D MACE_OPENMM_CUDA=ON`.
This path reads OpenMM's CUDA position buffer, restores particle order and
periodic image offsets, constructs neighbours in CUDA and adds the native
forces directly to OpenMM's CUDA force buffer. Only requested energy scalars
and changed image counters cross the host boundary. Separate OpenMM and ATen
CUDA contexts are synchronized explicitly. The log reports
`MACE_OPENMM_DEVICE_IO cuda`; absence of that line means this route is not active.
The build installs a companion `libmace_openmm_cuda` module beside
`libmace_openmm`. Keep both files together. The adapter loads its CUDA platform
dependency with local symbol visibility, as OpenMM does for its own plugins;
other installed platforms can remain enabled.

CPU and local-energy packages use the CustomCPPForce host-array path. Complete
step timings include the transfers and synchronization used by the selected path. No Force XML serializer is supplied: build the
System force again from the same native artifact before restoring a State.

## Verification

Build the driver with `-D MACE_OPENMM_TEST_DRIVER=/absolute/path/to/tests/integrations/openmm/driver.cpp`.
Set `MACE_OPENMM_DRIVER` to the resulting executable and run:

```sh
MACE_REQUIRE_CAPS=bin_openmm_native pytest tests/parity/test_openmm_native.py -m 'not gpu'
# On a CUDA host with cuEq and OpenMM's CUDA plugin:
MACE_REQUIRE_CAPS=bin_openmm_native,gpu,cueq pytest tests/parity/test_openmm_native.py -m gpu
# Polar device buffers: charged triclinic cells, image offsets, double/mixed precision:
MACE_REQUIRE_CAPS=bin_openmm_native,gpu,cueq pytest tests/parity/test_native_polar.py -m gpu
```

The executable runs actual integration steps and geometry updates. Both frozen
energy anchors are compared with legacy at the measured positions. Force group
exclusion and library traces are checked in the real native process; the traces
must contain OpenMM/libmace and must not contain Python or libtorch_python.

[Measured CPU/GPU parity and complete-step performance](../../../docs/reforge/gate3/2026-11-07/deployment-completion.md)
include the installed client, cuEq float32/float64 and the host-transfer costs.
