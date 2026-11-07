# GROMACS native MACE integration

This integration connects GROMACS 2026.3's NNPot interface to V1 libmace and
its existing AOTInductor `native.pt2`. Model evaluation loads neither Python
nor a scripted model. Python is used to export the artifact and locate the
build dependencies. The canonical V1 checkpoint and reduced weights are unchanged.

This is a MACE integration patch for the pinned GROMACS source, not support
for loading the artifact in an unmodified GROMACS binary. The patch preserves
GROMACS's preprocessing, atom collection and rank-local force distribution.
Its `MACE` build path excludes `torchmodel.cpp` and the Python linkage used by
the upstream `TORCH` path.

## Build

First build and install [V1 libmace](../libmace/README.md). The Fortran module
is not needed by GROMACS (`MACE_WITH_FORTRAN=OFF`). For GPU model execution,
build libmace with `CUEQ_OPS_DIR` and the matching CUDA toolkit.

```sh
git clone https://github.com/gromacs/gromacs.git gromacs-source
git -C gromacs-source checkout 121090014570a53a17ea391bcddae45e5ea05eb4
cmake -D GROMACS_SOURCE_DIR="$PWD/gromacs-source" \
  -P packages/mace-torch/gromacs/apply.cmake
cmake -S gromacs-source -B gromacs-build \
  -D CMAKE_BUILD_TYPE=Release -D CMAKE_CXX_STANDARD=20 \
  -D GMX_NNPOT=MACE -D CMAKE_PREFIX_PATH="$PWD/libmace-install" \
  -D MACE_GROMACS_SOURCE_DIR="$PWD/packages/mace-torch/gromacs" \
  -D GMX_DOUBLE=ON -D GMX_GPU=OFF -D GMX_OPENMP=OFF \
  -D GMX_FFT_LIBRARY=fftpack -D GMX_SIMD=NONE -D BUILD_TESTING=OFF \
  -D GMXAPI=OFF -D GMX_INSTALL_NBLIB_API=OFF \
  -D GMX_USE_PLUMED=OFF -D GMX_USE_COLVARS=NONE \
  -D GMX_VERSION_STRING_OF_FORK=MACE-V1-AOT
cmake --build gromacs-build -j 2
```

The patch helper checks the exact revision and refuses other tracked edits,
leaving them untouched. Applying the same patch twice is harmless. A wheel
installation includes these sources in `mace_torch/deploy/gromacs`.

The GROMACS build uses double precision for the parity checks. `GMX_GPU=OFF`
refers to GROMACS's own classical kernels; a CUDA native MACE artifact still
executes on the GPU through libmace. Its device/backend/precision are pinned
at export time. The main rank uses the first visible GPU. Set
`CUDA_VISIBLE_DEVICES` accordingly. Neighbour construction uses the host;
complete-step measurements are recorded in the deployment evidence.

## Input

Use the existing native export, currently exposed by the LAMMPS subcommand:

```sh
mace export lammps model.json --native 1024 65536 --halo-exchange --output model.mace-deploy
```

CUDA/cuEq exports add `--device cuda --backend cueq` and are built on the
intended GPU architecture. The node budget includes one padding atom.
In the GROMACS `.mdp` file:

```ini
nnpot-active = true
nnpot-modelfile = /absolute/path/model.mace-deploy/native.pt2
nnpot-input-group = System
nnpot-model-input1 = atom-positions
nnpot-model-input2 = atom-numbers
nnpot-model-input3 = box
nnpot-model-input4 = pbc
```

The topology must supply correct atomic numbers and masses. The supported
scope is the whole-system energy model. Hybrid NNP/MM embedding, link atoms,
nonzero NNP charge and other model inputs are rejected. Neighbours
are rebuilt using the model's own cutoff. `grompp` checks the input contract;
the native artifact is loaded and validated by `mdrun`, so CUDA is not needed
merely to preprocess the topology.

Full periodicity and XY slabs are verified. GROMACS's Verlet scheme requires
two walls for `pbc=xy`; the parity test uses zero wall interactions so that the
MACE energy is compared independently. Fully nonperiodic `pbc=no` is refused
by this GROMACS version before model evaluation. This restriction does not
apply to libmace or native OpenMM, which both support isolated molecules.

Positions and box vectors convert from nm to Å, energy from eV to kJ/mol,
and forces to kJ/mol/nm. GROMACS's own SI constants define that conversion.
The adapter supplies the direct virial contribution `Xi=-W/2`, where
`W=-V*stress` is the libmace virial; GROMACS uses `2*(Ekin-Xi)/V` for pressure.
The full-system model executes on the main rank after GROMACS collects the
coordinates. Forces return to the owning ranks; energy and virial are added
once. This preserves domain decomposition correctness without promising
model scaling across multiple GPUs.

## Verification

```sh
export MACE_GROMACS_BINARY="$PWD/gromacs-build/bin/gmx_d"
MACE_REQUIRE_CAPS=bin_gromacs pytest tests/integrations/gromacs tests/parity/test_gromacs.py -m 'not gpu'
```

The default build uses GROMACS thread-MPI. An external MPI build adds
`GMX_MPI=ON` and `GMX_THREAD_MPI=OFF`; set `MACE_GROMACS_MPIEXEC` to `mpiexec`
and use the resulting `gmx_mpi_d` binary. GPU checks require `gpu,cueq` as well.

The suite compares each MD frame with frozen legacy, including molecular and
triclinic systems, a deforming cell, energy, forces and virials. Rerun cases
cross the model cutoff and move atoms between MPI domains, including an empty
domain. GROMACS intentionally omits virials from rerun output, so the MD cases
verify them separately. Loader traces check that the real simulation process
loads libmace/libtorch and no Python library. Reference files and tolerance
rows are unchanged. There is no in-tree legacy GROMACS adapter; the numerical
oracle is the frozen legacy model evaluated at the GROMACS geometries.

[Boundary validation and performance evidence](../../../docs/reforge/gate3/2026-11-07/deployment-completion.md)
record XY slabs, the nonperiodic restriction, the neighbour-radius correction,
and complete-step/host/MPI timings at 512 and 4096 atoms on Dais B200.
