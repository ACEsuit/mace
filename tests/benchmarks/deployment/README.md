# Deployment throughput

These opt-in programs measure the same converted MACE-MP-0 small checkpoint
on rattled periodic silicon crystals containing 512 and 4096 atoms. They use
cuEq float32 on one GPU. Compilation and artifact loading are excluded from
the steady-state measurements. No reference or tolerance is regenerated.

## Complete steps

Set the executable paths described in the native
[GROMACS](../../integrations/gromacs/README.md) and
[OpenMM](../../integrations/openmm/native.md) guides. For external MPI, set
`MACE_GROMACS_MPIEXEC` to `mpiexec`. Export the model on the intended GPU with
`N+1` node slots and the next power of two above `40*N` edge slots.

```sh
python -m tests.benchmarks.deployment.throughput model.json output/gromacs \
  --repeat 4 --mode gromacs --native model.mace-deploy/native.pt2
python -m tests.benchmarks.deployment.throughput model.json output/openmm \
  --repeat 4 --mode openmm --native model.mace-deploy/native.pt2
python -m tests.benchmarks.deployment.throughput model.json output/torchsim \
  --repeat 4 --mode torchsim
```

Repeat with `--repeat 8` and its larger native artifact. GROMACS uses one and
two MPI ranks on the same node; its model still executes on the main rank's
single GPU. Three runs each measure 51 steps after a ten-step reset, using
GROMACS's direct `ms/step` column. OpenMM times three batches of 60 integration
steps after ten warm-up steps. Both use a 0.1 fs timestep. The OpenMM driver
uses double precision for its integrator and the artifact's precision for MACE.
The GROMACS input retains `nstlist=1` from the migration harness, so it rebuilds
its domains every step. Its two-rank overhead includes this aggressive setting;
it must not be extrapolated to a production run with less frequent rebuilding.

TorchSim times ten NVE steps after three warm-up steps with CUDA
synchronization at each boundary. It also measures graph preparation including
host transfers, and compiled evaluation on a prepared graph. These diagnostics
are separate calls, so their sum need not equal the complete-step measurement.
The integration timestep is 0.001 in TorchSim units. The final state is checked
against eager V1 under the existing `FP32` bounds. TorchSim uses its default
power-of-two atom padding, while native artifacts use `N+1`; the runtime timing
comparison therefore includes different padding and integration implementations.
It cannot isolate Python overhead by subtraction.

## Diagnostic stages and MPI

`native_profile.py SOURCE OUTPUT` creates an instrumented copy of libmace's
source. Build it in a separate directory against the same pinned dependencies.
Use `MACE_PROFILE=1` and that directory in `LD_LIBRARY_PATH` for a separate
GROMACS run. The profile reports host neighbour construction, edge preparation,
H2D, synchronized model evaluation, D2H and force assembly. Its extra GPU
synchronization changes scheduling, so retain uninstrumented wall times as
the throughput result. The production runtime has no timing instrumentation.

```sh
mkdir -p output/validation
mpicxx -O2 -shared -fPIC tests/benchmarks/deployment/mpi_profile.cpp \
  -o output/validation/mpi_profile.so
python -m tests.benchmarks.deployment.validate output/validation \
  --case 4 model-512/native.pt2 --case 8 model-4096/native.pt2
```

The validation requires the cached frozen MACE-MP-0 small source whose SHA256
is pinned in the script. It compares native total energy, site energies and
forces to legacy float64 using the existing FP32 bounds. It also records the
time spent by each of two ranks in the position and force Allreduces, including waiting.
The non-root rank waits for the main rank's model evaluation; that elapsed
time must not be presented as network cost. The main-rank measurement captures
its collective wait and transfer after evaluation. Both buffers contain
`3*N` doubles. The pinned provider gathers positions before evaluating MACE
and distributing forces, so the harness verifies and separates the two calls
per evaluation. Local atom mapping and domain redistribution are additional
GROMACS work.

Keep every sample, dependency version, native artifact hash and Slurm outcome
with the result. Preparation failures and cancelled/dependency-only jobs do
not count as numerical evidence. The dated deployment evidence records the
measurements from Dais separately from this reproducible harness.
