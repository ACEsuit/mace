# Deployment and TorchSim

The [integration support matrix](../reforge/integration_support.md) records the
supported adapters and artifacts for ASE, LAMMPS ML-IAP, libmace, native OpenMM,
GROMACS and TorchSim. Follow that matrix for the current family/backend and
precision restrictions. The standalone Symmetrix interface remains deferred
pending the decision about a kernel backend.

LAMMPS ML-IAP exchanges ghost features between layers. A plain CPU coupling
without feature exchange can run only the supported single-layer path; a
successful import does not prove the coupling has the required entry points.
Native libmace/OpenMM/GROMACS drivers use separate executable capabilities.
Contract tests without their binaries cannot certify a real trajectory.

TorchSim requires its optional installation and supported Python version. Its
batch/cell conventions and output precision belong to the adapter contract,
not to a generic dictionary of model outputs. `torchsim` tests exercise this
boundary. Use the same model, head, units and initial conditions when comparing
ASE and another simulator. Export once from a native checkpoint; do not load a
legacy pickle at runtime.
