# OpenMM

The contract tier checks dependency errors and prevents OpenMM's optional
callback serialization from creating a pickled model. Unit and trajectory
cases live in `packages/mace-torch/tests/test_mace_torch_openmm_integration.py`.
Real trajectories require `bin_openmm`; `MACE_REQUIRE_CAPS=bin_openmm` makes a
missing or broken plugin fail. The nightly job builds the pinned direct
PyTorch interface recorded in `requirements/openmm.txt`.

See [integration support](../../../docs/reforge/integration_support.md) for
the units, runtime scope and rebuild procedure.
