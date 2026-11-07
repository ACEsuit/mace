"""OpenMM units and a real direct-PyTorch simulation, without serialized code."""

import numpy as np
import pytest
import torch
from ase import Atoms, units
from conftest import fp64_only
from mace_torch.calculators import MACECalculator
from mace_torch.deploy.bundle import export_lammps
from mace_torch.deploy.openmm import OpenMMEvaluator
from test_mace_torch_calculator import water
from test_mace_torch_cli_commands import artifact as artifact

from tests.golden.harness import FP64_CPU_REFERENCE as ROW


def close(actual, expected):
    np.testing.assert_allclose(actual, expected, atol=ROW.atol, rtol=ROW.rtol)


@pytest.mark.parametrize("periodic", [False, True])
def test_openmm_callback_units_and_changing_neighbors(artifact, dtype, periodic):
    atoms = water(periodic)
    if periodic:
        atoms.set_cell([[4, 0, 0], [0.4, 4.3, 0], [0.2, 0.3, 4.5]])
    evaluator = OpenMMEvaluator(artifact, atoms.numbers, periodic=periodic)
    deployed = evaluator.deployed
    for displacement in (0.0, 0.15, 6.0, 0.0):
        current = atoms.copy()
        current.positions[2, 0] += displacement
        positions = torch.tensor(current.positions / 10, dtype=dtype)
        cell = torch.tensor(current.cell.array / 10, dtype=dtype) if periodic else None
        current.positions = positions.double().numpy() * 10
        if periodic:
            current.cell = cell.double().numpy() * 10
        current.calc = MACECalculator(artifact)
        with torch.inference_mode():
            energy, forces = evaluator.evaluate(positions, cell)
        assert not energy.requires_grad and not forces.requires_grad
        close(
            energy.cpu().numpy() * units.kJ / units.mol, current.get_potential_energy()
        )
        close(forces.cpu().numpy() * units.kJ / units.mol / 10, current.get_forces())
        assert evaluator.deployed is deployed


def test_openmm_refuses_inconsistent_inputs(artifact):
    with pytest.raises(ValueError, match="species"):
        OpenMMEvaluator(artifact, [26])
    with pytest.raises(ValueError, match="integer"):
        OpenMMEvaluator(artifact, [1.0])
    with pytest.raises(ValueError, match="not declared"):
        OpenMMEvaluator(artifact, [1, 8], inputs={"unknown": 1})
    evaluator = OpenMMEvaluator(artifact, [8, 1, 1], periodic=True)
    with pytest.raises(ValueError, match="cell"):
        evaluator.evaluate(torch.zeros(3, 3))
    with pytest.raises(ValueError, match="one three-vector"):
        evaluator.evaluate(torch.zeros(2, 3))


@fp64_only
@pytest.mark.bin_openmm
@pytest.mark.parametrize("periodic", [False, True])
def test_openmm_real_trajectory_matches_ase(artifact, periodic, monkeypatch):
    import openmm
    from openmm import unit
    from openmmtorch import PythonTorchForce

    def refuse_script(*args, **kwargs):
        raise AssertionError("OpenMM deployment must use the direct PyTorch interface")

    for name in ("script", "trace", "load", "save"):
        monkeypatch.setattr(torch.jit, name, refuse_script)
    atoms = water(periodic)
    bundle = export_lammps(artifact, halo_exchange=False)
    evaluator = OpenMMEvaluator(bundle, atoms.numbers, periodic=periodic)
    force = evaluator.as_force()
    assert isinstance(force, PythonTorchForce)
    assert force.usesPeriodicBoundaryConditions() == periodic
    system = openmm.System()
    for mass in atoms.get_masses():
        system.addParticle(float(mass))
    system.addForce(force)
    if periodic:
        system.setDefaultPeriodicBoxVectors(*(atoms.cell.array / 10))
    integrator = openmm.VerletIntegrator(0.0001)
    context = openmm.Context(
        system, integrator, openmm.Platform.getPlatformByName("Reference")
    )
    context.setPositions(atoms.positions / 10)
    try:
        for step in range(5):
            if periodic and step == 2:
                context.setPeriodicBoxVectors(
                    *([[0.42, 0, 0], [0.03, 0.43, 0], [0.02, 0.01, 0.45]])
                )
            state = context.getState(energy=True, forces=True, positions=True)
            current = Atoms(
                atoms.numbers,
                positions=state.getPositions(asNumpy=True).value_in_unit(unit.nanometer)
                * 10,
                cell=state.getPeriodicBoxVectors(asNumpy=True).value_in_unit(
                    unit.nanometer
                )
                * 10,
                pbc=periodic,
            )
            current.calc = MACECalculator(artifact)
            energy = state.getPotentialEnergy().value_in_unit(unit.kilojoules_per_mole)
            forces = state.getForces(asNumpy=True).value_in_unit(
                unit.kilojoules_per_mole / unit.nanometer
            )
            close(energy * units.kJ / units.mol, current.get_potential_energy())
            close(forces * units.kJ / units.mol / 10, current.get_forces())
            integrator.step(1)
    finally:
        del context, integrator
