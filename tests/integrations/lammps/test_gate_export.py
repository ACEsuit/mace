"""Exercise the real LAMMPS export CLI with a small untrained nonlinear model."""

import copy
import subprocess
import sys

import numpy as np
import pytest
import torch
from ase import Atoms

with torch.serialization.safe_globals([slice]):
    from e3nn import o3

from mace import modules
from mace.calculators import LAMMPS_MACE
from mace.modules.gate import GatedEquivariantBlock
from tests.integrations.lammps._harness import model_batch


@pytest.mark.parametrize("legacy", [False, True])
def test_lammps_cli_preserves_energy_forces_virial(tmp_path, legacy):
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(1713)
        model = modules.ScaleShiftMACE(
            r_max=3.0,
            num_bessel=4,
            num_polynomial_cutoff=5,
            max_ell=2,
            interaction_cls=modules.RealAgnosticResidualNonLinearInteractionBlock,
            interaction_cls_first=modules.RealAgnosticInteractionBlock,
            num_interactions=2,
            num_elements=2,
            hidden_irreps=o3.Irreps("8x0e + 8x1o"),
            MLP_irreps=o3.Irreps("4x0e"),
            gate=torch.nn.functional.silu,
            atomic_energies=np.array([-1.0, -5.0]),
            avg_num_neighbors=3.0,
            atomic_numbers=[1, 6],
            correlation=2,
            atomic_inter_scale=1.0,
            atomic_inter_shift=0.0,
        ).double()
    gates = [m for m in model.modules() if isinstance(m, GatedEquivariantBlock)]
    assert gates, "The model must actually exercise the affected nonlinear gate"
    reference = LAMMPS_MACE(copy.deepcopy(model))
    atoms = Atoms(
        numbers=[6, 1, 1], positions=[[0, 0, 0], [0.8, 0.3, 0], [-0.3, 0.9, 0.2]]
    )
    batch = model_batch(model, atoms)
    mask = torch.ones(len(atoms), dtype=torch.float64)
    expected = reference(copy.deepcopy(batch), mask, True)
    if legacy:
        for gate in gates:
            del gate._has_act_scalar
            del gate._has_act_gate
    checkpoint = tmp_path / "tiny.model"
    torch.save(model, checkpoint)
    process = subprocess.run(
        [sys.executable, "-m", "mace.cli.create_lammps_model", str(checkpoint)],
        text=True,
        capture_output=True,
        timeout=90,
    )
    assert process.returncode == 0, process.stdout + process.stderr
    exported = torch.jit.load(str(checkpoint) + "-lammps.pt")
    actual = exported(copy.deepcopy(batch), mask, True)
    for key in ("total_energy_local", "node_energy", "forces", "virials"):
        assert expected[key] is not None and actual[key] is not None
        assert torch.isfinite(actual[key]).all()
        torch.testing.assert_close(actual[key], expected[key], rtol=1e-9, atol=1e-10)
