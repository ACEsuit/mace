from __future__ import annotations

import json
import math
import os
import subprocess
import sys
from pathlib import Path
import ase.io
import numpy as np
import pytest
import torch
from ase import Atoms
from ase.io import read
from e3nn import o3

from mace import data
from mace.calculators import MACECalculator
from mace.calculators.foundations_models import mace_polar
from mace.modules import interaction_classes
from mace.modules.extensions import PolarMACE
from mace.tools import torch_geometric, utils

# pylint: disable=redefined-outer-name

try:
    from ase.calculators.fd import (
        calculate_numerical_forces,
        calculate_numerical_stress,
    )
except (ImportError, ModuleNotFoundError):

    def calculate_numerical_forces(
        atoms: Atoms,
        eps: float = 1e-6,
        iatoms=None,
        icarts=None,
        *,
        force_consistent: bool = False,
    ) -> np.ndarray:
        positions = atoms.get_positions().copy()
        natoms = len(atoms)
        forces = np.zeros((natoms, 3), dtype=float)
        atom_indices = range(natoms) if iatoms is None else iatoms
        cart_indices = range(3) if icarts is None else icarts

        for a in atom_indices:
            for c in cart_indices:
                displaced = positions.copy()
                displaced[a, c] += eps
                atoms.set_positions(displaced)
                eplus = atoms.get_potential_energy(force_consistent=force_consistent)

                displaced[a, c] -= 2.0 * eps
                atoms.set_positions(displaced)
                eminus = atoms.get_potential_energy(force_consistent=force_consistent)

                forces[a, c] = -(eplus - eminus) / (2.0 * eps)

        atoms.set_positions(positions)
        return forces

    def calculate_numerical_stress(
        atoms: Atoms,
        eps: float = 1e-6,
        voigt: bool = True,
        *,
        force_consistent: bool = True,
    ) -> np.ndarray:
        stress = np.zeros((3, 3), dtype=float)
        cell = atoms.cell.copy()
        volume = atoms.get_volume()
        for i in range(3):
            x = np.eye(3)
            x[i, i] = 1.0 + eps
            atoms.set_cell(cell @ x, scale_atoms=True)
            eplus = atoms.get_potential_energy(force_consistent=force_consistent)

            x[i, i] = 1.0 - eps
            atoms.set_cell(cell @ x, scale_atoms=True)
            eminus = atoms.get_potential_energy(force_consistent=force_consistent)

            stress[i, i] = (eplus - eminus) / (2 * eps * volume)
            x[i, i] = 1.0

            j = i - 2
            x[i, j] = x[j, i] = +0.5 * eps
            atoms.set_cell(cell @ x, scale_atoms=True)
            eplus = atoms.get_potential_energy(force_consistent=force_consistent)

            x[i, j] = x[j, i] = -0.5 * eps
            atoms.set_cell(cell @ x, scale_atoms=True)
            eminus = atoms.get_potential_energy(force_consistent=force_consistent)

            stress[i, j] = stress[j, i] = (eplus - eminus) / (2 * eps * volume)

        atoms.set_cell(cell, scale_atoms=True)
        return stress.flat[[0, 4, 8, 5, 2, 1]] if voigt else stress


try:
    import graph_longrange  # noqa: F401

    GRAPH_LONGRANGE_AVAILABLE = True
except (ImportError, ModuleNotFoundError):
    GRAPH_LONGRANGE_AVAILABLE = False
pytestmark = pytest.mark.skipif(
    not GRAPH_LONGRANGE_AVAILABLE, reason="graph_longrange is not installed"
)

REPO_ROOT = Path(__file__).resolve().parents[1]
RUN_TRAIN = REPO_ROOT / "mace" / "cli" / "run_train.py"

# ---------------------------------------------------------------------------
# Common helpers
# ---------------------------------------------------------------------------


def _water_atoms() -> Atoms:
    return Atoms(
        numbers=[8, 1, 1],
        positions=np.array(
            [
                [0.0, 0.0, 0.0],
                [0.9572, 0.0, 0.0],
                [-0.2390, 0.9270, 0.0],
            ],
            dtype=float,
        ),
    )


def _clone_batch(batch: dict) -> dict:
    return {k: (v.clone() if torch.is_tensor(v) else v) for k, v in batch.items()}


def _skip_if_model_unavailable(exc: Exception, model_name: str) -> None:
    if isinstance(exc, RuntimeError):
        msg = str(exc)
        if "Model download failed and no local model found" not in msg:
            raise exc
    pytest.skip(f"Polar model {model_name} not available")


def _random_rotation(device, dtype):
    A = torch.randn(3, 3, device=device, dtype=dtype)
    Q, _ = torch.linalg.qr(A)
    if torch.det(Q) < 0:
        Q[:, 0] = -Q[:, 0]
    return Q


# ---------------------------------------------------------------------------
# Model builders
# ---------------------------------------------------------------------------


def _build_full_model(
    device, dtype, *, r_max=6.0, num_bessel=8, num_polynomial_cutoff=6
):
    atomic_numbers = list(range(1, 84))
    num_elements = len(atomic_numbers)
    hidden_irreps = o3.Irreps("512x0e + 512x1o + 512x2e")
    MLP_irreps = o3.Irreps("16x0e")
    edge_irreps = o3.Irreps("128x0e + 128x1o + 128x2e")
    fixedpoint_update_config = {
        "type": "AgnosticEmbeddedOneBodyVariableUpdate",
        "potential_embedding_cls": "AgnosticChargeBiasedLinearPotentialEmbedding",
        "nonlinearity_cls": "MLPNonLinearity",
    }
    field_readout_config = {"type": "OneBodyMLPFieldReadout"}
    model = PolarMACE(
        r_max=r_max,
        num_bessel=num_bessel,
        num_polynomial_cutoff=num_polynomial_cutoff,
        max_ell=3,
        interaction_cls=interaction_classes[
            "RealAgnosticResidualNonLinearInteractionBlock"
        ],
        interaction_cls_first=interaction_classes[
            "RealAgnosticResidualNonLinearInteractionBlock"
        ],
        num_interactions=3,
        num_elements=num_elements,
        hidden_irreps=hidden_irreps,
        MLP_irreps=MLP_irreps,
        atomic_energies=torch.zeros(num_elements, dtype=dtype, device=device),
        avg_num_neighbors=3.0,
        atomic_numbers=atomic_numbers,
        correlation=3,
        gate=torch.nn.functional.silu,
        radial_MLP=[64, 64, 64],
        radial_type="bessel",
        kspace_cutoff_factor=1.0,
        atomic_multipoles_max_l=1,
        atomic_multipoles_smearing_width=1.5,
        field_feature_max_l=1,
        field_feature_widths=[1.5, 3.0],
        field_feature_norms=[20.0, 20.0, 0.5, 0.5],
        num_recursion_steps=2,
        field_si=False,
        include_electrostatic_self_interaction=True,
        add_local_electron_energy=True,
        field_dependence_type="AgnosticEmbeddedOneBodyVariableUpdate",
        final_field_readout_type="OneBodyMLPFieldReadout",
        return_electrostatic_potentials=False,
        heads=["Default"],
        field_norm_factor=1.0,
        fixedpoint_update_config=fixedpoint_update_config,
        field_readout_config=field_readout_config,
        edge_irreps=edge_irreps,
    ).to(device=device, dtype=dtype)
    return model


def _build_minimal_model(device, dtype):
    num_elements = 2
    atomic_numbers = [1, 8]
    hidden_irreps = o3.Irreps("4x0e + 4x1o")
    MLP_irreps = o3.Irreps("8x0e")

    fixedpoint_update_config = {
        "type": "AgnosticEmbeddedOneBodyVariableUpdate",
        "potential_embedding_cls": "AgnosticChargeBiasedLinearPotentialEmbedding",
        "nonlinearity_cls": "MLPNonLinearity",
    }
    field_readout_config = {
        "type": "OneBodyMLPFieldReadout",
    }

    model = PolarMACE(
        r_max=4.0,
        num_bessel=4,
        num_polynomial_cutoff=3,
        max_ell=1,
        interaction_cls=interaction_classes[
            "RealAgnosticResidualNonLinearInteractionBlock"
        ],
        interaction_cls_first=interaction_classes[
            "RealAgnosticResidualNonLinearInteractionBlock"
        ],
        num_interactions=2,
        num_elements=num_elements,
        hidden_irreps=hidden_irreps,
        MLP_irreps=MLP_irreps,
        atomic_energies=torch.zeros(num_elements, dtype=dtype, device=device),
        avg_num_neighbors=3.0,
        atomic_numbers=atomic_numbers,
        correlation=1,
        gate=torch.nn.functional.silu,
        radial_MLP=[16, 16],
        radial_type="bessel",
        kspace_cutoff_factor=1.0,
        atomic_multipoles_max_l=1,
        atomic_multipoles_smearing_width=1.0,
        field_feature_max_l=1,
        field_feature_widths=[1.0],
        field_feature_norms=[1.0, 1.0],
        num_recursion_steps=1,
        field_si=False,
        include_electrostatic_self_interaction=False,
        add_local_electron_energy=True,
        field_dependence_type="AgnosticEmbeddedOneBodyVariableUpdate",
        final_field_readout_type="OneBodyMLPFieldReadout",
        return_electrostatic_potentials=False,
        heads=["Default"],
        field_norm_factor=1.0,
        fixedpoint_update_config=fixedpoint_update_config,
        field_readout_config=field_readout_config,
    ).to(device=device, dtype=dtype)
    return model


def _build_minimal_batch(device, dtype):
    positions = torch.tensor(
        [[0.0, 0.0, 0.0], [1.2, 0.1, -0.2]], device=device, dtype=dtype
    )
    n = positions.shape[0]
    batch = torch.zeros(n, dtype=torch.long, device=device)
    ptr = torch.tensor([0, n], dtype=torch.long, device=device)

    src, dst = [], []
    for i in range(n):
        for j in range(n):
            if i != j:
                src.append(i)
                dst.append(j)
    edge_index = torch.tensor([src, dst], dtype=torch.long, device=device)
    shifts = torch.zeros((edge_index.shape[1], 3), dtype=dtype, device=device)

    node_attrs = torch.tensor([[1.0, 0.0], [0.0, 1.0]], device=device, dtype=dtype)

    cell_len = 10.0
    cell = torch.eye(3, dtype=dtype, device=device).unsqueeze(0) * cell_len
    rcell = torch.eye(3, dtype=dtype, device=device).unsqueeze(0) * (
        2.0 * math.pi / cell_len
    )
    volume = torch.ones((1,), dtype=dtype, device=device)
    pbc = torch.ones((1, 3), dtype=torch.bool, device=device)
    external_field = torch.zeros((1, 3), dtype=dtype, device=device)
    fermi_level = torch.zeros((1,), dtype=dtype, device=device)
    total_charge = torch.zeros((1,), dtype=dtype, device=device)
    total_spin = torch.zeros((1,), dtype=dtype, device=device)

    data_batch = {
        "positions": positions,
        "edge_index": edge_index,
        "shifts": shifts,
        "node_attrs": node_attrs,
        "batch": batch,
        "ptr": ptr,
        "cell": cell.view(-1, 9),
        "rcell": rcell.view(-1, 9),
        "volume": volume,
        "pbc": pbc,
        "external_field": external_field,
        "fermi_level": fermi_level,
        "total_charge": total_charge,
        "total_spin": total_spin,
        "density_coefficients": torch.zeros((n, 1), dtype=dtype, device=device),
    }
    return data_batch


def _build_model_batch(model: torch.nn.Module) -> dict:
    atoms = _water_atoms()
    z_table = utils.AtomicNumberTable([int(z) for z in model.atomic_numbers])
    keyspec = data.KeySpecification(
        info_keys={
            "total_spin": "spin",
            "total_charge": "charge",
            "external_field": "external_field",
        },
        arrays_keys={"Qs": "charges"},
    )
    config = data.config_from_atoms(
        atoms, key_specification=keyspec, head_name="Default"
    )
    loader = torch_geometric.dataloader.DataLoader(
        dataset=[
            data.AtomicData.from_config(
                config,
                z_table=z_table,
                cutoff=float(model.r_max),
                heads=model.heads,
            )
        ],
        batch_size=1,
        shuffle=False,
        drop_last=False,
    )
    batch = next(iter(loader)).to("cpu").to_dict()
    model_dtype = next(model.parameters()).dtype
    for key, value in batch.items():
        if torch.is_tensor(value) and value.dtype.is_floating_point:
            batch[key] = value.to(model_dtype)
    return batch


def _build_water_batch(device, dtype):
    O = torch.tensor([0.0000, 0.0000, 0.0000], dtype=dtype, device=device)
    H1 = torch.tensor([0.9572, 0.0000, 0.0000], dtype=dtype, device=device)
    H2 = torch.tensor([-0.2390, 0.9270, 0.0000], dtype=dtype, device=device)
    positions = torch.stack([O, H1, H2], dim=0)
    n = positions.shape[0]

    node_attrs = torch.zeros((n, 83), dtype=dtype, device=device)
    node_attrs[0, 7] = 1.0
    node_attrs[1, 0] = 1.0
    node_attrs[2, 0] = 1.0

    src, dst = [], []
    for i in range(n):
        for j in range(n):
            if i != j:
                src.append(i)
                dst.append(j)
    edge_index = torch.tensor([src, dst], dtype=torch.long, device=device)
    shifts = torch.zeros((edge_index.shape[1], 3), dtype=dtype, device=device)
    batch = torch.zeros(n, dtype=torch.long, device=device)
    ptr = torch.tensor([0, n], dtype=torch.long, device=device)
    cell_len = 12.0
    cell = torch.eye(3, dtype=dtype, device=device).unsqueeze(0) * cell_len
    rcell = torch.eye(3, dtype=dtype, device=device).unsqueeze(0) * (
        2.0 * math.pi / cell_len
    )
    data_batch = {
        "positions": positions,
        "edge_index": edge_index,
        "shifts": shifts,
        "node_attrs": node_attrs,
        "batch": batch,
        "ptr": ptr,
        "cell": cell.view(-1, 9),
        "rcell": rcell.view(-1, 9),
        "volume": torch.ones((1,), dtype=dtype, device=device),
        "pbc": torch.ones((1, 3), dtype=torch.bool, device=device),
        "external_field": torch.zeros((1, 3), dtype=dtype, device=device),
        "fermi_level": torch.zeros((1,), dtype=dtype, device=device),
        "total_charge": torch.zeros((1,), dtype=dtype, device=device),
        "total_spin": torch.zeros((1,), dtype=dtype, device=device),
        "density_coefficients": torch.zeros((n, 1), dtype=dtype, device=device),
    }
    return data_batch


# ---------------------------------------------------------------------------
# Invariance tests
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("dtype", [torch.float32])
def test_energy_invariance_under_rotation_and_translation(dtype):
    device = torch.device("cpu")
    torch.manual_seed(0)
    model = _build_minimal_model(device, dtype)
    data_batch = _build_minimal_batch(device, dtype)

    out = model(data_batch, training=False, compute_force=False)
    E = out["energy"].detach()

    torch.manual_seed(1)
    R = _random_rotation(device, dtype)
    data_rot = {
        k: (v.clone() if isinstance(v, torch.Tensor) else v)
        for k, v in data_batch.items()
    }
    data_rot["positions"] = data_batch["positions"] @ R.T
    data_rot["cell"] = (data_batch["cell"].view(-1, 3, 3) @ R.T).view(-1, 9)
    data_rot["rcell"] = (data_batch["rcell"].view(-1, 3, 3) @ R.T).view(-1, 9)
    out_rot = model(data_rot, training=False, compute_force=False)
    E_rot = out_rot["energy"].detach()

    assert torch.allclose(E, E_rot, atol=3e-1, rtol=1e-5)

    t = torch.tensor([0.5, -0.3, 0.8], device=device, dtype=dtype)
    data_tr = {
        k: (v.clone() if isinstance(v, torch.Tensor) else v)
        for k, v in data_batch.items()
    }
    data_tr["positions"] = data_batch["positions"] + t
    out_tr = model(data_tr, training=False, compute_force=False)
    E_tr = out_tr["energy"].detach()

    assert torch.allclose(E, E_tr, atol=5e-3, rtol=1e-5)


# ---------------------------------------------------------------------------
# Checkpoint evaluation
# ---------------------------------------------------------------------------

POLAR_MODELS = [
    # fukui-2 release digests:
    # 1L: 04379a98b3a6152eb4faa16721a4b4b8256ced50f062b4364ca2a14cdef057d3
    # 2L: 79358c165698dac456c656520cc2b78feac038e95f40ce14bdbc8db08cdbc60f
    # 3L: 49d3db53257822a50e6cd5e7d21c104718d4fb520fe618b2e232f090e1eeb409
    ("polar-1-l", -2079.86474609375),
    ("polar-1-m", -2079.86376953125),
    ("polar-1-s", -2079.86474609375),
]

POLAR_MODELS_FLOAT64 = [
    ("polar-1-l", -2079.8646699254614),
    ("polar-1-m", -2079.8637632777622),
    ("polar-1-s", -2079.864637957125),
]

POLAR_COMPONENTS = {
    "float32": {
        "polar-1-l": {
            "total": -2079.86474609375,
            "interaction": -0.5236091613769531,
            "electrostatic": 0.03348321095108986,
            "electron": 0.198118656873703,
            "local": -2080.096347961575,
        },
        "polar-1-m": {
            "total": -2079.863525390625,
            "interaction": -0.4711507260799408,
            "electrostatic": 0.03635868802666664,
            "electron": 0.14373645186424255,
            "local": -2080.043620530516,
        },
        "polar-1-s": {
            "total": -2079.86474609375,
            "interaction": -0.025278955698013306,
            "electrostatic": 0.022633790969848633,
            "electron": -0.2894555926322937,
            "local": -2079.5979242920876,
        },
    },
    "float64": {
        "polar-1-l": {
            "total": -2079.8646699254614,
            "interaction": -0.5236091496146468,
            "electrostatic": 0.03335496356847624,
            "electron": 0.19812072909037864,
            "local": -2080.0961456181203,
        },
        "polar-1-m": {
            "total": -2079.8637632777622,
            "interaction": -0.47115078949941797,
            "electrostatic": 0.03618856259411802,
            "electron": 0.14373541764930942,
            "local": -2080.043687257,
        },
        "polar-1-s": {
            "total": -2079.864637957125,
            "interaction": -0.02527944256184389,
            "electrostatic": 0.022633475333368347,
            "electron": -0.28945552139056663,
            "local": -2079.5978159110675,
        },
    },
}

POLAR_CHECKPOINT_ATOL = {"float32": 5e-4, "float64": 1e-8}
POLAR_COMPONENT_ATOL = {"float32": 5e-4, "float64": 1e-8}


@pytest.mark.parametrize("model_name, expected_energy", POLAR_MODELS)
def test_polar_checkpoint_evaluates(model_name, expected_energy):
    try:
        calc = mace_polar(model=model_name, device="cpu")
        model = mace_polar(model=model_name, device="cpu", return_raw_model=True)
    except (FileNotFoundError, ValueError, RuntimeError) as exc:
        _skip_if_model_unavailable(exc, model_name)

    assert model.__class__.__name__ == "PolarMACE"

    atoms = _water_atoms()
    atoms.calc = calc

    energy = atoms.get_potential_energy()
    assert np.isfinite(energy)
    assert abs(float(energy) - expected_energy) < POLAR_CHECKPOINT_ATOL["float32"]


@pytest.mark.parametrize("model_name, expected_energy", POLAR_MODELS_FLOAT64)
def test_polar_checkpoint_evaluates_float64(model_name, expected_energy):
    try:
        calc = mace_polar(model=model_name, device="cpu", default_dtype="float64")
        model = mace_polar(model=model_name, device="cpu", return_raw_model=True)
    except (FileNotFoundError, ValueError, RuntimeError) as exc:
        _skip_if_model_unavailable(exc, model_name)

    assert model.__class__.__name__ == "PolarMACE"

    atoms = _water_atoms()
    atoms.calc = calc

    energy = atoms.get_potential_energy()
    assert np.isfinite(energy)
    assert abs(float(energy) - expected_energy) < POLAR_CHECKPOINT_ATOL["float64"]


@pytest.mark.parametrize(
    "dtype_name,dtype", [("float32", torch.float32), ("float64", torch.float64)]
)
@pytest.mark.parametrize("model_name, _", POLAR_MODELS)
def test_polar_checkpoint_energy_components(dtype_name, dtype, model_name, _):
    try:
        model = mace_polar(model=model_name, device="cpu", return_raw_model=True).eval()
    except (FileNotFoundError, ValueError, RuntimeError) as exc:
        _skip_if_model_unavailable(exc, model_name)

    if dtype is torch.float64:
        model = model.double()

    previous_default_dtype = torch.get_default_dtype()
    try:
        torch.set_default_dtype(dtype)
        batch = _build_model_batch(model)
        out = model(_clone_batch(batch), compute_stress=False, training=False)
    finally:
        torch.set_default_dtype(previous_default_dtype)

    total = float(out["energy"][0].detach().cpu().item())
    interaction = float(out["interaction_energy"][0].detach().cpu().item())
    electrostatic = float(out["electrostatic_energy"][0].detach().cpu().item())
    electron = float(out["electron_energy"][0].detach().cpu().item())
    local = total - electrostatic - electron

    values = {
        "total": total,
        "interaction": interaction,
        "electrostatic": electrostatic,
        "electron": electron,
        "local": local,
    }
    expected = POLAR_COMPONENTS[dtype_name][model_name]
    atol = POLAR_COMPONENT_ATOL[dtype_name]

    for key, expected_value in expected.items():
        assert np.isfinite(values[key])
        assert abs(values[key] - expected_value) < atol


# ---------------------------------------------------------------------------
# Evaluation with charge and spin
# ---------------------------------------------------------------------------


def test_water_energy_changes_with_charge_and_spin():
    device = torch.device("cpu")
    dtype = torch.float32

    try:
        model = mace_polar(
            model="polar-1-l",
            device="cpu",
            return_raw_model=True,
        )
    except (FileNotFoundError, ValueError, RuntimeError) as exc:
        _skip_if_model_unavailable(exc, "polar-1-l")

    model = model.to(device=device, dtype=dtype)
    model.eval()

    data_batch = _build_water_batch(device, dtype)

    out0 = model(data_batch, training=False, compute_force=False)
    E0 = out0["energy"].detach()

    data_spin = {
        k: (v.clone() if isinstance(v, torch.Tensor) else v)
        for k, v in data_batch.items()
    }
    data_spin["total_spin"] = torch.tensor([1.0], dtype=dtype, device=device)
    out_spin = model(data_spin, training=False, compute_force=False)
    E_spin = out_spin["energy"].detach()

    data_charge = {
        k: (v.clone() if isinstance(v, torch.Tensor) else v)
        for k, v in data_batch.items()
    }
    data_charge["total_charge"] = torch.tensor([1.0], dtype=dtype, device=device)
    out_charge = model(data_charge, training=False, compute_force=False)
    E_charge = out_charge["energy"].detach()

    if torch.allclose(E0, E_spin) and torch.allclose(E0, E_charge):
        pytest.skip(
            "Model weights did not respond to spin/charge changes "
            "in this environment"
        )


# ---------------------------------------------------------------------------
# Fine-tuning
# ---------------------------------------------------------------------------


def _write_polar_data(tmp_path):
    configs = []
    atoms = Atoms(
        numbers=[8, 1, 1],
        positions=[[0, 0, 0], [1, 0, 0], [0, 1, 0]],
        cell=[6.0, 6.0, 6.0],
        pbc=[True, True, True],
    )
    rng = np.random.default_rng(123)
    for _ in range(4):
        sample = atoms.copy()
        sample.positions += rng.normal(0, 0.1, size=sample.positions.shape)
        sample.info["REF_energy"] = rng.normal(0, 1e-2)
        sample.new_array("REF_forces", rng.normal(0, 1e-2, size=sample.positions.shape))
        sample.info["REF_stress"] = rng.normal(0, 1e-2, size=6)
        configs.append(sample)
    path = tmp_path / "polar_smoke.xyz"
    ase.io.write(path, configs)
    return path, configs


def _write_polar_dipole_data(tmp_path):
    configs = []
    atoms = Atoms(
        numbers=[8, 1, 1],
        positions=[[0, 0, 0], [0.9572, 0, 0], [-0.2390, 0.9270, 0]],
        cell=[12.0, 12.0, 12.0],
        pbc=[False, False, False],
    )
    rng = np.random.default_rng(321)
    for _ in range(6):
        sample = atoms.copy()
        sample.positions += rng.normal(0, 0.08, size=sample.positions.shape)
        sample.info["REF_energy"] = float(rng.normal(0, 1e-2))
        sample.new_array("REF_forces", rng.normal(0, 1e-2, size=sample.positions.shape))
        sample.info["REF_dipoles"] = rng.normal(0, 1e-2, size=3)
        configs.append(sample)
    path = tmp_path / "polar_dipole_smoke.xyz"
    ase.io.write(path, configs)
    return path, configs


def _write_polar_multihead_data(tmp_path, configs):
    dft_configs = []
    mp2_configs = []
    for i, atoms in enumerate(configs):
        sample = atoms.copy()
        if i % 2 == 0:
            sample.info["head"] = "DFT"
            dft_configs.append(sample)
        else:
            sample.info["head"] = "MP2"
            mp2_configs.append(sample)

    dft_path = tmp_path / "polar_dft.xyz"
    mp2_path = tmp_path / "polar_mp2.xyz"
    ase.io.write(dft_path, dft_configs)
    ase.io.write(mp2_path, mp2_configs)
    return dft_path, mp2_path


def _write_heads_config(path, dft_path, mp2_path):
    yaml_str = "\n".join(
        [
            "heads:",
            "  DFT:",
            f"    train_file: {dft_path}",
            "    E0s: foundation",
            "  MP2:",
            f"    train_file: {mp2_path}",
            "    E0s: foundation",
        ]
    )
    path.write_text(yaml_str, encoding="utf-8")


def _run_train(params, extra_env=None):
    run_env = os.environ.copy()
    repo = Path(__file__).resolve().parents[1]
    pythonpath = run_env.get("PYTHONPATH")
    run_env["PYTHONPATH"] = ":".join(filter(None, [str(repo), pythonpath]))
    if extra_env:
        run_env.update(extra_env)
    cmd = [sys.executable, str(RUN_TRAIN)]
    for key, value in params.items():
        if value is None:
            cmd.append(f"--{key}")
        else:
            cmd.append(f"--{key}={value}")
    try:
        subprocess.run(cmd, env=run_env, check=True, capture_output=True, text=True)
    except subprocess.CalledProcessError as exc:
        print(exc.stdout)
        print(exc.stderr)
        if "Model download failed and no local model found" in (
            exc.stdout or ""
        ) or "Model download failed and no local model found" in (exc.stderr or ""):
            pytest.skip("Polar foundation model not available in this environment")
        raise


def _assert_model_predicts(model_path, configs, heads=("Default",)):
    for head in heads:
        calc = MACECalculator(
            model_paths=model_path,
            device="cpu",
            default_dtype="float64",
            head=head,
        )
        for atoms in configs:
            test_atoms = atoms.copy()
            test_atoms.calc = calc
            assert np.isfinite(test_atoms.get_potential_energy())


def _base_train_params(tmp_path, train_file, name):
    return {
        "name": name,
        "train_file": train_file,
        "checkpoints_dir": str(tmp_path),
        "model_dir": str(tmp_path),
        "valid_fraction": 0.2,
        "model": "PolarMACE",
        "hidden_irreps": "16x0e",
        "r_max": 3.5,
        "batch_size": 2,
        "max_num_epochs": 1,
        "lr": 1e-4,
        "loss": "stress",
        "default_dtype": "float64",
        "device": "cpu",
        "E0s": "average",
        "energy_key": "REF_energy",
        "forces_key": "REF_forces",
        "stress_key": "REF_stress",
    }


def test_run_train_polar_finetuning_from_checkpoint(tmp_path):
    train_file, configs = _write_polar_data(tmp_path)
    base_params = _base_train_params(tmp_path, train_file, "polar_base")
    base_params["valid_fraction"] = 0.3
    _run_train(base_params)

    ft_params = {
        **base_params,
        "name": "polar_ft",
        "foundation_model": str(tmp_path / "polar_base.model"),
        "force_mh_ft_lr": True,
    }
    _run_train(ft_params)

    model_path = tmp_path / "polar_ft.model"
    model = torch.load(model_path, map_location="cpu", weights_only=False)
    assert model.__class__.__name__ == "PolarMACE"
    assert model.heads == ["Default"]
    _assert_model_predicts(model_path, configs, heads=("Default",))


@pytest.mark.parametrize("foundation_model", ["polar-1-m", "polar-1-l"])
def test_run_train_polar_finetuning_foundation_model(tmp_path, foundation_model):
    train_file, configs = _write_polar_data(tmp_path)
    params = _base_train_params(tmp_path, train_file, f"polar_{foundation_model}_ft")
    params["foundation_model"] = foundation_model
    params["force_mh_ft_lr"] = True
    params["loss"] = "weighted"
    params["stress_weight"] = 0.0
    _run_train(params)

    model_path = tmp_path / f"polar_{foundation_model}_ft.model"
    model = torch.load(model_path, map_location="cpu", weights_only=False)
    assert model.__class__.__name__ == "PolarMACE"
    assert model.heads == ["Default"]
    if foundation_model == "polar-1-m":
        _assert_model_predicts(model_path, configs, heads=("Default",))


def test_run_train_polar_finetuning_energy_forces_dipole(tmp_path):
    train_file, configs = _write_polar_dipole_data(tmp_path)
    params = _base_train_params(tmp_path, train_file, "polar_dipole_ft")
    params.update(
        {
            "foundation_model": "polar-1-m",
            "force_mh_ft_lr": True,
            "loss": "energy_forces_dipole",
            "stress_weight": 0.0,
            "dipole_key": "REF_dipoles",
            "dipole_weight": 1.0,
        }
    )
    _run_train(params)

    model_path = tmp_path / "polar_dipole_ft.model"
    model = torch.load(model_path, map_location="cpu", weights_only=False)
    assert model.__class__.__name__ == "PolarMACE"
    assert model.heads == ["Default"]

    calc = MACECalculator(
        model_paths=model_path,
        device="cpu",
        default_dtype="float64",
        head="Default",
    )
    for atoms in configs:
        test_atoms = atoms.copy()
        test_atoms.info["charge"] = 0
        test_atoms.info["spin"] = 1
        test_atoms.calc = calc

        energy = test_atoms.get_potential_energy()
        forces = test_atoms.get_forces()
        dipole = np.asarray(calc.results["dipole"])

        assert np.isfinite(energy)
        assert np.all(np.isfinite(forces))
        assert np.all(np.isfinite(dipole))


@pytest.mark.parametrize("foundation_model", ["polar-1-m", "polar-1-l"])
def test_run_train_polar_multihead_finetuning_foundation_model(
    tmp_path, foundation_model
):
    pt_train_file, configs = _write_polar_data(tmp_path)
    dft_path, mp2_path = _write_polar_multihead_data(tmp_path, configs)
    config_path = tmp_path / "polar_heads.yaml"
    _write_heads_config(config_path, dft_path, mp2_path)

    params = {
        "name": "polar_2l_mh",
        "config": config_path,
        "pt_train_file": pt_train_file,
        "checkpoints_dir": str(tmp_path),
        "model_dir": str(tmp_path),
        "valid_fraction": 0.2,
        "model": "PolarMACE",
        "hidden_irreps": "16x0e",
        "r_max": 3.5,
        "batch_size": 2,
        "max_num_epochs": 1,
        "lr": 1e-4,
        "loss": "weighted",
        "force_mh_ft_lr": True,
        "num_samples_pt": 4,
        "default_dtype": "float64",
        "foundation_model": foundation_model,
        "device": "cpu",
    }
    _run_train(params)

    model_path = tmp_path / "polar_2l_mh.model"
    model = torch.load(model_path, map_location="cpu", weights_only=False)
    assert model.__class__.__name__ == "PolarMACE"
    assert set(model.heads) == {"DFT", "MP2", "pt_head"}
    _assert_model_predicts(model_path, configs, heads=("DFT", "MP2", "pt_head"))


# ---------------------------------------------------------------------------
# Finite difference
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def polar_calc_fd() -> MACECalculator:  # pylint: disable=inconsistent-return-statements
    try:
        calc = mace_polar(
            model="polar-1-m",
            device="cpu",
            default_dtype="float64",
        )
    except (FileNotFoundError, ValueError, RuntimeError):
        pytest.skip("Polar model polar-1-m not available")
    return calc


def _periodic_water(cell: np.ndarray) -> Atoms:
    atoms = Atoms(
        numbers=[8, 1, 1],
        positions=np.array(
            [
                [0.1, 0.2, 0.3],
                [1.0572, -0.11, 0.05],
                [-0.199, 0.947, 0.12],
            ],
            dtype=float,
        ),
        cell=cell,
        pbc=True,
    )
    atoms.info["charge"] = 0
    atoms.info["spin"] = 1
    return atoms


def test_polar_forces_match_finite_difference(
    polar_calc_fd: MACECalculator,
) -> None:
    atoms = _periodic_water(np.diag([12.3, 11.7, 10.9]))
    atoms.calc = polar_calc_fd

    forces = atoms.get_forces()
    forces_fd = calculate_numerical_forces(atoms, eps=1e-5, force_consistent=False)

    np.testing.assert_allclose(forces, forces_fd, rtol=0.0, atol=1e-6)


@pytest.mark.parametrize(
    "cell, atol",
    [
        (
            np.array([[40.0, 2.0, 0.0], [0.5, 42.0, 1.5], [0.0, 0.2, 38.0]]),
            2.0e-7,
        ),
        (
            np.array([[60.0, 3.0, 0.5], [1.0, 63.0, 2.0], [0.2, 0.5, 57.0]]),
            6.0e-8,
        ),
        (
            np.array([[80.0, 4.0, 1.0], [1.5, 84.0, 2.5], [0.4, 0.8, 76.0]]),
            3.0e-8,
        ),
    ],
)
def test_polar_stress_matches_fd_large_periodic_boxes(
    polar_calc_fd: MACECalculator,
    cell: np.ndarray,
    atol: float,
) -> None:
    atoms = _periodic_water(cell)
    atoms.calc = polar_calc_fd

    stress = atoms.get_stress(voigt=True)
    stress_fd = calculate_numerical_stress(atoms, eps=1e-5, force_consistent=False)

    np.testing.assert_allclose(stress, stress_fd, rtol=0.0, atol=atol)


# ---------------------------------------------------------------------------
# Regression values
# ---------------------------------------------------------------------------

_REG_REF_PATH = (
    Path(__file__).resolve().parent / "references" / "polar_regression_reference.json"
)
_LOCAL_BENCH_ROOT = (
    Path(__file__).resolve().parent / "references" / "x23_lattice_energy"
)

if _REG_REF_PATH.exists():
    _REF = json.loads(_REG_REF_PATH.read_text())
    STRUCTURE_KEYS = sorted(_REF.get("structures", {}).keys())
    BENCH_ROOT = Path(_REF.get("bench_root", ""))
    if not BENCH_ROOT.exists() and _LOCAL_BENCH_ROOT.exists():
        BENCH_ROOT = _LOCAL_BENCH_ROOT
else:
    _REF = {"structures": {}}
    STRUCTURE_KEYS = []
    BENCH_ROOT = Path("")

ATOL_BY_DTYPE = {
    "float32": 5e-6,
    "float64": 1e-9,
}


@pytest.fixture(scope="module", params=["float32", "float64"])
def polar_calc_regression(request):
    dtype = request.param
    try:
        calc = mace_polar(
            model="polar-1-m",
            device="cpu",
            default_dtype=dtype,
        )
    except (FileNotFoundError, ValueError, RuntimeError) as exc:
        _skip_if_model_unavailable(exc, "polar-1-m")
    return dtype, calc


@pytest.mark.skipif(
    not _REG_REF_PATH.exists(),
    reason="Regression reference JSON not available",
)
@pytest.mark.skipif(
    not BENCH_ROOT.exists(),
    reason="benchmarks-mp X23 structures not available",
)
@pytest.mark.parametrize("structure_relpath", STRUCTURE_KEYS)
def test_polar_2l_regression_hardcoded_values(polar_calc_regression, structure_relpath):
    dtype, calc = polar_calc_regression
    expected = _REF["structures"][structure_relpath][dtype]

    at = read(BENCH_ROOT / structure_relpath, index=0)
    at.info["charge"] = 0
    at.info["spin"] = 1
    at.calc = calc

    energy = float(at.get_potential_energy())
    forces = at.get_forces()
    stress = at.get_stress()

    atol = ATOL_BY_DTYPE[dtype]
    np.testing.assert_allclose(energy, expected["energy"], rtol=0.0, atol=atol)
    np.testing.assert_allclose(
        forces, np.array(expected["forces"]), rtol=0.0, atol=atol
    )
    np.testing.assert_allclose(
        stress, np.array(expected["stress"]), rtol=0.0, atol=atol
    )
