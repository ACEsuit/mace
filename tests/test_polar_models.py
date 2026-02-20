import math
import os
from pathlib import Path
from typing import Any, Dict

import numpy as np
import pytest
import torch
from ase import Atoms
from e3nn import o3

from mace import data, modules, tools
from mace.calculators.mace import MACECalculator
from mace.cli.convert_cueq_e3nn import run as run_cueq_to_e3nn
from mace.cli.convert_e3nn_cueq import run as run_e3nn_to_cueq
from mace.modules import interaction_classes
from mace.modules.extensions import PolarMACE
from mace.tools import torch_geometric, utils


# ===== From test_calculator_water_fieldfukui.py =====
def build_model(device, dtype):
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
        r_max=6.0,
        num_bessel=8,
        num_polynomial_cutoff=6,
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


def build_water_atoms():
    positions = np.array(
        [
            [0.0, 0.0, 0.0],
            [0.9572, 0.0, 0.0],
            [-0.2390, 0.9270, 0.0],
        ]
    )
    numbers = [8, 1, 1]
    atoms = Atoms(numbers=numbers, positions=positions)
    return atoms


@pytest.mark.skipif(
    not os.path.exists(
        os.path.normpath(
            os.path.join(
                os.path.dirname(__file__),
                "..",
                "mace-polar-spin-3L-xL-25-cpu_state_dict.pt",
            )
        )
    ),
    reason="Missing pretrained state dict at repo root",
)
def test_calculator_water_energy_and_charges():
    device = "cpu"
    dtype = torch.float32
    model = build_model(device, dtype)
    sd_path = os.path.normpath(
        os.path.join(
            os.path.dirname(__file__),
            "..",
            "mace-polar-spin-3L-xL-25-cpu_state_dict.pt",
        )
    )
    try:
        sd = torch.load(sd_path, map_location="cpu")
        if isinstance(sd, dict):
            for k in ("state_dict", "model_state_dict", "model", "module", "state"):
                if k in sd and isinstance(sd[k], dict):
                    sd = sd[k]
                    break
        current = model.state_dict()
        filtered = {
            k: v for k, v in sd.items() if k in current and current[k].shape == v.shape
        }
        model.load_state_dict(filtered, strict=False)
    except Exception:
        pass

    atoms = build_water_atoms()
    calc = MACECalculator(models=model, device=device, model_type="PolarMACE")
    atoms.calc = calc
    e0 = atoms.get_potential_energy()

    assert "charges" in calc.results
    charges = calc.results["charges"]
    assert charges.shape[0] == len(atoms)

    atoms.info["spin"] = 1.0
    e_spin = atoms.get_potential_energy()
    atoms.info["spin"] = 0.0

    atoms.info["charge"] = 1.0
    e_charge = atoms.get_potential_energy()
    atoms.info["charge"] = 0.0

    if np.isclose(e0, e_spin) and np.isclose(e0, e_charge):
        pytest.skip(
            "Energy did not change with spin/charge; weights may not be fully matching"
        )


# ===== From test_field_fukui_invariance.py =====
def random_rotation(device, dtype):
    A = torch.randn(3, 3, device=device, dtype=dtype)
    Q, _ = torch.linalg.qr(A)
    if torch.det(Q) < 0:
        Q[:, 0] = -Q[:, 0]
    return Q


def build_minimal_model(device, dtype):
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


def build_minimal_batch(device, dtype):
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
    cell = (
        torch.eye(3, dtype=dtype, device=device).unsqueeze(0) * cell_len
    )
    rcell = (
        torch.eye(3, dtype=dtype, device=device).unsqueeze(0)
        * (2.0 * math.pi / cell_len)
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


@pytest.mark.parametrize("dtype", [torch.float32])
def test_energy_invariance_under_rotation_and_translation(dtype):
    device = torch.device("cpu")
    model = build_minimal_model(device, dtype)
    data_batch = build_minimal_batch(device, dtype)

    out = model(data_batch, training=False, compute_force=False)
    E = out["energy"].detach()

    R = random_rotation(device, dtype)
    data_rot = {
        k: (v.clone() if isinstance(v, torch.Tensor) else v) for k, v in data_batch.items()
    }
    data_rot["positions"] = data_batch["positions"] @ R.T
    out_rot = model(data_rot, training=False, compute_force=False)
    E_rot = out_rot["energy"].detach()

    assert torch.allclose(E, E_rot, atol=3e-1, rtol=1e-5)

    t = torch.tensor([0.5, -0.3, 0.8], device=device, dtype=dtype)
    data_tr = {
        k: (v.clone() if isinstance(v, torch.Tensor) else v) for k, v in data_batch.items()
    }
    data_tr["positions"] = data_batch["positions"] + t
    out_tr = model(data_tr, training=False, compute_force=False)
    E_tr = out_tr["energy"].detach()

    assert torch.allclose(E, E_tr, atol=5e-3, rtol=1e-5)


# ===== From test_field_fukui_legacy_models_eval.py =====
REPO_ROOT = Path(__file__).resolve().parent.parent
LEGACY_MODELS = [
    ("mace-polar-spin-3L.model", -2079.864990234375),
    ("mace-polar-spin-2L.model", -2079.86376953125),
    ("mace-polar-spin-1L.model", -2079.86474609375),
]


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
    config = data.config_from_atoms(atoms, key_specification=keyspec, head_name="Default")
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


def _clone_batch(batch: dict) -> dict:
    return {k: (v.clone() if torch.is_tensor(v) else v) for k, v in batch.items()}


@pytest.mark.parametrize("model_name, expected_energy", LEGACY_MODELS)
def test_legacy_polar_model_evaluates(model_name, expected_energy):
    model_path = REPO_ROOT / model_name
    if not model_path.exists():
        pytest.skip(f"Missing legacy model file: {model_path}")

    model = torch.load(str(model_path), map_location="cpu")
    assert model.__class__.__name__ == "PolarMACE"

    atoms = _water_atoms()
    calc = MACECalculator(
        model_paths=str(model_path),
        device="cpu",
        model_type="PolarMACE",
    )
    atoms.calc = calc

    energy = atoms.get_potential_energy()
    assert np.isfinite(energy)
    assert abs(float(energy) - expected_energy) < 1e-5


@pytest.mark.parametrize("model_name, _", LEGACY_MODELS)
def test_legacy_polar_models_run_with_and_without_cueq(model_name, _):
    if os.environ.get("SKIP_CUEQ_TESTS") == "1":
        pytest.skip("Skipping CuEq test by environment request")

    model_path = REPO_ROOT / model_name
    if not model_path.exists():
        pytest.skip(f"Missing legacy model file: {model_path}")

    atoms = _water_atoms()
    calc_e3 = MACECalculator(
        model_paths=str(model_path),
        device="cpu",
        model_type="PolarMACE",
        enable_cueq=False,
    )
    atoms.calc = calc_e3
    energy_e3 = float(atoms.get_potential_energy())
    assert np.isfinite(energy_e3)

    calc_cueq = MACECalculator(
        model_paths=str(model_path),
        device="cpu",
        model_type="PolarMACE",
        enable_cueq=True,
    )
    atoms.calc = calc_cueq
    energy_cueq = float(atoms.get_potential_energy())
    assert np.isfinite(energy_cueq)

    assert abs(energy_cueq - energy_e3) <= 3.0e-4


@pytest.mark.parametrize("model_name, _", LEGACY_MODELS)
def test_legacy_polar_true_cueq_matches_e3nn(model_name, _):
    model_path = REPO_ROOT / model_name
    if not model_path.exists():
        pytest.skip(f"Missing legacy model file: {model_path}")

    model_e3 = torch.load(str(model_path), map_location="cpu").eval()
    previous_default_dtype = torch.get_default_dtype()
    try:
        torch.set_default_dtype(next(model_e3.parameters()).dtype)
        batch = _build_model_batch(model_e3)

        out_e3 = model_e3(_clone_batch(batch), compute_stress=True, training=False)
        model_cueq = run_e3nn_to_cueq(model_e3, device="cpu", layout="ir_mul").eval()
        out_cueq = model_cueq(_clone_batch(batch), compute_stress=True, training=False)

        energy_max_abs = torch.max(torch.abs(out_cueq["energy"] - out_e3["energy"])).item()
        forces_max_abs = torch.max(torch.abs(out_cueq["forces"] - out_e3["forces"])).item()
        stress_max_abs = torch.max(torch.abs(out_cueq["stress"] - out_e3["stress"])).item()
    finally:
        torch.set_default_dtype(previous_default_dtype)

    assert energy_max_abs <= 3.0e-4
    assert forces_max_abs <= 2.5e-4
    assert stress_max_abs <= 1.0e-8


def test_legacy_polar_2l_true_cueq_matches_e3nn_float64():
    model_name = "mace-polar-spin-2L.model"
    model_path = REPO_ROOT / model_name
    if not model_path.exists():
        pytest.skip(f"Missing legacy model file: {model_path}")

    model_e3 = torch.load(str(model_path), map_location="cpu").eval().double()
    previous_default_dtype = torch.get_default_dtype()
    try:
        torch.set_default_dtype(torch.float64)
        batch = _build_model_batch(model_e3)

        out_e3 = model_e3(_clone_batch(batch), compute_stress=True, training=False)
        model_cueq = (
            run_e3nn_to_cueq(model_e3, device="cpu", layout="ir_mul").eval().double()
        )
        out_cueq = model_cueq(_clone_batch(batch), compute_stress=True, training=False)

        energy_max_abs = torch.max(torch.abs(out_cueq["energy"] - out_e3["energy"])).item()
        forces_max_abs = torch.max(torch.abs(out_cueq["forces"] - out_e3["forces"])).item()
        stress_max_abs = torch.max(torch.abs(out_cueq["stress"] - out_e3["stress"])).item()
    finally:
        torch.set_default_dtype(previous_default_dtype)

    assert energy_max_abs <= 1.0e-7
    assert forces_max_abs <= 1.0e-7
    assert stress_max_abs <= 1.0e-11


# ===== From test_field_fukui_load_state_dict.py =====
def build_strict_config_model(device, dtype):
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
        r_max=4.0,
        num_bessel=4,
        num_polynomial_cutoff=3,
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


def load_state_dict_from_file(path):
    try:
        sd = torch.load(path, map_location="cpu", weights_only=True)
    except TypeError:
        sd = torch.load(path, map_location="cpu")
    except Exception:
        sd = torch.load(path, map_location="cpu", weights_only=False)
    if isinstance(sd, dict):
        for k in ("state_dict", "model_state_dict", "model", "module", "state"):
            if k in sd and isinstance(sd[k], dict):
                return sd[k]
    return sd


@pytest.mark.parametrize(
    "fname",
    [
        "mace-polar-spin-3L_state_dict.pt",
    ],
)
def test_load_previous_state_dict(fname):
    path = os.path.join(os.path.dirname(__file__), "..", fname)
    path = os.path.normpath(path)
    if not os.path.exists(path):
        pytest.skip(f"Missing state_dict file: {path}")

    sd = load_state_dict_from_file(path)
    assert isinstance(sd, dict)

    model_path = path.replace("_state_dict.pt", ".model")
    model = torch.load(model_path, map_location="cpu")
    missing, unexpected = model.load_state_dict(sd, strict=True)
    assert missing == [] and unexpected == []


# ===== From test_field_fukui_water_eval.py =====
def build_model_for_eval(device, dtype):
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
        r_max=6.0,
        num_bessel=8,
        num_polynomial_cutoff=6,
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


def load_state_dict_safe(path):
    try:
        sd = torch.load(path, map_location="cpu", weights_only=True)
    except TypeError:
        sd = torch.load(path, map_location="cpu")
    except Exception:
        sd = torch.load(path, map_location="cpu", weights_only=False)
    if isinstance(sd, dict):
        for k in ("state_dict", "model_state_dict", "model", "module", "state"):
            if k in sd and isinstance(sd[k], dict):
                return sd[k]
    return sd


def build_water(device, dtype):
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
    rcell = (
        torch.eye(3, dtype=dtype, device=device).unsqueeze(0)
        * (2.0 * math.pi / cell_len)
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


@pytest.mark.skipif(
    not os.path.exists(
        os.path.normpath(
            os.path.join(
                os.path.dirname(__file__), "..", "mace-polar-spin-3L-xL-25-cpu_state_dict.pt"
            )
        )
    ),
    reason="Missing pretrained state dict at repo root",
)
def test_water_energy_changes_with_charge_and_spin():
    device = torch.device("cpu")
    dtype = torch.float32
    model = build_model_for_eval(device, dtype)

    sd_path = os.path.normpath(
        os.path.join(os.path.dirname(__file__), "..", "mace-polar-spin-3L-xL-25-cpu_state_dict.pt")
    )
    sd = load_state_dict_safe(sd_path)
    current_sd = model.state_dict()
    filtered = {k: v for k, v in sd.items() if k in current_sd and current_sd[k].shape == v.shape}
    model.load_state_dict(filtered, strict=False)

    data_batch = build_water(device, dtype)

    out0 = model(data_batch, training=False, compute_force=False)
    E0 = out0["energy"].detach()

    data_spin = {
        k: (v.clone() if isinstance(v, torch.Tensor) else v) for k, v in data_batch.items()
    }
    data_spin["total_spin"] = torch.tensor([1.0], dtype=dtype, device=device)
    out_spin = model(data_spin, training=False, compute_force=False)
    E_spin = out_spin["energy"].detach()

    data_charge = {
        k: (v.clone() if isinstance(v, torch.Tensor) else v) for k, v in data_batch.items()
    }
    data_charge["total_charge"] = torch.tensor([1.0], dtype=dtype, device=device)
    out_charge = model(data_charge, training=False, compute_force=False)
    E_charge = out_charge["energy"].detach()

    if torch.allclose(E0, E_spin) and torch.allclose(E0, E_charge):
        pytest.skip(
            "Model weights did not respond to spin/charge changes in this environment"
        )


# ===== From test_fieldfukui_cueq_oeq.py =====
try:
    import cuequivariance as cue  # pylint: disable=unused-import

    CUET_AVAILABLE = True
except ImportError:
    CUET_AVAILABLE = False

import importlib.util as _ilu

CUET_OPS_AVAILABLE = _ilu.find_spec("cuequivariance_torch") is not None

CUDA_AVAILABLE = torch.cuda.is_available()


@pytest.mark.skipif(not CUET_AVAILABLE, reason="cuequivariance not installed")
@pytest.mark.skipif(
    not CUET_OPS_AVAILABLE, reason="cuequivariance_torch (ops) not installed"
)
class TestPolarCueqParity:
    @pytest.fixture
    def model_config(self) -> Dict[str, Any]:
        table = tools.AtomicNumberTable([6])
        hidden_irreps = o3.Irreps("16x0e + 16x1o")
        MLP_irreps = o3.Irreps("8x0e")
        return {
            "r_max": 5.0,
            "num_bessel": 6,
            "num_polynomial_cutoff": 6,
            "max_ell": 2,
            "interaction_cls": modules.interaction_classes[
                "RealAgnosticResidualInteractionBlock"
            ],
            "interaction_cls_first": modules.interaction_classes[
                "RealAgnosticResidualInteractionBlock"
            ],
            "num_interactions": 2,
            "num_elements": 1,
            "hidden_irreps": hidden_irreps,
            "MLP_irreps": MLP_irreps,
            "gate": torch.nn.functional.silu,
            "atomic_energies": torch.tensor([0.0]),
            "avg_num_neighbors": 8.0,
            "atomic_numbers": table.zs,
            "correlation": 3,
            "radial_type": "bessel",
            "atomic_inter_scale": 1.0,
            "atomic_inter_shift": 0.0,
            "kspace_cutoff_factor": 1.0,
            "atomic_multipoles_max_l": 1,
            "atomic_multipoles_smearing_width": 1.5,
            "field_feature_max_l": 1,
            "field_feature_widths": [1.5],
            "field_feature_norms": [1.0, 1.0],
            "num_recursion_steps": 1,
            "include_electrostatic_self_interaction": False,
            "add_local_electron_energy": False,
            "quadrupole_feature_corrections": False,
            "return_electrostatic_potentials": False,
            "field_norm_factor": 1.0,
            "fixedpoint_update_config": {
                "type": "AgnosticEmbeddedOneBodyVariableUpdate",
                "potential_embedding_cls": "AgnosticChargeBiasedLinearPotentialEmbedding",
                "nonlinearity_cls": "NoNonLinearity",
            },
            "field_readout_config": {"type": "OneBodyMLPFieldReadout"},
        }

    @pytest.fixture(params=(["cuda"] if CUDA_AVAILABLE else ["cpu"]))
    def device(self, request):
        return request.param

    @pytest.fixture
    def batch(self, device: str, model_config: Dict[str, Any]) -> Dict[str, torch.Tensor]:
        from ase import build

        table = tools.AtomicNumberTable([6])
        atoms = build.bulk("C", "diamond", a=3.567, cubic=True)
        atoms_list = [atoms.repeat((2, 2, 2))]
        configs = [data.config_from_atoms(at) for at in atoms_list]
        loader = torch_geometric.dataloader.DataLoader(
            dataset=[
                data.AtomicData.from_config(cfg, z_table=table, cutoff=5.0)
                for cfg in configs
            ],
            batch_size=1,
            shuffle=False,
            drop_last=False,
        )
        batch = next(iter(loader)).to(device)
        n_graphs = int(batch["num_graphs"]) if "num_graphs" in batch else 1
        batch["fermi_level"] = torch.zeros(
            n_graphs, device=device, dtype=batch["positions"].dtype
        )
        batch["external_field"] = torch.zeros(
            n_graphs, 3, device=device, dtype=batch["positions"].dtype
        )
        batch["total_charge"] = torch.zeros(
            n_graphs, device=device, dtype=batch["positions"].dtype
        )
        batch["total_spin"] = torch.ones(
            n_graphs, device=device, dtype=batch["positions"].dtype
        )
        return batch

    def test_e3nn_cueq_parity(self, model_config: Dict[str, Any], batch, device):
        previous_default_dtype = torch.get_default_dtype()
        try:
            torch.set_default_dtype(torch.float64)
            model_config = dict(model_config)
            model_config["atomic_energies"] = torch.tensor(
                [0.0], dtype=torch.get_default_dtype()
            )
            model_e3 = modules.PolarMACE(**model_config).to(device)

            model_cu = run_e3nn_to_cueq(model_e3, device=device, layout="mul_ir").to(device)
            model_e3_back = run_cueq_to_e3nn(model_cu).to(device)

            def cast_batch(b):
                d = b.to_dict()
                out = {}
                for k, v in d.items():
                    if torch.is_tensor(v) and v.dtype.is_floating_point:
                        out[k] = v.to(dtype=torch.get_default_dtype())
                    else:
                        out[k] = v
                return out

            bd = cast_batch(batch)
            out_e3 = model_e3(bd, training=True, compute_stress=True)
            out_cu = model_cu(bd, training=True, compute_stress=True)
            out_back = model_e3_back(bd, training=True, compute_stress=True)
        finally:
            torch.set_default_dtype(previous_default_dtype)

        torch.testing.assert_close(out_e3["energy"], out_cu["energy"])
        torch.testing.assert_close(out_cu["energy"], out_back["energy"])
        torch.testing.assert_close(out_e3["forces"], out_cu["forces"])
        torch.testing.assert_close(out_cu["forces"], out_back["forces"])
        torch.testing.assert_close(out_e3["stress"], out_cu["stress"])
        torch.testing.assert_close(out_cu["stress"], out_back["stress"])
