import argparse
import importlib.util
import os
from pathlib import Path

import ase.io
import numpy as np
import pytest
import torch
from ase.atoms import Atoms
from e3nn import o3

from mace.calculators import MACECalculator, MagneticMACECalculator
from mace.cli.eval_configs import run as mace_eval_configs_run
from mace.cli.run_train import run as mace_run
from mace.tools.arg_parser import build_default_arg_parser
from mace.tools.torch_tools import default_dtype

from mace.modules.extensions import MagneticScaleShiftMACE, MagneticSCFMACE
from mace.modules import interaction_classes

# ----------------------------------------------------------
# Environment flags
# ----------------------------------------------------------
CUDA_AVAILABLE = torch.cuda.is_available()

# ----------------------------------------------------------
# Fixtures
# ----------------------------------------------------------
@pytest.fixture(name="magnetic_configs")
def fixture_magnetic_configs():
    """Generate a small synthetic magnetic dataset."""

    # A simple 2-atom Fe dimer with random perturbations
    base = Atoms(numbers=[26, 26],
                 positions=[[0, 0, 0], [0, 0, 2.0]],
                 cell=[6.0] * 3,
                 pbc=[True] * 3)
    fit_configs = [
        Atoms(numbers=[26], positions=[[0, 0, 0]], cell=[6] * 3),
    ]

    fit_configs[0].info["REF_energy"] = 0.0
    fit_configs[0].info["config_type"] = "IsolatedAtom"
    fit_configs[0].arrays["REF_magmom"] = np.array([[0.0, 0.0, 2.2]])

    np.random.seed(5)
    for _ in range(20):
        c = base.copy()
        c.positions += np.random.normal(0, 0.05, size=c.positions.shape)
        c.info["REF_energy"] = np.random.normal(0.0, 0.01)
        c.new_array("REF_forces", np.random.normal(0, 0.01, size=c.positions.shape))
        c.new_array("REF_magforces", np.random.normal(0, 0.01, size=c.positions.shape))
        c.new_array("REF_magmom", np.tile([[0.0, 0.0, 2.2]], (len(c), 1)))
        fit_configs.append(c)

    return fit_configs


_magnetic_mace_params = {
    "name": "MACE",
    "valid_fraction": 0.05,
    "energy_weight": 1.0,
    "forces_weight": 10.0,
    "stress_weight": 1.0,
    "magforces_weight": 5.0,
    "model": "MagneticScaleShiftMACE",
    "interaction_first": "MagneticRealAgnosticSpinOrbitCoupledDensityInteractionBlock",
    "interaction": "MagneticRealAgnosticSpinOrbitCoupledDensityInteractionBlock",
    "hidden_irreps": "128x0e",
    "r_max": 3.5,
    "m_max": 10.0,
    "batch_size": 5,
    "max_num_epochs": 10,
    "swa": None,
    "start_swa": 5,
    "ema": None,
    "ema_decay": 0.99,
    "amsgrad": None,
    "restart_latest": None,
    "device": "cpu",
    "seed": 5,
    "loss": "stress",
    "energy_key": "REF_energy",
    "forces_key": "REF_forces",
    "magforces_key": "REF_magforces",
    "magmom_key": "REF_magmom",
    "eval_interval": 1,
    "use_reduced_cg": False,
}


# ----------------------------------------------------------
# Training tests
# ----------------------------------------------------------
def test_run_train_magnetic_mace(tmp_path, magnetic_configs):
    """Train a tiny magnetic MACE model on synthetic data and check energies."""
    ase.io.write(tmp_path / "fit.xyz", magnetic_configs)

    mace_params = _magnetic_mace_params.copy()
    mace_params["checkpoints_dir"] = str(tmp_path)
    mace_params["model_dir"] = str(tmp_path)
    mace_params["train_file"] = tmp_path / "fit.xyz"
    
    args = build_default_arg_parser().parse_args(
        [f"--{k}={v}" if v is not None else f"--{k}" for k, v in mace_params.items()]
    )

    # Run CLI training (mock Magnetic MACE mode)
    mace_run(args)

    model_path = tmp_path / "MACE.model"
    assert model_path.exists()

    calc = MagneticMACECalculator(
        model_paths=model_path, device="cpu", magmom_key="REF_magmom"
    )

    Es = []
    for at in magnetic_configs:
        at.calc = calc
        Es.append(at.get_potential_energy())

    assert all(np.isfinite(Es)), "Non-finite energies in magnetic MACE output."


def test_run_eval_magnetic_mace(tmp_path, magnetic_configs):
    """Run magnetic model evaluation and verify magnetic fields are written."""
    # Save fake model to disk
    with default_dtype(torch.float32):
        model = MagneticScaleShiftMACE(
            r_max=3.5,
            num_bessel=4,
            num_polynomial_cutoff=4,
            max_ell=2,
            interaction_cls=interaction_classes["MagneticRealAgnosticSpinOrbitCoupledDensityInteractionBlock"],
            interaction_cls_first=interaction_classes["MagneticRealAgnosticSpinOrbitCoupledDensityInteractionBlock"],
            num_interactions=1,
            num_elements=1,
            hidden_irreps=o3.Irreps("8x0e"),
            MLP_irreps=o3.Irreps("4x0e"),
            atomic_energies=np.zeros(1),
            avg_num_neighbors=1.0,
            atomic_numbers=[26],
            correlation=[1],
            gate=torch.nn.functional.silu,
            atomic_inter_shift=0.0,
            atomic_inter_scale=1.0,
            # == magmoms ===
            m_max=[3.0],
            num_mag_radial_basis = 8,
            num_mag_radial_basis_one_body = 10,
            max_m_ell = 1,
            use_magmom_one_body=False,
        )
        model_path = tmp_path / "magmace.model"
        torch.save(model, model_path)

    ase.io.write(tmp_path / "fit.xyz", magnetic_configs)
    output_path = tmp_path / "output.xyz"

    args = argparse.Namespace(
        model=str(model_path),
        configs=str(tmp_path / "fit.xyz"),
        output=str(output_path),
        device="cpu",
        default_dtype="float32",
        batch_size=1,
        compute_stress=False,
        compute_bec=False,
        enable_cueq=False,
        return_contributions=False,
        return_descriptors=False,
        return_node_energies=False,
        return_magforces=True,
        info_prefix="MACE_",
        head=None,
        magmom_key="REF_magmom",
    )

    mace_eval_configs_run(args)

    assert output_path.exists(), "Output file missing after evaluation."
    output_atoms = ase.io.read(str(output_path), index=":")
    assert len(output_atoms) == len(magnetic_configs)
    
    for at in output_atoms:
        assert "MACE_energy" in at.info
        assert "MACE_forces" in at.arrays
        assert "MACE_magforces" in at.arrays or "MACE_magmoms" in at.arrays


# ----------------------------------------------------------
# SCF wrapper test
# ----------------------------------------------------------
def test_run_magnetic_scf(tmp_path, magnetic_configs):
    """Check that the MagneticSCFMACE wrapper runs SCF relaxation cycles."""
    # Create minimal model and SCF wrapper
    with default_dtype(torch.float32):
        model = MagneticScaleShiftMACE(
            r_max=3.5,
            num_bessel=4,
            num_polynomial_cutoff=4,
            max_ell=2,
            m_max=[3.0],
            num_mag_radial_basis=4,
            max_m_ell=1,
            interaction_cls=interaction_classes["MagneticRealAgnosticSpinOrbitCoupledDensityInteractionBlock"],
            interaction_cls_first=interaction_classes["MagneticRealAgnosticSpinOrbitCoupledDensityInteractionBlock"],
            num_interactions=1,
            num_elements=1,
            hidden_irreps=o3.Irreps("8x0e"),
            MLP_irreps=o3.Irreps("4x0e"),
            atomic_energies=np.zeros(1),
            avg_num_neighbors=1.0,
            atomic_numbers=[26],
            correlation=[1],
            gate=torch.nn.functional.silu,
            use_magmom_one_body=False,
            num_mag_radial_basis_one_body=4,
            atomic_inter_shift=0.0,
            atomic_inter_scale=1.0,
        )
    scf_model = MagneticSCFMACE(model=model, n_scf_step=2, scf_logging=True)

    # Convert to data dict. magnetic_configs[0] is the IsolatedAtom (single Fe); use the dimer.
    at = magnetic_configs[1]
    data = {
        "positions": torch.tensor(at.positions, dtype=torch.float32),
        "cell": torch.tensor(at.cell.array, dtype=torch.float32).unsqueeze(0),
        "batch": torch.zeros(len(at), dtype=torch.int64),
        "ptr": torch.tensor([0, len(at)], dtype=torch.int64),
        "node_attrs": torch.nn.functional.one_hot(torch.tensor([0, 0]), num_classes=1).float(),
        "magmom": torch.tensor(at.arrays["REF_magmom"], dtype=torch.float32),
        "edge_index": torch.tensor([[0], [1]], dtype=torch.int64),
        "unit_shifts": torch.zeros((1, 3), dtype=torch.float32),
        "shifts": torch.zeros((1, 3), dtype=torch.float32),
    }

    out = scf_model(data)
    assert "equilibrated_magmom" in out
    assert torch.isfinite(out["equilibrated_magmom"]).all()


# ----------------------------------------------------------
# extract_config_mace_model round-trip
# ----------------------------------------------------------
def test_extract_config_magnetic_round_trip():
    """extract_config_mace_model should round-trip MagneticScaleShiftMACE."""
    from mace.tools.scripts_utils import extract_config_mace_model

    with default_dtype(torch.float32):
        model = MagneticScaleShiftMACE(
            r_max=3.5,
            num_bessel=4,
            num_polynomial_cutoff=4,
            max_ell=2,
            interaction_cls=interaction_classes[
                "MagneticRealAgnosticSpinOrbitCoupledDensityInteractionBlock"
            ],
            interaction_cls_first=interaction_classes[
                "MagneticRealAgnosticSpinOrbitCoupledDensityInteractionBlock"
            ],
            num_interactions=1,
            num_elements=1,
            hidden_irreps=o3.Irreps("8x0e"),
            MLP_irreps=o3.Irreps("4x0e"),
            atomic_energies=np.zeros(1),
            avg_num_neighbors=1.0,
            atomic_numbers=[26],
            correlation=[1],
            gate=torch.nn.functional.silu,
            atomic_inter_shift=0.0,
            atomic_inter_scale=1.0,
            m_max=[3.0],
            num_mag_radial_basis=8,
            num_mag_radial_basis_one_body=10,
            max_m_ell=1,
            use_magmom_one_body=True,
        )

    cfg = extract_config_mace_model(model)
    assert "error" not in cfg, cfg
    assert cfg["m_max"] == [3.0]
    assert cfg["max_m_ell"] == 1
    assert cfg["num_mag_radial_basis"] == 8
    assert cfg["num_mag_radial_basis_one_body"] == 10
    assert cfg["use_magmom_one_body"] is True


def test_inherit_magnetic_hyperparameters_from_foundation(monkeypatch):
    """inherit_magnetic_hyperparameters_from_foundation should copy m_max etc onto args."""
    from types import SimpleNamespace

    from mace.tools.multihead_tools import (
        inherit_magnetic_hyperparameters_from_foundation,
    )

    args = SimpleNamespace(
        m_max=[1, 1, 1],
        max_m_ell=1,
        num_mag_radial_basis=1,
        num_mag_radial_basis_one_body=1,
    )
    foundation_config = {
        "m_max": torch.tensor([2, 3, 4], dtype=torch.int64),
        "max_m_ell": 5,
        "num_mag_radial_basis": 6,
        "num_mag_radial_basis_one_body": 7,
    }

    monkeypatch.setattr(
        "mace.tools.multihead_tools.extract_config_mace_model",
        lambda model: foundation_config,
    )

    inherited = inherit_magnetic_hyperparameters_from_foundation(args, object())

    assert args.m_max == [2, 3, 4]
    assert args.max_m_ell == 5
    assert args.num_mag_radial_basis == 6
    assert args.num_mag_radial_basis_one_body == 7
    assert inherited == {
        "m_max_len": 3,
        "max_m_ell": 5,
        "num_mag_radial_basis": 6,
        "num_mag_radial_basis_one_body": 7,
    }


# ----------------------------------------------------------
# resolve_m_max
# ----------------------------------------------------------
def test_resolve_m_max_dict_form():
    from mace.tools.scripts_utils import resolve_m_max

    out = resolve_m_max(["{26: 1.8, 8: 0.5}"], [1, 6, 8, 26], default=1.0)
    assert out == [1.0, 1.0, 0.5, 1.8]


def test_resolve_m_max_fast_path_float_list():
    from mace.tools.scripts_utils import resolve_m_max

    out = resolve_m_max([0.1, 0.2, 0.3, 0.4], [1, 6, 8, 26])
    assert out == [0.1, 0.2, 0.3, 0.4]


def test_resolve_m_max_legacy_string_tokens():
    """argparse(nargs='+', type=str) on the legacy form yields stringified floats."""
    from mace.tools.scripts_utils import resolve_m_max

    out = resolve_m_max(["0.1", "0.2", "0.3", "0.4"], [1, 6, 8, 26])
    assert out == [0.1, 0.2, 0.3, 0.4]


def test_resolve_m_max_single_float_broadcast():
    from mace.tools.scripts_utils import resolve_m_max

    out = resolve_m_max(["1.5"], [1, 6, 8, 26])
    assert out == [1.5, 1.5, 1.5, 1.5]


def test_resolve_m_max_none_passthrough():
    from mace.tools.scripts_utils import resolve_m_max

    assert resolve_m_max(None, [1, 6, 8, 26]) is None


def test_resolve_m_max_wrong_length_raises():
    from mace.tools.scripts_utils import resolve_m_max

    with pytest.raises(ValueError, match="expected 4"):
        resolve_m_max([0.1, 0.2], [1, 6, 8, 26])


def test_resolve_m_max_extra_dict_keys_ignored():
    """A dict over-spec'd with elements not in the current z_table is OK: extras get ignored, present elements resolved."""
    from mace.tools.scripts_utils import resolve_m_max

    # Z=99 isn't in z_table; should be ignored, not raise.
    out = resolve_m_max(["{26: 1.8, 99: 1.0}"], [1, 6, 8, 26], default=1.0)
    assert out == [1.0, 1.0, 1.0, 1.8]


def test_resolve_m_max_numpy_atomic_numbers():
    """atomic_numbers can be np.int64 (as it comes from z_table)."""
    from mace.tools.scripts_utils import resolve_m_max

    zs = [np.int64(1), np.int64(6), np.int64(26)]
    out = resolve_m_max(["{26: 8.0}"], zs, default=4.0)
    assert out == [4.0, 4.0, 8.0]


# ----------------------------------------------------------
# O(3)-equivariance tests
# ----------------------------------------------------------
def _random_rotation(seed, dtype=torch.float32):
    g = torch.Generator().manual_seed(seed)
    A = torch.randn(3, 3, generator=g, dtype=dtype)
    Q, _ = torch.linalg.qr(A)
    if torch.det(Q) < 0:
        Q[:, 0] = -Q[:, 0]
    return Q


def _make_magnetic_cluster_data(dtype=torch.float32):
    """Non-collinear 2-Fe cluster with a hand-built neighbor list (no PBC edges)."""
    positions = torch.tensor(
        [[0.0, 0.0, 0.0], [1.6, 0.4, -0.3]], dtype=dtype
    )
    magmom = torch.tensor(
        [[0.5, 1.7, 0.9], [-1.1, 0.8, -0.6]], dtype=dtype
    )
    n = positions.shape[0]
    edge_index = torch.tensor([[0, 1], [1, 0]], dtype=torch.long)
    shifts = torch.zeros((edge_index.shape[1], 3), dtype=dtype)
    node_attrs = torch.nn.functional.one_hot(
        torch.zeros(n, dtype=torch.long), num_classes=1
    ).to(dtype)
    return {
        "positions": positions,
        "magmom": magmom,
        "edge_index": edge_index,
        "shifts": shifts,
        "unit_shifts": shifts.clone(),
        "cell": torch.eye(3, dtype=dtype).unsqueeze(0) * 10.0,
        "node_attrs": node_attrs,
        "batch": torch.zeros(n, dtype=torch.long),
        "ptr": torch.tensor([0, n], dtype=torch.long),
    }


def _build_small_magnetic_model(seed=42):
    torch.manual_seed(seed)
    with default_dtype(torch.float32):
        return MagneticScaleShiftMACE(
            r_max=3.5,
            num_bessel=4,
            num_polynomial_cutoff=4,
            max_ell=2,
            interaction_cls=interaction_classes[
                "MagneticRealAgnosticSpinOrbitCoupledDensityInteractionBlock"
            ],
            interaction_cls_first=interaction_classes[
                "MagneticRealAgnosticSpinOrbitCoupledDensityInteractionBlock"
            ],
            num_interactions=1,
            num_elements=1,
            hidden_irreps=o3.Irreps("8x0e"),
            MLP_irreps=o3.Irreps("4x0e"),
            atomic_energies=np.zeros(1),
            avg_num_neighbors=1.0,
            atomic_numbers=[26],
            correlation=[1],
            gate=torch.nn.functional.silu,
            atomic_inter_shift=0.0,
            atomic_inter_scale=1.0,
            m_max=[3.0],
            num_mag_radial_basis=8,
            num_mag_radial_basis_one_body=10,
            max_m_ell=1,
            use_magmom_one_body=False,
        )


def test_magnetic_mace_rotation_equivariance():
    """Rotating positions and magmoms together by R leaves E invariant and rotates F, magforces by R."""
    model = _build_small_magnetic_model().eval()
    R = _random_rotation(seed=1)

    data = _make_magnetic_cluster_data()
    data_rot = _make_magnetic_cluster_data()
    data_rot["positions"] = (data_rot["positions"] @ R.T).detach()
    data_rot["magmom"] = (data_rot["magmom"] @ R.T).detach()

    out = model(data, training=False, compute_force=True, compute_magforces=True)
    out_rot = model(
        data_rot, training=False, compute_force=True, compute_magforces=True
    )

    E = out["energy"].detach()
    F = out["forces"].detach()
    MF = out["magforces"].detach()
    E_r = out_rot["energy"].detach()
    F_r = out_rot["forces"].detach()
    MF_r = out_rot["magforces"].detach()

    assert torch.allclose(E, E_r, atol=1e-4, rtol=1e-4)
    assert torch.allclose(F_r, F @ R.T, atol=1e-4, rtol=1e-4)
    assert torch.allclose(MF_r, MF @ R.T, atol=1e-4, rtol=1e-4)


def test_magnetic_mace_inversion_parity():
    """Flipping both positions and magmoms leaves E invariant; forces and magforces flip with them."""
    model = _build_small_magnetic_model().eval()

    data = _make_magnetic_cluster_data()
    data_inv = _make_magnetic_cluster_data()
    data_inv["positions"] = (-data_inv["positions"]).detach()
    data_inv["magmom"] = (-data_inv["magmom"]).detach()

    out = model(data, training=False, compute_force=True, compute_magforces=True)
    out_inv = model(
        data_inv, training=False, compute_force=True, compute_magforces=True
    )

    E = out["energy"].detach()
    F = out["forces"].detach()
    MF = out["magforces"].detach()
    E_i = out_inv["energy"].detach()
    F_i = out_inv["forces"].detach()
    MF_i = out_inv["magforces"].detach()

    assert torch.allclose(E, E_i, atol=1e-4, rtol=1e-4)
    assert torch.allclose(F_i, -F, atol=1e-4, rtol=1e-4)
    assert torch.allclose(MF_i, -MF, atol=1e-4, rtol=1e-4)
