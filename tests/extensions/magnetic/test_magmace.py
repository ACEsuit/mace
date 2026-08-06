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
from mace.data import AtomicData, KeySpecification, config_from_atoms
from mace.tools import torch_geometric, utils
from mace.tools.arg_parser import build_default_arg_parser
from mace.tools.torch_tools import default_dtype

from mace.modules.extensions import MagneticScaleShiftMACE, MagneticSCFMACE
from mace.modules import interaction_classes
from mace.tools.scripts_utils import get_optimizer, get_params_options

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
    """Rotating positions and magmoms TOGETHER by R leaves E invariant and rotates F, magforces by R.

    Together is the operative word. This model has spin-orbit coupling, so the
    energy depends on how the spins are oriented relative to the lattice. Only
    the joint rotation is a symmetry; rotating the spins while holding the
    positions is expected to change the energy and is NOT tested here.

    Spin-only invariance is the non-SOC property. It is not built into this
    architecture, it is induced during training by --data_aug_magmom, which
    rotates the moments while leaving the positions alone. Test that by
    training with the augmentation, not by calling this model.
    """
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
    """Flipping BOTH positions and magmoms leaves E invariant; forces and magforces flip with them.

    Note what this does not claim. Inverting the positions while holding the
    spins is not a symmetry of an SOC model, and the energy does change under
    it: magnetic moments are axial vectors, so a parity operation that flips
    the lattice but not the spins alters their relative orientation, which is
    exactly what the SOC term reads. See the rotation test above for how the
    spin-only case is handled instead.
    """
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


# ----------------------------------------------------------
# Optimizer parameter-registration guard
# ----------------------------------------------------------
def test_magnetic_mace_registers_all_trainable_parameters():
    """Every trainable MagneticScaleShiftMACE parameter must be claimed by an
    optimizer group; get_params_options raises otherwise, so a successful
    optimizer build is the assertion."""
    model = _build_small_magnetic_model()
    args = argparse.Namespace(
        lr=0.01,
        weight_decay=5e-7,
        amsgrad=True,
        beta=0.9,
        freeze=None,
        optimizer="adam",
        lr_params_factors="{}",
        train_one_body_contribution=True,
    )
    get_optimizer(args, get_params_options(args, model))


# ----------------------------------------------------------
# Accelerated-backend wiring
# ----------------------------------------------------------
def _tiny_magnetic_model():
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


def test_magnetic_calculator_rejects_hybrid_backends():
    """There is no hybrid cueq+oeq path here, so asking for both must not proceed."""
    with pytest.raises(ValueError, match="hybrid"):
        MagneticMACECalculator(
            models=[_tiny_magnetic_model()],
            device="cpu",
            enable_cueq=True,
            enable_oeq=True,
        )


@pytest.mark.parametrize(
    "flag,available_attr,library",
    [
        ("enable_cueq", "CUEQQ_AVAILABLE", "cuequivariance"),
        ("enable_oeq", "OEQ_AVAILABLE", "openequivariance"),
    ],
)
def test_magnetic_calculator_reports_missing_backend(
    monkeypatch, flag, available_attr, library
):
    """A missing backend is an ImportError, not a TypeError on a None converter."""
    from mace.calculators import mace as mace_calc_mod

    monkeypatch.setattr(mace_calc_mod, available_attr, False)
    with pytest.raises(ImportError, match=library):
        MagneticMACECalculator(
            models=[_tiny_magnetic_model()],
            device="cpu",
            **{flag: True},
        )


def test_magnetic_calculator_converts_to_cueq_once(monkeypatch, tmp_path):
    """Loading from model_paths used to convert once on load and again later.

    The second call handed an already-converted model to a converter that
    expects the e3nn layout.
    """
    from mace.calculators import mace as mace_calc_mod

    with default_dtype(torch.float32):
        model_path = tmp_path / "magmace.model"
        torch.save(_tiny_magnetic_model(), model_path)

    calls = []

    def fake_convert(model, device=None):
        calls.append(model)
        return model

    monkeypatch.setattr(mace_calc_mod, "CUEQQ_AVAILABLE", True)
    monkeypatch.setattr(mace_calc_mod, "run_e3nn_to_cueq", fake_convert)

    MagneticMACECalculator(
        model_paths=[str(model_path)],
        device="cpu",
        default_dtype="float32",
        enable_cueq=True,
    )

    assert len(calls) == 1


def test_magnetic_committee_rmax_mismatch_reports_values():
    """A committee r_max mismatch must name the cutoffs, not raise TypeError.

    The message interpolated ' '.join over a NumPy float array, so the real
    configuration problem was hidden behind a TypeError.
    """
    model_a = _tiny_magnetic_model()
    model_b = _tiny_magnetic_model()
    model_b.r_max = torch.tensor(4.5)

    with pytest.raises(ValueError, match="committee r_max are not all the same"):
        MagneticMACECalculator(models=[model_a, model_b], device="cpu")


def test_magnetic_check_state_tracks_magmoms(magnetic_configs):
    """Changing magmoms in place must invalidate the cached results.

    magmom_key is REF_magmom, which is not one of ASE's all_changes, so the
    base check_state did not see it and served stale energies and forces.
    """
    calc = MagneticMACECalculator(
        models=[_tiny_magnetic_model()], device="cpu", default_dtype="float32"
    )
    atoms = magnetic_configs[1].copy()
    atoms.calc = calc
    atoms.get_potential_energy()

    # nothing touched yet, so nothing to recompute
    assert calc.check_state(atoms) == []

    atoms.arrays["REF_magmom"] = atoms.arrays["REF_magmom"] + 0.5
    assert "REF_magmom" in calc.check_state(atoms)


def test_eval_configs_unwraps_scf_wrapped_models(tmp_path, magnetic_configs):
    """mace_eval_configs must read model metadata off the inner module.

    MagneticSCFMACE wraps the real model in `magmom_mace` and keeps none of
    heads / atomic_numbers / r_max on itself, so reading them off the loaded
    object raised AttributeError for every SCF checkpoint. The ASE calculator
    already unwrapped with getattr(model, "magmom_mace", model); the eval CLI
    did not, so the two disagreed about what a loaded model looks like.
    """
    with default_dtype(torch.float32):
        scf_model = MagneticSCFMACE(
            model=_tiny_magnetic_model(), n_scf_step=2, scf_logging=False
        )
        model_path = tmp_path / "scf.model"
        torch.save(scf_model, model_path)

    # reachable after a save/load round trip, which is where a __getattr__ that
    # touched self.magmom_mace instead of __dict__ would recurse
    loaded = torch.load(model_path, map_location="cpu")
    assert float(loaded.r_max) == float(loaded.magmom_mace.r_max)

    ase.io.write(tmp_path / "fit.xyz", magnetic_configs[1:3])
    output_path = tmp_path / "out.xyz"
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
        return_magforces=False,
        magmom_key="REF_magmom",
        info_prefix="MACE_",
        head=None,
    )
    mace_eval_configs_run(args)

    assert output_path.exists()
    assert len(ase.io.read(str(output_path), index=":")) == 2


def test_eval_configs_refuses_magforces_for_scf_models(tmp_path, magnetic_configs):
    """SCF wrappers take no compute_magforces, so asking for it must say so.

    MagneticSCFMACE.forward accepts only data/training/compute_force/
    compute_virials/compute_stress/compute_displacement. Passing the flag
    through would surface as a bare TypeError from the forward call.
    """
    with default_dtype(torch.float32):
        model_path = tmp_path / "scf.model"
        torch.save(
            MagneticSCFMACE(
                model=_tiny_magnetic_model(), n_scf_step=2, scf_logging=False
            ),
            model_path,
        )

    ase.io.write(tmp_path / "fit.xyz", magnetic_configs[1:2])
    args = argparse.Namespace(
        model=str(model_path),
        configs=str(tmp_path / "fit.xyz"),
        output=str(tmp_path / "out.xyz"),
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
        magmom_key="REF_magmom",
        info_prefix="MACE_",
        head=None,
    )
    with pytest.raises(ValueError, match="return_magforces"):
        mace_eval_configs_run(args)


def test_random_rotation_loader_over_real_atomic_data(magnetic_configs):
    """Pin the augmentation's contract through the loader it is wired into.

    --data_aug_magmom wraps the train loader with create_random_rotation_loader,
    so every ASE sample passes through Random3DRotation as a real AtomicData.
    Two things are easy to get wrong about that path and both have been:

    1. Random3DRotation.forward assigns onto `data`, which reads like it
       corrupts the dataset, since for the ASE path the dataset is a plain list
       and __getitem__ returns the same object every epoch. It does not:
       TransformedDataset calls the transform, and BaseTransform.__call__ is
       `self.forward(copy.copy(data))`, so forward only ever sees a copy.
    2. "Fixing" 1. by copying inside forward crashes, because AtomicData cannot
       be cloned: Data.clone rebuilds via cls() and AtomicData.__init__ takes
       27 required arguments.

    So this asserts the stored samples come through untouched after repeated
    epochs, without requiring forward itself to copy. A stub-based unit test
    can see neither point, since it takes a different copy path.
    """
    pytest.importorskip(
        "torch_geometric", reason="loader path needs real torch_geometric"
    )
    from mace.data.augmentation import create_random_rotation_loader

    z_table = utils.AtomicNumberTable([26])
    keyspec = KeySpecification(
        info_keys={"energy": "REF_energy"},
        arrays_keys={"magmom": "REF_magmom", "magforces": "REF_magforces"},
    )
    dataset = [
        AtomicData.from_config(
            config_from_atoms(at, key_specification=keyspec),
            z_table=z_table,
            cutoff=3.5,
        )
        for at in magnetic_configs[1:5]
    ]
    stored = [d.magmom.clone() for d in dataset]

    base_loader = torch_geometric.dataloader.DataLoader(
        dataset=dataset, batch_size=2, shuffle=False, drop_last=False
    )
    loader = create_random_rotation_loader(base_loader)

    for _ in range(2):  # two epochs over the same underlying objects
        batches = list(loader)
        assert batches, "loader yielded nothing"
        for batch in batches:
            assert batch.magmom.shape[-1] == 3
            assert torch.isfinite(batch.magmom).all()

    for original, item in zip(stored, dataset):
        assert torch.equal(item.magmom, original), "dataset sample was mutated"


@pytest.mark.parametrize(
    "attr",
    [
        "heads",
        "atomic_numbers",
        "r_max",
        "num_interactions",
        "products",
        "interactions",
        "radial_embedding",
        "atomic_energies_fn",
    ],
)
def test_scf_wrapper_delegates_inner_model_attributes(attr):
    """The wrapper must answer for the model it wraps.

    MagneticSCFMACE defines none of these: they live on magmom_mace. Without
    delegation every consumer has to know that, and fixing them one report at
    a time has already missed the eval CLI twice, first for heads /
    atomic_numbers / r_max and then for the descriptors path's
    num_interactions / products. Others are still reachable this way:
    create_lammps_model and select_head read model.heads, and fine-tuning
    reads model_foundation.interactions and .atomic_energies_fn.
    """
    scf = MagneticSCFMACE(
        model=_tiny_magnetic_model(), n_scf_step=2, scf_logging=False
    )
    assert getattr(scf, attr) is getattr(scf.magmom_mace, attr)


def test_scf_wrapper_keeps_its_own_forward_and_still_raises():
    """Delegation must not swallow real mistakes or shadow the wrapper itself."""
    scf = MagneticSCFMACE(
        model=_tiny_magnetic_model(), n_scf_step=2, scf_logging=False
    )
    # forward is the wrapper's, not the inner model's
    assert scf.forward.__func__ is MagneticSCFMACE.forward
    # compute_magforces is still absent, which is why eval refuses it
    import inspect

    assert "compute_magforces" not in inspect.signature(scf.forward).parameters
    with pytest.raises(AttributeError):
        scf.definitely_not_a_real_attribute


def test_eval_configs_descriptors_for_scf_wrapped_models(tmp_path, magnetic_configs):
    """--return_descriptors needs num_interactions and products, both inner-only."""
    with default_dtype(torch.float32):
        model_path = tmp_path / "scf.model"
        torch.save(
            MagneticSCFMACE(
                model=_tiny_magnetic_model(), n_scf_step=2, scf_logging=False
            ),
            model_path,
        )

    ase.io.write(tmp_path / "fit.xyz", magnetic_configs[1:3])
    output_path = tmp_path / "out.xyz"
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
        return_descriptors=True,
        descriptor_num_layers=-1,
        descriptor_aggregation_method=None,
        descriptor_invariants_only=True,
        return_node_energies=False,
        return_magforces=False,
        magmom_key="REF_magmom",
        info_prefix="MACE_",
        head=None,
    )
    mace_eval_configs_run(args)

    out = ase.io.read(str(output_path), index=":")
    assert len(out) == 2
    assert any("descriptor" in k.lower() for at in out for k in at.arrays)


def test_hessian_refused_for_scf_wrapped_models():
    """An SCF wrapper cannot give a hessian, and the inner one is not a substitute.

    MagneticSCFMACE.forward takes no compute_hessian, so the request used to
    surface as a bare TypeError from the forward call. Falling through to
    magmom_mace would be worse than the error: that hessian holds the magnetic
    moments fixed and so omits the dm*/dr term, making it the hessian of a
    different energy than the one the calculator reports.
    """
    with default_dtype(torch.float32):
        scf = MagneticSCFMACE(
            model=_tiny_magnetic_model(), n_scf_step=2, scf_logging=False
        )

    atoms = Atoms(
        numbers=[26, 26],
        positions=[[0.0, 0.0, 0.0], [0.0, 0.0, 2.0]],
        cell=[6.0] * 3,
        pbc=True,
    )
    atoms.arrays["REF_magmom"] = np.tile([[0.0, 0.0, 2.2]], (2, 1))
    calc = MagneticMACECalculator(
        models=[scf], device="cpu", default_dtype="float32", magmom_key="REF_magmom"
    )

    with pytest.raises(NotImplementedError, match="SCF-wrapped"):
        calc.get_hessian(atoms=atoms)


def test_hessian_still_works_for_plain_magnetic_models():
    """The refusal must be scoped to wrappers, not to magnetic models at large."""
    with default_dtype(torch.float32):
        model = _tiny_magnetic_model()

    atoms = Atoms(
        numbers=[26, 26],
        positions=[[0.0, 0.0, 0.0], [0.0, 0.0, 2.0]],
        cell=[6.0] * 3,
        pbc=True,
    )
    atoms.arrays["REF_magmom"] = np.tile([[0.0, 0.0, 2.2]], (2, 1))
    calc = MagneticMACECalculator(
        models=[model], device="cpu", default_dtype="float32", magmom_key="REF_magmom"
    )

    hessian = calc.get_hessian(atoms=atoms)
    assert hessian.shape == (3 * len(atoms), len(atoms), 3)
    assert np.isfinite(hessian).all()
