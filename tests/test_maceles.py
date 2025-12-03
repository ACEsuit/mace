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

from mace.calculators import MACECalculator
from mace.cli.eval_configs import run as mace_eval_configs_run
from mace.cli.run_train import run as mace_run
from mace.modules import interaction_classes
from mace.modules.extensions import MACELES
from mace.modules.models import ScaleShiftMACE
from mace.tools.arg_parser import build_default_arg_parser
from mace.tools.torch_tools import default_dtype

LES_AVAILABLE = bool((spec := importlib.util.find_spec("les")) is not None)
CUET_AVAILABLE = bool((spec := importlib.util.find_spec("cuequivariance")) is not None)
CUDA_AVAILABLE = torch.cuda.is_available()


@pytest.fixture(name="fitting_configs")
def fixture_fitting_configs():
    water = Atoms(
        numbers=[8, 1, 1],
        positions=[[0, -2.0, 0], [1, 0, 0], [0, 1, 0]],
        cell=[4] * 3,
        pbc=[True] * 3,
    )
    fit_configs = [
        Atoms(numbers=[8], positions=[[0, 0, 0]], cell=[6] * 3),
        Atoms(numbers=[1], positions=[[0, 0, 0]], cell=[6] * 3),
    ]
    fit_configs[0].info["REF_energy"] = 0.0
    fit_configs[0].info["config_type"] = "IsolatedAtom"
    fit_configs[1].info["REF_energy"] = 0.0
    fit_configs[1].info["config_type"] = "IsolatedAtom"

    np.random.seed(5)
    for _ in range(20):
        c = water.copy()
        c.positions += np.random.normal(0.1, size=c.positions.shape)
        c.info["REF_energy"] = np.random.normal(0.1)
        print(c.info["REF_energy"])
        c.new_array("REF_forces", np.random.normal(0.1, size=c.positions.shape))
        c.info["REF_stress"] = np.random.normal(0.1, size=6)
        fit_configs.append(c)

    return fit_configs


_mace_params = {
    "name": "MACE",
    "valid_fraction": 0.05,
    "energy_weight": 1.0,
    "forces_weight": 10.0,
    "stress_weight": 1.0,
    "model": "MACELES",
    "hidden_irreps": "128x0e",
    "r_max": 3.5,
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
    "stress_key": "REF_stress",
    "eval_interval": 2,
    "use_reduced_cg": False,
}


@pytest.mark.skipif(not LES_AVAILABLE, reason="LES library is not available")
def test_run_train(tmp_path, fitting_configs):
    ase.io.write(tmp_path / "fit.xyz", fitting_configs)

    mace_params = _mace_params.copy()
    mace_params["checkpoints_dir"] = str(tmp_path)
    mace_params["model_dir"] = str(tmp_path)
    mace_params["train_file"] = tmp_path / "fit.xyz"
    args = build_default_arg_parser().parse_args(
        [f"--{k}={v}" if v is not None else f"--{k}" for k, v in mace_params.items()]
    )

    mace_run(args)

    calc = MACECalculator(model_paths=tmp_path / "MACE.model", device="cpu")

    Es = []
    for at in fitting_configs:
        at.calc = calc
        Es.append(at.get_potential_energy())

    print("Es", Es)
    ref_Es = [
        0.004919160731848143,
        0.5906680240792959,
        0.47887544882572264,
        0.4176002467254094,
        0.5606673227439406,
        0.40181714730443363,
        0.3367534132795259,
        0.27118917957971056,
        0.47967529915910134,
        0.32077479180773283,
        1.2865402405977537,
        0.3472478715875782,
        0.427734507004752,
        0.8092185237225293,
        0.38348242384362774,
        0.14448973657513398,
        0.5650118900854595,
        0.429029669763921,
        0.4837945154901776,
        0.2244894146891574,
        0.3667896493444026,
        0.23811703879534651,
    ]

    assert np.allclose(Es, ref_Es)


@pytest.mark.skipif(not LES_AVAILABLE, reason="LES library is not available")
def test_run_train_with_mp(tmp_path, fitting_configs):
    ase.io.write(tmp_path / "fit.xyz", fitting_configs)

    mace_params = _mace_params.copy()
    mace_params["checkpoints_dir"] = str(tmp_path)
    mace_params["foundation_model"] = "small"
    mace_params["hidden_irreps"] = "128x0e"
    mace_params["r_max"] = 6.0
    mace_params["default_dtype"] = "float64"
    mace_params["num_radial_basis"] = 10
    mace_params["interaction_first"] = "RealAgnosticResidualInteractionBlock"
    mace_params["multiheads_finetuning"] = False
    mace_params["model_dir"] = str(tmp_path)
    mace_params["train_file"] = tmp_path / "fit.xyz"
    args = build_default_arg_parser().parse_args(
        [f"--{k}={v}" if v is not None else f"--{k}" for k, v in mace_params.items()]
    )

    mace_run(args)

    calc = MACECalculator(model_paths=tmp_path / "MACE.model", device="cpu")

    Es = []
    for at in fitting_configs:
        at.calc = calc
        Es.append(at.get_potential_energy())

    print("Es", Es)
    ref_Es = [
        1.9041867483100463,
        0.9795927664122093,
        0.6143645372728241,
        0.540857367104403,
        0.2175412746398953,
        0.5204824602823621,
        0.42691720944924566,
        0.47462694450178505,
        0.5809854217525379,
        0.3586733195403562,
        3.755376867799749,
        0.6308930408544482,
        0.5298001079484215,
        0.7923006837586871,
        0.7015445400430391,
        0.5558430181089493,
        0.6546531810601435,
        0.7309926712585781,
        0.6821026693847355,
        0.30473441126045364,
        0.5945371974398417,
        0.6601282822585335,
    ]

    assert np.allclose(Es, ref_Es)


@pytest.mark.skipif(
    not (LES_AVAILABLE and CUET_AVAILABLE and CUDA_AVAILABLE),
    reason="Testing MACELES cueq training requires LES, cuequivariance, and CUDA to be available",
)
def test_run_train_maceles_cueq(tmp_path, fitting_configs):
    ase.io.write(tmp_path / "fit.xyz", fitting_configs)
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":16:8"
    mace_params = _mace_params.copy()
    mace_params["checkpoints_dir"] = str(tmp_path)
    mace_params["model_dir"] = str(tmp_path)
    mace_params["train_file"] = tmp_path / "fit.xyz"
    mace_params["device"] = "cuda"
    mace_params["enable_cueq"] = True
    args = build_default_arg_parser().parse_args(
        [f"--{k}={v}" if v is not None else f"--{k}" for k, v in mace_params.items()]
    )
    # Seed torch, and enable deterministic algorithms for reproducibility
    torch.manual_seed(5)
    torch.use_deterministic_algorithms(True)

    # Run the training
    mace_run(args)

    calc = MACECalculator(model_paths=tmp_path / "MACE.model", device="cpu")

    Es = []
    for at in fitting_configs:
        at.calc = calc
        Es.append(at.get_potential_energy())

    print("Es", Es)
    ref_Es = [
        0.004919160731848143,
        0.5906680240792959,
        0.47887544882572264,
        0.4176002467254094,
        0.5606673227439406,
        0.40181714730443363,
        0.3367534132795259,
        0.27118917957971056,
        0.47967529915910134,
        0.32077479180773283,
        1.2865402405977537,
        0.3472478715875782,
        0.427734507004752,
        0.8092185237225293,
        0.38348242384362774,
        0.14448973657513398,
        0.5650118900854595,
        0.429029669763921,
        0.4837945154901776,
        0.2244894146891574,
        0.3667896493444026,
        0.23811703879534651,
    ]
    assert np.allclose(Es, ref_Es)


MODEL_CONFIG = dict(
    r_max=5,
    num_bessel=8,
    num_polynomial_cutoff=6,
    max_ell=2,
    interaction_cls=interaction_classes["RealAgnosticResidualInteractionBlock"],
    interaction_cls_first=interaction_classes["RealAgnosticResidualInteractionBlock"],
    num_interactions=5,
    num_elements=2,
    hidden_irreps=o3.Irreps("32x0e + 32x1o"),
    MLP_irreps=o3.Irreps("16x0e"),
    gate=torch.nn.functional.silu,
    atomic_energies=np.zeros(2),
    avg_num_neighbors=8,
    atomic_numbers=[1, 8],
    correlation=3,
    radial_type="bessel",
    atomic_inter_shift=0.0,
    atomic_inter_scale=1.0,
)


@pytest.fixture(name="mace_model_path")
def mace_model_path_fixture(tmp_path: Path) -> Path:
    """Create and save a standard ScaleShiftMACE model."""
    with default_dtype(torch.float32):
        model = ScaleShiftMACE(**MODEL_CONFIG)
        path = tmp_path / "mace.model"
        torch.save(model, path)
    return path


@pytest.mark.skipif(not LES_AVAILABLE, reason="LES library is not available")
@pytest.fixture(name="maceles_model_path")
def maceles_model_path_fixture(tmp_path: Path) -> Path:
    """Create and save a MACELES model."""
    with default_dtype(torch.float32):
        model = MACELES(**MODEL_CONFIG)
        path = tmp_path / "maceles.model"
        torch.save(model, path)
    return path


@pytest.mark.skipif(not LES_AVAILABLE, reason="LES library is not available")
def test_run_eval_with_bec(tmp_path: Path, maceles_model_path: Path, fitting_configs):
    """Tests running evaluation with BEC computation enabled."""
    output_path = tmp_path / "output.xyz"
    ase.io.write(tmp_path / "fit.xyz", fitting_configs)
    args = argparse.Namespace(
        model=str(maceles_model_path),
        configs=str(tmp_path / "fit.xyz"),
        output=str(output_path),
        device="cpu",
        default_dtype="float32",
        batch_size=1,
        compute_stress=False,
        compute_bec=True,
        enable_cueq=False,
        return_contributions=False,
        return_descriptors=False,
        return_node_energies=False,
        info_prefix="MACE_",
        head=None,
    )
    mace_eval_configs_run(args)

    assert output_path.exists()
    output_atoms = ase.io.read(str(output_path), index=":")
    assert len(output_atoms) == len(fitting_configs)

    for at in output_atoms:
        assert isinstance(at, Atoms)
        assert "MACE_BEC" in at.arrays
        assert "MACE_latent_charges" in at.arrays
        assert at.arrays["MACE_BEC"].shape == (len(at), 9)
        assert at.arrays["MACE_latent_charges"].shape == ((len(at),))


def test_run_eval_fail_with_wrong_model(
    tmp_path: Path, mace_model_path: Path, fitting_configs
):
    # Test script fails if BEC is requested with a non-MACELES model
    ase.io.write(tmp_path / "fit.xyz", fitting_configs)
    args = argparse.Namespace(
        model=str(mace_model_path),
        configs=str(tmp_path / "fit.xyz"),
        output=str(tmp_path / "output.xyz"),
        device="cpu",
        default_dtype="float32",
        batch_size=1,
        compute_stress=False,
        compute_bec=True,  # Request BEC with wrong model
        enable_cueq=False,
        return_contributions=False,
        return_descriptors=False,
        return_node_energies=False,
        info_prefix="MACE_",
        head=None,
    )

    with pytest.raises(
        ValueError, match="BEC can only be computed with MACELES model."
    ):
        mace_eval_configs_run(args)


@pytest.mark.skipif(not LES_AVAILABLE, reason="LES library is not available")
def test_run_eval_no_bec(tmp_path: Path, maceles_model_path: Path, fitting_configs):
    """Tests running evaluation without requesting BEC."""
    output_path = tmp_path / "output.xyz"
    ase.io.write(tmp_path / "fit.xyz", fitting_configs)
    args = argparse.Namespace(
        model=str(maceles_model_path),
        configs=str(tmp_path / "fit.xyz"),
        output=str(output_path),
        device="cpu",
        default_dtype="float32",
        batch_size=1,
        compute_stress=True,
        compute_bec=False,  # BEC computation is off
        enable_cueq=False,
        return_contributions=False,
        return_descriptors=False,
        return_node_energies=False,
        info_prefix="MACE_",
        head=None,
    )
    mace_eval_configs_run(args)

    # Check that the output file exists
    assert output_path.exists()
    output_atoms = ase.io.read(str(output_path), index=":")
    assert len(output_atoms) == len(fitting_configs)
    for at in output_atoms:
        assert isinstance(at, Atoms)
        # Ensure BEC and latent charges are not present
        assert "MACE_BEC" not in at.arrays
        assert "MACE_latent_charges" not in at.arrays
        # Check that other expected arrays are present
        assert "MACE_energy" in at.info
        assert "MACE_stress" in at.info
        assert "MACE_forces" in at.arrays
