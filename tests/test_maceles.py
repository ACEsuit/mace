
import ase.io
import numpy as np
import pytest
from ase.atoms import Atoms
from mace.tools.arg_parser import build_default_arg_parser
from mace.cli.run_train import run as mace_run
from mace.calculators import MACECalculator

try:
    from les import Les
    LES_AVAILABLE = True
except ImportError:
    LES_AVAILABLE = False



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
