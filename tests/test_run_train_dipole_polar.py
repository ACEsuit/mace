import os
import subprocess
import sys
from pathlib import Path

import ase.io
import numpy as np
import pytest
from ase.atoms import Atoms

from mace.calculators import MACECalculator

try:
    import cuequivariance as cue  # pylint: disable=unused-import

    CUET_AVAILABLE = True
except ImportError:
    CUET_AVAILABLE = False

run_train = Path(__file__).parent.parent / "mace" / "cli" / "run_train.py"


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
        c.info["REF_dipoles"] = np.random.normal(0.1, size=3)
        c.info["REF_polarizability"] = np.random.normal(0.1, size=(3, 3))

        fit_configs.append(c)

    return fit_configs



@pytest.fixture(name="pretraining_configs")
def fixture_pretraining_configs():
    configs = []
    for _ in range(10):
        atoms = Atoms(
            numbers=[8, 1, 1],
            positions=np.random.rand(3, 3) * 3,
            cell=[5, 5, 5],
            pbc=[True] * 3,
        )
        atoms.info["REF_energy"] = np.random.normal(0, 1)
        atoms.arrays["REF_forces"] = np.random.normal(0, 1, size=(3, 3))
        atoms.info["REF_stress"] = np.random.normal(0, 1, size=6)
        atoms.info["REF_dipoles"] = np.random.normal(0.1, size=3)
        atoms.info["REF_polarizability"] = np.random.normal(0.1, size=(3, 3))
        configs.append(atoms)
    configs.append(
        Atoms(numbers=[8], positions=[[0, 0, 0]], cell=[6] * 3, pbc=[True] * 3),
    )
    configs.append(
        Atoms(numbers=[1], positions=[[0, 0, 0]], cell=[6] * 3, pbc=[True] * 3)
    )
    configs[-2].info["REF_energy"] = -2.0
    configs[-2].info["config_type"] = "IsolatedAtom"
    configs[-1].info["REF_energy"] = -4.0
    configs[-1].info["config_type"] = "IsolatedAtom"
    return configs

_mace_params_dipole = {
    "name": "DipolesMACE",
    "valid_fraction": 0.05,
    "energy_weight": 1.0,
    "forces_weight": 10.0,
    "stress_weight": 1.0,
    "dipole_weight": 1.0,
    "model": "AtomicDipolesMACE",
    "r_max": 3.5,
    "max_L": 1,
    "batch_size": 5,
    "max_num_epochs": 10,
    "ema_decay": 0.99,
    "amsgrad": None,
    "restart_latest": None,
    "device": "cpu",
    "seed": 5,
    "loss": "dipole",
    "error_table": "DipoleRMSE",
    "energy_key": "REF_energy",
    "forces_key": "REF_forces",
    "stress_key": "REF_stress",
    "dipole_key": "REF_dipoles",
    "eval_interval": 2,
    "use_reduced_cg": False,
    "compute_atomic_dipole":True,
}


def test_run_train_dipole(tmp_path, fitting_configs):
    ase.io.write(tmp_path / "fit.xyz", fitting_configs)

    mace_params = _mace_params_dipole.copy()
    mace_params["checkpoints_dir"] = str(tmp_path)
    mace_params["model_dir"] = str(tmp_path)
    mace_params["train_file"] = tmp_path / "fit.xyz"

    # make sure run_train.py is using the mace that is currently being tested
    run_env = os.environ.copy()
    sys.path.insert(0, str(Path(__file__).parent.parent))
    run_env["PYTHONPATH"] = ":".join(sys.path)
    print("DEBUG subprocess PYTHONPATH", run_env["PYTHONPATH"])

    cmd = (
        sys.executable
        + " "
        + str(run_train)
        + " "
        + " ".join(
            [
                (f"--{k}={v}" if v is not None else f"--{k}")
                for k, v in mace_params.items()
            ]
        )
    )

    p = subprocess.run(cmd.split(), env=run_env, check=True)
    assert p.returncode == 0

    calc = MACECalculator(
        model_paths=tmp_path / "DipolesMACE.model", model_type="DipoleMACE", device="cpu"
    )

    Mus = []
    for at in fitting_configs:
        at.calc = calc
        Mus.append(at.get_dipole_moment())
    print("Mus", Mus)
    # Obtained for MACE from the 08/08/2025
    ref_Mus = [np.array([0., 0., 0.]),
               np.array([0., 0., 0.]),
               np.array([ 0.00187852,  0.00347198, -0.00038684]),
               np.array([0.11058126, 0.06134831, 0.03802504]),
               np.array([ 0.06034323, -0.0547419 ,  0.01613814]),
               np.array([ 0.15007136, -0.03509335, -0.09119466]),
               np.array([-0.00953865, -0.00407877, -0.00667097]),
               np.array([-0.01117118, -0.09511023, -0.08482508]),
               np.array([-0.13181527, -0.02788298, -0.02729993]),
               np.array([ 0.00195789,  0.00375182, -0.00490151]),
               np.array([-0.01337001, -0.01065634, -0.04872033]),
               np.array([-0.00657708,  0.00228718,  0.01085754]),
               np.array([-0.00254496, -0.00755704,  0.00789225]),
               np.array([ 0.04717364, -0.07645017, -0.0346127 ]),
               np.array([-0.00453682, -0.01823174, -0.0004906 ]),
               np.array([0.00906295, 0.00062374, 0.00377266]),
               np.array([ 0.01442582, -0.07775026, -0.02393251]),
               np.array([-0.26496943,  0.05997367, -0.09351538]),
               np.array([-0.00118423,  0.00119765,  0.0065717 ]),
               np.array([-0.00270274,  0.00782406, -0.01322653]),
               np.array([-0.02180288, -0.02410089,  0.04983753]),
               np.array([-0.00539038, -0.00231573, -0.01304079])
               ]

    assert np.allclose(Mus, ref_Mus)

_mace_params_dipole_polar = {
    "name": "DielectricMACE",
    "valid_fraction": 0.05,
    "energy_weight": 1.0,
    "forces_weight": 10.0,
    "stress_weight": 1.0,
    "dipole_weight": 1.0,
    "polarizability_weight": 1.0,
    "model": "AtomicDielectricMACE",
    "r_max": 3.5,
    "max_L": 2,
    "batch_size": 5,
    "max_num_epochs": 10,
    "ema_decay": 0.99,
    "amsgrad": None,
    "restart_latest": None,
    "device": "cpu",
    "seed": 5,
    "loss": "dipole_polar",
    "error_table": "DipolePolarRMSE",
    "energy_key": "REF_energy",
    "forces_key": "REF_forces",
    "stress_key": "REF_stress",
    "dipole_key": "REF_dipoles",
    "polarizability_key": "REF_polarizability",
    "eval_interval": 2,
    "use_reduced_cg": False,
    "compute_polarizability": True,
}


def test_run_train_dipole_polar(tmp_path, fitting_configs):
    ase.io.write(tmp_path / "fit.xyz", fitting_configs)

    mace_params = _mace_params_dipole_polar.copy()
    mace_params["checkpoints_dir"] = str(tmp_path)
    mace_params["model_dir"] = str(tmp_path)
    mace_params["train_file"] = tmp_path / "fit.xyz"

    # make sure run_train.py is using the mace that is currently being tested
    run_env = os.environ.copy()
    sys.path.insert(0, str(Path(__file__).parent.parent))
    run_env["PYTHONPATH"] = ":".join(sys.path)
    print("DEBUG subprocess PYTHONPATH", run_env["PYTHONPATH"])

    cmd = (
        sys.executable
        + " "
        + str(run_train)
        + " "
        + " ".join(
            [
                (f"--{k}={v}" if v is not None else f"--{k}")
                for k, v in mace_params.items()
            ]
        )
    )

    p = subprocess.run(cmd.split(), env=run_env, check=True)
    assert p.returncode == 0

    calc = MACECalculator(
        model_paths=tmp_path / "DielectricMACE.model",
        model_type="DipolePolarizabilityMACE",
        device="cpu",
    )

    Mus = []
    alphas = []
    for at in fitting_configs:
        at.calc = calc
        Mus.append(at.get_dipole_moment())
        alphas.append(calc.get_property("polarizability", at))
    # Obtained for MACE from the 08/08/2025
    ref_Mus = [
        np.array([0., 0., 0.]),
        np.array([0., 0., 0.]),
        np.array([-0.00405055,  0.04989444, -0.03235187]),
        np.array([0.07428956, 0.17534762, 0.0344151 ]),
        np.array([0.22928963, 0.22789771, 0.01154527]),
        np.array([ 0.03826594, -0.07032791, -0.04752111]),
        np.array([-0.01850784, -0.0397861 , -0.02993542]),
        np.array([-0.08677283, -0.16031447, -0.07839491]),
        np.array([-0.10964872, -0.016287  , -0.02204037]),
        np.array([0.07918389, 0.08737918, 0.04145608]),
        np.array([-0.02907002,  0.01990042, -0.025413  ]),
        np.array([-0.04857212,  0.07961402, -0.01174765]),
        np.array([-0.00435445,  0.06191188, -0.01095293]),
        np.array([ 0.12814346, -0.16236763, -0.08407766]),
        np.array([0.01998741, 0.06349773, 0.10308535]),
        np.array([-0.01837022,  0.14428979, -0.08086759]),
        np.array([-0.01595687, -0.10272478, -0.0236392 ]),
        np.array([-0.1400921 , -0.087709  , -0.05998242]),
        np.array([-0.0009531 ,  0.11478384,  0.01394257]),
        np.array([0.01886107, 0.06399308, 0.03219055]),
        np.array([-0.0578738 , -0.01193058,  0.09853157]),
        np.array([0.06845842, 0.55215253, 0.16271066])
    ]

    ref_alphas = [
        np.array([[0., 0., 0.],
                [0., 0., 0.],
                [0., 0., 0.]]),

        np.array([[0., 0., 0.],
                [0., 0., 0.],
                [0., 0., 0.]]),

        np.array([[-0.01774253,  0.00307792,  0.01980522],
                [ 0.00307792, -0.01334953, -0.01574966],
                [ 0.01980522, -0.01574966, -0.00921213]]),

        np.array([[-0.10119078, -0.13912649, -0.06179769],
                [-0.13912649,  0.01985949,  0.0011849 ],
                [-0.06179769,  0.0011849 ,  0.09128444]]),

        np.array([[-0.08015337,  0.28906984, -0.05851742],
                [ 0.28906984, -0.10814388,  0.05704551],
                [-0.05851742,  0.05704551,  0.19103685]]),

        np.array([[-0.10977471, -0.04570234,  0.28398318],
                [-0.04570234,  0.23783969, -0.08478971],
                [ 0.28398318, -0.08478971, -0.01218864]]),

        np.array([[-0.00033354,  0.00791392,  0.01818803],
                [ 0.00791392, -0.01052066,  0.01542726],
                [ 0.01818803,  0.01542726, -0.02606526]]),

        np.array([[ 0.1590121 , -0.0117003 ,  0.02571063],
                [-0.0117003 , -0.11642923, -0.18587264],
                [ 0.02571063, -0.18587264,  0.00499395]]),

        np.array([[-0.14361682, -0.02639792, -0.04795591],
                [-0.02639792,  0.12444872, -0.00903552],
                [-0.04795591, -0.00903552,  0.07012009]]),

        np.array([[-0.02385325,  0.00278218, -0.00900929],
                [ 0.00278218, -0.02249226,  0.03559279],
                [-0.00900929,  0.03559279, -0.03665955]]),

        np.array([[ 0.19766853, -0.05722875, -0.15204635],
                [-0.05722875,  0.16751105, -0.20498558],
                [-0.15204635, -0.20498558, -0.25885304]]),

        np.array([[-3.33497839e-02,  5.84507102e-03,  1.03779892e-02],
                [ 5.84507102e-03, -3.23186866e-02,  8.11547509e-03],
                [ 1.03779892e-02,  8.11547509e-03,  5.04013953e-05]]),

        np.array([[-0.03121629, -0.00526901,  0.00280257],
                [-0.00526901, -0.01555394, -0.00570221],
                [ 0.00280257, -0.00570221, -0.00175081]]),

        np.array([[ 0.08495039,  0.20898154,  0.15077567],
                [ 0.20898154, -0.06027019, -0.18349961],
                [ 0.15077567, -0.18349961,  0.13150356]]),

        np.array([[ 0.02664201, -0.01412724,  0.01773262],
                [-0.01412724, -0.01550847, -0.00363691],
                [ 0.01773262, -0.00363691, -0.00834004]]),

        np.array([[-0.02283877, -0.00013611, -0.01089784],
                [-0.00013611, -0.01043637, -0.00651823],
                [-0.01089784, -0.00651823, -0.03756097]]),

        np.array([[ 0.02721329, -0.0434384 ,  0.00925382],
                [-0.0434384 , -0.04067187, -0.01123916],
                [ 0.00925382, -0.01123916,  0.03137444]]),

        np.array([[ 0.13941369,  0.1125973 ,  0.0079224 ],
                [ 0.1125973 , -0.13189636,  0.07080912],
                [ 0.0079224 ,  0.07080912, -0.02471156]]),

        np.array([[-0.00891008,  0.00219625, -0.00342596],
                [ 0.00219625,  0.00338428,  0.00039052],
                [-0.00342596,  0.00039052, -0.03264827]]),

        np.array([[-0.00305826,  0.03268619, -0.00097526],
                [ 0.03268619, -0.02749376,  0.02723039],
                [-0.00097526,  0.02723039, -0.01024948]]),

        np.array([[-0.0152999 , -0.01413736,  0.01099587],
                [-0.01413736, -0.01621853,  0.01282218],
                [ 0.01099587,  0.01282218, -0.02029655]]),

        np.array([[-0.11326049,  0.00939965, -0.00715356],
                [ 0.00939965, -0.2316651 ,  0.03025837],
                [-0.00715356,  0.03025837, -0.10757094]])
]

    assert np.allclose(Mus, ref_Mus)
    assert np.allclose(alphas, ref_alphas)
