import os
import subprocess
import sys
from copy import deepcopy
from pathlib import Path

import ase.io
import numpy as np
import pytest
from ase.atoms import Atoms

from mace.calculators.mace import MACECalculator
from mace.cli.run_train import run as run_mace_train
from mace.data.utils import KeySpecification
from mace.tools import build_default_arg_parser

run_train = Path(__file__).parent.parent / "mace" / "cli" / "run_train.py"


_mace_params = {
    "name": "MACE",
    "valid_fraction": 0.05,
    "energy_weight": 1.0,
    "forces_weight": 10.0,
    "stress_weight": 1.0,
    "model": "MACE",
    "hidden_irreps": "128x0e",
    "max_num_epochs": 10,
    "swa": None,
    "start_swa": 5,
    "ema": None,
    "ema_decay": 0.99,
    "amsgrad": None,
    "device": "cpu",
    "seed": 5,
    "loss": "weighted",
    "energy_key": "REF_energy",
    "forces_key": "REF_forces",
    "stress_key": "REF_stress",
    "interaction_first": "RealAgnosticResidualInteractionBlock",
    "batch_size": 1,
    "valid_batch_size": 1,
    "num_samples_pt": 50,
    "subselect_pt": "random",
    "eval_interval": 2,
    "num_radial_basis": 10,
    "r_max": 6.0,
    "default_dtype": "float64",
    "use_reduced_cg": False,
}


def configs_numbered_keys():
    np.random.seed(0)
    water = Atoms(
        numbers=[8, 1, 1],
        positions=[[0, -2.0, 0], [1, 0, 0], [0, 1, 0]],
        cell=[4] * 3,
        pbc=[True] * 3,
    )

    energies = list(np.random.normal(0.1, size=15))
    forces = list(np.random.normal(0.1, size=(15, 3, 3)))

    trial_configs_lists = []
    # some keys present, some not
    keys_to_use = (
        ["REF_energy"]
        + ["2_energy"] * 2
        + ["3_energy"] * 3
        + ["4_energy"] * 4
        + ["5_energy"] * 5
    )

    force_keys_to_use = (
        ["REF_forces"]
        + ["2_forces"] * 2
        + ["3_forces"] * 3
        + ["4_forces"] * 4
        + ["5_forces"] * 5
    )

    for ind in range(15):
        c = deepcopy(water)
        c.info[keys_to_use[ind]] = energies[ind]
        c.arrays[force_keys_to_use[ind]] = forces[ind]
        c.positions += np.random.normal(0.1, size=(3, 3))
        trial_configs_lists.append(c)

    return trial_configs_lists


def trial_yamls_and_and_expected():
    yamls = {}
    command_line_kwargs = {"energy_key": "2_energy", "forces_key": "2_forces"}

    yamls["no_heads"] = {}

    yamls["one_head_no_dicts"] = {
        "heads": {
            "Default": {
                "energy_key": "3_energy",
            }
        }
    }

    yamls["one_head_with_dicts"] = {
        "heads": {
            "Default": {
                "info_keys": {
                    "energy": "3_energy",
                },
                "arrays_keys": {
                    "forces": "3_forces",
                },
            }
        }
    }

    yamls["two_heads_no_dicts"] = {
        "heads": {
            "dft": {
                "train_file": "fit_multihead_dft.xyz",
                "energy_key": "3_energy",
            },
            "mp2": {
                "train_file": "fit_multihead_mp2.xyz",
                "energy_key": "4_energy",
            },
        }
    }

    yamls["two_heads_mixed"] = {
        "heads": {
            "dft": {
                "train_file": "fit_multihead_dft.xyz",
                "info_keys": {
                    "energy": "3_energy",
                },
                "arrays_keys": {
                    "forces": "3_forces",
                },
                "forces_key": "4_forces",
            },
            "mp2": {
                "train_file": "fit_multihead_mp2.xyz",
                "energy_key": "4_energy",
            },
        }
    }
    all_arg_sets = {
        "with_command_line": {
            key: {**command_line_kwargs, **value} for key, value in yamls.items()
        },
        "without_command_line": yamls,
    }

    all_expected_outputs = {
        "with_command_line": {
            "no_heads": [
                1.0037831178668188,
                1.0183291323603265,
                1.0120784084221528,
                0.9935695881012243,
                1.0021641561865526,
                0.9999135609205868,
                0.9809440616323108,
                1.0025784765050076,
                1.0017901145495376,
                1.0136913185404515,
                1.006798563238269,
                1.0187758397828384,
                1.0180201540775071,
                1.0132368725061702,
                0.9998734173248169,
            ],
            "one_head_no_dicts": [
                1.0028437510688613,
                1.0514693378041775,
                1.059933403321331,
                1.034719940573569,
                1.0438040675561824,
                1.019719477728329,
                0.9841759692947915,
                1.0435266573857496,
                1.0339501989779065,
                1.0501795448530264,
                1.0402594216704781,
                1.0604998765679152,
                1.0633411200246015,
                1.0539071190201297,
                1.0393496428177804,
            ],
            "one_head_with_dicts": [
                0.8638341551096959,
                1.0078341354784144,
                1.0149701178418595,
                0.9945723048460148,
                1.0184158011731292,
                0.9992135295205004,
                0.8943420783639198,
                1.0327920054084088,
                0.9905731198078909,
                0.9838325204450648,
                1.0018725575620482,
                1.007263052421034,
                1.0335213929231966,
                1.0033503312511205,
                1.0174433894759563,
            ],
            "two_heads_no_dicts": [
                0.9836377578288774,
                1.0196844186291318,
                1.0151628222871238,
                0.957307281711648,
                0.985574141310865,
                0.9629670134047853,
                0.9242583185138095,
                0.9807770070311039,
                0.9973679440479541,
                1.0221127246963275,
                1.0031807967874216,
                1.0358701219543687,
                1.0434208761164758,
                1.0235606028124515,
                0.9797494630655053,
            ],
            "two_heads_mixed": [
                0.8664108574741868,
                0.9907166576278023,
                1.0051969372365164,
                0.978702477000018,
                1.025500166764692,
                0.9940095566375018,
                0.9034029726954119,
                1.0391739502744488,
                0.9717327061183668,
                0.972292103670355,
                1.0012510461663253,
                0.9978051155885286,
                1.0378611651753475,
                1.0003207628186224,
                1.0209509292189651,
            ],
        },
        "without_command_line": {
            "no_heads": [
                0.9352605307451007,
                0.991084559389268,
                0.9940350095024881,
                0.9953849198103668,
                0.9954705498032904,
                0.9964815693808411,
                0.9663142667436776,
                0.9947223808739147,
                0.9897776682803257,
                0.989027769690667,
                0.9910280920241263,
                0.992067980667518,
                0.9917276132506404,
                0.9902848752169671,
                0.9928585982942544,
            ],
            "one_head_no_dicts": [
                0.9425342207393083,
                1.0149788456087416,
                1.0249228965652788,
                1.0247924743285792,
                1.02732103964481,
                1.0168852937950326,
                0.9771283495170653,
                1.0261776335561517,
                1.0130461033368028,
                1.0162619153561783,
                1.019995179866916,
                1.0209512298344965,
                1.0219971755636952,
                1.0195791901659124,
                1.0234662527729408,
            ],
            "one_head_with_dicts": [
                0.8638341551096959,
                1.0078341354784144,
                1.0149701178418595,
                0.9945723048460148,
                1.0184158011731292,
                0.9992135295205004,
                0.8943420783639198,
                1.0327920054084088,
                0.9905731198078909,
                0.9838325204450648,
                1.0018725575620482,
                1.007263052421034,
                1.0335213929231966,
                1.0033503312511205,
                1.0174433894759563,
            ],
            "two_heads_no_dicts": [
                0.9933763730233168,
                0.9986480398559268,
                1.0042486164355315,
                1.0025568793877726,
                1.0032598081704625,
                0.9926714183717912,
                0.9920385249670881,
                1.0020278841030676,
                1.0012474150830537,
                1.0039289677261019,
                1.0022718878661814,
                1.003586385624809,
                1.003436450009097,
                1.003805673887942,
                1.001450261102316,
            ],
            "two_heads_mixed": [
                0.8781767864616707,
                0.9843563603794138,
                1.0145197579049248,
                0.9835060778675391,
                1.0419060462994596,
                0.9917393978520056,
                0.9091521032773944,
                1.0605463095070453,
                0.9685381713826684,
                0.9866493058823766,
                1.00305061187164,
                1.0051273128414386,
                1.037964258398104,
                1.0106663924241408,
                1.0274351814133602,
            ],
        },
    }

    list_of_all = []
    for key, value in all_arg_sets.items():
        for key2, value2 in value.items():
            list_of_all.append(
                (value2, (key, key2), np.asarray(all_expected_outputs[key][key2]))
            )

    return list_of_all


def dict_to_yaml_str(data, indent=0):
    yaml_str = ""
    for key, value in data.items():
        yaml_str += " " * indent + str(key) + ":"
        if isinstance(value, dict):
            yaml_str += "\n" + dict_to_yaml_str(value, indent + 2)
        else:
            yaml_str += " " + str(value) + "\n"
    return yaml_str


_trial_yamls_and_and_expected = trial_yamls_and_and_expected()


@pytest.mark.parametrize(
    "yaml_contents, name, expected_value", _trial_yamls_and_and_expected
)
def test_key_specification_methods(tmp_path, yaml_contents, name, expected_value):
    fitting_configs = configs_numbered_keys()

    ase.io.write(tmp_path / "fit_multihead_dft.xyz", fitting_configs)
    ase.io.write(tmp_path / "fit_multihead_mp2.xyz", fitting_configs)
    ase.io.write(tmp_path / "duplicated_fit_multihead_dft.xyz", fitting_configs)

    mace_params = _mace_params.copy()
    mace_params["valid_fraction"] = 0.1
    mace_params["checkpoints_dir"] = str(tmp_path)
    mace_params["model_dir"] = str(tmp_path)
    mace_params["train_file"] = "fit_multihead_dft.xyz"
    mace_params["E0s"] = "{1:0.0,8:1.0}"
    mace_params["valid_file"] = "duplicated_fit_multihead_dft.xyz"
    del mace_params["valid_fraction"]
    mace_params["max_num_epochs"] = 1  # many tests to do
    del mace_params["energy_key"]
    del mace_params["forces_key"]
    del mace_params["stress_key"]

    mace_params["name"] = "MACE_"

    filename = tmp_path / "config.yaml"
    with open(filename, "w", encoding="utf-8") as file:
        file.write(dict_to_yaml_str(yaml_contents))
    if len(yaml_contents) > 0:
        mace_params["config"] = str(tmp_path / "config.yaml")

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

    p = subprocess.run(cmd.split(), env=run_env, cwd=tmp_path, check=True)
    assert p.returncode == 0

    if "heads" in yaml_contents:
        headname = list(yaml_contents["heads"].keys())[0]
    else:
        headname = "Default"

    calc = MACECalculator(
        tmp_path / "MACE_.model", device="cpu", default_dtype="float64", head=headname
    )

    Es = []
    for at in fitting_configs:
        at.calc = calc
        Es.append(at.get_potential_energy())

    print(name)
    print("Es", Es)

    assert np.allclose(
        np.asarray(Es), expected_value, rtol=1e-8, atol=1e-8
    ), f"Expected {expected_value} but got {Es} with error {np.max(np.abs(Es - expected_value))}"


def test_multihead_finetuning_does_not_modify_default_keyspec(tmp_path):
    fitting_configs = configs_numbered_keys()
    ase.io.write(tmp_path / "fit_multihead_dft.xyz", fitting_configs)

    args = build_default_arg_parser().parse_args(
        [
            "--name",
            "_MACE_",
            "--train_file",
            str(tmp_path / "fit_multihead_dft.xyz"),
            "--foundation_model",
            "small",
            "--device",
            "cpu",
            "--E0s",
            "{1:0.0,8:1.0}",
            "--energy_key",
            "2_energy",
            "--dry_run",
        ]
    )
    default_key_spec = KeySpecification.from_defaults()
    default_key_spec.info_keys["energy"] = "2_energy"
    run_mace_train(args)
    assert args.key_specification == default_key_spec


# for creating values
def make_output():
    outputs = {}
    for yaml_contents, name, expected_value in _trial_yamls_and_and_expected:
        if name[0] not in outputs:
            outputs[name[0]] = {}
        expected = test_key_specification_methods(
            Path("."), yaml_contents, name, expected_value, debug_test=False
        )
        outputs[name[0]][name[1]] = expected
    print(outputs)
