import os
import subprocess
import sys
from pathlib import Path

import ase.io
import numpy as np
import pytest
from ase.atoms import Atoms
from copy import deepcopy

from mace.calculators.mace import MACECalculator
import mace
np.random.seed(0)

run_train = Path(__file__).parent.parent / "mace" / "cli" / "run_train.py"



_mace_params = {
    "name": "MACE",
    "valid_fraction": 0.05,
    "energy_weight": 1.0,
    "forces_weight": 10.0,
    "stress_weight": 1.0,
    "model": "MACE",
    "hidden_irreps": "128x0e",
    "r_max": 3.5,
    "batch_size": 5,
    "max_num_epochs": 10,
    "swa": None,
    "start_swa": 5,
    "ema": None,
    "ema_decay": 0.99,
    "amsgrad": None,
    "device": "cpu",
    "seed": 5,
    "loss": "stress",
    "energy_key": "REF_energy",
    "forces_key": "REF_forces",
    "stress_key": "REF_stress",
    "eval_interval": 2,
}


def configs_numbered_keys():
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
    keys_to_use = ["REF_energy"] + \
        ["2_energy"]*2 + \
        ["3_energy"]*3 + \
        ["4_energy"]*4 + \
        ["5_energy"]*5

    force_keys_to_use = ["REF_forces"] + \
        ["2_forces"]*2 + \
        ["3_forces"]*3 + \
        ["4_forces"]*4 + \
        ["5_forces"]*5

    for ind in range(15):
        c = deepcopy(water)
        c.info[keys_to_use[ind]] = energies[ind]
        c.arrays[force_keys_to_use[ind]] = forces[ind]
        c.positions += np.random.normal(0.1, size=(3,3))
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
        "without_command_line": {key: value for key, value in yamls.items()},
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
                1.0000166263499473,
                1.0021620416915131,
                1.0046772896383978,
                1.001465441141607,
                1.0055517812192685,
                1.0015992637882436,
                1.0020402319259156,
                1.0054369609690694,
                1.0048820789691186,
                1.004069245195459,
                1.0036930315433792,
                1.0045657994185517,
                1.0049657202069904,
                1.0054495991318766,
                1.0059574240719107,
            ],
            "one_head_with_dicts": [
                0.9824761809087968,
                0.982723323954806,
                0.9804037844582393,
                0.9892979892015554,
                0.990123250174031,
                0.9872765633686582,
                0.9792985720223041,
                0.9834849185579561,
                0.9855709706241268,
                0.9838176625524332,
                0.9802380433794929,
                0.9798924747115749,
                0.9941246312362003,
                0.9843619552495816,
                1.0234402440454935,
            ],
            "two_heads_no_dicts": [
                0.9533172241443488,
                0.971143149409332,
                0.9591034423596022,
                0.9259180388268078,
                0.9866672025915887,
                0.9468387512978088,
                0.972806503955744,
                0.9268821579802152,
                0.9399783569634511,
                0.9566909477955546,
                1.0280484765877604,
                0.9638781804485581,
                0.9386762390303685,
                0.9513720471682103,
                1.061099519224079,
            ],
            "two_heads_mixed": [
                1.0008821794271117,
                0.9921658975489234,
                1.0128605897789047,
                1.0177680320432732,
                1.0040635968372489,
                1.0134535284156263,
                0.9900156994903402,
                0.9950077226207892,
                0.9931748657782218,
                0.9970869871816835,
                1.0036266515981311,
                0.9882332649269495,
                0.9973620987054619,
                1.0089283927259747,
                0.9984375026446699,
            ],
        },
        "without_command_line": {
            "no_heads": [
                0.9723249939003304,
                0.99830004939027,
                0.9976857883262907,
                1.0026915904907623,
                0.9986047122447201,
                1.0056392530400915,
                0.9955992271879338,
                0.9925618058915322,
                0.9992873743817391,
                1.0017751144824205,
                0.9965424145952742,
                0.9980104982304532,
                0.996970035434205,
                1.0017462160896793,
                1.00453025524217,
            ],
            "one_head_no_dicts": [
                0.9668728328024694,
                0.9559554052674338,
                0.9558003309868804,
                0.9568681942948057,
                0.9471374531635678,
                0.9573665902279203,
                0.9509504944430629,
                0.9449430732494284,
                0.9487872001503757,
                0.9515435134805473,
                0.9616246560028083,
                0.9652201708552365,
                0.9518567860504985,
                0.9695448453855497,
                0.9595931614125687,
            ],
            "one_head_with_dicts": [
                0.9904238487805224,
                0.9787489784129528,
                0.9980000798872206,
                1.0081047579760913,
                0.970990405481672,
                1.0296635919726917,
                1.0070991842774164,
                0.9977357706770508,
                0.9729041794133619,
                0.9952167479342705,
                1.0256795692987708,
                1.0005027614317226,
                1.0042896304620599,
                0.9933015438418198,
                0.9941762126172496,
            ],
            "two_heads_no_dicts": [
                0.8234141049979373,
                0.8486132642907047,
                0.8761921831858267,
                0.8086446850523645,
                0.8185616207749478,
                0.8349295066652644,
                0.8695339796701849,
                0.8783625449137391,
                0.8513575832201994,
                0.8428073015147357,
                0.8514345324682252,
                0.8774982178381736,
                0.8724648944295484,
                0.9071025824523504,
                0.8671562526370659,
            ],
            "two_heads_mixed": [
                1.0142275963817828,
                0.9252946269851097,
                0.9905802472120683,
                1.0104854763203601,
                1.0627569806879018,
                0.894635070244004,
                0.9570335273959514,
                0.9917699286224028,
                0.9731498108644769,
                1.02712188692559,
                1.0255958579172193,
                1.0134291318470228,
                0.9601947878290134,
                0.9593860448787849,
                1.0044099804202045,
            ],
        },
    }


    list_of_all = []
    for key, value in all_arg_sets.items():
        print(key)
        for key2, value2 in value.items():
            print('  ', key2)
            list_of_all.append((value2, (key, key2), np.asarray(all_expected_outputs[key][key2])))

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

@pytest.mark.parametrize("yaml_contents, name, expected_value", _trial_yamls_and_and_expected)
def test_key_specification_methods(tmp_path, yaml_contents, name, expected_value, debug_test=False):
    fitting_configs = configs_numbered_keys()

    ase.io.write(tmp_path / "fit_multihead_dft.xyz", fitting_configs)
    ase.io.write(tmp_path / "fit_multihead_mp2.xyz", fitting_configs)
    ase.io.write(tmp_path / "duplicated_fit_multihead_dft.xyz", fitting_configs)

    mace_params = _mace_params.copy()
    mace_params["valid_fraction"] = 0.1
    mace_params["checkpoints_dir"] = str(tmp_path)
    mace_params["model_dir"] = str(tmp_path)
    mace_params["loss"] = "weighted"
    mace_params["hidden_irreps"] = "128x0e"
    mace_params["r_max"] = 6.0
    mace_params["default_dtype"] = "float64"
    mace_params["num_radial_basis"] = 10
    mace_params["interaction_first"] = "RealAgnosticResidualInteractionBlock"
    mace_params["batch_size"] = 1
    mace_params["valid_batch_size"] = 1
    mace_params["num_samples_pt"] = 50
    mace_params["subselect_pt"] = "random"
    mace_params["train_file"] = "fit_multihead_dft.xyz"
    mace_params["E0s"] = "{1:0.0,8:1.0}"
    mace_params["valid_file"] = "duplicated_fit_multihead_dft.xyz"
    del mace_params["valid_fraction"]
    mace_params["max_num_epochs"] = 1 # many tests to do
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

    if debug_test:
        new_cmd = cmd.replace('--', '\n--')
        print('calling run train with {name}')
        print('command line args:\n', new_cmd)
        print('config.yaml:\n', dict_to_yaml_str(yaml_contents), flush=True)

    p = subprocess.run(cmd.split(), env=run_env, cwd=tmp_path, check=True)
    assert p.returncode == 0

    if 'heads' in yaml_contents:
        headname = list(yaml_contents['heads'].keys())[0]
    else:
        headname = 'Default'

    calc = MACECalculator(
        tmp_path / "MACE_.model", device="cpu", default_dtype="float64", head=headname
    )

    Es = []
    for at in fitting_configs:
        at.calc = calc
        Es.append(at.get_potential_energy())
    
    print(np.asarray(Es))
    print(expected_value)
    print(type(np.asarray(Es)))
    print(type(expected_value))
    assert np.allclose(np.asarray(Es), expected_value)