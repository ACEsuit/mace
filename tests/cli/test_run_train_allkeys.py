import subprocess
from copy import deepcopy
from pathlib import Path

import ase.io
import numpy as np
import numpy.testing as npt
import pytest
from ase.atoms import Atoms

from mace.calculators.mace import MACECalculator
from mace.cli.run_train import run as run_mace_train
from mace.data.utils import KeySpecification
from mace.tools import build_default_arg_parser

run_train = "mace_run_train"


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

    # UPDATED alongside https://github.com/ACEsuit/mace/pull/1009
    # fmt: off
    all_expected_outputs = {
        "with_command_line": {
            "no_heads": [0.9434231912341413, 0.9872075608810948, 0.9906143965220142, 0.9726121782519731, 0.9578010984173894, 0.9420714760605446, 0.9061579202759136, 0.9584825272265999, 0.9858665110888107, 0.9772673602984945, 0.9816583216318682, 1.012073473792806, 1.0016146799095524, 0.9791275872632423, 0.9496380114992509],
            "one_head_no_dicts": [1.0221501386275818, 1.0127991986133698, 1.0178930827901373, 1.009839280165339, 1.0091871434036621, 0.9990643090260968, 1.0010816061498387, 1.0068819221950205, 1.0136994314740237, 1.0180977966177092, 1.0158805131588555, 1.028259408302649, 1.0191959433696696, 1.016577233359923, 1.000627442548274],
            "one_head_with_dicts": [0.7480831408068953, 0.951775499630758, 1.0015875095002453, 0.9819165271183335, 0.9928739014963112, 0.9614852272615066, 0.83099400509797, 1.0355993341035963, 0.9911904393853932, 0.9720007139825775, 0.9784314683102276, 1.0077504173821927, 1.009084672590903, 0.9737687751059977, 0.9694892306335727],
            "two_heads_no_dicts": [0.8762304085576093, 0.9458940890671718, 0.961882252329402, 0.9375438031664655, 0.9163270917668694, 0.881081422334797, 0.8258144720311926, 0.9174176287558287, 0.9533235586682338, 0.9362733843989092, 0.9434567246677942, 0.992833432548966, 0.982777176888275, 0.9443328757805891, 0.8944409250063878],
            "two_heads_mixed": [0.7505230825276259, 0.9360605986852908, 0.9824451776524006, 0.9632659476016673, 0.9762177017115145, 0.9375653314293295, 0.8205702307782611, 1.0081361528288093, 0.9612295727721258, 0.939657836029402, 0.9572187550638243, 0.9857629293507899, 0.9907264352949687, 0.9511807884430297, 0.9519482557893972],
        },
        "without_command_line": {
            "no_heads": [0.931768752711591, 0.9916714543730648, 0.9955288156262, 0.9958519182940311, 0.9904178679694474, 0.9894399967666271, 0.9542436617420907, 0.9946465561027785, 0.9956050119199121, 0.9916911377209953, 0.9916984290642834, 0.996324137021483, 0.9939567238948687, 0.9909581253120823, 0.9904920939988081],
            "one_head_no_dicts": [0.9765377790674641, 1.0054716763172307, 1.0140060318323534, 1.014363714901333, 1.0189184329775378, 1.015198938226285, 0.9992268425405949, 1.0218196096722967, 1.012171037832985, 1.0173505198555735, 1.0140449474551874, 1.0130127776446485, 1.0108538058363181, 1.0148440634512879, 1.015659601309868],
            "one_head_with_dicts": [0.7480831408068953, 0.951775499630758, 1.0015875095002453, 0.9819165271183335, 0.9928739014963112, 0.9614852272615066, 0.83099400509797, 1.0355993341035963, 0.9911904393853932, 0.9720007139825775, 0.9784314683102276, 1.0077504173821927, 1.009084672590903, 0.9737687751059977, 0.9694892306335727],
            "two_heads_no_dicts": [0.9842012169306136, 0.9890475366032125, 0.9934824346196575, 0.9917470396384933, 0.9950894872514671, 0.9844190209597082, 0.9839036775420837, 0.9920259110928641, 0.9923836558086693, 0.9945216107682886, 0.9919768217686048, 0.993627763929503, 0.995455114471042, 0.9945126003013522, 0.9910098234565261],
            "two_heads_mixed": [0.7438737469070258, 0.9249199356959119, 0.9911499614997534, 0.9789329546372744, 1.0145182675811162, 0.9554436622303858, 0.8501758653336061, 1.047155482913313, 0.987552573441212, 0.9849693009422372, 0.9751310293263727, 0.9939523931936435, 0.9983463511859075, 0.9780938932717245, 0.9718457913217105],
        },
    }
    # fmt: on

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
    ("yaml_contents", "name", "expected_value"), _trial_yamls_and_and_expected
)
def test_key_specification_methods(tmp_path, yaml_contents, name, expected_value):
    fitting_configs = configs_numbered_keys()

    ase.io.write(tmp_path / "fit_multihead_dft.xyz", fitting_configs)
    ase.io.write(tmp_path / "fit_multihead_mp2.xyz", fitting_configs)
    ase.io.write(tmp_path / "duplicated_fit_multihead_dft.xyz", fitting_configs)

    mace_params = _mace_params.copy()
    mace_params["valid_fraction"] = 0.1
    mace_params["checkpoints_dir"] = (tmp_path).as_posix()
    mace_params["model_dir"] = (tmp_path).as_posix()
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
        mace_params["config"] = (tmp_path / "config.yaml").as_posix()

    cmd = (
        run_train
        + " "
        + " ".join(
            [
                (f"--{k}={v}" if v is not None else f"--{k}")
                for k, v in mace_params.items()
            ]
        )
    )

    p = subprocess.run(cmd.split(), cwd=tmp_path, check=True)
    assert p.returncode == 0

    if "heads" in yaml_contents:
        headname = next(iter(yaml_contents["heads"].keys()))
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

    npt.assert_allclose(np.asarray(Es), expected_value, rtol=1e-8, atol=1e-8)


def test_multihead_finetuning_does_not_modify_default_keyspec(tmp_path):
    fitting_configs = configs_numbered_keys()
    ase.io.write(tmp_path / "fit_multihead_dft.xyz", fitting_configs)

    args = build_default_arg_parser().parse_args(
        [
            "--name",
            "_MACE_",
            "--train_file",
            (tmp_path / "fit_multihead_dft.xyz").as_posix(),
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
