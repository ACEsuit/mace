import ase.io
import numpy as np
import pytest
import torch

from mace.calculators import MACECalculator
from tests.helpers import CUET_AVAILABLE, base_mace_params, run_mace_train  # noqa: F401  # pylint: disable=unused-import

pytestmark = [pytest.mark.network]

device = "cuda" if torch.cuda.is_available() else "cpu"
# device = "cpu"

# fitting_configs / pretraining_configs fixtures come from tests/conftest.py

# same as the canonical params except: fewer epochs, GPU if available, and
# no explicit use_reduced_cg flag (this file never passed it)
_mace_params = base_mace_params()
_mace_params["max_num_epochs"] = 2
_mace_params["device"] = device
del _mace_params["use_reduced_cg"]


def test_run_train_freeze(tmp_path, fitting_configs):
    ase.io.write(tmp_path / "fit.xyz", fitting_configs)

    mace_params = _mace_params.copy()
    mace_params["checkpoints_dir"] = str(tmp_path)
    mace_params["model_dir"] = str(tmp_path)
    mace_params["train_file"] = tmp_path / "fit.xyz"
    mace_params["loss"] = "weighted"
    mace_params["foundation_model"] = "small"
    mace_params["hidden_irreps"] = "128x0e"
    mace_params["r_max"] = 6.0
    mace_params["default_dtype"] = "float64"
    mace_params["num_radial_basis"] = 10
    mace_params["interaction_first"] = "RealAgnosticResidualInteractionBlock"
    mace_params["multiheads_finetuning"] = False
    mace_params["freeze"] = 6

    p = run_mace_train(mace_params)
    assert p.returncode == 0

    calc = MACECalculator(
        model_paths=tmp_path / "MACE.model", device=device, default_dtype="float64"
    )

    Es = []
    for at in fitting_configs:
        at.calc = calc
        Es.append(at.get_potential_energy())

    print("Es", Es)

    ref_Es = [
        5.348334089807952,
        2.4128907878403982,
        8.5566950528953,
        7.743803832228654,
        5.788643738738498,
        9.103127501095454,
        8.719323994063377,
        8.169843256425096,
        8.077166786336269,
        8.679676296893602,
        12.189297325152948,
        6.911712148654615,
        8.290506707079263,
        5.303821445834231,
        7.296761518032694,
        5.946962420990914,
        9.043336244248948,
        7.446979685692335,
        5.764245581904601,
        6.975111618768769,
        6.931624082425803,
        6.72206658924676,
    ]

    assert np.allclose(Es, ref_Es)


def test_run_train_soft_freeze(tmp_path, fitting_configs):
    ase.io.write(tmp_path / "fit.xyz", fitting_configs)

    mace_params = _mace_params.copy()
    mace_params["checkpoints_dir"] = str(tmp_path)
    mace_params["model_dir"] = str(tmp_path)
    mace_params["train_file"] = tmp_path / "fit.xyz"
    mace_params["loss"] = "weighted"
    mace_params["foundation_model"] = "small"
    mace_params["hidden_irreps"] = "128x0e"
    mace_params["r_max"] = 6.0
    mace_params["default_dtype"] = "float64"
    mace_params["num_radial_basis"] = 10
    mace_params["interaction_first"] = "RealAgnosticResidualInteractionBlock"
    mace_params["multiheads_finetuning"] = False
    mace_params["lr_params_factors"] = (
        '{"embedding_lr_factor": 0.0, "interactions_lr_factor": 1.0, "products_lr_factor": 1.0, "readouts_lr_factor": 1.0}'
    )

    p = run_mace_train(mace_params)
    assert p.returncode == 0

    calc = MACECalculator(
        model_paths=tmp_path / "MACE.model", device=device, default_dtype="float64"
    )

    Es = []
    for at in fitting_configs:
        at.calc = calc
        Es.append(at.get_potential_energy())

    print("Es", Es)

    ref_Es = [
        4.077101520328611,
        1.9125514950167353,
        4.6390361860381795,
        4.6415570296531214,
        3.9153698530138845,
        4.487578378535444,
        4.439674506695098,
        4.906251552572849,
        4.6943771636613985,
        4.443480673870315,
        12.392544826986759,
        4.8014551746345475,
        4.6380462142293455,
        4.126315015844008,
        4.923222049125721,
        4.442558518514199,
        4.556565520687697,
        4.935513763430022,
        4.077869607943539,
        4.4407761603911124,
        5.10253699303561,
        4.537672050884654,
    ]

    assert np.allclose(Es, ref_Es)
