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

    # Regenerated after the finetuning loader stopped promoting the foundation
    # model's bessel_weights buffer to a trainable parameter: its (never
    # applied) gradient previously inflated the global gradient-clipping norm
    # and thereby rescaled all real updates.
    ref_Es = [
        5.345878155766861,
        2.411497782262181,
        8.550769991387254,
        7.738357505975316,
        5.78692808954139,
        9.095810905419432,
        8.712064778178565,
        8.174232379109277,
        8.071851652669764,
        8.672488636685976,
        12.194668758226612,
        6.908159446354922,
        8.284846585075943,
        5.3022876912879795,
        7.293411605988036,
        5.9452096934658565,
        9.036063031815054,
        7.443092610445011,
        5.762661936284168,
        6.97092296392942,
        6.928275856019228,
        6.719897367367488,
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

    # Regenerated after the finetuning loader stopped promoting the foundation
    # model's bessel_weights buffer to a trainable parameter: its (never
    # applied) gradient previously inflated the global gradient-clipping norm
    # and thereby rescaled all real updates.
    ref_Es = [
        4.072541620349025,
        1.9092678593639831,
        4.63517891170805,
        4.63855251203928,
        3.926328353729092,
        4.485477647143206,
        4.438278362095131,
        4.975977425662781,
        4.690904320412827,
        4.442832738970156,
        12.430264396148296,
        4.796766710009153,
        4.634183194000717,
        4.119115519077613,
        4.916594304626028,
        4.434547650005907,
        4.553428360835846,
        4.9286736924280365,
        4.068148288356933,
        4.440014853872592,
        5.0961304885631655,
        4.529497272833853,
    ]

    assert np.allclose(Es, ref_Es)
