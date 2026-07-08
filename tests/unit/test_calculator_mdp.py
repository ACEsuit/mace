"""MACE-MDP calculator tests (from PR #1439), relocated to unit/: they build
models locally and monkeypatch downloads, so they need neither network nor
optional dependencies.
"""

from pathlib import Path

import numpy as np
import pytest
import torch
import torch.nn.functional as F
from ase import build
from e3nn import o3

import mace.calculators.foundations_models as foundations_models
from mace import modules, tools
from mace.calculators import mace_mdp
from mace.calculators.mace import MACECalculator
from mace.modules.models import AtomicDielectricMACE
from mace.tools import AtomicNumberTable
from mace.tools.scripts_utils import extract_config_mace_model


def test_mace_mdp_local_model(tmp_path):
    table = tools.AtomicNumberTable([1, 6])
    model = modules.AtomicDielectricMACE(
        r_max=5.0,
        num_bessel=4,
        num_polynomial_cutoff=3,
        max_ell=2,
        interaction_cls=modules.interaction_classes[
            "RealAgnosticResidualInteractionBlock"
        ],
        interaction_cls_first=modules.interaction_classes[
            "RealAgnosticResidualInteractionBlock"
        ],
        num_interactions=2,
        num_elements=2,
        hidden_irreps=o3.Irreps("4x0e + 4x1o + 4x2e"),
        MLP_irreps=o3.Irreps("4x0e + 4x1o + 4x2e"),
        gate=F.silu,
        atomic_energies=None,
        avg_num_neighbors=2.0,
        atomic_numbers=table.zs,
        correlation=3,
        radial_type="bessel",
    )
    model_path = tmp_path / "mace_mdp_test.model"
    torch.save(model, model_path)

    with pytest.warns(
        UserWarning,
        match="The MACE-MDP model is designed for predicting dipoles and polarizabilities of organic systems only",
    ):
        calc = mace_mdp(model=model_path, device="cpu", default_dtype="float64")

    assert isinstance(calc, MACECalculator)
    assert calc.model_type == "DipolePolarizabilityMACE"
    assert len(calc.models) == 1
    assert isinstance(calc.models[0], AtomicDielectricMACE)

    atoms = build.molecule("CH4")
    mu = calc.get_property("dipole", atoms)
    alpha = calc.get_property("polarizability", atoms)

    assert mu is not None
    assert alpha is not None
    assert np.asarray(mu).shape == (3,)
    assert np.asarray(alpha).shape == (3, 3)




def test_mace_mdp_download_uses_raw_githubusercontent_urls(tmp_path, monkeypatch):
    downloaded = {}

    def fake_download(url, filename, timeout=120):
        downloaded["url"] = url
        Path(filename).write_bytes(b"fake")
        return filename, {}

    class DummyCalculator:
        def __init__(self, model_paths, device, default_dtype, model_type, **kwargs):
            self.model_paths = model_paths
            self.device = device
            self.default_dtype = default_dtype
            self.model_type = model_type
            self.kwargs = kwargs

    monkeypatch.setattr(foundations_models, "get_cache_dir", lambda: str(tmp_path))
    monkeypatch.setattr(foundations_models, "_urlretrieve_with_timeout", fake_download)
    monkeypatch.setattr(foundations_models, "MACECalculator", DummyCalculator)

    with pytest.warns(
        UserWarning,
        match="The MACE-MDP model is designed for predicting dipoles and polarizabilities of organic systems only",
    ):
        calc = foundations_models.mace_mdp(device="cpu")

    expected_url = foundations_models.mace_mdp_default_url
    expected_path = tmp_path / Path(expected_url).name
    assert downloaded["url"] == expected_url
    assert downloaded["url"].startswith("https://raw.githubusercontent.com/")
    assert calc.model_paths == str(expected_path)
    assert calc.model_type == "DipolePolarizabilityMACE"
    assert expected_path.exists()




def test_extract_config_mace_mdp_local_model(tmp_path):
    table = AtomicNumberTable([1, 6])
    model = modules.AtomicDielectricMACE(
        r_max=5.0,
        num_bessel=4,
        num_polynomial_cutoff=3,
        max_ell=2,
        interaction_cls=modules.interaction_classes[
            "RealAgnosticResidualInteractionBlock"
        ],
        interaction_cls_first=modules.interaction_classes[
            "RealAgnosticResidualInteractionBlock"
        ],
        num_interactions=2,
        num_elements=len(table),
        hidden_irreps=o3.Irreps("4x0e + 4x1o + 4x2e"),
        MLP_irreps=o3.Irreps("4x0e + 4x1o + 4x2e"),
        gate=torch.nn.functional.silu,
        atomic_energies=None,
        avg_num_neighbors=2.0,
        atomic_numbers=table.zs,
        correlation=3,
        radial_type="bessel",
    )
    model_path = tmp_path / "mace_mdp_test.model"
    torch.save(model, model_path)

    with pytest.warns(
        UserWarning,
        match="The MACE-MDP model is designed for predicting dipoles and polarizabilities of organic systems only",
    ):
        loaded_model = mace_mdp(
            model=model_path,
            device="cpu",
            default_dtype="float64",
            return_raw_model=True,
        )

    assert isinstance(loaded_model, modules.AtomicDielectricMACE)

    config = extract_config_mace_model(loaded_model)
    model_copy = modules.AtomicDielectricMACE(**config)
    model_copy.load_state_dict(loaded_model.state_dict())

    assert isinstance(model_copy, modules.AtomicDielectricMACE)
    assert torch.equal(model_copy.atomic_numbers, loaded_model.atomic_numbers)
