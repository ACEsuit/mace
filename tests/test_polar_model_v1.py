"""Tests for the v1 PolarModel registry/backend/extras API."""

import numpy as np
import pytest
import torch
from e3nn import o3

from mace import data
from mace.calculators.mace import MACECalculator
from mace.modules import interaction_classes
from mace.modules import models as models_module
from mace.modules.models import (
    PolarModel,
    ReferenceElectrostaticsBackend,
    build_electrostatics_backend,
    get_electrostatics_backend,
    get_model_class,
    list_models,
    register_electrostatics_backend,
    register_model,
)
from mace.tools import torch_geometric, utils

POLAR_FLAGS = (
    "kspace_cutoff_factor",
    "atomic_multipoles_max_l",
    "atomic_multipoles_smearing_width",
    "field_feature_max_l",
    "field_feature_widths",
    "num_recursion_steps",
    "field_si",
    "include_electrostatic_self_interaction",
    "add_local_electron_energy",
    "quadrupole_feature_corrections",
    "return_electrostatic_potentials",
    "field_feature_norms",
    "field_norm_factor",
    "fixedpoint_update_config",
    "field_readout_config",
)

POLAR_VALUES = {
    "kspace_cutoff_factor": 1.25,
    "atomic_multipoles_max_l": 1,
    "atomic_multipoles_smearing_width": 0.75,
    "field_feature_max_l": 2,
    "field_feature_widths": [1.0, 2.0],
    "num_recursion_steps": 3,
    "field_si": True,
    "include_electrostatic_self_interaction": False,
    "add_local_electron_energy": True,
    "quadrupole_feature_corrections": False,
    "return_electrostatic_potentials": True,
    "field_feature_norms": [0.5, 0.25],
    "field_norm_factor": 0.05,
    "fixedpoint_update_config": {"max_iter": 7},
    "field_readout_config": {"hidden": 8},
}


def _build_v1_model(**overrides):
    kwargs = dict(
        r_max=4.0,
        num_bessel=4,
        num_polynomial_cutoff=3,
        max_ell=1,
        interaction_cls=interaction_classes["RealAgnosticResidualInteractionBlock"],
        interaction_cls_first=interaction_classes["RealAgnosticResidualInteractionBlock"],
        num_interactions=1,
        num_elements=2,
        hidden_irreps=o3.Irreps("4x0e + 4x1o"),
        MLP_irreps=o3.Irreps("8x0e"),
        atomic_energies=np.zeros(2, dtype=np.float32),
        avg_num_neighbors=3.0,
        atomic_numbers=[1, 8],
        correlation=1,
        gate=torch.nn.functional.silu,
        radial_MLP=[16, 16],
        heads=["Default"],
        **POLAR_VALUES,
    )
    kwargs.update(overrides)
    return PolarModel(**kwargs)


def _make_atoms():
    from ase import Atoms

    return Atoms("HO", positions=[[0.0, 0.0, 0.0], [0.0, 0.0, 1.0]], cell=[10, 10, 10], pbc=False)


def _build_v1_batch(model):
    z_table = utils.AtomicNumberTable([int(z) for z in model.atomic_numbers])
    config = data.config_from_atoms(_make_atoms())
    loader = torch_geometric.dataloader.DataLoader(
        dataset=[
            data.AtomicData.from_config(
                config,
                z_table=z_table,
                cutoff=float(model.r_max),
                heads=model.heads,
            )
        ],
        batch_size=1,
        shuffle=False,
    )
    return next(iter(loader))


class CountingBackend(ReferenceElectrostaticsBackend):
    def __init__(self, config=None):
        super().__init__(config)
        self.calls = 0

    def compute(self, positions, charges, **kwargs):
        self.calls += 1
        return {
            "electrostatic_energy": torch.zeros(1, device=positions.device),
            "electrostatic_forces": torch.zeros_like(positions),
        }


@pytest.fixture
def clean_registries():
    model_registry = dict(models_module._model_registry)
    backend_registry = dict(models_module._electrostatics_backend_registry)
    yield
    models_module._model_registry.clear()
    models_module._model_registry.update(model_registry)
    models_module._electrostatics_backend_registry.clear()
    models_module._electrostatics_backend_registry.update(backend_registry)


def test_polar_model_is_registered():
    assert list_models()
    assert "PolarModel" in list_models()
    assert get_model_class("PolarModel") is PolarModel


def test_polarmace_is_denylisted():
    with pytest.raises(ValueError):
        register_model("PolarMACE", torch.nn.Module)


def test_register_model_supports_direct_decorator_and_idempotent(clean_registries):
    @register_model("TmpA")
    class A(torch.nn.Module):
        pass

    assert get_model_class("TmpA") is A
    register_model("TmpA", A)

    class B(torch.nn.Module):
        pass

    with pytest.raises(ValueError):
        register_model("TmpA", B)


def test_reference_backend_is_registered_and_buildable():
    assert get_electrostatics_backend("reference") is ReferenceElectrostaticsBackend
    backend = build_electrostatics_backend("reference", {"pbc_handling": "auto"})
    assert isinstance(backend, ReferenceElectrostaticsBackend)
    assert backend.pbc_handling == "auto"


def test_backend_registration_is_idempotent_and_conflicts(clean_registries):
    class MyBackend(ReferenceElectrostaticsBackend):
        pass

    register_electrostatics_backend("mine", MyBackend)
    register_electrostatics_backend("mine", MyBackend)
    assert get_electrostatics_backend("mine") is MyBackend

    class OtherBackend(ReferenceElectrostaticsBackend):
        pass

    with pytest.raises(ValueError):
        register_electrostatics_backend("mine", OtherBackend)


def test_custom_backend_identity_recorded_in_config(clean_registries):
    class MyBackend(ReferenceElectrostaticsBackend):
        pass

    register_electrostatics_backend("mine", MyBackend)
    model = _build_v1_model(electrostatics_backend="mine")
    assert model.config["electrostatics_backend"] == "mine"
    assert model.config["resolved"]["electrostatics_backend"] == "MyBackend"
    assert model.config["resolved"]["electrostatics_backend_name"] == "mine"
    assert isinstance(model.electrostatics_backend, MyBackend)


def test_polar_model_forces_backbone_options():
    model = _build_v1_model(use_agnostic_product=False)
    assert model.keep_last_layer_irreps is True
    assert model.use_agnostic_product is True
    assert model._permute_to_e3nn_convention is True


def test_polar_model_stores_all_fifteen_flags():
    model = _build_v1_model()
    for flag in POLAR_FLAGS:
        assert hasattr(model, flag), flag
        assert flag in model.config, flag
        assert getattr(model, flag) == POLAR_VALUES[flag], flag


def test_polar_model_forward_returns_extras():
    model = _build_v1_model()
    batch = _build_v1_batch(model)
    out = model(batch)
    assert "extras" in out
    extras = out["extras"]
    for key in (
        "interaction_energy",
        "electrostatic_energy",
        "electron_energy",
        "density_coefficients",
        "spin_charge_density",
        "fukui_functions",
        "dipole",
        "charges",
    ):
        assert key in extras, key


def test_polar_model_unrolls_fixed_number_of_recursion_steps(clean_registries):
    register_electrostatics_backend("counting", CountingBackend)
    model = _build_v1_model(electrostatics_backend="counting", num_recursion_steps=4)
    batch = _build_v1_batch(model)
    model(batch)
    assert model.electrostatics_backend.calls == 4


def test_legacy_polarmace_is_distinct_from_polar_model():
    from mace.modules.extensions import PolarMACE

    assert PolarMACE is not PolarModel
    assert "PolarMACE" not in list_models()


def test_calculator_merges_polar_extras_into_results():
    model = _build_v1_model(return_electrostatic_potentials=True)
    atoms = _make_atoms()
    calc = MACECalculator(models=model, model_type="PolarModel", device="cpu")
    atoms.calc = calc
    atoms.get_potential_energy()
    for key in (
        "interaction_energy",
        "electrostatic_energy",
        "electron_energy",
        "density_coefficients",
        "spin_charge_density",
        "fukui_functions",
        "dipole",
    ):
        assert key in calc.results, key
