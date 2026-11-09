"""Physical input declarations agree before and after model construction."""

import pytest
from mace_core.config.data import DataConfig
from mace_core.config.model import ModelConfig
from mace_torch.data.inputs import graph_inputs_of, key_specification
from mace_torch.models.registry import model_inputs
from test_mace_torch_joint_dipole import build


@pytest.mark.parametrize(
    ("family", "expected"),
    [
        ("scale_shift", ()),
        ("plain", ()),
        ("dipole", ("charges",)),
        ("dielectric", ("total_charge",)),
        ("polar", ("total_charge", "total_spin", "external_field")),
        ("magnetic", ("magmom",)),
        ("les", ("external_field",)),
        ("lc_mace", ("external_field",)),
        ("lsc_mace", ("external_field",)),
        ("qeq", ("total_charge", "external_field")),
        ("fixed_point_scf", ("total_charge", "external_field")),
    ],
)
def test_every_family_declares_its_physical_inputs(family, expected):
    settings = {family: {}} if family in {"qeq", "fixed_point_scf"} else {}
    config = ModelConfig(model=family, **settings)
    assert graph_inputs_of(config) == expected
    keys = key_specification(DataConfig(), model=config)
    for spec in model_inputs(config):
        assert spec.name in (keys.atom_keys if spec.per_atom else keys.graph_keys)


def test_joint_dipole_declares_the_fixed_charges_to_all_consumers():
    engine, config = build()
    inputs = engine.backbone.input_specs
    assert inputs == model_inputs(config.model)
    assert [(spec.name, spec.per_atom, spec.units) for spec in inputs] == [
        ("charges", True, "e")
    ]


def test_embedded_physical_input_keeps_its_units_and_single_declaration():
    config = ModelConfig(
        model="polar",
        graph_features=[
            {
                "name": "total_charge",
                "input_dim": 1,
                "per": "graph",
                "kind": "continuous",
                "embedding_dim": 2,
            }
        ],
    )
    inputs = model_inputs(config)
    charges = [spec for spec in inputs if spec.name == "total_charge"]
    assert len(charges) == 1 and charges[0].units == "e"
