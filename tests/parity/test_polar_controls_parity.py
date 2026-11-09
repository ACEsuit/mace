"""Polar controls, full multipoles and graph inputs through both model stacks."""

import argparse
import json

import numpy as np
import pytest
import torch
from ase.io import write
from mace_core.config.legacy import LegacyFlagError
from mace_core.observables import DEFAULT_CATALOGUE
from mace_torch.cli import migrate
from mace_torch.cli.run_train import parse
from mace_torch.deploy.loader import load_deployed
from mace_torch.electrostatics.reference.gto_utils import gto_basis_kspace_cutoff
from mace_torch.physics import DerivativeEngine
from mace_torch.train import (
    run_data_stage,
    run_model_stage,
    run_train_stage,
    write_model,
)

from tests.golden.harness import (
    CLOSED_FORM_FP64,
    EXACT,
    FP64_ACCELERATED_BACKEND,
    FP64_CPU_REFERENCE,
)
from tests.parity.test_polar_parity import (
    CASES,
    ENERGY,
    convert,
    legacy_batch,
    legacy_polar,
    v1_batch,
    v1_graph,
    with_inputs,
)

pytestmark = [pytest.mark.polar, pytest.mark.usefixtures("fp64", "isolated")]
DEVICES = ["cpu", pytest.param("cuda", marks=pytest.mark.gpu)]
VARIANTS = {
    "baseline": {},
    "cutoff": {"kspace_cutoff_factor": 0.65},
    "multipole_order": {"atomic_multipoles_max_l": 0},
    "multipole_width": {"atomic_multipoles_smearing_width": 0.85},
    "feature_order": {
        "field_feature_max_l": 0,
        "field_feature_norms": [2.0, 3.0],
    },
    "feature_widths": {"field_feature_widths": [0.8, 1.7]},
    "feature_norms": {"field_feature_norms": [1.1, 1.4, 0.8, 0.6]},
    "recursions": {"num_recursion_steps": 3},
    "feature_self": {"field_si": True},
    "energy_self": {"include_electrostatic_self_interaction": False},
    "local_energy": {"add_local_electron_energy": False},
    "quadrupole_correction": {"quadrupole_feature_corrections": True},
}


def close(actual, expected, row):
    assert actual.shape == expected.shape
    assert torch.isfinite(actual).all() and torch.isfinite(expected).all()
    torch.testing.assert_close(actual, expected, atol=row.atol, rtol=row.rtol)


def on_device(graph, device):
    return {key: value.to(device).clone() for key, value in graph.items()}


def structures(name):
    atoms, _, inputs = CASES[name]
    first = with_inputs(atoms, *inputs)
    second = first.copy()
    second.positions[1] += [0.09, -0.04, 0.03]
    second.info.update(charge=1.0, spin=2.0, external_field=[-0.015, 0.01, 0.025])
    return first, second


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("name", ["charged doublet in a field", "crystal", "slab"])
@pytest.mark.parametrize("variant", VARIANTS)
def test_polar_controls_and_outputs_match_legacy(device, name, variant):
    from e3nn import o3

    legacy = legacy_polar(
        hidden_irreps=o3.Irreps("4x0e + 4x1o + 4x2e"), **VARIANTS[variant]
    )
    profile, normal = CASES[name][1]
    model, config = convert(legacy, profile, normal)
    legacy.to(device)
    model.to(device)
    atoms = structures(name)
    numbers = config["atomic_numbers"]
    old = on_device(legacy_batch(legacy, *atoms), device)
    new = on_device(
        v1_batch(*(v1_graph(a, numbers, config["cutoff"]) for a in atoms)), device
    )
    old["fermi_level"] = torch.tensor([0.3, -0.7], device=device)
    new["fermi_level"] = old["fermi_level"].clone()
    old_positions = old["positions"].requires_grad_(True)
    new_positions = new["positions"].requires_grad_(True)
    expected = legacy(old, training=True, compute_force=True)
    actual = DerivativeEngine(model, ENERGY)(new, compute=("forces",), training=True)
    row = FP64_ACCELERATED_BACKEND if device == "cuda" else FP64_CPU_REFERENCE
    pairs = [
        (actual.total_energy, expected["energy"]),
        (actual.forces, expected["forces"]),
    ]
    pairs += [
        (actual.dipole, expected["dipole"]),
        (actual.node_energies, expected["node_energy"]),
    ]
    for key in (
        "interaction_energy",
        "electrostatic_energy",
        "electron_energy",
        "total_charge",
        "charges",
        "spins",
        "density_coefficients",
        "spin_density",
        "spin_charge_density",
        "fukui_functions",
        "fermi_level",
        "external_field",
    ):
        pairs.append((actual.extras[key], expected[key]))
    for left, right in pairs:
        close(left, right, row)
    extras = actual.extras
    width = (legacy.atomic_multipoles_max_l + 1) ** 2
    assert extras["spin_density"].shape == (sum(map(len, atoms)), width)
    close(
        extras["spin_density"],
        extras["spin_charge_density"][:, 0] - extras["spin_charge_density"][:, 1],
        EXACT,
    )
    close(extras["spins"], extras["spin_density"][:, 0], EXACT)
    close(extras["external_field"], new["external_field"], EXACT)
    close(extras["fermi_level"], torch.zeros(2, device=device), EXACT)
    declared = {spec.name: spec for spec in model.extra_observables}
    assert declared["spin_density"].per_atom
    assert not declared["external_field"].per_atom
    assert not declared["fermi_level"].per_atom
    close(extras["total_charge"], new["total_charge"], CLOSED_FORM_FP64)
    # Shared nonuniform cotangents include the force path's double backward.
    objectives = [0, 0]
    for left, right in pairs:
        direction = torch.linspace(0.1, 0.9, left.numel(), device=device).reshape(
            left.shape
        )
        objectives[0] += (left * direction).sum()
        objectives[1] += (right * direction).sum()
    gradients = [
        torch.autograd.grad(value, positions)[0]
        for value, positions in zip(
            objectives, (new_positions, old_positions), strict=True
        )
    ]
    close(*gradients, row)


FLAGS = {
    "kspace_cutoff_factor": 0.85,
    "atomic_multipoles_max_l": 1,
    "atomic_multipoles_smearing_width": 0.9,
    "field_feature_max_l": 1,
    "field_feature_widths": [0.8, 1.3],
    "field_feature_norms": [1.1, 1.2, 1.3, 1.4],
    "num_recursion_steps": 2,
    "field_si": True,
    "include_electrostatic_self_interaction": True,
    "add_local_electron_energy": True,
    "quadrupole_feature_corrections": True,
    "fixedpoint_update_config": {
        "type": "AgnosticEmbeddedOneBodyVariableUpdate",
        "potential_embedding_cls": "AgnosticChargeBiasedLinearPotentialEmbedding",
        "nonlinearity_cls": "MLPNonLinearity",
    },
    "field_readout_config": {"type": "OneBodyMLPFieldReadout"},
}
EXPECTED = {
    "multipole_max_l": 1,
    "multipole_width": 0.9,
    "feature_max_l": 1,
    "feature_widths": (0.8, 1.3),
    "feature_norms": (1.1, 1.2, 1.3, 1.4),
    "num_recursion_steps": 2,
    "feature_self_interaction": True,
    "energy_self_interaction": True,
    "add_local_electron_energy": True,
    "quadrupole_feature_corrections": True,
}


def config_file(tmp_path):
    frames = []
    for index in range(4):
        atoms = structures("charged doublet in a field")[index % 2]
        atoms.positions[0, 0] += 0.02 * index
        atoms.info["REF_energy"] = -13.0 + index * 0.1
        atoms.arrays["REF_forces"] = np.full((len(atoms), 3), 0.01 * (index + 1))
        frames.append(atoms)
    dataset = tmp_path / "train.xyz"
    write(dataset, frames)
    document = {
        "runtime": {"work_dir": str(tmp_path), "seed": 3},
        "data": {
            "heads": {
                "default": {
                    "train_file": str(dataset),
                    "e0s": {"kind": "table", "values": {1: -1.3, 8: -4.7}},
                }
            },
            "valid_fraction": 0.25,
        },
        "model": {
            "model": "polar",
            "num_channels": 2,
            "num_interactions": 2,
            "max_ell": 2,
            "correlation": 2,
            "hidden_irreps": "0e+1o+2e",
            "interaction_first": "plain",
            "readout": {"mlp_irreps": "4x0e"},
        },
        "electrostatics": {"enabled": True, "periodicity_profile": "per_structure"},
        "training": {
            "max_num_epochs": 1,
            "batch_size": 3,
            "valid_batch_size": 1,
            "lr": 0.0001,
        },
    }
    path = tmp_path / "config.json"
    path.write_text(json.dumps(document))
    return path


@pytest.mark.parametrize("route", ["flags", "native_json"])
def test_all_polar_controls_build_train_and_survive_reload(tmp_path, route):
    path = config_file(tmp_path)
    if route == "native_json":
        document = json.loads(path.read_text())
        document["model"]["polar"] = EXPECTED
        document["electrostatics"]["kspace_cutoff_factor"] = 0.85
        path.write_text(json.dumps(document))
        arguments = []
    else:
        arguments = [
            token
            for name, value in FLAGS.items()
            for token in ("--" + name, repr(value))
        ]
    config = parse(["--config", str(path), *arguments])
    assert config.model.polar.model_dump() == EXPECTED
    assert config.electrostatics.kspace_cutoff_factor == 0.85
    data = run_data_stage(config, DEFAULT_CATALOGUE)
    built = run_model_stage(config, data, DEFAULT_CATALOGUE)
    model = built.engine.get_submodule("model")
    assert model.config.polar.model_dump() == EXPECTED
    assert model.solver_spec.kspace_cutoff == pytest.approx(
        0.85
        * gto_basis_kspace_cutoff(
            [model.config.polar.multipole_width, *model.config.polar.feature_widths],
            max(model.config.polar.multipole_max_l, model.config.polar.feature_max_l),
        )
    )
    assert len(model.updates) == 2 and model.density_width == 4
    assert model.solver_spec.features.include_self_interaction
    assert model.solver_spec.features.quadrupole_corrections
    before = {
        name: parameter.detach().clone()
        for name, parameter in built.engine.named_parameters()
    }
    run_train_stage(config, built, checkpoint_path=tmp_path / "polar")
    assert any(
        not torch.equal(before[name], parameter)
        for name, parameter in built.engine.named_parameters()
    )
    artifact = write_model(tmp_path / "checked", built.engine, built.metadata)
    loaded = load_deployed(artifact)
    assert loaded.model.config.polar == model.config.polar
    graph = next(iter(next(iter(data.valid_loaders.values())))).graph
    actual = loaded.compute(dict(graph), compute=("forces",))
    expected = built.engine(dict(graph), compute=("forces",))
    for name in ("total_energy", "forces", "dipole"):
        close(getattr(actual, name), getattr(expected, name), EXACT)
    assert actual.extras.keys() == expected.extras.keys()
    for name in actual.extras:
        close(actual.extras[name], expected.extras[name], EXACT)


@pytest.mark.parametrize(
    "flag,value,match",
    [
        ("fixedpoint_update_config", {"typo": 2}, "typo"),
        ("field_readout_config", {"typo": 2}, "typo"),
        ("fixedpoint_update_config", [], "mapping"),
        ("field_readout_config", False, "mapping"),
        ("fixedpoint_update_config", {"type": "unknown"}, "unknown"),
        ("fixedpoint_update_config", {"potential_embedding_cls": "unknown"}, "unknown"),
        ("field_readout_config", {"type": "unknown"}, "unknown"),
    ],
)
@pytest.mark.parametrize("route", ["flags", "yaml"])
def test_invalid_polar_block_settings_are_not_discarded(
    tmp_path, flag, value, match, route
):
    path = config_file(tmp_path)
    if route == "yaml":
        path = tmp_path / "legacy.yaml"
        path.write_text(
            json.dumps(
                {
                    "model": "polar",
                    "train_file": str(tmp_path / "train.xyz"),
                    flag: value,
                }
            )
        )
        args = []
    else:
        args = ["--" + flag, repr(value)]
    with pytest.raises(LegacyFlagError, match=match):
        if route == "yaml":
            migrate.run(
                argparse.Namespace(command=[str(path)], output=tmp_path / "v1.yaml")
            )
        else:
            parse(["--config", str(path), *args])


def test_the_legacy_nonlinearity_selector_is_inert(tmp_path):
    models = [
        legacy_polar(
            fixedpoint_update_config={
                **FLAGS["fixedpoint_update_config"],
                "nonlinearity_cls": name,
            }
        )
        for name in ("NoNonLinearity", "MLPNonLinearity")
    ]
    assert models[0].state_dict().keys() == models[1].state_dict().keys()
    for key, value in models[0].state_dict().items():
        close(value, models[1].state_dict()[key], EXACT)
    outputs = [
        model(
            legacy_batch(model, *structures("charged doublet in a field")),
            compute_force=True,
        )
        for model in models
    ]
    for key, value in outputs[0].items():
        if isinstance(value, torch.Tensor):
            close(value, outputs[1][key], EXACT)
    path = config_file(tmp_path)
    configs = [
        parse(
            [
                "--config",
                str(path),
                "--fixedpoint_update_config",
                repr({**FLAGS["fixedpoint_update_config"], "nonlinearity_cls": name}),
            ]
        )
        for name in ("NoNonLinearity", "MLPNonLinearity")
    ]
    assert configs[0] == configs[1]


def test_legacy_yaml_translates_all_polar_controls(tmp_path):
    path = tmp_path / "legacy.yaml"
    path.write_text(json.dumps({"model": "polar", "train_file": "unused.xyz", **FLAGS}))
    converted = tmp_path / "v1.yaml"
    migrate.run(argparse.Namespace(command=[str(path)], output=converted))
    config = parse(["--config", str(converted)])
    assert config.model.polar.model_dump() == EXPECTED
    assert config.electrostatics.kspace_cutoff_factor == 0.85
