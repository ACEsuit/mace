"""Explicit output requests, calculator inputs and backend selection boundaries."""

from __future__ import annotations

import json
from typing import Any

import numpy as np
import pytest
import torch
from ase import Atoms
from ase.io import read, write
from conftest import fp64_only
from mace_torch.calculators import MACECalculator
from mace_torch.cli.run_train import parse, train
from mace_torch.deploy.loader import load_deployed
from test_mace_torch_full_batch import configuration

from tests.golden.harness import FP64_CPU_REFERENCE

FLAGS = {
    "compute_forces": "forces",
    "compute_stress": "stress",
    "compute_atomic_dipole": "dipole",
    "compute_polarizability": "polarizability",
}


@pytest.mark.parametrize("flag,observable", FLAGS.items())
@pytest.mark.parametrize("enabled", [False, True])
def test_output_flags_change_only_the_named_observable(
    tmp_path, flag, observable, enabled
):
    initial = ["energy", "forces", "stress", "dipole", "polarizability"]
    if enabled:
        initial.remove(observable)
    path = tmp_path / "config.json"
    path.write_text(json.dumps({"model": {"observables": initial}}))
    actual = parse(["--config", str(path), "--" + flag, str(enabled)])
    expected = (
        [*initial, observable] if enabled else [n for n in initial if n != observable]
    )
    assert actual.model.observables == tuple(expected)


@pytest.mark.parametrize("flag,observable", FLAGS.items())
def test_enabled_output_keeps_the_native_default_forces(flag, observable):
    actual = parse(["--" + flag, "True"])
    assert actual.model.observables == tuple(
        dict.fromkeys(["energy", "forces", observable])
    )


@fp64_only
@pytest.mark.parametrize("flag,observable", FLAGS.items())
@pytest.mark.parametrize("enabled", [False, True])
def test_output_flag_trains_and_reloads_the_same_model_as_native_config(
    tmp_path, flag, observable, enabled
):
    base = configuration(tmp_path, max_num_epochs=1)
    source = base.data.heads["default"].train_file
    frames = read(source, index=":")
    for atoms in frames:
        atoms.info["dipole"] = [0.1, -0.2, 0.3]
        atoms.info["polarizability"] = np.eye(3) * 0.2
        atoms.info["REF_stress"] = np.zeros(6)
        atoms.set_cell(np.eye(3) * 8.0)
        atoms.pbc = True
    write(source, frames)
    all_outputs = (
        ["dipole", "polarizability"]
        if observable == "polarizability"
        else list(dict.fromkeys(["energy", "forces", observable]))
    )
    outputs = (
        all_outputs if enabled else [name for name in all_outputs if name != observable]
    )
    initial = (
        [name for name in all_outputs if name != observable] if enabled else all_outputs
    )
    checkpoints = []
    for route in ("native", "flag"):
        doc = base.to_resolved_dict()
        doc["runtime"]["work_dir"] = str(tmp_path / route)
        doc["runtime"].pop("error_table")
        doc["model"].update(max_ell=2, hidden_irreps="0e+1o+2e")
        if observable == "polarizability":
            doc["model"]["model"] = "dielectric"
        doc["loss"]["weights"] = dict.fromkeys(outputs, 1.0)
        doc["model"]["observables"] = outputs if route == "native" else initial
        path = tmp_path / f"{route}.json"
        path.write_text(json.dumps(doc))
        args = ["--config", str(path)]
        if route == "flag":
            args += ["--" + flag, str(enabled)]
        assert train(parse(args)) == 0
        checkpoints.append(load_deployed(tmp_path / route / "mace.json"))
    for name, value in checkpoints[0].engine.state_dict().items():
        torch.testing.assert_close(
            checkpoints[1].engine.state_dict()[name], value, atol=0, rtol=0
        )
    for model in checkpoints:
        assert model.config.model.observables == tuple(outputs)
    calculators = [MACECalculator(models=model) for model in checkpoints]
    assert isinstance(frames[-1], Atoms)
    for calculator in calculators:
        calculator.calculate(frames[-1])
    for name in outputs:
        assert np.isfinite(calculators[0].results[name]).all()
        np.testing.assert_array_equal(
            calculators[0].results[name], calculators[1].results[name]
        )


@pytest.mark.parametrize("flag", ["enable_cueq", "enable_oeq"])
def test_calculator_backend_flags_cannot_be_silently_ignored(flag):
    options: dict[str, Any] = {flag: True}
    with pytest.raises(TypeError, match=flag + ".*model.backend"):
        MACECalculator(model_paths="missing.json", **options)


@fp64_only
@pytest.mark.parametrize("padding", [False, True])
def test_array_key_mapping_changes_predictions_and_invalidates_cache(tmp_path, padding):
    from mace_core.config.resolved import ResolvedConfig
    from mace_core.observables import DEFAULT_CATALOGUE
    from mace_torch.train import run_data_stage, run_model_stage, write_model

    base = configuration(tmp_path, max_num_epochs=0)
    document = base.to_resolved_dict()
    document["model"]["graph_features"] = [
        {"name": "site_value", "kind": "continuous", "per": "atom", "embedding_dim": 2}
    ]
    document["data"]["embedding_specs"] = {
        "site_value": {"per": "atom", "key": "training_value"}
    }
    frames = read(base.data.heads["default"].train_file, index=":")
    for atoms in frames:
        atoms.arrays["training_value"] = np.arange(len(atoms), dtype=float) + 1
    write(base.data.heads["default"].train_file, frames)
    config = ResolvedConfig.from_dict(document)
    data = run_data_stage(config, DEFAULT_CATALOGUE)
    built = run_model_stage(config, data, DEFAULT_CATALOGUE)
    path = write_model(tmp_path / "conditioned", built.model, built.metadata)
    calculator = MACECalculator(
        path,
        arrays_keys={"site_value": "evaluation_value"},
        pad_num_atoms=16 if padding else 0,
        pad_num_edges=2048 if padding else 0,
    )
    assert isinstance(frames[-1], Atoms)
    atoms = frames[-1].copy()
    atoms.arrays["evaluation_value"] = np.arange(len(atoms), dtype=float) + 3
    atoms.calc = calculator
    first = atoms.get_potential_energy()
    assert calculator.check_state(atoms) == []
    atoms.arrays["evaluation_value"][0] += 2
    assert calculator.check_state(atoms)
    second = atoms.get_potential_energy()
    assert first != second
    direct = atoms.copy()
    direct.arrays["training_value"] = atoms.arrays["evaluation_value"].copy()
    direct.calc = MACECalculator(path)
    np.testing.assert_allclose(
        second,
        direct.get_potential_energy(),
        atol=FP64_CPU_REFERENCE.atol,
        rtol=FP64_CPU_REFERENCE.rtol,
    )
    np.testing.assert_allclose(
        atoms.get_forces(),
        direct.get_forces(),
        atol=FP64_CPU_REFERENCE.atol,
        rtol=FP64_CPU_REFERENCE.rtol,
    )
