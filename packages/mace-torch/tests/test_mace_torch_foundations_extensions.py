"""Adding species/readouts keeps trained predictions and survives native reload."""

import json
import os
from pathlib import Path

import numpy as np
import pytest
import torch
from conftest import TOLERANCES, fp64_only
from mace_core.model_registry import get_model
from mace_core.observables import ObservableSpec
from mace_torch.backends.precision import interface_dtype
from mace_torch.calculators.foundations import load_foundation
from mace_torch.deploy.loader import load_deployed
from mace_torch.deploy.reference import _evaluate
from mace_torch.finetune.extend import NewSpeciesInit, extend_elements
from mace_torch.finetune.readouts import extend_readouts
from mace_torch.serialization import canonical_state
from test_mace_torch_extend_elements import water_foundation, waters


@pytest.fixture(scope="module")
def foundation(tmp_path_factory):
    return water_foundation(tmp_path_factory.mktemp("readouts"))


def outputs(model):
    return _evaluate(
        model.engine,
        waters()[2],
        z_table=model.z_table,
        cutoff=model.config.model.r_max,
        head=0,
    )


def assert_same_predictions(first, second, *, species=False):
    expected = outputs(second)
    for name, value in outputs(first).items():
        if species:
            # FT-3 rescales raw weights for the expanded element fan-in.
            # This preserves the function up to the existing arithmetic row.
            atol, rtol = TOLERANCES[interface_dtype(first.engine)]
            np.testing.assert_allclose(value, expected[name], atol=atol, rtol=rtol)
        else:
            np.testing.assert_array_equal(value, expected[name])


def extend_and_check(parent, tmp_path):
    spec = ObservableSpec(name="test_scalar", irreps="0e", per_atom=False, units="eV")
    path = extend_readouts(parent, [spec], tmp_path / "readout", seed=21)
    expanded = load_deployed(path)
    before, after = canonical_state(parent.engine), canonical_state(expanded.engine)
    for operator, tensors in before.items():
        for name, value in tensors.items():
            assert torch.equal(value, after[operator][name])
    assert after.keys() > before.keys()
    assert_same_predictions(parent, expanded)
    captured = []
    hook = expanded.engine.register_forward_hook(
        lambda module, args, output: captured.append(output)
    )
    values = _evaluate(
        expanded.engine,
        waters()[2],
        z_table=expanded.z_table,
        cutoff=expanded.config.model.r_max,
        head=0,
        quantities=("test_scalar",),
    )
    hook.remove()
    assert np.isfinite(values["test_scalar"]).all()
    loss = captured[0].get("test_scalar").square().sum()
    gradients = torch.autograd.grad(
        loss, tuple(expanded.engine.parameters()), allow_unused=True
    )
    assert any(
        gradient is not None
        and torch.isfinite(gradient).all()
        and torch.count_nonzero(gradient)
        for gradient in gradients
    )
    assert expanded.config.catalogue is not None
    assert expanded.config.catalogue.observable("test_scalar") == spec
    assert expanded.metadata.parents[0].metadata == parent.metadata
    # The new trainable tensors differ with seed; the trained ones never do.
    second = load_deployed(extend_readouts(parent, [spec], tmp_path / "other", seed=22))
    other = canonical_state(second.engine)
    assert any(
        not torch.equal(value, other[op][key])
        for op in after.keys() - before.keys()
        for key, value in after[op].items()
    )
    return expanded


@fp64_only
@pytest.mark.parametrize("model_dtype", ["float64", "float32"])
def test_extend_loaded_foundation_roundtrip(foundation, tmp_path, model_dtype):
    parent = load_foundation(foundation, default_dtype=model_dtype, device="cpu")
    extend_and_check(parent, tmp_path)
    added = load_deployed(
        extend_elements(
            parent,
            tmp_path / "species",
            {head: {6: -1030.0} for head in parent.heads},
            NewSpeciesInit(seed=2),
        )
    )
    assert_same_predictions(parent, added, species=True)
    assert added.config.precision == parent.config.precision


@fp64_only
def test_readout_replacement_is_refused_without_writing(foundation, tmp_path):
    parent = load_foundation(foundation, device="cpu")
    with pytest.raises(ValueError, match="must not replace"):
        extend_readouts(
            parent,
            [ObservableSpec(name="energy", irreps="0e", per_atom=False, units="eV")],
            tmp_path / "bad",
        )
    assert not (tmp_path / "bad.json").exists()


@fp64_only
@pytest.mark.network
def test_small_registry_artifact_extensions_roundtrip(tmp_path):
    mapping = os.getenv("MACE_REGISTRY_ARTIFACTS")
    entry = get_model("mace-off23-small")
    if mapping:
        reference = Path(json.loads(Path(mapping).read_text())[entry.model_id])
    elif entry.artifact_ref:
        reference = entry.model_id
    else:
        pytest.skip("native publication pending; supply local FM-2 candidate map")
    parent = load_foundation(reference, family="mace_off", device="cpu")
    assert parent.config.model == entry.hyperparameters()
    extend_and_check(parent, tmp_path)
    added = load_deployed(
        extend_elements(
            parent,
            tmp_path / "silicon",
            {head: {14: -100.0} for head in parent.heads},
            NewSpeciesInit(seed=8),
        )
    )
    assert 14 in added.z_table.zs
    assert_same_predictions(parent, added, species=True)
