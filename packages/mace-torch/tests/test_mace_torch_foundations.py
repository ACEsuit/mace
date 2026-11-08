"""Native loader behavior and the released D3 keyword contract."""

import sys
import types

import pytest
import torch
from ase import units
from conftest import fp64_only
from mace_core.config.precision import PrecisionConfig
from mace_torch.calculators import foundations as fm
from mace_torch.deploy.loader import load_deployed
from mace_torch.serialization import canonical_state
from test_mace_torch_extend_elements import water_foundation


@pytest.fixture(scope="module")
def foundation(tmp_path_factory):
    return water_foundation(tmp_path_factory.mktemp("foundations"))


@fp64_only
@pytest.mark.parametrize(
    "loader,expected", [(fm.mace_mp, torch.float32), (fm.mace_off, torch.float64)]
)
def test_local_loader_uses_its_family_precision_and_raw_model(
    foundation, loader, expected
):
    calc = loader(foundation, device="cpu")
    assert isinstance(calc, fm.MACECalculator)
    raw = loader(foundation, device="", return_raw_model=True)
    assert isinstance(raw, torch.nn.Module)
    from mace_torch.backends.precision import interface_dtype

    assert interface_dtype(calc.models[0].engine) == expected
    assert interface_dtype(raw) == expected
    converted = loader(foundation, device="cpu", default_dtype="float64")
    reference = load_deployed(foundation)
    for op, tensors in canonical_state(reference.engine).items():
        for key, value in tensors.items():
            assert torch.equal(
                canonical_state(converted.models[0].engine)[op][key], value
            )


@fp64_only
@pytest.mark.parametrize("raw", [False, True])
@pytest.mark.parametrize("policy", [PrecisionConfig.float64(), PrecisionConfig.mixed()])
def test_explicit_precision_reads_weights_directly_from_the_checkpoint(
    foundation, raw, policy
):
    result = fm.mace_mp(
        foundation, device="cpu", return_raw_model=raw, precision=policy
    )
    if isinstance(result, fm.MACECalculator):
        deployed = result.models[0]
        assert deployed.config.precision == policy
        assert (
            deployed.metadata.config.resolved["precision"]
            == deployed.config.to_resolved_dict()["precision"]
        )
        result = deployed.engine
    assert isinstance(result, torch.nn.Module)
    actual = canonical_state(result)
    expected = canonical_state(load_deployed(foundation, precision=policy).engine)
    for op, tensors in expected.items():
        for key, value in tensors.items():
            torch.testing.assert_close(actual[op][key], value, atol=0, rtol=0)


@fp64_only
@pytest.mark.parametrize(
    "loader", [fm.mace_mp, fm.mace_off, fm.mace_omol, fm.mace_mdp, fm.mace_polar]
)
def test_every_loader_refuses_pickle_before_reading(loader, tmp_path):
    path = tmp_path / "old.model"
    path.write_bytes(b"not a native checkpoint")
    with pytest.raises(ValueError, match="mace model convert"):
        loader(path, device="cpu")


@fp64_only
def test_mdp_and_omol_enforce_their_boundary_contracts(monkeypatch):
    def refuse(*args, **kwargs):
        raise AssertionError("no model should load")

    monkeypatch.setattr(fm, "load_foundation", refuse)
    with pytest.raises(ValueError, match="model_type"):
        fm.mace_mdp(model_type="energy")
    with pytest.raises(ValueError, match="omol"):
        fm.mace_omol(head="wrong")
    with pytest.raises(ValueError, match="explicit model"):
        fm.mace_polar(None)  # ty: ignore[invalid-argument-type]


@pytest.fixture
def d3_kwargs(monkeypatch):
    recorded = {}

    class FakeTorchDFTD3Calculator:
        def __init__(self, **kwargs):
            recorded.update(kwargs)

    module = types.ModuleType("torch_dftd.torch_dftd3_calculator")
    module.__dict__["TorchDFTD3Calculator"] = FakeTorchDFTD3Calculator
    monkeypatch.setitem(sys.modules, "torch_dftd", types.ModuleType("torch_dftd"))
    monkeypatch.setitem(sys.modules, "torch_dftd.torch_dftd3_calculator", module)
    monkeypatch.setattr(fm, "_calculator", lambda *args, **kwargs: object())
    monkeypatch.setattr(fm, "SumCalculator", lambda calcs: calcs)
    return recorded


@fp64_only
@pytest.mark.parametrize(
    "arguments,key,expected",
    [
        ({"dispersion_damping": "zerom"}, "damping", "zerom"),
        ({"damping": "zero"}, "damping", "zero"),
        ({}, "damping", "bj"),
        ({"dispersion_xc": "revpbe"}, "xc", "revpbe"),
        ({}, "xc", "pbe"),
        ({"dispersion_cutoff": 25.0}, "cutoff", 25.0),
        ({}, "cutoff", 40.0 * units.Bohr),
        ({"dispersion_cutoff": 30.0, "dispersion_coord_cutoff": 12.0}, "cnthr", 12.0),
    ],
)
def test_released_dispersion_arguments(d3_kwargs, arguments, key, expected):
    fm.mace_mp(model="small", device="cpu", dispersion=True, **arguments)
    assert d3_kwargs[key] == expected


@fp64_only
def test_dispersion_options_do_not_reach_the_model_calculator(monkeypatch, d3_kwargs):
    seen = {}

    def calculator(family, model, device, dtype, raw, kwargs):
        seen.update(kwargs)
        return object()

    monkeypatch.setattr(fm, "_calculator", calculator)
    fm.mace_mp(
        model="small",
        device="cpu",
        dispersion=True,
        dispersion_xc="revpbe",
        dispersion_cutoff=25,
        damping="zero",
    )
    assert d3_kwargs["damping"] == "zero"
    assert not any(key.startswith("dispersion") or key == "damping" for key in seen)


@fp64_only
def test_dispersion_uses_the_requested_interface_dtype(d3_kwargs):
    fm.mace_mp(
        "small", device="cpu", dispersion=True, precision=PrecisionConfig.float64()
    )
    assert d3_kwargs["dtype"] == torch.float64
    assert "precision" not in d3_kwargs


@fp64_only
@pytest.mark.parametrize("license_id", ["MIT", "ASL"])
def test_named_default_downloads_the_pair_and_reuses_cache(
    foundation, tmp_path, monkeypatch, caplog, capsys, license_id
):
    import io
    import json
    import logging

    from mace_core import artifacts, model_registry
    from mace_core.artifacts import ArtifactRef
    from mace_core.metadata import HeadSummary, ModelMetadata

    metadata = ModelMetadata.from_json(
        json.dumps(json.loads(foundation.read_text())["config"])
    )
    metadata.heads = {head: HeadSummary() for head in metadata.heads}
    template = model_registry.get_model("mace-mp-0-small")
    entry = template.model_copy(
        update={
            "metadata": metadata,
            "default": True,
            "license": template.license.model_copy(update={"identifier": license_id}),
            "artifact_ref": ArtifactRef(
                url="https://models.example/native.json?release=1"
            ),
        }
    )
    monkeypatch.setattr(model_registry, "_entries", lambda: (entry,))
    monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path / "cache"))
    requests = []

    def download(url, **kwargs):
        requests.append(url)
        data = (
            foundation if ".json?" in url else foundation.with_suffix(".safetensors")
        ).read_bytes()
        result = io.BytesIO(data)
        result.__dict__["headers"] = {"Content-Length": str(len(data))}
        result.__dict__["geturl"] = lambda: url
        return result

    monkeypatch.setattr(artifacts.urllib.request, "urlopen", download)
    with caplog.at_level(logging.INFO):
        first = fm.mace_mp(device="cpu", default_dtype="float64")
    assert isinstance(first, fm.MACECalculator)
    assert entry.model_id in caplog.text
    assert len(requests) == 2
    with pytest.warns(RuntimeWarning, match=entry.model_id):
        second = fm.mace_mp("small", device="cpu", default_dtype="float64")
    assert len(requests) == 2
    assert isinstance(second, fm.MACECalculator)
    assert capsys.readouterr().out.count(model_registry.ASL_NOTICE) == (
        2 if license_id == "ASL" else 0
    )
    for op, tensors in canonical_state(first.models[0].engine).items():
        for name, value in tensors.items():
            assert torch.equal(
                value, canonical_state(second.models[0].engine)[op][name]
            )


@fp64_only
@pytest.mark.parametrize(
    "version,accepted",
    [("1.0", True), ("1.1", False), ("1.2", False), ("1.3", False), ("2.0", False)],
)
def test_loader_checks_metadata_version_before_model_construction(
    foundation, tmp_path, version, accepted
):
    import json

    from mace_core.metadata import SCHEMA_VERSION

    metadata = json.loads(foundation.read_text())
    metadata["config"]["schema_version"] = version
    path = tmp_path / "versioned.json"
    path.write_text(json.dumps(metadata))
    path.with_suffix(".safetensors").symlink_to(foundation.with_suffix(".safetensors"))
    if accepted:
        assert isinstance(fm.mace_mp(path, device="cpu"), fm.MACECalculator)
    else:
        with pytest.raises(ValueError) as error:
            fm.mace_mp(path, device="cpu")
        assert version in str(error.value) and SCHEMA_VERSION in str(error.value)
