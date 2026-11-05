"""Loud load refusals, named path alignment and the framework boundary."""

import io
import json
import subprocess
import sys
import urllib.request

import jax
import numpy as np
import pytest
from jax_conformance import TINY, ModelSurface
from mace_core.artifacts import cache_name
from mace_core.kernels import available_backends
from mace_jax.deploy.neutral_io import ModelLoadError, load_model
from safetensors.numpy import load_file, save_file

from tests.golden import harness


@pytest.mark.parametrize("suffix", [".json", ".safetensors"])
def test_downloaded_checkpoint_pair_reloads_offline_with_golden_parity(
    tmp_path, monkeypatch, suffix
):
    """Both cached files keep the artifact bytes and the frozen predictions."""
    monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path))
    root = "https://example.org/tiny"
    bodies = {
        root + ext: TINY.with_suffix(ext).read_bytes()
        for ext in (".json", ".safetensors")
    }
    requested = []

    class Response(io.BytesIO):
        def __init__(self, body):
            super().__init__(body)
            self.headers = {
                "Content-Length": str(len(body)),
                "Content-Type": "application/octet-stream",
            }

    def opener(url, timeout):
        requested.append(url)
        return Response(bodies[url])

    monkeypatch.setattr(urllib.request, "urlopen", opener)
    apply, params = load_model(root + suffix)
    assert requested == list(bodies)
    for url, body in bodies.items():
        assert (tmp_path / "mace" / cache_name(url)).read_bytes() == body

    def refuse(*args, **kwargs):
        raise AssertionError("the cached checkpoint tried to use the network")

    monkeypatch.setattr(urllib.request, "urlopen", refuse)
    reloaded, state = load_model(root + suffix)
    reference = harness.load_reference(
        harness.REFERENCES_DIR / "tiny_scaleshift_e3nn_cpu_fp64.json"
    )
    row = harness.FP64_CPU_REFERENCE
    for name, atoms in harness.load_fixtures(
        names=["water_cluster", "triclinic_bulk"]
    ).items():
        actual = ModelSurface(apply, params).golden_outputs(atoms)
        again = ModelSurface(reloaded, state).golden_outputs(atoms)
        expected = reference["fixtures"][name]["outputs"]
        channels = (
            ("energy", "forces", "stress") if atoms.pbc.any() else ("energy", "forces")
        )
        for key in channels:
            np.testing.assert_allclose(
                actual[key], expected[key]["value"], atol=row.atol, rtol=row.rtol
            )
            np.testing.assert_array_equal(again[key], actual[key])
    assert len(list((tmp_path / "mace").iterdir())) == 2
    jax.clear_caches()


def edited(tmp_path, edit):
    document = json.loads(TINY.read_text())
    flat = load_file(str(TINY.with_suffix(".safetensors")))
    edit(document, flat)
    path = tmp_path / "edited.json"
    path.write_text(json.dumps(document))
    # safetensors' numpy writer consumes contiguous storage; a path permutation
    # creates a strided view. Preserve tensor values, not that view's byte order.
    save_file(
        {name: value.copy(order="C") for name, value in flat.items()},
        str(path.with_suffix(".safetensors")),
    )
    return path


@pytest.mark.parametrize(
    "field,value,match",
    [
        ("clebsch_gordan_basis", "full", "reduced"),
        ("observables", ["energy", "hessian"], "hessian"),
        ("observables", ["energy", "magforces"], "magforces"),
        ("interaction", "nonlinear", "nonlinear"),
        ("radial_basis", {"kind": "gaussian", "num_basis": 8}, "gaussian"),
        ("distance_transform", {"kind": "soft"}, "soft"),
        ("readout", {"gate": "tanh"}, "tanh"),
    ],
)
def test_unsupported_architectures_are_named(tmp_path, field, value, match):
    path = edited(
        tmp_path,
        lambda doc, weights: doc["config"]["config"]["resolved"]["model"].__setitem__(
            field, value
        ),
    )
    with pytest.raises(ValueError, match=match):
        load_model(path)


def test_multihead_points_to_explicit_selection(tmp_path):
    def change(doc, weights):
        heads = doc["config"]["config"]["resolved"]["data"]["heads"]
        heads["other"] = dict(next(iter(heads.values())))

    with pytest.raises(ModelLoadError, match="mace model select-head"):
        load_model(edited(tmp_path, change))


@pytest.mark.parametrize("path", ["legacy.model", "legacy.pt", "legacy.pkl"])
def test_pickle_is_refused_before_reading(path):
    with pytest.raises(ModelLoadError, match="mace model convert"):
        load_model(path)


def test_future_metadata_version_is_refused(tmp_path):
    with pytest.raises(ValueError, match="schema_version"):
        load_model(
            edited(
                tmp_path,
                lambda doc, weights: doc["config"].__setitem__(
                    "schema_version", "999.0"
                ),
            )
        )


def test_named_paths_can_be_permuted_without_changing_the_model(tmp_path):
    def permute(doc, weights):
        for name, layout in doc["layout"].items():
            if "paths" not in layout:
                continue
            order = np.random.default_rng(83).permutation(len(layout["paths"]))
            layout["paths"] = [layout["paths"][index] for index in order]
            weights[name + "::weight"] = weights[name + "::weight"][:, order, :]

    apply, params = load_model(edited(tmp_path, permute))
    actual = ModelSurface(apply, params).golden_outputs(
        harness.load_fixtures(names=["water_cluster"])["water_cluster"]
    )
    reference = harness.load_reference(
        harness.REFERENCES_DIR / "tiny_scaleshift_e3nn_cpu_fp64.json"
    )["fixtures"]["water_cluster"]["outputs"]
    row = harness.FP64_CPU_REFERENCE
    np.testing.assert_allclose(
        actual["energy"], reference["energy"]["value"], atol=row.atol, rtol=row.rtol
    )
    np.testing.assert_allclose(
        actual["forces"], reference["forces"]["value"], atol=row.atol, rtol=row.rtol
    )
    jax.clear_caches()


def test_unknown_named_path_is_not_interpreted_by_position(tmp_path):
    def break_name(doc, weights):
        layout = next(value for value in doc["layout"].values() if "paths" in value)
        layout["paths"][0] = "unknown coupling"

    with pytest.raises(ModelLoadError, match="named reduced contraction paths"):
        load_model(edited(tmp_path, break_name))


@pytest.mark.parametrize("fault", ["undeclared", "duplicate", "shape"])
def test_tensor_manifest_is_enforced(tmp_path, fault):
    def change(doc, weights):
        if fault == "undeclared":
            weights["unexpected::weight"] = np.zeros(1)
        elif fault == "duplicate":
            doc["tensors"].append(dict(doc["tensors"][0]))
        else:
            doc["tensors"][0]["shape"] = [987]

    with pytest.raises(ModelLoadError, match=r"canonical tensor|disagrees"):
        load_model(edited(tmp_path, change))


def test_backend_discovery_and_no_training_claim():
    from mace_jax.kernels.reference import ReferenceBackend

    assert any(
        row.name == "reference" and row.loaded for row in available_backends("jax")
    )
    assert not ReferenceBackend().capabilities().supports_double_backward


def test_fresh_process_owns_x64_and_imports_no_torch():
    script = """import sys
import jax
jax.config.update("jax_enable_x64",False)
from mace_jax.deploy.neutral_io import load_model
apply, params = load_model(sys.argv[1])
assert jax.config.jax_enable_x64
sys.path.insert(0, sys.argv[2])
from jax_conformance import ModelSurface
from tests.golden import harness
import numpy as np
atoms = harness.load_fixtures(names=["water_cluster"])["water_cluster"]
actual = ModelSurface(apply, params).golden_outputs(atoms)
expected = harness.load_reference(
    harness.REFERENCES_DIR / "tiny_scaleshift_e3nn_cpu_fp64.json"
)["fixtures"]["water_cluster"]["outputs"]
for key in ("energy", "forces"):
    assert actual[key].dtype == np.float64
    np.testing.assert_allclose(actual[key], expected[key]["value"],
        atol=harness.FP64_CPU_REFERENCE.atol, rtol=harness.FP64_CPU_REFERENCE.rtol)
for name in ("torch","mace","mace_torch","e3nn","e3nn_jax"):
    assert name not in sys.modules
"""
    result = subprocess.run(
        [sys.executable, "-c", script, str(TINY), str(TINY.parent.parent)],
        text=True,
        capture_output=True,
    )
    assert result.returncode == 0, result.stderr


@pytest.mark.parametrize("fault", [None, "model", "heads", "elements"])
def test_named_artifact_uses_core_identity_and_checks_its_metadata(monkeypatch, fault):
    from mace_core import model_registry
    from mace_core.artifacts import ArtifactRef, CheckpointFiles
    from mace_core.metadata import ModelMetadata
    from mace_jax.deploy import neutral_io
    from mace_jax.models.registry import SUPPORTED_MODELS

    metadata = ModelMetadata.from_json(
        json.dumps(json.loads(TINY.read_text())["config"])
    )
    if fault == "model":
        metadata.config.resolved["model"]["num_channels"] += 1
    elif fault == "heads":
        metadata.heads = {"wrong_head": next(iter(metadata.heads.values()))}
    elif fault == "elements":
        metadata.elements = metadata.elements[:-1]
    entry = model_registry.get_model("mace-mp-0-small").model_copy(
        update={
            "artifact_ref": ArtifactRef(url="https://models.example/test.json"),
            "metadata": metadata,
        }
    )
    monkeypatch.setattr(model_registry, "_entries", lambda: (entry,))
    requests = []

    def resolve(reference):
        requests.append(reference)
        return CheckpointFiles(TINY, TINY.with_suffix(".safetensors"))

    monkeypatch.setattr(neutral_io, "resolve_checkpoint", resolve)
    assert entry.artifact_ref is not None
    assert SUPPORTED_MODELS[entry.model_id].artifact_ref == entry.artifact_ref.url
    if fault:
        with pytest.raises(ModelLoadError, match="disagree with registry"):
            load_model(entry.model_id)
    else:
        apply, _ = load_model(entry.model_id)
        assert apply.config.model == entry.hyperparameters()
    assert requests == [entry.artifact_ref.url]
