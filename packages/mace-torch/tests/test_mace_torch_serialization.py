"""The checkpoint format: what it holds, and what it refuses.

Two properties are the point and both are about what is *not* in the file. It
holds no pickle, so reading one does not execute anything. And it holds no
backend module tree, so a model trained with one set of kernels is not tied to
them.

The refusals get as much attention as the round trip, because every one of them
replaces a way a checkpoint can load and quietly give a different model.
"""

from __future__ import annotations

import json

import pytest
import torch
from conftest import fp64_only
from mace_core.elements import AtomicNumberTable, ResolvedE0s
from mace_core.kernels.precision import PrecisionConfig
from mace_torch.backends.reference import ReferenceBackend
from mace_torch.models import EnergyOutputHead, ScaleShiftSpec
from mace_torch.nn import MACEBackbone
from mace_torch.serialization import (
    FORMAT,
    VERSION,
    CheckpointError,
    canonical_state,
    load_checkpoint,
    save_checkpoint,
)
from safetensors.torch import load_file, save_file

SETTINGS = dict(
    atomic_numbers=[1, 8],
    num_layers=2,
    num_features=4,
    lmax=2,
    hidden_irreps="0e+1o",
    correlation=2,
    avg_num_neighbors=6.0,
)


def build(config):
    return MACEBackbone(ReferenceBackend(), **config)


def trained(seed: int = 0):
    torch.manual_seed(seed)
    model = build(dict(SETTINGS))
    generator = torch.Generator().manual_seed(seed + 1)
    with torch.no_grad():
        for parameter in model.parameters():
            parameter.copy_(
                torch.rand(parameter.shape, generator=generator, dtype=parameter.dtype)
                - 0.5
            )
    return model


@fp64_only
def test_a_model_survives_the_round_trip(tmp_path):
    """Compared on what it computes, not on its weights."""
    import numpy as np
    from mace_core.neighbors import get_neighborhood

    model = trained()
    save_checkpoint(tmp_path / "anchor", model, dict(SETTINGS))
    restored = load_checkpoint(tmp_path / "anchor", build)

    positions = np.array([[0.0, 0.0, 0.0], [0.95, 0.0, 0.0], [-0.24, 0.93, 0.0]])
    neighborhood = get_neighborhood(positions, 5.0, (False, False, False), None)
    graph = {
        "positions": torch.tensor(positions),
        "atomic_numbers": torch.tensor([8, 1, 1]),
        "edge_index": torch.tensor(neighborhood.edge_index),
        "shifts": torch.tensor(neighborhood.shifts),
    }
    before = model(graph)
    after = restored(graph)
    for index, (first, second) in enumerate(zip(before, after, strict=True)):
        assert torch.equal(first, second), f"layer {index} changed"
    assert float(before[0].abs().max()) > 1e-6, "the model computes nothing"


@fp64_only
def test_the_file_holds_no_pickle_and_no_module_tree(tmp_path):
    """What is written is each operator's canonical form, keyed by position.

    Not a `state_dict`: those name a backend's own submodules, so renaming one
    stops a checkpoint loading and swapping the backend means it never did.
    """
    model = trained()
    sidecar = save_checkpoint(tmp_path / "anchor", model, dict(SETTINGS))
    document = json.loads(sidecar.read_text())

    assert document["format"] == FORMAT
    assert document["version"] == VERSION
    assert document["config"] == SETTINGS

    written = {entry["module"] for entry in document["tensors"]}
    assert written == set(canonical_state(model))

    raw = (tmp_path / "anchor.safetensors").read_bytes()
    for marker in (b"__reduce__", b"torch.storage", b"pickle"):
        assert marker not in raw, f"the tensor file contains {marker!r}"


@fp64_only
def test_a_pickled_checkpoint_is_refused_by_name(tmp_path):
    """Reading one executes whatever it holds, so it is not read at all."""
    (tmp_path / "old.pt").write_bytes(b"not really a pickle")
    with pytest.raises(CheckpointError, match="executes whatever"):
        load_checkpoint(tmp_path / "old.pt", build)


@fp64_only
def test_a_missing_sidecar_is_refused(tmp_path):
    model = trained()
    save_checkpoint(tmp_path / "anchor", model, dict(SETTINGS))
    (tmp_path / "anchor.json").unlink()
    with pytest.raises(CheckpointError, match="two files"):
        load_checkpoint(tmp_path / "anchor", build)


@fp64_only
@pytest.mark.parametrize(
    "field,value,message",
    [("format", "someone-elses", "only reads"), ("version", 99, "written for")],
)
def test_a_file_from_elsewhere_is_refused(tmp_path, field, value, message):
    model = trained()
    sidecar = save_checkpoint(tmp_path / "anchor", model, dict(SETTINGS))
    document = json.loads(sidecar.read_text())
    document[field] = value
    sidecar.write_text(json.dumps(document))

    with pytest.raises(CheckpointError, match=message):
        load_checkpoint(tmp_path / "anchor", build)


@fp64_only
def test_tensors_the_sidecar_did_not_declare_are_refused(tmp_path):
    """The sidecar is the manifest, so an extra tensor is a disagreement."""
    model = trained()
    sidecar = save_checkpoint(tmp_path / "anchor", model, dict(SETTINGS))
    document = json.loads(sidecar.read_text())
    document["tensors"] = document["tensors"][:-1]
    sidecar.write_text(json.dumps(document))

    with pytest.raises(CheckpointError, match="Undeclared"):
        load_checkpoint(tmp_path / "anchor", build)


@fp64_only
def test_a_model_of_a_different_shape_is_refused(tmp_path):
    """A partial load leaves weights at their initial values and still runs.

    A third layer changes more than the operator count. It also moves which
    layer is last, and the last one carries only scalars, so the middle layer's
    contraction changes shape too. The layout record catches that first and
    names the operator, which is the more useful of the two refusals: an
    operator count says something is different, a path count on a named
    operator says what.
    """
    model = trained()
    save_checkpoint(tmp_path / "anchor", model, dict(SETTINGS))

    def build_deeper(config):
        return MACEBackbone(ReferenceBackend(), **{**config, "num_layers": 3})

    with pytest.raises(CheckpointError, match=r"products\.1\.contraction"):
        load_checkpoint(tmp_path / "anchor", build_deeper)


@fp64_only
def test_a_file_missing_an_operator_entirely_is_refused(tmp_path):
    """The other half of the refusal, which the layout check deliberately skips.

    Here every shared operator agrees, so only the operator set differs. That
    one belongs to `load_canonical_state`, and this pins that it still fires
    rather than being shadowed by the layout check.
    """
    model = trained()
    save_checkpoint(tmp_path / "anchor", model, dict(SETTINGS))
    sidecar = tmp_path / "anchor.json"
    document = json.loads(sidecar.read_text())
    dropped = "products.1.contraction"
    document["layout"].pop(dropped)
    document["tensors"] = [
        entry for entry in document["tensors"] if entry["module"] != dropped
    ]
    sidecar.write_text(json.dumps(document))

    tensors = load_file(str(tmp_path / "anchor.safetensors"))
    for key in [key for key in tensors if key.startswith(f"{dropped}::")]:
        tensors.pop(key)
    save_file(tensors, str(tmp_path / "anchor.safetensors"))

    with pytest.raises(CheckpointError, match="do not hold the same operators"):
        load_checkpoint(tmp_path / "anchor", build)


def energy_head(precision: str):
    return EnergyOutputHead(
        ResolvedE0s({"default": {1: -13.6, 8: -2040.0}}),
        ["default"],
        AtomicNumberTable([1, 8]),
        ScaleShiftSpec("std", (0.5,), (0.25,)),
        PrecisionConfig(model=precision, accumulate="float64"),
    )


@fp64_only
def test_the_isolated_atom_table_stays_float64_through_the_file(tmp_path):
    """Even when the model computes in float32.

    The table holds energies of thousands of eV. Writing it at the model's own
    precision would round them on the way out and no later cast recovers them.
    """
    head = energy_head("float32")
    save_checkpoint(tmp_path / "head", head, {})
    restored = load_checkpoint(tmp_path / "head", lambda _: energy_head("float32"))

    assert restored.e0_table.dtype == torch.float64
    assert torch.equal(restored.e0_table, head.e0_table)
    assert restored.scale.dtype == torch.float32


@fp64_only
def test_a_narrowed_isolated_atom_table_is_refused(tmp_path):
    head = energy_head("float64")
    with pytest.raises(ValueError, match="float64"):
        head.load_canonical(
            {
                "e0_table": head.e0_table.float(),
                "scale": head.scale,
                "shift": head.shift,
            }
        )
