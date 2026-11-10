"""Run native checkpoints against unchanged legacy values and gradients.

The gradient maps contain only parameter-coordinate transformations captured
from the existing converter. Reference outputs are read from tests/golden.
"""

import hashlib
import importlib.abc
import json
import sys
from pathlib import Path

import ase.io
import numpy as np
import pytest
import torch
from mace_core.data import KeySpecification, configuration_from_atoms
from mace_torch.data.inputs import graph_inputs_of
from mace_torch.deploy.loader import load_deployed
from mace_torch.deploy.reference import verify_against_reference
from mace_torch.graph import build_batch
from mace_torch.neighbors import resolve_neighbor_backend
from safetensors.torch import load_file

from tests.golden.harness import FP64_CPU_REFERENCE

ARTIFACTS = Path(__file__).parent / "artifacts"
GOLDEN = Path(__file__).parents[1] / "golden"
ENERGY_ANCHORS = ("tiny_scaleshift", "tiny_mace")
ANCHORS = (*ENERGY_ANCHORS, "tiny_dipoles", "tiny_dielectric", "tiny_magnetic")


class _NoLegacy(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split(".")[0] in {"mace", "e3nn", "mace_launcher"}:
            raise AssertionError(f"retirement gate imported {fullname}")
        return None


@pytest.fixture(autouse=True)
def forbid_legacy():
    finder = _NoLegacy()
    sys.meta_path.insert(0, finder)
    try:
        yield
    finally:
        sys.meta_path.remove(finder)


def test_artifact_provenance():
    manifest = json.loads((ARTIFACTS / "provenance.json").read_text())
    assert set(manifest["artifacts"]) == {*ANCHORS, "tiny_maceles"}
    for name, expected in manifest["sha256"].items():
        assert (
            hashlib.sha256((ARTIFACTS / name).read_bytes()).hexdigest() == expected
        ), name


@pytest.mark.parametrize(
    "anchor", (*ANCHORS, pytest.param("tiny_maceles", marks=pytest.mark.les))
)
def test_native_checkpoint_reproduces_existing_golden(anchor):
    loaded = load_deployed(ARTIFACTS / f"{anchor}.json")
    report = verify_against_reference(
        loaded.engine,
        GOLDEN / "references" / f"{anchor}_e3nn_cpu_fp64.json",
        GOLDEN / "fixtures",
        z_table=loaded.z_table,
        cutoff=loaded.config.model.r_max,
        graph_inputs=graph_inputs_of(loaded.config.model),
        atol=FP64_CPU_REFERENCE.atol,
        rtol=FP64_CPU_REFERENCE.rtol,
    )
    assert report.deviations
    assert report.passed, report.describe()


def training_step(anchor):
    loaded = load_deployed(ARTIFACTS / f"{anchor}.json")
    structures = [
        atoms
        for atoms in ase.io.read(GOLDEN / "fixtures/tiny_train.xyz", ":")
        if len(atoms) > 1
    ][:4]
    assert len(structures) == 4
    configurations = [
        configuration_from_atoms(
            atoms, KeySpecification.from_defaults(), head_name=loaded.heads[0]
        )
        for atoms in structures
    ]
    graph = build_batch(
        configurations,
        atomic_number_table=loaded.z_table,
        cutoff=loaded.config.model.r_max,
        heads=loaded.heads,
        neighbor_backend=resolve_neighbor_backend(),
        dtype=torch.float64,
    )
    output = loaded.engine(graph, compute=("forces",), training=True)
    energy = torch.tensor(
        [atoms.info["REF_energy"] for atoms in structures], dtype=torch.float64
    )
    forces = torch.tensor(
        np.concatenate([atoms.arrays["REF_forces"] for atoms in structures]),
        dtype=torch.float64,
    )
    counts = torch.tensor([len(atoms) for atoms in structures], dtype=torch.float64)
    reference = json.loads(
        (GOLDEN / "references" / f"{anchor}_train_step_grad_fp64.json").read_text()
    )
    weights = reference["metadata"]["loss_weights"]
    loss = (
        weights["energy_weight"]
        * ((output.total_energy - energy) / counts).square().mean()
    )
    loss = loss + weights["forces_weight"] * (output.forces - forces).square().mean()
    loss.backward()
    return loaded, loss, reference


@pytest.mark.parametrize("anchor", ENERGY_ANCHORS)
def test_every_parameter_gradient_matches_existing_digest(anchor):
    loaded, loss, reference = training_step(anchor)
    maps = json.loads((ARTIFACTS / f"{anchor}_gradient_map.json").read_text())
    transforms = load_file(ARTIFACTS / f"{anchor}_gradient_map.safetensors")
    parameters = dict(loaded.engine.named_parameters())
    assert {entry["source"] for entry in maps} == set(reference["parameters"])
    assert {entry["target"] for entry in maps} == {
        name for name, value in parameters.items() if value.numel()
    }
    norm_squared = 0.0
    for entry in maps:
        gradient = parameters[entry["target"]].grad
        assert gradient is not None, entry["target"]
        if entry["kind"] == "permutation":
            gradient = gradient.flatten()[transforms[entry["tensor"]]]
        elif entry["kind"] == "contraction":
            gradient = torch.einsum(
                "pq,zqc->zpc", transforms[entry["tensor"]], gradient
            )
        else:
            assert entry["kind"] == "identity"
        gradient = gradient.reshape(entry["shape"])
        flat = gradient.flatten()
        expected = reference["parameters"][entry["source"]]
        assert list(gradient.shape) == expected["shape"]
        assert flat.numel() == expected["count"]
        actual = {
            "sum": flat.sum().item(),
            "abs_sum": flat.abs().sum().item(),
            "sq_sum": flat.square().sum().item(),
            "projection": (
                flat * torch.cos(torch.arange(flat.numel(), dtype=torch.float64) + 1)
            )
            .sum()
            .item(),
        }
        norm_squared += actual["sq_sum"]
        for field, value in actual.items():
            assert value == pytest.approx(
                expected[field],
                abs=FP64_CPU_REFERENCE.atol,
                rel=FP64_CPU_REFERENCE.rtol,
            ), (entry["source"], field)
    assert loss.item() == pytest.approx(
        reference["loss"], abs=FP64_CPU_REFERENCE.atol, rel=FP64_CPU_REFERENCE.rtol
    )
    assert norm_squared**0.5 == pytest.approx(
        reference["global_gradient_norm"],
        abs=FP64_CPU_REFERENCE.atol,
        rel=FP64_CPU_REFERENCE.rtol,
    )
