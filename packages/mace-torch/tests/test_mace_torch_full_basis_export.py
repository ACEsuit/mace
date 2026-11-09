"""The external representative preserves predictions and is not a checkpoint."""

import json

import numpy as np
import pytest
import torch
from conftest import assert_close
from mace_core.clebsch_gordan import CouplingTree, full_to_reduced
from mace_core.cli import main
from mace_core.config.resolved import ResolvedConfig
from mace_core.observables import DEFAULT_CATALOGUE
from mace_torch.calculators import MACECalculator
from mace_torch.cli.model import _rebuild
from mace_torch.deploy import load_deployed
from mace_torch.deploy.bundle import export_lammps
from mace_torch.deploy.full_basis import export_full_basis_weights
from mace_torch.serialization import (
    CheckpointError,
    canonical_state,
    load_canonical_state,
)
from mace_torch.train import run_data_stage, run_model_stage, write_model
from safetensors.torch import load_file
from test_mace_torch_calculator import water
from test_mace_torch_full_batch import configuration


@pytest.fixture(params=[False, True], ids=["species", "shared"])
def source(tmp_path, dtype, request):
    doc = configuration(tmp_path, max_num_epochs=1).to_resolved_dict()
    doc["model"].update(
        use_agnostic_product=request.param,
        hidden_irreps="0e+1o+2e",
        max_ell=2,
        correlation=3,
        observables=["energy", "forces", "stress"],
    )
    doc["precision"]["model"] = str(dtype).removeprefix("torch.")
    config = ResolvedConfig.model_validate(doc)
    data = run_data_stage(config, DEFAULT_CATALOGUE)
    built = run_model_stage(config, data, DEFAULT_CATALOGUE)
    return write_model(tmp_path / "reduced", built.engine, built.metadata)


def exported_engine(source, artifact):
    """A test consumer that installs the exported tensors in full-basis ops."""
    deployed = load_deployed(source)
    config = deployed.config.model_copy(
        update={
            "model": deployed.config.model.model_copy(
                update={"clebsch_gordan_basis": "full"}
            )
        }
    )
    engine = _rebuild(deployed, config, deployed.heads)
    state = canonical_state(deployed.engine)
    record = json.loads(artifact.read_text())
    tensors = load_file(artifact.with_suffix(".safetensors"))
    for block in record["blocks"]:
        state[block["component"]] = {
            "weight": tensors[block["tensor"]],
            "path_counts": torch.tensor([group["count"] for group in block["groups"]]),
        }
    load_canonical_state(engine, state)
    return engine


def test_full_basis_function_and_roundtrip(source, dtype):
    artifact = export_full_basis_weights(source)
    record = json.loads(artifact.read_text())
    tensors = load_file(artifact.with_suffix(".safetensors"))
    reduced = canonical_state(load_deployed(source).engine)
    assert record["format"] == "mace-full-basis-weights" and record["export_only"]
    assert record["species"] == [1, 8]
    assert set(tensors) == {block["tensor"] for block in record["blocks"]}
    for block in record["blocks"]:
        full = tensors[block["tensor"]]
        assert list(full.shape) == block["shape"]
        assert full.dtype == dtype
        assert block["element_rows"] == (
            "shared" if full.shape[0] == 1 else "species_order"
        )
        recovered, offset = [], 0
        for group in block["groups"]:
            assert group["offset"] == offset
            labels = [CouplingTree.parse(label) for label in group["full_path_labels"]]
            assert len(set(labels)) == len(labels) == group["count"]
            assert all(label.correlation == group["order"] for label in labels)
            assert all(str(label.target) == group["target"] for label in labels)
            assert all(norm > 0 for norm in group["basis_frobenius_norms"])
            assert group["forward_factors"] == [1.0] * len(labels)
            piece = full[:, offset : offset + group["count"]].numpy()
            recovered.append(
                full_to_reduced(
                    piece, block["irreps_in"], group["order"], group["target"]
                )
            )
            offset += group["count"]
        assert offset == full.shape[1]
        assert_close(
            np.concatenate(recovered, axis=1), reduced[block["component"]]["weight"]
        )
    full_engine = exported_engine(source, artifact)
    calculator = MACECalculator(source)
    for periodic in (False, True):
        graph, _ = calculator._graph(water(periodic), padded=False)
        expected = calculator.models[0].compute(graph, compute=("forces", "stress"))
        actual = full_engine(graph, compute=("forces", "stress"), training=False)
        for key in ("total_energy", "forces", "stress"):
            assert_close(getattr(actual, key), getattr(expected, key).detach(), key)


def test_full_basis_cli_bundle_and_refusals(source, tmp_path, dtype, capsys):
    # Runtime pins cannot downcast or require a GPU to export the stored weights.
    bundle = export_lammps(source, halo_exchange=True, dtype="float32", backend="cueq")
    output = tmp_path / "external.json"
    assert (
        main(["export", "full-basis-weights", str(bundle), "--output", str(output)])
        == 0
    )
    with pytest.raises(CheckpointError, match=r"format|checkpoint"):
        load_deployed(output)
    with pytest.raises(FileExistsError):
        export_full_basis_weights(bundle, output)
    with pytest.raises(FileExistsError):
        export_full_basis_weights(source, source)
    assert (
        main(
            [
                "model",
                "convert",
                str(source),
                "--output",
                str(tmp_path / "bad"),
                "--to-basis",
                "full",
            ]
        )
        == 2
    )
    assert "under-determined" in capsys.readouterr().err
    assert not (tmp_path / "bad.json").exists()
    assert all(
        value.dtype == dtype
        for value in load_file(output.with_suffix(".safetensors")).values()
    )
