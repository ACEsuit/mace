"""Custom scalar conditioning survives training and every graph consumer."""

import copy
from dataclasses import replace

import numpy as np
import pytest
from ase import Atoms
from ase.io import read, write
from conftest import fp64_only
from mace_core.cli import main
from mace_core.config.pseudolabels import PseudolabelConfig
from mace_core.config.resolved import ResolvedConfig
from mace_core.data.xyz import configuration_from_atoms
from mace_core.observables import DEFAULT_CATALOGUE
from mace_torch.calculators import FixedPadding, MACECalculator
from mace_torch.cli.run_train import run
from mace_torch.data import GraphDataset, make_loader
from mace_torch.data.inputs import key_specification
from mace_torch.deploy.loader import load_deployed
from mace_torch.finetune.foundation import read_foundation
from mace_torch.finetune.pseudolabels import generate_pseudolabels
from mace_torch.nn.graph_features import GraphFeatureEmbedding

from tests.golden.harness import CLOSED_FORM_FP64


def close(actual, expected):
    np.testing.assert_allclose(
        actual, expected, atol=CLOSED_FORM_FP64.atol, rtol=CLOSED_FORM_FP64.rtol
    )


@pytest.fixture
def device():
    return "cpu"


@pytest.fixture
def conditioned(tmp_path, request, device):
    storage, kind = request.param
    frames = []
    for index, count in enumerate((2, 3, 4, 2, 3, 4)):
        atoms = Atoms(
            "H" * count,
            positions=np.column_stack((np.arange(count) * 0.8, np.zeros((count, 2)))),
        )
        atoms.info["REF_energy"] = float(index) * 0.02
        atoms.arrays["REF_forces"] = np.zeros((count, 3))
        if kind == "continuous":
            value = (
                np.arange(count * 2).reshape(count, 2) * 0.1
                if storage == "atom"
                else np.array([index * 0.1, 0.3])
            )
        else:
            value = np.arange(count) % 3 + 2 if storage == "atom" else index % 3 + 2
        if storage == "atom":
            atoms.arrays["source_feature"] = np.asarray(value)
        else:
            atoms.info["source_feature"] = value
        frames.append(atoms)
    train, valid = tmp_path / "train.xyz", tmp_path / "valid.xyz"
    write(train, frames[:4])
    write(valid, frames[4:])
    config = ResolvedConfig.model_validate(
        {
            "runtime": {"work_dir": str(tmp_path), "seed": 17, "device": device},
            "data": {
                "heads": {
                    "default": {
                        "train_file": str(train),
                        "valid_file": str(valid),
                        "e0s": {"kind": "table", "values": {1: 0.0}},
                    }
                },
                "embedding_specs": {
                    "conditioning": {"per": storage, "key": "source_feature"}
                },
                "pin_memory": False,
            },
            "model": {
                "observables": ["energy", "forces"],
                "num_channels": 2,
                "num_interactions": 2,
                "max_ell": 1,
                "correlation": 2,
                "radial_mlp": [4],
                "readout": {"mlp_irreps": "4x0e"},
                "scaling": "none",
                "graph_features": [
                    {
                        "name": "conditioning",
                        "kind": kind,
                        "per": storage,
                        "input_dim": 2 if kind == "continuous" else 1,
                        "num_classes": 3 if kind == "categorical" else 0,
                        "offset": -2 if kind == "categorical" else 0,
                        "embedding_dim": 2,
                    }
                ],
            },
            "training": {
                "max_num_epochs": 1,
                "batch_size": 4,
                "valid_batch_size": 2,
                "scheduler": {"kind": "constant"},
            },
        }
    )
    trained = run(config)
    assert trained.checkpoint_path is not None
    return config, frames, trained.checkpoint_path


@fp64_only
@pytest.mark.parametrize("device", ["cpu", pytest.param("cuda", marks=pytest.mark.gpu)])
@pytest.mark.parametrize(
    "conditioned",
    [
        (storage, kind)
        for storage in ("atom", "graph")
        for kind in ("continuous", "categorical")
    ],
    indirect=True,
)
def test_conditioned_checkpoint_cli_calculator_padding_and_cache(
    conditioned, tmp_path, device
):
    config, frames, path = conditioned
    loaded = load_deployed(path, device=device)
    keys = key_specification(config.data, model=config.model)
    configurations = [configuration_from_atoms(atoms, keys) for atoms in frames]
    dataset = GraphDataset(
        configurations,
        cutoff=loaded.r_max,
        z_table=loaded.z_table,
        targets=(),
        features=config.model.feature_storages,
    )
    graph = (
        next(iter(make_loader(dataset, batch_size=6, shuffle=False))).to(device).graph
    )
    direct = loaded.compute(graph, compute=("forces",))
    assert direct.total_energy is not None and direct.forces is not None
    expected_energy = direct.total_energy.detach().cpu().numpy()
    expected_forces = direct.forces.detach().cpu().numpy()
    storage = config.model.graph_features[0].per
    expected_input = (
        np.concatenate([atoms.arrays["source_feature"] for atoms in frames])
        if storage == "atom"
        else np.stack([atoms.info["source_feature"] for atoms in frames])
    )
    np.testing.assert_array_equal(graph["conditioning"].cpu().numpy(), expected_input)
    assert loaded.config.model.feature_storages == {"conditioning": storage}

    all_frames = tmp_path / "all.xyz"
    write(all_frames, frames)
    for size in (1, 3):
        output = tmp_path / f"evaluated-{size}.xyz"
        assert (
            main(
                [
                    "eval",
                    "--device",
                    device,
                    "--model",
                    str(path),
                    "--configs",
                    str(all_frames),
                    "--output",
                    str(output),
                    "--batch_size",
                    str(size),
                ]
            )
            == 0
        )
        evaluated = read(output, index=":")
        close([atoms.info["MACE_energy"] for atoms in evaluated], expected_energy)
        # extxyz atom columns are written to eight decimal places.
        close(
            np.concatenate([atoms.arrays["MACE_forces"] for atoms in evaluated]),
            np.round(expected_forces, 8),
        )

    calculator = MACECalculator(model_paths=path, device=device)
    padded = MACECalculator(
        model_paths=path,
        device=device,
        padding=FixedPadding(nodes_budget=9, edges_budget=64, graphs_budget=2),
    )
    offset = 0
    for index, original in enumerate(frames):
        atoms = original.copy()
        for selected in (calculator, padded):
            atoms.calc = selected
            close(atoms.get_potential_energy(), expected_energy[index])
            close(atoms.get_forces(), expected_forces[offset : offset + len(atoms)])
        offset += len(atoms)
    atoms = copy.deepcopy(frames[0])
    atoms.calc = calculator
    before = atoms.get_potential_energy()
    values = atoms.arrays if storage == "atom" else atoms.info
    values["source_feature"] += 1
    assert calculator.check_state(atoms)
    after = atoms.get_potential_energy()
    assert before != after
    atoms.calc = MACECalculator(model_paths=path, device=device)
    close(atoms.get_potential_energy(), after)
    values.pop("source_feature")
    with pytest.raises(ValueError, match="conditioning"):
        atoms.get_potential_energy()

    if config.model.graph_features[0].kind == "categorical":
        # Both members accept raw label 3. Their category-zero file values,
        # 2 and 3, differ, so padding must use their common real input domain.
        second = load_deployed(path, device=device)
        feature = config.model.graph_features[0].model_copy(update={"offset": -3})
        second = replace(
            second,
            config=second.config.model_copy(
                update={
                    "model": second.config.model.model_copy(
                        update={"graph_features": (feature,)}
                    )
                }
            ),
        )
        embedding = second.model.get_submodule("backbone.graph_embedding")
        assert isinstance(embedding, GraphFeatureEmbedding)
        embedding.specs = [
            spec.model_copy(update={"offset": -3}) for spec in embedding.specs
        ]
        atoms = copy.deepcopy(frames[0])
        if storage == "atom":
            atoms.arrays["source_feature"][:] = 3
        else:
            atoms.info["source_feature"] = 3
        members = [load_deployed(path, device=device), second]
        atoms.calc = MACECalculator(models=members, device=device)
        energy, forces = atoms.get_potential_energy(), atoms.get_forces()
        atoms.calc = MACECalculator(
            models=members,
            device=device,
            padding=FixedPadding(nodes_budget=9, edges_budget=64, graphs_budget=2),
        )
        close(atoms.get_potential_energy(), energy)
        close(atoms.get_forces(), forces)

    # Foundation consumers build their own batches, including a mixed-size
    # pseudolabel batch and one kernel_spec batch per structure.
    foundation = read_foundation(path, DEFAULT_CATALOGUE)
    descriptors = foundation.structure_descriptors(configurations)
    assert descriptors.shape[0] == len(frames) and np.isfinite(descriptors).all()
    labelled = generate_pseudolabels(
        foundation.engine,
        configurations,
        PseudolabelConfig(),
        z_table=foundation.z_table,
        cutoff=loaded.r_max,
        head_index=0,
        features=config.model.feature_storages,
        batch_size=3,
    )
    close([item.properties["energy"] for item in labelled], expected_energy)
    close(
        np.concatenate([item.properties["forces"] for item in labelled]),
        expected_forces,
    )
    for original, result in zip(configurations, labelled, strict=True):
        np.testing.assert_array_equal(
            original.properties["conditioning"], result.properties["conditioning"]
        )


@fp64_only
@pytest.mark.parametrize(
    "conditioned", [("atom", "continuous"), ("graph", "continuous")], indirect=True
)
def test_feature_shape_errors_name_the_input(conditioned):
    _, frames, path = conditioned
    calculator = MACECalculator(model_paths=path)
    atoms = frames[0].copy()
    values = atoms.arrays if "source_feature" in atoms.arrays else atoms.info
    # Same number of scalar values can hide a swapped atom/component axis.
    values["source_feature"] = np.zeros((1, 4))
    atoms.calc = calculator
    with pytest.raises(ValueError, match="conditioning"):
        atoms.get_potential_energy()
