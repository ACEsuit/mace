"""Prepared v2 data against frozen readers and whole-dataset statistics."""

import json

import numpy as np
import pytest
import torch
from ase.io import read, write
from mace_core.data import KeySpecification, open_dataset
from mace_torch.cli.preprocess_data import main

from tests.golden.harness import FP64_CPU_REFERENCE
from tests.parity.test_data_backend_parity import assert_same, frames


@pytest.mark.parametrize("custom_properties", [False, True])
def test_custom_labels_weights_and_missing_properties_survive_preparation(
    tmp_path, custom_properties
):
    from mace.data import KeySpecification as FrozenKeys
    from mace.data import config_from_atoms

    configurations = frames(5, 7)
    info = {
        name: f"label_{name}"
        for name in ("energy", "stress", "virials", "dipole", "polarizability")
    }
    arrays = {name: f"label_{name}" for name in ("forces", "charges")}
    for index, atoms in enumerate(configurations):
        atoms.info.update(
            label_energy=atoms.info.pop("REF_energy"),
            label_stress=atoms.info.pop("REF_stress"),
            label_virials=np.eye(3) * index,
            label_dipole=np.arange(3, dtype=float) + index,
            label_polarizability=np.eye(3) * (index + 1),
            config_weight=0.5,
            config_energy_weight=0.25,
            config_forces_weight=0.75,
            config_type="bulk" if index % 2 else "molecule",
        )
        atoms.arrays["label_forces"] = atoms.arrays.pop("REF_forces")
        atoms.arrays["label_charges"] = np.arange(3, dtype=float) / 10
    configurations[-1].info.pop("label_dipole")
    configurations[-1].arrays.pop("label_forces")
    source = tmp_path / "custom.xyz"
    write(source, configurations)
    output = tmp_path / "prepared"
    if not custom_properties:
        info = {"energy": info["energy"]}
        arrays = {"forces": arrays["forces"]}
    flags = [
        arg
        for name, key in {**info, **arrays}.items()
        for arg in (f"--{name}_key", key)
    ]
    main(
        [
            "--train_file",
            str(source),
            "--valid_file",
            str(source),
            "--test_file",
            str(source),
            "--output",
            str(output),
            "--E0s",
            "{1: 0.0, 8: 0.0}",
            "--shuffle",
            "false",
            "--config_type_weights",
            "{'bulk': 2.0, 'molecule': 0.25}",
            "--num_process",
            "1",
            "--shard_size",
            "2",
            *flags,
        ]
    )
    assert not (output / "statistics.json").exists()
    keys = FrozenKeys.from_defaults().update(info_keys=info, arrays_keys=arrays)
    expected = [
        config_from_atoms(
            atoms,
            key_specification=keys,
            config_type_weights={"bulk": 2.0, "molecule": 0.25},
        )
        for atoms in read(source, index=":")
    ]
    for split in ("train", "valid", "test"):
        actual = open_dataset(output / split, key_spec=KeySpecification.from_defaults())
        assert len(actual) == len(expected) == 5
        for index, frozen in enumerate(expected):
            assert_same(actual[index], frozen, (split, index))
        assert actual[len(actual) - 1].properties["forces"] is None
        assert actual[len(actual) - 1].property_weights["forces"] == 0.0
        assert actual[len(actual) - 1].properties["dipole"] is None
        assert actual[len(actual) - 1].property_weights["dipole"] == 0.0


def test_uneven_shard_statistics_match_the_frozen_whole_dataset(tmp_path, isolated):
    from mace.data import AtomicData, config_from_atoms
    from mace.data import KeySpecification as FrozenKeys
    from mace.modules import compute_statistics
    from mace.tools import AtomicNumberTable
    from mace.tools.torch_geometric.dataloader import DataLoader

    source = tmp_path / "train.xyz"
    write(source, frames(7, 12))
    output = tmp_path / "prepared"
    main(
        [
            "--train_file",
            str(source),
            "--valid_file",
            str(source),
            "--output",
            str(output),
            "--E0s",
            "{1: -0.2, 8: -1.0}",
            "--num_process",
            "1",
            "--shard_size",
            "3",
            "--shuffle",
            "false",
            "--r_max",
            "3.0",
            "--compute_statistics",
        ]
    )
    torch.set_default_dtype(torch.float64)
    graphs = [
        AtomicData.from_config(
            config_from_atoms(atoms, key_specification=FrozenKeys.from_defaults()),
            z_table=AtomicNumberTable([1, 8]),
            cutoff=3.0,
        )
        for atoms in read(source, index=":")
    ]
    neighbors, mean, rms = compute_statistics(
        DataLoader(graphs, batch_size=2), np.array([-0.2, -1.0])
    )
    actual = json.loads((output / "statistics.json").read_text())
    np.testing.assert_allclose(
        [actual["avg_num_neighbors"], actual["mean"], actual["std"]],
        [neighbors, mean.item(), rms.item()],
        atol=FP64_CPU_REFERENCE.atol,
        rtol=FP64_CPU_REFERENCE.rtol,
    )
    assert actual["atomic_energies"] == {"1": -0.2, "8": -1.0}
    assert actual["atomic_numbers"] == [1, 8]
