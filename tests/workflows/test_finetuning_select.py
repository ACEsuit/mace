import ase.io as aio
import numpy as np
import pytest
from ase import Atoms
from ase.build import molecule

from mace.cli.fine_tuning_select import (
    FilteringType,
    SelectionSettings,
    SubselectType,
    _filter_pretraining_data,
    _load_descriptors,
    _maybe_save_descriptors,
    filter_atoms,
    select_samples,
)


@pytest.fixture(name="train_atoms_fixture")
def train_atoms():
    return [
        molecule("H2O"),
        molecule("CH4"),
        Atoms("Fe2O3"),
        Atoms("C"),
        Atoms("FeON"),
        Atoms("Fe"),
    ]


@pytest.fixture(name="train_atom_descriptors_fixture")
def train_atom_descriptors(train_atoms_fixture):
    return [
        {x: np.zeros(5) + i for x in atoms.symbols}
        for i, atoms in enumerate(train_atoms_fixture)
    ]


@pytest.mark.parametrize(
    "filtering_type, passes_filter, element_sublist",
    [
        (FilteringType.NONE, [True] * 6, []),
        (FilteringType.NONE, [True] * 6, ["C", "U", "Anything really"]),
        (
            FilteringType.COMBINATIONS,
            [False, False, True, False, False, True],
            ["O", "Fe"],
        ),
        (
            FilteringType.INCLUSIVE,
            [False, False, True, False, True, False],
            ["O", "Fe"],
        ),
        (
            FilteringType.EXCLUSIVE,
            [False, False, True, False, False, False],
            ["O", "Fe"],
        ),
    ],
)
def test_filter_data(
    train_atoms_fixture, filtering_type, passes_filter, element_sublist
):
    filtered, _, passes = _filter_pretraining_data(
        train_atoms_fixture, filtering_type, element_sublist
    )
    assert passes == passes_filter
    assert len(filtered) == sum(passes_filter)


@pytest.mark.parametrize(
    "passes_filter", [[True] * 6, [False, True, False, True, False, True]]
)
def test_load_descriptors(
    train_atoms_fixture, train_atom_descriptors_fixture, passes_filter, tmp_path
):
    for i, atoms in enumerate(train_atoms_fixture):
        atoms.info["mace_descriptors"] = train_atom_descriptors_fixture[i]
    save_path = tmp_path / "test.xyz"
    _maybe_save_descriptors(train_atoms_fixture, save_path.as_posix())
    assert all(not "mace_descriptors" in atoms.info for atoms in train_atoms_fixture)
    filtered_atoms = [
        x for x, passes in zip(train_atoms_fixture, passes_filter) if passes
    ]
    descriptors_path = save_path.as_posix().replace(".xyz", "_descriptors.npy")

    _load_descriptors(
        filtered_atoms,
        passes_filter,
        descriptors_path=descriptors_path,
        calc=None,
        full_data_length=len(train_atoms_fixture),
    )
    expected_descriptors = [
        train_atom_descriptors_fixture[i]
        for i, passes in enumerate(passes_filter)
        if passes
    ]
    for i, atoms in enumerate(filtered_atoms):
        assert "mace_descriptors" in atoms.info
        for key, value in expected_descriptors[i].items():
            assert np.allclose(atoms.info["mace_descriptors"][key], value)


def test_select_samples_random(train_atoms_fixture, tmp_path):
    input_file_path = tmp_path / "input.xyz"
    aio.write(input_file_path, train_atoms_fixture, format="extxyz")
    output_file_path = tmp_path / "output.xyz"

    settings = SelectionSettings(
        configs_pt=input_file_path.as_posix(),
        output=output_file_path.as_posix(),
        num_samples=2,
        subselect=SubselectType.RANDOM,
        filtering_type=FilteringType.NONE,
    )
    select_samples(settings)

    # Check if output file is created
    assert output_file_path.exists()
    combined_output_file_path = tmp_path / "output_combined.xyz"
    assert combined_output_file_path.exists()

    output_atoms = aio.read(output_file_path, index=":")
    assert isinstance(output_atoms, list)
    assert len(output_atoms) == 2

    combined_output_atoms = aio.read(combined_output_file_path, index=":")
    assert isinstance(combined_output_atoms, list)
    assert (
        len(combined_output_atoms) == 2
    )  # combined same as output since no FT data provided


def test_select_samples_ft_provided(train_atoms_fixture, tmp_path):
    input_file_path = tmp_path / "input.xyz"
    aio.write(input_file_path, train_atoms_fixture, format="extxyz")
    output_file_path = tmp_path / "output.xyz"
    ft_file_path = tmp_path / "ft_data.xyz"
    ft_data = [Atoms("FeO")]
    aio.write(ft_file_path.as_posix(), ft_data, format="extxyz")

    settings = SelectionSettings(
        configs_pt=input_file_path.as_posix(),
        output=output_file_path.as_posix(),
        num_samples=2,
        subselect=SubselectType.RANDOM,
        configs_ft=ft_file_path.as_posix(),
    )
    select_samples(settings)

    # Check if output file is created
    assert output_file_path.exists()
    combined_output_file_path = tmp_path / "output_combined.xyz"
    assert combined_output_file_path.exists()

    output_atoms = aio.read(output_file_path, index=":")
    assert isinstance(output_atoms, list)
    assert len(output_atoms) == 2
    assert all(filter_atoms(x, ["Fe", "O"]) for x in output_atoms)

    combined_atoms = aio.read(combined_output_file_path, index=":")
    assert isinstance(combined_atoms, list)
    assert len(combined_atoms) == len(output_atoms) + len(ft_data)
