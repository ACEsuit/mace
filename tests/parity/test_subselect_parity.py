"""The rewrite's random selection against the frozen tree's script.

Both are handed the same file and the same seed, and have to keep the same
structures. Random selection is the one both can be held to exactly: the
frozen tree seeds the global generator and draws once, and the rewrite draws
the same stream from a generator of its own. Farthest-point sampling is not
compared, because the frozen tree's falls back to a random draw when
`fpsample` is missing, which it is here, and that would be comparing two
random draws under another name.
"""

from __future__ import annotations

import ase.io
import pytest
from ase import Atoms
from ase.build import molecule
from mace_core.data import KeySpecification
from mace_core.data.xyz import read_configurations
from mace_torch.finetune import select

SIX = [
    molecule("H2O"),
    molecule("CH4"),
    Atoms("Fe2O3"),
    Atoms("C"),
    Atoms("FeON"),
    Atoms("Fe"),
]


def signature(atoms_or_configuration) -> str:
    numbers = getattr(atoms_or_configuration, "atomic_numbers", None)
    if numbers is None:
        numbers = atoms_or_configuration.get_atomic_numbers()
    return "-".join(str(int(n)) for n in numbers)


@pytest.mark.parametrize(
    ("num_samples", "filtering", "elements"),
    [(2, "none", ()), (3, "none", ()), (4, "combinations", (8, 26))],
)
@pytest.mark.parametrize("seed", [0, 42, 1234])
def test_a_random_selection_keeps_what_the_frozen_tree_keeps(
    isolated, tmp_path, num_samples, filtering, elements, seed
):
    from mace.cli.fine_tuning_select import (
        FilteringType,
        SelectionSettings,
        SubselectType,
        select_samples,
    )

    source = tmp_path / "pool.xyz"
    ase.io.write(source, SIX, format="extxyz")
    output = tmp_path / "kept.xyz"
    select_samples(
        SelectionSettings(
            configs_pt=str(source),
            output=str(output),
            num_samples=num_samples,
            subselect=SubselectType.RANDOM,
            filtering_type=FilteringType(filtering),
            filter_atomic_numbers_pt=list(elements) or None,
            seed=seed,
        )
    )
    legacy = [signature(atoms) for atoms in ase.io.read(output, index=":")]

    pool = read_configurations(
        source, KeySpecification.from_defaults(), no_data_ok=True
    ).configurations
    kept = select(
        pool,
        num_samples=num_samples,
        method="random",
        filtering=filtering,
        elements=elements,
        seed=seed,
    )
    assert [signature(item) for item in kept] == legacy
