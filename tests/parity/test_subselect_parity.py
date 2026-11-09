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
    [
        (2, "none", ()),
        (3, "none", ()),
        (4, "combinations", (8, 26)),
        (1, "inclusive", (8, 26)),
        (3, "exclusive", (8, 26)),
    ],
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


@pytest.mark.parametrize(
    ("filtering", "elements"),
    [
        ("none", ()),
        ("combinations", (1, 6, 8)),
        ("inclusive", (26,)),
        ("exclusive", (6,)),
    ],
)
def test_precomputed_descriptor_rows_match_the_frozen_filtered_input(
    isolated, tmp_path, monkeypatch, filtering, elements
):
    import numpy as np
    from mace_core.config.resolved import ResolvedConfig
    from mace_torch.finetune.stages import select_structures
    from mace_torch.finetune.subselect import ABSENT, passes_filter
    from mace_torch.train import data_stage

    from mace.cli.fine_tuning_select import _load_descriptors

    species = sorted(
        {symbol for atoms in SIX for symbol in atoms.get_chemical_symbols()}
    )
    sparse = [
        {
            symbol: np.array([index, index + 0.5])
            for symbol in set(atoms.get_chemical_symbols())
        }
        for index, atoms in enumerate(SIX)
    ]
    dense = np.array(
        [
            np.concatenate([row.get(symbol, np.full(2, ABSENT)) for symbol in species])
            for row in sparse
        ]
    )
    legacy_file, native_file = tmp_path / "legacy.npy", tmp_path / "numeric.npy"
    np.save(legacy_file, sparse, allow_pickle=True)
    np.save(native_file, dense, allow_pickle=False)
    mask = [passes_filter(atoms.numbers, elements, filtering) for atoms in SIX]
    filtered = [atoms.copy() for atoms, keep in zip(SIX, mask, strict=True) if keep]
    _load_descriptors(filtered, mask, str(legacy_file), None, len(SIX))
    expected = np.array(
        [
            np.concatenate(
                [
                    atoms.info["mace_descriptors"].get(symbol, np.full(2, ABSENT))
                    for symbol in species
                ]
            )
            for atoms in filtered
        ]
    )
    captured = []
    original = data_stage.select

    def observe(*args, **kwargs):
        captured.append(kwargs["structure_descriptors"])
        return original(*args, **kwargs)

    monkeypatch.setattr(data_stage, "select", observe)
    source = tmp_path / "pool.xyz"
    ase.io.write(source, SIX, format="extxyz")
    config = ResolvedConfig.model_validate(
        {
            "data": {
                "heads": {
                    "selected": {
                        "train_file": str(source),
                        "subselect": {
                            "method": "fps",
                            "num_samples": 1,
                            "filtering": filtering,
                            "elements": elements,
                            "structure_descriptors_file": str(native_file),
                        },
                    }
                }
            },
        }
    )
    select_structures(config, "selected", tmp_path / "selected.xyz")
    assert len(captured) == 1
    np.testing.assert_array_equal(captured[0], expected)
