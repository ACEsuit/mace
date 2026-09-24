"""The magnetic-moment augmentation: which symmetry it draws, and where.

Ported from the frozen tree's own tests of the same rotation, and extended with
what they could not check there: that it is drawn again on every read, that it
reaches the training structures and not the evaluated ones, and that it is
selected from a configuration.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch
from mace_core.data.configuration import Configuration
from mace_torch.data.augmentation import (
    AUGMENTATION_REGISTRY,
    UnknownAugmentationError,
    build_augmentations,
)

POSITIONS = np.array([[0.0, 0.0, 0.0], [1.8, 0.0, 0.0]])


def configuration(moments, forces=None) -> Configuration:
    properties = {"magmom": np.asarray(moments, dtype=float)}
    if forces is not None:
        properties["magforces"] = np.asarray(forces, dtype=float)
    return Configuration(
        atomic_numbers=np.array([26, 26]),
        positions=POSITIONS,
        properties=properties,
    )


def augmentation(mode="non-soc"):
    (augment,) = build_augmentations([("magnetic_moments", {"mode": mode})])
    return augment


def test_the_moments_and_their_forces_turn_by_one_transform():
    """Every inner product survives, the cross terms between the moments and
    their forces included, which is what one shared transform preserves."""
    torch.manual_seed(0)
    moments = np.array([[0.0, 0.0, 1.0], [1.0, 0.0, 0.0]])
    forces = np.array([[0.0, 1.0, 0.0], [0.0, 0.0, 2.0]])
    drawn = augmentation()(configuration(moments, forces))
    before = np.vstack([moments, forces])
    after = np.vstack([drawn.properties["magmom"], drawn.properties["magforces"]])
    np.testing.assert_allclose(after @ after.T, before @ before.T, atol=1e-12)
    assert not np.allclose(after, before)


def test_the_positions_are_never_turned():
    torch.manual_seed(1)
    drawn = augmentation()(configuration([[0.0, 0.0, 2.0], [0.0, 1.0, 0.0]]))
    assert np.array_equal(drawn.positions, POSITIONS)


@pytest.mark.parametrize("draws", [1, 5])
def test_the_orientations_are_uniform(draws):
    """A uniform rotation takes a fixed unit vector to one whose z component
    is uniform on [-1, 1], with standard deviation 1/sqrt(3). The reversal
    keeps that, and so does a product of uniform draws."""
    torch.manual_seed(0)
    augment = augmentation()
    heights = []
    for _ in range(4000):
        drawn = configuration([[0.0, 0.0, 1.0], [0.0, 0.0, 1.0]])
        for _ in range(draws):
            drawn = augment(drawn)
        heights.append(drawn.properties["magmom"][0, 2])
    heights = np.array(heights)
    assert abs(heights.mean()) < 0.03
    assert abs(heights.std() - 1.0 / np.sqrt(3.0)) < 0.03


def test_with_spin_orbit_coupling_only_the_reversal_is_drawn():
    torch.manual_seed(0)
    augment = augmentation("soc")
    moments = np.array([[0.3, -0.4, 1.2], [1.0, 0.5, 0.0]])
    signs = []
    for _ in range(400):
        drawn = augment(configuration(moments)).properties["magmom"]
        sign = 1.0 if np.allclose(drawn, moments) else -1.0
        np.testing.assert_allclose(drawn, sign * moments)
        signs.append(sign)
    assert abs(np.mean(signs)) < 0.15


def test_a_structure_without_moments_is_left_as_it_is():
    plain = Configuration(
        atomic_numbers=np.array([26]),
        positions=np.zeros((1, 3)),
        properties={"energy": -1.0},
    )
    assert augmentation()(plain) is plain


def test_a_mode_that_is_neither_is_refused():
    with pytest.raises(ValueError, match="non-soc"):
        augmentation("both")


def test_an_augmentation_nobody_registered_is_named():
    with pytest.raises(UnknownAugmentationError, match="magnetic_moments"):
        build_augmentations([("spin_flip", {})])
    assert "magnetic_moments" in AUGMENTATION_REGISTRY


def test_it_is_drawn_on_every_read_of_a_training_structure_and_of_no_other():
    from mace_core.elements import AtomicNumberTable
    from mace_core.observables import DEFAULT_CATALOGUE, resolve_requested
    from mace_torch.data.batch import GraphDataset
    from mace_torch.data.graphs import target_specs

    specs = target_specs(resolve_requested(["energy"], DEFAULT_CATALOGUE))
    structure = configuration([[0.0, 0.0, 2.0], [0.0, 2.0, 0.0]])
    structure.properties["energy"] = -3.0

    def dataset(augmentations=()):
        return GraphDataset(
            [structure],
            cutoff=3.0,
            z_table=AtomicNumberTable([26]),
            targets=specs,
            graph_inputs=("magmom",),
            augmentations=augmentations,
        )

    training = dataset([augmentation()])
    evaluated = dataset()
    torch.manual_seed(3)
    reads = [training[0][0]["magmom"] for _ in range(4)]
    assert len({tuple(np.round(read.ravel(), 12)) for read in reads}) == 4
    assert np.array_equal(evaluated[0][0]["magmom"], structure.properties["magmom"])
    assert np.array_equal(
        structure.properties["magmom"], [[0.0, 0.0, 2.0], [0.0, 2.0, 0.0]]
    )
