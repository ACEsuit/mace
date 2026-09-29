"""The data transforms: the registry, the three presets, and a written one.

What is checked is the arithmetic of each preset and the property every one of
them has to have: a shift that is constant per structure changes no force, and
a mask leaves the structure and its other properties alone.

The last test is the documented example of writing one, run as written. An
example that is only in a docstring is an example nobody has checked.
"""

from __future__ import annotations

import numpy as np
import pytest
from mace_core.data.configuration import Configuration
from mace_torch.data.transforms import (
    TRANSFORM_REGISTRY,
    TransformError,
    UnknownTransformError,
    apply_transforms,
    register_transform,
)

POSITIONS = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]])


def structure(energy, config_type="Default", **properties):
    return Configuration(
        atomic_numbers=np.array([1, 1]),
        positions=POSITIONS,
        properties={"energy": energy, "forces": np.ones((2, 3)), **properties},
        config_type=config_type,
    )


@pytest.fixture(name="clean_registry")
def fixture_clean_registry():
    """Registration is a global side effect, so a test that does it undoes it."""
    before = dict(TRANSFORM_REGISTRY)
    yield TRANSFORM_REGISTRY
    TRANSFORM_REGISTRY.clear()
    TRANSFORM_REGISTRY.update(before)


# ---------------------------------------------------------------------------
# Relative energies
# ---------------------------------------------------------------------------


def test_each_group_is_measured_from_its_own_lowest_structure():
    """Two sets on different absolute scales, which is why this exists: fitting
    the absolute numbers spends the model on the offset between them."""
    data = [
        structure(-10.0, "dimer"),
        structure(-8.0, "dimer"),
        structure(-1000.0, "bulk"),
        structure(-995.0, "bulk"),
    ]
    shifted = apply_transforms(data, [("relative_energy", {})])
    assert [item.properties["energy"] for item in shifted] == [0.0, 2.0, 0.0, 5.0]


def test_a_constant_shift_leaves_the_forces_alone():
    """It is a constant per structure, so its gradient is zero. A transform
    that moved the forces would be changing the physics, not the target."""
    data = [structure(-10.0), structure(-8.0)]
    shifted = apply_transforms(data, [("relative_energy", {})])
    for item in shifted:
        assert np.array_equal(item.properties["forces"], np.ones((2, 3)))


def test_a_structure_with_no_energy_is_refused_rather_than_left_absolute():
    """It would sit at its absolute value beside shifted neighbours, which is
    the one outcome nothing downstream could notice."""
    data = [
        structure(-10.0),
        Configuration(
            atomic_numbers=np.array([1, 1]), positions=POSITIONS, properties={}
        ),
    ]
    with pytest.raises(TransformError, match="relative_energy"):
        apply_transforms(data, [("relative_energy", {})])


def test_the_group_can_be_any_key_the_data_carries():
    data = [structure(-10.0, source="a"), structure(-4.0, source="b")]
    shifted = apply_transforms(data, [("relative_energy", {"group_by": "source"})])
    assert [item.properties["energy"] for item in shifted] == [0.0, 0.0]


# ---------------------------------------------------------------------------
# Subtracting one property from another
# ---------------------------------------------------------------------------


def test_the_interaction_energy_is_what_is_left():
    data = [structure(-10.0, intra_energy=-9.0)]
    result = apply_transforms(
        data, [("subtract_property", {"subtract": "intra_energy"})]
    )
    assert result[0].properties["energy"] == pytest.approx(-1.0)


def test_the_subtracted_property_is_removed_by_default():
    """Left in, a model that declared it would train on a quantity it has just
    been told to ignore."""
    data = [structure(-10.0, intra_energy=-9.0)]
    result = apply_transforms(
        data, [("subtract_property", {"subtract": "intra_energy"})]
    )
    assert "intra_energy" not in result[0].properties


def test_it_can_be_kept_when_something_downstream_wants_it():
    data = [structure(-10.0, intra_energy=-9.0)]
    result = apply_transforms(
        data, [("subtract_property", {"subtract": "intra_energy", "drop": False})]
    )
    assert result[0].properties["intra_energy"] == -9.0


def test_a_missing_part_is_refused_rather_than_taken_as_zero():
    """Zero would claim the fragments have no energy."""
    with pytest.raises(TransformError, match="intra_energy"):
        apply_transforms(
            [structure(-10.0)], [("subtract_property", {"subtract": "intra_energy"})]
        )


# ---------------------------------------------------------------------------
# Masking
# ---------------------------------------------------------------------------


def test_an_extreme_value_stops_counting_for_its_own_property():
    data = [structure(-10.0), structure(-8.0, forces=np.full((2, 3), 1000.0))]
    masked = apply_transforms(
        data, [("mask_above", {"property_name": "forces", "threshold": 10.0})]
    )
    assert masked[0].property_weights.get("forces", 1.0) == 1.0
    assert masked[1].property_weights["forces"] == 0.0


def test_the_masked_structure_and_its_other_properties_are_kept():
    """Dropping it would throw away a good energy because a force was wrong."""
    data = [structure(-8.0, forces=np.full((2, 3), 1000.0))]
    masked = apply_transforms(
        data, [("mask_above", {"property_name": "forces", "threshold": 10.0})]
    )
    assert len(masked) == 1
    assert masked[0].properties["energy"] == -8.0
    assert "energy" not in masked[0].property_weights


# ---------------------------------------------------------------------------
# The registry
# ---------------------------------------------------------------------------


def test_the_order_is_part_of_the_meaning():
    """Shifting and then masking is not the same run as the other way round."""
    data = [structure(-10.0), structure(-8.0)]
    shifted_then = apply_transforms(
        data,
        [
            ("relative_energy", {}),
            ("mask_above", {"property_name": "energy", "threshold": 1.0}),
        ],
    )
    masked_then = apply_transforms(
        data,
        [
            ("mask_above", {"property_name": "energy", "threshold": 1.0}),
            ("relative_energy", {}),
        ],
    )
    assert shifted_then[1].property_weights["energy"] == 0.0
    assert masked_then[0].property_weights["energy"] == 0.0


def test_an_unknown_transform_lists_what_there_is():
    with pytest.raises(UnknownTransformError, match="relative_energy"):
        apply_transforms([structure(-1.0)], [("no_such_transform", {})])


def test_registering_one_name_twice_is_refused(clean_registry):
    @register_transform("probe_duplicate_transform")
    def first():
        return lambda items: list(items)

    with pytest.raises(ValueError, match="already a registered transform"):

        @register_transform("probe_duplicate_transform")
        def second():
            return lambda items: list(items)


def test_a_transform_written_outside_this_package_runs_as_documented(clean_registry):
    """The example, as a reader would write it: a factory that takes its
    settings and returns the function that rewrites the structures."""

    @register_transform("scale_energy")
    def scale_energy(factor: float = 1.0):
        def transform(configurations):
            return [
                Configuration(
                    atomic_numbers=item.atomic_numbers,
                    positions=item.positions,
                    properties={
                        **item.properties,
                        "energy": item.properties["energy"] * factor,
                    },
                )
                for item in configurations
            ]

        return transform

    result = apply_transforms([structure(-10.0)], [("scale_energy", {"factor": 0.5})])
    assert result[0].properties["energy"] == pytest.approx(-5.0)
