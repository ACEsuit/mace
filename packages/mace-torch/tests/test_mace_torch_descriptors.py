"""Atomic descriptors reduced per structure."""

from typing import Any, cast

import pytest
import torch
from conftest import assert_close
from mace_torch.descriptors import aggregate_atomic_descriptors


def test_the_descriptors_aggregate_per_structure():
    descriptors = torch.arange(10.0).reshape(5, 2)
    batch = torch.tensor([0, 0, 1, 1, 1])
    element = torch.tensor([0, 1, 1, 1, 2])
    means = aggregate_atomic_descriptors(descriptors, batch, 2, "mean")
    assert_close(means, [[1.0, 2.0], [6.0, 7.0]], "mean")
    per_element = aggregate_atomic_descriptors(
        descriptors,
        batch,
        2,
        "per_element_mean",
        element=element,
        atomic_numbers=(1, 6, 8),
    )
    assert sorted(per_element[0]) == [1, 6]
    assert sorted(per_element[1]) == [6, 8]
    assert_close(per_element[1][6], [5.0, 6.0], "per element")
    assert aggregate_atomic_descriptors(descriptors, batch, 2, None) is descriptors
    with pytest.raises(ValueError, match="aggregation"):
        aggregate_atomic_descriptors(descriptors, batch, 2, cast(Any, "median"))
