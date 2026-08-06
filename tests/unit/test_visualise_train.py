"""Tests for plot-time data-loader key handling."""

import pytest

# mace.tools first on purpose: mace.tools.train imports TrainingPlotter, so
# importing mace.cli.visualise_train before it hits a partially initialised
# module. The cycle predates this test.
import mace.tools  # noqa: F401  # pylint: disable=unused-import
from mace.cli.visualise_train import belongs_to_head


@pytest.mark.parametrize(
    "name,head,expected",
    [
        # train/valid keys put the head last, test keys put it first
        ("train_H2O", "H2O", True),
        ("valid_H2O", "H2O", True),
        ("H2O_test", "H2O", True),
        ("H2O_liquid_test", "H2O", True),
        # a substring match would hand every H2O set to the H2 head as well
        ("train_H2O", "H2", False),
        ("valid_H2O", "H2", False),
        ("H2O_test", "H2", False),
        # and would claim the H2 sets for H2O in neither direction
        ("train_H2", "H2O", False),
        ("H2_test", "H2O", False),
        # unrelated heads never match
        ("train_solid", "H2O", False),
    ],
)
def test_belongs_to_head(name, head, expected):
    assert belongs_to_head(name, head) is expected
