"""Configuration files and direct solver calls reject the same invalid inputs."""

from dataclasses import replace
from typing import Any

import pytest
from mace_core.config import ResolvedConfig
from mace_core.config.electrostatics import ElectrostaticsConfig
from mace_core.electrostatics import ElectrostaticsSpec, FeatureProjection, ScfSpec


def solve(**values):
    arguments: dict[str, Any] = (
        dict(
            periodicity_profile="molecular",
            multipole_max_l=0,
            kspace_cutoff=1.0,
            smearing_width=1.0,
        )
        | values
    )
    return ElectrostaticsSpec(**arguments)


@pytest.mark.parametrize("axis", [-1, 3, 7, 1.5, True])
def test_slab_axis_validation_agrees_at_both_boundaries(axis):
    values: dict[str, Any] = dict(periodicity_profile="z_slab", slab_normal=axis)
    for construct in (ElectrostaticsConfig, solve):
        with pytest.raises(ValueError, match="slab_normal"):
            construct(**values)
    with pytest.raises(ValueError, match="slab_normal"):
        ResolvedConfig.from_dict({"electrostatics": values})
    with pytest.raises(ValueError, match="slab_normal"):
        ElectrostaticsConfig().model_copy(update=values)


@pytest.mark.parametrize("value", [0, -1, float("nan"), float("inf"), -float("inf")])
def test_lengths_cutoffs_and_scf_tolerances_are_positive_and_finite(value):
    for name in ("kspace_cutoff", "smearing_width"):
        with pytest.raises(ValueError, match=name):
            solve(**{name: value})
    with pytest.raises(ValueError, match="kspace_cutoff_factor"):
        ElectrostaticsConfig(kspace_cutoff_factor=value)
    with pytest.raises(ValueError, match="widths"):
        FeatureProjection(max_l=0, widths=(value,))
    with pytest.raises(ValueError, match="tolerance"):
        ScfSpec(tolerance=value)


@pytest.mark.parametrize("value", [1.5, True, 0, -1])
def test_scf_iteration_limit_is_a_positive_integer(value):
    with pytest.raises(ValueError, match="max_iters"):
        ScfSpec(max_iters=value)


def test_options_and_replacements_are_validated_without_losing_hashability():
    with pytest.raises(ValueError, match="precision"):
        solve(precision="invalid")
    with pytest.raises(ValueError, match="normalization"):
        FeatureProjection(
            **dict[str, Any](max_l=0, widths=(1.0,), normalization="invalid")
        )
    with pytest.raises(ValueError, match="kspace_cutoff"):
        replace(solve(), kspace_cutoff=float("nan"))
    assert len({solve(), replace(solve())}) == 1
