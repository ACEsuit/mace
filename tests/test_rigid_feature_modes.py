from __future__ import annotations

import pytest
import torch

from mace.data.rigid_features import (
    mask_inertia_irreps,
    rigid_feature_spec,
    select_rigid_features,
    validate_rigid_feature_mode,
)


@pytest.mark.parametrize(
    ("mode", "dimension"),
    (
        ("none", 0),
        ("isotropic", 1),
        ("traceless_moi", 5),
        ("moi", 6),
        ("gyration", 6),
        ("steric_extent", 6),
        ("electrostatic_quadrupole", 6),
    ),
)
def test_selected_dimensions(mode: str, dimension: int) -> None:
    full = torch.randn(4, 6)
    assert select_rigid_features(full, mode).shape == (4, dimension)
    assert rigid_feature_spec(mode).dimension == dimension


def test_mode_validation() -> None:
    assert validate_rigid_feature_mode(" MOI ") == "moi"
    with pytest.raises(ValueError):
        validate_rigid_feature_mode("bad")


@pytest.mark.parametrize(
    "mode",
    (
        "none",
        "isotropic",
        "traceless_moi",
        "moi",
        "gyration",
        "steric_extent",
        "electrostatic_quadrupole",
    ),
)
def test_mask_preserves_six_component_layout(mode: str) -> None:
    full = torch.randn(3, 6)
    assert mask_inertia_irreps(full, mode).shape == full.shape


def test_masks_are_exact() -> None:
    full = torch.arange(18, dtype=torch.float64).reshape(3, 6)

    none = mask_inertia_irreps(full, "none")
    isotropic = mask_inertia_irreps(full, "isotropic")
    traceless = mask_inertia_irreps(full, "traceless_moi")
    moi = mask_inertia_irreps(full, "moi")
    gyration = mask_inertia_irreps(full, "gyration")
    steric_extent = mask_inertia_irreps(full, "steric_extent")
    electrostatic_quadrupole = mask_inertia_irreps(
        full,
        "electrostatic_quadrupole",
    )

    torch.testing.assert_close(none, torch.zeros_like(full))
    torch.testing.assert_close(isotropic[:, :1], full[:, :1])
    torch.testing.assert_close(isotropic[:, 1:], torch.zeros_like(full[:, 1:]))
    torch.testing.assert_close(traceless[:, :1], torch.zeros_like(full[:, :1]))
    torch.testing.assert_close(traceless[:, 1:], full[:, 1:])
    torch.testing.assert_close(moi, full)
    torch.testing.assert_close(gyration, full)
    torch.testing.assert_close(steric_extent, full)
    torch.testing.assert_close(electrostatic_quadrupole, full)
