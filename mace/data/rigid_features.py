from __future__ import annotations

from dataclasses import dataclass

import torch


VALID_RIGID_FEATURE_MODES = (
    "none",
    "isotropic",
    "traceless_moi",
    "moi",
)


@dataclass(frozen=True)
class RigidFeatureSpec:
    mode: str
    irreps: str
    dimension: int


RIGID_FEATURE_SPECS = {
    "none": RigidFeatureSpec(
        mode="none",
        irreps="",
        dimension=0,
    ),
    "isotropic": RigidFeatureSpec(
        mode="isotropic",
        irreps="1x0e",
        dimension=1,
    ),
    "traceless_moi": RigidFeatureSpec(
        mode="traceless_moi",
        irreps="1x2e",
        dimension=5,
    ),
    "moi": RigidFeatureSpec(
        mode="moi",
        irreps="1x0e + 1x2e",
        dimension=6,
    ),
}


def validate_rigid_feature_mode(mode: str) -> str:
    normalized = mode.strip().lower()

    if normalized not in RIGID_FEATURE_SPECS:
        valid = ", ".join(VALID_RIGID_FEATURE_MODES)
        raise ValueError(
            f"Unknown rigid feature mode {mode!r}. "
            f"Expected one of: {valid}."
        )

    return normalized


def rigid_feature_spec(mode: str) -> RigidFeatureSpec:
    return RIGID_FEATURE_SPECS[
        validate_rigid_feature_mode(mode)
    ]


def select_rigid_features(
    inertia_irreps: torch.Tensor,
    mode: str,
) -> torch.Tensor:
    """
    Select rigid-body node features from the full MOI decomposition.

    The expected full representation is:

        1x0e + 1x2e

    stored as six Cartesian-tensor irrep components, with the scalar
    component first and the five l=2 components following it.
    """
    mode = validate_rigid_feature_mode(mode)

    if inertia_irreps.ndim != 2:
        raise ValueError(
            "inertia_irreps must have shape (N, 6); "
            f"got {tuple(inertia_irreps.shape)}."
        )

    if inertia_irreps.shape[1] != 6:
        raise ValueError(
            "Expected six inertia-irrep components for "
            "'1x0e + 1x2e'; "
            f"got {inertia_irreps.shape[1]}."
        )

    if mode == "none":
        return inertia_irreps.new_zeros(
            (inertia_irreps.shape[0], 0)
        )

    if mode == "isotropic":
        return inertia_irreps[:, :1]

    if mode == "traceless_moi":
        return inertia_irreps[:, 1:]

    if mode == "moi":
        return inertia_irreps

    raise AssertionError(
        f"Unhandled rigid feature mode: {mode}"
    )
