from __future__ import annotations

import math

import numpy as np
import torch
from scipy.spatial.transform import Rotation

from mace.modules.rigid_pair import (
    rigid_pair_geometry,
    rigid_pair_harmonics,
)

DTYPE = torch.float64
ATOL = 2.0e-10
RTOL = 2.0e-10


def _wxyz(rotation: Rotation) -> torch.Tensor:
    x, y, z, w = rotation.as_quat()
    return torch.tensor(
        [w, x, y, z],
        dtype=DTYPE,
    )


def _axis_angle(axis, angle: float) -> Rotation:
    axis = np.asarray(axis, dtype=float)
    axis /= np.linalg.norm(axis)
    return Rotation.from_rotvec(axis * angle)


def _one_edge_geometry(
    body_i: Rotation,
    body_j: Rotation,
    edge_vector,
):
    quaternions = torch.stack(
        (
            _wxyz(body_i),
            _wxyz(body_j),
        )
    )

    edge_index = torch.tensor(
        [[0], [1]],
        dtype=torch.long,
    )

    edge_vectors = torch.tensor(
        [edge_vector],
        dtype=DTYPE,
    )

    return rigid_pair_geometry(
        quaternions,
        edge_index,
        edge_vectors,
    )


def test_pair_geometry_matches_scipy_body_frames():
    body_i = _axis_angle(
        [1.0, 2.0, 0.5],
        0.61,
    )
    body_j = _axis_angle(
        [-0.3, 1.0, 2.0],
        -0.44,
    )

    edge = np.array(
        [1.2, -0.7, 2.1],
        dtype=float,
    )

    geometry = _one_edge_geometry(
        body_i,
        body_j,
        edge,
    )

    rhat = edge / np.linalg.norm(edge)

    expected_i = body_i.inv().apply(rhat)
    expected_j = body_j.inv().apply(rhat)

    expected_relative = (
        body_i.inv() * body_j
    ).as_matrix()

    torch.testing.assert_close(
        geometry["body_i_direction"][0],
        torch.tensor(expected_i, dtype=DTYPE),
        atol=ATOL,
        rtol=RTOL,
    )

    torch.testing.assert_close(
        geometry["body_j_direction"][0],
        torch.tensor(expected_j, dtype=DTYPE),
        atol=ATOL,
        rtol=RTOL,
    )

    torch.testing.assert_close(
        geometry["relative_rotation"][0],
        torch.tensor(expected_relative, dtype=DTYPE),
        atol=ATOL,
        rtol=RTOL,
    )


def test_pair_geometry_is_global_rotation_invariant():
    body_i = _axis_angle(
        [1.0, 0.0, 0.0],
        0.28,
    )
    body_j = _axis_angle(
        [0.0, 1.0, 0.0],
        -0.67,
    )

    global_rotation = _axis_angle(
        [1.0, 2.0, 3.0],
        0.91,
    )

    edge = np.array(
        [1.3, -0.8, 2.4],
        dtype=float,
    )

    before = _one_edge_geometry(
        body_i,
        body_j,
        edge,
    )

    after = _one_edge_geometry(
        global_rotation * body_i,
        global_rotation * body_j,
        global_rotation.apply(edge),
    )

    for key in (
        "distance",
        "body_i_direction",
        "body_j_direction",
        "relative_rotation",
    ):
        torch.testing.assert_close(
            before[key],
            after[key],
            atol=ATOL,
            rtol=RTOL,
        )


def test_rotating_only_neighbor_changes_relative_orientation():
    body_i = Rotation.identity()

    body_j_a = Rotation.identity()

    body_j_b = _axis_angle(
        [0.0, 0.0, 1.0],
        math.pi / 3.0,
    )

    edge = [1.0, 0.3, -0.2]

    geometry_a = _one_edge_geometry(
        body_i,
        body_j_a,
        edge,
    )

    geometry_b = _one_edge_geometry(
        body_i,
        body_j_b,
        edge,
    )

    # Body i and the bead centers have not changed.
    torch.testing.assert_close(
        geometry_a["distance"],
        geometry_b["distance"],
        atol=ATOL,
        rtol=RTOL,
    )

    torch.testing.assert_close(
        geometry_a["body_i_direction"],
        geometry_b["body_i_direction"],
        atol=ATOL,
        rtol=RTOL,
    )

    # But neighbor orientation is a distinct physical state.
    assert not torch.allclose(
        geometry_a["relative_rotation"],
        geometry_b["relative_rotation"],
        atol=1.0e-8,
        rtol=1.0e-8,
    )

    assert not torch.allclose(
        geometry_a["body_j_direction"],
        geometry_b["body_j_direction"],
        atol=1.0e-8,
        rtol=1.0e-8,
    )


def test_rotating_only_center_changes_body_frame_direction():
    body_i_a = Rotation.identity()

    body_i_b = _axis_angle(
        [0.0, 0.0, 1.0],
        math.pi / 4.0,
    )

    body_j = Rotation.identity()

    edge = [1.0, 0.4, 0.7]

    geometry_a = _one_edge_geometry(
        body_i_a,
        body_j,
        edge,
    )

    geometry_b = _one_edge_geometry(
        body_i_b,
        body_j,
        edge,
    )

    torch.testing.assert_close(
        geometry_a["distance"],
        geometry_b["distance"],
        atol=ATOL,
        rtol=RTOL,
    )

    assert not torch.allclose(
        geometry_a["body_i_direction"],
        geometry_b["body_i_direction"],
        atol=1.0e-8,
        rtol=1.0e-8,
    )

    assert not torch.allclose(
        geometry_a["relative_rotation"],
        geometry_b["relative_rotation"],
        atol=1.0e-8,
        rtol=1.0e-8,
    )


def test_body_frame_directions_are_related_by_relative_rotation():
    body_i = _axis_angle(
        [1.0, 1.0, 0.0],
        0.52,
    )

    body_j = _axis_angle(
        [0.2, 1.0, 2.0],
        -0.39,
    )

    geometry = _one_edge_geometry(
        body_i,
        body_j,
        [0.7, -1.3, 2.2],
    )

    ui = geometry["body_i_direction"]
    uj = geometry["body_j_direction"]
    Rij = geometry["relative_rotation"]

    # u_j = R_j^T rhat
    #     = (R_i^T R_j)^T R_i^T rhat
    #     = R_ij^T u_i
    predicted_uj = torch.einsum(
        "ei,eij->ej",
        ui,
        Rij,
    )

    torch.testing.assert_close(
        predicted_uj,
        uj,
        atol=ATOL,
        rtol=RTOL,
    )


def test_quaternion_sign_does_not_change_pair_geometry():
    body_i = _axis_angle(
        [1.0, 2.0, 3.0],
        0.4,
    )

    body_j = _axis_angle(
        [-1.0, 0.5, 2.0],
        -0.7,
    )

    quaternions = torch.stack(
        (
            _wxyz(body_i),
            _wxyz(body_j),
        )
    )

    edge_index = torch.tensor(
        [[0], [1]],
        dtype=torch.long,
    )

    edge_vectors = torch.tensor(
        [[1.0, 2.0, -0.4]],
        dtype=DTYPE,
    )

    reference = rigid_pair_geometry(
        quaternions,
        edge_index,
        edge_vectors,
    )

    flipped = rigid_pair_geometry(
        -quaternions,
        edge_index,
        edge_vectors,
    )

    for key in (
        "distance",
        "body_i_direction",
        "body_j_direction",
        "relative_rotation",
    ):
        torch.testing.assert_close(
            reference[key],
            flipped[key],
            atol=ATOL,
            rtol=RTOL,
        )


def test_harmonic_pair_basis_is_global_rotation_invariant():
    body_i = _axis_angle(
        [1.0, 0.2, -0.4],
        0.57,
    )

    body_j = _axis_angle(
        [-0.2, 1.0, 0.7],
        -0.31,
    )

    global_rotation = _axis_angle(
        [1.0, 2.0, 3.0],
        0.83,
    )

    edge = np.array(
        [0.8, -1.1, 2.0],
        dtype=float,
    )

    geometry_before = _one_edge_geometry(
        body_i,
        body_j,
        edge,
    )

    geometry_after = _one_edge_geometry(
        global_rotation * body_i,
        global_rotation * body_j,
        global_rotation.apply(edge),
    )

    basis_before = rigid_pair_harmonics(
        geometry_before,
        lmax=2,
        jmax=2,
    )

    basis_after = rigid_pair_harmonics(
        geometry_after,
        lmax=2,
        jmax=2,
    )

    torch.testing.assert_close(
        basis_before["body_i_harmonics"],
        basis_after["body_i_harmonics"],
        atol=ATOL,
        rtol=RTOL,
    )

    torch.testing.assert_close(
        basis_before["body_j_harmonics"],
        basis_after["body_j_harmonics"],
        atol=ATOL,
        rtol=RTOL,
    )

    for before, after in zip(
        basis_before["wigner_D"],
        basis_after["wigner_D"],
    ):
        torch.testing.assert_close(
            before,
            after,
            atol=ATOL,
            rtol=RTOL,
        )


def test_l1_j1_basis_detects_relative_pair_rotation():
    body_i = Rotation.identity()

    body_j_a = Rotation.identity()
    body_j_b = _axis_angle(
        [1.0, 1.0, 0.0],
        0.63,
    )

    edge = [1.0, -0.3, 0.8]

    basis_a = rigid_pair_harmonics(
        _one_edge_geometry(
            body_i,
            body_j_a,
            edge,
        ),
        lmax=1,
        jmax=1,
    )

    basis_b = rigid_pair_harmonics(
        _one_edge_geometry(
            body_i,
            body_j_b,
            edge,
        ),
        lmax=1,
        jmax=1,
    )

    # l=1 body-i direction remains unchanged because body i and
    # the bead positions have not changed.
    torch.testing.assert_close(
        basis_a["body_i_harmonics"],
        basis_b["body_i_harmonics"],
        atol=ATOL,
        rtol=RTOL,
    )

    # The J=1 representation of relative orientation must change.
    assert not torch.allclose(
        basis_a["wigner_D"][1],
        basis_b["wigner_D"][1],
        atol=1.0e-8,
        rtol=1.0e-8,
    )
