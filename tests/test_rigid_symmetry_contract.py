from __future__ import annotations

import math

import numpy as np
import torch
from scipy.spatial.transform import Rotation

from mace.data.rigid_body import (
    inertia_edge_invariants,
    principal_tensor_from_values,
    quaternion_to_matrix,
)

DTYPE = torch.float64
ATOL = 1.0e-12
RTOL = 1.0e-12


def _torch_wxyz(rotation: Rotation) -> torch.Tensor:
    """Convert SciPy's xyzw quaternion to MACE's wxyz convention."""
    x, y, z, w = rotation.as_quat()
    return torch.tensor([w, x, y, z], dtype=DTYPE)


def _axis_angle(axis, angle: float) -> Rotation:
    axis = np.asarray(axis, dtype=float)
    axis /= np.linalg.norm(axis)
    return Rotation.from_rotvec(axis * angle)


def test_quaternion_to_matrix_matches_scipy_reference():
    """MACE wxyz -> rotation matrix agrees with an independent implementation."""
    rotation = _axis_angle([1.0, 2.0, -3.0], 0.73)
    q = _torch_wxyz(rotation)

    expected = torch.tensor(rotation.as_matrix(), dtype=DTYPE)
    actual = quaternion_to_matrix(q)

    torch.testing.assert_close(
        actual,
        expected,
        atol=ATOL,
        rtol=RTOL,
    )


def test_quaternion_double_cover_is_physically_identical():
    """q and -q must describe exactly the same rigid-body orientation."""
    q = _torch_wxyz(
        _axis_angle([1.0, 2.0, -3.0], 0.73)
    )

    torch.testing.assert_close(
        quaternion_to_matrix(q),
        quaternion_to_matrix(-q),
        atol=ATOL,
        rtol=RTOL,
    )


def test_global_rotation_composition_matches_scipy():
    """A global rotation acts on a body orientation as S R_i."""
    body = _axis_angle([1.0, 1.0, 0.0], 0.42)
    global_rotation = _axis_angle([0.0, 1.0, 1.0], -0.81)

    q_body = _torch_wxyz(body)

    # SciPy multiplication means:
    #
    #     (global_rotation * body).apply(v)
    #       == global_rotation.apply(body.apply(v))
    #
    # hence this represents S R_i.
    q_global_body = _torch_wxyz(
        global_rotation * body
    )

    principal = torch.tensor(
        [1.0, 2.0, 4.0],
        dtype=DTYPE,
    )

    tensor = principal_tensor_from_values(
        q_body,
        principal,
    )

    rotated_tensor = principal_tensor_from_values(
        q_global_body,
        principal,
    )

    S = torch.tensor(
        global_rotation.as_matrix(),
        dtype=DTYPE,
    )

    torch.testing.assert_close(
        rotated_tensor,
        S @ tensor @ S.T,
        atol=ATOL,
        rtol=RTOL,
    )


def test_rank2_pair_invariants_are_global_rotation_invariant():
    """Rotate bodies and displacement together: scalar edge features cannot change."""
    body_i = _axis_angle(
        [1.0, 0.0, 0.0],
        0.22,
    )

    body_j = _axis_angle(
        [0.0, 1.0, 0.0],
        -0.47,
    )

    global_rotation = _axis_angle(
        [1.0, 2.0, 3.0],
        0.91,
    )

    principal = torch.tensor(
        [1.0, 2.0, 5.0],
        dtype=DTYPE,
    )

    inertia = torch.stack(
        (
            principal_tensor_from_values(
                _torch_wxyz(body_i),
                principal,
            ),
            principal_tensor_from_values(
                _torch_wxyz(body_j),
                principal,
            ),
        )
    )

    edge_index = torch.tensor(
        [[0], [1]],
        dtype=torch.long,
    )

    edge_vector = torch.tensor(
        [[1.2, -0.7, 2.1]],
        dtype=DTYPE,
    )

    before = inertia_edge_invariants(
        inertia,
        edge_index,
        edge_vector,
    )

    S = torch.tensor(
        global_rotation.as_matrix(),
        dtype=DTYPE,
    )

    rotated_inertia = S @ inertia @ S.T

    # Row-vector convention used for edge vectors here.
    rotated_edge_vector = edge_vector @ S.T

    after = inertia_edge_invariants(
        rotated_inertia,
        edge_index,
        rotated_edge_vector,
    )

    torch.testing.assert_close(
        before,
        after,
        atol=ATOL,
        rtol=RTOL,
    )


def test_rank2_pair_invariants_detect_inequivalent_single_body_rotation():
    """Holding centers fixed while rotating one anisotropic bead must be detectable."""
    identity = Rotation.identity()

    body_j_rotated = _axis_angle(
        [0.0, 0.0, 1.0],
        math.pi / 3.0,
    )

    principal = torch.tensor(
        [1.0, 2.0, 5.0],
        dtype=DTYPE,
    )

    edge_index = torch.tensor(
        [[0], [1]],
        dtype=torch.long,
    )

    edge_vector = torch.tensor(
        [[1.0, 1.0, 0.4]],
        dtype=DTYPE,
    )

    inertia_a = torch.stack(
        (
            principal_tensor_from_values(
                _torch_wxyz(identity),
                principal,
            ),
            principal_tensor_from_values(
                _torch_wxyz(identity),
                principal,
            ),
        )
    )

    inertia_b = torch.stack(
        (
            principal_tensor_from_values(
                _torch_wxyz(identity),
                principal,
            ),
            principal_tensor_from_values(
                _torch_wxyz(body_j_rotated),
                principal,
            ),
        )
    )

    features_a = inertia_edge_invariants(
        inertia_a,
        edge_index,
        edge_vector,
    )

    features_b = inertia_edge_invariants(
        inertia_b,
        edge_index,
        edge_vector,
    )

    assert not torch.allclose(
        features_a,
        features_b,
        atol=1.0e-10,
        rtol=1.0e-10,
    )


def test_generic_moi_has_d2_stabilizer():
    """A generic diagonal rank-2 tensor is blind to principal-axis pi rotations."""
    principal = torch.tensor(
        [1.0, 2.0, 5.0],
        dtype=DTYPE,
    )

    reference = principal_tensor_from_values(
        _torch_wxyz(Rotation.identity()),
        principal,
    )

    for axis in np.eye(3):
        symmetry = Rotation.from_rotvec(
            axis * math.pi
        )

        transformed = principal_tensor_from_values(
            _torch_wxyz(symmetry),
            principal,
        )

        torch.testing.assert_close(
            reference,
            transformed,
            atol=ATOL,
            rtol=RTOL,
        )


def test_axisymmetric_moi_has_continuous_stabilizer():
    """If two principal moments coincide, rotations about that axis are invisible."""
    principal = torch.tensor(
        [2.0, 2.0, 5.0],
        dtype=DTYPE,
    )

    reference = principal_tensor_from_values(
        _torch_wxyz(Rotation.identity()),
        principal,
    )

    for angle in (0.17, 0.91, 2.4):
        symmetry = _axis_angle(
            [0.0, 0.0, 1.0],
            angle,
        )

        transformed = principal_tensor_from_values(
            _torch_wxyz(symmetry),
            principal,
        )

        torch.testing.assert_close(
            reference,
            transformed,
            atol=ATOL,
            rtol=RTOL,
        )
