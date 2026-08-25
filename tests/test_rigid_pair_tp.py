from __future__ import annotations

import numpy as np
import torch
from e3nn import o3
from scipy.spatial.transform import Rotation

from mace.modules.rigid_pair_tp import (
    RigidPairTensorProductFeatures,
)

DTYPE = torch.float64
ATOL = 5.0e-7
RTOL = 5.0e-7


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


def _single_edge(
    model,
    body_i: Rotation,
    body_j: Rotation,
    edge,
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
        [edge],
        dtype=DTYPE,
    )

    return model(
        quaternions,
        edge_index,
        edge_vectors,
    )


def test_output_dimension_is_full_input_product():
    """FullTensorProduct must not silently compress the pair representation."""
    model = RigidPairTensorProductFeatures(
        lmax=2
    ).to(dtype=DTYPE)

    # spherical harmonics through l=2:
    #
    #   1 + 3 + 5 = 9
    #
    # each complete frame:
    #
    #   3 axes * 3 Cartesian components = 9
    #
    # Full TP preserves the full tensor-product dimension.
    expected_dimension = 9 * 9 * 9

    assert model.edge_irreps.dim == 9
    assert model.frame_irreps.dim == 9
    assert model.irreps_out.dim == expected_dimension


def test_frame_features_are_three_l1_vectors():
    model = RigidPairTensorProductFeatures(
        lmax=1
    ).to(dtype=DTYPE)

    body = _axis_angle(
        [1.0, 2.0, -0.5],
        0.61,
    )

    from mace.data.rigid_body import quaternion_to_matrix

    R = quaternion_to_matrix(
        _wxyz(body)
    )

    features = model._frame_features(
        R.unsqueeze(0)
    )[0]

    expected = torch.tensor(
        body.as_matrix().T.reshape(-1),
        dtype=DTYPE,
    )

    torch.testing.assert_close(
        features,
        expected,
        atol=ATOL,
        rtol=RTOL,
    )


def test_pair_tensor_product_is_globally_equivariant():
    """The complete TP output must transform exactly according to irreps_out."""
    model = RigidPairTensorProductFeatures(
        lmax=2
    ).to(dtype=DTYPE)

    body_i = _axis_angle(
        [1.0, 0.3, -0.4],
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

    edge = np.asarray(
        [0.8, -1.1, 2.0],
        dtype=float,
    )

    before = _single_edge(
        model,
        body_i,
        body_j,
        edge,
    )

    after = _single_edge(
        model,
        global_rotation * body_i,
        global_rotation * body_j,
        global_rotation.apply(edge),
    )

    S = torch.tensor(
        global_rotation.as_matrix(),
        dtype=DTYPE,
    )

    D = model.irreps_out.D_from_matrix(S)

    expected_after = before @ D.T

    torch.testing.assert_close(
        after,
        expected_after,
        atol=ATOL,
        rtol=RTOL,
    )


def test_pair_features_change_when_only_neighbor_rotates():
    model = RigidPairTensorProductFeatures(
        lmax=2
    ).to(dtype=DTYPE)

    body_i = Rotation.identity()

    body_j_a = Rotation.identity()

    body_j_b = _axis_angle(
        [0.3, 1.0, 0.2],
        0.71,
    )

    edge = [1.0, -0.4, 0.7]

    features_a = _single_edge(
        model,
        body_i,
        body_j_a,
        edge,
    )

    features_b = _single_edge(
        model,
        body_i,
        body_j_b,
        edge,
    )

    assert not torch.allclose(
        features_a,
        features_b,
        atol=1.0e-8,
        rtol=1.0e-8,
    )


def test_pair_features_change_when_only_center_rotates():
    model = RigidPairTensorProductFeatures(
        lmax=2
    ).to(dtype=DTYPE)

    body_i_a = Rotation.identity()

    body_i_b = _axis_angle(
        [0.0, 0.0, 1.0],
        0.63,
    )

    body_j = _axis_angle(
        [0.0, 1.0, 0.0],
        -0.29,
    )

    edge = [1.0, 0.2, -0.5]

    features_a = _single_edge(
        model,
        body_i_a,
        body_j,
        edge,
    )

    features_b = _single_edge(
        model,
        body_i_b,
        body_j,
        edge,
    )

    assert not torch.allclose(
        features_a,
        features_b,
        atol=1.0e-8,
        rtol=1.0e-8,
    )


def test_quaternion_double_cover_does_not_change_features():
    model = RigidPairTensorProductFeatures(
        lmax=2
    ).to(dtype=DTYPE)

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

    reference = model(
        quaternions,
        edge_index,
        edge_vectors,
    )

    flipped = model(
        -quaternions,
        edge_index,
        edge_vectors,
    )

    torch.testing.assert_close(
        reference,
        flipped,
        atol=ATOL,
        rtol=RTOL,
    )


def test_feature_norm_is_global_rotation_invariant():
    """A simple invariant contraction of the equivariant feature is unchanged."""
    model = RigidPairTensorProductFeatures(
        lmax=2
    ).to(dtype=DTYPE)

    body_i = _axis_angle(
        [0.2, 1.0, -0.3],
        0.41,
    )

    body_j = _axis_angle(
        [1.0, -0.4, 0.6],
        -0.52,
    )

    global_rotation = _axis_angle(
        [1.0, 3.0, -2.0],
        1.01,
    )

    edge = np.asarray(
        [1.3, -0.6, 2.1],
        dtype=float,
    )

    before = _single_edge(
        model,
        body_i,
        body_j,
        edge,
    )

    after = _single_edge(
        model,
        global_rotation * body_i,
        global_rotation * body_j,
        global_rotation.apply(edge),
    )

    torch.testing.assert_close(
        torch.sum(before * before, dim=-1),
        torch.sum(after * after, dim=-1),
        atol=ATOL,
        rtol=RTOL,
    )


def test_multiple_edges_have_expected_shape():
    model = RigidPairTensorProductFeatures(
        lmax=1
    ).to(dtype=DTYPE)

    rotations = [
        Rotation.identity(),
        _axis_angle([1.0, 0.0, 0.0], 0.2),
        _axis_angle([0.0, 1.0, 0.0], -0.5),
    ]

    quaternions = torch.stack(
        [_wxyz(rotation) for rotation in rotations]
    )

    edge_index = torch.tensor(
        [
            [0, 0, 1],
            [1, 2, 2],
        ],
        dtype=torch.long,
    )

    edge_vectors = torch.tensor(
        [
            [1.0, 0.0, 0.3],
            [-0.4, 1.2, 0.7],
            [0.6, -0.3, 1.1],
        ],
        dtype=DTYPE,
    )

    output = model(
        quaternions,
        edge_index,
        edge_vectors,
    )

    assert output.shape == (
        3,
        model.irreps_out.dim,
    )


def test_output_contains_scalar_channels():
    """The rigid-pair TP already contains globally invariant contractions."""
    model = RigidPairTensorProductFeatures(
        lmax=2
    ).to(dtype=DTYPE)

    scalar_multiplicity = sum(
        mul
        for mul, irrep in model.irreps_out
        if irrep == o3.Irrep("0e")
    )

    assert scalar_multiplicity > 0
