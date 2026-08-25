"""Standalone equivariant tensor-product features for rigid-body pairs.

This module is diagnostic only and is not wired into MACE interactions.

A rigid orientation matrix R is represented by its three body axes in the
lab frame. Each axis transforms as an l=1 polar vector under proper global
rotations, giving orientation irreps

    3x1o.

For an edge i -> j we construct

    Y_l(rhat_ij) x frame_i x frame_j

using full e3nn tensor products.

Because FullTensorProduct retains every allowed Clebsch-Gordan path, this
stage does not introduce a learned compression of the angular information.
"""

from __future__ import annotations

import torch
from e3nn import o3

from mace.data.rigid_body import quaternion_to_matrix


class RigidPairTensorProductFeatures(torch.nn.Module):
    """Full equivariant tensor-product basis for an ordered rigid pair."""

    def __init__(self, lmax: int = 2):
        super().__init__()

        if lmax < 0:
            raise ValueError("lmax must be >= 0")

        self.lmax = lmax

        self.edge_irreps = o3.Irreps.spherical_harmonics(lmax)
        self.frame_irreps = o3.Irreps("3x1o")

        # First couple positional angular information to the center frame.
        self.edge_center_tp = o3.FullTensorProduct(
            self.edge_irreps,
            self.frame_irreps,
        )

        # Then couple the neighbor frame.
        self.pair_tp = o3.FullTensorProduct(
            self.edge_center_tp.irreps_out,
            self.frame_irreps,
        )

        self.irreps_out = self.pair_tp.irreps_out

    @staticmethod
    def _frame_features(rotation_matrices: torch.Tensor) -> torch.Tensor:
        """Convert rotation matrices to 3x1o body-axis features.

        ``rotation_matrices[..., :, a]`` is body axis ``a`` expressed in
        lab coordinates.

        e3nn expects multiplicity-major layout

            [axis_0_xyz, axis_1_xyz, axis_2_xyz],

        hence the transpose before flattening.
        """
        return rotation_matrices.transpose(-1, -2).reshape(
            *rotation_matrices.shape[:-2],
            9,
        )

    def forward(
        self,
        quaternions: torch.Tensor,
        edge_index: torch.Tensor,
        edge_vectors: torch.Tensor,
    ) -> torch.Tensor:
        """Construct equivariant rigid-pair features.

        Parameters
        ----------
        quaternions
            ``(num_nodes, 4)`` scalar-first ``[w, x, y, z]`` quaternions.
        edge_index
            ``(2, num_edges)``.
        edge_vectors
            ``(num_edges, 3)``.

        Returns
        -------
        torch.Tensor
            ``(num_edges, irreps_out.dim)`` equivariant pair features.
        """
        if quaternions.ndim != 2 or quaternions.shape[-1] != 4:
            raise ValueError(
                "quaternions must have shape (num_nodes, 4)"
            )

        if edge_index.ndim != 2 or edge_index.shape[0] != 2:
            raise ValueError(
                "edge_index must have shape (2, num_edges)"
            )

        if edge_vectors.ndim != 2 or edge_vectors.shape[-1] != 3:
            raise ValueError(
                "edge_vectors must have shape (num_edges, 3)"
            )

        if edge_vectors.shape[0] != edge_index.shape[1]:
            raise ValueError(
                "edge_vectors and edge_index must contain the same "
                "number of edges"
            )

        distances = torch.linalg.vector_norm(
            edge_vectors,
            dim=-1,
        )

        if torch.any(distances <= 1.0e-12):
            raise ValueError(
                "RigidPairTensorProductFeatures does not support "
                "zero-length edges"
            )

        directions = edge_vectors / distances.unsqueeze(-1)

        edge_features = o3.spherical_harmonics(
            self.edge_irreps,
            directions,
            normalize=True,
            normalization="component",
        )

        rotations = quaternion_to_matrix(quaternions)

        frame_features = self._frame_features(rotations)

        i = edge_index[0]
        j = edge_index[1]

        frame_i = frame_features[i]
        frame_j = frame_features[j]

        edge_center = self.edge_center_tp(
            edge_features,
            frame_i,
        )

        return self.pair_tp(
            edge_center,
            frame_j,
        )


class RigidPairEdgeEmbedding(torch.nn.Module):
    """Project complete rigid-pair TP features into standard MACE edge irreps.

    The full tensor product is used as an information-rich reference
    representation, then an equivariant Linear layer compresses it back
    into the same irreps as the ordinary MACE spherical-harmonic edge
    attributes.

    This lets rigid orientation information enter existing MACE
    InteractionBlocks without modifying those blocks.
    """

    def __init__(
        self,
        lmax: int,
        edge_irreps: o3.Irreps,
    ):
        super().__init__()

        self.full_pair = RigidPairTensorProductFeatures(
            lmax=lmax,
        )

        self.edge_irreps = o3.Irreps(edge_irreps)

        self.projection = o3.Linear(
            self.full_pair.irreps_out,
            self.edge_irreps,
        )

    def forward(
        self,
        quaternions: torch.Tensor,
        edge_index: torch.Tensor,
        edge_vectors: torch.Tensor,
    ) -> torch.Tensor:
        full_features = self.full_pair(
            quaternions,
            edge_index,
            edge_vectors,
        )

        return self.projection(full_features)


def validate_rigid_pair_mode(mode: str) -> str:
    """Validate the optional rigid-pair MACE interaction mode."""
    valid = (
        "none",
        "full_frame",
    )

    if mode not in valid:
        raise ValueError(
            f"Unknown rigid_pair_mode={mode!r}; "
            f"expected one of {valid}"
        )

    return mode
