"""Standalone rigid-body pair geometry diagnostics.

This module is deliberately not wired into MACE interactions.

For an ordered pair i -> j, the geometry is represented in the body frame
of particle i by

    distance              r_ij
    body_i_direction      R_i^T rhat_ij
    relative_rotation     R_i^T R_j

These quantities are invariant under a common global rotation

    R_i -> S R_i
    R_j -> S R_j
    r_ij -> S r_ij

and, for labeled rigid bodies, retain the complete relative pair geometry
before quotienting by molecular point-group symmetries.
"""

from __future__ import annotations

from typing import Dict, List

import torch
from e3nn import o3

from mace.data.rigid_body import quaternion_to_matrix


def rigid_pair_geometry(
    quaternions: torch.Tensor,
    edge_index: torch.Tensor,
    edge_vectors: torch.Tensor,
    eps: float = 1.0e-12,
) -> Dict[str, torch.Tensor]:
    """Construct rigid-pair geometry for directed graph edges.

    Parameters
    ----------
    quaternions
        Shape ``(num_nodes, 4)`` in MACE scalar-first ``[w, x, y, z]``
        convention.
    edge_index
        Shape ``(2, num_edges)`` with center/source indices in row 0 and
        neighbor/target indices in row 1, matching the convention used by
        the existing rigid edge-invariant code.
    edge_vectors
        Shape ``(num_edges, 3)``. Vector from body i to body j.

    Returns
    -------
    dict
        ``distance``:
            ``(num_edges,)``

        ``space_direction``:
            normalized edge vectors in the lab frame, ``(num_edges, 3)``

        ``body_i_direction``:
            edge direction expressed in body-i coordinates,
            ``R_i^T rhat_ij``.

        ``body_j_direction``:
            edge direction expressed in body-j coordinates,
            ``R_j^T rhat_ij``.

        ``relative_rotation``:
            ``R_i^T R_j``, shape ``(num_edges, 3, 3)``.
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

    rotations = quaternion_to_matrix(quaternions)

    i = edge_index[0]
    j = edge_index[1]

    Ri = rotations[i]
    Rj = rotations[j]

    distance = torch.linalg.vector_norm(
        edge_vectors,
        dim=-1,
    )

    if torch.any(distance <= eps):
        raise ValueError(
            "rigid_pair_geometry does not support zero-length edges"
        )

    space_direction = edge_vectors / distance.unsqueeze(-1)

    # Column-vector convention:
    #
    #     u_body = R^T u_space
    #
    # In batched row-vector PyTorch notation this is u_space @ R.
    body_i_direction = torch.einsum(
        "ei,eij->ej",
        space_direction,
        Ri,
    )

    body_j_direction = torch.einsum(
        "ei,eij->ej",
        space_direction,
        Rj,
    )

    relative_rotation = Ri.transpose(-1, -2) @ Rj

    return {
        "distance": distance,
        "space_direction": space_direction,
        "body_i_direction": body_i_direction,
        "body_j_direction": body_j_direction,
        "relative_rotation": relative_rotation,
    }


def rigid_pair_harmonics(
    geometry: Dict[str, torch.Tensor],
    lmax: int = 2,
    jmax: int = 2,
    normalize: bool = True,
) -> Dict[str, object]:
    """Expand rigid-pair geometry in spherical harmonics and Wigner-D matrices.

    This is a diagnostic basis, not yet a MACE interaction feature.

    ``body_i_direction`` and ``body_j_direction`` are expanded using
    spherical harmonics. ``relative_rotation`` is expanded using the
    integer-l irreducible representations of SO(3).

    The raw geometry returned by :func:`rigid_pair_geometry` should remain
    the primary object for completeness diagnostics.
    """
    if lmax < 0:
        raise ValueError("lmax must be >= 0")
    if jmax < 0:
        raise ValueError("jmax must be >= 0")

    normalization = "component" if normalize else "norm"

    body_i_harmonics = o3.spherical_harmonics(
        list(range(lmax + 1)),
        geometry["body_i_direction"],
        normalize=True,
        normalization=normalization,
    )

    body_j_harmonics = o3.spherical_harmonics(
        list(range(lmax + 1)),
        geometry["body_j_direction"],
        normalize=True,
        normalization=normalization,
    )

    relative_rotation = geometry["relative_rotation"]

    wigner_D: List[torch.Tensor] = []
    for ell in range(jmax + 1):
        irrep = o3.Irrep(ell, 1)
        wigner_D.append(
            irrep.D_from_matrix(relative_rotation)
        )

    return {
        "body_i_harmonics": body_i_harmonics,
        "body_j_harmonics": body_j_harmonics,
        "wigner_D": wigner_D,
    }
