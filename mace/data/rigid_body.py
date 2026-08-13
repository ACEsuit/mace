"""Rigid-body feature utilities for MACE.

Quaternion convention is scalar-first (w, x, y, z), matching the supplied XYZ.
The ellipsoid diameters are interpreted as full principal diameters.  With unit
mass, principal moments are Ixx=(b^2+c^2)/20 etc.  Set mass_scale if needed.
"""
from __future__ import annotations

import torch
from e3nn import o3
from e3nn.io import CartesianTensor


def quaternion_to_matrix(q: torch.Tensor, eps: float = 1.0e-12) -> torch.Tensor:
    q = q / torch.clamp(torch.linalg.vector_norm(q, dim=-1, keepdim=True), min=eps)
    w, x, y, z = q.unbind(-1)
    return torch.stack(
        (
            1 - 2 * (y*y + z*z), 2 * (x*y - z*w),     2 * (x*z + y*w),
            2 * (x*y + z*w),     1 - 2 * (x*x + z*z), 2 * (y*z - x*w),
            2 * (x*z - y*w),     2 * (y*z + x*w),     1 - 2 * (x*x + y*y),
        ),
        dim=-1,
    ).reshape(q.shape[:-1] + (3, 3))


def ellipsoid_inertia_tensor(
    quaternions: torch.Tensor,
    diameters: torch.Tensor,
    mass_scale: float | torch.Tensor = 1.0,
) -> torch.Tensor:
    """Return lab-frame inertia tensors, shape [N,3,3]."""
    a, b, c = diameters.unbind(-1)
    mass = torch.as_tensor(mass_scale, dtype=diameters.dtype, device=diameters.device)
    principal = mass * torch.stack((b*b + c*c, a*a + c*c, a*a + b*b), dim=-1) / 20.0
    rotation = quaternion_to_matrix(quaternions)
    return rotation @ torch.diag_embed(principal) @ rotation.transpose(-1, -2)


def rotate_body_tensor(
    quaternions: torch.Tensor,
    body_tensor: torch.Tensor,
) -> torch.Tensor:
    """Rotate body-frame symmetric tensors into the lab frame."""
    rotation = quaternion_to_matrix(quaternions)
    return rotation @ body_tensor @ rotation.transpose(-1, -2)


def principal_tensor_from_values(
    quaternions: torch.Tensor,
    principal_values: torch.Tensor,
) -> torch.Tensor:
    """Return lab-frame tensors from body-frame principal values."""
    return rotate_body_tensor(quaternions, torch.diag_embed(principal_values))


def steric_extent_tensor(
    quaternions: torch.Tensor,
    diameters: torch.Tensor,
) -> torch.Tensor:
    """Return lab-frame squared semi-axis extent tensors.

    diameters are full principal diameters, so the body-frame tensor is
    diag((a/2)^2, (b/2)^2, (c/2)^2).
    """
    semiaxes = 0.5 * diameters
    return principal_tensor_from_values(quaternions, semiaxes * semiaxes)


def ellipsoid_gyration_tensor(
    quaternions: torch.Tensor,
    diameters: torch.Tensor,
) -> torch.Tensor:
    """Return a uniform-ellipsoid gyration tensor fallback.

    Prefer file-provided gyration principal values when available. This fallback
    uses <x_i^2> = a_i^2 / 5 for a uniform ellipsoid with semi-axis a_i.
    """
    semiaxes = 0.5 * diameters
    return principal_tensor_from_values(quaternions, semiaxes * semiaxes / 5.0)


def quadrupole_tensor_to_irreps(
    tensor: torch.Tensor,
    eps: float = 1.0e-12,
) -> torch.Tensor:
    """Convert signed/traceless quadrupole tensors to 1x0e + 1x2e features.

    The scalar channel is the Frobenius norm. The l=2 channels come from the
    Frobenius-normalized, traceless tensor. This avoids trace normalization for
    tensors whose trace is zero or sign-indefinite.
    """
    if tensor.shape[-2:] != (3, 3):
        raise ValueError(
            "Expected quadrupole tensors with final shape (3, 3), "
            f"received {tuple(tensor.shape)}"
        )

    tensor = 0.5 * (tensor + tensor.transpose(-1, -2))
    trace = tensor.diagonal(dim1=-2, dim2=-1).sum(dim=-1)
    eye = torch.eye(3, dtype=tensor.dtype, device=tensor.device)
    traceless = tensor - trace[..., None, None] * eye / 3.0
    norm = torch.linalg.matrix_norm(traceless, ord="fro", dim=(-2, -1))
    normalized = traceless / norm.clamp_min(eps)[..., None, None]

    converter = CartesianTensor("ij=ji")
    irreps = converter.from_cartesian(normalized)
    irreps = irreps.clone()
    irreps[..., :1] = norm[..., None]
    return irreps


def cartesian_tensor_to_irreps(
    tensor: torch.Tensor,
    eps: float = 1.0e-12,
) -> torch.Tensor:
    """Convert symmetric inertia tensors to normalized 0e + 2e features.

    The input has shape ``[..., 3, 3]``. Dividing by the trace removes
    the arbitrary inertia-unit scale while retaining the complete shape
    and orientation of each ellipsoid.

    The output has shape ``[..., 6]`` and transforms as ``1x0e + 1x2e``.
    """
    if tensor.shape[-2:] != (3, 3):
        raise ValueError(
            "Expected inertia tensors with final shape (3, 3), "
            f"received {tuple(tensor.shape)}"
        )

    # Eliminate small numerical asymmetries before decomposition.
    tensor = 0.5 * (tensor + tensor.transpose(-1, -2))

    trace = tensor.diagonal(
        dim1=-2,
        dim2=-1,
    ).sum(dim=-1)

    normalized_tensor = tensor / trace.clamp_min(eps)[..., None, None]

    converter = CartesianTensor("ij=ji")
    return converter.from_cartesian(normalized_tensor)



def inertia_edge_invariants(
    inertia: torch.Tensor,
    edge_index: torch.Tensor,
    edge_vectors: torch.Tensor,
    eps: float = 1.0e-12,
) -> torch.Tensor:
    """Five invariant anisotropy features for each directed edge.

    [tr(I_i), tr(I_j), rhat.I_i.rhat, rhat.I_j.rhat,
     Frobenius(I_i, I_j)]
    """
    sender, receiver = edge_index[0], edge_index[1]
    rhat = edge_vectors / torch.clamp(
        torch.linalg.vector_norm(edge_vectors, dim=-1, keepdim=True), min=eps
    )
    Ii, Ij = inertia[sender], inertia[receiver]
    longitudinal_i = torch.einsum("ei,eij,ej->e", rhat, Ii, rhat)
    longitudinal_j = torch.einsum("ei,eij,ej->e", rhat, Ij, rhat)
    overlap = torch.einsum("eij,eij->e", Ii, Ij)
    return torch.stack(
        (Ii.diagonal(dim1=-2, dim2=-1).sum(-1),
         Ij.diagonal(dim1=-2, dim2=-1).sum(-1),
         longitudinal_i, longitudinal_j, overlap),
        dim=-1,
    )
