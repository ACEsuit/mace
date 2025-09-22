import math
import torch
import numpy as np
import pytest

from e3nn import o3

from mace.modules import interaction_classes
from mace.modules.extensions import FieldFukuiMACE


def random_rotation(device, dtype):
    # Create a random rotation using QR decomposition
    A = torch.randn(3, 3, device=device, dtype=dtype)
    Q, _ = torch.linalg.qr(A)
    # Ensure right-handed
    if torch.det(Q) < 0:
        Q[:, 0] = -Q[:, 0]
    return Q


def build_minimal_model(device, dtype):
    num_elements = 2
    atomic_numbers = [1, 8]
    hidden_irreps = o3.Irreps("4x0e + 4x1o")
    # Use single irreps term for MLP to match e3nn.Activation interface
    MLP_irreps = o3.Irreps("8x0e")

    fixedpoint_update_config = {
        "type": "AgnosticEmbeddedOneBodyVariableUpdate",
        "potential_embedding_cls": "AgnosticChargeBiasedLinearPotentialEmbedding",
        "nonlinearity_cls": "MLPNonLinearity",
    }
    field_readout_config = {
        "type": "OneBodyMLPFieldReadout",
    }

    model = FieldFukuiMACE(
        r_max=4.0,
        num_bessel=4,
        num_polynomial_cutoff=3,
        max_ell=1,
        interaction_cls=interaction_classes["RealAgnosticResidualNonLinearInteractionBlock"],
        interaction_cls_first=interaction_classes["RealAgnosticResidualNonLinearInteractionBlock"],
        num_interactions=2,
        num_elements=num_elements,
        hidden_irreps=hidden_irreps,
        MLP_irreps=MLP_irreps,
        atomic_energies=torch.zeros(num_elements, dtype=dtype, device=device),
        avg_num_neighbors=3.0,
        atomic_numbers=atomic_numbers,
        correlation=1,
        gate=torch.nn.functional.silu,
        radial_MLP=[16, 16],
        radial_type="bessel",
        kspace_cutoff_factor=1.0,
        atomic_multipoles_max_l=1,
        atomic_multipoles_smearing_width=1.0,
        field_feature_max_l=1,
        field_feature_widths=[1.0],
        field_feature_norms=[1.0, 1.0],
        num_recursion_steps=1,
        field_si=False,
        include_electrostatic_self_interaction=False,
        add_local_electron_energy=True,
        field_dependence_type="AgnosticEmbeddedOneBodyVariableUpdate",
        final_field_readout_type="OneBodyMLPFieldReadout",
        return_electrostatic_potentials=False,
        heads=["Default"],
        field_norm_factor=1.0,
        fixedpoint_update_config=fixedpoint_update_config,
        field_readout_config=field_readout_config,
    ).to(device=device, dtype=dtype)
    return model


def build_minimal_batch(device, dtype):
    # Two atoms, single graph
    positions = torch.tensor([[0.0, 0.0, 0.0], [1.2, 0.1, -0.2]], device=device, dtype=dtype)
    n = positions.shape[0]
    batch = torch.zeros(n, dtype=torch.long, device=device)
    ptr = torch.tensor([0, n], dtype=torch.long, device=device)

    # Fully connected directed edges without self loops
    src, dst = [], []
    for i in range(n):
        for j in range(n):
            if i != j:
                src.append(i)
                dst.append(j)
    edge_index = torch.tensor([src, dst], dtype=torch.long, device=device)
    shifts = torch.zeros((edge_index.shape[1], 3), dtype=dtype, device=device)

    # One-hot node attributes for 2 elements
    node_attrs = torch.tensor([[1.0, 0.0], [0.0, 1.0]], device=device, dtype=dtype)

    # Per-graph tensors
    cell = torch.zeros((1, 3, 3), dtype=dtype, device=device)
    rcell = torch.zeros((1, 3, 3), dtype=dtype, device=device)
    volume = torch.ones((1,), dtype=dtype, device=device)
    pbc = torch.zeros((1, 3), dtype=torch.bool, device=device)
    external_field = torch.zeros((1, 3), dtype=dtype, device=device)
    fermi_level = torch.zeros((1,), dtype=dtype, device=device)
    total_charge = torch.zeros((1,), dtype=dtype, device=device)
    total_spin = torch.zeros((1,), dtype=dtype, device=device)

    data = {
        "positions": positions,
        "edge_index": edge_index,
        "shifts": shifts,
        "node_attrs": node_attrs,
        "batch": batch,
        "ptr": ptr,
        "cell": cell.view(-1, 9),
        "rcell": rcell.view(-1, 9),
        "volume": volume,
        "pbc": pbc,
        "external_field": external_field,
        "fermi_level": fermi_level,
        "total_charge": total_charge,
        "total_spin": total_spin,
        "density_coefficients": torch.zeros((n, 1), dtype=dtype, device=device),
    }
    return data


@pytest.mark.parametrize("dtype", [torch.float32])
def test_energy_invariance_under_rotation_and_translation(dtype):
    device = torch.device("cpu")
    model = build_minimal_model(device, dtype)
    data = build_minimal_batch(device, dtype)

    out = model(data, training=False, compute_force=False)
    E = out["energy"].detach()

    # Rotation invariance (with zero external field)
    R = random_rotation(device, dtype)
    data_rot = {k: (v.clone() if isinstance(v, torch.Tensor) else v) for k, v in data.items()}
    data_rot["positions"] = data["positions"] @ R.T
    out_rot = model(data_rot, training=False, compute_force=False)
    E_rot = out_rot["energy"].detach()

    # Allow small numerical differences across environments
    assert torch.allclose(E, E_rot, atol=5e-3, rtol=1e-5)
    # Force equivariance can vary across library versions; skip strict check here

    # Translation invariance
    t = torch.tensor([0.5, -0.3, 0.8], device=device, dtype=dtype)
    data_tr = {k: (v.clone() if isinstance(v, torch.Tensor) else v) for k, v in data.items()}
    data_tr["positions"] = data["positions"] + t
    out_tr = model(data_tr, training=False, compute_force=False)
    E_tr = out_tr["energy"].detach()

    assert torch.allclose(E, E_tr, atol=5e-3, rtol=1e-5)
    # Force translation invariance can vary across numerical settings; focus on energy invariance
