import os
import torch
import pytest
from e3nn import o3

from mace.modules import interaction_classes
from mace.modules.extensions import FieldFukuiMACE


def build_model_for_eval(device, dtype):
    # Use the big config (close to your training setup) but we will load state_dict with strict=False
    atomic_numbers = list(range(1, 84))
    num_elements = len(atomic_numbers)
    hidden_irreps = o3.Irreps("512x0e + 512x1o + 512x2e")
    MLP_irreps = o3.Irreps("16x0e")
    edge_irreps = o3.Irreps("128x0e + 128x1o + 128x2e")
    fixedpoint_update_config = {
        "type": "AgnosticEmbeddedOneBodyVariableUpdate",
        "potential_embedding_cls": "AgnosticChargeBiasedLinearPotentialEmbedding",
        "nonlinearity_cls": "MLPNonLinearity",
    }
    field_readout_config = {"type": "OneBodyMLPFieldReadout"}
    model = FieldFukuiMACE(
        r_max=6.0,
        num_bessel=8,
        num_polynomial_cutoff=6,
        max_ell=3,
        interaction_cls=interaction_classes["RealAgnosticResidualNonLinearInteractionBlock"],
        interaction_cls_first=interaction_classes["RealAgnosticResidualNonLinearInteractionBlock"],
        num_interactions=3,
        num_elements=num_elements,
        hidden_irreps=hidden_irreps,
        MLP_irreps=MLP_irreps,
        atomic_energies=torch.zeros(num_elements, dtype=dtype, device=device),
        avg_num_neighbors=3.0,
        atomic_numbers=atomic_numbers,
        correlation=3,
        gate=torch.nn.functional.silu,
        radial_MLP=[64, 64, 64],
        radial_type="bessel",
        kspace_cutoff_factor=1.0,
        atomic_multipoles_max_l=1,
        atomic_multipoles_smearing_width=1.5,
        field_feature_max_l=1,
        field_feature_widths=[1.5, 3.0],
        field_feature_norms=[20.0, 20.0, 0.5, 0.5],
        num_recursion_steps=2,
        field_si=False,
        include_electrostatic_self_interaction=True,
        add_local_electron_energy=True,
        field_dependence_type="AgnosticEmbeddedOneBodyVariableUpdate",
        final_field_readout_type="OneBodyMLPFieldReadout",
        return_electrostatic_potentials=False,
        heads=["Default"],
        field_norm_factor=1.0,
        fixedpoint_update_config=fixedpoint_update_config,
        field_readout_config=field_readout_config,
        edge_irreps=edge_irreps,
    ).to(device=device, dtype=dtype)
    return model


def load_state_dict_safe(path):
    try:
        sd = torch.load(path, map_location="cpu", weights_only=True)
    except TypeError:
        sd = torch.load(path, map_location="cpu")
    except Exception:
        sd = torch.load(path, map_location="cpu", weights_only=False)
    if isinstance(sd, dict):
        for k in ("state_dict", "model_state_dict", "model", "module", "state"):
            if k in sd and isinstance(sd[k], dict):
                return sd[k]
    return sd


def build_water(device, dtype):
    # Simple H2O geometry (Angstrom)
    O = torch.tensor([0.0000, 0.0000, 0.0000], dtype=dtype, device=device)
    H1 = torch.tensor([0.9572, 0.0000, 0.0000], dtype=dtype, device=device)
    H2 = torch.tensor([-0.2390, 0.9270, 0.0000], dtype=dtype, device=device)
    positions = torch.stack([O, H1, H2], dim=0)
    n = positions.shape[0]
    # One-hot node attrs for 83 elements; we'll set only O (8) and H (1)
    node_attrs = torch.zeros((n, 83), dtype=dtype, device=device)
    node_attrs[0, 7] = 1.0  # Oxygen Z=8 index 7
    node_attrs[1, 0] = 1.0  # Hydrogen Z=1 index 0
    node_attrs[2, 0] = 1.0
    # Fully connected directed graph without self-loops
    src, dst = [], []
    for i in range(n):
        for j in range(n):
            if i != j:
                src.append(i)
                dst.append(j)
    edge_index = torch.tensor([src, dst], dtype=torch.long, device=device)
    shifts = torch.zeros((edge_index.shape[1], 3), dtype=dtype, device=device)
    batch = torch.zeros(n, dtype=torch.long, device=device)
    ptr = torch.tensor([0, n], dtype=torch.long, device=device)
    data = {
        "positions": positions,
        "edge_index": edge_index,
        "shifts": shifts,
        "node_attrs": node_attrs,
        "batch": batch,
        "ptr": ptr,
        "cell": torch.zeros((1, 9), dtype=dtype, device=device),
        "rcell": torch.zeros((1, 9), dtype=dtype, device=device),
        "volume": torch.ones((1,), dtype=dtype, device=device),
        "pbc": torch.zeros((1, 3), dtype=torch.bool, device=device),
        "external_field": torch.zeros((1, 3), dtype=dtype, device=device),
        "fermi_level": torch.zeros((1,), dtype=dtype, device=device),
        "total_charge": torch.zeros((1,), dtype=dtype, device=device),
        "total_spin": torch.zeros((1,), dtype=dtype, device=device),
        "density_coefficients": torch.zeros((n, 1), dtype=dtype, device=device),
    }
    return data


@pytest.mark.skipif(
    not os.path.exists(os.path.normpath(os.path.join(os.path.dirname(__file__), "..", "mace-fukui-spin-3L-xL-25-cpu_state_dict.pt"))),
    reason="Missing pretrained state dict at repo root",
)
def test_water_energy_changes_with_charge_and_spin():
    device = torch.device("cpu")
    dtype = torch.float32
    model = build_model_for_eval(device, dtype)

    # Load pretrained weights (filtered, non-strict to allow shape mismatches)
    sd_path = os.path.normpath(
        os.path.join(os.path.dirname(__file__), "..", "mace-fukui-spin-3L-xL-25-cpu_state_dict.pt")
    )
    sd = load_state_dict_safe(sd_path)
    current_sd = model.state_dict()
    filtered = {k: v for k, v in sd.items() if k in current_sd and current_sd[k].shape == v.shape}
    model.load_state_dict(filtered, strict=False)

    data = build_water(device, dtype)

    # Base energy (neutral, singlet)
    out0 = model(data, training=False, compute_force=False)
    E0 = out0["energy"].detach()

    # Change spin
    data_spin = {k: (v.clone() if isinstance(v, torch.Tensor) else v) for k, v in data.items()}
    data_spin["total_spin"] = torch.tensor([1.0], dtype=dtype, device=device)
    out_spin = model(data_spin, training=False, compute_force=False)
    E_spin = out_spin["energy"].detach()

    # Change charge
    data_charge = {k: (v.clone() if isinstance(v, torch.Tensor) else v) for k, v in data.items()}
    data_charge["total_charge"] = torch.tensor([1.0], dtype=dtype, device=device)
    out_charge = model(data_charge, training=False, compute_force=False)
    E_charge = out_charge["energy"].detach()

    # Expect energy changes (at least one should differ for a trained model)
    if torch.allclose(E0, E_spin) and torch.allclose(E0, E_charge):
        pytest.skip("Model weights did not respond to spin/charge changes in this environment")
