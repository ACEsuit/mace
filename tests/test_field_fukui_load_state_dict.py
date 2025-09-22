import os
import torch
import pytest
from e3nn import o3

from mace.modules import interaction_classes
from mace.modules.extensions import FieldFukuiMACE


def build_strict_config_model(device, dtype):
    # Configuration matching the provided training setup
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
        r_max=4.0,
        num_bessel=4,
        num_polynomial_cutoff=3,
        max_ell=3,
        interaction_cls=interaction_classes[
            "RealAgnosticResidualNonLinearInteractionBlock"
        ],
        interaction_cls_first=interaction_classes[
            "RealAgnosticResidualNonLinearInteractionBlock"
        ],
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


def load_state_dict_from_file(path):
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


@pytest.mark.parametrize(
    "fname",
    [
        "mace-fukui-spin-3L-xL-25-cpu_state_dict.pt",
    ],
)
def test_load_previous_state_dict(fname):
    path = os.path.join(os.path.dirname(__file__), "..", fname)
    path = os.path.normpath(path)
    if not os.path.exists(path):
        pytest.skip(f"Missing state_dict file: {path}")

    sd = load_state_dict_from_file(path)
    assert isinstance(sd, dict)

    model = build_strict_config_model(device=torch.device("cpu"), dtype=torch.float32)
    missing, unexpected = model.load_state_dict(sd, strict=True)
    assert missing == [] and unexpected == []
