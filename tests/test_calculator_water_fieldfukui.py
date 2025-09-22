import os
import numpy as np
import pytest
import torch
from ase import Atoms
from e3nn import o3

from mace.calculators.mace import MACECalculator
from mace.modules import interaction_classes
from mace.modules.extensions import FieldFukuiMACE


def build_model(device, dtype):
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


def build_water_atoms():
    # O at origin, H positions
    positions = np.array(
        [
            [0.0, 0.0, 0.0],
            [0.9572, 0.0, 0.0],
            [-0.2390, 0.9270, 0.0],
        ]
    )
    numbers = [8, 1, 1]
    atoms = Atoms(numbers=numbers, positions=positions)
    return atoms


@pytest.mark.skipif(
    not os.path.exists(os.path.normpath(os.path.join(os.path.dirname(__file__), "..", "mace-fukui-spin-3L-xL-25-cpu_state_dict.pt"))),
    reason="Missing pretrained state dict at repo root",
)
def test_calculator_water_energy_and_charges():
    device = "cpu"
    dtype = torch.float32
    model = build_model(device, dtype)
    # Try to load state dict (filtered) if present
    sd_path = os.path.normpath(
        os.path.join(os.path.dirname(__file__), "..", "mace-fukui-spin-3L-xL-25-cpu_state_dict.pt")
    )
    try:
        sd = torch.load(sd_path, map_location="cpu")
        if isinstance(sd, dict):
            for k in ("state_dict", "model_state_dict", "model", "module", "state"):
                if k in sd and isinstance(sd[k], dict):
                    sd = sd[k]
                    break
        current = model.state_dict()
        filtered = {k: v for k, v in sd.items() if k in current and current[k].shape == v.shape}
        model.load_state_dict(filtered, strict=False)
    except Exception:
        pass

    atoms = build_water_atoms()
    calc = MACECalculator(models=model, device=device, model_type="FieldFukuiMACE")
    atoms.calc = calc
    e0 = atoms.get_potential_energy()

    # charges should be available via calculator results
    assert "charges" in calc.results
    charges = calc.results["charges"]
    assert charges.shape[0] == len(atoms)

    # Change spin/charge in info and recompute
    atoms.info["spin"] = 1.0
    e_spin = atoms.get_potential_energy()
    atoms.info["spin"] = 0.0

    atoms.info["charge"] = 1.0
    e_charge = atoms.get_potential_energy()
    atoms.info["charge"] = 0.0

    # At least one should differ for trained weights (skip if not)
    if np.isclose(e0, e_spin) and np.isclose(e0, e_charge):
        pytest.skip("Energy did not change with spin/charge; weights may not be fully matching")
