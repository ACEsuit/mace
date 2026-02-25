import os

import numpy as np
import torch
from ase import Atoms
from e3nn import o3

import mace
from mace.calculators.mace import MACECalculator
from mace.modules import interaction_classes
from mace.modules.blocks import NonLinearBiasReadoutBlock
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
        num_polynomial_cutoff=5,
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
        avg_num_neighbors=30.0,
        atomic_numbers=atomic_numbers,
        correlation=3,
        gate=torch.nn.functional.silu,
        radial_MLP=[64, 64, 64],
        radial_type="bessel",
        kspace_cutoff_factor=1.5,
        atomic_multipoles_max_l=1,
        atomic_multipoles_smearing_width=1.5,
        field_feature_max_l=1,
        field_feature_widths=[1.5, 3.0],
        field_feature_norms=[20.0, 20.0, 0.5, 0.5],
        num_recursion_steps=2,
        include_electrostatic_self_interaction=True,
        readout_cls=NonLinearBiasReadoutBlock,
        add_local_electron_energy=True,
        field_dependence_type="AgnosticEmbeddedOneBodyVariableUpdate",
        final_field_readout_type="OneBodyMLPFieldReadout",
        return_electrostatic_potentials=False,
        apply_cutoff=False,
        heads=["Default"],
        use_reduced_cg=False,
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


def test_calculator_water_energy_and_charges():
    device = "cpu"
    dtype = torch.float32
    model = build_model(device, dtype)
    # Try to load state dict (filtered) if present
    sd_path = os.path.normpath(
        os.path.join(
            "/Users/Lilyes/Documents/GitHub/mace-field/mace-fukui-spin-3L_state_dict.pt"
        )
    )
    sd = torch.load(sd_path, map_location="cpu")
    sd["scale_shift.scale"] = torch.tensor([1.0], dtype=dtype)
    sd["scale_shift.shift"] = torch.tensor([0.0], dtype=dtype)
    print("keys", sd.keys())
    sd["interactions.0.alpha"] += 20.0
    sd["interactions.1.alpha"] += 20.0
    sd["interactions.2.alpha"] += 20.0
    model.load_state_dict(sd, strict=True)
    print("model fukui", model.fukui_source_map.non_linearity.__dict__)
    # model = model.double().to(device)
    # save the model
    torch.save(
        model, "/Users/Lilyes/Documents/GitHub/mace-field/mace-fukui-spin-3L.model"
    )

    atoms = build_water_atoms()
    # print barycenter of atoms (not mass weighted)
    barycenter = np.mean(atoms.get_positions(), axis=0)
    print("barycenter", barycenter)

    calc = MACECalculator(
        models=model,
        device=device,
        model_type="FieldFukuiMACE",
        default_dtype="float64",
    )
    print("calc model", calc.models[0])

    # Change spin/charge in info and recompute
    atoms.calc = calc
    # Change spin/charge in info and recompute
    atoms.info["spin"] = 2.0
    atoms.info["charge"] = 0.0
    e_spin = atoms.get_potential_energy()
    print("results calc spin=1", calc.results)

    atoms.info["spin"] = 1.0
    atoms.info["charge"] = 1.0
    e_charge = atoms.get_potential_energy()

    print("results calc charge=1", calc.results)

    print(f"Energy with spin=1: {e_spin:.6f} eV")
    print(f"Energy with charge=1: {e_charge:.6f} eV")

    # add external field in z direction
    atoms.info["external_field"] = [1.0, 1.0, 1.0]  # V/Angstrom
    e_field = atoms.get_potential_energy()
    print("results calc field", calc.results)
    print(f"Energy with external field: {e_field:.6f} eV")
