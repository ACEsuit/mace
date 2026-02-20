#!/usr/bin/env python
import argparse
import json
import os
from typing import Dict, Any

import torch


def add_safe_globals_and_aliases(PolarMACE):
    try:
        from torch.serialization import add_safe_globals

        add_safe_globals([slice, PolarMACE])
    except Exception:
        pass
    # Alias old module path to new class so pickles referencing it work
    try:
        import types, sys

        if "macetools" not in sys.modules:
            sys.modules["macetools"] = types.ModuleType("macetools")
        if "macetools.electrostatics" not in sys.modules:
            sys.modules["macetools.electrostatics"] = types.ModuleType(
                "macetools.electrostatics"
            )
        alias_mod = types.ModuleType("macetools.electrostatics.field_fukui")
        alias_mod.PolarMACE = PolarMACE
        sys.modules["macetools.electrostatics.field_fukui"] = alias_mod
    except Exception:
        pass


def load_state_dict_any(path: str) -> Dict[str, torch.Tensor]:
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
    if not isinstance(sd, dict):
        raise RuntimeError(f"Unsupported checkpoint type: {type(sd)}")
    return sd


def default_config() -> Dict[str, Any]:
    return {
        "r_max": 6.0,
        "num_bessel": 8,
        "num_polynomial_cutoff": 6,
        "max_ell": 3,
        "num_interactions": 3,
        "hidden_irreps": "512x0e + 512x1o + 512x2e",
        "MLP_irreps": "16x0e",
        "edge_irreps": "128x0e + 128x1o + 128x2e",
        "atomic_numbers": list(range(1, 84)),
        "avg_num_neighbors": 3.0,
        "correlation": 3,
        "radial_MLP": [64, 64, 64],
        "radial_type": "bessel",
        "kspace_cutoff_factor": 1.0,
        "atomic_multipoles_max_l": 1,
        "atomic_multipoles_smearing_width": 1.5,
        "field_feature_max_l": 1,
        "field_feature_widths": [1.5, 3.0],
        "field_feature_norms": [20.0, 20.0, 0.5, 0.5],
        "num_recursion_steps": 2,
        "include_electrostatic_self_interaction": True,
        "add_local_electron_energy": True,
        "gate": "silu",
        "heads": ["Default"],
        "field_norm_factor": 1.0,
        "fixedpoint_update_config": {
            "type": "AgnosticEmbeddedOneBodyVariableUpdate",
            "potential_embedding_cls": "AgnosticChargeBiasedLinearPotentialEmbedding",
            "nonlinearity_cls": "MLPNonLinearity",
        },
        "field_readout_config": {"type": "OneBodyMLPFieldReadout"},
    }


def build_model_from_cfg(cfg: Dict[str, Any]) -> torch.nn.Module:
    from e3nn import o3
    from mace.modules import interaction_classes
    from mace.modules.extensions import PolarMACE

    # Parse strings
    hidden_irreps = o3.Irreps(cfg["hidden_irreps"]) if isinstance(cfg["hidden_irreps"], str) else cfg["hidden_irreps"]
    mlp_irreps = o3.Irreps(cfg["MLP_irreps"]) if isinstance(cfg["MLP_irreps"], str) else cfg["MLP_irreps"]
    edge_irreps = (
        o3.Irreps(cfg["edge_irreps"]) if isinstance(cfg.get("edge_irreps"), str) else cfg.get("edge_irreps")
    )
    gate = torch.nn.functional.silu if cfg.get("gate", "silu") == "silu" else torch.tanh
    atomic_numbers = cfg.get("atomic_numbers")
    num_elements = len(atomic_numbers)

    model = PolarMACE(
        r_max=cfg["r_max"],
        num_bessel=cfg["num_bessel"],
        num_polynomial_cutoff=cfg["num_polynomial_cutoff"],
        max_ell=cfg["max_ell"],
        interaction_cls=interaction_classes["RealAgnosticResidualNonLinearInteractionBlock"],
        interaction_cls_first=interaction_classes["RealAgnosticResidualNonLinearInteractionBlock"],
        num_interactions=cfg["num_interactions"],
        num_elements=num_elements,
        hidden_irreps=hidden_irreps,
        MLP_irreps=mlp_irreps,
        atomic_energies=torch.zeros(num_elements),
        avg_num_neighbors=cfg["avg_num_neighbors"],
        atomic_numbers=atomic_numbers,
        correlation=cfg["correlation"],
        gate=gate,
        radial_MLP=cfg["radial_MLP"],
        radial_type=cfg["radial_type"],
        kspace_cutoff_factor=cfg["kspace_cutoff_factor"],
        atomic_multipoles_max_l=cfg["atomic_multipoles_max_l"],
        atomic_multipoles_smearing_width=cfg["atomic_multipoles_smearing_width"],
        field_feature_max_l=cfg["field_feature_max_l"],
        field_feature_widths=cfg["field_feature_widths"],
        field_feature_norms=cfg["field_feature_norms"],
        num_recursion_steps=cfg["num_recursion_steps"],
        include_electrostatic_self_interaction=cfg["include_electrostatic_self_interaction"],
        add_local_electron_energy=cfg["add_local_electron_energy"],
        field_dependence_type=cfg["fixedpoint_update_config"]["type"],
        final_field_readout_type=cfg["field_readout_config"]["type"],
        return_electrostatic_potentials=False,
        heads=cfg["heads"],
        field_norm_factor=cfg["field_norm_factor"],
        fixedpoint_update_config=cfg["fixedpoint_update_config"].copy(),
        field_readout_config=cfg["field_readout_config"].copy(),
        edge_irreps=edge_irreps,
    )
    return model


def main():
    parser = argparse.ArgumentParser(description="Convert old Polar state_dict to new model layout")
    parser.add_argument("--old", required=True, help="Path to old state_dict .pt/.model")
    parser.add_argument("--out", required=True, help="Output path for converted state_dict .pt")
    parser.add_argument("--config", help="JSON file with model config (optional)")
    parser.add_argument("--print-report", action="store_true")
    args = parser.parse_args()

    # Import mace first (helps with PyTorch 2.6 weights_only default and e3nn constants)
    import mace  # noqa: F401
    from mace.modules.extensions import PolarMACE

    add_safe_globals_and_aliases(PolarMACE)

    old_sd = load_state_dict_any(args.old)

    cfg = default_config()
    if args.config:
        with open(args.config, "r", encoding="utf-8") as f:
            user_cfg = json.load(f)
        cfg.update(user_cfg)

    model = build_model_from_cfg(cfg)
    new_sd = model.state_dict()

    matched, skipped, mismatched = 0, 0, 0
    for k in new_sd.keys():
        if k in old_sd and old_sd[k].shape == new_sd[k].shape:
            new_sd[k] = old_sd[k]
            matched += 1
        elif k in old_sd and old_sd[k].numel() == new_sd[k].numel():
            # shape differs but same numel (rare): reshape conservatively
            new_sd[k] = old_sd[k].reshape_as(new_sd[k])
            matched += 1
        elif k in old_sd:
            mismatched += 1
        else:
            skipped += 1

    torch.save(new_sd, args.out)

    if args.print_report:
        print(f"Converted state_dict saved to: {args.out}")
        print(f"Matched: {matched}, Mismatched: {mismatched}, Skipped: {skipped}")


if __name__ == "__main__":
    main()

