#!/usr/bin/env python
import argparse
import pprint
import torch
import mace  # import mace first to avoid torch 2.6 weights_only pitfalls

# Delay heavy imports until after adjusting torch.load globals
o3 = None
interaction_classes = None
PolarMACE = None


def build_fallback_config():
    num_elements = 2
    cfg = dict(
        r_max=4.0,
        num_bessel=4,
        num_polynomial_cutoff=3,
        max_ell=1,
        interaction_cls=interaction_classes["RealAgnosticResidualNonLinearInteractionBlock"],
        interaction_cls_first=interaction_classes["RealAgnosticResidualNonLinearInteractionBlock"],
        num_interactions=2,
        num_elements=num_elements,
        hidden_irreps=o3.Irreps("4x0e + 4x1o"),
        MLP_irreps=o3.Irreps("8x0e + 8x1o"),
        atomic_energies=torch.zeros(num_elements),
        avg_num_neighbors=3.0,
        atomic_numbers=[1, 8],
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
        fixedpoint_update_config={
            "type": "AgnosticEmbeddedOneBodyVariableUpdate",
            "potential_embedding_cls": "AgnosticChargeBiasedLinearPotentialEmbedding",
            "nonlinearity_cls": "MLPNonLinearity",
        },
        field_readout_config={"type": "OneBodyMLPFieldReadout"},
    )
    return cfg


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("path", help="Path to a saved model .pt file")
    args = ap.parse_args()

    # Allow e3nn constants to be loaded under PyTorch 2.6 weights_only=True default
    try:
        from torch.serialization import add_safe_globals
        add_safe_globals([slice])
    except Exception:
        pass

    # Defer imports until globals are set
    global o3, interaction_classes, PolarMACE
    from e3nn import o3 as _o3
    from mace.modules import interaction_classes as _ic
    from mace.modules.extensions import PolarMACE as _FFM
    o3 = _o3
    interaction_classes = _ic
    PolarMACE = _FFM

    # Create compatibility alias to satisfy checkpoints referencing previous package paths
    try:
        import types, sys
        if 'macetools' not in sys.modules:
            sys.modules['macetools'] = types.ModuleType('macetools')
        if 'macetools.electrostatics' not in sys.modules:
            sys.modules['macetools.electrostatics'] = types.ModuleType('macetools.electrostatics')
        alias_mod = types.ModuleType('macetools.electrostatics.field_fukui')
        alias_mod.PolarMACE = PolarMACE
        sys.modules['macetools.electrostatics.field_fukui'] = alias_mod
        try:
            from torch.serialization import add_safe_globals as _asg
            _asg([PolarMACE])
        except Exception:
            pass
    except Exception:
        pass

    # Prefer weights_only=True to avoid importing original training modules
    try:
        obj = torch.load(args.path, map_location="cpu", weights_only=True)
    except TypeError:
        # Torch<2.6: doesn't support weights_only kw, just do default
        obj = torch.load(args.path, map_location="cpu")
    except Exception:
        # Fallback: allow full unpickling if needed (only if you trust the source)
        obj = torch.load(args.path, map_location="cpu", weights_only=False)
    print("Loaded object keys:", list(obj.keys()) if isinstance(obj, dict) else type(obj))

    # Try to get model config
    cfg = None
    if isinstance(obj, dict):
        for key in ("config", "model_config", "model_kwargs"):
            if key in obj and isinstance(obj[key], dict):
                cfg = obj[key]
                break
    if cfg is None:
        cfg = build_fallback_config()

    print("Using config:")
    pprint.pprint({k: (str(v) if k.endswith("irreps") else v) for k, v in cfg.items() if k not in ("interaction_cls", "interaction_cls_first")})

    model = PolarMACE(**cfg)
    print("Model instantiated.")

    # Try loading state_dict heuristically
    state_dict = None
    if isinstance(obj, dict):
        for k in ("state_dict", "model_state_dict", "model", "module"):
            if k in obj and isinstance(obj[k], dict):
                state_dict = obj[k]
                break
        if state_dict is None and all(isinstance(k, str) for k in obj.keys()):
            # Might already be a raw state_dict
            state_dict = obj
    if state_dict is not None:
        missing, unexpected = model.load_state_dict(state_dict, strict=False)
        print("Loaded state_dict (strict=False)")
        print("Missing keys:", len(missing))
        print("Unexpected keys:", len(unexpected))
    else:
        print("No state_dict detected in the file. Skipped loading weights.")

    print("Done.")


if __name__ == "__main__":
    main()
