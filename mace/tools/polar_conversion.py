"""Compatibility repair of PolarMACE whole-model artefacts for graph_longrange.

This does not convert weights or training checkpoint dictionaries. Load trusted
whole-model files with ``weights_only=False``; PolarMACE repairs them on restore.
"""

PBC_HANDLING_MODES = (
    "auto",
    "realspace",
    "pbc",
    "slab",
    "molecule_in_box",
    "mixed_periodic",
)


def validate_pbc_handling(mode: str) -> None:
    if mode not in PBC_HANDLING_MODES:
        raise ValueError(f"Unsupported pbc_handling {mode!r}; choose {PBC_HANDLING_MODES}")


def ensure_polar_compatibility(model):
    """Repair restored state without overriding an existing evaluator choice."""
    return convert_polar_model(model, getattr(model, "pbc_handling", "auto"))


def convert_polar_model(model, pbc_handling: str = "auto"):
    """Repair a legacy model in place and return it, preserving all tensors.

    Idempotent for compatible models. All self-interaction blocks, including
    nested real-space blocks, need their derived Python dimension restored.
    Dispatch is rebound on this instance through the public upstream setters.
    """
    from graph_longrange.energy import GTOElectrostaticEnergy
    from graph_longrange.features import GTOElectrostaticFeatures
    from graph_longrange.gto_utils import GTOSelfInteractionBlock

    from mace.modules.extensions import PolarMACE

    validate_pbc_handling(pbc_handling)
    if not isinstance(model, PolarMACE):
        raise TypeError("Expected a PolarMACE whole model, not a checkpoint state dict")
    blocks = (
        (model.electric_potential_descriptor, GTOElectrostaticFeatures),
        (model.coulomb_energy, GTOElectrostaticEnergy),
    )
    repairs = []
    for block, expected_type in blocks:
        if not isinstance(block, expected_type) or not hasattr(block, "set_pbc_handling"):
            raise TypeError(
                "Conversion requires graph_longrange 0.4.3 electrostatic blocks"
            )
        for module in block.modules():
            if isinstance(module, GTOSelfInteractionBlock):
                dimension = module.features_irreps.dim
                if getattr(module, "features_dim", dimension) != dimension:
                    raise ValueError("Conflicting self-interaction features_dim in model")
                repairs.append((module, dimension))
    descriptor = model.electric_potential_descriptor
    expected_permutation = descriptor._build_output_permutation(
        model.field_feature_max_l, len(model.field_feature_widths)
    )
    permutation = descriptor._buffers.get("output_permutation")
    if permutation is None or not permutation.cpu().equal(expected_permutation):
        raise ValueError("Unsupported electrostatic output_permutation buffer schema")
    for module, dimension in repairs:
        module.features_dim = dimension
    model.set_electrostatic_pbcs(pbc_handling)
    return model
