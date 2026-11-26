"""Live legacy models, and the v1 pieces that stand for them, for the parity
suites.

The live models share one small architecture, so a test states only what it
changes: the class, the irreps, the heads, a flag. Their weights are perturbed
so that no parameter sits at a value that hides it.
"""

import numpy as np
import torch
from e3nn import o3
from mace_core.elements import ResolvedE0s
from mace_torch.models import EnergyOutputHead

from mace import modules

#: The elements of the fixtures the parity suites run on.
ELEMENTS = (1, 6, 8)

#: Isolated-atom energies of the live models, one per element.
ATOMIC_ENERGIES = (-1.5, -5.25, -8.75)


def live_model(
    model_class,
    *,
    seed: int,
    hidden_irreps: str = "8x0e + 8x1o",
    mlp_irreps: str = "8x0e",
    correlation: int = 2,
    atomic_energies=ATOMIC_ENERGIES,
    **extra,
):
    """A float64 legacy model of the shared small architecture.

    Two layers, the plain interaction first and the residual one after, a
    cutoff of 3.5 A, eight Bessel functions. ``extra`` reaches the class's
    constructor as it is.
    """
    torch.manual_seed(seed)
    return model_class(
        r_max=3.5,
        num_bessel=8,
        num_polynomial_cutoff=5,
        max_ell=2,
        interaction_cls=modules.interaction_classes[
            "RealAgnosticResidualInteractionBlock"
        ],
        interaction_cls_first=modules.interaction_classes[
            "RealAgnosticInteractionBlock"
        ],
        num_interactions=2,
        num_elements=len(ELEMENTS),
        hidden_irreps=o3.Irreps(hidden_irreps),
        MLP_irreps=o3.Irreps(mlp_irreps),
        gate=torch.nn.functional.silu,
        atomic_energies=np.asarray(atomic_energies),
        avg_num_neighbors=3.0,
        atomic_numbers=list(ELEMENTS),
        correlation=correlation,
        **extra,
    ).double()


def perturb(model, seed: int, size: float = 0.1):
    """``model`` with every parameter moved by a seeded normal of ``size``,
    so that none sits at the value it was initialized to."""
    generator = torch.Generator().manual_seed(seed)
    with torch.no_grad():
        for parameter in model.parameters():
            parameter.add_(size * torch.randn(parameter.shape, generator=generator))
    return model


def resolved_e0s(legacy, heads) -> ResolvedE0s:
    """The legacy model's isolated-atom energies, one row per head."""
    table = torch.atleast_2d(legacy.atomic_energies_fn.atomic_energies).double()
    atomic_numbers = [int(z) for z in legacy.atomic_numbers]
    return ResolvedE0s(
        values={
            head: dict(zip(atomic_numbers, row.tolist(), strict=True))
            for head, row in zip(heads, table, strict=True)
        },
        provenance=dict.fromkeys(heads, "table"),
    )


def energy_head_from(legacy, precision: str = "float64") -> EnergyOutputHead:
    """The v1 energy head holding the legacy model's constants.

    A `ScaleShiftMACE` places the pair repulsion inside its scale and shift
    and accumulates the per-atom energies in float64; a plain `MACE` has no
    scale or shift of its own and keeps its own dtype.
    """
    heads = list(legacy.heads)
    scale_shift = hasattr(legacy, "scale_shift")
    if scale_shift:
        scale = _per_head(legacy.scale_shift.scale, len(heads))
        shift = _per_head(legacy.scale_shift.shift, len(heads))
    else:
        scale, shift = [1.0] * len(heads), [0.0] * len(heads)
    return EnergyOutputHead(
        resolved_e0s(legacy, heads),
        heads,
        [int(z) for z in legacy.atomic_numbers],
        scale,
        shift,
        zbl_in_scale_shift=scale_shift,
        precision=precision,
        node_energy_dtype="float64" if scale_shift else precision,
    )


def _per_head(value, num_heads: int) -> list[float]:
    values = torch.atleast_1d(value).double().tolist()
    return values * num_heads if len(values) == 1 else values
