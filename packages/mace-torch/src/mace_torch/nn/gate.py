"""The equivariant gate: a nonlinearity that leaves the higher irreps equivariant.

A pointwise activation of a vector would break its rotation, so the higher
irreps are only ever scaled, each copy by a gate: a scalar computed alongside
it and passed through a sigmoid. The scalars themselves go through a SiLU.
"""

# Gating and activation normalization inspired by e3nn.nn.Gate/Activation (MIT):
# https://github.com/e3nn/e3nn/blob/0.4.4/e3nn/nn/_gate.py
# https://github.com/e3nn/e3nn/blob/0.4.4/e3nn/nn/_activation.py
# MACE specializes this to even scalars, SiLU and sigmoid gates.

from __future__ import annotations

import torch
from mace_core.clebsch_gordan.irreps import Irreps
from mace_core.kernels.canonical import SILU_SECOND_MOMENT_SCALE
from torch import Tensor, nn

from mace_torch.backends import layout

__all__ = [
    "SIGMOID_SECOND_MOMENT_SCALE",
    "Gate",
    "ScalarActivation",
    "gate_input_irreps",
]

#: ``1 / sqrt(E[sigmoid(z)^2])`` for a standard normal ``z``, the factor that
#: gives the gate activation unit second moment. A Monte Carlo estimate, as
#: :data:`~mace_core.kernels.canonical.SILU_SECOND_MOMENT_SCALE` is, and the
#: number the frozen tree's gate multiplies by.
SIGMOID_SECOND_MOMENT_SCALE = 1.8467055342154763


class ScalarActivation(nn.Module):
    """A scalar activation with the frozen model's second-moment normalization.

    ``"None"`` is identity. Factors follow e3nn.math.normalize2mom (MIT),
    including its fixed Monte Carlo estimate for abs rather than the exact 1:
    https://github.com/e3nn/e3nn/blob/0.4.4/e3nn/math/_normalize_activation.py
    They are trained-model conventions, not recomputed at construction.
    """

    def __init__(self, gate: str = "silu") -> None:
        super().__init__()
        activations = {
            "silu": (torch.nn.functional.silu, SILU_SECOND_MOMENT_SCALE),
            "tanh": (torch.tanh, 1.5937334472592695),
            "abs": (torch.abs, 1.0011106008384667),
            "None": (nn.Identity(), 1.0),
        }
        if gate not in activations:
            raise ValueError(
                f"unknown readout gate {gate!r}; choose {tuple(activations)}"
            )
        self.activation, self.scale = activations[gate]

    def forward(self, features: Tensor) -> Tensor:
        """Preserve the shape and dtype of scalar features."""
        return self.scale * self.activation(features)


def gate_input_irreps(irreps_out: str) -> str:
    """What a gate producing ``irreps_out`` reads, grouped.

    One scalar block holding the output's scalars and then one gate per gated
    copy, followed by the gated terms in order. That is the sorted, simplified
    declaration the frozen tree's gate reads.
    """
    terms = Irreps.parse(irreps_out).terms
    scalars = sum(mul for mul, ir in terms if ir.degree == 0)
    gated = [f"{mul}x{ir}" for mul, ir in terms if ir.degree > 0]
    gates = sum(mul for mul, ir in terms if ir.degree > 0)
    return "+".join([f"{scalars + gates}x0e", *gated])


class Gate(nn.Module):
    """SiLU on the scalars, and each higher copy scaled by its sigmoid gate.

    Both activations carry their unit-second-moment factor. Input grouped as
    :func:`gate_input_irreps` declares it; output grouped as ``irreps_out``,
    the gates spent.

    Args:
        irreps_out: The output declaration, scalars first. A scalar term may
            only be even: an odd scalar through a SiLU is not odd any more.
    """

    def __init__(
        self, irreps_out: str, *, feature_layout: layout.Layout = layout.CANONICAL
    ) -> None:
        super().__init__()
        self.layout = feature_layout
        terms = Irreps.parse(irreps_out).terms
        for mul, ir in terms:
            if ir.degree == 0 and ir.parity != 1:
                raise ValueError(
                    f"irreps_out={irreps_out!r} has {mul}x{ir}; an odd scalar "
                    f"does not stay odd through a SiLU. Gate even scalars only."
                )
        self.num_scalars = sum(mul for mul, ir in terms if ir.degree == 0)
        self.gated = [(mul, ir.dimension) for mul, ir in terms if ir.degree > 0]
        self.num_gates = sum(mul for mul, _ in self.gated)

    def forward(self, features: Tensor) -> Tensor:
        """``[n, dim_in]`` grouped -> ``[n, dim_out]`` grouped."""
        scalars = SILU_SECOND_MOMENT_SCALE * torch.nn.functional.silu(
            features[:, : self.num_scalars]
        )
        if not self.num_gates:
            return scalars
        split = self.num_scalars + self.num_gates
        gates = SIGMOID_SECOND_MOMENT_SCALE * torch.sigmoid(
            features[:, self.num_scalars : split]
        )
        pieces, gate = [], 0
        blocks = self.layout.blocks(features[:, split:], tuple(self.gated))
        for block, (copies, _) in zip(blocks, self.gated, strict=True):
            pieces.append(block * gates[:, gate : gate + copies, None])
            gate += copies
        return torch.cat([scalars, self.layout.join(pieces)], dim=-1)
