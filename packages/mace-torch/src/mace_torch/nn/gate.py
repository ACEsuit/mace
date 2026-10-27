"""The equivariant gate: a nonlinearity that leaves the higher irreps equivariant.

A pointwise activation of a vector would break its rotation, so the higher
irreps are only ever scaled, each copy by a gate: a scalar computed alongside
it and passed through a sigmoid. The scalars themselves go through a SiLU.
"""

from __future__ import annotations

import torch
from mace_core.clebsch_gordan.irreps import Irreps
from mace_core.kernels.canonical import SILU_SECOND_MOMENT_SCALE
from torch import Tensor, nn

__all__ = ["SIGMOID_SECOND_MOMENT_SCALE", "Gate", "gate_input_irreps"]

#: ``1 / sqrt(E[sigmoid(z)^2])`` for a standard normal ``z``, the factor that
#: gives the gate activation unit second moment. A Monte Carlo estimate, as
#: :data:`~mace_core.kernels.canonical.SILU_SECOND_MOMENT_SCALE` is, and the
#: number the frozen tree's gate multiplies by.
SIGMOID_SECOND_MOMENT_SCALE = 1.8467055342154763


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

    def __init__(self, irreps_out: str) -> None:
        super().__init__()
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
        pieces = [scalars]
        start, gate = split, 0
        for copies, dimension in self.gated:
            block = features[:, start : start + copies * dimension]
            scaled = (
                block.reshape(-1, copies, dimension)
                * gates[:, gate : gate + copies, None]
            )
            pieces.append(scaled.reshape(-1, copies * dimension))
            start += copies * dimension
            gate += copies
        return torch.cat(pieces, dim=-1)
