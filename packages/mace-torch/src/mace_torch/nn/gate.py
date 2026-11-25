"""The equivariant gate: a nonlinearity that leaves the higher irreps equivariant.

A pointwise activation of a vector would break its rotation, so the higher
irreps are only ever scaled, each copy by a gate: a scalar computed alongside
it and passed through an activation, a sigmoid or a SiLU. The scalars
themselves go through a SiLU.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Literal

import torch
from mace_core.clebsch_gordan.irreps import Irrep, Irreps
from mace_core.kernels.canonical import SILU_SECOND_MOMENT_SCALE, gate_irreps_problem
from torch import Tensor, nn

__all__ = [
    "SIGMOID_SECOND_MOMENT_SCALE",
    "Gate",
    "gate_input_irreps",
    "scalars_and_gated",
]

#: ``1 / sqrt(E[sigmoid(z)^2])`` for a standard normal ``z``, the factor that
#: gives the gate activation unit second moment. A Monte Carlo estimate, as
#: :data:`~mace_core.kernels.canonical.SILU_SECOND_MOMENT_SCALE` is, and the
#: number the frozen tree's gate multiplies by.
SIGMOID_SECOND_MOMENT_SCALE = 1.8467055342154763

#: What a gate can go through, with the factor that gives it unit second moment.
_GATE_ACTIVATIONS: dict[str, tuple[Callable[[Tensor], Tensor], float]] = {
    "sigmoid": (torch.sigmoid, SIGMOID_SECOND_MOMENT_SCALE),
    "silu": (torch.nn.functional.silu, SILU_SECOND_MOMENT_SCALE),
}


def scalars_and_gated(irreps_out: str) -> tuple[int, list[tuple[int, Irrep]]]:
    """The scalar copies of a gate's output, and its terms of higher degree,
    each of whose copies takes a gate."""
    terms = Irreps.parse(irreps_out).terms
    scalars = sum(mul for mul, ir in terms if ir.degree == 0)
    return scalars, [(mul, ir) for mul, ir in terms if ir.degree > 0]


def gate_input_irreps(irreps_out: str) -> str:
    """What a gate producing ``irreps_out`` reads, grouped.

    One scalar block holding the output's scalars and then one gate per gated
    copy, followed by the gated terms in order. That is the sorted, simplified
    declaration the frozen tree's gate reads.
    """
    scalars, gated = scalars_and_gated(irreps_out)
    gates = sum(mul for mul, _ in gated)
    return "+".join([f"{scalars + gates}x0e", *(f"{mul}x{ir}" for mul, ir in gated)])


class Gate(nn.Module):
    """SiLU on the scalars, and each higher copy scaled by its activated gate.

    Both activations carry their unit-second-moment factor. Input grouped as
    :func:`gate_input_irreps` declares it; output grouped as ``irreps_out``,
    the gates spent.

    Args:
        irreps_out: The output declaration, as
            :func:`~mace_core.kernels.canonical.gate_irreps_problem` admits it.
        gate_activation: What the gates go through.
    """

    def __init__(
        self, irreps_out: str, gate_activation: Literal["sigmoid", "silu"] = "sigmoid"
    ) -> None:
        super().__init__()
        problem = gate_irreps_problem(Irreps.parse(irreps_out))
        if problem:
            raise ValueError(f"irreps_out={irreps_out!r} {problem}.")
        if gate_activation not in _GATE_ACTIVATIONS:
            raise ValueError(
                f"gate_activation={gate_activation!r}; a gate goes through one "
                f"of {sorted(_GATE_ACTIVATIONS)}."
            )
        self.activation, self.activation_scale = _GATE_ACTIVATIONS[gate_activation]
        self.num_scalars, gated = scalars_and_gated(irreps_out)
        self.gated = [(mul, ir.dimension) for mul, ir in gated]
        self.num_gates = sum(mul for mul, _ in self.gated)

    def forward(self, features: Tensor) -> Tensor:
        """``[n, dim_in]`` grouped -> ``[n, dim_out]`` grouped."""
        scalars = SILU_SECOND_MOMENT_SCALE * torch.nn.functional.silu(
            features[:, : self.num_scalars]
        )
        if not self.num_gates:
            return scalars
        split = self.num_scalars + self.num_gates
        gates = self.activation_scale * self.activation(
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
