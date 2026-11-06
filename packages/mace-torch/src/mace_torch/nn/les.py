"""Cartesian tensor readouts for latent Ewald multipoles."""

from __future__ import annotations

import torch
from mace_core.clebsch_gordan.irreps import Irreps
from mace_core.kernels.descriptors import LinearDescriptor
from torch import Tensor, nn


class TensorReadout(nn.Module):
    """Symmetric Cartesian tensor from scalar-weighted vector outer products.

    Adapted from LinearLesReadoutBlock and NonLinearLesReadoutBlock in
    mace/modules/blocks.py. Each parity has its own vector mixing matrix.
    """

    def __init__(
        self, backend, irreps: str, precision, *, nonlinear: bool, positive: bool
    ):
        super().__init__()
        self.linear = backend.make_linear(
            LinearDescriptor(irreps_in=irreps, irreps_out=irreps, precision=precision)
        )
        indices = {key: [] for key in ("0e", "1o", "1e")}
        for span, ir in Irreps.parse(irreps).slices():
            if str(ir) in indices:
                indices[str(ir)].extend(range(span.start, span.stop))
        self.register_buffer(
            "scalars", torch.tensor(indices["0e"], dtype=torch.long), persistent=False
        )
        self.positive = positive
        self.nonlinear = nonlinear
        self.mixing = nn.ModuleDict()
        for parity in ("1o", "1e"):
            columns = indices[parity]
            if not columns or (nonlinear and parity == "1e"):
                continue
            self.register_buffer(
                "vectors_" + parity, torch.tensor(columns), persistent=False
            )
            count = len(columns) // 3
            width = len(indices["0e"])
            self.mixing[parity] = (
                nn.Sequential(
                    nn.Linear(width, 32), nn.SiLU(), nn.Linear(32, count * count)
                )
                if nonlinear
                else nn.Linear(width, count)
            )
        if not self.mixing or not indices["0e"]:
            raise ValueError("LES tensor readouts require scalar and vector channels")

    def forward(self, features: Tensor) -> Tensor:
        values = self.linear(features)
        scalars = values.index_select(1, self.scalars)
        result = values.new_zeros((len(values), 3, 3))
        for parity, mixer in self.mixing.items():
            vectors = values.index_select(
                1, getattr(self, "vectors_" + parity)
            ).reshape(len(values), -1, 3)
            weights = mixer(scalars)
            if self.nonlinear:
                weights = weights.reshape(
                    len(values), vectors.shape[1], vectors.shape[1]
                )
                mixed = ((weights + weights.transpose(1, 2)) * 0.5) @ vectors
            else:
                if self.positive and parity == "1o":
                    weights = weights.square()
                mixed = weights.unsqueeze(-1) * vectors
            result = result + mixed.transpose(1, 2) @ vectors
        return (result + result.transpose(1, 2)) * 0.5

    def initialize_weights(self, seed: int) -> None:
        with torch.random.fork_rng(devices=[]):
            torch.random.default_generator.manual_seed(seed)
            for layer in self.mixing.modules():
                if isinstance(layer, nn.Linear):
                    layer.reset_parameters()

    def to_canonical(self):
        return self.mixing.state_dict()

    def load_canonical(self, state):
        self.mixing.load_state_dict(state)
