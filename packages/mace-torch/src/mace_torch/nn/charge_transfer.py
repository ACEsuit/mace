"""Directed scalar transfers from equivariant bond features.

The architecture and 1/40 normalization follow ACEsuit/mace-scf's
NoFieldSymmetricPredictionSourceBlock at 784a510339aacc4412b61b6926ef1a92bca387c0
(MIT). Operators here use V1's canonical backend contract, not e3nn.
"""

import torch
from mace_core.config.model import ModelConfig
from mace_core.kernels.descriptors import (
    ChannelwiseTPConvDescriptor,
    LinearDescriptor,
    RadialMLPDescriptor,
)
from mace_core.kernels.precision import Precision
from torch import Tensor, nn

from mace_torch.backends.layout import expanded_irreps
from mace_torch.nn.interaction import _radial_network


class ChargeTransfer(nn.Module):
    """One directed transfer, in e, per edge; no reduction over atoms yet."""

    def __init__(self, backend, config: ModelConfig, precision: Precision):
        super().__init__()
        self.channels = config.num_channels
        node_irreps = expanded_irreps(config.node_irreps, self.channels)
        edge_irreps = "+".join(
            f"{degree}{'e' if degree % 2 == 0 else 'o'}"
            for degree in range(config.max_ell + 1)
        )
        # Only scalar paths can reach the final scalar linear. Eliminating
        # the other paths leaves the function and every relevant weight intact.
        descriptor = ChannelwiseTPConvDescriptor(
            irreps_node=config.node_irreps,
            irreps_edge=edge_irreps,
            irreps_out="0e",
            num_features=self.channels,
            precision=precision,
        )
        self.paths = len(descriptor.paths)
        linear = LinearDescriptor(
            irreps_in=node_irreps, irreps_out=node_irreps, precision=precision
        )
        self.sender = backend.make_linear(linear)
        self.receiver = backend.make_linear(linear)
        self.product = backend.make_channelwise_tp_conv(descriptor)
        self.radial = _radial_network(
            backend,
            RadialMLPDescriptor(
                channels=(
                    config.radial_basis.num_basis,
                    64,
                    64,
                    64,
                    self.paths * self.channels,
                ),
                kind="normalized_silu",
                precision=precision,
            ),
        )
        self.readout = backend.make_linear(
            LinearDescriptor(
                irreps_in=expanded_irreps(descriptor.irreps_paths, self.channels),
                irreps_out="0e",
                precision=precision,
            )
        )

    def forward(
        self, features: Tensor, attributes: Tensor, radial: Tensor, edge_index: Tensor
    ) -> Tensor:
        """Node features and edge harmonics/basis → ``[n_edges]`` transfers."""
        sender, receiver = edge_index
        bonds = self.sender(features)[sender] + self.receiver(features)[receiver]
        edge_ids = torch.arange(sender.shape[0], device=sender.device)
        messages = self.product(
            bonds,
            attributes,
            self.radial(radial).reshape(-1, self.paths, self.channels),
            edge_ids,
            edge_ids,
            sender.shape[0],
        )
        return self.readout(messages).squeeze(-1) / 40.0


class SplitChargeSource(nn.Module):
    """A conservative transfer plus higher multipoles, with no unused monopole."""

    def __init__(self, backend, config: ModelConfig, degree: int, precision: Precision):
        super().__init__()
        self.transfer = ChargeTransfer(backend, config, precision)
        self.density_map = (
            backend.make_linear(
                LinearDescriptor(
                    irreps_in=expanded_irreps(config.node_irreps, config.num_channels),
                    irreps_out="+".join(
                        f"{ell}{'e' if ell % 2 == 0 else 'o'}"
                        for ell in range(1, degree + 1)
                    ),
                    precision=precision,
                )
            )
            if degree
            else None
        )

    def forward(self, features: Tensor) -> Tensor:
        """``[n_atoms, (lmax+1)^2]`` coefficients, with a zero monopole slot."""
        monopole = features.new_zeros((features.shape[0], 1))
        if self.density_map is None:
            return monopole
        return torch.cat((monopole, self.density_map(features)), dim=-1)
