"""Fixed-shape scalar residual maps for the native one-hot element input."""

from __future__ import annotations

import torch
from mace_core.clebsch_gordan.irreps import Irreps
from mace_core.kernels.canonical import fully_connected_tp_path_normalization
from mace_core.kernels.descriptors import FullyConnectedTPDescriptor
from torch import Tensor, nn

from mace_torch.backends.precision import boundary_precision


class GroupedScalarSkip(nn.Module):
    matrices: Tensor

    def __init__(self, matrices: Tensor) -> None:
        super().__init__()
        self.register_buffer("matrices", matrices)

    def forward(self, features: Tensor, attributes: Tensor) -> Tensor:
        nodes, channels = features.shape
        species, outputs, _ = self.matrices.shape
        if species == 1:
            return torch.nn.functional.linear(features, self.matrices[0])

        # With K nonempty species, the first row of each group needs a tile.
        # Every additional tile needs another "tile" rows. Smaller tiles bound
        # the padding on ranks with few atoms relative to the element table.
        tile = min(64, max(1, nodes // species))
        groups = min(species, nodes)
        tiles = groups + (nodes - groups) // tile
        elements, order = torch.sort(attributes.argmax(dim=1))
        labels = torch.arange(species + 1, device=features.device)
        starts = torch.searchsorted(elements, labels)
        counts = starts[1:] - starts[:-1]
        tile_ends = ((counts + tile - 1) // tile).cumsum(0)
        tile_starts = torch.cat((tile_ends.new_zeros(1), tile_ends[:-1]))
        offsets = tile_starts * tile - starts[:-1]
        sorted_slots = torch.arange(nodes, device=features.device) + offsets[elements]
        slots = torch.empty_like(order).scatter(0, order, sorted_slots)
        tile_species = torch.searchsorted(
            tile_ends, torch.arange(tiles, device=features.device), right=True
        ).clamp_max(species - 1)

        packed = features.new_zeros((tiles * tile, channels)).index_copy(
            0, slots, features
        )
        matrices = self.matrices.index_select(0, tile_species)
        result = torch.bmm(packed.view(tiles, tile, channels), matrices.transpose(1, 2))
        return result.reshape(tiles * tile, outputs).index_select(0, slots)


def lower_scalar_skips(interactions: nn.ModuleList) -> int:
    """Use canonical weights for scalar skip maps in the private export copy."""

    lowered = 0
    for layer in interactions:
        skip = getattr(layer, "skip", None)
        descriptor = getattr(skip, "descriptor", None)
        if skip is None or not isinstance(descriptor, FullyConnectedTPDescriptor):
            continue
        if (
            descriptor.accumulate_floor != descriptor.precision
            or boundary_precision(skip) is not None
        ):
            continue
        irreps = [
            Irreps.parse(v)
            for v in (
                descriptor.irreps_in1,
                descriptor.irreps_in2,
                descriptor.irreps_out,
            )
        ]
        if not all(
            len(r.terms) == 1
            and r.terms[0][1].degree == 0
            and r.terms[0][1].parity == 1
            for r in irreps
        ):
            continue
        inputs, species, outputs = (r.dimension for r in irreps)
        factor = fully_connected_tp_path_normalization(
            descriptor.irreps_in1, descriptor.irreps_in2, irreps[2].terms[0][1]
        )
        matrices = (
            skip.to_canonical()["weight"].reshape(species, outputs, inputs) * factor
        )
        layer.skip = GroupedScalarSkip(matrices)
        lowered += 1
    return lowered
