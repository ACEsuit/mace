"""Low-rank adapters, trained in place of the weights they adapt.

Each adapted weight becomes ``W + (alpha / rank) * B A``, with ``B`` starting at
zero so the adapted model is the original one until it trains, and only ``A``
and ``B`` receive gradients. Merging folds the update into ``W`` and removes
every trace of it, so a merged model has the shapes of one never adapted and
loads into any backend.

**The update lives in the canonical weight layout, which is what keeps it
equivariant.** An equivariant linear map has one weight per pair of copies of
the same irrep, shared by the ``2l + 1`` components of that irrep. So its
weights, grouped by irrep, are one matrix per irrep, output copies by input
copies, and a low-rank update of each of those matrices is again a map of that
shape: equivariant by construction, with no layout of its own. The frozen
tree builds the same thing as a pair of e3nn linear maps through a bottleneck
of ``rank`` copies of each shared irrep (``mace/modules/lora.py:13-26``); a
bottleneck of ``rank`` copies of an irrep is exactly a rank-``rank`` factor of
that irrep's matrix.

The dense layers get the ordinary update of their matrix. They are the radial
networks here, which the frozen tree adapts too, through its wrappers of
``nn.Linear`` and of e3nn's fully connected layer.

Both are torch parametrizations, so the op's forward reads the adapted weight
without being told, and merging is removing the parametrization with the
adapted value left behind.
"""

from __future__ import annotations

import math

import torch
from mace_core.clebsch_gordan.irreps import Irreps
from mace_core.kernels.canonical import linear_weight_table
from mace_core.kernels.descriptors import LinearDescriptor
from torch import Tensor, nn
from torch.nn.utils import parametrize

__all__ = ["DenseLowRank", "EquivariantLowRank", "inject_lora", "merge_lora"]

#: The spread of the adapters' first factor at the start. Small, and it does
#: not matter much: the second factor starts at zero, so the update does too.
_INITIAL_SPREAD = 1e-3


class EquivariantLowRank(nn.Module):
    """A rank-``rank`` update of each irrep's block of a canonical linear map.

    Args:
        irreps_in: The map's input declaration.
        irreps_out: Its output declaration.
        rank: The rank of each irrep's update.
        alpha: The update is scaled by ``alpha / rank``.
        like: The weight being adapted, for its size, dtype and device.
    """

    def __init__(
        self, irreps_in: str, irreps_out: str, rank: int, alpha: float, like: Tensor
    ) -> None:
        super().__init__()
        table = linear_weight_table(irreps_in, irreps_out)
        in_irreps = [irrep for _, irrep in Irreps.parse(irreps_in).slices()]
        out_irreps = [irrep for _, irrep in Irreps.parse(irreps_out).slices()]
        self.scaling = alpha / rank
        self.first = nn.ParameterList()
        self.second = nn.ParameterList()
        self.positions: list[Tensor] = []
        for irrep in sorted({out_irreps[o] for o, _ in table}):
            outs = [o for o, found in enumerate(out_irreps) if found == irrep]
            ins = [i for i, found in enumerate(in_irreps) if found == irrep]
            index = torch.tensor([[table[(o, i)] for i in ins] for o in outs])
            self.positions.append(index)
            generator = torch.Generator().manual_seed(len(self.positions))
            self.first.append(
                nn.Parameter(
                    (
                        torch.randn(
                            rank, len(ins), generator=generator, dtype=like.dtype
                        )
                        * _INITIAL_SPREAD
                        / math.sqrt(len(ins))
                    ).to(like.device)
                )
            )
            self.second.append(
                nn.Parameter(
                    torch.zeros(len(outs), rank, dtype=like.dtype, device=like.device)
                )
            )

    def forward(self, weight: Tensor) -> Tensor:
        update = torch.zeros_like(weight)
        for index, first, second in zip(
            self.positions, self.first, self.second, strict=True
        ):
            update = update.index_add(
                0, index.reshape(-1).to(weight.device), (second @ first).reshape(-1)
            )
        return weight + self.scaling * update


class DenseLowRank(nn.Module):
    """The ordinary low-rank update of a dense ``[in, out]`` matrix."""

    def __init__(self, like: Tensor, rank: int, alpha: float) -> None:
        super().__init__()
        rows, columns = like.shape
        self.scaling = alpha / rank
        generator = torch.Generator().manual_seed(rows * 7919 + columns)
        self.first = nn.Parameter(
            (
                torch.randn(rows, rank, generator=generator, dtype=like.dtype)
                * _INITIAL_SPREAD
                / math.sqrt(rows)
            ).to(like.device)
        )
        self.second = nn.Parameter(
            torch.zeros(rank, columns, dtype=like.dtype, device=like.device)
        )

    def forward(self, weight: Tensor) -> Tensor:
        return weight + self.scaling * (self.first @ self.second)


def inject_lora(model: nn.Module, rank: int = 4, alpha: float = 1.0) -> list[str]:
    """Adapt every linear map and every radial network, and freeze the rest.

    The defaults are the frozen tree's. What is adapted is what it adapts:
    the equivariant linear maps, wherever they are, and the dense layers.

    Returns:
        The paths of the adapted weights, in module order.

    Raises:
        ValueError: If nothing could be adapted, which is a model with no
            linear map in it and an adapter that would train nothing.
    """
    from mace_torch.nn.radial_mlp import RadialMLP

    adapted: list[str] = []
    for path, module in list(model.named_modules()):
        descriptor = getattr(module, "descriptor", None)
        weight = getattr(module, "weight", None)
        # A linear map, by what its descriptor says it is. The skip connection
        # holds a descriptor too and is a tensor product, which the frozen
        # tree does not adapt either.
        if (
            isinstance(descriptor, LinearDescriptor)
            and isinstance(weight, nn.Parameter)
            and weight.numel() > 0
            and not parametrize.is_parametrized(module, "weight")
        ):
            parametrize.register_parametrization(
                module,
                "weight",
                EquivariantLowRank(
                    descriptor.irreps_in, descriptor.irreps_out, rank, alpha, weight
                ),
            )
            adapted.append(f"{path}.weight")
        elif isinstance(module, RadialMLP):
            for index, matrix in enumerate(module.weights):
                parametrize.register_parametrization(
                    module.weights, str(index), DenseLowRank(matrix, rank, alpha)
                )
                adapted.append(f"{path}.weights.{index}")
    if not adapted:
        raise ValueError(
            "no weight in the model can take an adapter, so adapting it would "
            "train nothing."
        )
    for name, parameter in model.named_parameters():
        parameter.requires_grad = ".parametrizations." in name and not name.endswith(
            ".original"
        )
    return adapted


def merge_lora(model: nn.Module) -> None:
    """Fold every adapter into its weight and remove it.

    Afterwards the model has the parameters, the names and the shapes of one
    never adapted, and every parameter trains again, which is what a merged
    model loaded to be fine-tuned further expects.
    """
    for module in list(model.modules()):
        if not parametrize.is_parametrized(module):
            continue
        parametrizations = module.get_submodule("parametrizations")
        assert isinstance(parametrizations, nn.ModuleDict)
        for name in list(parametrizations.keys()):
            parametrize.remove_parametrizations(module, name, leave_parametrized=True)
    for parameter in model.parameters():
        parameter.requires_grad = True
