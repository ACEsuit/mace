"""Inputs embedded beside the elements: a structure's charge, spin, temperature.

Each declared feature is embedded on its own, a label by a table of one row
per class and a real value by a linear map, a SiLU and a linear map. The
embeddings of every feature are joined, projected to the node width by a
linear map without bias and passed through a SiLU, and the result is added to
the element embedding before the first layer. A feature of a structure is
first given to each of its atoms.

The weights are plain matrices rather than kernel ops: a gather of rows and a
few scalar maps, which no backend accelerates. Each feature's embedding saves
and draws its own; the joint embedding holds the projection.

Every map starts uniform within ``1 / sqrt(fan_in)``, its bias too, and every
table a standard normal: torch's own starting points
for a linear map and an embedding, drawn from the module's seed.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence

import torch
from mace_core.config import (
    CategoricalFeatureConfig,
    ContinuousFeatureConfig,
    GraphFeatureConfig,
)
from mace_core.kernels.precision import Precision
from torch import Tensor, nn

from mace_torch.backends.reference import torch_dtype
from mace_torch.kernels.initialization import standard_normal_like, uniform_like

__all__ = ["GraphFeatureEmbedding"]


def _zeros(*shape: int, dtype: torch.dtype) -> nn.Parameter:
    return nn.Parameter(torch.zeros(shape, dtype=dtype))


def _uniform_within_fan_in(weights: Sequence[Tensor], fan_in: int, seed: int) -> None:
    """Every tensor of one map uniform within ``1 / sqrt(fan_in)``, in turn
    from one generator."""
    generator = torch.Generator().manual_seed(seed)
    with torch.no_grad():
        for weight in weights:
            weight.copy_(uniform_like(weight, 1.0 / math.sqrt(fan_in), generator))


class _CategoricalEmbedding(nn.Module):
    """A label, offset into a table of one row per class."""

    def __init__(self, feature: CategoricalFeatureConfig, dtype: torch.dtype) -> None:
        super().__init__()
        self.offset = feature.offset
        self.table = _zeros(feature.num_classes, feature.embedding_dim, dtype=dtype)

    def forward(self, values: Tensor) -> Tensor:
        """``[n_nodes, 1]`` labels -> ``[n_nodes, embedding_dim]``."""
        return nn.functional.embedding(
            (values.squeeze(-1) + self.offset).long(), self.table
        )

    def initialize_weights(self, seed: int) -> None:
        with torch.no_grad():
            self.table.copy_(standard_normal_like(self.table, seed))

    def to_canonical(self) -> dict[str, Tensor]:
        """``table``, ``[num_classes, embedding_dim]``."""
        return {"table": self.table.detach()}

    def load_canonical(self, state: Mapping[str, Tensor]) -> None:
        with torch.no_grad():
            self.table.copy_(state["table"])


class _ContinuousEmbedding(nn.Module):
    """A real value through a linear map, a SiLU and a linear map."""

    def __init__(self, feature: ContinuousFeatureConfig, dtype: torch.dtype) -> None:
        super().__init__()
        width = feature.embedding_dim
        self.weight_in = _zeros(width, feature.input_dim, dtype=dtype)
        self.weight_out = _zeros(width, width, dtype=dtype)
        self.bias_in = _zeros(width, dtype=dtype) if feature.bias else None
        self.bias_out = _zeros(width, dtype=dtype) if feature.bias else None

    def forward(self, values: Tensor) -> Tensor:
        """``[n_nodes, input_dim]`` -> ``[n_nodes, embedding_dim]``."""
        hidden = nn.functional.silu(
            nn.functional.linear(
                values.to(self.weight_in.dtype), self.weight_in, self.bias_in
            )
        )
        return nn.functional.linear(hidden, self.weight_out, self.bias_out)

    def initialize_weights(self, seed: int) -> None:
        for offset, (weight, bias) in enumerate(
            ((self.weight_in, self.bias_in), (self.weight_out, self.bias_out))
        ):
            tensors = [weight] if bias is None else [weight, bias]
            _uniform_within_fan_in(tensors, weight.shape[1], seed + offset)

    def to_canonical(self) -> dict[str, Tensor]:
        """``weight_in``, ``weight_out`` and, with biases, ``bias_in`` and
        ``bias_out``, each map ``[out, in]``."""
        return {
            name: tensor.detach()
            for name, tensor in self.named_parameters(recurse=False)
        }

    def load_canonical(self, state: Mapping[str, Tensor]) -> None:
        with torch.no_grad():
            for name, tensor in self.named_parameters(recurse=False):
                tensor.copy_(state[name])


class GraphFeatureEmbedding(nn.Module):
    """The joint embedding of the declared graph features.

    Args:
        features: The declarations, in the order their embeddings are joined.
        num_channels: The width of the element embedding it is added to.
        precision: The dtype of the weights and of what it returns.

    """

    def __init__(
        self,
        features: Sequence[GraphFeatureConfig],
        num_channels: int,
        precision: Precision = "float64",
    ) -> None:
        super().__init__()
        if not features:
            raise ValueError(
                "a graph-feature embedding needs at least one feature; a model "
                "without graph features builds none."
            )
        dtype = torch_dtype(precision)
        self.features = tuple(features)
        #: Each feature's embedding, in the order of ``features``.
        self.embeddings = nn.ModuleList(
            [
                _CategoricalEmbedding(feature, dtype)
                if isinstance(feature, CategoricalFeatureConfig)
                else _ContinuousEmbedding(feature, dtype)
                for feature in self.features
            ]
        )
        total = sum(feature.embedding_dim for feature in self.features)
        self.projection = _zeros(num_channels, total, dtype=dtype)

    def forward(self, graph: Mapping[str, Tensor], batch: Tensor) -> Tensor:
        """``[n_nodes, num_channels]``, what is added to the element embedding.

        Args:
            graph: The flat dict; each feature is read under its name,
                ``[n_graphs]`` or ``[n_graphs, input_dim]`` for a feature of a
                structure, ``[n_nodes]`` or ``[n_nodes, input_dim]`` for one of
                an atom.
            batch: ``[n_nodes]``, each node's structure.
        """
        num_nodes = batch.shape[0]
        rows = {"atom": num_nodes, "graph": graph["ptr"].shape[0] - 1}
        pieces = []
        for feature, embedding in zip(self.features, self.embeddings, strict=True):
            value = graph.get(feature.name)
            components = (
                1
                if isinstance(feature, CategoricalFeatureConfig)
                else feature.input_dim
            )
            expected = rows[feature.per]
            if (
                value is None
                or value.ndim == 0
                or value.shape[0] != expected
                or value.numel() != expected * components
            ):
                found = "absent" if value is None else f"{tuple(value.shape)}"
                raise ValueError(
                    f"graph feature {feature.name!r} is {components} value(s) per "
                    f"{feature.per}, {expected} row(s) in this batch, and the "
                    f"graph holds it {found}."
                )
            if feature.per == "graph":
                value = value[batch]
            pieces.append(embedding(value.reshape(num_nodes, components)))
        joined = torch.cat(pieces, dim=-1)
        return nn.functional.silu(nn.functional.linear(joined, self.projection))

    def initialize_weights(self, seed: int) -> None:
        """The projection; each feature's embedding draws its own."""
        _uniform_within_fan_in([self.projection], self.projection.shape[1], seed)

    def to_canonical(self) -> dict[str, Tensor]:
        """``projection``, ``[num_channels, total embedding_dim]``."""
        return {"projection": self.projection.detach()}

    def load_canonical(self, state: Mapping[str, Tensor]) -> None:
        with torch.no_grad():
            self.projection.copy_(state["projection"])
