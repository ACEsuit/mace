"""Graph-level inputs, embedded and added to the node features.

Total charge, total spin, electronic temperature, Fermi level, an external
field. Each is one value per structure, so it is broadcast to that structure's
atoms before it can join them.

The frozen tree configures these through a dict of strings. Here each feature
carries a typed spec, so a misspelled kind is an error at construction with the
kinds listed, rather than a key that quietly does nothing.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any, Literal

import torch
from torch import Tensor, nn

__all__ = ["FeatureSpec", "GraphFeatureEmbedding"]

FeatureKind = Literal["categorical", "continuous"]
FeatureLevel = Literal["graph", "node"]


@dataclass(frozen=True)
class FeatureSpec:
    """One graph-level or per-node input feature.

    Attributes:
        name: The key it is read from.
        kind: ``categorical`` for a small integer label, ``continuous`` for a
            real value.
        embedding_dim: The width it is embedded to.
        num_classes: How many labels, for a categorical feature.
        input_dim: How many components, for a continuous feature. An external
            field has three.
        level: Whether the value is per structure or already per atom.
        offset: Added to a categorical value before it is used as a label, so
            a range that starts below zero indexes from zero: a total charge
            from -100 to 100 with an offset of 100 takes labels 0 to 200.
        use_bias: Whether a continuous feature's two linear maps carry biases.
    """

    name: str
    kind: FeatureKind
    embedding_dim: int
    num_classes: int = 0
    input_dim: int = 1
    level: FeatureLevel = "graph"
    offset: int = 0
    use_bias: bool = True

    def __post_init__(self) -> None:
        if self.kind == "categorical" and self.num_classes <= 0:
            raise ValueError(
                f"the categorical feature {self.name!r} needs num_classes, and "
                f"got {self.num_classes}."
            )
        if self.kind not in ("categorical", "continuous"):
            raise ValueError(
                f"{self.kind!r} is not a feature kind. The kinds are "
                f"'categorical' and 'continuous'."
            )


class GraphFeatureEmbedding(nn.Module):
    """One embedder per feature, concatenated, projected, added to the nodes.

    The result is added to the scalar channels of the node features rather than
    concatenated onto them, so the node declaration does not change with the
    number of features declared. The node features are flat and grouped by
    irrep, so their scalars are the leading ``num_features * num_scalars``
    columns.

    Args:
        specs: The features to embed.
        num_features: The channel width of the node features.
        num_scalars: How many scalar components one channel has. The sum lands
            on those and only those: adding to a higher irrep would break
            equivariance, since the added value does not rotate.
        precision: The dtype the embedders are built at.
    """

    def __init__(
        self,
        specs: Sequence[FeatureSpec],
        num_features: int,
        num_scalars: int,
        precision: str = "float64",
    ) -> None:
        super().__init__()
        if not specs:
            raise ValueError(
                "no features were declared. Leave the embedding out entirely "
                "rather than building one that adds zero."
            )
        dtype = torch.float64 if precision == "float64" else torch.float32
        self.specs = list(specs)
        self.width = num_features * num_scalars
        self.num_features = num_features
        self.num_scalars = num_scalars

        embedders: dict[str, nn.Module] = {}
        total = 0
        for spec in self.specs:
            if spec.kind == "categorical":
                embedders[spec.name] = nn.Embedding(
                    spec.num_classes, spec.embedding_dim, dtype=dtype
                )
            else:
                embedders[spec.name] = nn.Sequential(
                    nn.Linear(
                        spec.input_dim,
                        spec.embedding_dim,
                        bias=spec.use_bias,
                        dtype=dtype,
                    ),
                    nn.SiLU(),
                    nn.Linear(
                        spec.embedding_dim,
                        spec.embedding_dim,
                        bias=spec.use_bias,
                        dtype=dtype,
                    ),
                )
            total += spec.embedding_dim
        self.embedders = nn.ModuleDict(embedders)
        self.project = nn.Sequential(
            nn.Linear(total, self.width, bias=False, dtype=dtype), nn.SiLU()
        )

    def forward(self, graph: Mapping[str, Any], features: Tensor) -> Tensor:
        """The node features with the embedded inputs added to their scalars.

        Args:
            graph: The flat dict, read only.
            features: ``[n_atoms, dimension]``, flat and grouped by irrep.
        """
        return self.add_to(features, self.embed(graph, features.dtype))

    def embed(self, graph: Mapping[str, Any], dtype: torch.dtype) -> Tensor:
        """``[n_atoms, num_features * num_scalars]``: what is added."""
        batch = graph["batch"]
        pieces = []
        for spec in self.specs:
            if spec.name not in graph:
                raise KeyError(
                    f"the feature {spec.name!r} is declared but the graph "
                    f"carries no such key. The keys present are "
                    f"{sorted(graph)}."
                )
            value = graph[spec.name]
            if spec.kind == "categorical":
                embedded = self.embedders[spec.name](
                    (value + spec.offset).long().reshape(-1)
                )
            else:
                embedded = self.embedders[spec.name](
                    value.reshape(-1, spec.input_dim).to(dtype)
                )
            pieces.append(embedded if spec.level == "node" else embedded[batch])
        return self.project(torch.cat(pieces, dim=-1))

    def add_to(self, features: Tensor, addition: Tensor) -> Tensor:
        return torch.cat(
            [features[:, : self.width] + addition, features[:, self.width :]], dim=-1
        )

    def to_canonical(self) -> dict[str, Tensor]:
        """Every embedder's and the projection's parameters, by their path."""
        return {name: value.detach() for name, value in self.named_parameters()}

    def load_canonical(self, state: dict[str, Tensor]) -> None:
        with torch.no_grad():
            for name, value in self.named_parameters():
                value.copy_(state[name])

    def initialize_weights(self, seed: int) -> None:
        """A standard normal for each label's embedding, and the usual uniform
        draw within ``1 / sqrt(fan_in)`` for each linear map."""
        with torch.no_grad():
            for index, (_, module) in enumerate(self.named_modules()):
                generator = torch.Generator().manual_seed(seed + index)
                if isinstance(module, nn.Embedding):
                    draw = torch.randn(
                        module.weight.shape, generator=generator, dtype=torch.float64
                    )
                    module.weight.copy_(draw.to(module.weight.dtype))
                elif isinstance(module, nn.Linear):
                    bound = 1.0 / math.sqrt(module.in_features)
                    for value in (module.weight, module.bias):
                        if value is None:
                            continue
                        draw = torch.rand(
                            value.shape, generator=generator, dtype=torch.float64
                        )
                        value.copy_(((2 * draw - 1) * bound).to(value.dtype))
