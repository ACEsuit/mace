"""Declared per-node inputs, mixed into the node features.

A magnetic moment is a ``1o`` node input. A model that uses one does not get a
new block: it declares the input, and the same equivariant map that would
handle any other declared stream brings it in. Nothing here knows what a
magnetic moment is, and the word does not appear.

The mixing is a sum onto the node features rather than a concatenation, so the
feature declaration does not change with how many streams were declared. It is
equivariant because the map into the feature space is: a ``1o`` input reaches
the ``1o`` channels and cannot reach the scalars.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

from mace_core.kernels.descriptors import LinearDescriptor
from mace_core.kernels.precision import Precision
from mace_core.observables import InputSpec
from torch import Tensor, nn

from mace_torch.backends.layout import expanded_irreps, layout_of, terms

__all__ = ["NodeInputEmbedding"]


class NodeInputEmbedding(nn.Module):
    """One equivariant map per declared node input, summed into the features.

    Args:
        backend: The kernel backend. Consulted at construction only.
        specs: The per-node inputs to bring in. A per-structure input is not
            one of these: it is broadcast to nodes by the graph-feature
            embedding, which is a different module because the broadcast is
            the difference.
        hidden_irreps: One channel's node-feature declaration.
        num_features: The channel width.
        precision: The dtype the maps are built at.
    """

    def __init__(
        self,
        backend,
        specs: Sequence[InputSpec],
        hidden_irreps: str,
        num_features: int,
        precision: Precision = "float64",
    ) -> None:
        super().__init__()
        if not specs:
            raise ValueError(
                "no node inputs were declared. Leave the module out rather "
                "than building one that adds zero."
            )
        per_graph = [spec.name for spec in specs if not spec.per_atom]
        if per_graph:
            raise ValueError(
                f"{per_graph} are per-structure inputs, and this mixes "
                f"per-node ones. A per-structure input is broadcast to nodes "
                f"by the graph-feature embedding."
            )
        self.specs = list(specs)
        self.layout = layout_of(backend)
        self.input_terms = {spec.name: terms(spec.irreps) for spec in specs}
        grouped = expanded_irreps(hidden_irreps, num_features)
        self.maps = nn.ModuleDict(
            {
                spec.name: backend.make_linear(
                    LinearDescriptor(
                        irreps_in=spec.irreps,
                        irreps_out=grouped,
                        precision=precision,
                    )
                )
                for spec in specs
            }
        )

    def forward(self, graph: Mapping[str, Any], features: Tensor) -> Tensor:
        """The node features with every declared stream added.

        Args:
            graph: The flat dict, read only.
            features: ``[n_atoms, width]``, flat and grouped by irrep, in the
                backend's layout.
        """
        total = None
        for spec in self.specs:
            if spec.name not in graph:
                raise KeyError(
                    f"the node input {spec.name!r} is declared but the graph "
                    f"carries no such key. The keys present are "
                    f"{sorted(graph)}. Absence is the key being absent; a "
                    f"value of all zeros is a present input whose value is "
                    f"zero."
                )
            stream = graph[spec.name].to(features.dtype)
            stream = self.layout.from_canonical(stream, self.input_terms[spec.name])
            mapped = self.maps[spec.name](stream)
            total = mapped if total is None else total + mapped
        assert total is not None
        return features + total
