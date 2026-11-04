"""The whole model in one piece: descriptor, repulsion, observables.

A graph goes in and the typed output comes out. Still no gradient call
anywhere: forces and stress are taken around this by the derivative engine.

It exists because three things have to meet somewhere. The backbone produces
node features, the short-range repulsion produces a per-atom energy that has
nothing to do with them, and the output layer needs both. Wiring that at every
call site is how a model ends up with the repulsion added in two different
places, which is exactly the difference between the two model classes the
frozen tree has.

This is also the class a new model subclasses. The backbone is not, and the
reason is forced: a model like ``PolarMACE`` adds an energy, and an energy does
not exist until the output layer has run. What the backbone produces is node
features and nothing else.
"""

from __future__ import annotations

from collections.abc import Iterator, Mapping, Sequence
from contextlib import contextmanager
from typing import Any

import torch
from mace_core.clebsch_gordan.irreps import Irreps
from mace_core.config.model import ModelConfig
from mace_core.kernels.precision import Precision
from mace_core.observables import InputSpec, ObservableSpec
from mace_core.outputs import MACEOutput
from torch import Tensor, nn

from mace_torch.models.energy import EnergyOutputHead
from mace_torch.models.outputs import ENERGY_EXTRA_ROWS, MACEOutputs
from mace_torch.nn.backbone import MACEBackbone
from mace_torch.nn.graph_features import FeatureSpec
from mace_torch.nn.radial import ZBLBasis

__all__ = ["MACEModel"]


@contextmanager
def _constructed_in(precision: Precision) -> Iterator[None]:
    """Build under ``precision`` as the process default, then restore it.

    The radial basis, the cutoff envelope and the repulsion create their
    constants in the process default at construction. Left to the process, a
    float64 model built in a fresh interpreter, whose default is float32,
    holds those constants rounded to float32 and computes other numbers than
    the same checkpoint built elsewhere: measured, 9.6e-7 eV on a two-atom
    fixture.
    """
    previous = torch.get_default_dtype()
    torch.set_default_dtype(getattr(torch, precision))
    try:
        yield
    finally:
        torch.set_default_dtype(previous)


class MACEModel(nn.Module):
    """Backbone, optional pair repulsion, and declared observable heads.

    ``config`` is the sole architecture description. Element/head tables and
    measured neighbour statistics are supplied separately from that recipe.
    ``last_layer_irreps`` and ``edge_axes`` are the physical conventions of a
    model family; the standard energy model keeps scalars in Cartesian order.
    """

    def __init__(
        self,
        backend,
        config: ModelConfig,
        atomic_numbers: Sequence[int],
        observables: Sequence[ObservableSpec],
        energy_head: EnergyOutputHead | None = None,
        *,
        avg_num_neighbors: float = 1.0,
        precision: Precision = "float64",
        node_inputs: Sequence[InputSpec] = (),
        num_heads: int = 1,
        last_layer_irreps: str | None = None,
        edge_axes: tuple[int, int, int] = (0, 1, 2),
        radial_precision: Precision | None = None,
    ) -> None:
        super().__init__()
        if config.pair_repulsion and energy_head is None:
            raise ValueError("pair repulsion requires an energy head")
        if last_layer_irreps is None:
            last_layer_irreps = config.node_irreps
            if energy_head is not None:
                # Energy needs scalars; additional observables must retain their
                # own irreps too, or their final readout silently disappears.
                required = {
                    irrep
                    for spec in observables
                    for _, irrep in Irreps.parse(spec.irreps).terms
                }
                last_layer_irreps = "+".join(
                    f"{multiplicity}x{irrep}"
                    for multiplicity, irrep in Irreps.parse(config.node_irreps).terms
                    if irrep in required
                )
        self.config = config
        graph_features = tuple(
            FeatureSpec(
                name=feature.name,
                kind=feature.kind,
                embedding_dim=feature.embedding_dim,
                num_classes=feature.num_classes,
                input_dim=feature.input_dim,
                level="graph" if feature.per == "graph" else "node",
                offset=feature.offset,
                use_bias=feature.use_bias,
            )
            for feature in config.graph_features
        )
        with _constructed_in(precision):
            self.backbone = MACEBackbone(
                backend,
                config,
                atomic_numbers,
                avg_num_neighbors=avg_num_neighbors,
                precision=precision,
                node_inputs=node_inputs,
                last_layer_irreps=last_layer_irreps,
                graph_features=graph_features,
                edge_axes=edge_axes,
                radial_precision=radial_precision,
            )
            self.outputs = MACEOutputs(
                backend,
                list(observables),
                layer_irreps=tuple(
                    "+".join(
                        f"{mul // config.num_channels}x{ir}"
                        for mul, ir in Irreps.parse(layer).terms
                    )
                    for layer in self.backbone.layer_irreps
                ),
                num_features=config.num_channels,
                energy_head=energy_head,
                precision=precision,
                readout_irreps=config.readout.mlp_irreps,
                gate=config.readout.gate,
                num_heads=num_heads,
                last_only=config.readout.last_only,
                biased=config.readout.bias,
                embedding_irreps=(
                    self.backbone.embedding_irreps
                    if config.readout.from_embedding
                    else None
                ),
            )
            self.repulsion = (
                ZBLBasis(polynomial_order=config.cutoff.polynomial_order)
                if config.pair_repulsion
                else None
            )

    @property
    def extra_rows(self) -> dict[str, str]:
        """What each quantity the model adds to ``extras`` has a row for.

        Declared by the model that adds it, so whatever cuts a padded batch
        back reads it here rather than keeping a list of every model's.
        """
        rows = dict(ENERGY_EXTRA_ROWS) if self.outputs.energy_head is not None else {}
        if (
            self.outputs.energy_head is not None
            and not self.outputs.energy_head.zbl_in_scale_shift
        ):
            rows["contributions"] = "graph"
        return rows

    def forward(self, graph: Mapping[str, Any]) -> MACEOutput[Tensor]:
        """The declared observables. The graph is read and never written to."""
        features = list(self.backbone(graph).layers)
        repulsion: Tensor | None = None
        if self.repulsion is not None:
            positions = graph["positions"]
            sender, receiver = graph["edge_index"][0], graph["edge_index"][1]
            vectors = positions[receiver] - positions[sender] + graph["shifts"]
            repulsion = self.repulsion(
                vectors.norm(dim=-1, keepdim=True),
                graph["node_atomic_numbers"],
                graph["edge_index"],
            )
        embedding = (
            self.backbone.embedding(graph)
            if self.outputs.embedding_readout is not None
            else None
        )
        return self.outputs(graph, features, repulsion, embedding)

    def canonical_components(self) -> dict[str, nn.Module | None]:
        """Stable checkpoint roles, independent of implementation attributes."""
        return {
            "backbone": self.backbone,
            "outputs": self.outputs,
            "repulsion": self.repulsion,
        }
