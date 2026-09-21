"""The whole model in one piece: descriptor, repulsion, observables.

A graph goes in and the typed output comes out. Still no gradient call
anywhere: forces and stress are taken around this by the derivative engine.

It exists because three things have to meet somewhere. The backbone produces
node features, the short-range repulsion produces a per-atom energy that has
nothing to do with them, and the output layer needs both. Wiring that at every
call site is how a model ends up with the repulsion added in two different
places, which is exactly the difference between the two model classes the
frozen tree has.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

from mace_core.observables import InputSpec, ObservableSpec
from mace_core.outputs import MACEOutput
from torch import Tensor, nn

from mace_torch.models.energy import EnergyOutputHead
from mace_torch.models.outputs import MACEOutputs
from mace_torch.nn.backbone import MACEBackbone
from mace_torch.nn.radial import ZBLBasis

__all__ = ["MACEModel"]


class MACEModel(nn.Module):
    """Backbone, optional pair repulsion, and the output layer.

    Args:
        backend: The kernel backend. Consulted at construction only.
        atomic_numbers: The element table, ascending.
        observables: What the model reads out.
        energy_head: The energy head, when ``energy`` is declared.
        pair_repulsion: Whether to add the short-range repulsion.
        cutoff_order: The polynomial order of both envelopes, the radial
            basis's and the repulsion's. It comes from the model's cutoff
            setting rather than from either of them, and defaulting it where a
            trained model set something else moves the energy by 6.3e-3 eV and
            the repulsion by 0.41 eV.
        node_inputs: Declared per-node input streams.
        readout_hidden: The width of the last readout's middle.
        The rest are the backbone's.
    """

    def __init__(
        self,
        backend,
        atomic_numbers: Sequence[int],
        observables: Sequence[ObservableSpec],
        energy_head: EnergyOutputHead | None = None,
        num_layers: int = 2,
        num_features: int = 16,
        lmax: int = 2,
        hidden_irreps: str = "0e+1o",
        num_radial: int = 8,
        cutoff: float = 5.0,
        correlation: int = 3,
        avg_num_neighbors: float = 1.0,
        radial_kind: str = "bessel",
        precision: str = "float64",
        pair_repulsion: bool = False,
        cutoff_order: int = 6,
        node_inputs: Sequence[InputSpec] = (),
        readout_hidden: int = 16,
    ) -> None:
        super().__init__()
        self.backbone = MACEBackbone(
            backend,
            atomic_numbers=atomic_numbers,
            num_layers=num_layers,
            num_features=num_features,
            lmax=lmax,
            hidden_irreps=hidden_irreps,
            num_radial=num_radial,
            cutoff=cutoff,
            correlation=correlation,
            avg_num_neighbors=avg_num_neighbors,
            radial_kind=radial_kind,
            cutoff_order=cutoff_order,
            precision=precision,
            node_inputs=node_inputs,
        )
        self.outputs = MACEOutputs(
            backend,
            list(observables),
            layer_irreps=self.backbone.layer_irreps,
            num_features=num_features,
            energy_head=energy_head,
            precision=precision,
            hidden_scalars=readout_hidden,
        )
        self.repulsion = (
            ZBLBasis(polynomial_order=cutoff_order) if pair_repulsion else None
        )

    def forward(self, graph: Mapping[str, Any]) -> MACEOutput[Tensor]:
        """The declared observables. The graph is read and never written to."""
        features = self.backbone(graph)
        repulsion: Tensor | None = None
        if self.repulsion is not None:
            positions = graph["positions"]
            sender, receiver = graph["edge_index"][0], graph["edge_index"][1]
            vectors = (
                graph["vectors"]
                if "vectors" in graph
                else positions[receiver] - positions[sender] + graph["shifts"]
            )
            repulsion = self.repulsion(
                vectors.norm(dim=-1, keepdim=True),
                graph["atomic_numbers"],
                graph["edge_index"],
            )
        return self.outputs(graph, features, repulsion)
