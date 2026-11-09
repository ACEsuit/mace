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

from collections.abc import Mapping, Sequence
from typing import Any

from mace_core.clebsch_gordan.irreps import Irreps
from mace_core.config.model import ModelConfig
from mace_core.kernels.precision import Precision
from mace_core.observables import InputSpec, ObservableSpec
from mace_core.outputs import MACEOutput
from torch import Tensor, nn

from mace_torch.backends.precision import constructed_in
from mace_torch.models.energy import EnergyOutputHead
from mace_torch.models.inputs import CHARGES, with_embedded_inputs
from mace_torch.models.outputs import ENERGY_EXTRA_OBSERVABLES, MACEOutputs
from mace_torch.nn.backbone import MACEBackbone
from mace_torch.nn.radial import ZBLBasis

__all__ = ["MACEModel"]


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
        with constructed_in(precision):
            self.backbone = MACEBackbone(
                backend,
                config,
                atomic_numbers,
                avg_num_neighbors=avg_num_neighbors,
                precision=precision,
                node_inputs=node_inputs,
                last_layer_irreps=last_layer_irreps,
                graph_features=config.graph_features,
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

    @staticmethod
    def declared_inputs(config: ModelConfig) -> tuple[InputSpec, ...]:
        """Physical inputs beyond geometry; joint dipoles include fixed charges."""
        return (CHARGES,) if "dipole" in config.observables else ()

    @property
    def input_specs(self) -> tuple[InputSpec, ...]:
        """The same declarations used before model construction by data readers."""
        return with_embedded_inputs(self.config, self.declared_inputs(self.config))

    @property
    def extra_observables(self) -> tuple[ObservableSpec, ...]:
        """Declarations of diagnostics emitted alongside the configured heads."""
        if self.outputs.energy_head is None:
            return ()
        specs = list(ENERGY_EXTRA_OBSERVABLES)
        if not self.outputs.energy_head.zbl_in_scale_shift:
            specs.append(
                ObservableSpec(
                    name="contributions",
                    irreps=f"{self.config.num_interactions + 2}x0e",
                    per_atom=False,
                    units="eV",
                )
            )
        return tuple(specs)

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
        output = self.outputs(graph, features, repulsion, embedding)
        return self.transform_outputs(graph, features, output)

    def transform_outputs(
        self, graph: Mapping[str, Any], layers: list[Tensor], output: MACEOutput[Tensor]
    ) -> MACEOutput[Tensor]:
        """Apply a model-specific physical coupling before energy derivatives."""
        return output

    def canonical_components(self) -> dict[str, nn.Module | None]:
        """Stable checkpoint roles, independent of implementation attributes."""
        return {
            "backbone": self.backbone,
            "outputs": self.outputs,
            "repulsion": self.repulsion,
        }
