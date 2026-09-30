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
from typing import Any, Literal

import torch
from mace_core.kernels.descriptors import RadialKind
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
        readout_hidden: The last readout's middle, per head, as
            ``MLP_irreps``; an integer is that many scalars.
        num_heads: How many levels of theory the model reads out. They share
            the backbone and nothing after it: each has its own readout, and
            the energy head carries one row of constants per head.
        last_layer_irreps: What the last layer keeps; see the backbone.
        residual_first_layer: A residual first layer; see the backbone.
        learned_density_first_layer: A learned density in the first layer;
            see the backbone.
        learned_density: A learned density in every later layer.
        nonlinear_first_layer: A nonlinear first layer; see the backbone.
        nonlinear: Nonlinear later layers.
        convolution_irreps: What a nonlinear layer convolves; see the
            backbone.
        narrow_first_convolution: Convolve only its scalars in the first
            layer.
        distance_transform: The distance transform; see the backbone.
        apply_cutoff: Whether the radial basis carries the cutoff envelope;
            see the backbone.
        radial_hidden: The radial networks' hidden widths.
        graph_features: Inputs embedded beside the elements; see the
            backbone.
        readout_last_only: Read out only the last layer.
        readout_biased: The frozen tree's biased readout on the last layer.
        readout_from_embedding: Also read an energy out of the node embedding,
            before the first interaction.
        element_agnostic_product: One set of product weights for all
            elements; see the backbone.
        edge_axes: The axis order the spherical harmonics read; see the
            backbone.
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
        radial_kind: RadialKind = "bessel",
        precision: Precision = "float64",
        pair_repulsion: bool = False,
        cutoff_order: int = 6,
        distance_transform: Literal["none", "agnesi", "soft"] = "none",
        apply_cutoff: bool = True,
        radial_hidden: Sequence[int] = (64, 64, 64),
        graph_features: Sequence[FeatureSpec] = (),
        readout_last_only: bool = False,
        readout_biased: bool = False,
        readout_from_embedding: bool = False,
        node_inputs: Sequence[InputSpec] = (),
        readout_hidden: int | str = 16,
        num_heads: int = 1,
        last_layer_irreps: str = "0e",
        residual_first_layer: bool = False,
        learned_density_first_layer: bool = False,
        learned_density: bool = False,
        nonlinear_first_layer: bool = False,
        nonlinear: bool = False,
        convolution_irreps: str | None = None,
        narrow_first_convolution: bool = False,
        element_agnostic_product: bool = False,
        edge_axes: tuple[int, int, int] = (0, 1, 2),
    ) -> None:
        super().__init__()
        with _constructed_in(precision):
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
                distance_transform=distance_transform,
                apply_cutoff=apply_cutoff,
                radial_hidden=radial_hidden,
                graph_features=graph_features,
                precision=precision,
                node_inputs=node_inputs,
                last_layer_irreps=last_layer_irreps,
                residual_first_layer=residual_first_layer,
                learned_density_first_layer=learned_density_first_layer,
                learned_density=learned_density,
                nonlinear_first_layer=nonlinear_first_layer,
                nonlinear=nonlinear,
                convolution_irreps=convolution_irreps,
                narrow_first_convolution=narrow_first_convolution,
                element_agnostic_product=element_agnostic_product,
                edge_axes=edge_axes,
            )
            self.outputs = MACEOutputs(
                backend,
                list(observables),
                layer_irreps=self.backbone.layer_irreps,
                num_features=num_features,
                energy_head=energy_head,
                precision=precision,
                readout_irreps=readout_hidden,
                num_heads=num_heads,
                last_only=readout_last_only,
                biased=readout_biased,
                embedding_irreps=(
                    self.backbone.embedding_irreps if readout_from_embedding else None
                ),
            )
            self.repulsion = (
                ZBLBasis(polynomial_order=cutoff_order) if pair_repulsion else None
            )

    @property
    def extra_rows(self) -> dict[str, str]:
        """What each quantity the model adds to ``extras`` has a row for.

        Declared by the model that adds it, so whatever cuts a padded batch
        back reads it here rather than keeping a list of every model's.
        """
        return dict(ENERGY_EXTRA_ROWS) if self.outputs.energy_head is not None else {}

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
        embedding = (
            self.backbone.embedding(graph)
            if self.outputs.embedding_readout is not None
            else None
        )
        return self.outputs(graph, features, repulsion, embedding)
