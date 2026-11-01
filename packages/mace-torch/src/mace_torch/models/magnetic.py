"""An energy model that reads a magnetic moment on every atom.

The moments are an input, not an output: the model reports an energy, and the
derivative engine takes forces from it against the positions and ``magforces``
against the moments, ``-dE/dm``. Relaxing the moments to where that derivative
vanishes is a fixed point around the engine, not a second model.

What differs from the standard energy model is the backbone and one more site
energy. The readouts, the isolated-atom energies, the scale and shift and the
pair repulsion are the standard ones, and so is where the repulsion goes: into
the scaled sum, as in the frozen tree's scale-shift models. The one-body term
of the moment's length goes there too.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

from mace_core.config.model import ModelConfig
from mace_core.kernels.precision import Precision
from mace_core.observables import ObservableSpec
from mace_core.outputs import MACEOutput
from torch import Tensor, nn

from mace_torch.models.base import _constructed_in
from mace_torch.models.energy import EnergyOutputHead
from mace_torch.models.outputs import ENERGY_EXTRA_ROWS, MACEOutputs
from mace_torch.nn.magnetic import MagneticBackbone, OneBodyMomentEnergy
from mace_torch.nn.radial import ZBLBasis

__all__ = ["MAGNETIC_GRAPH_INPUTS", "MagneticModel"]

#: What the magnetic model reads from a structure beside its geometry: the
#: moments, from the ``magmom`` property, and none on an atom that has none.
MAGNETIC_GRAPH_INPUTS: tuple[str, ...] = ("magmom",)


class MagneticModel(nn.Module):
    """The magnetic backbone, the energy readouts, and two extra site energies.

    Args:
        backend: The kernel backend. Consulted at construction only.
        atomic_numbers: The element table, ascending.
        observables: What the model reads out. The energy, and only the
            energy: the moments are what it reads in.
        energy_head: The isolated-atom energies and the scale and shift.
        saturation: One moment saturation per element, in muB.
        one_body_basis: How many polynomials the one-body term of the moment
            length has, the constant included. Zero for no such term.
        pair_repulsion: Whether to add the short-range repulsion.
        readout_hidden: The last readout's middle, as ``MLP_irreps``.
        num_heads: How many heads.
        The rest are :class:`~mace_torch.nn.magnetic.MagneticBackbone`'s.
    """

    def __init__(
        self,
        backend,
        config: ModelConfig,
        atomic_numbers: Sequence[int],
        observables: Sequence[ObservableSpec],
        energy_head: EnergyOutputHead,
        saturation: Sequence[float],
        *,
        num_heads: int = 1,
        precision: Precision = "float64",
    ) -> None:
        super().__init__()
        self.config = config
        one_body_basis = (
            config.magnetic.one_body_basis if config.magnetic.one_body else 0
        )
        with _constructed_in(precision):
            self.backbone = MagneticBackbone(
                backend,
                atomic_numbers=atomic_numbers,
                saturation=saturation,
                num_layers=config.num_interactions,
                num_features=config.num_channels,
                lmax=config.max_ell,
                moment_lmax=config.magnetic.lmax,
                hidden_irreps=config.node_irreps,
                num_radial=config.radial_basis.num_basis,
                num_moment_basis=config.magnetic.num_basis,
                cutoff=config.r_max,
                cutoff_order=config.cutoff.polynomial_order,
                correlation=config.correlation,
                precision=precision,
            )
            self.outputs = MACEOutputs(
                backend,
                list(observables),
                layer_irreps=self.backbone.layer_irreps,
                num_features=config.num_channels,
                energy_head=energy_head,
                precision=precision,
                readout_irreps=config.readout.mlp_irreps,
                num_heads=num_heads,
            )
            self.repulsion = (
                ZBLBasis(polynomial_order=config.cutoff.polynomial_order)
                if config.pair_repulsion
                else None
            )
            self.one_body = (
                OneBodyMomentEnergy(
                    len(atomic_numbers), one_body_basis, num_heads, precision
                )
                if one_body_basis
                else None
            )

    @property
    def extra_rows(self) -> dict[str, str]:
        return dict(ENERGY_EXTRA_ROWS)

    def forward(self, graph: Mapping[str, Any]) -> MACEOutput[Tensor]:
        """The energy at the moments the graph carries under ``magmom``."""
        features = self.backbone(graph)
        site: Tensor | None = None
        if self.repulsion is not None:
            positions = graph["positions"]
            sender, receiver = graph["edge_index"][0], graph["edge_index"][1]
            vectors = (
                graph["vectors"]
                if "vectors" in graph
                else positions[receiver] - positions[sender] + graph["shifts"]
            )
            site = self.repulsion(
                vectors.norm(dim=-1, keepdim=True),
                graph["node_atomic_numbers"],
                graph["edge_index"],
            )
        if self.one_body is not None:
            element = self.backbone.element_index(graph["node_atomic_numbers"])
            squashed = self.backbone.moments.squashed_length(
                graph["magmom"].to(features[0].dtype), element
            )
            head = graph["head"][graph["batch"]] if self.outputs.num_heads > 1 else None
            term = self.one_body(squashed, element, head)
            site = term if site is None else site + term
        # The energy head adds a per-atom energy inside the scaled sum, which is
        # where both of these go.
        return self.outputs(graph, features, site)

    def canonical_components(self) -> dict[str, nn.Module | None]:
        """Stable checkpoint roles, independent of implementation attributes."""
        return {
            "backbone": self.backbone,
            "outputs": self.outputs,
            "repulsion": self.repulsion,
            "one_body": self.one_body,
        }
