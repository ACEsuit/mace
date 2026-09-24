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
        atomic_numbers: Sequence[int],
        observables: Sequence[ObservableSpec],
        energy_head: EnergyOutputHead,
        saturation: Sequence[float],
        num_layers: int = 2,
        num_features: int = 16,
        lmax: int = 2,
        moment_lmax: int = 2,
        hidden_irreps: str = "0e+1o",
        num_radial: int = 8,
        num_moment_basis: int = 6,
        one_body_basis: int = 0,
        cutoff: float = 5.0,
        cutoff_order: int = 6,
        correlation: int = 3,
        pair_repulsion: bool = False,
        readout_hidden: int | str = 16,
        num_heads: int = 1,
        precision: Precision = "float64",
    ) -> None:
        super().__init__()
        with _constructed_in(precision):
            self.backbone = MagneticBackbone(
                backend,
                atomic_numbers=atomic_numbers,
                saturation=saturation,
                num_layers=num_layers,
                num_features=num_features,
                lmax=lmax,
                moment_lmax=moment_lmax,
                hidden_irreps=hidden_irreps,
                num_radial=num_radial,
                num_moment_basis=num_moment_basis,
                cutoff=cutoff,
                cutoff_order=cutoff_order,
                correlation=correlation,
                precision=precision,
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
            )
            self.repulsion = (
                ZBLBasis(polynomial_order=cutoff_order) if pair_repulsion else None
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
                graph["atomic_numbers"],
                graph["edge_index"],
            )
        if self.one_body is not None:
            element = self.backbone.element_index(graph["atomic_numbers"])
            squashed = self.backbone.moments.squashed_length(
                graph["magmom"].to(features[0].dtype), element
            )
            head = graph["head"][graph["batch"]] if self.outputs.num_heads > 1 else None
            term = self.one_body(squashed, element, head)
            site = term if site is None else site + term
        # The energy head adds a per-atom energy inside the scaled sum, which is
        # where both of these go.
        return self.outputs(graph, features, site)
