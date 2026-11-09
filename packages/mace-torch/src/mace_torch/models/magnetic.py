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
from dataclasses import replace
from typing import Any

import torch
from mace_core.config.model import ModelConfig
from mace_core.kernels.precision import Precision
from mace_core.observables import DEFAULT_CATALOGUE, InputSpec, ObservableSpec
from mace_core.outputs import MACEOutput
from torch import Tensor, nn

from mace_torch.backends.precision import constructed_in
from mace_torch.models.energy import EnergyReadout
from mace_torch.models.outputs import ENERGY_EXTRA_OBSERVABLES, MACEOutputs
from mace_torch.nn.magnetic import MagneticBackbone, OneBodyMomentEnergy
from mace_torch.nn.radial import ZBLBasis

__all__ = ["MagneticModel"]


class MagneticModel(nn.Module):
    """The magnetic backbone, the energy readouts, and two extra site energies.

    Args:
        backend: The kernel backend. Consulted at construction only.
        atomic_numbers: The element table, ascending.
        observables: What the model reads out. The energy, and only the
            energy: the moments are what it reads in.
        energy_readout: The isolated-atom energies and the scale and shift.
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
        energy_readout: EnergyReadout,
        saturation: Sequence[float],
        *,
        num_heads: int = 1,
        precision: Precision = "float64",
    ) -> None:
        super().__init__()
        if config.magnetic.time_reversal not in {"none", "batched", "sequential"}:
            raise ValueError(
                "magnetic.time_reversal must be 'none', 'batched' or 'sequential'"
            )
        self.config = config
        one_body_basis = (
            config.magnetic.one_body_basis if config.magnetic.one_body else 0
        )
        with constructed_in(precision):
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
                energy_readout=energy_readout,
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

    @staticmethod
    def declared_inputs(config: ModelConfig) -> tuple[InputSpec, ...]:
        return (DEFAULT_CATALOGUE.input("magmom"),)

    @property
    def input_specs(self) -> tuple[InputSpec, ...]:
        from mace_torch.models.inputs import with_embedded_inputs

        return with_embedded_inputs(self.config, self.declared_inputs(self.config))

    @property
    def extra_observables(self) -> tuple[ObservableSpec, ...]:
        return ENERGY_EXTRA_OBSERVABLES

    def forward(self, graph: Mapping[str, Any]) -> MACEOutput[Tensor]:
        """The energy at the moments the graph carries under ``magmom``."""
        mode = self.config.magnetic.time_reversal
        if mode == "none":
            return self._forward(graph)
        moments = graph.get("magmom")
        if not isinstance(moments, Tensor) or moments.shape != graph["positions"].shape:
            raise ValueError("time reversal requires magmom with shape [n_atoms, 3]")
        if mode == "batched":
            result = self._forward(_opposite_moments(graph))
            # These are energies only. Differentiating their mean in the outer
            # engine supplies the minus sign for magnetic forces and all higher
            # derivatives, without a second table of derivative parities.
            assert result.total_energy is not None and result.node_energies is not None
            plus = MACEOutput(
                total_energy=result.total_energy.chunk(2)[0],
                node_energies=result.node_energies.chunk(2)[0],
                extras={key: value.chunk(2)[0] for key, value in result.extras.items()},
            )
            minus = MACEOutput(
                total_energy=result.total_energy.chunk(2)[1],
                node_energies=result.node_energies.chunk(2)[1],
                extras={key: value.chunk(2)[1] for key, value in result.extras.items()},
            )
        else:
            plus = self._forward(graph)
            minus = self._forward({**graph, "magmom": -moments})
        assert plus.total_energy is not None and minus.total_energy is not None
        assert plus.node_energies is not None and minus.node_energies is not None
        return replace(
            plus,
            total_energy=(plus.total_energy + minus.total_energy) / 2,
            node_energies=(plus.node_energies + minus.node_energies) / 2,
            extras={
                key: (value + minus.extras[key]) / 2
                for key, value in plus.extras.items()
            },
        )

    def _forward(self, graph: Mapping[str, Any]) -> MACEOutput[Tensor]:
        """The unprojected energy in eV; all derivative calls stay outside."""
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


def _opposite_moments(graph: Mapping[str, Any]) -> dict[str, Any]:
    """Two disconnected copies, with opposite moments and shared autograd leaves.

    Tensor rows follow the flat graph contract; ptr and connectivity need
    offsets, and host tuples carry one entry per structure. Input values and
    gradient flags are never mutated. Strain has already been applied by the
    derivative engine, so both copies differentiate against the original cell.
    """
    nodes = graph["positions"].shape[0]
    graphs = graph["ptr"].shape[0] - 1
    doubled = {
        name: torch.cat((value, value))
        if isinstance(value, Tensor) and value.ndim
        else value + value
        if isinstance(value, tuple)
        else value
        for name, value in graph.items()
        if name not in {"edge_index", "ptr", "batch", "magmom"}
    }
    doubled.update(
        edge_index=torch.cat((graph["edge_index"], graph["edge_index"] + nodes), dim=1),
        ptr=torch.cat((graph["ptr"], graph["ptr"][1:] + nodes)),
        batch=torch.cat((graph["batch"], graph["batch"] + graphs)),
        magmom=torch.cat((graph["magmom"], -graph["magmom"])),
    )
    return doubled
