"""The output layer: heads, reductions, and the typed result.

The second of the two layers. The backbone produced node features; this turns
them into the declared observables and returns them as the typed dataclass, not
as a dict whose keys every consumer has to know by heart.

The point of the shape is what it makes cheap. Adding a new spherical-tensor
observable is a line of configuration: the spec drives the head, the reduction
follows from ``per_atom``, and the value lands either on its own field of
:class:`~mace_core.outputs.MACEOutput` or in ``extras`` under its own name. No
file in this package is edited, and no new model class appears. The frozen tree
needs a model class per readout combination, which is why three of them exist
for the dipole cases alone.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any, cast

from mace_core.kernels.descriptors import LinearDescriptor
from mace_core.kernels.precision import Precision
from mace_core.observables import ObservableSpec
from mace_core.outputs import CORE_FIELD_NAMES, FIELD_BY_OBSERVABLE, MACEOutput
from torch import Tensor, nn

from mace_torch.kernels import segment_sum
from mace_torch.models.energy import EnergyOutputHead
from mace_torch.models.heads import ObservableHead, per_head_irreps
from mace_torch.nn.layout import expanded_irreps

__all__ = ["ENERGY_EXTRA_ROWS", "ENERGY_OBSERVABLE", "MACEOutputs"]

#: The one observable whose head is not a plain readout. Its site energies go
#: through the energy head, which owns the isolated-atom energies, the scale and
#: shift, and the two-reduction structure.
ENERGY_OBSERVABLE = "energy"

#: What each quantity the energy head adds to ``extras`` has a row for.
ENERGY_EXTRA_ROWS: dict[str, str] = {
    "interaction_energy": "graph",
    "node_interaction_energy": "atom",
}


class MACEOutputs(nn.Module):
    """Every declared observable, read out and reduced.

    Args:
        backend: The kernel backend. Consulted at construction only.
        observables: The declarations. One head is built per entry.
        layer_irreps: One channel's node-feature declaration per layer, as the
            backbone reports them.
        num_features: The channel width.
        energy_head: The energy head, required when ``energy`` is declared and
            rejected when it is not.
        nonlinear: Whether each head's last-layer readout carries a gate.
        readout_irreps: That gated readout's middle, as ``MLP_irreps``; an
            integer is that many scalars.
        precision: The dtype every op is built at.
        num_heads: How many levels of theory the model reads out. Every
            observable gets one readout per head, so what the heads share is
            the backbone and nothing after it.
        last_only: Read out only the last layer; see :class:`ObservableHead`.
        biased: The biased gated readout on the last layer; see
            :class:`ObservableHead`.
        embedding_irreps: One channel's declaration of the node embedding,
            when the energy also reads it out: a linear map to one scalar per
            head, added beside the isolated-atom energies. ``None`` for no
            such readout.
    """

    def __init__(
        self,
        backend,
        observables: Sequence[ObservableSpec],
        layer_irreps,
        num_features: int,
        energy_head: EnergyOutputHead | None = None,
        nonlinear: bool = True,
        precision: Precision = "float64",
        readout_irreps: int | str = 16,
        num_heads: int = 1,
        last_only: bool = False,
        biased: bool = False,
        embedding_irreps: str | None = None,
    ) -> None:
        super().__init__()
        names = [spec.name for spec in observables]
        duplicates = sorted({name for name in names if names.count(name) > 1})
        if duplicates:
            raise ValueError(
                f"{duplicates} are declared more than once. Each observable is "
                f"declared once, and its head is built from that declaration."
            )
        wants_energy = ENERGY_OBSERVABLE in names
        if wants_energy and energy_head is None:
            raise ValueError(
                f"{ENERGY_OBSERVABLE!r} is declared but no energy head was "
                f"given. The isolated-atom energies and the scale and shift "
                f"live there, so the energy cannot be read out without it."
            )
        if energy_head is not None and not wants_energy:
            raise ValueError(
                f"an energy head was given but {ENERGY_OBSERVABLE!r} is not "
                f"declared. The declarations are {names}."
            )

        if energy_head is not None and energy_head.e0_table.shape[0] != num_heads:
            raise ValueError(
                f"the energy head carries isolated-atom energies for "
                f"{energy_head.e0_table.shape[0]} heads and the readouts are "
                f"built for {num_heads}. A head is one row of each, so the two "
                f"counts are the same number or the rows do not line up."
            )

        self.specs = list(observables)
        self.energy_head = energy_head
        self.num_heads = num_heads
        self.heads = nn.ModuleDict(
            {
                spec.name: ObservableHead(
                    backend,
                    spec,
                    layer_irreps=layer_irreps,
                    num_features=num_features,
                    nonlinear=nonlinear,
                    readout_irreps=readout_irreps,
                    precision=precision,
                    num_heads=num_heads,
                    last_only=last_only,
                    biased=biased,
                )
                for spec in observables
            }
        )
        if embedding_irreps is not None and energy_head is None:
            raise ValueError(
                "an embedding readout was asked for and no energy is declared. "
                "It reads out an energy, so declare one or leave it out."
            )
        self.embedding_readout = (
            backend.make_linear(
                LinearDescriptor(
                    irreps_in=expanded_irreps(embedding_irreps, num_features),
                    irreps_out=per_head_irreps("0e", num_heads),
                    precision=precision,
                )
            )
            if embedding_irreps is not None
            else None
        )

    def _embedding_energy(
        self, embedding: Tensor | None, node_head: Tensor | None
    ) -> Tensor | None:
        """``[n_atoms]``: each atom's own head's readout of the embedding."""
        if self.embedding_readout is None:
            return None
        if embedding is None:
            raise ValueError(
                "the model reads out an energy from the node embedding and was "
                "not given one."
            )
        values = self.embedding_readout(embedding)
        if node_head is not None:
            values = values.gather(1, node_head.unsqueeze(1))
        return values.squeeze(-1)

    def required_property_keys(self) -> tuple[str, ...]:
        """The data keys a training run has to supply, one per observable."""
        return tuple(spec.name for spec in self.specs)

    def check_data(self, available: Sequence[str]) -> None:
        """Fail now if a declared observable has no data behind it.

        Args:
            available: The property keys the dataset actually carries.

        Raises:
            KeyError: Naming both the observable and the key that is missing.
                Without this the run trains a head against nothing and reports
                a loss that looks fine, because the term is simply absent.
        """
        present = set(available)
        missing = [spec.name for spec in self.specs if spec.name not in present]
        if missing:
            raise KeyError(
                f"observable(s) {missing} are declared but the data carries no "
                f"such key. The keys available are {sorted(present)}. Either "
                f"drop the declaration or supply the property."
            )

    def forward(
        self,
        graph: Mapping[str, Any],
        features: list[Tensor],
        zbl_node_energy: Tensor | None = None,
        embedding: Tensor | None = None,
    ) -> MACEOutput[Tensor]:
        """The declared observables, typed.

        Args:
            graph: The flat dict. Read and never written to.
            features: One ``[n_atoms, channels, width]`` tensor per layer.
            zbl_node_energy: The short-range pair repulsion per atom, if the
                model has one.
            embedding: The node features before the first interaction, when
                the energy has a readout of them.
        """
        batch = graph["batch"]
        num_graphs = int(graph["num_graphs"])
        # Gathered once, since every observable's readout reads it.
        node_head = graph["head"][batch] if self.num_heads > 1 else None
        fields: dict[str, Tensor] = {}
        extras: dict[str, Tensor] = {}

        for spec in self.specs:
            head = cast(ObservableHead, self.heads[spec.name])
            if spec.name == ENERGY_OBSERVABLE:
                assert self.energy_head is not None
                terms = self.energy_head(
                    [
                        value.squeeze(-1)
                        for value in head.per_layer(features, node_head)
                    ],
                    zbl_node_energy,
                    graph["element_index"],
                    graph["head"],
                    batch,
                    num_graphs,
                    self._embedding_energy(embedding, node_head),
                )
                fields["total_energy"] = terms.total_energy
                fields["node_energies"] = terms.node_energy
                extras["interaction_energy"] = terms.interaction_energy
                extras["node_interaction_energy"] = terms.node_interaction_energy
                continue

            value = head(features, node_head)
            if not spec.per_atom:
                value = segment_sum(value, batch, num_graphs)
            field = FIELD_BY_OBSERVABLE.get(spec.name, spec.name)
            if field in CORE_FIELD_NAMES:
                fields[field] = value
            else:
                extras[spec.name] = value

        return MACEOutput(**fields, extras=extras)
