"""The output layer: every declared observable, read out, reduced and typed.

The second of the two layers the model is split into. The backbone returns
node features; this turns them into a :class:`~mace_core.outputs.MACEOutput`.
A new observable is a new declaration and no new code: its irreps decide the
readout, its ``per_atom`` whether the result is summed over each structure,
and its name the field of the output it fills, or the entry of ``extras``. An
observable with a single scalar component is reported without a component
axis, as the energy is.

Observables read out together share one readout (``readout.groups``): the
frozen dielectric model reads its charge, dipole and polarizability out of one
map, and its weights only convert to one map.

Three things follow a readout, each chosen in ``readout``, none of them a
class of model:

* **The dipole's charge part**, ``sum_i q_i r_i``, from the structure's own
  charges or from the ``charges`` observable. Predicted charges are shifted
  first, every atom of a structure by one constant, so they sum to the
  structure's total charge.
* **A symmetric matrix** read out as its ``0e+2e`` components is reported as
  the ``3 x 3`` matrix too.
* **The per-atom values** of an observable of a structure, before the sum.

The energy is the one observable with more to it: its per-layer site energies
go through the :class:`~mace_torch.models.energy.EnergyOutputHead`. No
derivative is taken here.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import cast

import torch
from mace_core.clebsch_gordan import symmetric_matrix_basis
from mace_core.config.model import ModelConfig
from mace_core.kernels.precision import Precision
from mace_core.outputs import CORE_FIELD_NAMES, FIELD_BY_OBSERVABLE, MACEOutput
from mace_core.units import DEBYE_FROM_SI
from torch import Tensor, nn

from mace_torch.backends.reference import torch_dtype
from mace_torch.kernels import segment_sum
from mace_torch.models.energy import EnergyOutputHead
from mace_torch.models.readouts import ObservableReadout
from mace_torch.nn.backbone import NodeFeatures

__all__ = ["MACEOutputs"]

#: The observable the energy head reads out.
ENERGY = "energy"


class MACEOutputs(nn.Module):
    """Every declared observable, read out of the backbone's node features.

    Args:
        backend: The kernel backend. Consulted at construction only.
        config: The model's configuration: its ``observables`` and its
            ``readout``, which it has checked against each other.
        layer_irreps: Each layer's node-feature declaration, as the backbone
            reports it in ``layer_irreps``.
        num_heads: How many heads the model has.
        energy_head: The energy's head, given exactly when the energy is
            declared.
        precision: The dtype every op is built at.
    """

    matrix_basis: Tensor

    def __init__(
        self,
        backend,
        config: ModelConfig,
        layer_irreps: Sequence[str],
        num_heads: int,
        energy_head: EnergyOutputHead | None,
        precision: Precision = "float64",
    ) -> None:
        super().__init__()
        if num_heads < 1:
            raise ValueError(f"num_heads={num_heads}; a model has at least one head.")
        specs = {spec.name: spec for spec in config.observables.observables}
        if (ENERGY in specs) != (energy_head is not None):
            raise ValueError(
                f"the observables are {sorted(specs)} and an energy head was "
                f"{'not ' if energy_head is None else ''}given. The energy "
                f"needs one, and nothing else reads it."
            )
        if energy_head is not None and energy_head.e0_table.shape[0] != num_heads:
            raise ValueError(
                f"the energy head has isolated-atom energies for "
                f"{energy_head.e0_table.shape[0]} head(s) and the model has "
                f"{num_heads}."
            )
        if energy_head is not None and energy_head.scale.dtype != torch_dtype(
            precision
        ):
            raise ValueError(
                f"the energy head computes in {energy_head.scale.dtype} and the "
                f"output layer in {precision!r}; build both at the model's "
                f"precision."
            )
        for name in specs:
            if hasattr(nn.ModuleDict(), name):
                raise ValueError(
                    f"the observable {name!r} is named like an attribute of "
                    f"the module holding the readouts; declare it under "
                    f"another name."
                )
        readout = config.readout
        grouped = {name for group in readout.groups for name in group}
        groups = [list(group) for group in readout.groups] + [
            [name] for name in specs if name not in grouped
        ]
        self.specs = specs
        self.num_heads = num_heads
        self.energy_head = energy_head
        self.dipole_charges = readout.dipole_charges
        self.dipole_charge_factor = (
            1.0 / DEBYE_FROM_SI if readout.dipole_charges == "input" else 1.0
        )
        self.symmetric_matrices = readout.symmetric_matrices
        self.atomic_contributions = dict(readout.atomic_contributions)
        self.readouts = nn.ModuleDict(
            {
                "+".join(group): ObservableReadout(
                    backend,
                    [specs[name] for name in group],
                    layer_irreps,
                    num_heads,
                    readout.mlp_irreps,
                    last_only=readout.last_only,
                    precision=precision,
                )
                for group in groups
            }
        )
        self.register_buffer(
            "matrix_basis",
            torch.tensor(symmetric_matrix_basis(), dtype=torch.float64),
            persistent=False,
        )

    def forward(
        self,
        graph: Mapping[str, Tensor],
        features: NodeFeatures,
        pair_node_energy: Tensor | None = None,
    ) -> MACEOutput[Tensor]:
        """The declared observables of a batch.

        Args:
            graph: The flat dict. Reads ``batch``, ``ptr`` and ``head``, an
                absent ``head`` being head 0 for every structure; with a
                dipole charge part, ``positions`` and the input it reads:
                ``charges`` for input charges, ``total_charge`` for
                predicted ones.
            features: The backbone's output.
            pair_node_energy: ``[n_nodes]``, each atom's pair repulsion, for a
                model that has it.
        """
        num_nodes = features.element_index.shape[0]
        batch = graph["batch"][:num_nodes]
        num_graphs = graph["ptr"].shape[0] - 1
        head_index = graph.get("head")
        if head_index is None:
            head_index = torch.zeros(num_graphs, dtype=torch.int64, device=batch.device)
        node_head = head_index[batch]
        readout_head = node_head if self.num_heads > 1 else None

        per_layer: dict[str, list[Tensor]] = {}
        for module in self.readouts.values():
            readout = cast(ObservableReadout, module)
            contributions = readout(features.layers, readout_head)
            for name in readout.observables:
                per_layer[name] = []
            for value in contributions:
                for name, own in readout.split(value).items():
                    per_layer[name].append(own)

        fields: dict[str, Tensor] = {}
        extras: dict[str, Tensor] = {}

        def report(name: str, value: Tensor) -> None:
            field = FIELD_BY_OBSERVABLE.get(name, name)
            if field in CORE_FIELD_NAMES:
                fields[field] = value
            else:
                extras[name] = value

        atoms = {
            name: torch.stack(layers).sum(0)
            for name, layers in per_layer.items()
            if name != ENERGY
        }
        if self.dipole_charges == "predicted":
            atoms["charges"] = _charges_summing_to_total(
                atoms["charges"], graph, batch, num_graphs
            )
        for name, values in atoms.items():
            spec = self.specs[name]
            if name in self.atomic_contributions:
                report(self.atomic_contributions[name], _squeezed(values))
            if not spec.per_atom:
                values = segment_sum(values, batch, num_graphs)
                if name == "dipole" and self.dipole_charges != "none":
                    values = values + self._charge_dipole(
                        atoms, graph, batch, num_graphs, num_nodes
                    )
            if name in self.symmetric_matrices:
                report(f"{name}_sh", values)
                basis = self.matrix_basis.to(values.dtype)
                report(name, torch.einsum("kij,gk->gij", basis, values))
                continue
            report(name, _squeezed(values))

        if self.energy_head is not None:
            terms = self.energy_head(
                [value.squeeze(-1) for value in per_layer[ENERGY]],
                pair_node_energy,
                features.element_index,
                node_head,
                batch,
                num_graphs,
            )
            fields["total_energy"] = terms.total_energy
            fields["node_energies"] = terms.node_energy
            extras["interaction_energy"] = terms.interaction_energy
        return MACEOutput(**fields, extras=extras)

    def _charge_dipole(
        self,
        atoms: Mapping[str, Tensor],
        graph: Mapping[str, Tensor],
        batch: Tensor,
        num_graphs: int,
        num_nodes: int,
    ) -> Tensor:
        """``[n_graphs, 3]``: ``sum_i q_i r_i`` of each structure, in the
        dtype of the readouts."""
        dtype = atoms["dipole"].dtype
        if self.dipole_charges == "predicted":
            charges = atoms["charges"].squeeze(-1)
        else:
            charges = graph["charges"][:num_nodes].to(dtype)
        positions = graph["positions"][:num_nodes].to(dtype)
        dipoles = positions * charges.unsqueeze(-1) * self.dipole_charge_factor
        return segment_sum(dipoles, batch, num_graphs)


def _squeezed(values: Tensor) -> Tensor:
    """A single scalar component without its component axis."""
    return values.squeeze(-1) if values.shape[-1] == 1 else values


def _charges_summing_to_total(
    charges: Tensor, graph: Mapping[str, Tensor], batch: Tensor, num_graphs: int
) -> Tensor:
    """``[n_nodes, 1]``: every atom of a structure shifted by one constant so
    the structure's charges sum to its ``total_charge``."""
    charges = charges.squeeze(-1)
    counts = segment_sum(torch.ones_like(charges), batch, num_graphs)
    total = graph["total_charge"].to(charges.dtype)
    excess = segment_sum(charges, batch, num_graphs) / counts - total / counts
    return (charges - excess[batch]).unsqueeze(-1)
