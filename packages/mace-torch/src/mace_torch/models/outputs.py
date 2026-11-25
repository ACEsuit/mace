"""The output layer: every declared observable, read out, reduced and typed.

The second of the two layers the model is split into. The backbone returns
node features; this turns them into a :class:`~mace_core.outputs.MACEOutput`,
one readout per observable the model declares. A new observable is a new
declaration and no new code: its irreps decide the readout, its ``per_atom``
whether the result is summed over each structure, and its name the field of
the output it fills, or the entry of ``extras``.

The energy is the one observable with more to it: its per-layer site energies
go through the :class:`~mace_torch.models.energy.EnergyOutputHead`, which
adds the isolated-atom energies, the scale and shift and the pair repulsion.
No derivative is taken here.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence

import torch
from mace_core.config.model import ReadoutConfig
from mace_core.kernels.precision import Precision
from mace_core.observables import ObservableCatalogue, ObservableSpec, declared_as
from mace_core.outputs import CORE_FIELD_NAMES, FIELD_BY_OBSERVABLE, MACEOutput
from torch import Tensor, nn

from mace_torch.backends.reference import torch_dtype
from mace_torch.kernels import segment_sum
from mace_torch.models.energy import EnergyOutputHead
from mace_torch.models.readouts import ObservableReadout
from mace_torch.nn.backbone import NodeFeatures

__all__ = ["MACEOutputs"]

#: The observable the energy head reads out.
ENERGY = "energy"

#: What the energy head writes besides the total energy.
_ENERGY_WRITES = ("node_energies", "interaction_energy")


class MACEOutputs(nn.Module):
    """Every declared observable, read out of the backbone's node features.

    Args:
        backend: The kernel backend. Consulted at construction only.
        observables: What the model computes; one readout per observable.
        readout: How they are read out.
        layer_irreps: Each layer's node-feature declaration, as the backbone
            reports it in ``layer_irreps``.
        num_heads: How many heads the model has.
        energy_head: The energy's head, given exactly when the energy is
            declared.
        precision: The dtype every op is built at.
    """

    def __init__(
        self,
        backend,
        observables: ObservableCatalogue,
        readout: ReadoutConfig,
        layer_irreps: Sequence[str],
        num_heads: int,
        energy_head: EnergyOutputHead | None,
        precision: Precision = "float64",
    ) -> None:
        super().__init__()
        if num_heads < 1:
            raise ValueError(f"num_heads={num_heads}; a model has at least one head.")
        names = [spec.name for spec in observables.observables]
        if (ENERGY in names) != (energy_head is not None):
            raise ValueError(
                f"the observables are {names} and an energy head was "
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
        for spec in observables.observables:
            _check_destination(spec, energy_declared=energy_head is not None)
            if hasattr(nn.ModuleDict(), spec.name):
                raise ValueError(
                    f"the observable {spec.name!r} is named like an attribute "
                    f"of the module holding the readouts; declare it under "
                    f"another name."
                )
        self.specs = observables.observables
        self.num_heads = num_heads
        self.energy_head = energy_head
        self.readouts = nn.ModuleDict(
            {
                spec.name: ObservableReadout(
                    backend,
                    spec,
                    layer_irreps,
                    num_heads,
                    readout.mlp_irreps,
                    last_only=readout.last_only,
                    precision=precision,
                )
                for spec in self.specs
            }
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
                absent ``head`` being head 0 for every structure.
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

        fields: dict[str, Tensor] = {}
        extras: dict[str, Tensor] = {}
        for spec in self.specs:
            contributions = self.readouts[spec.name](features.layers, readout_head)
            if spec.name == ENERGY:
                assert self.energy_head is not None
                terms = self.energy_head(
                    [value.squeeze(-1) for value in contributions],
                    pair_node_energy,
                    features.element_index,
                    node_head,
                    batch,
                    num_graphs,
                )
                fields["total_energy"] = terms.total_energy
                fields["node_energies"] = terms.node_energy
                extras["interaction_energy"] = terms.interaction_energy
                continue
            value = torch.stack(contributions).sum(0)
            if not spec.per_atom:
                value = segment_sum(value, batch, num_graphs)
            field = FIELD_BY_OBSERVABLE.get(spec.name, spec.name)
            if field in CORE_FIELD_NAMES:
                fields[field] = value
            else:
                extras[spec.name] = value
        return MACEOutput(**fields, extras=extras)


def _check_destination(spec: ObservableSpec, energy_declared: bool) -> None:
    """Refuse an observable whose value would land where another one does.

    The energy head fills ``node_energies`` and ``extras["interaction_energy"]``,
    and a core field of the output has one shape. An observable written to
    either would overwrite a value or break the shape, depending on the order
    of the declarations.
    """
    if spec.name == ENERGY:
        return
    if energy_declared and spec.name in _ENERGY_WRITES:
        raise ValueError(
            f"the observable {spec.name!r} is a name the energy fills; "
            f"declare it under another name."
        )
    field = FIELD_BY_OBSERVABLE.get(spec.name, spec.name)
    if field not in CORE_FIELD_NAMES:
        return
    if field == "dipole" and declared_as(spec, "1o", per_atom=False):
        return
    raise ValueError(
        f"the observable {spec.name!r} would fill the output's {field!r} "
        f"field, which holds "
        + (
            "one 1o vector per structure"
            if field == "dipole"
            else "what the energy or its derivatives give"
        )
        + f"; it is declared {spec.irreps!r}, "
        f"{'per atom' if spec.per_atom else 'per structure'}. Declare it to "
        f"match, or under another name."
    )
