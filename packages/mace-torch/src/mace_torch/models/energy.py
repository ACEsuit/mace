"""The energy head: isolated-atom energies, scale and shift, and two sums.

Everything between the per-layer site energies and the energy of a structure,
none of which the backbone sees. The frozen tree runs it inside the forward of
each energy model class; the classes differ in where the pair repulsion is
added and in the dtype of the per-atom energies, and both become arguments
here.

Three rules this module keeps:

* **The E0 table is float64.** Held at a narrower type it rounds an
  isolated-atom energy of thousands of eV when the model is built, and no
  later cast recovers it.
* **Two sums, then one add.** The isolated-atom energies and the interaction
  energies are summed over a structure's atoms separately and added per
  structure. Summed together, the small interaction energies lose their digits
  to the large isolated-atom ones (about 0.7 eV at a thousand atoms in
  float32, about 68 eV at ten thousand).
* **A device without float64 says so when the model is built.** MPS cannot
  hold a float64 tensor at all, so a head built for it holds the table and
  the per-atom energies in the model's own dtype. A head built for a device
  with float64 cannot then be moved to one without: it is built for the
  device it runs on.
"""

from __future__ import annotations

import math
import warnings
from collections.abc import Mapping, Sequence
from dataclasses import dataclass

import torch
from mace_core.elements import ResolvedE0s, model_atomic_number_table
from mace_core.kernels.precision import Precision
from torch import Tensor, nn

from mace_torch.backends.reference import torch_dtype
from mace_torch.kernels import segment_sum

__all__ = ["EnergyOutputHead", "EnergyTerms"]

#: Device types without float64.
_NO_FLOAT64 = frozenset({"mps"})


@dataclass(frozen=True)
class EnergyTerms:
    """The energies of a batch, before any derivative is taken.

    Attributes:
        total_energy: ``[n_graphs]``, in the model's dtype. The quantity a
            derivative engine differentiates.
        node_energy: ``[n_nodes]``, each atom's isolated-atom energy plus its
            interaction energy, in the head's ``node_energy_dtype``.
        interaction_energy: ``[n_graphs]``, in the model's dtype: the total
            without the isolated-atom energies. Its position gradient is the
            total's, since the isolated-atom energies do not depend on the
            positions.
    """

    total_energy: Tensor
    node_energy: Tensor
    interaction_energy: Tensor


class EnergyOutputHead(nn.Module):
    """Isolated-atom energies, scale and shift, and the two reductions.

    Args:
        resolved: The isolated-atom energies of every head.
        heads: The head names, in the order a graph's ``head`` index counts.
        atomic_numbers: The model's element table, strictly ascending, in the
            order of the element index.
        scale: One scale per head, in ``heads`` order, positive.
        shift: One shift per head, in eV/atom.
        zbl_in_scale_shift: Where the pair repulsion is added. ``True`` puts it
            inside the scaled interaction energy, so it is scaled and shifted
            with it; ``False`` adds it afterwards, unscaled. A trained model
            fixes one of the two.
        precision: The model's dtype: the inputs', the sums', the scale and
            shift's.
        node_energy_dtype: The dtype the per-atom energies are added up and
            returned in, at least as wide as ``precision``. Each atom's
            isolated-atom energy is taken at ``precision`` first, as the
            frozen tree takes it; a wider dtype keeps the digits the sum of
            the two terms would round away.
        device: Where the model will run. On a device without float64 the
            model is not built in float64; the table and the per-atom
            energies are held at ``precision``, and the build warns.
    """

    e0_table: Tensor
    scale: Tensor
    shift: Tensor

    def __init__(
        self,
        resolved: ResolvedE0s,
        heads: Sequence[str],
        atomic_numbers: Sequence[int],
        scale: Sequence[float],
        shift: Sequence[float],
        *,
        zbl_in_scale_shift: bool,
        precision: Precision = "float64",
        node_energy_dtype: Precision = "float64",
        device: torch.device | str = "cpu",
    ) -> None:
        super().__init__()
        if len(set(heads)) != len(heads):
            raise ValueError(f"the heads {list(heads)} repeat a name.")
        for name, values in (("scale", scale), ("shift", shift)):
            if len(values) != len(heads):
                raise ValueError(
                    f"{name} has {len(values)} value(s) and the model has "
                    f"{len(heads)} head(s), {list(heads)}; there is one per head."
                )
            if not all(math.isfinite(value) for value in values):
                raise ValueError(f"{name} {list(values)} is not finite.")
        if not all(value > 0 for value in scale):
            raise ValueError(
                f"scale {list(scale)} is not positive; a head's interaction "
                f"energy is multiplied by it."
            )
        model_dtype = torch_dtype(precision, "precision")
        node_dtype = torch_dtype(node_energy_dtype, "node_energy_dtype")
        if node_dtype.itemsize < model_dtype.itemsize:
            raise ValueError(
                f"node_energy_dtype={node_energy_dtype!r} is narrower than the "
                f"model's precision {precision!r}; the per-atom energies are "
                f"added up at least as wide as they are computed."
            )
        table = model_atomic_number_table(atomic_numbers)

        device_type = torch.device(device).type
        if device_type in _NO_FLOAT64:
            if precision == "float64":
                raise ValueError(
                    f"precision='float64' on {device_type}, which has no "
                    f"float64; build the model in 'float32' there."
                )
            if node_energy_dtype == "float64":
                warnings.warn(
                    f"{device_type} has no float64: the isolated-atom energies "
                    f"are held, and the per-atom energies added up, in "
                    f"{precision!r}.",
                    stacklevel=2,
                )
            table_dtype, node_dtype = model_dtype, model_dtype
        else:
            table_dtype = torch.float64

        self.zbl_in_scale_shift = zbl_in_scale_shift
        self.node_energy_dtype = node_dtype
        self.register_buffer(
            "e0_table",
            torch.tensor(resolved.to_array(heads, table.zs), dtype=table_dtype),
        )
        self.register_buffer("scale", torch.tensor(scale, dtype=model_dtype))
        self.register_buffer("shift", torch.tensor(shift, dtype=model_dtype))

    def to_canonical(self) -> dict[str, Tensor]:
        """The table ``[n_heads, n_elements]`` as held, float64 except on a
        device without it, and the scale and shift ``[n_heads]``."""
        return {
            "e0_table": self.e0_table.detach(),
            "scale": self.scale.detach(),
            "shift": self.shift.detach(),
        }

    def load_canonical(self, state: Mapping[str, Tensor]) -> None:
        """Put a saved table, scale and shift back.

        Raises:
            ValueError: If the saved table is narrower than this head's: it
                was rounded before it was saved, and a wider copy of it would
                pass the rounding off as digits.
        """
        saved = state["e0_table"].dtype
        if saved.itemsize < self.e0_table.dtype.itemsize:
            raise ValueError(
                f"the saved isolated-atom energies are {saved} and this head "
                f"holds them as {self.e0_table.dtype}; the saved table has "
                f"already lost the digits this head would keep."
            )
        with torch.no_grad():
            self.e0_table.copy_(state["e0_table"])
            self.scale.copy_(state["scale"])
            self.shift.copy_(state["shift"])

    def forward(
        self,
        node_energy_layers: Sequence[Tensor],
        pair_node_energy: Tensor | None,
        element_index: Tensor,
        node_head: Tensor,
        batch: Tensor,
        num_graphs: int,
        embedding_node_energy: Tensor | None = None,
    ) -> EnergyTerms:
        """The energies of a batch from its per-layer site energies.

        Args:
            node_energy_layers: One ``[n_nodes]`` tensor per layer read out, in
                the model's dtype.
            pair_node_energy: ``[n_nodes]``, each atom's pair repulsion, or
                ``None`` for a model without it.
            element_index: ``[n_nodes]``, each atom's column of the element
                table.
            node_head: ``[n_nodes]``, each atom's head.
            batch: ``[n_nodes]``, each atom's structure.
            num_graphs: How many structures.
            embedding_node_energy: ``[n_nodes]``, the energy read out of the
                node embedding, or ``None``. It is added unscaled beside the
                isolated-atom energies, summed on its own, and counted in
                each atom's energy.
        """
        terms = list(node_energy_layers)
        if pair_node_energy is not None and self.zbl_in_scale_shift:
            terms.insert(0, pair_node_energy)
        node_interaction = torch.stack(terms).sum(0)
        node_interaction = (
            self.scale[node_head] * node_interaction + self.shift[node_head]
        )
        if pair_node_energy is not None and not self.zbl_in_scale_shift:
            node_interaction = node_interaction + pair_node_energy

        node_e0 = self.e0_table[node_head, element_index].to(node_interaction.dtype)
        interaction_energy = segment_sum(node_interaction, batch, num_graphs)
        e0_energy = segment_sum(node_e0, batch, num_graphs)
        if embedding_node_energy is not None:
            e0_energy = e0_energy + segment_sum(
                embedding_node_energy, batch, num_graphs
            )
        total_energy = e0_energy + interaction_energy

        node_dtype = self.node_energy_dtype
        # Each term widened before it is added, so a small one keeps the
        # digits a large isolated-atom energy would round away.
        node_energy = node_e0.to(node_dtype)
        if embedding_node_energy is not None:
            node_energy = node_energy + embedding_node_energy.to(node_dtype)
        node_energy = node_energy + node_interaction.to(node_dtype)
        return EnergyTerms(
            total_energy=total_energy,
            node_energy=node_energy,
            interaction_energy=interaction_energy,
        )
