"""The energy head: isolated-atom energies, scale and shift, and two sums.

Everything the backbone must not see. The frozen tree runs all of it inside
``forward`` of two model classes whose only real difference is where the
pair-repulsion term is added, so the difference becomes a field here rather
than a second class.

Three things are settled in this module and are not free choices:

* **The E0 table is float64 whatever the model computes in.** Registering it at
  the build-time default dtype rounds an isolated-atom energy of some tens of eV
  to fp32 at construction, and no later cast recovers it.
* **The two sums stay separate.** Summing site energies and isolated-atom
  energies together loses the small one: measured on the frozen tree, 0.69 eV at
  a thousand atoms in fp32 and about 68 eV at ten thousand. The fused form is
  cheaper and wrong.
* **The accumulation type is a floor, not a promise.** A device without float64
  gets the model's own type and says so at build time.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Literal

import torch
from mace_core.elements import AtomicNumberTable, ResolvedE0s
from mace_core.kernels.precision import PrecisionConfig
from torch import Tensor, nn

from mace_torch.kernels import segment_sum

__all__ = ["EnergyOutputHead", "EnergyTerms", "ScaleShiftSpec", "ScalingMethod"]

_TORCH_DTYPE = {
    "float64": torch.float64,
    "float32": torch.float32,
    "bfloat16": torch.bfloat16,
}

#: How the interaction energy's scale and shift were computed. One enumeration,
#: where the frozen tree has a registry and a CLI choice list that disagree:
#: the registry offers ``std_scaling``, ``rms_forces_scaling`` and
#: ``rms_dipoles_scaling``, the flag offers ``std_scaling``,
#: ``rms_forces_scaling`` and ``no_scaling``, and ``no_scaling`` is implemented
#: outside the registry by forcing the standard deviation to one. All four are
#: kept and named here.
ScalingMethod = Literal["std", "rms_forces", "rms_dipoles", "none"]


@dataclass(frozen=True)
class ScaleShiftSpec:
    """The scale and shift, per head, already computed.

    The numbers come from dataset statistics before the model is built, so this
    carries results rather than a recipe. Holding the method alongside them is
    what lets a checkpoint say how they were obtained without a second registry
    existing anywhere.

    Attributes:
        method: Which formula produced them.
        scale: One multiplier per head, in the head order the model uses.
        shift: One offset per head, same order.
    """

    method: ScalingMethod
    scale: tuple[float, ...]
    shift: tuple[float, ...]

    def __post_init__(self) -> None:
        if len(self.scale) != len(self.shift):
            raise ValueError(
                f"the scale has {len(self.scale)} entries and the shift has "
                f"{len(self.shift)}. There is one of each per head."
            )
        if not self.scale:
            raise ValueError("a scale and shift are needed for at least one head.")


@dataclass
class EnergyTerms:
    """What the energy head produces, before any derivative is taken.

    Three quantities rather than one because they are not interchangeable.
    ``total_energy`` is what the derivative engine differentiates.
    ``interaction_energy`` is the same thing without the isolated-atom part,
    and differentiating it gives the identical forces, since the E0 branch has
    no path back to the positions. Reporting both means a consumer never has to
    subtract a table to recover one.

    Attributes:
        total_energy: ``[n_graphs]``, in the model's own dtype.
        node_energy: ``[n_atoms]``, in the accumulation dtype.
        interaction_energy: ``[n_graphs]``, the total without the E0 sum.
    """

    total_energy: Tensor
    node_energy: Tensor
    interaction_energy: Tensor


class EnergyOutputHead(nn.Module):
    """Isolated-atom energies, scale and shift, and the two reductions.

    Args:
        resolved: The isolated-atom energies, already resolved to numbers.
        heads: The head names, in the order the model indexes them.
        z_table: The element table, in the order the model indexes it.
        scale_shift: The per-head scale and shift.
        precision: What to compute in and what to accumulate in.
        zbl_in_scale_shift: Where the short-range pair repulsion is added.
            ``False`` adds it unscaled, after the scale and shift, which is what
            the frozen tree's plain model does. ``True`` puts it inside the
            scaled sum, which is what its scale-shift model does, so it picks up
            ``scale * P + n * shift``. The two differ by about 77 eV on the
            probe geometry, so this is a fact about a trained model and not a
            preference.
        supports_float64: Whether the device this will run on has float64. The
            accumulation type degrades to the model's own when it does not.
    """

    e0_table: Tensor
    scale: Tensor
    shift: Tensor

    def __init__(
        self,
        resolved: ResolvedE0s,
        heads: Sequence[str],
        z_table: AtomicNumberTable,
        scale_shift: ScaleShiftSpec,
        precision: PrecisionConfig,
        zbl_in_scale_shift: bool = True,
        supports_float64: bool = True,
    ) -> None:
        super().__init__()
        if len(scale_shift.scale) != len(heads):
            raise ValueError(
                f"the scale and shift cover {len(scale_shift.scale)} head(s) "
                f"and the model has {len(heads)}: {list(heads)}."
            )
        self.heads = list(heads)
        self.zbl_in_scale_shift = zbl_in_scale_shift
        self.precision = precision
        self.accumulate = precision.resolved_accumulate(supports_float64)
        self.degraded = self.accumulate != precision.accumulate

        self.register_buffer(
            "e0_table",
            torch.tensor(
                resolved.to_array(self.heads, list(z_table.zs)), dtype=torch.float64
            ),
            persistent=True,
        )
        self.e0_table.requires_grad_(False)
        model_dtype = _TORCH_DTYPE[precision.model]
        self.register_buffer(
            "scale", torch.tensor(scale_shift.scale, dtype=model_dtype)
        )
        self.register_buffer(
            "shift", torch.tensor(scale_shift.shift, dtype=model_dtype)
        )
        self.scaling_method = scale_shift.method

    def build_report(self) -> str | None:
        """What was silently changed at build time, if anything."""
        if not self.degraded:
            return None
        return (
            f"this device has no float64, so energies accumulate in "
            f"{self.accumulate} rather than the requested "
            f"{self.precision.accumulate}. Sums over many atoms are less "
            f"accurate than the configuration asked for."
        )

    def forward(
        self,
        node_energy_layers: list[Tensor],
        zbl_node_energy: Tensor | None,
        element_index: Tensor,
        head_index: Tensor,
        batch: Tensor,
        num_graphs: int,
    ) -> EnergyTerms:
        """The energies, from per-layer site energies.

        Args:
            node_energy_layers: One ``[n_atoms]`` tensor per interaction layer.
            zbl_node_energy: The short-range pair repulsion per atom, or
                ``None``.
            element_index: ``[n_atoms]``, each atom's position in the element
                table.
            head_index: ``[n_graphs]``, which head each structure belongs to.
            batch: ``[n_atoms]``, which structure each atom belongs to.
            num_graphs: How many structures, as a plain int so it stays
                symbolic under tracing.
        """
        accumulate = _TORCH_DTYPE[self.accumulate]
        model_dtype = self.scale.dtype
        node_head = head_index[batch]

        interaction = sum(node_energy_layers)
        if zbl_node_energy is not None and self.zbl_in_scale_shift:
            interaction = interaction + zbl_node_energy
        scaled = self.scale[node_head] * interaction + self.shift[node_head]
        if zbl_node_energy is not None and not self.zbl_in_scale_shift:
            scaled = scaled + zbl_node_energy

        # Gathering a row of the table is the one-hot matmul the frozen tree
        # writes, with the same result and without materializing the one-hot.
        e0 = self.e0_table[node_head, element_index].to(model_dtype)

        node_energy = scaled.to(accumulate) + e0.to(accumulate)
        interaction_total = segment_sum(scaled.to(accumulate), batch, num_graphs).to(
            model_dtype
        )
        e0_total = segment_sum(e0.to(accumulate), batch, num_graphs).to(model_dtype)
        return EnergyTerms(
            total_energy=interaction_total + e0_total,
            node_energy=node_energy,
            interaction_energy=interaction_total,
        )
