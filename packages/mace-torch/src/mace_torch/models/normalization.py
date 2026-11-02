"""Equivariant output normalization with head-leading checkpoint buffers."""

from collections.abc import Sequence

import torch
from mace_core.data.normalization import NormalizationValues
from mace_core.observables import ObservableSpec
from torch import Tensor, nn

from mace_torch.canonical import require_member_state


class ObservableNormalization(nn.Module):
    scale: Tensor
    shift: Tensor

    def __init__(
        self,
        spec: ObservableSpec,
        values: Sequence[NormalizationValues],
        dtype: torch.dtype,
    ) -> None:
        super().__init__()
        self.per_atom = spec.per_atom
        self.extensive = spec.extensive
        self.register_buffer(
            "scale", torch.tensor([row.scale for row in values], dtype=dtype)
        )
        self.register_buffer(
            "shift", torch.tensor([row.shift for row in values], dtype=dtype)
        )

    def forward(
        self, value: Tensor, head: Tensor, batch: Tensor, ptr: Tensor
    ) -> Tensor:
        """Apply after graph reduction, multiplying an extensive shift by N."""
        indices = head[batch] if self.per_atom else head
        shift = self.shift[indices]
        if self.extensive and not self.per_atom:
            shift = shift * (ptr[1:] - ptr[:-1]).unsqueeze(-1)
        return value * self.scale[indices].unsqueeze(-1) + shift

    def to_canonical(self) -> dict[str, Tensor]:
        return {"scale": self.scale.detach(), "shift": self.shift.detach()}

    def load_canonical(self, state: dict[str, Tensor]) -> None:
        require_member_state(state, ("scale", "shift"), "normalization")
        for name, expected in (("scale", self.scale), ("shift", self.shift)):
            if state[name].shape != expected.shape:
                raise ValueError(
                    f"normalization.{name}: expected shape {tuple(expected.shape)}, "
                    f"got {tuple(state[name].shape)}"
                )
        with torch.no_grad():
            self.scale.copy_(state["scale"])
            self.shift.copy_(state["shift"])
