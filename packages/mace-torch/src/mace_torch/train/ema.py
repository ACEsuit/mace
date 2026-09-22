"""The exponentially moving average of the weights, written out.

Fifteen lines of arithmetic, kept here rather than taken from a dependency, and
the reason is the warm-up: the averaged weights start as a copy of the initial
ones, so for the first few hundred steps a decay of 0.99 is mostly averaging
noise. The frozen tree's dependency caps the decay by the step count for
exactly that reason, and a run whose reported validation error comes through
the average needs that behaviour stated where it can be read.

The averaged weights are what a run is judged on: validation runs through them
and the best checkpoint holds them. So this is not a smoothing detail, it is
the model the run produces.
"""

from __future__ import annotations

from collections.abc import Iterable, Iterator
from contextlib import contextmanager

import torch
from torch import Tensor, nn

__all__ = ["ExponentialMovingAverage"]

#: The warm-up cap. With `n` updates taken, the decay is at most
#: `(1 + n) / (10 + n)`, so it rises from 0.09 rather than starting at the
#: configured value while the average is still mostly its initial copy.
_WARMUP_OFFSET = 10.0


class ExponentialMovingAverage:
    """A running average of a model's parameters.

    Args:
        parameters: The parameters to track. Held by reference, so the same
            tensors the optimizer steps.
        decay: How much of the average survives each update. Higher is
            smoother and slower.
        warmup: Whether to cap the decay by the update count early on.
    """

    def __init__(
        self, parameters: Iterable[nn.Parameter], decay: float, warmup: bool = True
    ) -> None:
        if not 0.0 <= decay <= 1.0:
            raise ValueError(
                f"the decay is {decay} and it is a fraction of the average "
                f"that survives each update, so it lies in [0, 1]."
            )
        self.decay = decay
        self.warmup = warmup
        self.num_updates = 0
        self._parameters = [p for p in parameters if p.requires_grad]
        self.shadow: list[Tensor] = [p.detach().clone() for p in self._parameters]
        self._stashed: list[Tensor] = []

    def effective_decay(self) -> float:
        """What the next update will actually use."""
        if not self.warmup:
            return self.decay
        return min(
            self.decay, (1.0 + self.num_updates) / (_WARMUP_OFFSET + self.num_updates)
        )

    @torch.no_grad()
    def update(self) -> None:
        """Fold the current weights in. Called after each optimizer step."""
        decay = self.effective_decay()
        self.num_updates += 1
        for shadow, parameter in zip(self.shadow, self._parameters, strict=True):
            shadow.mul_(decay).add_(parameter.detach(), alpha=1.0 - decay)

    @contextmanager
    def average_parameters(self) -> Iterator[None]:
        """Run a block with the averaged weights in place, then put them back.

        The originals are kept rather than recomputed: the average is not
        invertible, so a block that forgot to restore would silently continue
        training from the smoothed weights.
        """
        self._stashed = [p.detach().clone() for p in self._parameters]
        with torch.no_grad():
            for parameter, shadow in zip(self._parameters, self.shadow, strict=True):
                parameter.copy_(shadow)
        try:
            yield
        finally:
            with torch.no_grad():
                for parameter, original in zip(
                    self._parameters, self._stashed, strict=True
                ):
                    parameter.copy_(original)
            self._stashed = []

    def state_dict(self) -> dict:
        """Enough to resume: the average and how many updates it has seen."""
        return {
            "decay": self.decay,
            "warmup": self.warmup,
            "num_updates": self.num_updates,
            "shadow": [tensor.clone() for tensor in self.shadow],
        }

    def load_state_dict(self, state: dict) -> None:
        if len(state["shadow"]) != len(self.shadow):
            raise ValueError(
                f"the saved average covers {len(state['shadow'])} tensor(s) "
                f"and this model has {len(self.shadow)}. Resuming would "
                f"average the wrong weights together."
            )
        self.decay = state["decay"]
        self.warmup = state["warmup"]
        self.num_updates = state["num_updates"]
        with torch.no_grad():
            for shadow, saved in zip(self.shadow, state["shadow"], strict=True):
                shadow.copy_(saved)
