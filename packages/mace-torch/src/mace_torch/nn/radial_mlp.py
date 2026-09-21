"""The small dense network that turns a radial embedding into path weights.

One per interaction. It reads the radial basis of an edge length and produces
one weight per tensor-product path per channel, which is what makes the
convolution depend on distance.

Two constants here are not free, and both are properties of the trained
artifacts rather than choices:

The layers are **unbiased and scaled by ``1/sqrt(fan_in)`` inside the forward**,
not at initialization. A layer holds its raw weight and divides on every call,
so the effective map is ``x @ (W / sqrt(h_in))``. Initializing instead would
give the same function only until the first optimizer step.

The activation carries a **fixed multiplier** so its output has unit second
moment. See :data:`SECOND_MOMENT_SCALE` for why the value is the one it is.
"""

from __future__ import annotations

from collections.abc import Sequence

import torch
from torch import Tensor, nn

__all__ = ["SECOND_MOMENT_SCALE", "RadialMLP", "exact_second_moment_scale"]

#: The multiplier that gives ``silu`` unit second moment on standard normal
#: input, as **every trained MACE artifact carries it**.
#:
#: It is not the exact value. The number comes from a million-sample Monte Carlo
#: estimate under a fixed seed, and the exact value by Gauss-Hermite quadrature
#: is 1.6765324703310915, which differs by 0.16 per cent. That is enormous next
#: to any tolerance these models are compared at, so the estimate is what has to
#: be carried: a model trained against it computes a different function under
#: the exact one. :func:`exact_second_moment_scale` derives the exact value, and
#: a test pins the gap so it stays a known deviation rather than a discovered
#: one.
SECOND_MOMENT_SCALE = 1.679176792398942


def exact_second_moment_scale() -> float:
    """The multiplier that would make ``silu`` exactly unit second moment.

    Gauss-Hermite quadrature against the standard normal weight. Here to be
    compared with :data:`SECOND_MOMENT_SCALE`, not to replace it.
    """
    import numpy as np

    nodes, weights = np.polynomial.hermite_e.hermegauss(200)
    weights = weights / np.sqrt(2.0 * np.pi)
    activated = nodes / (1.0 + np.exp(-nodes))
    return float(np.sum(weights * activated**2) ** -0.5)


class RadialMLP(nn.Module):
    """Radial embedding in, one weight per path and channel out.

    Args:
        num_radial: Width of the radial embedding.
        hidden: The hidden widths, in order.
        num_out: How many weights to produce per edge.
        precision: The dtype the weights are held at.
    """

    def __init__(
        self,
        num_radial: int,
        hidden: Sequence[int],
        num_out: int,
        precision: str = "float64",
    ) -> None:
        super().__init__()
        dtype = torch.float64 if precision == "float64" else torch.float32
        widths = [num_radial, *hidden, num_out]
        self.weights = nn.ParameterList(
            [
                nn.Parameter(torch.randn(widths[i], widths[i + 1], dtype=dtype))
                for i in range(len(widths) - 1)
            ]
        )
        self.widths = list(widths)
        self.register_buffer(
            "scales",
            torch.tensor(
                [widths[i] ** -0.5 for i in range(len(widths) - 1)], dtype=dtype
            ),
            persistent=False,
        )

    def forward(self, radial: Tensor) -> Tensor:
        """``[n_edges, num_radial]`` to ``[n_edges, num_out]``."""
        activated = radial
        last = len(self.weights) - 1
        for index, weight in enumerate(self.weights):
            activated = activated @ (weight * self.scales[index])
            if index != last:
                activated = SECOND_MOMENT_SCALE * torch.nn.functional.silu(activated)
        return activated
