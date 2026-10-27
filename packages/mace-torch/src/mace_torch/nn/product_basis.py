"""The many-body product basis, on the kernel backend's contraction.

The contraction is one backend op over the reduced Clebsch-Gordan basis, with
its weights per element in the canonical flat ``[Z, A, mul]``; the block around
it is a linear and the residual add. No code generation, no tracing, no
TorchScript.
"""

from __future__ import annotations

from mace_core.kernels.descriptors import (
    ClebschGordanBasis,
    LinearDescriptor,
    SymmetricContractionDescriptor,
)
from mace_core.kernels.precision import Precision
from torch import Tensor, nn

from mace_torch.backends.layout import expanded_irreps

__all__ = ["EquivariantProductBasisBlock"]


class EquivariantProductBasisBlock(nn.Module):
    """The symmetric contraction, then a linear that mixes channels.

    The skip a residual interaction carries is added **after the linear**,
    which is where a trained model puts it.

    Args:
        backend: The kernel backend. Consulted at construction only.
        irreps_in: One channel's input, the interaction's message.
        irreps_out: One channel's output, one copy of each irrep.
        correlation: The body order.
        num_elements: How many elements carry their own contraction weights.
            One shares them across elements.
        num_features: The channel count.
        basis: The Clebsch-Gordan basis the weights are written against,
            passed to the contraction's descriptor unchanged.
        precision: The dtype name every op is built at.
    """

    def __init__(
        self,
        backend,
        irreps_in: str,
        irreps_out: str,
        correlation: int,
        num_elements: int,
        num_features: int,
        basis: ClebschGordanBasis = "reduced",
        precision: Precision = "float64",
    ) -> None:
        super().__init__()
        out_flat = expanded_irreps(irreps_out, num_features)
        self.contraction = backend.make_symmetric_contraction(
            SymmetricContractionDescriptor(
                irreps_in=irreps_in,
                irreps_out=irreps_out,
                correlation=correlation,
                num_elements=num_elements,
                num_features=num_features,
                basis=basis,
                precision=precision,
            )
        )
        self.linear = backend.make_linear(
            LinearDescriptor(
                irreps_in=out_flat, irreps_out=out_flat, precision=precision
            )
        )

    def forward(
        self, message: Tensor, element: Tensor, skip: Tensor | None = None
    ) -> Tensor:
        """Grouped message in, grouped node features out.

        Args:
            message: ``[n_nodes, num_features * dim_in]``.
            element: ``[n_nodes]``, int64, each node's row of contraction
                weights.
            skip: The carried skip at this block's output declaration, or
                ``None`` when the interaction applied its own.
        """
        features = self.linear(self.contraction(message, element))
        if skip is None:
            return features
        return features + skip
