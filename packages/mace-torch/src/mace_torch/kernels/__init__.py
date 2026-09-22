"""The torch binding of the kernel contract.

:mod:`mace_core.kernels` states what an op is; this package is what runs it.
The ops live in :mod:`mace_torch.kernels.ops` as torch custom operators, and
the backends that build them live in :mod:`mace_torch.backends`.
"""

from mace_torch.kernels.initialization import initialize_model_weights, op_seed
from mace_torch.kernels.ops import (
    channelwise_tp_conv,
    segment_sum,
    symmetric_contraction,
)

__all__ = [
    "channelwise_tp_conv",
    "initialize_model_weights",
    "op_seed",
    "segment_sum",
    "symmetric_contraction",
]
