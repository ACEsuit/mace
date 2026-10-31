"""A CUDA cueq model whose kernels failed to load must raise, not run the naive path."""

import pytest
import torch

from mace.modules.wrapper_ops import CUET_AVAILABLE

pytestmark = pytest.mark.skipif(not CUET_AVAILABLE, reason="cuequivariance is not installed")


def _linear():
    import cuequivariance as cue
    import cuequivariance_torch as cuet

    irreps = cue.Irreps("O3", "2x0e + 1x1o")
    return cuet.Linear(irreps, irreps, layout=cue.mul_ir, method="naive", device="cpu")


def test_a_model_with_only_naive_polynomials_raises():
    from mace.modules.wrapper_ops import require_cueq_kernels

    with pytest.raises(RuntimeError, match="without its CUDA kernels"):
        require_cueq_kernels(torch.nn.ModuleList([_linear(), _linear()]))


def test_one_kernel_backed_polynomial_is_enough():
    import cuequivariance_torch as cuet

    from mace.modules.wrapper_ops import require_cueq_kernels

    # Linear layers are naive even with kernels; the tensor products are not. Stand one in.
    kernel = _linear()
    for m in kernel.modules():
        if isinstance(m, cuet.SegmentedPolynomial):
            m.method = "uniform_1d"
    require_cueq_kernels(torch.nn.ModuleList([_linear(), kernel]))


def test_a_model_without_cueq_modules_passes():
    from mace.modules.wrapper_ops import require_cueq_kernels

    require_cueq_kernels(torch.nn.Linear(2, 2))
