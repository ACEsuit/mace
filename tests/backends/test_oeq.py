"""OpenEquivariance backend: e3nn parity. OEQ JIT-compiles CUDA/HIP kernels,
so execution requires a GPU (capability gpu); works on NVIDIA and AMD.
"""
# pylint: disable=wrong-import-position
import os

os.environ["TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD"] = "1"

import pytest

from mace.cli.convert_e3nn_oeq import run as run_e3nn_to_oeq
from mace.cli.convert_oeq_e3nn import run as run_oeq_to_e3nn
from tests.backends.backend_parity import BackendTestBase
from tests.helpers import OEQ_AVAILABLE



@pytest.mark.oeq
@pytest.mark.gpu
@pytest.mark.skipif(not OEQ_AVAILABLE, reason="openequivariance not installed")
class TestOeq(BackendTestBase):
    @pytest.fixture
    def conversion_functions(self):
        return run_e3nn_to_oeq, run_oeq_to_e3nn

    # OpenEquivariance JIT-compiles CUDA/HIP kernels: it can only execute on a
    # GPU. Parametrizing with an empty list on CPU hosts would collect zero
    # tests silently; a plain "cuda" param plus the gpu marker keeps the tests
    # visible (skipped locally, enforced in GPU CI via MACE_REQUIRE_CAPS).
    @pytest.fixture(params=["cuda"])
    def device(self, request):
        return request.param
