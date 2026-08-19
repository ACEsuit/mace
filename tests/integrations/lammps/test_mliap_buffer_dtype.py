"""The ML-IAP wrapper's own buffers must follow the model, not the process.

`total_charge` and `total_spin` are handed to the forward beside the model's own
tensors, so a float32 pair under a float64 model does not raise: promotion makes
it a quietly less accurate number, and it is LAMMPS that consumes it.

They used to be built at `torch.get_default_dtype()`, and the export happened to
be correct only because `mace_create_lammps_model --format mliap` converts to the
cueq layout first and that converter set the default globally on its way through.
Restoring the default in the converter -- which every other caller wants, since
`MACECalculator` and `run_train` call it as a plain function -- removes that
accident, so the dtype has to come from the model instead.
"""

import pytest
import torch

from mace.calculators.lammps_mliap_mace import MACEEdgeForcesWrapper
from tests.integrations.lammps._harness import StubMACE


@pytest.fixture(name="process_default_float32", autouse=True)
def fixture_process_default_float32():
    """A float32 process, which is torch's default and what a fresh CLI has."""
    previous = torch.get_default_dtype()
    torch.set_default_dtype(torch.float32)
    yield
    torch.set_default_dtype(previous)


@pytest.mark.parametrize("dtype", [torch.float64, torch.float32])
def test_the_wrapper_buffers_follow_the_model_dtype(dtype):
    wrapper = MACEEdgeForcesWrapper(StubMACE(num_interactions=2, dtype=dtype))

    assert wrapper.total_charge.dtype is dtype
    assert wrapper.total_spin.dtype is dtype


def test_a_float64_model_is_not_downgraded_by_a_float32_process():
    """The regression the converter's restore would otherwise have introduced:
    `--dtype float64` is the export default, and nothing downstream raises."""
    wrapper = MACEEdgeForcesWrapper(StubMACE(num_interactions=2, dtype=torch.float64))

    assert torch.get_default_dtype() is torch.float32
    assert wrapper.total_charge.dtype is torch.float64


def test_explicit_buffers_still_win():
    """`total_charge=` is how a caller sets a charged system; it must be kept
    verbatim rather than recast to the model's dtype."""
    given = torch.tensor([-1.0], dtype=torch.float64)

    wrapper = MACEEdgeForcesWrapper(
        StubMACE(num_interactions=2, dtype=torch.float64), total_charge=given
    )

    assert wrapper.total_charge.item() == -1.0
    assert wrapper.total_charge.dtype is torch.float64
