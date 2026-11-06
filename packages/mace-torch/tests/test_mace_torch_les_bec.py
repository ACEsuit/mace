"""Born-charge force with both BEC layouts and all supported alpha layouts."""

import pytest
import torch
from conftest import assert_close
from mace_torch.models.les import field_forces

pytestmark = pytest.mark.les


@pytest.mark.parametrize("split", [False, True])
@pytest.mark.parametrize("layout", ["isotropic", "matrix", "flat"])
@pytest.mark.parametrize("neutral", [False, True])
def test_field_force_preserves_the_measured_bec(split, layout, neutral):
    full = torch.arange(27.0).reshape(3, 3, 3) / 10
    measured = torch.stack((full * 0.25, full * 0.75), 1) if split else full.clone()
    before = measured.clone()
    alpha = torch.tensor([0.5, 1.5, 2.5])
    tensors = alpha[:, None, None] * torch.eye(3)
    alphas = {"isotropic": alpha, "matrix": tensors, "flat": tensors.reshape(3, 9)}[
        layout
    ]
    field = torch.tensor([0.1, 0.3, -0.2])
    correction = field_forces(
        measured,
        alphas,
        field,
        volume=40,
        eps_infty=2.5,
        keep_neutral=neutral,
        electric_field_unit=1.7,
    )
    effective = (2.5 / (1 + 4.5 / (40 * 5.52635e-3))) ** 0.5
    expected = (
        torch.matmul(field, full - full.mean(0) if neutral else full) * effective * 1.7
    )
    assert_close(correction, expected)
    assert torch.equal(measured, before)
