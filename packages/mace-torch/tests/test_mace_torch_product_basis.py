"""The product basis block: the contraction, the linear, the residual add."""

from typing import Any

import pytest
import torch
from conftest import assert_close
from mace_core.clebsch_gordan.irreps import Irreps
from mace_torch.backends.reference import ReferenceBackend
from mace_torch.kernels import initialize_model_weights
from mace_torch.nn.product_basis import EquivariantProductBasisBlock

BACKEND = ReferenceBackend()


@pytest.fixture(autouse=True)
def no_gradients():
    with torch.no_grad():
        yield


def _block(**options: Any) -> EquivariantProductBasisBlock:
    precision = {torch.float64: "float64", torch.float32: "float32"}[
        torch.get_default_dtype()
    ]
    common: dict[str, Any] = {
        "irreps_in": "0e+1o+2e",
        "irreps_out": "0e+1o",
        "correlation": 3,
        "num_elements": 2,
        "num_features": 4,
        "precision": precision,
    }
    common.update(options)
    block = EquivariantProductBasisBlock(BACKEND, **common)
    initialize_model_weights(block, 2)
    return block


def _message() -> torch.Tensor:
    return torch.randn(5, 4 * 9, generator=torch.Generator().manual_seed(3))


def test_the_skip_is_added_after_the_linear():
    block = _block()
    message = _message()
    element = torch.tensor([0, 1, 1, 0, 1])
    skip = torch.randn(5, 4 * 4, generator=torch.Generator().manual_seed(4))
    without = block(message, element)
    assert_close(block(message, element, skip), without + skip, "residual add")
    assert_close(without, block.linear(block.contraction(message, element)), "order")


def test_each_element_reads_its_own_weights():
    block = _block()
    message = _message()
    assert not torch.allclose(
        block(message, torch.zeros(5, dtype=torch.long)),
        block(message, torch.ones(5, dtype=torch.long)),
    )


def test_one_element_row_serves_every_element():
    block = _block(num_elements=1)
    assert block.contraction.weights[0].shape[0] == 1
    assert block(_message(), torch.zeros(5, dtype=torch.long)).shape == (
        5,
        4 * Irreps.parse("0e+1o").dimension,
    )


@pytest.mark.parametrize("basis", ["reduced", "full"])
def test_the_basis_is_handed_to_the_contraction_unchanged(basis):
    assert _block(basis=basis).contraction.descriptor.basis == basis
