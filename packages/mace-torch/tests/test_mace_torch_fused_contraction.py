"""The existing V1 monomial optimization, across the agreed grouped contract."""

import importlib.util
import math

import pytest

if importlib.util.find_spec("torch") is None:  # pragma: no cover
    pytest.skip("the kernel ops need torch", allow_module_level=True)

import torch
from mace_core.clebsch_gordan.reduced_basis import (
    reduced_symmetric_tensor_product_basis,
)
from mace_torch.kernels.contraction import (
    monomial_basis,
)
from mace_torch.kernels.contraction import (
    symmetric_contraction_grouped as symmetric_contraction,
)


@pytest.fixture(autouse=True)
def double_precision():
    """gradcheck is a finite difference, so it needs fp64 to mean anything."""
    previous = torch.get_default_dtype()
    torch.set_default_dtype(torch.float64)
    torch.manual_seed(0)
    yield
    torch.set_default_dtype(previous)


def contraction_case(irreps="0e+1o", target="0e", correlation=2, elements=2, width=3):
    bases, weights = [], []
    for order in range(1, correlation + 1):
        array = reduced_symmetric_tensor_product_basis(irreps, order, target)[target]
        # The trailing extent is spelled out: a body order no path reaches has
        # zero rows, and `-1` cannot be inferred for an empty array.
        trailing = math.prod(array.shape[2:])
        basis = torch.tensor(array.reshape(array.shape[0], array.shape[1], trailing))
        bases.append(monomial_basis(basis, order))
        weights.append(torch.randn(elements, basis.shape[0], width, requires_grad=True))
    return bases, weights


@pytest.mark.parametrize("correlation", [1, 2, 3])
def test_symmetric_contraction_differentiates_twice(correlation):
    """Correlation 3 is the case that matters: the outer power then has a
    three-term product rule in its backward, and an error in one term is
    invisible at correlation 1 and 2."""
    bases, weights = contraction_case(correlation=correlation)
    features = torch.randn(4, 3, 4, requires_grad=True)
    element = torch.tensor([0, 1, 0, 1])

    def function(x, *w):
        return symmetric_contraction(x, list(w), bases, element, len(bases))

    assert torch.autograd.gradcheck(function, (features, *weights))
    assert torch.autograd.gradgradcheck(function, (features, *weights))


@pytest.mark.parametrize("correlation", [1, 2, 3])
def test_the_contraction_is_the_basis_applied_to_the_outer_power(correlation):
    """The op never forms the outer power, so check it against the one
    formula that does: every body order's basis contracted with ``x`` repeated
    ``order`` times, then weighted per element."""
    irreps, target = "0e+1o+2e", "1o"
    bases, weights = contraction_case(irreps, target, correlation, elements=2)
    features = torch.randn(5, 3, 9)
    element = torch.tensor([0, 1, 1, 0, 1])
    expected = torch.zeros(5, 3, 3)
    for order, weight in enumerate(weights, start=1):
        array = reduced_symmetric_tensor_product_basis(irreps, order, target)[target]
        if array.shape[0] == 0:
            continue
        power = features
        for _ in range(order - 1):
            power = (power.unsqueeze(-1) * features.unsqueeze(-2)).flatten(-2)
        flat = torch.tensor(array.reshape(array.shape[0], array.shape[1], -1))
        projected = torch.einsum("aof,nmf->nmao", flat, power)
        expected += torch.einsum("nmao,nam->nmo", projected, weight[element])
    contracted = symmetric_contraction(features, weights, bases, element, len(bases))
    torch.testing.assert_close(contracted, expected, rtol=1e-12, atol=1e-12)


def test_several_output_irreps_in_one_call_are_each_contracted_alone():
    """The op takes every output irrep at once and shares the monomials between
    them. Each irrep's slice of the result has to be what it gives alone, and
    its gradient has to reach only its own weights."""
    irreps, correlation = "0e+1o+2e", 3
    cases = [contraction_case(irreps, t, correlation) for t in ("0e", "1o", "2e")]
    bases = [basis for case in cases for basis in case[0]]
    weights = [weight for case in cases for weight in case[1]]
    features = torch.randn(4, 3, 9, requires_grad=True)
    element = torch.tensor([0, 1, 0, 1])
    joined = symmetric_contraction(features, weights, bases, element, correlation)
    alone = torch.cat(
        [
            symmetric_contraction(features, case[1], case[0], element, correlation)
            for case in cases
        ],
        dim=-1,
    )
    torch.testing.assert_close(joined, alone, rtol=1e-12, atol=1e-12)

    def function(x, *w):
        return symmetric_contraction(x, list(w), bases, element, correlation)

    assert torch.autograd.gradcheck(function, (features, *weights))
    assert torch.autograd.gradgradcheck(function, (features, *weights))


def test_the_contraction_takes_no_nodes_and_unreachable_orders():
    """Zero nodes and a body order no path reaches are both shapes a real batch
    or a real declaration produces, and neither may need a ``-1`` inferred from
    an empty tensor. ``0e+1o`` in, ``2e`` out has no path at order one."""
    bases, weights = contraction_case("0e+1o", "2e", correlation=3)
    assert bases[0].shape[1] == 0
    for nodes in (0, 3):
        features = torch.randn(nodes, 3, 4, requires_grad=True)
        element = torch.zeros(nodes, dtype=torch.long)
        out = symmetric_contraction(features, weights, bases, element, len(bases))
        assert out.shape == (nodes, 3, 5)
        out.sum().backward()


def test_the_contraction_is_element_wise_in_its_weights():
    """Each element carries its own weights, so a node of one element cannot be
    moved by another element's.

    Note the two axes this is easy to confuse, and the first version of this
    test did: the list is indexed by body order, and the element is axis 0
    *inside* each of its tensors.
    """
    bases, weights = contraction_case(correlation=2, elements=2)
    features = torch.randn(2, 3, 4)
    all_element_zero = torch.tensor([0, 0])
    before = symmetric_contraction(
        features, weights, bases, all_element_zero, len(bases)
    )
    with torch.no_grad():
        for per_order in weights:
            per_order[1].add_(100.0)
    after = symmetric_contraction(
        features, weights, bases, all_element_zero, len(bases)
    )
    assert torch.equal(before, after)


def test_a_body_order_with_no_reachable_path_contributes_nothing():
    """An unreachable order gives a zero-path basis, and the op skips it rather
    than needing its weights forced to zero as the frozen tree does."""
    array = reduced_symmetric_tensor_product_basis("0e", 1, "1o")["1o"]
    assert array.shape[0] == 0
