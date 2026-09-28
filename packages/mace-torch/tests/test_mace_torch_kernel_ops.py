"""The dispatched ops: their derivatives, and how they compile.

Two properties matter and neither is about the forward values.

The **second derivative** has to be right, because training on forces
differentiates a quantity that was itself produced by a backward pass. A
backward that is not itself differentiable does not fail: it gives wrong
forces, and a model trained on wrong forces looks like it is working.

And the ops have to be **opaque to the compiler and transparent to shapes**:
one node per op with no graph break, and the node count entering as a symbolic
dimension so one compiled frame serves every batch size.
"""

import importlib.util
import math

import pytest

if importlib.util.find_spec("torch") is None:  # pragma: no cover
    pytest.skip("the kernel ops need torch", allow_module_level=True)

import torch
from mace_core.clebsch_gordan.reduced_basis import (
    reduced_symmetric_tensor_product_basis,
)
from mace_torch.kernels.ops import (
    channelwise_tp_conv,
    monomial_basis,
    segment_sum,
    symmetric_contraction,
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


# ---------------------------------------------------------------------------
# segment_sum
# ---------------------------------------------------------------------------


def test_segment_sum_reduces_into_its_segments():
    values = torch.tensor([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
    index = torch.tensor([0, 0, 1])
    assert torch.equal(
        segment_sum(values, index, 2), torch.tensor([[4.0, 6.0], [5.0, 6.0]])
    )


def test_a_segment_nothing_points_at_is_zero_rather_than_missing():
    values = torch.tensor([[1.0]])
    assert torch.equal(
        segment_sum(values, torch.tensor([0]), 3), torch.tensor([[1.0], [0.0], [0.0]])
    )


def test_segment_sum_differentiates_twice():
    values = torch.randn(6, 3, requires_grad=True)
    index = torch.tensor([0, 0, 1, 1, 2, 2])
    function = lambda v: segment_sum(v, index, 3)  # noqa: E731
    assert torch.autograd.gradcheck(function, (values,))
    assert torch.autograd.gradgradcheck(function, (values,))


# ---------------------------------------------------------------------------
# symmetric_contraction
# ---------------------------------------------------------------------------


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


# ---------------------------------------------------------------------------
# channelwise_tp_conv
# ---------------------------------------------------------------------------


def convolution_case(nodes=3, edges=6, width=2, dim_in=4, dim_edge=3, paths=2, out=4):
    return {
        "node_features": torch.randn(nodes, width, dim_in, requires_grad=True),
        "edge_attributes": torch.randn(edges, dim_edge, requires_grad=True),
        "radial_weights": torch.randn(edges, paths, width, requires_grad=True),
        "coefficients": torch.randn(out, dim_in, dim_edge),
        "path_widths": [out // paths] * paths,
        "sender": torch.randint(0, nodes, (edges,)),
        "receiver": torch.randint(0, nodes, (edges,)),
        "num_nodes": nodes,
    }


def test_the_convolution_returns_node_level_values():
    """Always `[n_nodes, ...]`, never `[n_edges, ...]`. Whether the reduction is
    fused is the backend's business, which is what removes the six
    `conv_fusion` branches from the interaction blocks."""
    case = convolution_case(nodes=3, edges=6)
    assert channelwise_tp_conv(**case).shape == (3, 2, 4)


def test_the_convolution_differentiates_twice():
    case = convolution_case()

    def function(features, attributes, radial):
        return channelwise_tp_conv(
            features,
            attributes,
            radial,
            case["coefficients"],
            case["path_widths"],
            case["sender"],
            case["receiver"],
            case["num_nodes"],
        )

    arguments = (
        case["node_features"],
        case["edge_attributes"],
        case["radial_weights"],
    )
    assert torch.autograd.gradcheck(function, arguments)
    assert torch.autograd.gradgradcheck(function, arguments)


def test_the_convolution_is_each_path_coupled_weighted_and_summed():
    """The op never forms every path's messages at once, so check it against
    the formula that does: each path's block of the coupling applied to the
    sender's features and the edge attributes, weighted by that path's radial
    weights per channel, and summed onto the receivers."""
    case = convolution_case(nodes=4, edges=9, width=3, dim_in=4, paths=2, out=6)
    case["path_widths"] = [2, 4]
    coefficients = case["coefficients"]
    gathered = case["node_features"][case["sender"]]
    expected = torch.zeros(4, 3, 6)
    start = 0
    for path, width in enumerate(case["path_widths"]):
        block = coefficients[start : start + width]
        messages = (
            torch.einsum("oid,emi,ed->emo", block, gathered, case["edge_attributes"])
            * case["radial_weights"][:, path, :, None]
        )
        expected[:, :, start : start + width] = torch.zeros(4, 3, width).index_add(
            0, case["receiver"], messages
        )
        start += width
    torch.testing.assert_close(
        channelwise_tp_conv(**case), expected, rtol=1e-12, atol=1e-12
    )


def test_the_convolution_takes_a_structure_with_no_edges():
    case = convolution_case(nodes=3, edges=0)
    out = channelwise_tp_conv(**case)
    assert out.shape == (3, 2, 4) and not out.any()
    out.sum().backward()
    assert case["node_features"].grad is not None


def test_the_node_count_comes_from_the_argument_and_not_from_the_indices():
    """`edge_index.max()` would be a data-dependent host read: a recompile on
    every batch, and a broken CUDA-graph capture. A node no edge reaches still
    has to appear in the output."""
    case = convolution_case(nodes=3, edges=2)
    case["sender"] = torch.tensor([0, 1])
    case["receiver"] = torch.tensor([0, 1])
    case["num_nodes"] = 5
    assert channelwise_tp_conv(**case).shape[0] == 5


# ---------------------------------------------------------------------------
# Compilation
# ---------------------------------------------------------------------------


def test_the_ops_compile_whole_and_do_not_recompile_for_every_batch_size():
    """`fullgraph=True` raises on a graph break, so reaching the assertion is
    already half the claim. The other half is the frame count: five batch sizes
    must not mean five compilations.

    Two frames rather than one is dynamo's own first-call behaviour, a static
    graph followed by the dynamic one it settles on, and not a property of
    these ops. What would fail here is five.
    """
    from torch._dynamo import reset
    from torch._dynamo.testing import CompileCounter

    bases, weights = contraction_case()
    weights = [weight.detach() for weight in weights]
    coefficients = torch.randn(4, 4, 3)

    def step(features, element, attributes, radial, sender, receiver, nodes):
        contracted = symmetric_contraction(
            features, weights, bases, element, len(bases)
        )
        convolved = channelwise_tp_conv(
            features, attributes, radial, coefficients, [2, 2], sender, receiver, nodes
        )
        return segment_sum(contracted.flatten(1), element, 2).sum() + convolved.sum()

    reset()
    counter = CompileCounter()
    compiled = torch.compile(step, backend=counter, fullgraph=True, dynamic=True)
    for nodes in (3, 5, 8, 11, 16):
        edges = 2 * nodes
        compiled(
            torch.randn(nodes, 3, 4),
            torch.zeros(nodes, dtype=torch.long),
            torch.randn(edges, 3),
            torch.randn(edges, 2, 3),
            torch.randint(0, nodes, (edges,)),
            torch.randint(0, nodes, (edges,)),
            nodes,
        )
    assert counter.frame_count <= 2, (
        f"five batch sizes produced {counter.frame_count} compiled graphs. The "
        f"node count is meant to enter as a symbolic dimension, so anything "
        f"approaching one graph per size means something is reading a shape as "
        f"a concrete number."
    )


def test_the_meta_implementations_give_the_right_shape_without_running():
    """What `torch.compile` traces instead of the body. A wrong shape here is a
    wrong graph rather than a wrong number, which surfaces far from the cause."""
    bases, weights = contraction_case()
    with torch.device("meta"):
        features = torch.randn(7, 3, 4)
        element = torch.zeros(7, dtype=torch.long)
        meta_weights = [w.detach().to("meta") for w in weights]
        meta_bases = [b.to("meta") for b in bases]
        out = symmetric_contraction(
            features, meta_weights, meta_bases, element, len(bases)
        )
    assert out.shape == (7, 3, bases[0].shape[2])
