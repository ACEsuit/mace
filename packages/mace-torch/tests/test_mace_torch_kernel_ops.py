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

import pytest

if importlib.util.find_spec("torch") is None:  # pragma: no cover
    pytest.skip("the kernel ops need torch", allow_module_level=True)

import torch
from mace_core.clebsch_gordan.reduced_basis import (
    reduced_symmetric_tensor_product_basis,
)
from mace_torch.kernels.ops import (
    channelwise_tp_conv,
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
        basis = torch.tensor(array.reshape(array.shape[0], array.shape[1], -1))
        bases.append(basis)
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
        return symmetric_contraction(x, list(w), bases, element)

    assert torch.autograd.gradcheck(function, (features, *weights))
    assert torch.autograd.gradgradcheck(function, (features, *weights))


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
    before = symmetric_contraction(features, weights, bases, all_element_zero)
    with torch.no_grad():
        for per_order in weights:
            per_order[1].add_(100.0)
    after = symmetric_contraction(features, weights, bases, all_element_zero)
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
        "coefficients": torch.randn(paths, out, dim_in, dim_edge),
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
    coefficients = torch.randn(2, 4, 4, 3)

    def step(features, element, attributes, radial, sender, receiver, nodes):
        contracted = symmetric_contraction(features, weights, bases, element)
        convolved = channelwise_tp_conv(
            features, attributes, radial, coefficients, sender, receiver, nodes
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
        out = symmetric_contraction(features, meta_weights, meta_bases, element)
    assert out.shape == (7, 3, bases[0].shape[1])
