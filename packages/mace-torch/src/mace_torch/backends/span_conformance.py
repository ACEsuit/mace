"""Conformance for stateless edge-convolution spans, against reference ops.

This harness owns no accelerated implementation. Install a backend normally
and pass its registered name. Declined shapes are reported, not substituted.
The tolerance table is the existing backend-conformance table.
"""

from __future__ import annotations

from dataclasses import replace

import torch
from mace_core.config import (
    AgnesiTransformConfig,
    NoDistanceTransformConfig,
    SoftTransformConfig,
)
from mace_core.edges import plan_edges
from mace_core.kernels.descriptors import (
    ActivationLayout,
    ChannelwiseTPConvDescriptor,
    DistanceTransformDescriptor,
    RadialBasisDescriptor,
    RadialMLPDescriptor,
    SphericalHarmonicsDescriptor,
)
from mace_core.kernels.precision import Precision
from mace_core.kernels.registry import get_backend
from mace_core.kernels.spans import EdgeConvolutionSpan
from torch import Tensor, nn

from mace_torch.backends.conformance import (
    ConformanceResult,
    _block_rotation,
    _check_cuda_graph,
    _close,
    _rotation,
)
from mace_torch.backends.layout import Layout, expanded_irreps
from mace_torch.backends.reference import ReferenceBackend
from mace_torch.backends.spans import validate_span
from mace_torch.graph import EdgeRuns, apply_edge_plan
from mace_torch.nn.edges import EdgeInputs
from mace_torch.nn.embedding import RadialEmbeddingBlock, build_distance_transform

__all__ = ["run_span_conformance", "span_conformance_cases"]


def span_conformance_cases(
    precision: Precision = "float64",
) -> list[EdgeConvolutionSpan]:
    """Small angular convolutions, all bases and both distance transforms."""
    convolution = ChannelwiseTPConvDescriptor(
        irreps_node="0e+1o",
        irreps_edge="0e+1o",
        irreps_out="0e+1o",
        num_features=2,
        precision=precision,
    )
    base = EdgeConvolutionSpan(
        RadialBasisDescriptor(num_basis=3, cutoff=3.0, precision=precision),
        SphericalHarmonicsDescriptor(lmax=1, precision=precision),
        RadialMLPDescriptor(
            channels=(3, 4, len(convolution.paths) * 2), precision=precision
        ),
        convolution,
    )
    conditioned = replace(
        base.radial_network,
        channels=(7, 4, len(convolution.paths) * 2),
        kind="layer_norm_silu",
        num_elements=3,
        element_embedding_width=2,
        layer_norm_epsilon=3e-4,
    )
    return [
        base,
        replace(base, radial_basis=replace(base.radial_basis, kind="gaussian")),
        replace(base, radial_basis=replace(base.radial_basis, kind="chebyshev")),
        replace(
            base,
            distance_transform=DistanceTransformDescriptor(
                kind="agnesi", precision=precision
            ),
        ),
        replace(
            base,
            distance_transform=DistanceTransformDescriptor(
                kind="soft", precision=precision
            ),
        ),
        replace(
            base,
            density=RadialMLPDescriptor(channels=(3, 1), precision=precision),
        ),
        replace(base, radial_network=conditioned),
        replace(
            base,
            radial_network=conditioned,
            density=replace(conditioned, channels=(7, 3, 1)),
        ),
    ]


class _ReferenceConvolution(nn.Module):
    def __init__(self, shape: EdgeConvolutionSpan, device: str) -> None:
        super().__init__()
        backend = ReferenceBackend()
        basis = backend.make_radial_basis(shape.radial_basis)
        transform = (
            NoDistanceTransformConfig()
            if shape.distance_transform is None
            else AgnesiTransformConfig()
            if shape.distance_transform.kind == "agnesi"
            else SoftTransformConfig()
        )
        self.radial = RadialEmbeddingBlock(
            basis.basis, build_distance_transform(transform), basis.cutoff
        )
        self.harmonics = backend.make_spherical_harmonics(shape.spherical_harmonics)
        self.network = backend.make_radial_mlp(shape.radial_network)
        self.network.initialize_weights(321)
        self.density = (
            None if shape.density is None else backend.make_radial_mlp(shape.density)
        )
        if self.density is not None:
            self.density.initialize_weights(123)
        self.tables = nn.ParameterList()
        if shape.radial_network.num_elements:
            generator = torch.Generator().manual_seed(981)
            for _ in range(2):
                self.tables.append(
                    nn.Parameter(
                        torch.randn(
                            shape.radial_network.num_elements,
                            shape.radial_network.element_embedding_width,
                            generator=generator,
                            dtype=getattr(torch, shape.radial_network.precision),
                        )
                    )
                )
        self.convolution = backend.make_channelwise_tp_conv(shape.convolution)
        self.shape = shape
        self.to(device=device, dtype=getattr(torch, shape.convolution.precision))
        live = self.member_weights()
        indices = {id(value): index for index, value in enumerate(live)}
        self.weight_names = tuple(
            (name, indices[id(value)])
            for name, value in (*self.named_parameters(), *self.named_buffers())
            if id(value) in indices
        )

    def member_weights(self) -> tuple[Tensor, ...]:
        return (
            self.radial.span_weights()
            + self.network.span_weights()
            + (() if self.density is None else self.density.span_weights())
            + tuple(self.tables)
        )

    def forward(
        self,
        features: Tensor,
        edges: EdgeInputs,
        sender: Tensor,
        receiver: Tensor,
        nodes: int,
    ) -> Tensor | tuple[Tensor, Tensor]:
        radial, _ = self.radial(
            edges.vectors.norm(dim=-1, keepdim=True),
            edges.node_atomic_numbers,
            edges.edge_index,
        )
        if self.tables:
            assert edges.node_elements is not None
            radial = torch.cat(
                (
                    radial,
                    self.tables[0][edges.node_elements[sender]],
                    self.tables[1][edges.node_elements[receiver]],
                ),
                dim=-1,
            )
        weights = self.network(radial).reshape(
            -1, len(self.shape.convolution.paths), self.shape.convolution.num_features
        )
        message = self.convolution(
            features, self.harmonics(edges.vectors), weights, sender, receiver, nodes
        )
        if self.density is None:
            return message
        density = radial.new_zeros((nodes, 1)).index_add(
            0, receiver, torch.tanh(self.density(radial).square())
        )
        return message, density


def run_span_conformance(
    backend_name: str,
    device: str = "cpu",
    precision: Precision = "float64",
    layout: ActivationLayout = "mul_ir",
    training: bool = False,
    compile_spans: bool = False,
    cuda_graphs: bool = False,
) -> list[ConformanceResult]:
    """Check declared spans, including canonical weight gradients and double backward.

    CUDA-graph callers must run under their GPU capability gate. CPU requests
    for capture raise rather than reporting an unexecuted check as successful.
    """
    Layout(layout)
    if cuda_graphs and (
        torch.device(device).type != "cuda" or not torch.cuda.is_available()
    ):
        raise RuntimeError("CUDA graph conformance requires an available CUDA device.")
    backend = get_backend(backend_name)
    if backend.make_interaction_layer(()) is not None:
        raise AssertionError("an unrecognized span tuple must return None")
    results = []
    for shape in span_conformance_cases(precision):
        members = tuple(replace(member, layout=layout) for member in shape.descriptors)
        span = backend.make_interaction_layer(members)
        result = ConformanceResult(
            "interaction_layer", shape, span is not None, ["decline"]
        )
        results.append(result)
        if span is None:
            continue
        validate_span(span, backend_name)
        span.train(training)
        result.checks.append("stateless")
        oracle = _ReferenceConvolution(shape, device)
        _check_span(
            span,
            oracle,
            backend.capabilities().edge_features,
            device,
            precision,
            result,
            compile_spans,
            cuda_graphs,
            Layout(layout),
        )
    return results


def _check_span(
    span,
    oracle,
    features,
    device,
    precision,
    result,
    compile_spans,
    cuda_graphs,
    layout,
):
    dtype = getattr(torch, precision)
    positions = torch.tensor(
        [
            [0.1, 0.2, 0.3],
            [0.7, -0.2, 0.1],
            [1.1, 0.8, -0.1],
            [-0.3, 0.6, 0.4],
            [0.0, 0.0, 0.0],
        ],
        dtype=dtype,
        device=device,
    )
    generator = torch.Generator().manual_seed(437)
    nodes = torch.randn((5, 8), generator=generator, dtype=dtype).to(device)
    edge_index = torch.tensor([[2, 0, 1, 3, 0, 2], [0, 1, 2, 0, 3, 1]], device=device)
    atomic_numbers = torch.tensor([1, 6, 8, 1, 1], device=device)
    weights = tuple(value.detach().clone() for value in oracle.member_weights())
    shape = oracle.shape
    plan = plan_edges(features, frozenset())
    input_terms = layout.terms(expanded_irreps(shape.convolution.irreps_node, 2))
    output_terms = layout.terms(expanded_irreps(shape.convolution.irreps_paths, 2))
    elements = torch.tensor([0, 1, 2, 0, 0], device=device)

    def runner(indices, shifts, candidate):
        graph = apply_edge_plan(
            {"positions": positions, "edge_index": indices, "shifts": shifts}, plan
        )
        sender, receiver = graph["edge_index"]
        receiver_runs = (
            EdgeRuns.from_graph(graph, "receiver")
            if "receiver_runs" in features
            else None
        )
        sender_runs = (
            EdgeRuns.from_graph(graph, "sender") if "sender_runs" in features else None
        )
        runs = receiver_runs if receiver_runs is not None else sender_runs

        def run(points, node_features, *member_weights):
            edges = EdgeInputs(
                points[receiver] - points[sender] + graph["shifts"],
                atomic_numbers,
                graph["edge_index"],
                oracle.harmonics,
                oracle.radial,
                receiver_runs=receiver_runs,
                sender_runs=sender_runs,
                node_elements=elements,
            )
            if candidate:
                output = span(
                    layout.from_canonical(node_features, input_terms),
                    edges,
                    sender,
                    receiver,
                    len(points),
                    tuple(member_weights),
                    runs,
                )
                if isinstance(output, tuple):
                    message, density = output
                    return torch.cat(
                        (layout.to_canonical(message, output_terms), density), dim=-1
                    )
                return layout.to_canonical(output, output_terms)
            state = {name: member_weights[index] for name, index in oracle.weight_names}
            output = torch.func.functional_call(
                oracle, state, (node_features, edges, sender, receiver, len(points))
            )
            return torch.cat(output, dim=-1) if isinstance(output, tuple) else output

        return run

    shifts = positions.new_zeros((edge_index.shape[1], 3))
    candidate = runner(edge_index, shifts, True)
    reference = runner(edge_index, shifts, False)
    arguments = (positions, nodes, *weights)
    mine_inputs = tuple(value.clone().requires_grad_(True) for value in arguments)
    theirs_inputs = tuple(value.clone().requires_grad_(True) for value in arguments)
    actual, expected = candidate(*mine_inputs), reference(*theirs_inputs)
    _close(actual, expected, precision, "span values")
    result.checks.append("values")
    count = len(arguments) if span.capabilities.weight_gradients else 2
    seed = torch.randn(expected.shape, generator=generator, dtype=dtype).to(device)
    mine = torch.autograd.grad(
        actual, mine_inputs[:count], seed, create_graph=True, allow_unused=True
    )
    theirs = torch.autograd.grad(
        expected, theirs_inputs[:count], seed, create_graph=True, allow_unused=True
    )
    for index, (value, wanted) in enumerate(zip(mine, theirs, strict=True)):
        _close(
            torch.zeros_like(mine_inputs[index]) if value is None else value,
            torch.zeros_like(theirs_inputs[index]) if wanted is None else wanted,
            precision,
            f"span gradient {index}",
        )
    result.checks.append("first derivatives")
    if span.capabilities.weight_gradients:
        result.checks.append("weight gradients")
    if precision == "float64" and span.capabilities.double_backward:
        differentiable = (
            mine_inputs if span.capabilities.weight_gradients else mine_inputs[:2]
        )

        def twice(*values):
            return candidate(*values, *arguments[len(values) :])

        assert torch.autograd.gradgradcheck(
            twice, differentiable, atol=1e-6, fast_mode=True
        )
        result.checks.append("double backward")

    rotation = _rotation(13)
    # wigner_d_real returns the matrix acting on feature rows, without a transpose.
    matrix = torch.tensor(rotation, dtype=dtype, device=device)
    node_rotation = torch.tensor(
        _block_rotation(expanded_irreps(shape.convolution.irreps_node, 2), rotation),
        dtype=dtype,
        device=device,
    )
    output_rotation = torch.tensor(
        _block_rotation(expanded_irreps(shape.convolution.irreps_paths, 2), rotation),
        dtype=dtype,
        device=device,
    )
    if shape.density is not None:
        output_rotation = torch.block_diag(output_rotation, nodes.new_ones((1, 1)))
    _close(
        candidate(positions @ matrix.T, nodes @ node_rotation, *weights),
        expected.detach() @ output_rotation,
        precision,
        "rotation",
    )
    parity_in = torch.tensor([1, 1, -1, -1, -1, -1, -1, -1], dtype=dtype, device=device)
    from mace_core.clebsch_gordan.irreps import Irreps

    parity_out = nodes.new_tensor(
        [
            ir.parity
            for mul, ir in Irreps.parse(
                expanded_irreps(shape.convolution.irreps_paths, 2)
            )
            for _ in range(mul * ir.dimension)
        ]
    )
    if shape.density is not None:
        parity_out = torch.cat((parity_out, nodes.new_ones(1)))
    _close(
        candidate(-positions, nodes * parity_in, *weights),
        expected.detach() * parity_out,
        precision,
        "inversion",
    )
    _close(
        candidate(positions + positions.new_tensor([1.1, -0.3, 2.0]), nodes, *weights),
        expected.detach(),
        precision,
        "translation",
    )
    result.checks.append("equivariance")
    reversed_run = runner(edge_index.flip(1), shifts.flip(0), True)
    _close(reversed_run(*arguments), expected.detach(), precision, "edge order")
    result.checks.append("edge order")
    empty = runner(edge_index[:, :0], shifts[:0], True)(*arguments)
    assert torch.count_nonzero(empty) == 0
    padded = runner(
        torch.cat((edge_index, edge_index.new_tensor([[4], [4]])), dim=1),
        torch.cat(
            (shifts, positions.new_tensor([[2 * shape.radial_basis.cutoff, 0, 0]]))
        ),
        True,
    )
    # Padding edges belong to the dummy node, as in pad_batch. A conditioned
    # biased network need not vanish at a zero radial input.
    _close(padded(*arguments)[:4], expected.detach()[:4], precision, "padding")
    result.checks.extend(("empty", "padded"))
    if compile_spans or cuda_graphs:
        # Each shape is an independent compilation; closures share one code
        # object and otherwise exhaust Dynamo's per-code specialization limit.
        torch._dynamo.reset()
        # The explicit capture below owns the CUDA graph. reduce-overhead would
        # replay its own graph inside that capture, which CUDA does not allow.
        compiled = torch.compile(candidate, fullgraph=True, dynamic=False)
        with torch.no_grad():
            _close(compiled(*arguments), expected.detach(), precision, "compiled")
        result.checks.append("compiles")
        if cuda_graphs:
            _check_cuda_graph(
                compiled, list(arguments), expected.detach(), precision, "span"
            )
            result.checks.append("cuda graph")
    validate_span(span, "conformance candidate")
