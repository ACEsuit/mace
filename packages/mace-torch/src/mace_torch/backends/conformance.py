"""The checks every kernel backend passes, against the reference, op by op.

Point :func:`run_backend_conformance` at a backend and it builds each case the
backend says it supports, next to the reference's build of the same
kernel_spec, and checks:

* **the weights**: the candidate loads the reference's canonical weights and
  gives the same ones back, which is the whole of why one checkpoint loads into
  any backend without a converter;
* **the values and the first derivatives** against the reference, for the
  inputs and for the weights;
* **equivariance**: rotating the inputs rotates the output by the Wigner
  matrices of its declaration. Parity with an equivariant reference already
  implies it; checked on its own so that a failure says which property broke;
* **the second derivative**, by ``gradgradcheck`` at float64, for a backend
  that claims it. Training on forces differentiates through the backward, and a
  backward that is not itself differentiable gives wrong forces rather than
  none;
* **honesty**: a kernel_spec the backend declines is refused when asked for,
  rather than built into something that fails later;
* **one layout for the chain**: the backend's native feature layout is one it
  declares, and next to the reference the chain of ops resolves to exactly
  one layout, so no op of it transposes its features on the way in or out;
* on request, that the op **compiles without a graph break**, under
  ``torch.compile(fullgraph=True)``, that one compilation **serves every
  size**, since the node and edge counts are meant to enter as symbolic
  dimensions and never be read off a device tensor, and that it **runs inside
  a captured CUDA graph**, replaying to the same values. All three are what an
  inference loop does with a model, and a kernel that syncs with the host or
  allocates on every call fails them rather than degrading quietly.

A case the backend declines is recorded and skipped, never failed: declining
is how a backend says an op is the reference's.

The tolerances are :data:`TOLERANCES`, one row per precision, and a test pins
them. A change to them is its own reviewed change.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field, replace
from typing import Any

import numpy as np
import torch
from mace_core.clebsch_gordan.irreps import Irreps
from mace_core.clebsch_gordan.real_basis import wigner_d_real
from mace_core.kernels.capabilities import UnsupportedKernelSpecError
from mace_core.kernels.paths import channelwise_paths
from mace_core.kernels.precision import Precision
from mace_core.kernels.protocol import InternalWeights
from mace_core.kernels.registry import get_backend
from mace_core.kernels.reorder import decompose
from mace_core.kernels.specs import (
    ActivationLayout,
    ChannelwiseTPConvSpec,
    FullyConnectedTPSpec,
    LinearSpec,
    SegmentReduceSpec,
    SymmetricContractionSpec,
)
from torch import Tensor, nn

from mace_torch.backends.composite import CompositeBackend
from mace_torch.backends.layout import Layout, Terms, expanded_irreps
from mace_torch.backends.reference import ReferenceBackend

__all__ = [
    "TOLERANCES",
    "ConformanceResult",
    "conformance_cases",
    "run_backend_conformance",
]

#: ``(atol, rtol)`` per precision, for a backend against the reference on one
#: op. At float64 the two differ only in the order of the arithmetic; at
#: float32 an accelerated kernel may accumulate differently, and the row is the
#: repository's float32 row.
TOLERANCES: dict[str, tuple[float, float]] = {
    "float64": (1e-10, 1e-10),
    "float32": (5e-5, 1e-3),
}

_DTYPES = {"float64": torch.float64, "float32": torch.float32}

#: The op each kernel_spec type is built by.
_OPS = {
    LinearSpec: "linear",
    ChannelwiseTPConvSpec: "channelwise_tp_conv",
    SymmetricContractionSpec: "symmetric_contraction",
    FullyConnectedTPSpec: "fully_connected_tp",
    SegmentReduceSpec: "segment_reduce",
}


def conformance_cases(precision: Precision = "float64") -> list[Any]:
    """The kernel specifications every backend is checked on.

    The shapes of a MACE layer at small width: the up and down linears with
    and without a bias, the convolution from ``0e+1o`` nodes over harmonics to
    ``lmax = 3``, a correlation-three contraction onto ``0e+1o`` with two
    elements, the skip, and the reduction. A backend is expected to decline
    some of them.
    """
    features = 4
    node = "0e+1o"
    edge = "0e+1o+2e+3o"
    target = "0e+1o+2e+3o"
    paths = channelwise_paths(node, edge, target)
    path_flat = "+".join(f"{features}x{path.irrep}" for path in paths)
    target_flat = expanded_irreps(target, features)
    node_flat = expanded_irreps(node, features)
    return [
        LinearSpec(irreps_in=node_flat, irreps_out=node_flat, precision=precision),
        LinearSpec(irreps_in=path_flat, irreps_out=target_flat, precision=precision),
        LinearSpec(
            irreps_in=node_flat,
            irreps_out=f"{features}x0e+2x0e",
            has_bias=True,
            precision=precision,
        ),
        ChannelwiseTPConvSpec(
            irreps_node=node,
            irreps_edge=edge,
            irreps_out=target,
            num_features=features,
            precision=precision,
        ),
        SymmetricContractionSpec(
            irreps_in=target,
            irreps_out=node,
            correlation=3,
            num_elements=2,
            num_features=features,
            precision=precision,
        ),
        FullyConnectedTPSpec(
            irreps_in1=node_flat,
            irreps_in2="2x0e",
            irreps_out=node_flat,
            precision=precision,
        ),
        SegmentReduceSpec(num_features=6, precision=precision),
    ]


@dataclass
class ConformanceResult:
    """What happened to one case.

    Attributes:
        op: The factory name.
        kernel_spec: The case.
        built: Whether the backend built it. ``False`` means it declined.
        checks: The checks that ran and passed, by name.
    """

    op: str
    kernel_spec: Any
    built: bool
    checks: list[str] = field(default_factory=list)


def _block_rotation(irreps: str, rotation: np.ndarray) -> np.ndarray:
    """The rotation of a grouped value: one Wigner block per copy of a term."""
    blocks = []
    for multiplicity, irrep in Irreps.parse(irreps).terms:
        matrix = wigner_d_real(irrep.degree, rotation)
        blocks.extend([matrix] * multiplicity)
    size = sum(block.shape[0] for block in blocks)
    out = np.zeros((size, size))
    start = 0
    for block in blocks:
        end = start + block.shape[0]
        out[start:end, start:end] = block
        start = end
    return out


def _rotation(seed: int) -> np.ndarray:
    generator = np.random.default_rng(seed)
    matrix, _ = np.linalg.qr(generator.normal(size=(3, 3)))
    if np.linalg.det(matrix) < 0:
        matrix[:, 0] = -matrix[:, 0]
    return matrix


@dataclass
class _Inputs:
    """The arguments of one op, which of them are differentiable, and how each
    one and the output transform under a rotation."""

    arguments: list[Any]
    differentiable: list[int]
    rotate: Callable[[np.ndarray], tuple[list[Any], np.ndarray]] | None


def _inputs(kernel_spec: Any, dtype: torch.dtype, device: str, seed: int) -> _Inputs:
    generator = torch.Generator().manual_seed(seed)

    def randn(*shape: int) -> Tensor:
        return torch.randn(*shape, generator=generator, dtype=dtype).to(device)

    def rotated(values: Tensor, irreps: str, rotation: np.ndarray) -> Tensor:
        matrix = torch.tensor(_block_rotation(irreps, rotation), dtype=dtype)
        return values @ matrix.to(device).T

    if isinstance(kernel_spec, LinearSpec):
        source = randn(5, Irreps.parse(kernel_spec.irreps_in).dimension)

        def rotate_linear(rotation):
            out = _block_rotation(kernel_spec.irreps_out, rotation)
            return [rotated(source, kernel_spec.irreps_in, rotation)], out

        rotate = None if kernel_spec.has_bias else rotate_linear
        return _Inputs([source], [0], rotate)

    if isinstance(kernel_spec, ChannelwiseTPConvSpec):
        features = kernel_spec.num_features
        paths = channelwise_paths(
            kernel_spec.irreps_node, kernel_spec.irreps_edge, kernel_spec.irreps_out
        )
        node_irreps = expanded_irreps(kernel_spec.irreps_node, features)
        out_irreps = "+".join(f"{features}x{path.irrep}" for path in paths)
        nodes, sender, receiver = 4, [0, 1, 2, 3, 0, 2, 1], [1, 0, 3, 2, 2, 0, 3]
        node_features = randn(nodes, Irreps.parse(node_irreps).dimension)
        edge_attributes = randn(
            len(sender), Irreps.parse(kernel_spec.irreps_edge).dimension
        )
        weights = randn(len(sender), len(paths), features)
        senders = torch.tensor(sender, device=device)
        receivers = torch.tensor(receiver, device=device)
        arguments = [node_features, edge_attributes, weights, senders, receivers, nodes]

        def rotate_conv(rotation):
            moved = list(arguments)
            moved[0] = rotated(node_features, node_irreps, rotation)
            moved[1] = rotated(edge_attributes, kernel_spec.irreps_edge, rotation)
            return moved, _block_rotation(out_irreps, rotation)

        return _Inputs(arguments, [0, 1, 2], rotate_conv)

    if isinstance(kernel_spec, SymmetricContractionSpec):
        features = kernel_spec.num_features
        in_irreps = expanded_irreps(kernel_spec.irreps_in, features)
        out_irreps = expanded_irreps(kernel_spec.irreps_out, features)
        source = randn(5, Irreps.parse(in_irreps).dimension)
        element = (
            torch.tensor([0, 1, 1, 0, 1], device=device) % kernel_spec.num_elements
        )

        def rotate_contraction(rotation):
            moved = [rotated(source, in_irreps, rotation), element]
            return moved, _block_rotation(out_irreps, rotation)

        return _Inputs([source, element], [0], rotate_contraction)

    if isinstance(kernel_spec, FullyConnectedTPSpec):
        first = randn(5, Irreps.parse(kernel_spec.irreps_in1).dimension)
        second = randn(5, Irreps.parse(kernel_spec.irreps_in2).dimension)

        def rotate_skip(rotation):
            moved = [rotated(first, kernel_spec.irreps_in1, rotation), second]
            return moved, _block_rotation(kernel_spec.irreps_out, rotation)

        return _Inputs([first, second], [0, 1], rotate_skip)

    if isinstance(kernel_spec, SegmentReduceSpec):
        values = randn(7, kernel_spec.num_features)
        index = torch.tensor([0, 2, 1, 0, 2, 2, 1], device=device)
        return _Inputs([values, index, 3], [0], None)

    raise TypeError(f"no conformance inputs for {type(kernel_spec).__name__}")


# Layout-aware conformance adapts reforge/perf-integration (8dee319d).
def _layout_terms(kernel_spec: Any) -> tuple[dict[int, Terms], Terms]:
    """Which arguments of an op carry features, and the terms of each and of
    the output, so that a value can be moved between layouts."""
    terms = Layout.terms
    if isinstance(kernel_spec, LinearSpec):
        return {0: terms(kernel_spec.irreps_in)}, terms(kernel_spec.irreps_out)
    if isinstance(kernel_spec, ChannelwiseTPConvSpec):
        features = kernel_spec.num_features
        paths = channelwise_paths(
            kernel_spec.irreps_node, kernel_spec.irreps_edge, kernel_spec.irreps_out
        )
        node = terms(expanded_irreps(kernel_spec.irreps_node, features))
        edge = terms(kernel_spec.irreps_edge)
        return {0: node, 1: edge}, tuple(
            (features, path.irrep.dimension) for path in paths
        )
    if isinstance(kernel_spec, SymmetricContractionSpec):
        features = kernel_spec.num_features
        return (
            {0: terms(expanded_irreps(kernel_spec.irreps_in, features))},
            terms(expanded_irreps(kernel_spec.irreps_out, features)),
        )
    if isinstance(kernel_spec, FullyConnectedTPSpec):
        return (
            {0: terms(kernel_spec.irreps_in1), 1: terms(kernel_spec.irreps_in2)},
            terms(kernel_spec.irreps_out),
        )
    return {}, ()


class _InLayout(nn.Module):
    """An op built in another layout, fed and read in the canonical one.

    What lets every check compare it with the reference in ``mul_ir``: the
    inputs are moved into the op's layout and its output back, so a value, a
    gradient or a rotation that disagrees is the op's and not the layout's.
    """

    def __init__(self, op: Any, kernel_spec: Any) -> None:
        super().__init__()
        self.op = op
        self.layout = Layout(kernel_spec.layout)
        self.inputs, self.output = _layout_terms(kernel_spec)

    def forward(self, *arguments: Any) -> Tensor:
        moved = [
            self.layout.from_canonical(value, self.inputs[index])
            if index in self.inputs
            else value
            for index, value in enumerate(arguments)
        ]
        return self.layout.to_canonical(self.op(*moved), self.output)


def _canonical_to(state: Any, like: Tensor) -> Any:
    if isinstance(state, Tensor):
        return (
            state.to(device=like.device, dtype=like.dtype)
            if state.is_floating_point()
            else state.to(like.device)
        )
    return {name: _canonical_to(value, like) for name, value in state.items()}


def _close(actual: Tensor, expected: Tensor, precision: str, what: str) -> None:
    atol, rtol = TOLERANCES[precision]
    if not torch.allclose(actual, expected, atol=atol, rtol=rtol):
        error = (actual - expected).abs().max().item()
        raise AssertionError(
            f"{what}: max |difference| {error:.3e} against the reference, over "
            f"atol {atol:g} and rtol {rtol:g}"
        )


def _flat_canonical(state: Any) -> list[Tensor]:
    if isinstance(state, Tensor):
        return [state]
    return [piece for name in sorted(state) for piece in _flat_canonical(state[name])]


def run_backend_conformance(
    backend: str | Any,
    *,
    device: str = "cpu",
    precision: Precision = "float64",
    cases: list[Any] | None = None,
    seed: int = 0,
    compile_ops: bool = False,
    cuda_graphs: bool = False,
    layout: ActivationLayout = "mul_ir",
    max_block: int = 3,
) -> list[ConformanceResult]:
    """Check every case against the reference. Raises on the first failure.

    Args:
        backend: The backend under test, or the name it is registered under,
            which is resolved through the entry points as a model would resolve
            it.
        device: Where the ops run.
        precision: The dtype the cases are built at.
        cases: The kernel_specs. :func:`conformance_cases` by default.
        seed: For the weights, the inputs and the rotation.
        compile_ops: Also compile each op with ``fullgraph=True``, and check
            that one compilation serves inputs of three sizes.
        cuda_graphs: Also capture each op in a CUDA graph and replay it.
        layout: The activation layout to test against canonical reference values.
        max_block: The largest block an op's canonical to internal weight map
            may have. Three is the largest measured on the production grid.

    Returns:
        One result per case, including the declined ones.

    Raises:
        AssertionError: Naming the op, the kernel_spec and the check.
    """
    if isinstance(backend, str):
        backend = get_backend(backend)
    dtype = _DTYPES[precision]
    reference = ReferenceBackend()
    capabilities = backend.capabilities()
    _check_chain_layout(backend, capabilities)
    Layout(layout)  # Validate before iterating, including a declined layout.
    results = []
    given = cases or conformance_cases(precision)
    for number, kernel_spec in enumerate(given):
        kernel_spec = replace(kernel_spec, layout=layout)
        op = _OPS[type(kernel_spec)]
        make = getattr(backend, f"make_{op}", None)
        if op not in capabilities.ops or not capabilities.supports(kernel_spec):
            if make is not None and op in capabilities.ops:
                try:
                    make(kernel_spec)
                except UnsupportedKernelSpecError:
                    pass
                else:
                    raise AssertionError(
                        f"{backend.name} declines {kernel_spec} and built it "
                        f"anyway. A declined kernel_spec must be refused, or the "
                        f"model is built with an op its backend said it cannot "
                        f"compute."
                    )
            results.append(ConformanceResult(op, kernel_spec, False))
            continue
        if make is None:
            raise AssertionError(f"{backend.name} claims {op} and has no make_{op}")
        result = ConformanceResult(op, kernel_spec, True, ["layout"])
        where = f"{backend.name} {op} {kernel_spec}"
        candidate = make(kernel_spec).to(device)
        canonical = replace(kernel_spec, layout="mul_ir", product="ieee")
        expected_op = getattr(reference, f"make_{op}")(canonical).to(device)
        runner = candidate if layout == "mul_ir" else _InLayout(candidate, kernel_spec)

        if isinstance(expected_op, InternalWeights):
            if not isinstance(candidate, InternalWeights):
                raise AssertionError(
                    f"{where}: holds weights and has no canonical form"
                )
            expected_op.initialize_weights(seed + number)
            state = expected_op.to_canonical()
            like = next(expected_op.parameters())
            try:
                candidate.load_canonical(_canonical_to(state, like))
            except (RuntimeError, KeyError, ValueError) as failure:
                raise AssertionError(
                    f"{where}: cannot load the reference's canonical weights: {failure}"
                ) from failure
            for mine, theirs in zip(
                _flat_canonical(candidate.to_canonical()),
                _flat_canonical(state),
                strict=True,
            ):
                if mine.is_floating_point():
                    _close(
                        mine.to(theirs.device, theirs.dtype),
                        theirs,
                        precision,
                        f"{where} canonical round trip",
                    )
                elif not torch.equal(mine.cpu(), theirs.cpu()):
                    raise AssertionError(
                        f"{where}: canonical round trip changed {theirs}"
                    )
            result.checks.append("weights")
            if precision == "float64":
                _check_reorder(candidate, state, like, max_block, where)
                result.checks.append("reorder")

        inputs = _inputs(kernel_spec, dtype, device, seed + number)
        mine_arguments = [
            value.detach().clone().requires_grad_(index in inputs.differentiable)
            if isinstance(value, Tensor) and value.is_floating_point()
            else value
            for index, value in enumerate(inputs.arguments)
        ]
        their_arguments = [
            value.detach().clone().requires_grad_(index in inputs.differentiable)
            if isinstance(value, Tensor) and value.is_floating_point()
            else value
            for index, value in enumerate(inputs.arguments)
        ]
        mine = runner(*mine_arguments)
        theirs = expected_op(*their_arguments)
        _close(mine, theirs, precision, f"{where} values")
        result.checks.append("values")

        projection = torch.randn(
            theirs.shape, generator=torch.Generator().manual_seed(seed), dtype=dtype
        ).to(device)
        mine_leaves = [mine_arguments[i] for i in inputs.differentiable] + [
            p for p in candidate.parameters() if p.requires_grad
        ]
        their_leaves = [their_arguments[i] for i in inputs.differentiable]
        mine_grads = torch.autograd.grad((mine * projection).sum(), mine_leaves)
        their_grads = torch.autograd.grad((theirs * projection).sum(), their_leaves)
        for position, (a, b) in enumerate(
            zip(mine_grads[: len(their_leaves)], their_grads, strict=True)
        ):
            _close(
                a,
                b,
                precision,
                f"{where} gradient of input {inputs.differentiable[position]}",
            )
        result.checks.append("gradients")

        if inputs.rotate is not None:
            rotation = _rotation(seed + number)
            moved, output_rotation = inputs.rotate(rotation)
            with torch.no_grad():
                rotated_output = runner(
                    *[
                        value.detach() if isinstance(value, Tensor) else value
                        for value in moved
                    ]
                )
                expected_rotated = (
                    mine.detach()
                    @ torch.tensor(output_rotation, dtype=dtype, device=device).T
                )
            _close(rotated_output, expected_rotated, precision, f"{where} equivariance")
            result.checks.append("equivariance")

        if capabilities.supports_double_backward and precision == "float64":
            differentiable = [
                inputs.arguments[i].detach().clone().requires_grad_()
                for i in inputs.differentiable
            ]

            def function(*leaves, op=runner, given=inputs):
                arguments = list(given.arguments)
                for index, leaf in zip(given.differentiable, leaves, strict=True):
                    arguments[index] = leaf
                return op(*arguments)

            # Projected finite differences keep CPU backend conformance tractable.
            # The kernel suite also checks the complete second-order Jacobians.
            if not torch.autograd.gradgradcheck(
                function, tuple(differentiable), atol=1e-6, fast_mode=True
            ):
                raise AssertionError(f"{where}: gradgradcheck failed")
            result.checks.append("double backward")
        detached = [
            value.detach() if isinstance(value, Tensor) else value
            for value in inputs.arguments
        ]
        if isinstance(kernel_spec, ChannelwiseTPConvSpec):
            _check_no_edges(runner, detached, theirs.shape[1], where)
            result.checks.append("no edges")
        if compile_ops:
            torch._dynamo.reset()
            compiled = torch.compile(runner, fullgraph=True, dynamic=False)
            with torch.no_grad():
                _close(
                    compiled(*detached), mine.detach(), precision, f"{where} compiled"
                )
            result.checks.append("compiles")
            _check_every_size(runner, kernel_spec, detached, precision, where)
            result.checks.append("one compile for every size")
        if cuda_graphs:
            _check_cuda_graph(runner, detached, mine.detach(), precision, where)
            result.checks.append("cuda graph")
        if isinstance(candidate, InternalWeights) and any(
            p.requires_grad for p in candidate.parameters()
        ):
            _check_weight_gradients(
                candidate,
                expected_op,
                their_arguments,
                mine_grads[len(their_leaves) :],
                projection,
                precision,
                where,
            )
            result.checks.append("weight gradients")
        results.append(result)
    return results


def _flat_floats(state: Any) -> list[Tensor]:
    return [piece for piece in _flat_canonical(state) if piece.is_floating_point()]


def _with_floats(state: Any, values: Tensor) -> Any:
    """``state`` with its floating tensors replaced, in order, from ``values``."""
    offset = 0

    def rebuild(node: Any) -> Any:
        nonlocal offset
        if isinstance(node, Tensor):
            if not node.is_floating_point():
                return node
            piece = values[offset : offset + node.numel()].reshape(node.shape)
            offset += node.numel()
            return piece.to(node)
        return {name: rebuild(node[name]) for name in sorted(node)}

    return rebuild(state)


def _check_reorder(
    candidate: Any, state: Any, like: Tensor, max_block: int, where: str
) -> None:
    """The map from canonical weights to the op's own is block diagonal.

    Derived from the op, not asked of it: one canonical weight direction is
    loaded at a time and the op's parameters are read back, which gives the map
    whatever the backend does inside ``load_canonical``. The map has to split
    into independent square blocks, each invertible, none larger than
    ``max_block``, and a canonical state has to come back exactly. A future
    backend release whose internal order made the map dense fails here instead
    of silently reinterpreting a checkpoint.
    """
    canonical = _canonical_to(state, like)
    width = sum(piece.numel() for piece in _flat_floats(canonical))
    parameters = list(candidate.parameters())
    columns = []
    try:
        with torch.no_grad():
            for index in range(width):
                direction = torch.zeros(width, dtype=torch.float64)
                direction[index] = 1.0
                candidate.load_canonical(_with_floats(canonical, direction))
                columns.append(
                    torch.cat(
                        [p.detach().reshape(-1).double().cpu() for p in parameters]
                    )
                )
            weights = torch.randn(
                width, generator=torch.Generator().manual_seed(11), dtype=torch.float64
            )
            candidate.load_canonical(_with_floats(canonical, weights))
            back = torch.cat(
                [
                    p.double().reshape(-1).cpu()
                    for p in _flat_floats(candidate.to_canonical())
                ]
            )
    finally:
        candidate.load_canonical(canonical)
    held = torch.stack(columns, dim=1).numpy()
    blocks = decompose(held.T)
    if sum(len(block.canonical) for block in blocks) != width:
        raise AssertionError(
            f"{where}: some canonical weight reaches none of the op's own"
        )
    for block in blocks:
        if len(block.canonical) != len(block.backend):
            raise AssertionError(
                f"{where}: a block maps {len(block.canonical)} canonical weights "
                f"onto {len(block.backend)} of its own, so the map is not invertible"
            )
        if len(block.canonical) > max_block:
            raise AssertionError(
                f"{where}: a block couples {len(block.canonical)} weights, over "
                f"the bound of {max_block}; the map is closer to dense than the "
                f"layout allows"
            )
        if np.linalg.cond(block.matrix) > 1e8:
            raise AssertionError(f"{where}: a block of the weight map is singular")
    if not torch.allclose(back, weights, rtol=0, atol=1e-12):
        error = (back - weights).abs().max().item()
        raise AssertionError(
            f"{where}: a canonical state comes back off by {error:.2e} through "
            f"the op's own weights"
        )


def _check_weight_gradients(
    candidate: Any,
    reference: Any,
    arguments: list[Any],
    candidate_gradients: tuple[Tensor, ...],
    projection: Tensor,
    precision: str,
    where: str,
) -> None:
    """Compare the weight gradients along directions of the canonical form.

    A backend holds its weights in its own layout, and a gradient transforms
    against the map from the canonical one, not with it. What both backends
    must agree on is the derivative along a direction of canonical weights: the
    direction loaded into each op gives that op's own parameter direction, and
    the gradient projected on it is the directional derivative.
    """
    parameters = [p for p in reference.parameters() if p.requires_grad]
    gradients = torch.autograd.grad(
        (reference(*arguments) * projection).sum(), parameters
    )
    mine = [p for p in candidate.parameters() if p.requires_grad]
    reference_state = reference.to_canonical()
    candidate_state = candidate.to_canonical()
    generator = torch.Generator().manual_seed(7)

    def randomized(state: Any) -> Any:
        if isinstance(state, Tensor):
            if not state.is_floating_point():
                return state
            return torch.randn(
                state.shape, generator=generator, dtype=torch.float64
            ).to(state)
        return {name: randomized(value) for name, value in state.items()}

    try:
        for _ in range(3):
            direction = randomized(reference_state)
            with torch.no_grad():
                reference.load_canonical(direction)
                candidate.load_canonical(_canonical_to(direction, mine[0]))
            expected = sum(
                (g * p.detach()).sum()
                for g, p in zip(gradients, parameters, strict=True)
            )
            actual = sum(
                (g * p.detach()).sum()
                for g, p in zip(candidate_gradients, mine, strict=True)
            )
            _close(
                torch.as_tensor(actual).reshape(1),
                torch.as_tensor(expected).reshape(1).to(torch.as_tensor(actual)),
                precision,
                f"{where} weight gradient along a canonical direction",
            )
    finally:
        reference.load_canonical(reference_state)
        candidate.load_canonical(candidate_state)


def _check_chain_layout(backend: Any, capabilities: Any) -> None:
    """The chain of ops runs in one layout, and it is one the backend declares.

    What a model is built in is decided once, when the backend is resolved
    next to the reference: the backend's native layout if the reference can
    follow it, the canonical one if not. Either way every op of the chain
    reads and writes that one layout, so no op transposes its features at its
    boundary, and that only holds if the backend really builds its ops in the
    layout it calls native.
    """
    if "mul_ir" not in capabilities.layouts:
        raise AssertionError(f"{backend.name} cannot load canonical mul_ir weights")
    native = capabilities.native_layout
    if native not in capabilities.activation_layouts:
        raise AssertionError(
            f"{backend.name} calls {native!r} its native layout and declares "
            f"only {sorted(capabilities.activation_layouts)}: the chain would "
            f"be built in a layout its own ops decline."
        )
    chain = CompositeBackend(backend, ReferenceBackend()).layout.name
    if chain not in capabilities.activation_layouts:
        raise AssertionError(
            f"{backend.name} next to the reference resolves the chain to "
            f"{chain!r}, a layout its ops decline, so every op would transpose "
            f"at its boundary."
        )


def _tiled(kernel_spec: Any, arguments: list[Any], copies: int) -> list[Any]:
    """``copies`` independent copies of one op's inputs, side by side.

    Node rows, edge rows and segment rows repeat, and the indices of each copy
    point into its own nodes, so the result is the same op on a structure
    ``copies`` times larger.
    """
    if copies == 1:
        return list(arguments)

    def rows(value: Tensor) -> Tensor:
        return torch.cat([value] * copies)

    if isinstance(kernel_spec, ChannelwiseTPConvSpec):
        nodes, attributes, weights, sender, receiver, num_nodes = arguments
        offsets = [num_nodes * copy for copy in range(copies)]
        return [
            rows(nodes),
            rows(attributes),
            rows(weights),
            torch.cat([sender + offset for offset in offsets]),
            torch.cat([receiver + offset for offset in offsets]),
            num_nodes * copies,
        ]
    if isinstance(kernel_spec, SegmentReduceSpec):
        values, index, segments = arguments
        return [
            rows(values),
            torch.cat([index + segments * copy for copy in range(copies)]),
            segments * copies,
        ]
    return [rows(value) if isinstance(value, Tensor) else value for value in arguments]


def _check_every_size(
    candidate: Any, kernel_spec: Any, arguments: list[Any], precision: str, where: str
) -> None:
    """One compilation for inputs of three sizes, each equal to eager.

    Counted with dynamo's own counter, through AOT autograd, so the op's fake
    implementation is what gets traced. Two frames rather than one is dynamo's
    first-call behaviour and allowed; one per size means a size is being read
    as a concrete number.

    The sizes are two, three and five copies of the case, which keeps every
    node, edge and row count off every channel and component count. Dynamo
    gives two sizes that are equal at the first call one symbol, and a node
    count that happened to equal the channel count would be recompiled the
    first time the two parted, which is the test's coincidence rather than
    the op's.
    """
    from torch._dynamo.testing import CompileCounterWithBackend

    torch._dynamo.reset()
    counter = CompileCounterWithBackend("aot_eager")
    compiled = torch.compile(candidate, backend=counter, fullgraph=True, dynamic=True)
    sizes = (2, 3, 5)
    for copies in sizes:
        tiled = _tiled(kernel_spec, arguments, copies)
        with torch.no_grad():
            _close(
                compiled(*tiled),
                candidate(*tiled),
                precision,
                f"{where} compiled at {copies} times the size",
            )
    torch._dynamo.reset()
    if counter.frame_count > 2:
        raise AssertionError(
            f"{where}: {len(sizes)} sizes compiled {counter.frame_count} times. "
            f"The sizes are meant to enter as symbolic dimensions; one graph "
            f"per size means one of them is read as a concrete number."
        )


def _check_cuda_graph(
    candidate: Any, arguments: list[Any], expected: Tensor, precision: str, where: str
) -> None:
    """Capture one forward in a CUDA graph, replay it, compare.

    Warmed up on a side stream first, as capture requires: the first calls may
    allocate or pick a kernel, and neither may happen inside a capture.
    """
    static = [
        value.clone() if isinstance(value, Tensor) else value for value in arguments
    ]
    side = torch.cuda.Stream()
    side.wait_stream(torch.cuda.current_stream())
    with torch.no_grad(), torch.cuda.stream(side):
        for _ in range(3):
            candidate(*static)
    torch.cuda.current_stream().wait_stream(side)
    graph = torch.cuda.CUDAGraph()
    with torch.no_grad(), torch.cuda.graph(graph):
        output = candidate(*static)
    graph.replay()
    torch.cuda.synchronize()
    _close(output, expected, precision, f"{where} replayed from a CUDA graph")


def _check_no_edges(
    candidate: Any, arguments: list[Any], width: int, where: str
) -> None:
    """A structure with no edges, such as a single atom, gets one zero row per
    node. Inferring a width from zero rows is how this breaks."""
    nodes, edge_attributes, weights, sender, receiver, num_nodes = arguments
    with torch.no_grad():
        output = candidate(
            nodes,
            edge_attributes[:0],
            weights[:0],
            sender[:0],
            receiver[:0],
            num_nodes,
        )
    if tuple(output.shape) != (num_nodes, width) or bool(output.any()):
        raise AssertionError(
            f"{where}: with no edges it returned {tuple(output.shape)} "
            f"{'with values' if output.numel() and output.any() else ''}, and "
            f"expected zeros of shape {(num_nodes, width)}"
        )
