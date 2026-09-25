"""The checks every kernel backend passes, against the reference, op by op.

Point :func:`run_backend_conformance` at a backend and it builds each case the
backend says it supports, next to the reference's build of the same
descriptor, and checks:

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
* **honesty**: a descriptor the backend declines is refused when asked for,
  rather than built into something that fails later.

A case the backend declines is recorded and skipped, never failed: declining
is how a backend says an op is the reference's.

The tolerances are :data:`TOLERANCES`, one row per precision, and a test pins
them. A change to them is its own reviewed change.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import torch
from mace_core.clebsch_gordan.irreps import Irreps
from mace_core.clebsch_gordan.real_basis import wigner_d_real
from mace_core.kernels.capabilities import UnsupportedDescriptorError
from mace_core.kernels.descriptors import (
    ChannelwiseTPConvDescriptor,
    FullyConnectedTPDescriptor,
    LinearDescriptor,
    SegmentReduceDescriptor,
    SymmetricContractionDescriptor,
)
from mace_core.kernels.paths import channelwise_paths
from mace_core.kernels.precision import Precision
from mace_core.kernels.protocol import InternalWeights
from torch import Tensor

from mace_torch.backends.reference import ReferenceBackend
from mace_torch.nn.layout import expanded_irreps

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

#: The op each descriptor type is built by.
_OPS = {
    LinearDescriptor: "linear",
    ChannelwiseTPConvDescriptor: "channelwise_tp_conv",
    SymmetricContractionDescriptor: "symmetric_contraction",
    FullyConnectedTPDescriptor: "fully_connected_tp",
    SegmentReduceDescriptor: "segment_reduce",
}


def conformance_cases(precision: Precision = "float64") -> list[Any]:
    """The descriptors every backend is checked on.

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
        LinearDescriptor(
            irreps_in=node_flat, irreps_out=node_flat, precision=precision
        ),
        LinearDescriptor(
            irreps_in=path_flat, irreps_out=target_flat, precision=precision
        ),
        LinearDescriptor(
            irreps_in=node_flat,
            irreps_out=f"{features}x0e+2x0e",
            has_bias=True,
            precision=precision,
        ),
        ChannelwiseTPConvDescriptor(
            irreps_node=node,
            irreps_edge=edge,
            irreps_out=target,
            num_radial=8,
            num_features=features,
            precision=precision,
        ),
        SymmetricContractionDescriptor(
            irreps_in=target,
            irreps_out=node,
            correlation=3,
            num_elements=2,
            num_features=features,
            precision=precision,
        ),
        FullyConnectedTPDescriptor(
            irreps_in1=node_flat,
            irreps_in2="2x0e",
            irreps_out=node_flat,
            precision=precision,
        ),
        SegmentReduceDescriptor(num_features=6, precision=precision),
    ]


@dataclass
class ConformanceResult:
    """What happened to one case.

    Attributes:
        op: The factory name.
        descriptor: The case.
        built: Whether the backend built it. ``False`` means it declined.
        checks: The checks that ran and passed, by name.
    """

    op: str
    descriptor: Any
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


def _inputs(descriptor: Any, dtype: torch.dtype, device: str, seed: int) -> _Inputs:
    generator = torch.Generator().manual_seed(seed)

    def randn(*shape: int) -> Tensor:
        return torch.randn(*shape, generator=generator, dtype=dtype).to(device)

    def rotated(values: Tensor, irreps: str, rotation: np.ndarray) -> Tensor:
        matrix = torch.tensor(_block_rotation(irreps, rotation), dtype=dtype)
        return values @ matrix.to(device).T

    if isinstance(descriptor, LinearDescriptor):
        source = randn(5, Irreps.parse(descriptor.irreps_in).dimension)

        def rotate_linear(rotation):
            out = _block_rotation(descriptor.irreps_out, rotation)
            return [rotated(source, descriptor.irreps_in, rotation)], out

        rotate = None if descriptor.has_bias else rotate_linear
        return _Inputs([source], [0], rotate)

    if isinstance(descriptor, ChannelwiseTPConvDescriptor):
        features = descriptor.num_features
        paths = channelwise_paths(
            descriptor.irreps_node, descriptor.irreps_edge, descriptor.irreps_out
        )
        node_irreps = expanded_irreps(descriptor.irreps_node, features)
        out_irreps = "+".join(f"{features}x{path.irrep}" for path in paths)
        nodes, sender, receiver = 4, [0, 1, 2, 3, 0, 2, 1], [1, 0, 3, 2, 2, 0, 3]
        node_features = randn(nodes, Irreps.parse(node_irreps).dimension)
        edge_attributes = randn(
            len(sender), Irreps.parse(descriptor.irreps_edge).dimension
        )
        weights = randn(len(sender), len(paths), features)
        senders = torch.tensor(sender, device=device)
        receivers = torch.tensor(receiver, device=device)
        arguments = [node_features, edge_attributes, weights, senders, receivers, nodes]

        def rotate_conv(rotation):
            moved = list(arguments)
            moved[0] = rotated(node_features, node_irreps, rotation)
            moved[1] = rotated(edge_attributes, descriptor.irreps_edge, rotation)
            return moved, _block_rotation(out_irreps, rotation)

        return _Inputs(arguments, [0, 1, 2], rotate_conv)

    if isinstance(descriptor, SymmetricContractionDescriptor):
        features = descriptor.num_features
        in_irreps = expanded_irreps(descriptor.irreps_in, features)
        out_irreps = expanded_irreps(descriptor.irreps_out, features)
        source = randn(5, Irreps.parse(in_irreps).dimension)
        element = torch.tensor([0, 1, 1, 0, 1], device=device) % descriptor.num_elements

        def rotate_contraction(rotation):
            moved = [rotated(source, in_irreps, rotation), element]
            return moved, _block_rotation(out_irreps, rotation)

        return _Inputs([source, element], [0], rotate_contraction)

    if isinstance(descriptor, FullyConnectedTPDescriptor):
        first = randn(5, Irreps.parse(descriptor.irreps_in1).dimension)
        second = randn(5, Irreps.parse(descriptor.irreps_in2).dimension)

        def rotate_skip(rotation):
            moved = [rotated(first, descriptor.irreps_in1, rotation), second]
            return moved, _block_rotation(descriptor.irreps_out, rotation)

        return _Inputs([first, second], [0, 1], rotate_skip)

    if isinstance(descriptor, SegmentReduceDescriptor):
        values = randn(7, descriptor.num_features)
        index = torch.tensor([0, 2, 1, 0, 2, 2, 1], device=device)
        return _Inputs([values, index, 3], [0], None)

    raise TypeError(f"no conformance inputs for {type(descriptor).__name__}")


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
    backend: Any,
    *,
    device: str = "cpu",
    precision: Precision = "float64",
    cases: list[Any] | None = None,
    seed: int = 0,
) -> list[ConformanceResult]:
    """Check every case against the reference. Raises on the first failure.

    Args:
        backend: The backend under test.
        device: Where the ops run.
        precision: The dtype the cases are built at.
        cases: The descriptors. :func:`conformance_cases` by default.
        seed: For the weights, the inputs and the rotation.

    Returns:
        One result per case, including the declined ones.

    Raises:
        AssertionError: Naming the op, the descriptor and the check.
    """
    dtype = _DTYPES[precision]
    reference = ReferenceBackend()
    capabilities = backend.capabilities()
    results = []
    for number, descriptor in enumerate(cases or conformance_cases(precision)):
        op = _OPS[type(descriptor)]
        make = getattr(backend, f"make_{op}", None)
        if op not in capabilities.ops or not capabilities.supports(descriptor):
            if make is not None and op in capabilities.ops:
                try:
                    make(descriptor)
                except UnsupportedDescriptorError:
                    pass
                else:
                    raise AssertionError(
                        f"{backend.name} declines {descriptor} and built it "
                        f"anyway. A declined descriptor must be refused, or the "
                        f"model is built with an op its backend said it cannot "
                        f"compute."
                    )
            results.append(ConformanceResult(op, descriptor, False))
            continue
        if make is None:
            raise AssertionError(f"{backend.name} claims {op} and has no make_{op}")
        result = ConformanceResult(op, descriptor, True)
        where = f"{backend.name} {op} {descriptor}"
        candidate = make(descriptor).to(device)
        expected_op = getattr(reference, f"make_{op}")(descriptor).to(device)

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

        inputs = _inputs(descriptor, dtype, device, seed + number)
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
        mine = candidate(*mine_arguments)
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
                rotated_output = candidate(
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

            def function(*leaves, op=candidate, given=inputs):
                arguments = list(given.arguments)
                for index, leaf in zip(given.differentiable, leaves, strict=True):
                    arguments[index] = leaf
                return op(*arguments)

            if not torch.autograd.gradgradcheck(
                function, tuple(differentiable), atol=1e-6
            ):
                raise AssertionError(f"{where}: gradgradcheck failed")
            result.checks.append("double backward")
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
