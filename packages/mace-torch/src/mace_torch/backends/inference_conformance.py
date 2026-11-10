"""Value and input-derivative checks for explicitly prepared inference regions."""

from __future__ import annotations

from collections.abc import Callable, Sequence

import torch
from torch import Tensor

from mace_torch.backends.conformance import TOLERANCES
from mace_torch.backends.inference import PreparedInference

__all__ = ["run_inference_conformance"]


def _evaluate(function, arguments, twice):
    inputs = tuple(
        tensor.detach().clone().requires_grad_(tensor.is_floating_point())
        for tensor in arguments
    )
    original = tuple(tensor.detach().clone() for tensor in inputs)
    outputs = function(inputs)
    for actual, before in zip(inputs, original, strict=True):
        torch.testing.assert_close(actual, before, atol=0, rtol=0)
    differentiable = tuple(tensor for tensor in inputs if tensor.requires_grad)
    # Nonuniform cotangents exercise every returned feature and readout component.
    objective = torch.stack(
        [
            (
                output
                * torch.arange(1, output.numel() + 1, device=output.device).reshape(
                    output.shape
                )
            ).sum()
            for output in outputs
        ]
    ).sum()
    gradients = torch.autograd.grad(
        objective, differentiable, create_graph=twice, allow_unused=True
    )
    second = ()
    if twice:
        terms = [
            gradient.square().sum()
            for gradient in gradients
            if gradient is not None and gradient.requires_grad
        ]
        if terms:
            second = torch.autograd.grad(
                torch.stack(terms).sum(), differentiable, allow_unused=True
            )
        else:
            second = (None,) * len(differentiable)
    # An unused input and an explicitly returned zero adjoint are equivalent.
    gradients = tuple(
        torch.zeros_like(tensor) if gradient is None else gradient
        for tensor, gradient in zip(differentiable, gradients, strict=True)
    )
    if twice:
        second = tuple(
            torch.zeros_like(tensor) if gradient is None else gradient
            for tensor, gradient in zip(differentiable, second, strict=True)
        )
    return (*outputs, *gradients, *second)


def run_inference_conformance(
    prepared: PreparedInference,
    reference: Callable[[tuple[Tensor, ...]], tuple[Tensor, ...]],
    cases: Sequence[tuple[Tensor, ...]],
    *,
    precision: str,
) -> None:
    """Compare values, input adjoints and declared second derivatives.

    Supply independent reference operations with the same canonical weights.
    Cases must include the geometry/topology changes relevant to the backend.
    Uses the existing backend tolerance rows, never a new relaxed tolerance.
    Compilation, native registration, equivariance and CUDA capture remain
    additional gates for a concrete implementation, not inferred from this check.
    """
    if not cases:
        raise ValueError("inference conformance needs at least one case")
    atol, rtol = TOLERANCES[precision]
    buffers = {name: tensor.clone() for name, tensor in prepared.named_buffers()}
    for index, arguments in enumerate(cases):
        twice = prepared.capabilities.double_backward
        expected = _evaluate(reference, arguments, twice)
        actual = _evaluate(prepared, arguments, twice)
        if len(actual) != len(expected):
            raise AssertionError(f"case {index}: output or derivative count differs")
        for value, target in zip(actual, expected, strict=True):
            torch.testing.assert_close(value, target, atol=atol, rtol=rtol)
        for name, tensor in prepared.named_buffers():
            torch.testing.assert_close(tensor, buffers[name], atol=0, rtol=0)
