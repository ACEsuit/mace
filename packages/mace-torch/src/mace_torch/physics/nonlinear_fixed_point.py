"""Converged nonlinear SCF with implicit first and second derivatives.

Newton's linearization supplies implicit derivatives of the residual equation.
The convergence trajectory is detached. Two differentiable Newton corrections
at its root retain the dependence of the Jacobian on the root, which a single
correction misses at second order. This is independent of the number of solve
iterations; gradgradcheck and unrolled-reference tests pin the distinction.
"""

from typing import NamedTuple

import torch
from torch import Tensor, nn


class ResponseInputs(NamedTuple):
    """Inputs to the nonlinear SCF residual, with canonical activation layout.

    Density/local: [atoms,multipoles] in e Å**l; projection: [N*F,N*D]
    maps density to normalized potential. Applied field: [N,F] in eV/e;
    Fermi response: [N*F,graphs] per eV. Membership: [N,graphs].
    Charge [graphs] is in e; Fermi [graphs] in eV. Remaining tensors are
    fixed descriptors [N,C], one-hot elements [N,Z], edge attributes [E,9],
    radial embeddings [E,R] and sender/receiver indices [2,E].
    """

    local: Tensor
    projection: Tensor
    applied: Tensor
    fermi_response: Tensor
    membership: Tensor
    charge: Tensor
    fermi: Tensor
    nodes: Tensor
    attributes: Tensor
    edge_attributes: Tensor
    edge_radial: Tensor
    edge_index: Tensor


def residual(
    state: Tensor,
    response: nn.Module,
    parameters: dict[str, Tensor],
    inputs: ResponseInputs,
    constant_charge: bool,
) -> Tensor:
    count = inputs.local.numel()
    density = state[:count].reshape_as(inputs.local)
    fermi = state[count:] if constant_charge else inputs.fermi
    field = (
        inputs.projection @ density.reshape(-1)
        + inputs.applied.reshape(-1)
        + inputs.fermi_response @ fermi
    ).reshape(inputs.applied.shape)
    delta = torch.func.functional_call(
        response,
        parameters,
        (
            inputs.nodes,
            inputs.attributes,
            field,
            inputs.edge_attributes,
            inputs.edge_radial,
            inputs.edge_index,
        ),
    )
    result = (density - inputs.local - delta).reshape(-1)
    if constant_charge:
        result = torch.cat(
            (result, inputs.membership.T @ density[:, 0] - inputs.charge)
        )
    return result


def newton_step(
    state: Tensor,
    response: nn.Module,
    parameters: dict[str, Tensor],
    inputs: ResponseInputs,
    constant_charge: bool,
) -> Tensor:
    args = (state, response, parameters, inputs, constant_charge)
    jacobian = torch.func.jacrev(residual, argnums=0)(*args)
    return state - torch.linalg.solve(jacobian, residual(*args))


class NonlinearFixedPoint(nn.Module):
    """Dense reference solve, density in e Å**l and Fermi level in eV.

    Refuses a nonconverged root instead of returning an unconverged iterate.
    The deterministic initial state is the local density and supplied Fermi
    level (zero for fixed charge); no persistent warm-start cache is hidden.
    """

    def __init__(
        self, constant_charge: bool, max_iters: int = 50, tolerance: float = 1e-10
    ):
        super().__init__()
        self.constant_charge = constant_charge
        self.max_iters = max_iters
        self.tolerance = tolerance

    def forward(
        self,
        response: nn.Module,
        parameters: dict[str, Tensor],
        inputs: ResponseInputs,
    ) -> tuple[Tensor, Tensor]:
        state = inputs.local.reshape(-1)
        if self.constant_charge:
            state = torch.cat((state, inputs.fermi))
        with torch.no_grad():
            for _ in range(self.max_iters):
                value = residual(
                    state, response, parameters, inputs, self.constant_charge
                )
                norm = value.abs().amax()
                if not torch.compiler.is_compiling() and bool(norm <= self.tolerance):
                    break
                proposal = newton_step(
                    state, response, parameters, inputs, self.constant_charge
                )
                # Bounded backtracking is tensor-only, so full-graph capture does
                # not branch on a device scalar or synchronize each iteration.
                step = proposal - state
                candidate = proposal
                for _ in range(8):
                    trial = (
                        residual(
                            candidate,
                            response,
                            parameters,
                            inputs,
                            self.constant_charge,
                        )
                        .abs()
                        .amax()
                    )
                    step = torch.where(trial > norm, step * 0.5, step)
                    candidate = state + step
                state = torch.where(norm <= self.tolerance, state, candidate)
            error = (
                residual(state, response, parameters, inputs, self.constant_charge)
                .abs()
                .amax()
            )
            torch._assert_async(
                torch.isfinite(error) & (error <= self.tolerance),
                "FixedPointSCF did not converge; increase max_iters "
                "or review the response and initial state",
            )
        state = state.detach()
        for _ in range(2):
            state = newton_step(
                state, response, parameters, inputs, self.constant_charge
            )
        density = state[: inputs.local.numel()].reshape_as(inputs.local)
        return density, state[
            inputs.local.numel() :
        ] if self.constant_charge else inputs.fermi
