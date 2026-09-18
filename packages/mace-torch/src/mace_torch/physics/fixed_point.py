"""Relaxing a declared input to a fixed point, around the engine.

Some models do not evaluate once. A magnetic model relaxes the moments until
the derivative against them vanishes, and only then reports the energy. That
loop wraps the **engine**, not the model: each iteration is an ordinary
evaluation plus the derivative against the relaxed variable, which the engine
already produces by name for any declared differentiable input.

Three clauses here are the contract rather than the implementation, and each
replaces something the frozen tree does in a way that is hard to see.

**It does not differentiate through its own iterations.** The converged
variable is detached before the final evaluation, so what is reported comes
from one evaluation at the fixed point. A gradient taken through the trajectory
would be a gradient of the solver, not of the physics, and would cost memory
proportional to the step count.

**The warm start is explicit.** The frozen tree caches the converged value by
writing an attribute onto the model inside its forward, so two calls with the
same inputs return different numbers depending on what ran before, with nothing
in the signature to say so. Here it is off unless asked for, it lives on the
driver, and it has a reset.

**A second derivative through the fixed point is refused.** The chain the
converged variable would contribute is exactly the one that was detached, so a
second derivative computed anyway is not wrong by a little: it silently omits
the term the fixed point contributes. The frozen tree refuses hessians for
these models, and this refuses them by name.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from typing import Any, Protocol, runtime_checkable

import torch
from mace_core.config import FixedPointSpec
from mace_core.observables.derivatives import derivative_name
from mace_core.outputs import MACEOutput
from torch import Tensor, nn

__all__ = ["FixedPointDriver", "RelaxableEngine"]


@runtime_checkable
class RelaxableEngine(Protocol):
    """What the driver needs from whatever it wraps.

    A protocol rather than the engine class, because the loop only ever uses
    two things: which inputs are differentiable, and an evaluation that returns
    an energy and the derivative against one of them. Saying so keeps the
    driver testable against a problem whose fixed point is known in closed
    form, which the real model is not.
    """

    differentiable_inputs: list[Any]

    def derivative_names(self) -> dict[str, str]: ...

    def __call__(self, graph: Mapping[str, Any], compute: Iterable[str], **kwargs): ...


class FixedPointDriver(nn.Module):
    """An outer loop that relaxes one declared input until its force vanishes.

    Args:
        engine: The derivative engine to evaluate. It has to declare the
            relaxed variable as a differentiable input, since the update is
            that input's own derivative.
        spec: What to relax and how.
    """

    def __init__(self, engine: RelaxableEngine, spec: FixedPointSpec) -> None:
        super().__init__()
        self.engine = engine
        self.spec = spec
        declared = {item.name for item in self.spec_inputs(engine)}
        if spec.variable not in declared:
            raise ValueError(
                f"the fixed point relaxes {spec.variable!r}, which the engine "
                f"does not carry as a differentiable input. The ones it does "
                f"are {sorted(declared)}. Declare it, or relax one of those."
            )
        self.derivative = derivative_name("energy", spec.variable)
        self._cache: Tensor | None = None

    @staticmethod
    def spec_inputs(engine: RelaxableEngine) -> list[Any]:
        """The engine's differentiable inputs, read through the protocol.

        Read once here rather than at each use, because reaching for it through
        an `nn.Module` goes via `__getattr__` and loses the type.
        """
        return list(engine.differentiable_inputs)

    def reset(self) -> None:
        """Forget the warm start.

        Named and public because a cached starting point makes a call depend on
        what ran before it, and a caller comparing two structures needs to be
        able to say so.
        """
        self._cache = None

    @property
    def cached(self) -> Tensor | None:
        """The converged value the next warm start would begin from."""
        return self._cache

    def relax(self, graph: Mapping[str, Any]) -> tuple[Tensor, list[float], int, bool]:
        """Drive the variable to its fixed point.

        Returns:
            The converged variable, the energy at each step, the number of
            steps taken, and whether the derivative actually reached the
            tolerance.

        Raises:
            RuntimeError: If the energy stops being finite. An energy with no
                lower bound in the relaxed variable sends the solver off to
                infinity, and the number that comes back looks like an energy.
                It is worth saying out loud: this is a statement about the
                model, not about the solver's settings.
        """
        start = graph[self.spec.variable]
        if self.spec.warm_start and self._cache is not None:
            if self._cache.shape != start.shape:
                raise ValueError(
                    f"the warm start holds a {self.spec.variable!r} of shape "
                    f"{tuple(self._cache.shape)} and this graph wants "
                    f"{tuple(start.shape)}. Call `reset()` between structures "
                    f"of different sizes, or leave `warm_start` off."
                )
            start = self._cache
        # The direction to project onto for a collinear relaxation is the one
        # the caller supplied, not the one the warm start happens to hold.
        reference = graph[self.spec.variable].detach()
        variable = start.detach().clone().requires_grad_(True)

        optimizer = torch.optim.LBFGS(
            [variable],
            lr=self.spec.step_size,
            max_iter=self.spec.max_iter,
            tolerance_grad=self.spec.tolerance,
            line_search_fn="strong_wolfe",
        )
        history: list[float] = []

        def closure() -> Tensor:
            optimizer.zero_grad()
            moving = dict(graph)
            moving[self.spec.variable] = variable
            result = self.engine(moving, compute=(self.derivative,))
            energy = result.total_energy.sum()
            # The derivative is minus the gradient, by the naming rule that
            # gave it its name, so the gradient is minus it back.
            gradient = -result.extras[self.derivative].detach()
            if self.spec.collinear:
                gradient = _project_onto(gradient, reference)
            variable.grad = gradient
            history.append(float(energy.detach()))
            return energy.detach()

        optimizer.step(closure)
        final_gradient = variable.grad
        settled = bool(
            final_gradient is not None
            and float(final_gradient.abs().max()) <= self.spec.tolerance
        )
        if self.spec.require_convergence and not settled:
            reached = (
                float(final_gradient.abs().max())
                if final_gradient is not None
                else float("nan")
            )
            raise RuntimeError(
                f"the fixed point in {self.spec.variable!r} did not settle in "
                f"{len(history)} step(s): the largest component of "
                f"d(energy)/d({self.spec.variable}) is {reached:.3e} against a "
                f"tolerance of {self.spec.tolerance:.3e}, and the energy went "
                f"from {history[0]:.6f} to {history[-1]:.6f}. A run away to a "
                f"very large magnitude means the energy has no lower bound in "
                f"that variable, which is about the model and not about "
                f"`max_iter`. Set `require_convergence=False` to accept a "
                f"capped loop and read `fixed_point_converged` instead."
            )
        converged = variable.detach()
        if self.spec.collinear:
            converged = _project_onto(converged, reference)
        if self.spec.warm_start:
            self._cache = converged.clone()
        return converged, history, len(history), settled

    def forward(
        self,
        graph: Mapping[str, Any],
        compute: Iterable[str] = ("forces",),
        training: bool = False,
        second_derivatives: bool = False,
    ) -> MACEOutput[Tensor]:
        """The observables at the fixed point.

        Args:
            graph: The flat dict, read only.
            compute: Which derivatives the final evaluation should produce.
            training: Passed to the final evaluation.
            second_derivatives: Refused. The argument exists so that asking
                produces an error naming this driver rather than a number that
                silently omits the fixed point's own contribution.
        """
        if second_derivatives:
            raise NotImplementedError(
                f"a second derivative through {type(self).__name__} is not "
                f"available: the loop detaches its converged "
                f"{self.spec.variable!r} on purpose, so a second derivative "
                f"computed through it would omit the term the fixed point "
                f"contributes and look plausible anyway. Evaluate the engine "
                f"directly at a fixed {self.spec.variable!r} if you need one."
            )
        converged, history, steps, settled = self.relax(graph)
        final = dict(graph)
        final[self.spec.variable] = converged

        output = self.engine(final, compute=compute, training=training)
        # Telemetry, in extras rather than as fields: it describes how the
        # answer was reached and is not itself an observable.
        output.extras["fixed_point_history"] = torch.tensor(
            history, dtype=converged.dtype
        )
        output.extras["fixed_point_steps"] = torch.tensor(steps)
        output.extras[f"converged_{self.spec.variable}"] = converged
        # Whether it settled, rather than only how many steps it took. A loop
        # that hit `max_iter` and one that reached the tolerance both report a
        # step count, and only one of them found a fixed point.
        output.extras["fixed_point_converged"] = torch.tensor(settled)
        return output


def _project_onto(values: Tensor, reference: Tensor) -> Tensor:
    """Keep only the component of each row along the reference's direction.

    For magnetic moments this is what makes a relaxation collinear: the
    magnitudes move and the directions do not. A reference row of length zero
    has no direction to keep, so its whole row is dropped.
    """
    norm = reference.norm(dim=-1, keepdim=True)
    direction = reference / torch.where(norm > 0, norm, torch.ones_like(norm))
    return (values * direction).sum(dim=-1, keepdim=True) * direction
