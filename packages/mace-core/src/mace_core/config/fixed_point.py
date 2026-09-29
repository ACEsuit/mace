"""Describing a self-consistent loop, without running one.

Some models do not evaluate once. A magnetic model relaxes the moments until
they stop moving, and only then reports the energy. The loop is an outer solver
over the model, and this says what it does; running it belongs to the framework
package.

Two clauses here are contracts rather than settings, and both reproduce what
the frozen tree does for reasons it discovered the hard way.

The loop **does not differentiate through its own iterations**. What is
reported comes from one final evaluation at the converged variable. A gradient
taken through the iterations would be a gradient of the solver's trajectory,
which is not a physical quantity, and it costs memory proportional to the
number of steps.

The warm start is **explicit state with a reset**. The frozen tree caches the
equilibrated moments by writing an attribute onto the model during its forward,
which makes two calls with the same inputs return different numbers depending
on what ran before, with nothing in the signature to say so.
"""

from __future__ import annotations

from typing import Literal

from pydantic import ConfigDict, Field, model_validator

from mace_core.config.base import ConfigSection

__all__ = ["FixedPointSpec", "SolverKind"]

SolverKind = Literal["lbfgs"]


class FixedPointSpec(ConfigSection):
    """An outer loop that relaxes one declared input to a fixed point.

    Attributes:
        variable: The declared input that is relaxed. It has to be a
            differentiable one, since the update is its own derivative.
        solver: Which solver drives it.
        max_iter: The iteration ceiling.
        tolerance: The gradient norm below which it has converged.
        step_size: The solver's step scale.
        collinear: Project the variable onto its initial direction after each
            step. For magnetic moments this is a collinear relaxation: the
            magnitudes move and the directions do not.
        warm_start: Begin from the previous call's converged value. Off by
            default, because a warm start makes a call depend on what ran
            before it and that should be asked for rather than inherited.
        variational: Whether the fixed point is a stationary point of the
            energy that gets reported. It is, when the loop reaches it by
            minimising that energy, and then a force taken at the fixed point
            with the converged variable detached is already the total
            derivative: the term the variable would contribute is multiplied by
            a gradient that vanished. A fixed point of something else, a linear
            system solved for charges say, has no such cancellation, and a
            derivative through it needs an implicit backward that this solver
            does not have. Declaring it false is therefore a refusal of
            derivative training rather than a setting that changes a number.
        require_convergence: Fail when the loop stops without reaching the
            tolerance. On by default, which is a **declared deviation** from
            the frozen tree: it returns the energy at wherever the solver
            stopped, with nothing to say the loop did not settle, and an
            unconverged energy is indistinguishable from a converged one at
            the call site. Turn it off for a training loop that accepts a
            capped number of iterations, and read `fixed_point_converged`.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    variable: str
    solver: SolverKind = "lbfgs"
    max_iter: int = Field(default=20, ge=1)
    tolerance: float = Field(default=1e-5, gt=0.0)
    step_size: float = Field(default=1.0, gt=0.0)
    collinear: bool = False
    warm_start: bool = False
    variational: bool = True
    require_convergence: bool = True

    @model_validator(mode="after")
    def _validate(self) -> FixedPointSpec:
        if not self.variable:
            raise ValueError(
                "a fixed point needs the name of the input it relaxes, and "
                "the name is empty."
            )
        return self
