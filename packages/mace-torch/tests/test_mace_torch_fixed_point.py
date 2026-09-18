"""The fixed-point driver, against a problem whose answer is known.

The driver is tested on a synthetic engine whose energy is a quadratic in the
relaxed variable, so the fixed point has a closed form and the test compares
against it rather than against another solver's output. A real model is used
only where the point is that the driver and the real engine fit together.
"""

from __future__ import annotations

import pytest
import torch
from conftest import fp64_only
from mace_core.config import FixedPointSpec
from mace_core.outputs import MACEOutput
from mace_torch.physics import FixedPointDriver
from mace_torch_engine_fixtures import (
    NODE_INPUT,
    build_engine,
    build_graph,
    molecule,
    node_input_values,
)

VARIABLE = NODE_INPUT.name


class QuadraticEngine(torch.nn.Module):
    """``E = sum((v - target)^2)``, so the fixed point is ``target``.

    Stands in for a model. The driver's job is to find where the derivative
    against the variable vanishes, and here that place is known exactly, so a
    disagreement is the driver's and not a question of which answer is better.
    """

    def __init__(self, target: torch.Tensor) -> None:
        super().__init__()
        self.target = target
        self.differentiable_inputs = [NODE_INPUT]
        self.evaluations = 0

    def derivative_names(self):
        return {VARIABLE: "magforces"}

    def forward(self, graph, compute=(), training=False):
        self.evaluations += 1
        value = graph[VARIABLE]
        leaf = value if value.requires_grad else value.clone().requires_grad_(True)
        energy = ((leaf - self.target) ** 2).sum().reshape(1)
        output = MACEOutput(total_energy=energy)
        if "magforces" in set(compute):
            output.extras["magforces"] = -torch.autograd.grad(
                energy.sum(), leaf, create_graph=training
            )[0]
        return output


def quadratic_setup(count: int = 4, **spec_kwargs):
    torch.manual_seed(0)
    target = torch.randn(count, 3, dtype=torch.float64)
    engine = QuadraticEngine(target)
    spec = dict(variable=VARIABLE, max_iter=60, tolerance=1e-10)
    spec.update(spec_kwargs)
    driver = FixedPointDriver(engine, FixedPointSpec(**spec))
    graph = {VARIABLE: torch.zeros(count, 3, dtype=torch.float64)}
    return driver, graph, target


@fp64_only
def test_the_loop_finds_the_known_fixed_point():
    driver, graph, target = quadratic_setup()
    result = driver(graph, compute=("magforces",))

    converged = result.extras[f"converged_{VARIABLE}"]
    assert torch.allclose(converged, target, atol=1e-8, rtol=1e-8), (
        f"the fixed point is off by {float((converged - target).abs().max()):.3e}"
    )
    assert bool(result.extras["fixed_point_converged"])
    assert float(result.extras["magforces"].abs().max()) < 1e-7


@fp64_only
def test_the_telemetry_says_how_it_got_there():
    driver, graph, _ = quadratic_setup()
    result = driver(graph, compute=("magforces",))

    history = result.extras["fixed_point_history"]
    assert int(result.extras["fixed_point_steps"]) == len(history)
    assert float(history[-1]) < float(history[0]), "the energy did not go down"
    assert float(history[-1]) == pytest.approx(0.0, abs=1e-14)


@fp64_only
def test_the_telemetry_is_in_extras_and_not_a_core_field():
    """How the answer was reached is not itself an observable."""
    driver, graph, _ = quadratic_setup()
    result = driver(graph, compute=("magforces",))

    for key in ("fixed_point_history", "fixed_point_steps", "fixed_point_converged"):
        assert key in result.extras
    assert result.forces is None and result.stress is None


@fp64_only
def test_the_input_graph_is_left_alone():
    driver, graph, _ = quadratic_setup()
    before = graph[VARIABLE].clone()
    driver(graph, compute=("magforces",))
    assert torch.equal(graph[VARIABLE], before)


@fp64_only
def test_a_loop_that_does_not_settle_fails_and_says_why():
    """The frozen tree returns the energy at wherever the solver stopped.

    An unconverged energy and a converged one are indistinguishable at the
    call site, which is the shape of failure this replaces.
    """
    driver, graph, _ = quadratic_setup(max_iter=1, tolerance=1e-14)
    with pytest.raises(RuntimeError, match="did not settle"):
        driver(graph, compute=("magforces",))


@fp64_only
def test_a_capped_loop_can_be_accepted_on_purpose():
    driver, graph, _ = quadratic_setup(
        max_iter=1, tolerance=1e-14, require_convergence=False
    )
    result = driver(graph, compute=("magforces",))
    assert not bool(result.extras["fixed_point_converged"])


@fp64_only
def test_the_collinear_projection_keeps_the_directions():
    """Magnitudes move, directions do not. The projection, stated generically."""
    count = 4
    torch.manual_seed(1)
    target = torch.randn(count, 3, dtype=torch.float64)
    engine = QuadraticEngine(target)
    driver = FixedPointDriver(
        engine,
        FixedPointSpec(
            variable=VARIABLE,
            max_iter=60,
            tolerance=1e-10,
            collinear=True,
            require_convergence=False,
        ),
    )
    start = torch.randn(count, 3, dtype=torch.float64)
    result = driver({VARIABLE: start}, compute=("magforces",))

    converged = result.extras[f"converged_{VARIABLE}"]
    cosine = torch.nn.functional.cosine_similarity(converged, start, dim=-1)
    assert torch.allclose(cosine.abs(), torch.ones_like(cosine), atol=1e-10)


@fp64_only
def test_the_warm_start_is_off_until_asked_for():
    driver, graph, _ = quadratic_setup()
    driver(graph, compute=("magforces",))
    assert driver.cached is None


@fp64_only
def test_the_warm_start_is_explicit_state_with_a_reset():
    driver, graph, target = quadratic_setup(warm_start=True)
    driver(graph, compute=("magforces",))

    assert driver.cached is not None
    assert torch.allclose(driver.cached, target, atol=1e-8)

    before = driver.engine.evaluations
    driver(graph, compute=("magforces",))
    warm = driver.engine.evaluations - before

    driver.reset()
    assert driver.cached is None
    before = driver.engine.evaluations
    driver(graph, compute=("magforces",))
    cold = driver.engine.evaluations - before

    assert warm <= cold, (
        f"starting from the converged value took {warm} evaluations and "
        f"starting from scratch took {cold}, so the cache is not being used"
    )


@fp64_only
def test_a_warm_start_of_the_wrong_shape_is_refused():
    driver, graph, _ = quadratic_setup(warm_start=True)
    driver(graph, compute=("magforces",))
    with pytest.raises(ValueError, match="reset"):
        driver({VARIABLE: torch.zeros(7, 3, dtype=torch.float64)}, compute=())


@fp64_only
def test_a_second_derivative_through_the_fixed_point_is_refused_by_name():
    """Not returned wrong. The chain the converged variable would contribute
    is exactly the one the loop detached."""
    driver, graph, _ = quadratic_setup()
    with pytest.raises(NotImplementedError, match="FixedPointDriver"):
        driver(graph, compute=("magforces",), second_derivatives=True)


@fp64_only
def test_the_loop_does_not_keep_a_graph_through_its_iterations():
    driver, graph, _ = quadratic_setup()
    converged, history, steps, settled = driver.relax(graph)
    assert converged.grad_fn is None and not converged.requires_grad
    assert settled and steps == len(history)


@fp64_only
def test_relaxing_an_input_the_engine_does_not_declare_is_refused():
    engine = build_engine()
    with pytest.raises(ValueError, match="differentiable input"):
        FixedPointDriver(engine, FixedPointSpec(variable="magmom"))


@fp64_only
def test_the_driver_runs_on_the_real_engine():
    """Not a numerical claim: that the two fit together at all.

    The random-weight reference model has no lower bound in this input, so it
    has no fixed point to find, and the driver says so rather than returning
    the energy it ran away to.
    """
    engine = build_engine(inputs=[NODE_INPUT])
    positions, numbers = molecule()
    graph = build_graph(positions, numbers)
    graph[VARIABLE] = node_input_values(len(numbers))

    driver = FixedPointDriver(
        engine, FixedPointSpec(variable=VARIABLE, max_iter=8, tolerance=1e-8)
    )
    with pytest.raises(RuntimeError, match="no lower bound"):
        driver(graph, compute=("magforces",))


@fp64_only
def test_no_attribute_is_written_onto_a_module_during_a_forward():
    """The cache lives on the driver, and the driver says when it changes.

    The frozen tree writes the converged value onto the model inside its
    forward, so two calls with the same inputs differ by what ran before.
    """
    driver, graph, _ = quadratic_setup()
    before = dict(driver.engine.__dict__)
    driver(graph, compute=("magforces",))
    after = dict(driver.engine.__dict__)

    changed = {
        key
        for key in set(before) | set(after)
        if key != "evaluations"
        and (key not in before or key not in after or before[key] is not after[key])
    }
    assert not changed, f"the forward wrote {sorted(changed)} onto the engine"
