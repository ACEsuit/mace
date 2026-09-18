"""Derivatives against declared inputs other than the positions.

A magnetic moment and an external field are, to this machinery, two declared
inputs with different irreps. Neither name appears in the engine or the
backbone, and the test that asserts so is the point of the design: a model that
wants a new conditioned input declares it, and nothing here is edited.
"""

from __future__ import annotations

import ast
import inspect

import numpy as np
import pytest
import torch
from conftest import fp64_only
from mace_core.observables import InputSpec
from mace_core.observables.derivatives import derivation_mode, derivative_name
from mace_torch_engine_fixtures import (
    NODE_INPUT,
    build_engine,
    build_graph,
    crystal,
    molecule,
    node_input_values,
)

PERIODIC = (True, True, True)


def conditioned_graph(seed: int = 7):
    positions, numbers = molecule()
    graph = build_graph(positions, numbers)
    graph[NODE_INPUT.name] = node_input_values(len(numbers), seed)
    return graph, positions, numbers


# ---------------------------------------------------------------------------
# The naming and the mode, derived rather than declared
# ---------------------------------------------------------------------------


@fp64_only
def test_the_derivation_mode_follows_from_the_target():
    """Declaring the mode beside the target would let the two disagree.

    A declaration saying `wrt: pos` and `derivation: grad_input` is spellable
    and meaningless, and nothing would catch it. What the target is already
    decides how the derivative can be taken.
    """
    assert derivation_mode("pos") == "autograd"
    assert derivation_mode("cell") == "grad_strain"
    assert derivation_mode("magmom") == "grad_input"
    assert derivation_mode("external_field") == "grad_input"


@fp64_only
def test_a_declared_input_gets_its_name_from_the_same_rule_as_forces():
    assert derivative_name("energy", "pos") == "forces"
    assert derivative_name("energy", "magmom") == "magforces"
    assert derivative_name("energy", "charge") == "d_energy_d_charge"


@fp64_only
def test_the_engine_reports_the_name_the_rule_gives():
    engine = build_engine(inputs=[NODE_INPUT])
    assert engine.derivative_names() == {"magmom": "magforces"}


# ---------------------------------------------------------------------------
# The numbers
# ---------------------------------------------------------------------------


@fp64_only
def test_the_input_derivative_matches_finite_differences():
    """The analogue of the force check, for a non-positional input.

    The frozen tree has no test of this at all: its magnetic suite covers
    training, evaluation, equivariance and the refusals, and never compares
    dE/dm against a reference.
    """
    engine = build_engine(inputs=[NODE_INPUT])
    graph, _positions, _numbers = conditioned_graph()
    values = graph[NODE_INPUT.name].numpy()

    result = engine(graph, compute=("magforces",))
    derivative = result.extras["magforces"].detach().numpy()
    assert np.abs(derivative).max() > 1e-3, "the derivative is essentially zero"

    step = 1e-5
    reference = np.zeros_like(values)
    for row in range(len(values)):
        for axis in range(3):
            samples = []
            for offset in (-2, -1, 1, 2):
                moved = values.copy()
                moved[row, axis] += offset * step
                probe = dict(graph)
                probe[NODE_INPUT.name] = torch.tensor(moved)
                samples.append(float(engine(probe, compute=()).total_energy.detach()))
            reference[row, axis] = -(
                samples[0] - 8 * samples[1] + 8 * samples[2] - samples[3]
            ) / (12 * step)

    deviation = np.abs(derivative - reference).max()
    assert deviation < 1e-6, (
        f"the input derivative differs from the five-point central difference "
        f"by {deviation:.3e}"
    )


@fp64_only
def test_every_derivative_comes_from_one_grad_call(monkeypatch):
    """Forces, virials and the input derivative in a single call.

    Separate calls would need `retain_graph` and would not reproduce the same
    numbers under `create_graph=True`, which is the shape the frozen tree
    already uses for its fused magnetic path.
    """
    engine = build_engine(inputs=[NODE_INPUT])
    positions, numbers, cell = crystal()
    graph = build_graph(positions, numbers, cell, PERIODIC)
    graph[NODE_INPUT.name] = node_input_values(len(numbers))

    calls = []
    original = torch.autograd.grad

    def counting(*args, **kwargs):
        calls.append(len(kwargs.get("inputs", args[1] if len(args) > 1 else [])))
        return original(*args, **kwargs)

    monkeypatch.setattr(torch.autograd, "grad", counting)
    engine(graph, compute=("forces", "virials", "stress", "magforces"))

    assert calls == [3], f"expected one grad call over three targets, got {calls}"


@fp64_only
def test_the_derivative_carries_the_sign_the_rule_gives():
    """Minus the gradient, like forces. Asserted against the raw gradient."""
    engine = build_engine(inputs=[NODE_INPUT])
    graph, _, _ = conditioned_graph()

    result = engine(graph, compute=("magforces",))
    leaf = graph[NODE_INPUT.name].clone().requires_grad_(True)
    probe = dict(graph)
    probe[NODE_INPUT.name] = leaf
    energy = engine(probe, compute=()).total_energy.sum()
    raw = torch.autograd.grad(energy, leaf)[0]

    assert torch.allclose(result.extras["magforces"], -raw, atol=1e-12, rtol=1e-12)


# ---------------------------------------------------------------------------
# The errors
# ---------------------------------------------------------------------------


@fp64_only
def test_a_derivative_of_an_undeclared_input_is_refused():
    engine = build_engine()
    graph, _, _ = conditioned_graph()
    with pytest.raises(ValueError, match="magforces"):
        engine(graph, compute=("magforces",))


@fp64_only
def test_a_declared_but_non_differentiable_input_says_which_flag_is_missing():
    fixed = InputSpec(
        name="magmom", irreps="1o", per_atom=True, units="muB", differentiable=False
    )
    engine = build_engine(inputs=[fixed])
    graph, _, _ = conditioned_graph()
    with pytest.raises(ValueError, match="differentiable"):
        engine(graph, compute=("magforces",))


@fp64_only
def test_an_input_missing_from_the_graph_is_refused_at_call_time():
    engine = build_engine(inputs=[NODE_INPUT])
    positions, numbers = molecule()
    with pytest.raises(KeyError, match="magmom"):
        engine(build_graph(positions, numbers), compute=("magforces",))


@fp64_only
def test_an_all_zero_input_is_present_and_not_absent():
    """The frozen tree treats an all-zero external field as no field at all.

    A genuinely zero field and no field have to be distinguishable, so absence
    is the key being absent and nothing else.
    """
    engine = build_engine(inputs=[NODE_INPUT])
    graph, _, numbers = conditioned_graph()
    graph[NODE_INPUT.name] = torch.zeros(len(numbers), 3, dtype=torch.float64)

    result = engine(graph, compute=("magforces",))
    assert result.extras["magforces"].shape == (len(numbers), 3)


# ---------------------------------------------------------------------------
# No name knows itself
# ---------------------------------------------------------------------------


@fp64_only
def test_no_input_is_named_in_the_engine_or_the_backbone():
    """The whole point, asserted where a shortcut would be taken.

    A single `if name == "magmom"` anywhere below would make the next
    conditioned input a code change instead of a declaration.
    """
    from mace_torch.models import energy, heads, outputs
    from mace_torch.nn import backbone, interaction, node_inputs, product_basis
    from mace_torch.physics import fixed_point
    from mace_torch.physics import outputs as engine

    forbidden = {"magmom", "magforces", "external_field", "BEC"}
    modules = (
        backbone,
        interaction,
        product_basis,
        node_inputs,
        energy,
        heads,
        outputs,
        engine,
        fixed_point,
    )
    for module in modules:
        tree = ast.parse(inspect.getsource(module))
        for node in ast.walk(tree):
            if isinstance(node, ast.Constant) and node.value in forbidden:
                pytest.fail(
                    f"{module.__name__} names {node.value!r} as a literal. A "
                    f"conditioned input is declared, never recognised."
                )
