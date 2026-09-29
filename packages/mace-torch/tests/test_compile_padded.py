"""The compiled calculator: one graph per padding budget, and no graph break.

The model is compiled whole, ``fullgraph=True``, at static shapes, so a graph
break raises instead of splitting it and every new batch shape is a new
compile. Padding holds the shape fixed while structures change, so a budget
is compiled once; a structure over it grows the budget and compiles exactly
once more. Counted with dynamo's own counter, which also keeps these runs free
of a C++ toolchain.
"""

from __future__ import annotations

import copy
import inspect

import pytest
import torch
from conftest import fp64_only
from mace_core.config.precision import PrecisionConfig
from mace_torch.calculators import MACECalculator, ase_calculator
from mace_torch.calculators.padding import PaddingOverflowError, PaddingPolicy
from mace_torch.physics.outputs import DerivativeEngine
from mace_torch_compile_fixtures import cluster, deployed
from torch._dynamo.testing import CompileCounter

pytestmark = fp64_only


@pytest.fixture(name="counter")
def fixture_counter(monkeypatch):
    torch._dynamo.reset()
    counter = CompileCounter()

    def compiled(engine, mode):
        engine = copy.copy(engine)
        engine.compile_model(backend=counter)
        return engine

    monkeypatch.setattr(ase_calculator, "_compiled", compiled)
    yield counter
    torch._dynamo.reset()


def evaluate(calculator, atoms):
    atoms = atoms.copy()
    atoms.calc = calculator
    return atoms.get_potential_energy(), atoms.get_forces()


def test_structures_under_one_budget_share_one_compile(counter):
    calculator = MACECalculator(models=deployed(), compile_mode="default")
    for molecules in (3, 1, 2, 3, 2):
        evaluate(calculator, cluster(molecules, seed=molecules))
    assert counter.frame_count == 1


def test_a_periodic_structure_compiles_once_with_its_stress(counter):
    calculator = MACECalculator(models=deployed(), compile_mode="default")
    for molecules in (3, 2, 1):
        atoms = cluster(molecules, periodic=True)
        atoms.calc = calculator
        assert atoms.get_stress().shape == (6,)
    assert counter.frame_count == 1


def test_a_structure_over_the_budget_recompiles_exactly_once(counter, caplog):
    calculator = MACECalculator(models=deployed(), compile_mode="default")
    evaluate(calculator, cluster(2))
    assert counter.frame_count == 1
    with caplog.at_level("WARNING", logger=ase_calculator.__name__):
        evaluate(calculator, cluster(4))
    assert counter.frame_count == 2
    assert any("over the padding budget" in r.getMessage() for r in caplog.records)
    for molecules in (4, 3, 1):
        evaluate(calculator, cluster(molecules, seed=molecules))
    assert counter.frame_count == 2


def test_a_fixed_budget_that_refuses_to_grow_raises():
    calculator = MACECalculator(
        models=deployed(),
        padding=PaddingPolicy(
            mode="fixed", nodes_budget=3, edges_budget=64, on_overflow="error"
        ),
    )
    with pytest.raises(PaddingOverflowError, match="Raise the budget"):
        evaluate(calculator, cluster(3))


def test_compiling_leaves_the_engine_it_was_given_eager():
    model = deployed()
    MACECalculator(models=model, compile_mode="default")
    assert model.engine._compiled_model is None


def test_the_compiled_region_holds_no_derivative():
    """``autograd.grad`` is the derivative engine's alone, and it stays out of
    what is compiled: the compiled callable is the model call and nothing
    around it."""
    engine = deployed().engine
    assert isinstance(engine, DerivativeEngine)
    engine.compile_model(backend=CompileCounter())
    compiled = engine._compiled_model
    assert compiled is not None
    assert inspect.unwrap(compiled) == engine.model_forward


def test_mixed_precision_compiles_whole(counter):
    calculator = MACECalculator(
        models=deployed(PrecisionConfig.mixed()), compile_mode="default"
    )
    for molecules in (2, 1, 2):
        _, forces = evaluate(calculator, cluster(molecules))
        assert forces.shape == (3 * molecules, 3)
    assert counter.frame_count == 1
