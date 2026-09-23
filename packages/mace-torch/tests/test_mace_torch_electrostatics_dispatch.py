"""Finding a solver, and the rules it is resolved by.

A solver is found through its entry point group, resolved once when the model
is built, and never substituted: one that declines a solve is an error, one
that cannot be differentiated twice is refused for training on derivatives,
and one that failed to import is named with the reason it failed.
"""

from __future__ import annotations

import subprocess
import sys
from types import NotImplementedType

import pytest
import torch
from mace_core.electrostatics import (
    ENTRY_POINT_GROUPS,
    ElectrostaticsSolverDescriptor,
    SolverCapabilities,
    SolverNotAvailableError,
    UnsupportedSolveError,
    available_solvers,
)
from mace_core.electrostatics import registry as registry_module
from mace_torch.electrostatics import (
    LongRangeEnergy,
    ReferenceSolver,
    build_long_range,
    build_scf_solve,
)
from torch import nn

SOLVE = ElectrostaticsSolverDescriptor("full_periodic", 1, 3.0, 1.0)


class _EntryPoint:
    def __init__(self, name: str, target) -> None:
        self.name = name
        self._target = target

    def load(self):
        if isinstance(self._target, BaseException):
            raise self._target
        return self._target


class _Span(nn.Module):
    pass


class _Accelerated:
    """A solver shaped like an accelerated one: not the reference's numbers,
    inference only, bulk crystals only, with a fused self-consistent solve."""

    name = "accelerated"
    capabilities = SolverCapabilities(
        periodicity_profiles=frozenset({"full_periodic"}),
        supports_double_backward=False,
        bit_parity=False,
    )

    def long_range_energy(self, descriptor):
        return nn.Identity()

    def make_scf_solve(self, descriptor) -> nn.Module | NotImplementedType:
        return _Span()


@pytest.fixture(name="registered")
def fixture_registered(monkeypatch):
    """The reference, an accelerated solver, and one that does not import."""
    points = [
        _EntryPoint("reference", ReferenceSolver),
        _EntryPoint("accelerated", _Accelerated),
        _EntryPoint("broken", ImportError("libnvsomething.so: not found")),
    ]

    def entry_points(group: str):
        assert group == ENTRY_POINT_GROUPS["torch"]
        return points

    monkeypatch.setattr(registry_module, "entry_points", entry_points)


# ---------------------------------------------------------------------------
# Discovery and resolution
# ---------------------------------------------------------------------------


def test_the_reference_is_found_through_its_entry_point():
    """Registered in this package's metadata, the way a third party's is."""
    found = {solver.name: solver for solver in available_solvers()}
    assert found["reference"].loaded


def test_a_solver_that_does_not_import_is_listed_with_why(registered):
    broken = {solver.name: solver for solver in available_solvers()}["broken"]
    assert not broken.loaded and "libnvsomething" in broken.reason


def test_asking_for_a_solver_that_does_not_import_names_the_failure(registered):
    with pytest.raises(SolverNotAvailableError, match="libnvsomething"):
        build_long_range(SOLVE, solver="broken")


def test_asking_for_a_solver_nobody_registered_lists_the_ones_there_are(registered):
    with pytest.raises(SolverNotAvailableError, match="accelerated"):
        build_long_range(SOLVE, solver="missing")


def test_the_op_is_resolved_once_and_not_again_in_forward(registered, monkeypatch):
    op = build_long_range(SOLVE)
    assert isinstance(op, LongRangeEnergy) and op.solver == "reference"

    def resolving(*_, **__):
        raise AssertionError("a solver was resolved in forward")

    monkeypatch.setattr(registry_module, "entry_points", resolving)
    graph = {
        "positions": torch.rand(3, 3, dtype=torch.float64),
        "batch": torch.zeros(3, dtype=torch.long),
        "cell": 6.0 * torch.eye(3, dtype=torch.float64).unsqueeze(0),
        "pbc": torch.tensor([[True, True, True]]),
    }
    op(graph, torch.rand(3, 4, dtype=torch.float64))


# ---------------------------------------------------------------------------
# What is refused
# ---------------------------------------------------------------------------


def test_a_solver_that_cannot_differentiate_twice_is_refused_for_training(
    registered,
):
    with pytest.raises(UnsupportedSolveError, match="differentiated twice"):
        build_long_range(SOLVE, solver="accelerated", trains_derivatives=True)


def test_it_is_usable_for_inference(registered):
    assert isinstance(build_long_range(SOLVE, solver="accelerated"), nn.Identity)


def test_a_solver_that_declines_a_system_is_an_error_not_a_fallback(registered):
    """Not the reference's numbers, so the reference is never handed back in
    its place: that would change the model."""
    molecule = ElectrostaticsSolverDescriptor("molecular", 1, 3.0, 1.0)
    with pytest.raises(UnsupportedSolveError, match="accelerated"):
        build_long_range(molecule, solver="accelerated")


# ---------------------------------------------------------------------------
# The fused self-consistent solve
# ---------------------------------------------------------------------------


def test_a_solver_without_a_fused_solve_leaves_the_loop_to_the_model(registered):
    assert build_scf_solve(SOLVE) is None


def test_a_solver_with_one_hands_it_over(registered):
    assert isinstance(build_scf_solve(SOLVE, solver="accelerated"), _Span)


# ---------------------------------------------------------------------------
# What it pulls in
# ---------------------------------------------------------------------------


def test_the_solver_imports_neither_e3nn_nor_the_frozen_tree():
    """The in-tree solver replaced both of the original's imports of them."""
    probe = (
        "import sys\n"
        "from mace_core.electrostatics import ElectrostaticsSolverDescriptor as D\n"
        "from mace_torch.electrostatics import build_long_range\n"
        "build_long_range(D('partial', 1, 3.0, 1.0))\n"
        "roots = {name.split('.')[0] for name in sys.modules}\n"
        "leaked = sorted(roots & {'e3nn', 'mace'})\n"
        "print(leaked)\n"
    )
    result = subprocess.run(
        [sys.executable, "-c", probe], capture_output=True, text=True, check=True
    )
    assert result.stdout.strip() == "[]"


# ---------------------------------------------------------------------------
# The configuration section
# ---------------------------------------------------------------------------


def _run_with(tmp_path, **electrostatics):
    from mace_core.observables import load_default_catalogue
    from mace_torch.train import run_data_stage, run_model_stage
    from test_mace_torch_full_batch import configuration

    config = configuration(tmp_path)
    config = config.model_copy(
        update={
            "electrostatics": config.electrostatics.model_copy(update=electrostatics)
        }
    )
    catalogue = load_default_catalogue()
    return run_model_stage(config, run_data_stage(config, catalogue), catalogue)


def test_the_section_is_off_unless_written(tmp_path):
    _run_with(tmp_path)


def test_an_enabled_section_is_refused_rather_than_trained_without(tmp_path):
    from mace_torch.train import ModelStageError

    with pytest.raises(ModelStageError, match=r"electrostatics\.enabled"):
        _run_with(tmp_path, enabled=True)


def test_an_enabled_section_naming_no_solver_says_so(tmp_path):
    with pytest.raises(SolverNotAvailableError, match="'fast'"):
        _run_with(tmp_path, enabled=True, solver="fast")


def test_the_section_is_not_among_the_advertised_flags(capsys):
    """Reachable from a configuration file, and not advertised by --help."""
    from mace_torch.cli.run_train import parse

    with pytest.raises(SystemExit):
        parse(["--help"])
    advertised = capsys.readouterr().out
    assert "--config" in advertised
    assert "electrostatics" not in advertised
