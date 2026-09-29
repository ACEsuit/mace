"""The electrostatics solve and what a solver declares, without a framework.

The descriptor refuses a solve that cannot be meant, and the capability
record decides support: a solver that does bulk crystals only declines a
molecule or a mixed batch rather than doing something else with it.
"""

from __future__ import annotations

import subprocess
import sys

import pytest
from mace_core.electrostatics import (
    PERIODICITY_PROFILES,
    REALSPACE_METHODS,
    ElectrostaticsSolverDescriptor,
    FeatureProjection,
    ScfSpec,
    SolverCapabilities,
    UnsupportedSolveError,
)


def solve(profile="full_periodic", **settings) -> ElectrostaticsSolverDescriptor:
    return ElectrostaticsSolverDescriptor(
        profile,
        multipole_max_l=settings.pop("multipole_max_l", 1),
        kspace_cutoff=settings.pop("kspace_cutoff", 3.0),
        smearing_width=settings.pop("smearing_width", 1.0),
        **settings,
    )


def test_the_four_profiles_are_the_ones_named():
    assert PERIODICITY_PROFILES == ("full_periodic", "z_slab", "molecular", "partial")


def test_an_unknown_profile_is_refused():
    with pytest.raises(ValueError, match="periodicity profile"):
        solve("bulk")


def test_a_slab_needs_its_normal_and_nothing_else_takes_one():
    with pytest.raises(ValueError, match="slab_normal"):
        solve("z_slab")
    with pytest.raises(ValueError, match="slab_normal"):
        solve("full_periodic", slab_normal=2)
    assert solve("z_slab", slab_normal=2).slab_normal == 2


def test_a_solve_with_no_width_or_no_cutoff_is_refused():
    with pytest.raises(ValueError, match="positive"):
        solve(smearing_width=0.0)
    with pytest.raises(ValueError, match="positive"):
        solve(kspace_cutoff=-1.0)


def test_a_self_consistent_loop_has_to_be_one_that_can_end():
    with pytest.raises(ValueError, match="max_iters"):
        ScfSpec(max_iters=0)
    with pytest.raises(ValueError, match="mixing"):
        ScfSpec(mixing=1.5)


def test_a_descriptor_can_be_compared_and_hashed():
    """It is what a checkpoint records and a cache is keyed by."""
    assert solve() == solve()
    assert len({solve(), solve(), solve("molecular")}) == 2


def test_a_bulk_only_solver_declines_a_molecule_and_a_mixed_batch():
    bulk = SolverCapabilities(
        periodicity_profiles=frozenset({"full_periodic", "z_slab"})
    )
    assert bulk.supports(solve())
    assert not bulk.supports(solve("molecular"))
    assert not bulk.supports(solve("partial"))


def test_a_solver_declines_a_precision_it_does_not_compute_in():
    assert not SolverCapabilities().supports(solve(precision="float32"))


def test_a_solver_declines_a_multipole_order_above_its_limit():
    assert not SolverCapabilities(max_multipole_l=0).supports(solve())


def test_a_declined_solve_names_the_solver_and_what_it_declared():
    with pytest.raises(UnsupportedSolveError, match=r"'bulk'.*full_periodic"):
        SolverCapabilities(periodicity_profiles=frozenset({"full_periodic"})).require(
            solve("molecular"), "bulk"
        )


def test_a_solver_that_cannot_differentiate_twice_says_why_it_is_refused():
    with pytest.raises(UnsupportedSolveError, match="training on forces"):
        SolverCapabilities().require_double_backward("fast", "training on forces")


def test_the_real_space_method_is_one_of_the_named_ones():
    assert REALSPACE_METHODS == ("finite_difference",)
    with pytest.raises(ValueError, match="real-space method"):
        solve(realspace_method="analytical")


def test_a_projection_needs_a_width_and_counts_its_components():
    with pytest.raises(ValueError, match="widths"):
        FeatureProjection(max_l=1, widths=())
    assert FeatureProjection(max_l=1, widths=(1.0, 1.5)).dimension == 8


def test_a_solver_without_the_projection_declines_a_solve_that_reads_one():
    """A model reads its features and its energy from one solver, so a solver
    that has only the energy cannot serve a model that projects."""
    energy_only = SolverCapabilities()
    projecting = solve(features=FeatureProjection(max_l=1, widths=(1.0,)))
    assert energy_only.supports(solve())
    assert not energy_only.supports(projecting)
    both = SolverCapabilities(
        ops=frozenset({"long_range_energy", "long_range_features"})
    )
    assert both.supports(projecting)


def test_a_solver_declines_a_real_space_method_it_does_not_have():
    """Two methods are two sets of numbers for an open system."""
    assert not SolverCapabilities(realspace_methods=frozenset()).supports(solve())


@pytest.mark.parametrize("framework", ["torch", "jax"])
def test_the_electrostatics_layer_imports_no_framework(framework):
    probe = (
        f"import sys, mace_core.electrostatics\nprint({framework!r} in sys.modules)\n"
    )
    result = subprocess.run(
        [sys.executable, "-c", probe], capture_output=True, text=True, check=True
    )
    assert result.stdout.strip() == "False"
