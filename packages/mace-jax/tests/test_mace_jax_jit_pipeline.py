"""Neighbour-list rebuilds, explicit geometry derivatives and capacity overflow."""

import jax
import numpy as np
import pytest
from ase import Atoms
from jax_conformance import TINY, ModelSurface
from mace_core.graph import AutoPadding, FixedPadding, NoPadding, PaddingOverflowError
from mace_jax.deploy.neutral_io import load_model
from mace_jax.inference import JITPipeline

from tests.golden import harness


def test_fixed_bucket_energy_rebuilds_and_depends_on_explicit_positions_and_cell():
    apply, params = load_model(TINY)
    surface = ModelSurface(apply, params)
    atoms = Atoms(
        "HH", positions=[[0, 0, 0], [1, 0, 0]], cell=[4, 8, 9], pbc=[True, False, False]
    )
    changed = atoms.copy()
    changed.positions[1] = [1.5, 2.9, 0]
    graphs = [surface.graph(frame) for frame in (atoms, changed)]
    assert graphs[0]["edge_index"].shape != graphs[1]["edge_index"].shape
    pipeline = JITPipeline(
        apply,
        params,
        FixedPadding(
            nodes_budget=5, edges_budget=32, graphs_budget=2, on_overflow="error"
        ),
    )
    for graph in graphs:
        padded, info = pipeline.prepare(graph)
        total = pipeline.energy(padded["positions"], padded["cell"], padded)
        result = pipeline(graph)
        np.testing.assert_allclose(
            total[: info.n_real_graphs], result.total_energy, atol=1e-12
        )
    assert pipeline.energy_traces == pipeline.output_traces == 1

    padded, _ = pipeline.prepare(graphs[0])
    pos, cell = padded["positions"], padded["cell"]

    def real_energy(p, c):
        return pipeline.energy(p, c, padded)[0]

    f = -jax.grad(real_energy)(pos, cell)
    expected = apply(params, graphs[0])
    assert expected.forces is not None
    np.testing.assert_allclose(f[: len(atoms)], expected.forces, atol=1e-10)
    # Hold positions fixed and move only a lattice vector. This is sensitive
    # to a stale Cartesian shift even when the positions gradient is correct.
    step = 1e-5
    shifted = atoms.copy()
    shifted.cell[0, 0] += step
    changed_energy = real_energy(pos, cell.at[0, 0, 0].add(step))
    shifted_total = apply(params, surface.graph(shifted), compute=()).total_energy
    assert shifted_total is not None
    np.testing.assert_allclose(
        changed_energy,
        shifted_total[0],
        atol=1e-12,
        rtol=0,
    )
    assert abs(float(changed_energy - real_energy(pos, cell))) > 1e-12


@pytest.mark.parametrize("overflow", ["grow", "error"])
def test_auto_bucket_overflow_is_one_retrace_or_a_loud_error(overflow):
    apply, params = load_model(TINY)
    surface = ModelSurface(apply, params)
    fixtures = harness.load_fixtures(
        names=["dimer_short", "water_cluster", "isolated_atom"]
    )
    graphs = {k: surface.graph(v) for k, v in fixtures.items()}
    pipeline = JITPipeline(
        apply, params, AutoPadding(edge_multiple=8, on_overflow=overflow)
    )
    with pytest.raises(ValueError, match="padding"):
        JITPipeline(apply, params, NoPadding())
    pipeline(graphs["dimer_short"])
    first = pipeline.capacity
    assert first is not None
    if overflow == "error":
        with pytest.raises(PaddingOverflowError, match="does not fit"):
            pipeline(graphs["water_cluster"])
        assert pipeline.capacity == first
    else:
        pipeline(graphs["water_cluster"])
        assert pipeline.capacity is not None
        assert pipeline.capacity.nodes > first.nodes
        assert pipeline.capacity.edges % 8 == 0
    pipeline(graphs["isolated_atom"])
    assert (
        pipeline.output_traces
        == pipeline.energy_traces
        == (2 if overflow == "grow" else 1)
    )
