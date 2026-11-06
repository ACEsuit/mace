"""Natural batches, real-model derivatives, empty graphs and the inference CLI."""

import subprocess
import sysconfig
from pathlib import Path

import ase.io
import jax
import jax.numpy as jnp
import numpy as np
import pytest
from ase.constraints import FixAtoms
from jax_conformance import TINY, ModelSurface, run_jax_model_conformance
from mace_jax.deploy.neutral_io import load_model
from mace_jax.graph import graph_from_atoms
from mace_jax.inference import JITPipeline

from tests.golden import harness


@pytest.fixture(scope="module")
def loaded():
    model = load_model(TINY)
    yield model
    jax.clear_caches()


def graph_for(apply, atoms):
    return graph_from_atoms(
        atoms,
        elements=apply.elements,
        cutoff=apply.config.model.r_max,
        dtype=apply.dtype,
        head=next(iter(apply.config.data.heads)),
    )


def test_mixed_batch_keeps_isolated_atoms_and_periodic_stress(loaded):
    apply, params = loaded
    fixtures = harness.load_fixtures(
        names=["water_cluster", "triclinic_bulk", "isolated_atom"]
    )
    atoms = list(fixtures.values())
    separate = [apply(params, graph_for(apply, item)) for item in atoms]
    graph = graph_for(apply, atoms)
    result = apply(params, graph)
    compiled = JITPipeline(apply, params)(graph)
    row = harness.FP64_CPU_REFERENCE
    for name in ("total_energy", "node_energies", "forces", "stress", "virials"):
        np.testing.assert_allclose(
            getattr(compiled, name), getattr(result, name), atol=row.atol, rtol=row.rtol
        )
        np.testing.assert_allclose(
            getattr(result, name),
            np.concatenate([getattr(item, name) for item in separate]),
            atol=row.atol,
            rtol=row.rtol,
        )
    for index, structure in enumerate(atoms):
        if not structure.pbc.any():
            np.testing.assert_array_equal(result.stress[index], 0)
    assert np.max(np.abs(result.stress)) > 0
    isolated = next(i for i, item in enumerate(atoms) if len(item) == 1)
    np.testing.assert_array_equal(separate[isolated].forces, 0)
    assert np.isfinite(np.asarray(result.total_energy)).all()


def test_force_and_symmetric_strain_are_energy_derivatives(loaded):
    apply, params = loaded
    atoms = harness.load_fixtures(names=["triclinic_bulk"])["triclinic_bulk"]
    graph = graph_for(apply, atoms)
    output = apply(params, graph)
    step = 1e-5
    plus, minus = dict(graph), dict(graph)
    plus["positions"] = graph["positions"].at[1, 0].add(step)
    minus["positions"] = graph["positions"].at[1, 0].add(-step)
    finite_force = -(
        apply(params, plus, compute=()).total_energy
        - apply(params, minus, compute=()).total_energy
    ) / (2 * step)
    np.testing.assert_allclose(
        output.forces[1, 0], finite_force[0], atol=1e-8, rtol=1e-6
    )
    # An off-diagonal symmetric strain moves both positions and periodic shifts.
    direction = jnp.asarray([[0.0, 0.5, 0.0], [0.5, 0.0, 0.0], [0.0, 0.0, 0.0]])
    for key in ("positions", "shifts", "cell"):
        plus[key] = graph[key] + step * (graph[key] @ direction)
        minus[key] = graph[key] - step * (graph[key] @ direction)
    derivative = (
        apply(params, plus, compute=()).total_energy
        - apply(params, minus, compute=()).total_energy
    ) / (2 * step)
    np.testing.assert_allclose(
        -output.virials[0, 0, 1], derivative[0], atol=1e-8, rtol=1e-6
    )
    np.testing.assert_allclose(
        output.stress * abs(np.linalg.det(atoms.cell)), -output.virials, atol=1e-12
    )
    assert apply(params, graph, compute=()).forces is None
    with pytest.raises(ValueError, match="hessian"):
        apply(params, graph, compute=("hessian",))


def test_cli_writes_predictions_for_every_structure(loaded, tmp_path):
    apply, params = loaded
    atoms = harness.load_fixtures(names=["triclinic_bulk", "isolated_atom"])
    # extxyz preserves FixAtoms. The command must write raw model forces.
    atoms["triclinic_bulk"].set_constraint(FixAtoms(indices=[0]))
    source, target = tmp_path / "input.xyz", tmp_path / "output.xyz"
    ase.io.write(source, list(atoms.values()), format="extxyz")
    script = Path(sysconfig.get_path("scripts")) / "mace-jax"
    subprocess.run(
        [
            str(script),
            "--model",
            str(TINY),
            "--configs",
            str(source),
            "--output",
            str(target),
        ],
        check=True,
    )
    written = ase.io.read(target, index=":")
    assert len(written) == len(atoms)
    for original, actual in zip(atoms.values(), written, strict=True):
        reference = ModelSurface(apply, params).golden_outputs(original)
        if original.constraints:
            assert actual.constraints and np.max(np.abs(reference["forces"][0])) > 1e-5
        np.testing.assert_allclose(
            actual.info["MACE_energy"], reference["energy"], atol=1e-6, rtol=0
        )
        np.testing.assert_allclose(
            actual.arrays["MACE_forces"], reference["forces"], atol=1e-6, rtol=0
        )
        np.testing.assert_allclose(
            actual.info["MACE_stress"], reference["stress"], atol=1e-6, rtol=0
        )


@pytest.mark.parametrize("dtype", ["float64", "float32"])
def test_plain_energy_anchor(dtype):
    run_jax_model_conformance(
        "tiny_plain",
        dtypes=(dtype,),
        artifact_path=Path(__file__).parent / "fixtures" / "tiny_plain.json",
    )
