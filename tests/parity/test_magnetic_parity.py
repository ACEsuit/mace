"""The magnetic model against the live frozen one, and against its reference.

The committed magnetic anchor, a seeded ``MagneticScaleShiftMACE`` over oxygen
and iron with the one-body term and the pair repulsion on, is converted and
evaluated on the structures its reference was taken on. The energy, the
per-atom energies, the forces and ``magforces`` are compared with the live
model in the same process, and with the committed numbers at the fp64 row.

Loading the anchor needs sphericart, because the frozen model holds a
sphericart module. The rewrite does not: its moment harmonics are its own
harmonic polynomials, which are what sphericart's solid harmonics compute, so
only this file is marked ``magnetic``.
"""

from __future__ import annotations

import json

import numpy as np
import pytest
import torch
from ase import Atoms
from mace_core.neighbors import get_neighborhood
from mace_core.observables import DEFAULT_CATALOGUE
from mace_torch.physics import DerivativeEngine

from tests.golden import harness
from tests.parity.magnetic_convert import convert_magnetic

pytestmark = pytest.mark.magnetic

#: Same process, same inputs: agreement to rounding.
TOLERANCE = 1e-12
REFERENCE = harness.REFERENCES_DIR / "tiny_magnetic_e3nn_cpu_fp64.json"
ENERGY = DEFAULT_CATALOGUE.observable("energy")
MAGMOM = DEFAULT_CATALOGUE.input("magmom")


def magnetic_surfaces():
    from tests.golden import magnetic_surfaces as surfaces

    return surfaces


def v1_graph(atoms, numbers, cutoff, moments):
    positions = atoms.get_positions()
    pbc = tuple(bool(p) for p in atoms.pbc)
    cell = np.asarray(atoms.cell) if any(pbc) else None
    neighborhood = get_neighborhood(positions, cutoff, pbc, cell)
    return {
        "positions": torch.tensor(positions),
        "atomic_numbers": torch.tensor(atoms.get_atomic_numbers()),
        "element_index": torch.tensor(
            [numbers.index(int(z)) for z in atoms.get_atomic_numbers()]
        ),
        "edge_index": torch.tensor(neighborhood.edge_index),
        "shifts": torch.tensor(neighborhood.shifts),
        "unit_shifts": torch.tensor(neighborhood.unit_shifts),
        "cell": torch.tensor(np.asarray(neighborhood.cell, dtype=float)).reshape(
            1, 3, 3
        ),
        "pbc": torch.tensor([pbc]),
        "batch": torch.zeros(len(atoms), dtype=torch.long),
        "num_graphs": 1,
        "head": torch.zeros(1, dtype=torch.long),
        "magmom": torch.tensor(np.asarray(moments, dtype=float)),
    }


def gap(a, b) -> float:
    return float(
        (torch.as_tensor(a).detach() - torch.as_tensor(b).detach()).abs().max()
    )


@pytest.fixture(name="converted", scope="module")
def fixture_converted():
    previous = torch.get_default_dtype()
    torch.set_default_dtype(torch.float64)
    try:
        legacy = magnetic_surfaces().load_anchor()
        engine = DerivativeEngine(
            convert_magnetic(legacy, [ENERGY]), ENERGY, None, inputs=[MAGMOM]
        )
        yield legacy, engine
    finally:
        torch.set_default_dtype(previous)


def evaluate(engine, legacy, atoms, moments, compute=("forces", "magforces")):
    numbers = [int(z) for z in legacy.atomic_numbers.tolist()]
    return engine(
        v1_graph(atoms, numbers, float(legacy.r_max), moments), compute=compute
    )


@pytest.mark.parametrize(
    "name",
    [
        "mag_fe3_canted",
        "mag_fe_atom",
        "mag_fe_dimer_afm",
        "mag_fe_dimer_fm",
        "mag_feo_cluster",
    ],
)
def test_the_converted_model_matches_the_live_legacy_model(fp64, converted, name):
    surfaces = magnetic_surfaces()
    legacy, engine = converted
    atoms = surfaces.magnetic_fixtures()[name]
    moments = atoms.arrays[surfaces.MAGMOM_KEY]
    reference = legacy(
        surfaces.build_batch(legacy, atoms),
        compute_force=True,
        compute_magforces=True,
    )
    result = evaluate(engine, legacy, atoms, moments)

    assert gap(result.total_energy, reference["energy"]) < TOLERANCE
    assert gap(result.node_energies, reference["node_energy"]) < TOLERANCE
    assert gap(result.forces, reference["forces"]) < TOLERANCE
    assert gap(result.extras["magforces"], reference["magforces"]) < TOLERANCE
    # The moments move the energy, so the comparison is not of a model that
    # ignores them.
    assert float(reference["magforces"].abs().max()) > 1e-3


def test_the_converted_model_reproduces_the_committed_reference(fp64, converted):
    """The acceptance number: the committed reference, at the fp64 row."""
    surfaces = magnetic_surfaces()
    legacy, engine = converted
    reference = json.loads(REFERENCE.read_text(encoding="utf-8"))
    row = harness.FP64_CPU_REFERENCE
    assert reference["provenance"]["tolerance_row"] == row.name
    fixtures = surfaces.magnetic_fixtures()
    assert sorted(fixtures) == sorted(reference["fixtures"])
    for name, entry in reference["fixtures"].items():
        atoms = fixtures[name]
        moments = np.asarray(entry["inputs"]["magmom"]["value"], dtype=float)
        assert np.array_equal(moments, atoms.arrays[surfaces.MAGMOM_KEY]), name
        result = evaluate(engine, legacy, atoms, moments)
        produced = {
            "energy": result.total_energy.detach().numpy()[0],
            "energies": result.node_energies.detach().numpy(),
            "forces": result.forces.detach().numpy(),
            "magforces": result.extras["magforces"].detach().numpy(),
        }
        assert sorted(produced) == sorted(entry["outputs"]), name
        for channel, recorded in entry["outputs"].items():
            np.testing.assert_allclose(
                produced[channel],
                np.asarray(recorded["value"], dtype=float),
                atol=row.atol,
                rtol=row.rtol,
                err_msg=f"{name}/{channel}",
            )


def test_a_periodic_crystal_s_stress_matches_the_live_legacy_model(fp64, converted):
    """The fixtures are all molecules, which have no stress. A bcc iron cell
    with canted moments has one, and the strain goes through the moment
    blocks' edges like any other."""
    surfaces = magnetic_surfaces()
    legacy, engine = converted
    atoms = Atoms(
        "Fe2",
        scaled_positions=[[0.0, 0.0, 0.0], [0.5, 0.5, 0.5]],
        cell=np.eye(3) * 2.87 + np.array([[0.0, 0.05, 0.0]] * 3),
        pbc=True,
    )
    moments = np.array([[0.2, 0.1, 2.2], [-0.1, 0.3, -2.1]])
    atoms.arrays[surfaces.MAGMOM_KEY] = moments
    reference = legacy(
        surfaces.build_batch(legacy, atoms),
        compute_force=True,
        compute_stress=True,
        compute_magforces=True,
    )
    result = evaluate(engine, legacy, atoms, moments, ("forces", "stress", "magforces"))
    assert gap(result.total_energy, reference["energy"]) < TOLERANCE
    assert gap(result.forces, reference["forces"]) < TOLERANCE
    assert gap(result.stress, reference["stress"]) < TOLERANCE
    assert gap(result.extras["magforces"], reference["magforces"]) < TOLERANCE
    assert float(reference["stress"].abs().max()) > 1e-4
