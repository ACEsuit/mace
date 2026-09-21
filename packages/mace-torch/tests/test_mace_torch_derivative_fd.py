"""Derivatives against finite differences, and the sign conventions.

The engine's numbers are checked against the definition of the derivative
rather than against another implementation of it. A parity run tells you two
implementations agree; a finite difference tells you which one is right.

Everything here is fp64 only. A central difference at ``h = 1e-4`` is rounding
noise in fp32, so running these at both precisions would report a failure of
the method as a failure of the code.
"""

from __future__ import annotations

import numpy as np
import torch
from conftest import fp64_only
from mace_torch_engine_fixtures import (
    build_engine,
    build_graph,
    crystal,
    energy_of,
    molecule,
)

PERIODIC = (True, True, True)


def five_point_forces(engine, positions, numbers, step=1e-4):
    """``-dE/dx`` by a five-point central difference, atom by atom."""
    gradient = np.zeros_like(positions)
    for atom in range(len(positions)):
        for axis in range(3):
            samples = []
            for offset in (-2, -1, 1, 2):
                moved = positions.copy()
                moved[atom, axis] += offset * step
                samples.append(energy_of(engine, moved, numbers))
            gradient[atom, axis] = (
                samples[0] - 8 * samples[1] + 8 * samples[2] - samples[3]
            ) / (12 * step)
    return -gradient


def strain_gradient(engine, positions, numbers, cell, step=1e-5):
    """``dE/dstrain`` by a central difference in each strain component."""

    def energy_at(strain):
        symmetric = 0.5 * (strain + strain.T)
        return energy_of(
            engine,
            positions + positions @ symmetric,
            numbers,
            cell + cell @ symmetric,
            PERIODIC,
        )

    gradient = np.zeros((3, 3))
    for row in range(3):
        for column in range(3):
            plus = np.zeros((3, 3))
            plus[row, column] = step
            gradient[row, column] = (energy_at(plus) - energy_at(-plus)) / (2 * step)
    return 0.5 * (gradient + gradient.T)


@fp64_only
def test_forces_are_minus_the_energy_gradient():
    engine = build_engine()
    positions, numbers = molecule()

    forces = engine(build_graph(positions, numbers), compute=("forces",)).forces
    assert forces is not None
    reference = five_point_forces(engine, positions, numbers)

    assert np.abs(forces.detach().numpy()).max() > 1e-3, (
        "the forces are essentially zero, so agreeing with a finite difference "
        "says nothing about the sign or the magnitude"
    )
    deviation = np.abs(forces.detach().numpy() - reference).max()
    assert deviation < 1e-6, (
        f"the forces differ from the five-point central difference by "
        f"{deviation:.3e} eV/A, which is far above the method's own error"
    )


@fp64_only
def test_the_forces_of_an_isolated_molecule_sum_to_zero():
    """Newton's third law, which no finite difference is needed to state."""
    engine = build_engine()
    positions, numbers = molecule()
    forces = engine(build_graph(positions, numbers), compute=("forces",)).forces
    assert np.abs(forces.detach().numpy().sum(axis=0)).max() < 1e-12


@fp64_only
def test_the_stress_is_the_strain_gradient_over_the_volume():
    """All six independent components, against finite strains."""
    engine = build_engine()
    positions, numbers, cell = crystal()
    result = engine(
        build_graph(positions, numbers, cell, PERIODIC), compute=("stress", "virials")
    )
    stress = result.stress.detach().numpy()[0]
    volume = float(np.linalg.det(cell))
    reference = strain_gradient(engine, positions, numbers, cell) / volume

    assert np.abs(stress).max() > 1e-6, (
        "the stress is essentially zero, so the comparison below is vacuous"
    )
    for row in range(3):
        for column in range(row, 3):
            deviation = abs(stress[row, column] - reference[row, column])
            assert deviation < 1e-8, (
                f"stress component ({row}, {column}) is "
                f"{stress[row, column]:.8e} against a finite-strain "
                f"{reference[row, column]:.8e}, off by {deviation:.3e}"
            )


@fp64_only
def test_the_virial_is_minus_the_strain_gradient():
    """The sign, stated against the definition rather than against the stress.

    This is the one the ticket calls the first candidate for a silent porting
    bug, and it is worth pinning on its own: the virial and the stress carry
    **opposite** signs, and a port that gives them the same one produces
    numbers of the right magnitude throughout.
    """
    engine = build_engine()
    positions, numbers, cell = crystal()
    result = engine(
        build_graph(positions, numbers, cell, PERIODIC), compute=("stress", "virials")
    )
    virials = result.virials.detach().numpy()[0]
    reference = strain_gradient(engine, positions, numbers, cell)

    assert np.abs(virials).max() > 1e-6
    assert np.abs(virials + reference).max() < 1e-7, (
        "the virial is not minus the strain gradient"
    )


@fp64_only
def test_the_stress_and_the_virial_are_opposite():
    """``stress * V == -virials``, to the rounding of one divide and multiply.

    Both come from the same gradient in the same call, so the only thing
    between them is dividing by the volume and multiplying it back. That is
    not exact in floating point, so the bound is relative to the magnitudes
    rather than zero: an earlier version of this test asserted exact equality
    and passed only because those particular numbers happened to round back.
    """
    engine = build_engine()
    positions, numbers, cell = crystal()
    result = engine(
        build_graph(positions, numbers, cell, PERIODIC), compute=("stress", "virials")
    )
    volume = float(np.linalg.det(cell))
    virials = result.virials.detach().numpy()
    residue = np.abs(result.stress.detach().numpy() * volume + virials).max()
    bound = 1e-12 * max(np.abs(virials).max(), 1.0)
    assert residue < bound, (
        f"stress * V + virials is {residue:.3e}, above the {bound:.3e} that one "
        f"divide and multiply can account for"
    )


@fp64_only
def test_the_stress_of_an_aperiodic_structure_is_zero():
    """A molecule's cell is the padding box the neighbour list invents.

    Dividing a real strain derivative by that box's volume gives a number set
    by the padding, and the same molecule in a larger box would report a
    different stress for no physical reason.
    """
    engine = build_engine()
    positions, numbers = molecule()
    result = engine(build_graph(positions, numbers), compute=("stress", "virials"))

    assert torch.all(result.stress == 0.0)
    assert np.abs(result.virials.detach().numpy()).max() > 0.0, (
        "the virial was masked as well, and it should not be: it needs no "
        "volume, so it is the stress alone that is undefined here"
    )


@fp64_only
def test_the_force_path_differentiates_twice():
    """Force training backpropagates through the derivative, so the second
    derivative has to exist and be correct."""
    engine = build_engine()
    positions, numbers = molecule()
    graph = build_graph(positions, numbers)
    graph["positions"] = graph["positions"].clone().requires_grad_(True)

    forces = engine(graph, compute=("forces",), training=True).forces
    second = torch.autograd.grad(forces.pow(2).sum(), graph["positions"])[0]

    assert second is not None
    assert torch.isfinite(second).all()
    assert float(second.abs().max()) > 0.0


@fp64_only
def test_the_forces_pass_gradcheck():
    """The whole chain, against numerical differentiation of the chain."""
    engine = build_engine()
    positions, numbers = molecule()
    graph = build_graph(positions, numbers)

    def energy(coordinates):
        moving = dict(graph)
        moving["positions"] = coordinates
        return engine(moving, compute=()).total_energy

    start = graph["positions"].clone().requires_grad_(True)
    assert torch.autograd.gradcheck(energy, (start,), eps=1e-6, atol=1e-7)
    assert torch.autograd.gradgradcheck(energy, (start,), eps=1e-6, atol=1e-5)
