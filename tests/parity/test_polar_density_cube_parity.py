"""The density-cube interpolators against the frozen tree's, on the same inputs.

The coefficients are made up rather than predicted, since what is compared is
the sampling of a given density: in reciprocal space on a crystal and a slab,
density, potential and corrected potential alike, and in real space on a
molecule and on a structure periodic along two axes that are not a slab's.
"""

from __future__ import annotations

import numpy as np
import pytest
from ase import Atoms
from mace_torch.electrostatics.density_cube import (
    FourierDensity,
    RealSpaceDensity,
    make_grid,
)

pytest.importorskip("graph_longrange")

pytestmark = pytest.mark.polar

#: The fp64 row of the golden tolerance table, as in the other parity tests.
TOLERANCE = 1e-6

MULTIPOLES = np.array(
    [
        [-0.4, 0.01, 0.02, 0.03],
        [0.25, -0.01, 0.0, 0.02],
        [0.15, 0.0, -0.02, -0.01],
    ]
)
POSITIONS = [[2.0, 2.0, 2.5], [2.9, 2.1, 2.4], [1.8, 2.8, 2.6]]


def structure(pbc) -> Atoms:
    return Atoms(
        numbers=[8, 1, 1],
        positions=POSITIONS,
        cell=[[5.0, 0.0, 0.0], [0.4, 5.5, 0.0], [0.0, 0.0, 7.0]],
        pbc=pbc,
    )


@pytest.mark.parametrize("pbc", [(True, True, True), (True, True, False)])
def test_the_reciprocal_space_density_and_potential_are_the_frozen_tree_s(
    fp64, isolated, pbc
):
    from mace.cli.polar_density_cube import PotentialInterpolator

    atoms = structure(pbc)
    coords = make_grid(atoms, (6, 7, 9))
    field, fermi = np.array([0.01, -0.02, 0.03]), 0.2
    settings = {"sigma": 0.9, "multipoles_max_l": 1, "kspace_cutoff": 4.0}
    with pytest.warns(UserWarning):
        legacy = PotentialInterpolator(**settings)
    reference = legacy(atoms, MULTIPOLES, field, fermi, coords)
    result = FourierDensity(**settings)(atoms, MULTIPOLES, field, fermi, coords)
    for expected, got in zip(reference, result, strict=True):
        assert np.max(np.abs(expected - got)) < TOLERANCE
        assert np.max(np.abs(expected - got)) < 1e-12


@pytest.mark.parametrize("pbc", [(False, False, False), (True, False, True)])
def test_the_real_space_density_is_the_frozen_tree_s(fp64, isolated, pbc):
    from mace.cli.polar_density_cube import RealSpaceDensityInterpolator

    atoms = structure(pbc)
    coords = make_grid(atoms, (6, 7, 9))
    settings = {"sigma": 0.9, "multipoles_max_l": 1, "cutoff_factor": 5.0}
    reference, _, _ = RealSpaceDensityInterpolator(**settings, chunk_size=13)(
        atoms, MULTIPOLES, coords
    )
    result, _, _ = RealSpaceDensity(**settings, chunk_size=13)(
        atoms, MULTIPOLES, coords
    )
    assert np.max(np.abs(reference - result)) < 1e-12
