"""The two calculator surfaces that scale or select, neither of them asserted.

`energy_units_to_eV` and `length_units_to_A` multiply every energy, force and
stress the calculator returns. Nothing exercised them, and a wrong factor is the
quietest kind of wrong: the numbers stay plausible and a user comparing against
a reference blames the model.

The committee keys are the other half. `energy_comm` and `energy_var` are what an
active-learning loop reads to decide which structures to label next, and only the
energy mean and variance were ever asserted -- at the default conversion of 1.0,
where a factor and its square are indistinguishable. They are not:

    var(c * X) == c**2 * var(X)

which is what these tests pin, and what the calculator got wrong in two separate
ways. `MACECalculator` scaled the ensemble array in place, and `.numpy()` shares
storage with a cpu tensor, so the variance was then taken over already-converted
values and scaled again -- `unit_conv**3`. `MagneticMACECalculator` had the
exponent alone. Both were invisible while the factors stayed at 1.
"""

import numpy as np
import pytest
import torch
import torch.nn.functional as F
from ase import Atoms
from e3nn import o3

from mace import modules
import mace.calculators.mace as mace_calculator_module
from mace.calculators import MACECalculator
from mace.tools import AtomicNumberTable

TABLE = AtomicNumberTable([1, 8])

#: not a round number, so a factor and its square cannot coincide, and neither
#: can a factor and its reciprocal
ENERGY_FACTOR = 2.5
LENGTH_FACTOR = 1.25


def _tiny_model(seed):
    torch.manual_seed(seed)
    return modules.ScaleShiftMACE(
        r_max=4.0,
        num_bessel=4,
        num_polynomial_cutoff=5,
        max_ell=2,
        interaction_cls=modules.interaction_classes[
            "RealAgnosticResidualInteractionBlock"
        ],
        interaction_cls_first=modules.interaction_classes[
            "RealAgnosticInteractionBlock"
        ],
        num_interactions=2,
        num_elements=2,
        hidden_irreps=o3.Irreps("8x0e"),
        MLP_irreps=o3.Irreps("4x0e"),
        gate=F.silu,
        atomic_energies=np.array([-1.0, -5.0]),
        avg_num_neighbors=4.0,
        atomic_numbers=TABLE.zs,
        correlation=2,
        atomic_inter_scale=1.0,
        atomic_inter_shift=0.0,
    ).double()


@pytest.fixture(name="water")
def fixture_water():
    atoms = Atoms(
        "H2O",
        positions=[[0, 0, 0], [0.95, 0, 0], [-0.24, 0.93, 0]],
        cell=[8, 8, 8],
        pbc=True,
    )
    return atoms


def _results(water, models, **kwargs):
    """Evaluate in a float64 scope, since the graph reads the default dtype."""
    previous = torch.get_default_dtype()
    torch.set_default_dtype(torch.float64)
    try:
        atoms = water.copy()
        atoms.calc = MACECalculator(
            models=models, device="cpu", default_dtype="float64", **kwargs
        )
        atoms.get_potential_energy()
        return dict(atoms.calc.results)
    finally:
        torch.set_default_dtype(previous)


# ---------------------------------------------------------------------------
# the unit conversions
# ---------------------------------------------------------------------------


def test_the_energy_conversion_scales_the_energy_and_the_forces(water):
    """`energy_units_to_eV` is linear in the energy and in the force, since a
    force is an energy over a length and only the energy is being converted."""
    plain = _results(water, [_tiny_model(1)])
    scaled = _results(water, [_tiny_model(1)], energy_units_to_eV=ENERGY_FACTOR)

    assert scaled["energy"] == pytest.approx(plain["energy"] * ENERGY_FACTOR)
    assert scaled["forces"] == pytest.approx(plain["forces"] * ENERGY_FACTOR)


def test_the_length_conversion_divides_the_forces_and_cubes_for_the_stress(water):
    """A force is an energy per length and a stress an energy per volume, so the
    length factor enters once and three times respectively -- and inversely."""
    plain = _results(water, [_tiny_model(1)])
    scaled = _results(water, [_tiny_model(1)], length_units_to_A=LENGTH_FACTOR)

    assert scaled["energy"] == pytest.approx(plain["energy"])
    assert scaled["forces"] == pytest.approx(plain["forces"] / LENGTH_FACTOR)
    assert scaled["stress"] == pytest.approx(plain["stress"] / LENGTH_FACTOR**3)


def test_the_default_conversions_change_nothing(water):
    """The identity case, so a test above cannot pass by the factors being
    ignored altogether."""
    plain = _results(water, [_tiny_model(1)])
    explicit = _results(
        water, [_tiny_model(1)], energy_units_to_eV=1.0, length_units_to_A=1.0
    )

    assert explicit["energy"] == pytest.approx(plain["energy"])
    assert explicit["forces"] == pytest.approx(plain["forces"])


# ---------------------------------------------------------------------------
# the committee spread
# ---------------------------------------------------------------------------


def test_a_committee_reports_the_spread_of_the_members_it_averaged(water):
    """The mean is the energy, and the variance is the variance of the members
    the mean came from. Asserted for the arrays as well as the scalar: only the
    energy pair was covered before."""
    results = _results(water, [_tiny_model(1), _tiny_model(2)])

    assert results["energy"] == pytest.approx(np.mean(results["energy_comm"]))
    assert results["energy_var"] == pytest.approx(np.var(results["energy_comm"]))
    assert results["forces"] == pytest.approx(np.mean(results["forces_comm"], axis=0))
    assert results["forces_var"] == pytest.approx(
        np.var(results["forces_comm"], axis=0)
    )


def test_the_spread_stays_the_spread_under_a_unit_conversion(water):
    """The bug this closes. A variance carries the square of whatever scales the
    values, and the calculator applied the factor once -- to data it had already
    converted in place, reaching `unit_conv**3` for the energy.

    Stated as an internal consistency rather than against a remembered number:
    whatever the conversion, the reported variance is the variance of the
    reported members.
    """
    scaled = _results(
        water,
        [_tiny_model(1), _tiny_model(2)],
        energy_units_to_eV=ENERGY_FACTOR,
    )

    assert scaled["energy_var"] == pytest.approx(np.var(scaled["energy_comm"]))
    assert scaled["forces_var"] == pytest.approx(
        np.var(scaled["forces_comm"], axis=0)
    )


def test_the_variance_scales_as_the_square_and_the_members_linearly(water):
    """The same claim from the outside, which is what a caller sees: converting
    the units multiplies the members by the factor and the variance by its
    square."""
    plain = _results(water, [_tiny_model(1), _tiny_model(2)])
    scaled = _results(
        water,
        [_tiny_model(1), _tiny_model(2)],
        energy_units_to_eV=ENERGY_FACTOR,
    )

    assert scaled["energy_comm"] == pytest.approx(
        plain["energy_comm"] * ENERGY_FACTOR
    )
    assert scaled["energy_var"] == pytest.approx(
        plain["energy_var"] * ENERGY_FACTOR**2
    )


def test_a_single_model_reports_no_spread_at_all(water):
    """The keys are a committee feature, so one model must not invent them."""
    results = _results(water, [_tiny_model(1)])

    assert "energy_comm" not in results
    assert "energy_var" not in results
    assert "forces_var" not in results


# ---------------------------------------------------------------------------
# the stress and dipole halves of the committee
# ---------------------------------------------------------------------------


def _dipole_model(seed):
    """A dipole committee, since `dipole_comm` and `dipole_var` only appear for
    a model whose readout is a dipole."""
    torch.manual_seed(seed)
    return modules.AtomicDipolesMACE(
        r_max=4.0,
        num_bessel=4,
        num_polynomial_cutoff=5,
        max_ell=2,
        interaction_cls=modules.interaction_classes[
            "RealAgnosticResidualInteractionBlock"
        ],
        interaction_cls_first=modules.interaction_classes[
            "RealAgnosticInteractionBlock"
        ],
        num_interactions=2,
        num_elements=2,
        hidden_irreps=o3.Irreps("8x0e + 8x1o"),
        MLP_irreps=o3.Irreps("4x0e"),
        gate=F.silu,
        atomic_energies=None,
        avg_num_neighbors=4.0,
        atomic_numbers=TABLE.zs,
        correlation=2,
    ).double()


def _dipole_results(water, models, **kwargs):
    previous = torch.get_default_dtype()
    torch.set_default_dtype(torch.float64)
    try:
        atoms = water.copy()
        calc = MACECalculator(
            models=models,
            device="cpu",
            default_dtype="float64",
            model_type="DipoleMACE",
            **kwargs,
        )
        calc.calculate(atoms)
        return dict(calc.results)
    finally:
        torch.set_default_dtype(previous)


def test_a_committee_reports_the_stress_of_the_members_it_averaged(water):
    results = _results(water, [_tiny_model(0), _tiny_model(1)])

    members = results["stress_comm"]
    assert members.shape == (2, 6), "one Voigt-6 stress per member"
    assert not np.allclose(members[0], members[1]), "two seeds, two stresses"
    assert np.allclose(results["stress_var"], np.var(members, axis=0))


def test_the_stress_spread_is_in_the_same_layout_as_the_stress(water):
    """One layout for the mean and its spread, so a caller can index them the
    same way. `MACECalculator` used to leave these two 3x3 while `stress` was
    Voigt-6, which also disagreed with `MagneticMACECalculator`.
    """
    results = _results(water, [_tiny_model(0), _tiny_model(1)])

    assert np.shape(results["stress"]) == (6,)
    assert np.shape(results["stress_var"]) == (6,)
    assert np.shape(results["stress_comm"]) == (2, 6)


def test_the_committee_axis_survives_the_voigt_conversion(water):
    """The conversion broadcasts, so three members stay three members."""
    results = _results(water, [_tiny_model(0), _tiny_model(1), _tiny_model(2)])

    assert np.shape(results["stress_comm"]) == (3, 6)
    assert np.allclose(np.mean(results["stress_comm"], axis=0), results["stress"])
    assert np.allclose(np.var(results["stress_comm"], axis=0), results["stress_var"])


def test_the_members_are_symmetric_so_the_conversion_loses_nothing(water):
    """Voigt-6 keeps six of nine components and averages the off-diagonal pairs,
    which is only lossless because a MACE stress is symmetric. Asserted on the
    raw 3x3 tensors, since after the conversion the evidence is gone.
    """
    previous = torch.get_default_dtype()
    torch.set_default_dtype(torch.float64)
    try:
        atoms = water.copy()
        calc = MACECalculator(
            models=[_tiny_model(0), _tiny_model(1)],
            device="cpu",
            default_dtype="float64",
        )
        seen = []
        original = mace_calculator_module.full_3x3_to_voigt_6_stress

        def watching(value):
            seen.append(np.asarray(value).copy())
            return original(value)

        mace_calculator_module.full_3x3_to_voigt_6_stress = watching
        try:
            calc.calculate(atoms)
        finally:
            mace_calculator_module.full_3x3_to_voigt_6_stress = original
    finally:
        torch.set_default_dtype(previous)

    tensors = [array for array in seen if array.shape[-2:] == (3, 3)]
    assert tensors, "nothing was converted, so the assertion below proves nothing"
    for array in tensors:
        assert np.allclose(array, np.swapaxes(array, -1, -2))


def test_the_stress_variance_scales_as_the_square_of_the_conversion(water):
    """Same rule as the energy and forces above: a variance carries the square,
    and for stress the conversion is itself energy over length cubed."""
    plain = _results(water, [_tiny_model(0), _tiny_model(1)])
    scaled = _results(
        water, [_tiny_model(0), _tiny_model(1)], energy_units_to_eV=2.0
    )

    assert np.allclose(scaled["stress_comm"], plain["stress_comm"] * 2.0)
    assert np.allclose(scaled["stress_var"], plain["stress_var"] * 4.0)


def test_a_dipole_committee_reports_its_spread(water):
    results = _dipole_results(water, [_dipole_model(0), _dipole_model(1)])

    members = results["dipole_comm"]
    assert np.shape(results["dipole"]) == (3,)
    assert members.shape == (2, 3)
    assert not np.allclose(members[0], members[1])
    assert np.allclose(results["dipole_var"], np.var(members, axis=0))


def test_the_dipole_spread_keeps_the_layout_of_the_dipole(water):
    """Unlike stress: `dipole` and `dipole_var` are both (3,), so the pair is
    indexable the same way."""
    results = _dipole_results(water, [_dipole_model(0), _dipole_model(1)])

    assert np.shape(results["dipole"]) == np.shape(results["dipole_var"])


def test_a_single_dipole_model_reports_no_spread(water):
    """The committee keys are a committee's, not a decoration on every run."""
    results = _dipole_results(water, [_dipole_model(0)])

    assert "dipole" in results
    assert "dipole_comm" not in results
    assert "dipole_var" not in results
