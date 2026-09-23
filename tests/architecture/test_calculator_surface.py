"""Every name the frozen tree's calculator writes, accounted for in v1.

Layer (b) of the output surface: the keys ``MACECalculator.calculate`` can put
in ``results``, scanned from the frozen source rather than listed by hand. The
v1 energy-family calculator has to produce each energy-family key, and every
other key has to belong to a family another ticket owns. A key in neither is
one the rewrite would lose without anyone deciding to.
"""

from __future__ import annotations

import pytest

from tests.architecture.observable_coverage import legacy_calculator_keys

pytest.importorskip("mace_torch")

from mace_torch.calculators.ase_calculator import declared_properties

#: The keys the dipole, dielectric, polar, LES and magnetic calculators write,
#: which are the model-family surfaces on the one calculator, a ticket of their
#: own. ``dipole`` is among them, and with it its two committee keys.
OTHER_FAMILIES = frozenset(
    {
        "LES_alphas",
        "LES_kappas",
        "MACE_magmoms",
        "bec",
        "charges",
        "density_coefficients",
        "dipole",
        "dipole_comm",
        "dipole_var",
        "electron_energy",
        "electrostatic_energy",
        "fukui_functions",
        "interaction_energy",
        "polarizability",
        "polarizability_sh",
        "spin_charge_density",
        "spins",
    }
)


def test_every_legacy_key_is_produced_or_belongs_to_another_family():
    produced = set(declared_properties(committee=True, atomic_stresses=True))
    legacy = legacy_calculator_keys()
    unaccounted = sorted(legacy - produced - OTHER_FAMILIES)
    assert not unaccounted, (
        f"{unaccounted} are written by the frozen tree's calculator and "
        f"neither produced by the v1 energy family nor assigned to another."
    )
    assert not produced & OTHER_FAMILIES
    assert produced <= legacy, sorted(produced - legacy)
