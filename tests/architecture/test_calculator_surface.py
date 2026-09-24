"""Every name the frozen tree's calculator writes, accounted for in v1.

Layer (b) of the output surface: the keys ``MACECalculator.calculate`` can put
in ``results``, scanned from the frozen source rather than listed by hand. The
one v1 calculator has to produce each key of the model families v1 builds,
energy, charge-aware, dipole and dielectric, and every other key has to belong
to a family whose model another ticket owns. A key in neither is one the
rewrite would lose without anyone deciding to.
"""

from __future__ import annotations

import pytest

from tests.architecture.observable_coverage import legacy_calculator_keys

pytest.importorskip("mace_torch")

from mace_torch.calculators.ase_calculator import (
    declared_properties,
    produced_outputs,
)
from mace_torch.models.dipoles import DipoleSettings
from mace_torch.models.electrostatics import POLAR_EXTRA_ROWS, PolarModel
from mace_torch.models.outputs import ENERGY_EXTRA_ROWS

#: The keys whose model v1 does not build yet, and who builds it. LES reads
#: its applied field through the Born charges it predicts, and the calculator
#: term that turns them into forces is ported with that model.
OTHER_FAMILIES = {
    "LES_alphas": "the LES model",
    "LES_kappas": "the LES model",
    "bec": "the LES model",
    "MACE_magmoms": "the magnetic model",
}

FIXED = DipoleSettings(charges="fixed")
PREDICTED = DipoleSettings(charges="predicted", polarizability=True)

#: Each v1 model family, as the outputs a calculator reads from it.
FAMILIES = {
    "energy": produced_outputs(
        ("energy", "forces"), ENERGY_EXTRA_ROWS, atomic_stresses=True
    ),
    "charge-aware": produced_outputs(
        ("energy", "forces"),
        {**ENERGY_EXTRA_ROWS, **POLAR_EXTRA_ROWS},
        atomic_stresses=True,
        produced=sorted(PolarModel.PRODUCED),
    ),
    "dipole": produced_outputs(("dipole",), FIXED.extra_rows, produced=FIXED.produced),
    "dielectric": produced_outputs(
        ("dipole", "polarizability"),
        PREDICTED.extra_rows,
        produced=PREDICTED.produced,
    ),
}


def produced_keys() -> set[str]:
    return {
        key
        for outputs in FAMILIES.values()
        for key in declared_properties(outputs, committee=True)
    }


def test_every_legacy_key_is_produced_or_belongs_to_another_family():
    produced = produced_keys()
    legacy = legacy_calculator_keys()
    unaccounted = sorted(legacy - produced - set(OTHER_FAMILIES))
    assert not unaccounted, (
        f"{unaccounted} are written by the frozen tree's calculator and "
        f"neither produced by a v1 model family nor assigned to another."
    )
    assert not produced & set(OTHER_FAMILIES)
    assert produced <= legacy, sorted(produced - legacy)


@pytest.mark.parametrize(
    ("family", "expected"),
    [
        ("dipole", {"dipole"}),
        ("dielectric", {"dipole", "charges", "polarizability", "polarizability_sh"}),
    ],
)
def test_a_response_model_writes_the_frozen_tree_s_keys_and_no_energy(
    family, expected
):
    """The frozen tree's ``DipoleMACE`` and ``DipolePolarizabilityMACE``
    implemented properties, with ``dipole_comm`` and ``dipole_var`` for a
    committee."""
    single = set(declared_properties(FAMILIES[family], committee=False))
    assert single == expected
    committee = set(declared_properties(FAMILIES[family], committee=True))
    assert committee == expected | {"dipole_comm", "dipole_var"}
