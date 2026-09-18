"""`charges_key` must actually select the atoms.arrays field the charges come from.

`KeySpecification.arrays_keys` maps PROPERTY NAME -> ATOMS ARRAY KEY:
`config_from_atoms` iterates `for name, atoms_key in arrays_keys.items()` and
stores `atoms.arrays[atoms_key]` under `properties[name]`. MACECalculator wrote
the pair the other way round (`{charges_key: "charges"}`), so "charges" was
never a property name at all, `AtomicData.from_config` fell back to its zeros
default, and `charges_key` was silently inert -- including for DipoleMACE /
EnergyDipoleMACE, whose fixed-charge dipole baseline reads `data["charges"]`
and was therefore always zero.
"""

import numpy as np
import pytest
import torch
from ase import Atoms
from e3nn import o3

from mace import modules
from mace.calculators import MACECalculator
from mace.tools import AtomicNumberTable

CHARGES = np.array([-0.8, 0.4, 0.4])


@pytest.fixture(scope="module", name="dipole_model")
def dipole_model_fixture():
    table = AtomicNumberTable([1, 8])
    torch.manual_seed(0)
    return modules.AtomicDipolesMACE(
        r_max=5,
        num_bessel=8,
        num_polynomial_cutoff=5,
        max_ell=2,
        interaction_cls=modules.interaction_classes[
            "RealAgnosticResidualInteractionBlock"
        ],
        interaction_cls_first=modules.interaction_classes[
            "RealAgnosticResidualInteractionBlock"
        ],
        num_interactions=2,
        num_elements=2,
        hidden_irreps=o3.Irreps("8x0e + 8x1o"),
        MLP_irreps=o3.Irreps("8x0e"),
        gate=torch.nn.functional.silu,
        atomic_energies=None,
        avg_num_neighbors=3,
        atomic_numbers=table.zs,
        correlation=3,
        radial_type="gaussian",
    )


def _water(charges_array_name=None):
    atoms = Atoms(
        numbers=[8, 1, 1],
        positions=[[0.0, -2.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]],
        cell=[6.0] * 3,
        pbc=True,
    )
    if charges_array_name is not None:
        atoms.arrays[charges_array_name] = CHARGES.copy()
    return atoms


def _calc(dipole_model, **kwargs):
    return MACECalculator(
        models=[dipole_model],
        device="cpu",
        default_dtype="float64",
        model_type="DipoleMACE",
        **kwargs,
    )


@pytest.mark.parametrize("charges_key", ["Qs", "REF_charges", "my_charges"])
def test_charges_key_selects_the_arrays_field(dipole_model, charges_key):
    calc = _calc(dipole_model, charges_key=charges_key)
    batch = calc._atoms_to_batch(_water(charges_key))  # pylint: disable=protected-access
    assert np.allclose(batch["charges"].numpy(), CHARGES)


def test_arrays_keys_maps_property_name_to_atoms_key(dipole_model):
    calc = _calc(dipole_model, charges_key="Qs")
    calc._atoms_to_batch(_water("Qs"))  # pylint: disable=protected-access
    assert calc.arrays_keys["charges"] == "Qs"


def test_unconfigured_key_is_not_read(dipole_model):
    """A charges_key the atoms do not carry must not pick up some other field."""
    calc = _calc(dipole_model, charges_key="Qs")
    batch = calc._atoms_to_batch(_water("REF_charges"))  # pylint: disable=protected-access
    assert np.allclose(batch["charges"].numpy(), 0.0)


def test_charges_reach_the_fixed_charge_dipole_baseline(dipole_model):
    """DipoleMACE adds sum(q_i r_i) to the predicted dipole; zeros hid the bug."""
    with_charges = _water("Qs")
    with_charges.calc = _calc(dipole_model, charges_key="Qs")
    without_charges = _water()
    without_charges.calc = _calc(dipole_model, charges_key="Qs")
    assert not np.allclose(
        with_charges.get_dipole_moment(), without_charges.get_dipole_moment()
    )
