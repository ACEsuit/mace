"""MACEOutput: the typed replacement for the forward's dictionary of tensors.

The tests run on numpy arrays throughout. That is the point rather than a
convenience: ``MACEOutput`` is generic over the tensor type so that one class
serves torch, jax and neither, and a test suite that only ever exercised it
with torch would not notice the day it stopped being framework-free.
"""

import subprocess
import sys

import numpy as np
import pytest
from mace_core.observables import load_default_catalogue
from mace_core.outputs import (
    CORE_FIELD_NAMES,
    FIELD_BY_OBSERVABLE,
    OBSERVABLE_BY_FIELD,
    MACEOutput,
)


def test_the_six_core_fields_are_the_declared_ones():
    """The type's shape, pinned. `extras` is not one of them."""
    assert CORE_FIELD_NAMES == (
        "total_energy",
        "node_energies",
        "forces",
        "stress",
        "virials",
        "dipole",
    )


def test_an_empty_output_holds_nothing():
    output = MACEOutput[np.ndarray]()
    assert output.names() == ()
    assert output.get("forces") is None
    assert "forces" not in output


def test_core_fields_round_trip_numpy_arrays():
    forces = np.zeros((4, 3))
    output = MACEOutput(total_energy=np.array([-1.5]), forces=forces)
    assert output.get("total_energy") is output.total_energy
    assert output.get("forces") is forces
    assert output.names() == ("energy", "forces")


def test_energy_reaches_the_total_energy_field_under_either_name():
    """The one place an observable name and a field name differ."""
    assert FIELD_BY_OBSERVABLE == {"energy": "total_energy"}
    assert OBSERVABLE_BY_FIELD == {"total_energy": "energy"}
    output = MACEOutput(total_energy=np.array([2.0]))
    assert output.get("energy") is output.get("total_energy")
    assert "energy" in output


def test_extras_carries_what_the_core_fields_do_not():
    """The escape hatch is load-bearing: 43 legacy keys, six core fields."""
    output = MACEOutput(extras={"latent_charges": np.zeros(4)})
    assert "latent_charges" in output
    latent_charges = output.get("latent_charges")
    assert latent_charges is not None
    assert latent_charges.shape == (4,)
    assert output.names() == ("latent_charges",)


def test_a_name_nobody_wrote_is_absent_rather_than_an_error():
    """A consumer iterating declared observables asks for names that a given
    model did not compute; that is not an error, it is a `None`."""
    output = MACEOutput(extras={"charges": np.zeros(2)})
    assert output.get("magforces") is None
    assert "magforces" not in output


def test_a_core_field_left_none_is_not_reported_as_present():
    output = MACEOutput(forces=None, extras={})
    assert output.names() == ()


def test_two_outputs_do_not_share_an_extras_dictionary():
    """`extras` has a default factory; a shared mutable default would make one
    model's outputs appear in another's."""
    first = MACEOutput[np.ndarray]()
    second = MACEOutput[np.ndarray]()
    first.extras["dipole_moment"] = np.zeros(3)
    assert second.extras == {}


def test_the_class_is_generic_over_the_tensor_type():
    """Subscripting has to work with no framework installed at all: the kernel
    Protocol reuses this pattern, so it cannot depend on torch being there."""
    assert MACEOutput[np.ndarray] is not None
    assert isinstance(MACEOutput[np.ndarray](), MACEOutput)


@pytest.mark.parametrize("framework", ["torch", "jax", "e3nn"])
def test_importing_mace_core_imports_no_framework(framework):
    """Run in a fresh interpreter on purpose. Asserting this in-process would
    pass whenever some earlier test in the session had already imported torch,
    which is the case the assertion exists for."""
    probe = (
        "import sys, mace_core, mace_core.observables, mace_core.outputs\n"
        f"assert {framework!r} not in sys.modules, "
        f"'importing mace_core pulled in {framework}'\n"
    )
    subprocess.run([sys.executable, "-c", probe], check=True)


@pytest.mark.parametrize("key", ["forces", "energy", "total_energy", "dipole"])
def test_extras_may_not_shadow_a_core_field(key):
    """A value written into `extras` under a core field's name would be stored
    and never read: the field it shadows stays `None`."""
    with pytest.raises(ValueError, match="cannot also be"):
        MACEOutput(extras={key: np.zeros(3)})


def test_a_name_that_is_not_a_core_field_is_fine_in_extras():
    """Deliberately not a near-miss of a field name. `node_energy` would be a
    bad example here: it is the legacy spelling of `node_energies`, so using it
    would read as an endorsement of putting the per-atom energy in `extras`
    while the field that owns it stays `None`. v1 renames that key instead."""
    out = MACEOutput(extras={"latent_charges": np.zeros(3)})
    assert out.names() == ("latent_charges",)


def test_membership_and_listing_agree_on_every_name():
    """The two accessors have to use one vocabulary.

    They did not: `"energy" in output` resolved the alias and `names()` yielded
    the storage field, so a consumer intersecting a catalogue's names with an
    output's dropped the energy while membership said it was there. Asserted
    over both directions rather than over the one alias, so a second entry in
    `FIELD_BY_OBSERVABLE` cannot reopen it.
    """
    output = MACEOutput(
        total_energy=np.array([-1.5]),
        forces=np.zeros((4, 3)),
        extras={"latent_charges": np.zeros(4)},
    )
    for name in output.names():
        assert name in output, name
        assert output.get(name) is not None, name
    for observable, field in FIELD_BY_OBSERVABLE.items():
        assert field not in output.names()
        if getattr(output, field) is not None:
            assert observable in output.names()


def test_a_catalogue_name_finds_its_value_in_an_output():
    """The correlation the two vocabularies exist to allow."""
    catalogue = load_default_catalogue()
    output = MACEOutput(total_energy=np.array([-1.5]), forces=np.zeros((4, 3)))
    shared = set(catalogue.names()) & set(output.names())
    assert shared == {"energy", "forces"}
    assert all(output.get(name) is not None for name in shared)
