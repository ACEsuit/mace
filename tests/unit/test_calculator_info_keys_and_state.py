"""`info_keys` and `MACECalculator.check_state`.

Two parts of the calculator's contract that nothing exercised.

`info_keys` maps a PROPERTY NAME to the `atoms.info` key it is read from, the same
direction as `arrays_keys` (see tests/unit/test_calculator_charges_key.py). Its
default is not empty, so a caller who overrides it partially loses the entries
they did not restate, which is the kind of thing a default that nobody asserts
drifts into.

`check_state` is ASE's recalculation trigger: whatever it returns non-empty for
gets recomputed, and whatever it misses is served from cache. MACE overrides it to
add `info`, because a MACE model can read `atoms.info` (total charge, total spin,
external field) and ASE's own implementation looks only at numbers, positions,
cell and pbc. So a change to a charge would otherwise return the previous energy.
`MagneticMACECalculator` has its own override and its own test; this is the base
one.
"""

import numpy as np
import pytest
import torch
from ase import Atoms
from e3nn import o3

from mace import modules
from mace.calculators import MACECalculator
from mace.tools import AtomicNumberTable


@pytest.fixture(scope="module", name="model_path")
def fixture_model_path(tmp_path_factory):
    table = AtomicNumberTable([1, 8])
    torch.manual_seed(0)
    model = modules.MACE(
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
        atomic_energies=np.array([1.0, 3.0]),
        avg_num_neighbors=3,
        atomic_numbers=table.zs,
        correlation=3,
        radial_type="gaussian",
    ).double()
    path = tmp_path_factory.mktemp("info_keys") / "model.pt"
    torch.save(model, path)
    return path


def water():
    return Atoms(
        numbers=[8, 1, 1],
        positions=[[0.0, -2.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]],
        cell=[6.0] * 3,
        pbc=True,
    )


def calculator(model_path, **kwargs):
    return MACECalculator(
        model_paths=str(model_path), device="cpu", default_dtype="float64", **kwargs
    )


# ---------------------------------------------------------------------------
# info_keys
# ---------------------------------------------------------------------------


def test_the_default_info_keys_are_the_three_documented_ones(model_path):
    """Stated here so removing one is a test failure rather than a silent change
    in what the calculator reads off a structure."""
    calc = calculator(model_path)

    assert calc.info_keys == {
        "total_spin": "spin",
        "total_charge": "charge",
        "external_field": "external_field",
    }


def test_info_keys_maps_property_name_to_the_info_key(model_path):
    """Same direction as `arrays_keys`: property name first, `atoms.info` key
    second. Reversed, the property would silently never be found."""
    calc = calculator(model_path, info_keys={"total_charge": "Q"})

    assert calc.info_keys["total_charge"] == "Q"


def test_an_override_replaces_the_defaults_rather_than_extending_them(model_path):
    """The trap in overriding it: the entries not restated are gone."""
    calc = calculator(model_path, info_keys={"total_charge": "Q"})

    assert "total_spin" not in calc.info_keys


def test_the_configured_info_key_is_the_one_read(model_path):
    """Behaviour rather than the attribute: the value has to arrive in the batch
    the model is given, through the configured key.

    Checked on the batch instead of the energy because a plain `MACE` ignores the
    total charge, so an energy comparison would pass whether the key was read or
    not.
    """
    calc = calculator(model_path, info_keys={"total_charge": "Q"})
    atoms = water()
    atoms.info["Q"] = -1.0

    batch = calc._atoms_to_batch(atoms)  # pylint: disable=protected-access
    batch = batch[0] if isinstance(batch, tuple) else batch

    assert float(batch.to_dict()["total_charge"]) == -1.0


def test_the_default_key_is_not_read_once_it_has_been_remapped(model_path):
    """The other half: after remapping to `Q`, a value left under `charge` is
    ignored. Without this, a calculator that read both would pass the test above.
    """
    calc = calculator(model_path, info_keys={"total_charge": "Q"})
    atoms = water()
    atoms.info["charge"] = -1.0

    batch = calc._atoms_to_batch(atoms)  # pylint: disable=protected-access
    batch = batch[0] if isinstance(batch, tuple) else batch

    assert float(batch.to_dict()["total_charge"]) == 0.0


# ---------------------------------------------------------------------------
# check_state
# ---------------------------------------------------------------------------


def test_nothing_changed_means_nothing_to_recompute(model_path):
    calc = calculator(model_path)
    atoms = water()
    atoms.calc = calc
    atoms.get_potential_energy()

    assert calc.check_state(atoms) == []


def test_moving_an_atom_is_a_change(model_path):
    """ASE's own part of the contract, which the override must not lose."""
    calc = calculator(model_path)
    atoms = water()
    atoms.calc = calc
    atoms.get_potential_energy()

    moved = atoms.copy()
    moved.positions[0, 0] += 0.1

    assert "positions" in calc.check_state(moved)


def test_a_changed_info_entry_is_a_change(model_path):
    """The reason for the override. ASE looks at numbers, positions, cell and pbc,
    so a total charge that the model reads would otherwise return a cached energy.
    """
    calc = calculator(model_path)
    atoms = water()
    atoms.info["charge"] = 0.0
    atoms.calc = calc
    atoms.get_potential_energy()

    charged = atoms.copy()
    charged.info["charge"] = -1.0

    assert "info" in calc.check_state(charged)


def test_a_new_info_entry_is_a_change(model_path):
    calc = calculator(model_path)
    atoms = water()
    atoms.calc = calc
    atoms.get_potential_energy()

    annotated = atoms.copy()
    annotated.info["charge"] = -1.0

    assert "info" in calc.check_state(annotated)


def test_an_array_valued_info_entry_is_not_compared(model_path):
    """The override skips ndarray values on purpose: `!=` on an array is an array,
    and truth-testing it raises. Pinned so the skip is a decision, not a bug
    waiting to be reintroduced as an exception.
    """
    calc = calculator(model_path)
    atoms = water()
    atoms.info["external_field"] = np.array([0.0, 0.0, 0.0])
    atoms.calc = calc
    atoms.get_potential_energy()

    changed_field = atoms.copy()
    changed_field.info["external_field"] = np.array([0.0, 0.0, 1.0])

    assert calc.check_state(changed_field) == []
