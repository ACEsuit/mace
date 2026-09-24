"""Turning an E0 declaration into one energy per element.

An isolated-atom energy is a per-species constant that every energy the head
trains on is shifted by, so a missing one does not read as a missing number. It
reads as a model that fits slightly worse than it should, forever. The four
refusals below are the module's reason to exist: each replaces a place where
the frozen tree notices the same thing and carries on.
"""

import numpy as np
import pytest
from mace_core.config.e0s import (
    E0sAverage,
    E0sEstimated,
    E0sFromFoundation,
    E0sIsolatedAtoms,
    E0sTable,
)
from mace_core.data.configuration import Configuration
from mace_core.data.e0_resolution import E0ResolutionError, resolve_e0s
from mace_core.data.xyz import ISOLATED_ATOM_CONFIG_TYPE
from mace_core.elements.number_table import AtomicNumberTable

Z_TABLE = AtomicNumberTable([1, 8])

#: The two references the fits below should recover.
HYDROGEN, OXYGEN = -13.6, -2040.0


def configuration(numbers, energy, config_type="Default", positions=None):
    numbers = np.asarray(numbers)
    return Configuration(
        atomic_numbers=numbers,
        positions=np.zeros((len(numbers), 3)) if positions is None else positions,
        properties={"energy": energy},
        config_type=config_type,
    )


def isolated(number):
    """An isolated atom carrying its reference energy."""
    return configuration(
        [number], HYDROGEN if number == 1 else OXYGEN, ISOLATED_ATOM_CONFIG_TYPE
    )


def unlabelled_isolated(number):
    """An isolated atom the file carried no energy for.

    Its own constructor rather than `isolated(number, energy=None)`: the
    absent energy is the subject of two tests, and a default argument that
    also means "give me the reference" would have them pass on the wrong one.
    """
    return configuration([number], None, ISOLATED_ATOM_CONFIG_TYPE)


#: Structures whose compositions differ, so the least-squares system has rank 2.
DATASET = [
    isolated(1),
    isolated(8),
    configuration([1, 1, 8], 2 * HYDROGEN + OXYGEN),
    configuration([1, 8], HYDROGEN + OXYGEN),
    configuration([1, 1, 8, 8], 2 * HYDROGEN + 2 * OXYGEN),
]


def zero_energy(configurations):
    return [0.0] * len(configurations)


# ---------------------------------------------------------------------------
# The five kinds
# ---------------------------------------------------------------------------


def test_a_table_is_taken_as_given():
    values, provenance = resolve_e0s(E0sTable(values={1: HYDROGEN, 8: OXYGEN}), Z_TABLE)
    assert values == {1: HYDROGEN, 8: OXYGEN}
    assert provenance.solver is None


def test_isolated_atoms_are_read_off_the_structures_marked_as_such():
    values, provenance = resolve_e0s(E0sIsolatedAtoms(), Z_TABLE, DATASET)
    assert values == {1: HYDROGEN, 8: OXYGEN}
    assert provenance.num_configurations == len(DATASET)


def test_a_least_squares_fit_recovers_the_references_it_was_built_from():
    """The energies are exactly the sum of the two, so the fit is exact."""
    values, provenance = resolve_e0s(E0sAverage(), Z_TABLE, DATASET)
    assert values[1] == pytest.approx(HYDROGEN)
    assert values[8] == pytest.approx(OXYGEN)
    assert provenance.solver == "least_squares"
    assert provenance.rank == 2


def test_a_foundation_table_is_copied():
    values, provenance = resolve_e0s(
        E0sFromFoundation(),
        Z_TABLE,
        foundation_e0s={1: HYDROGEN, 8: OXYGEN},
        foundation_model="medium",
    )
    assert values == {1: HYDROGEN, 8: OXYGEN}
    assert provenance.foundation_model == "medium"


def test_an_estimated_fit_corrects_against_the_foundation_model():
    """Against a model predicting zero it is the plain fit, which is what says
    the correction is the only difference between the two kinds."""
    corrected, _ = resolve_e0s(
        E0sEstimated(), Z_TABLE, DATASET, predict_energy=zero_energy
    )
    plain, _ = resolve_e0s(E0sAverage(), Z_TABLE, DATASET)
    assert corrected == pytest.approx(plain)


def test_the_correction_moves_the_answer():
    """The guard against the previous test passing for the wrong reason."""

    def predicts_one_eV_low(configurations):
        return [-1.0] * len(configurations)

    corrected, _ = resolve_e0s(
        E0sEstimated(), Z_TABLE, DATASET, predict_energy=predicts_one_eV_low
    )
    plain, _ = resolve_e0s(E0sAverage(), Z_TABLE, DATASET)
    assert corrected != pytest.approx(plain)


# ---------------------------------------------------------------------------
# The four refusals
# ---------------------------------------------------------------------------


def test_an_isolated_atom_with_no_energy_is_refused():
    """Legacy warns and contributes 0.0 (`mace/data/utils.py:333-337`)."""
    data = [unlabelled_isolated(1), isolated(8)]
    with pytest.raises(E0ResolutionError, match="carry no energy"):
        resolve_e0s(E0sIsolatedAtoms(), Z_TABLE, data)


def test_a_singular_fit_is_refused_rather_than_zeroed():
    """Legacy logs an error and zeroes every element (`:381-387`).

    Both structures here have one hydrogen per oxygen, so the composition
    cannot separate the two energies and the system has rank 1.
    """
    data = [
        configuration([1, 8], HYDROGEN + OXYGEN),
        configuration([1, 1, 8, 8], 2 * (HYDROGEN + OXYGEN)),
    ]
    with pytest.raises(E0ResolutionError, match="singular"):
        resolve_e0s(E0sAverage(), Z_TABLE, data)


def test_an_element_the_foundation_model_does_not_cover_is_refused():
    """Legacy pads it with `head_energies.get(z, 0.0)` (`run_train.py:578-586`)."""
    with pytest.raises(E0ResolutionError, match="covers"):
        resolve_e0s(E0sFromFoundation(), Z_TABLE, foundation_e0s={1: HYDROGEN})


def test_a_table_that_does_not_cover_the_model_is_refused():
    with pytest.raises(E0ResolutionError, match=r"\[8\]"):
        resolve_e0s(E0sTable(values={1: HYDROGEN}), Z_TABLE)


def test_every_refusal_names_the_elements_and_what_legacy_does():
    """The message is the whole value of refusing rather than defaulting."""
    with pytest.raises(E0ResolutionError) as caught:
        resolve_e0s(E0sFromFoundation(), Z_TABLE, foundation_e0s={1: HYDROGEN})
    message = str(caught.value)
    assert "8" in message
    assert "0.0" in message


# ---------------------------------------------------------------------------
# The policies that have to be asked for
# ---------------------------------------------------------------------------


def test_the_zero_padding_can_be_asked_for_and_is_recorded():
    data = [unlabelled_isolated(1), isolated(8)]
    values, provenance = resolve_e0s(
        E0sIsolatedAtoms(on_missing_energy="zero"), Z_TABLE, data
    )
    assert values[1] == 0.0
    assert provenance.missing_filled == (1,)


def test_an_uncovered_element_is_fitted_with_the_covered_ones_held_fixed():
    """The restricted least squares: each structure's energy less its covered
    atoms' energies is a sum over its uncovered atoms, solved for those."""
    data = [
        configuration([1, 8], HYDROGEN + OXYGEN),
        configuration([1, 1, 8], 2 * HYDROGEN + OXYGEN),
        configuration([8, 8], 2 * OXYGEN),
    ]
    values, provenance = resolve_e0s(
        E0sFromFoundation(missing="average"),
        Z_TABLE,
        data,
        foundation_e0s={1: HYDROGEN + 0.25},
    )
    assert values[1] == HYDROGEN + 0.25
    residuals = [
        (HYDROGEN + OXYGEN) - (HYDROGEN + 0.25),
        (2 * HYDROGEN + OXYGEN) - 2 * (HYDROGEN + 0.25),
        2 * OXYGEN,
    ]
    counts = np.array([[1.0], [1.0], [2.0]])
    expected = np.linalg.lstsq(counts, np.array(residuals), rcond=None)[0][0]
    assert values[8] == pytest.approx(expected, abs=1e-12)
    assert provenance.missing_filled == (8,)


def test_fitting_an_element_no_structure_holds_is_refused():
    data = [configuration([1, 1], 2 * HYDROGEN)]
    with pytest.raises(E0ResolutionError, match="rank 0"):
        resolve_e0s(
            E0sFromFoundation(missing="average"),
            Z_TABLE,
            data,
            foundation_e0s={1: HYDROGEN},
        )


def test_the_zero_padding_is_allowed_where_nothing_trains_against_it():
    data = [configuration([1, 1], 2 * HYDROGEN)]
    values, provenance = resolve_e0s(
        E0sFromFoundation(missing="zero"), Z_TABLE, data, foundation_e0s={1: HYDROGEN}
    )
    assert values[8] == 0.0
    assert provenance.missing_filled == (8,)


def test_the_zero_padding_of_an_element_the_data_holds_is_refused():
    """A head would train against a reference energy of zero for oxygen."""
    data = [configuration([1, 8], HYDROGEN + OXYGEN)]
    with pytest.raises(E0ResolutionError, match=r"pads \[8\]"):
        resolve_e0s(
            E0sFromFoundation(missing="zero"),
            Z_TABLE,
            data,
            foundation_e0s={1: HYDROGEN},
        )


# ---------------------------------------------------------------------------
# What is missing rather than wrong
# ---------------------------------------------------------------------------


def test_a_kind_that_reads_data_says_so_when_it_gets_none():
    with pytest.raises(E0ResolutionError, match="training structures"):
        resolve_e0s(E0sAverage(), Z_TABLE)


def test_estimated_says_it_needs_a_model():
    with pytest.raises(E0ResolutionError, match="foundation model"):
        resolve_e0s(E0sEstimated(), Z_TABLE, DATASET)


def test_a_prediction_of_the_wrong_length_is_refused():
    """The correction is per structure, so a mismatch is not recoverable."""
    with pytest.raises(E0ResolutionError, match="line up"):
        resolve_e0s(
            E0sEstimated(), Z_TABLE, DATASET, predict_energy=lambda cs: [0.0, 0.0]
        )


def test_a_multi_atom_structure_marked_isolated_is_refused():
    """Reading an energy off it would attribute a whole structure to one species."""
    data = [configuration([1, 1], -27.0, ISOLATED_ATOM_CONFIG_TYPE)]
    with pytest.raises(E0ResolutionError, match="one atom"):
        resolve_e0s(E0sIsolatedAtoms(), Z_TABLE, data)


def test_a_fit_with_no_labelled_energy_says_that_rather_than_fitting_nothing():
    data = [configuration([1, 8], None)]
    with pytest.raises(E0ResolutionError, match="carries one"):
        resolve_e0s(E0sAverage(), Z_TABLE, data)


# ---------------------------------------------------------------------------
# Provenance
# ---------------------------------------------------------------------------


def test_the_fingerprint_follows_the_data_the_fit_reads():
    """Two datasets that fit differently have to be distinguishable."""
    other = [*DATASET[:-1], configuration([1, 1, 8, 8], -4000.0)]
    _, first = resolve_e0s(E0sAverage(), Z_TABLE, DATASET)
    _, second = resolve_e0s(E0sAverage(), Z_TABLE, other)
    assert first.dataset_fingerprint != second.dataset_fingerprint


def test_the_fingerprint_does_not_follow_geometry():
    """Two datasets differing only in positions give the same E0s.

    A fingerprint that moved with them would report a change that did not
    happen, and the number it identifies is the fit's, not the dataset's.
    """
    moved = [
        configuration(
            c.atomic_numbers,
            c.properties["energy"],
            c.config_type,
            positions=np.full((len(c.atomic_numbers), 3), 7.0),
        )
        for c in DATASET
    ]
    _, first = resolve_e0s(E0sAverage(), Z_TABLE, DATASET)
    _, second = resolve_e0s(E0sAverage(), Z_TABLE, moved)
    assert first.dataset_fingerprint == second.dataset_fingerprint
