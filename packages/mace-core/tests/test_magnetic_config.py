"""The magnetic model's settings, from a configuration and from the frozen flags.

The frozen tree takes ``--m_max`` as a list of strings and resolves it against
the element table when it builds the model: one dict literal, one number for
every element, or one number per element in the table's order. The eight cases
its suite pins are pinned here through the legacy translation and the build's
own resolution, with one deviation: no ``--m_max`` at all saturates every
element at one, where the frozen tree cannot build the model.
"""

from __future__ import annotations

from argparse import Namespace

import numpy as np
import pytest
from mace_core.config.legacy import from_namespace
from mace_core.config.model import MagneticConfig
from mace_core.config.resolved import ResolvedConfig

TABLE = [1, 6, 8, 26]


def saturation(tokens, table=TABLE) -> list[float]:
    values = from_namespace(Namespace(model="magnetic", m_max=tokens))
    config = ResolvedConfig.model_validate(values)
    return config.model.magnetic.saturation_for(table)


def test_a_dict_literal_saturates_what_it_lists_and_one_elsewhere():
    assert saturation(["{26: 1.8, 8: 0.5}"]) == [1.0, 1.0, 0.5, 1.8]


def test_a_list_of_numbers_is_one_per_element_in_the_table_s_order():
    assert saturation(["0.1", "0.2", "0.3", "0.4"]) == [0.1, 0.2, 0.3, 0.4]
    assert MagneticConfig(saturation=(0.1, 0.2, 0.3, 0.4)).saturation_for(TABLE) == [
        0.1,
        0.2,
        0.3,
        0.4,
    ]


def test_one_number_saturates_every_element():
    assert saturation(["1.5"]) == [1.5, 1.5, 1.5, 1.5]


def test_no_saturation_saturates_every_element_at_one():
    assert saturation(None) == [1.0, 1.0, 1.0, 1.0]


def test_a_list_of_the_wrong_length_is_refused_at_build():
    with pytest.raises(ValueError, match="expected 4"):
        saturation(["0.1", "0.2"])


def test_an_element_outside_the_table_is_ignored():
    assert saturation(["{26: 1.8, 99: 1.0}"]) == [1.0, 1.0, 1.0, 1.8]


def test_the_table_may_hold_numpy_integers():
    table = [np.int64(1), np.int64(6), np.int64(26)]
    assert saturation(["{26: 8.0}"], table) == [1.0, 1.0, 8.0]


def test_the_other_magnetic_flags_land_on_the_section():
    values = from_namespace(
        Namespace(
            max_m_ell=2,
            num_mag_radial_basis=6,
            num_mag_radial_basis_one_body=5,
            use_magmom_one_body=True,
            train_one_body_contribution=False,
            data_aug_magmom=True,
            data_aug_magmom_mode="soc",
        )
    )
    config = ResolvedConfig.model_validate(values)
    assert config.model.magnetic == MagneticConfig(
        lmax=2, num_basis=6, one_body_basis=5, one_body=True, train_one_body=False
    )
    (augmentation,) = config.data.augmentations
    assert augmentation.name == "magnetic_moments"
    assert augmentation.settings == {"mode": "soc"}


def test_no_augmentation_unless_asked():
    values = from_namespace(
        Namespace(data_aug_magmom=False, data_aug_magmom_mode="soc")
    )
    assert ResolvedConfig.model_validate(values).data.augmentations == ()
