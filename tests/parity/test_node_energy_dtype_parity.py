"""Which energy families widen their per-atom energies, on both stacks.

The frozen tree's list is read off its source by the legacy characterization
suite, so a model class it adds has to be classified there before that suite
passes. This pins that the model factory's own list says the same for every
family both stacks have, so the two cannot drift apart silently.
"""

from __future__ import annotations

from mace_torch.models import build

from tests.unit.test_scale_shift_dtype import WIDENS_NODE_ENERGY

#: The frozen tree's class behind each v1 energy family.
LEGACY_CLASS = {
    "plain": "MACE",
    "scale_shift": "ScaleShiftMACE",
    "polar": "PolarMACE",
    "magnetic": "MagneticScaleShiftMACE",
    "les": "MACELES",
}


def test_every_energy_family_has_a_frozen_counterpart():
    assert set(build._WIDENS_NODE_ENERGY) == set(LEGACY_CLASS)


def test_the_model_factory_widens_exactly_where_the_frozen_tree_does():
    frozen = {family: WIDENS_NODE_ENERGY[name] for family, name in LEGACY_CLASS.items()}
    assert frozen == build._WIDENS_NODE_ENERGY
