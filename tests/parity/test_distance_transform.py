"""The distance transforms, against the frozen tree that defines them.

Thirteen published models evaluate their radial basis on the Agnesi transform
of each length rather than on the length itself, the MACE-MP-0b2 and later
generations and the multi-head ones among them. The transform scales each
pair by its covalent radii, and those are the model's own: the frozen tree
copies them from ASE when it builds the model, the published models checked
hold 0.2 for elements 97 onward, and ASE 3.29 has 2.0 there. So they are carried
with the weights and never looked up again. The soft transform scales by the
same radii, and no published model uses it.
"""

from __future__ import annotations

import pytest
import torch
from mace_torch.nn.radial import AgnesiTransform, SoftTransform

from tests.golden.anchors import load_training_structures
from tests.golden.build_mace_anchor import ATOMIC_NUMBERS
from tests.parity.test_anchor_as_foundation import energies
from tests.parity.test_interaction_variants import (
    DENSITY,
    DENSITY_RESIDUAL,
    RESIDUAL,
    converted,
    legacy_energies,
    legacy_model,
)

#: ``(first, later)``: the standard pairing, and MACE-MH-0's.
PAIRINGS = {"residual": (RESIDUAL, RESIDUAL), "density": (DENSITY, DENSITY_RESIDUAL)}


def assert_matches(legacy, tmp_path):
    """The converted model, once it has been held to the live one."""
    structures = load_training_structures()
    imported = converted(legacy, tmp_path)
    got = energies(imported.engine, legacy, structures)
    expected = legacy_energies(legacy, structures)
    assert torch.allclose(got, expected, atol=1e-12, rtol=0), (
        f"largest difference {float((got - expected).abs().max()):.3g} eV"
    )
    return imported


@pytest.mark.parametrize("pairing", sorted(PAIRINGS))
@pytest.mark.parametrize("transform", ["Agnesi", "Soft"])
def test_each_transform_matches_the_live_legacy_model(
    transform, pairing, fp64, isolated, tmp_path
):
    imported = assert_matches(
        legacy_model(*PAIRINGS[pairing], distance_transform=transform), tmp_path
    )
    built = imported.engine.backbone.backbone.distance_transform
    assert isinstance(built, {"Agnesi": AgnesiTransform, "Soft": SoftTransform}[transform])


def test_the_radii_are_the_model_s_own(fp64, isolated, tmp_path):
    """Radii other than the ones this ASE has, for exactly the elements the
    structures contain: a converter that rebuilt them from ASE would move the
    energy, and this one does not."""
    legacy = legacy_model(RESIDUAL, RESIDUAL, distance_transform="Agnesi")
    radii = legacy.radial_embedding.distance_transform.covalent_radii
    with torch.no_grad():
        radii[list(ATOMIC_NUMBERS)] *= 1.3
    imported = assert_matches(legacy, tmp_path)
    transform = imported.engine.backbone.backbone.distance_transform
    assert isinstance(transform, AgnesiTransform)
    assert torch.equal(transform.covalent_radii, radii)


def test_the_radii_move_the_energy(fp64, isolated):
    """What makes the check above worth having."""
    structures = load_training_structures()
    legacy = legacy_model(RESIDUAL, RESIDUAL, distance_transform="Agnesi")
    before = legacy_energies(legacy, structures)
    with torch.no_grad():
        radii = legacy.radial_embedding.distance_transform.covalent_radii
        radii[list(ATOMIC_NUMBERS)] *= 1.3
    assert not torch.allclose(before, legacy_energies(legacy, structures), atol=1e-6)
