"""The interaction blocks, each against the frozen tree that defines it.

The committed anchors have one pairing, the plain first block and the residual
one after it. The published models have others: MACE-MP-0 takes its first
layer residual, as the frozen tree's default does, so its skip comes from the
element embedding and the product basis adds it; the multi-head models
normalize their messages by a density learned per atom rather than by the
average neighbour count. Each pairing is built here from the anchor's recipe,
converted through the production extraction, and held to the live legacy
model.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch
from e3nn import o3
from mace import modules
from mace_core.observables import load_default_catalogue
from mace_torch.deploy.neutral_io import import_neutral
from mace_torch.nn.interaction import InteractionBlock, ResidualInteractionBlock

from tests.golden.build_mace_anchor import (
    ANCHOR_CONFIG,
    ATOMIC_ENERGIES,
    ATOMIC_NUMBERS,
    AVG_NUM_NEIGHBORS,
    SEED,
)
from tests.parity.test_anchor_as_foundation import energies
from tests.parity.test_fm_convert_legacy import extract_here

CATALOGUE = load_default_catalogue()


PLAIN = "RealAgnosticInteractionBlock"
RESIDUAL = "RealAgnosticResidualInteractionBlock"
DENSITY = "RealAgnosticDensityInteractionBlock"
DENSITY_RESIDUAL = "RealAgnosticDensityResidualInteractionBlock"

#: ``(first, later)`` pairings, and which of the published models has each.
PAIRINGS = {
    "anchors": (PLAIN, RESIDUAL),
    "mace-mp-0": (RESIDUAL, RESIDUAL),
    "mace-mh-0": (DENSITY, DENSITY_RESIDUAL),
    "density-residual-first": (DENSITY_RESIDUAL, DENSITY_RESIDUAL),
    "density-later-only": (RESIDUAL, DENSITY_RESIDUAL),
}


def legacy_model(first: str, later: str = RESIDUAL) -> torch.nn.Module:
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    return modules.MACE(
        r_max=ANCHOR_CONFIG["r_max"],
        num_bessel=ANCHOR_CONFIG["num_bessel"],
        num_polynomial_cutoff=ANCHOR_CONFIG["num_polynomial_cutoff"],
        max_ell=ANCHOR_CONFIG["max_ell"],
        interaction_cls=modules.interaction_classes[later],
        interaction_cls_first=modules.interaction_classes[first],
        num_interactions=ANCHOR_CONFIG["num_interactions"],
        num_elements=ANCHOR_CONFIG["num_elements"],
        hidden_irreps=o3.Irreps(ANCHOR_CONFIG["hidden_irreps"]),
        MLP_irreps=o3.Irreps(ANCHOR_CONFIG["MLP_irreps"]),
        atomic_energies=np.array(ATOMIC_ENERGIES, dtype=float),
        avg_num_neighbors=AVG_NUM_NEIGHBORS,
        atomic_numbers=ATOMIC_NUMBERS,
        correlation=ANCHOR_CONFIG["correlation"],
        gate=modules.gate_dict[ANCHOR_CONFIG["gate"]],
        pair_repulsion=ANCHOR_CONFIG["pair_repulsion"],
        distance_transform=ANCHOR_CONFIG["distance_transform"],
        radial_type=ANCHOR_CONFIG["radial_type"],
        use_reduced_cg=ANCHOR_CONFIG["use_reduced_cg"],
    ).to(torch.float64)


def converted(legacy, tmp_path):
    path = tmp_path / "legacy.model"
    torch.save(legacy, path)
    return import_neutral(extract_here(path, tmp_path / "legacy"), CATALOGUE)


def legacy_energies(legacy, structures):
    from tests.golden.anchors import anchor_batch

    return legacy(
        anchor_batch(legacy, structures, torch.float64).to_dict(),
        training=False,
        compute_force=False,
    )["energy"].detach()


@pytest.mark.parametrize("pairing", sorted(PAIRINGS))
def test_each_pairing_matches_the_live_legacy_model(pairing, fp64, isolated, tmp_path):
    from tests.golden.anchors import load_training_structures

    first, later = PAIRINGS[pairing]
    legacy = legacy_model(first, later)
    structures = load_training_structures()
    got = energies(converted(legacy, tmp_path).engine, legacy, structures)
    expected = legacy_energies(legacy, structures)
    assert torch.allclose(got, expected, atol=1e-12, rtol=0), (
        f"largest difference {float((got - expected).abs().max()):.3g} eV"
    )


@pytest.mark.parametrize(
    ("first", "block", "density"),
    [
        (PLAIN, InteractionBlock, False),
        (RESIDUAL, ResidualInteractionBlock, False),
        (DENSITY, InteractionBlock, True),
        (DENSITY_RESIDUAL, ResidualInteractionBlock, True),
    ],
)
def test_each_first_block_is_built_as_named(first, block, density, fp64, isolated, tmp_path):
    imported = converted(legacy_model(first), tmp_path)
    built = imported.engine.backbone.backbone.interactions[0]
    assert isinstance(built, block)
    assert (built.body.density is not None) is density


def test_the_pairings_are_different_models(fp64, isolated):
    """Same seed, same everything else, different energies: none of these is
    a rename of another."""
    from tests.golden.anchors import load_training_structures

    structures = load_training_structures()
    seen = [legacy_energies(legacy_model(*pair), structures) for pair in PAIRINGS.values()]
    for index, first in enumerate(seen):
        for second in seen[index + 1 :]:
            assert not torch.allclose(first, second, atol=1e-6)
