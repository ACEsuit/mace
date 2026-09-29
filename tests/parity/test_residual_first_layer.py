"""A residual first layer, against the frozen tree that defines it.

The frozen tree's first interaction is ``RealAgnosticResidualInteractionBlock``
by default, and MACE-MP-0 was trained so: the first layer takes its skip from
the element embedding and the product basis adds it, as every later layer
does. The committed anchors have the plain first block, so this builds the
anchor's recipe with the residual one, converts it through the production
extraction, and holds it to the live legacy model.
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


def legacy_model(first: str) -> torch.nn.Module:
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    return modules.MACE(
        r_max=ANCHOR_CONFIG["r_max"],
        num_bessel=ANCHOR_CONFIG["num_bessel"],
        num_polynomial_cutoff=ANCHOR_CONFIG["num_polynomial_cutoff"],
        max_ell=ANCHOR_CONFIG["max_ell"],
        interaction_cls=modules.interaction_classes[ANCHOR_CONFIG["interaction_cls"]],
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


@pytest.fixture(name="residual")
def fixture_residual(fp64, isolated, tmp_path):
    legacy = legacy_model("RealAgnosticResidualInteractionBlock")
    path = tmp_path / "residual.model"
    torch.save(legacy, path)
    imported = import_neutral(extract_here(path, tmp_path / "residual"), CATALOGUE)
    return legacy, imported


def test_the_first_layer_is_built_residual(residual):
    _, imported = residual
    first = imported.engine.backbone.backbone.interactions[0]
    assert isinstance(first, ResidualInteractionBlock)


def test_a_residual_first_layer_matches_the_live_legacy_model(residual):
    from tests.golden.anchors import anchor_batch, load_training_structures

    legacy, imported = residual
    structures = load_training_structures()
    expected = legacy(
        anchor_batch(legacy, structures, torch.float64).to_dict(),
        training=False,
        compute_force=False,
    )["energy"].detach()
    got = energies(imported.engine, legacy, structures)
    assert torch.allclose(got, expected, atol=1e-12, rtol=0), (
        f"largest difference {float((got - expected).abs().max()):.3g} eV"
    )


def test_it_is_a_different_model_from_the_plain_first_layer(fp64, isolated):
    """The two spellings are two functions: same seed, same everything else,
    and the energies differ. Building one for the other is not a rename."""
    from tests.golden.anchors import anchor_batch, load_training_structures

    structures = load_training_structures()
    out = {}
    for first in ("RealAgnosticInteractionBlock", "RealAgnosticResidualInteractionBlock"):
        legacy = legacy_model(first)
        out[first] = legacy(
            anchor_batch(legacy, structures, torch.float64).to_dict(),
            training=False,
            compute_force=False,
        )["energy"].detach()
    assert not torch.allclose(*out.values(), atol=1e-6)


def test_the_plain_first_layer_is_still_built_plain(fp64, isolated, tmp_path):
    legacy = legacy_model("RealAgnosticInteractionBlock")
    path = tmp_path / "plain.model"
    torch.save(legacy, path)
    imported = import_neutral(extract_here(path, tmp_path / "plain"), CATALOGUE)
    first = imported.engine.backbone.backbone.interactions[0]
    assert isinstance(first, InteractionBlock)
