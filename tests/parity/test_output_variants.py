"""The readout and input-feature settings, each against the frozen tree.

MACE-OMOL uses all of them at once: the total charge and spin embedded beside
the elements, an energy read out of that embedding and added beside the
isolated-atom energies, a readout of the last layer only, and the biased
readout there. Each is built here from the anchor's recipe, alone and
together, converted through the production extraction and held to the live
legacy model.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch
from e3nn import o3
from mace import modules
from mace_core.observables import load_default_catalogue
from mace_torch.deploy.neutral_io import import_neutral

from tests.golden.build_mace_anchor import (
    ANCHOR_CONFIG,
    ATOMIC_ENERGIES,
    ATOMIC_NUMBERS,
    AVG_NUM_NEIGHBORS,
    SEED,
)
from tests.parity.test_fm_convert_legacy import extract_here

CATALOGUE = load_default_catalogue()

CHARGE = {
    "type": "categorical",
    "per": "graph",
    "in_dim": 1,
    "emb_dim": 6,
    "num_classes": 11,
    "offset": 5,
}
SPIN = {
    "type": "categorical",
    "per": "graph",
    "in_dim": 1,
    "emb_dim": 4,
    "num_classes": 7,
    "offset": 0,
}
CONTINUOUS_CHARGE = {"type": "continuous", "per": "graph", "in_dim": 1, "emb_dim": 5}

#: Each setting as the frozen tree's keyword arguments.
SETTINGS = {
    "graph-features": {"embedding_specs": {"total_charge": CHARGE, "total_spin": SPIN}},
    "continuous-feature": {"embedding_specs": {"total_charge": CONTINUOUS_CHARGE}},
    "embedding-readout": {
        "embedding_specs": {"total_charge": CHARGE},
        "use_embedding_readout": True,
    },
    "last-readout-only": {"use_last_readout_only": True},
    "biased-readout": {"readout_cls": modules.NonLinearBiasReadoutBlock},
    "all-of-mace-omol": {
        "embedding_specs": {"total_spin": SPIN, "total_charge": CHARGE},
        "use_embedding_readout": True,
        "use_last_readout_only": True,
        "readout_cls": modules.NonLinearBiasReadoutBlock,
    },
}

#: Varied, so that an input read from the wrong structure shows.
CHARGES = (-2, 0, 1, 3)
SPINS = (1, 2, 3, 1)


def legacy_model(scale_shift: bool, **settings) -> torch.nn.Module:
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    common = dict(
        r_max=ANCHOR_CONFIG["r_max"],
        num_bessel=ANCHOR_CONFIG["num_bessel"],
        num_polynomial_cutoff=ANCHOR_CONFIG["num_polynomial_cutoff"],
        max_ell=ANCHOR_CONFIG["max_ell"],
        interaction_cls=modules.interaction_classes[ANCHOR_CONFIG["interaction_cls"]],
        interaction_cls_first=modules.interaction_classes[
            ANCHOR_CONFIG["interaction_cls_first"]
        ],
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
        **settings,
    )
    if scale_shift:
        model = modules.ScaleShiftMACE(
            atomic_inter_scale=1.7, atomic_inter_shift=0.3, **common
        )
    else:
        model = modules.MACE(**common)
    model = model.to(torch.float64)
    # The frozen tree starts every bias at zero, where a readout that dropped
    # one would compute the same.
    generator = torch.Generator().manual_seed(SEED)
    with torch.no_grad():
        for name, parameter in model.named_parameters():
            if name.endswith(".bias") and parameter.numel():
                parameter.copy_(
                    torch.randn(parameter.shape, generator=generator, dtype=torch.float64)
                )
    return model


def structures():
    from tests.golden.anchors import load_training_structures

    found = load_training_structures()
    for index, atoms in enumerate(found):
        atoms.info["total_charge"] = CHARGES[index % len(CHARGES)]
        atoms.info["total_spin"] = SPINS[index % len(SPINS)]
    return found


def legacy_energies(legacy, found):
    from tests.golden.anchors import anchor_batch

    return legacy(
        anchor_batch(legacy, found, torch.float64).to_dict(),
        training=False,
        compute_force=False,
    )["energy"].detach()


def v1_energies(engine, legacy, found):
    from tests.golden.anchors import anchor_batch

    batch = anchor_batch(legacy, found, torch.float64)
    numbers = [int(z) for z in legacy.atomic_numbers.tolist()]
    graph = {
        "positions": batch.positions,
        "atomic_numbers": torch.tensor(
            [numbers[index] for index in batch.node_attrs.argmax(1).tolist()]
        ),
        "element_index": batch.node_attrs.argmax(1),
        "edge_index": batch.edge_index,
        "shifts": batch.shifts,
        "unit_shifts": batch.unit_shifts,
        "cell": batch.cell.reshape(-1, 3, 3),
        "batch": batch.batch,
        "num_graphs": int(batch.num_graphs),
        "head": torch.zeros(int(batch.num_graphs), dtype=torch.long),
        "total_charge": batch.total_charge,
        "total_spin": batch.total_spin,
    }
    return engine(graph, compute=("forces",)).total_energy.detach()


def converted(legacy, tmp_path):
    path = tmp_path / "legacy.model"
    torch.save(legacy, path)
    return import_neutral(extract_here(path, tmp_path / "legacy"), CATALOGUE)


@pytest.mark.parametrize("scale_shift", [False, True], ids=["plain", "scale-shift"])
@pytest.mark.parametrize("setting", sorted(SETTINGS))
def test_each_setting_matches_the_live_legacy_model(
    setting, scale_shift, fp64, isolated, tmp_path
):
    legacy = legacy_model(scale_shift, **SETTINGS[setting])
    found = structures()
    got = v1_energies(converted(legacy, tmp_path).engine, legacy, found)
    expected = legacy_energies(legacy, found)
    assert torch.allclose(got, expected, atol=1e-12, rtol=0), (
        f"largest difference {float((got - expected).abs().max()):.3g} eV"
    )


@pytest.mark.parametrize("setting", sorted(SETTINGS))
def test_each_setting_changes_the_model(setting, fp64, isolated):
    found = structures()
    plain = legacy_energies(legacy_model(True), found)
    changed = legacy_energies(legacy_model(True, **SETTINGS[setting]), found)
    assert not torch.allclose(plain, changed, atol=1e-6)


def test_the_embedded_inputs_are_read_per_structure(fp64, isolated, tmp_path):
    """The same structures with their charges permuted compute different
    energies, and the conversion follows them."""
    legacy = legacy_model(True, **SETTINGS["all-of-mace-omol"])
    engine = converted(legacy, tmp_path).engine
    found = structures()
    before = v1_energies(engine, legacy, found)
    for atoms in found:
        atoms.info["total_charge"] = -atoms.info["total_charge"]
    after = v1_energies(engine, legacy, found)
    assert not torch.allclose(before, after, atol=1e-6)
    assert torch.allclose(after, legacy_energies(legacy, found), atol=1e-12, rtol=0)


def test_a_checkpoint_of_it_reads_back_as_the_same_model(fp64, isolated, tmp_path):
    """The embedded inputs are concatenated in their declared order, and a
    checkpoint's configuration is written with its keys sorted: the order has
    to survive that, or the projection's columns land on the wrong inputs."""
    from mace_torch.finetune.foundation import read_foundation
    from mace_torch.train import write_model

    legacy = legacy_model(True, **SETTINGS["all-of-mace-omol"])
    imported = converted(legacy, tmp_path)
    path = write_model(tmp_path / "v1", imported.engine, imported.metadata)
    read_back = read_foundation(path, CATALOGUE)
    assert [feature.name for feature in read_back.config.model.graph_features] == [
        "total_spin",
        "total_charge",
    ]
    found = structures()
    assert torch.allclose(
        v1_energies(read_back.engine, legacy, found),
        legacy_energies(legacy, found),
        atol=1e-12,
        rtol=0,
    )
