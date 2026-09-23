"""Adapters and freezing: what trains, what the model still computes.

The adapter cases are the frozen tree's, from `tests/unit/test_lora.py`: only
the adapters train, the model stays symmetric, merging reproduces it, removes
every trace of the adapters, and lets everything train again. The freezing
cases are its `--freeze` thresholds, `mace/tools/scripts_utils.py:944-960`.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch
from conftest import fp64_only
from mace_core.config.resolved import ResolvedConfig
from mace_core.data.backend import DatasetStatistics
from mace_core.elements import AtomicNumberTable, ResolvedE0s
from mace_core.neighbors import get_neighborhood
from mace_core.observables import load_default_catalogue
from mace_torch.finetune.freeze import FREEZE_LEVELS, freeze, frozen_groups
from mace_torch.finetune.lora import inject_lora, merge_lora
from mace_torch.serialization import canonical_state
from mace_torch.train.model_stage import build_model
from mace_torch.train.optimizers import parameter_groups

CATALOGUE = load_default_catalogue()
WATER = np.array([[0.0, 0.0, 0.0], [0.96, 0.0, 0.0], [-0.24, 0.93, 0.0]])


def model(seed=0):
    config = ResolvedConfig.model_validate(
        {
            "runtime": {"seed": seed},
            "data": {"heads": {"a": {"train_file": "x.xyz"}}},
            "model": {
                "observables": ["energy", "forces"],
                "r_max": 3.0,
                "num_channels": 4,
                "max_ell": 1,
                "num_interactions": 2,
                "correlation": 2,
            },
        }
    )
    engine, _ = build_model(
        config,
        CATALOGUE,
        z_table=AtomicNumberTable([1, 8]),
        heads=("a",),
        e0s=ResolvedE0s({"a": {1: -13.6, 8: -2040.0}}),
        statistics=DatasetStatistics(avg_num_neighbors=2.0, std=1.0),
    )
    return engine, config


def graph(rotation=None):
    positions = WATER if rotation is None else WATER @ rotation.T
    neighborhood = get_neighborhood(positions, 3.0, (False, False, False), None)
    return {
        "positions": torch.tensor(positions),
        "atomic_numbers": torch.tensor([8, 1, 1]),
        "element_index": torch.tensor([1, 0, 0]),
        "edge_index": torch.tensor(neighborhood.edge_index),
        "shifts": torch.tensor(neighborhood.shifts),
        "batch": torch.zeros(3, dtype=torch.long),
        "num_graphs": 1,
        "head": torch.zeros(1, dtype=torch.long),
    }


def energy(engine, rotation=None):
    return engine(graph(rotation), compute=()).total_energy


def nudge_adapters(engine):
    """Move every adapter off its start, so the adapted model differs."""
    generator = torch.Generator().manual_seed(9)
    with torch.no_grad():
        for parameter in engine.parameters():
            if parameter.requires_grad:
                parameter.add_(
                    0.05
                    * torch.randn(
                        parameter.shape, generator=generator, dtype=parameter.dtype
                    )
                )


ROTATION = torch.tensor(
    [[0.36, 0.48, -0.8], [-0.8, 0.6, 0.0], [0.48, 0.64, 0.6]], dtype=torch.float64
).numpy()


# ---------------------------------------------------------------------------
# Adapters
# ---------------------------------------------------------------------------


@fp64_only
def test_an_adapter_starts_as_the_model_it_adapts():
    engine, _ = model()
    before = energy(engine)
    inject_lora(engine)
    assert torch.equal(energy(engine), before)


@fp64_only
def test_only_the_adapters_train():
    engine, _ = model()
    inject_lora(engine)
    trainable = [name for name, p in engine.named_parameters() if p.requires_grad]
    assert trainable
    assert all(".parametrizations." in name for name in trainable)
    assert not any(name.endswith(".original") for name in trainable)


@fp64_only
def test_only_the_adapters_receive_gradients():
    engine, _ = model()
    inject_lora(engine)
    nudge_adapters(engine)
    energy(engine).sum().backward()
    for name, parameter in engine.named_parameters():
        if not parameter.requires_grad:
            assert parameter.grad is None, name


@fp64_only
def test_every_linear_map_and_radial_network_is_adapted():
    engine, _ = model()
    adapted = inject_lora(engine)
    assert any(path.endswith("node_embedding.weight") for path in adapted)
    assert any(".radial.weights." in path for path in adapted)
    assert any(".readouts." in path for path in adapted)


@fp64_only
def test_an_adapted_model_stays_symmetric():
    """A low-rank update of each irrep's block is still a map of that shape.
    An update that mixed components would break this."""
    engine, _ = model()
    inject_lora(engine)
    nudge_adapters(engine)
    assert torch.allclose(energy(engine, ROTATION), energy(engine), atol=1e-10)


@fp64_only
def test_merging_reproduces_the_adapted_model():
    engine, _ = model()
    inject_lora(engine)
    nudge_adapters(engine)
    adapted = energy(engine).detach()
    merge_lora(engine)
    assert torch.allclose(energy(engine), adapted, rtol=0, atol=1e-12)


@fp64_only
def test_a_merged_model_has_the_shapes_of_one_never_adapted():
    """What makes the merged checkpoint load into any backend."""
    engine, _ = model()
    plain = {
        path: {name: value.shape for name, value in tensors.items()}
        for path, tensors in canonical_state(engine).items()
    }
    inject_lora(engine)
    nudge_adapters(engine)
    merge_lora(engine)
    merged = {
        path: {name: value.shape for name, value in tensors.items()}
        for path, tensors in canonical_state(engine).items()
    }
    assert merged == plain
    assert not any("parametrizations" in name for name, _ in engine.named_parameters())


@fp64_only
def test_a_merged_model_trains_again():
    engine, _ = model()
    inject_lora(engine)
    merge_lora(engine)
    assert all(parameter.requires_grad for parameter in engine.parameters())


@fp64_only
def test_a_merged_model_stays_symmetric():
    engine, _ = model()
    inject_lora(engine)
    nudge_adapters(engine)
    merge_lora(engine)
    assert torch.allclose(energy(engine, ROTATION), energy(engine), atol=1e-10)


@fp64_only
def test_evaluating_leaves_what_is_frozen_frozen():
    """The frozen tree's last case: an evaluation must not unfreeze."""
    engine, _ = model()
    inject_lora(engine)
    engine.eval()
    energy(engine)
    engine.train()
    trainable = [name for name, p in engine.named_parameters() if p.requires_grad]
    assert all(".parametrizations." in name for name in trainable)


# ---------------------------------------------------------------------------
# Freezing
# ---------------------------------------------------------------------------


def test_the_thresholds_are_the_frozen_trees():
    assert FREEZE_LEVELS == {
        "embedding": 1,
        "interactions": 5,
        "products": 6,
        "readouts": 7,
    }


@pytest.mark.parametrize(
    ("level", "groups"),
    [
        (None, ()),
        (0, ()),
        (1, ("embedding",)),
        (4, ("embedding",)),
        (5, ("embedding", "interactions")),
        (6, ("embedding", "interactions", "products")),
        (7, ("embedding", "interactions", "products", "readouts")),
        (99, ("embedding", "interactions", "products", "readouts")),
    ],
)
def test_a_level_freezes_the_groups_the_frozen_tree_freezes(level, groups):
    assert frozen_groups(level) == groups


@fp64_only
@pytest.mark.parametrize("level", [1, 5, 6, 7])
def test_a_frozen_group_takes_no_gradient_and_has_no_learning_rate(level):
    engine, config = model()
    factors = freeze(engine, level)
    assert factors == {group: 0.0 for group in frozen_groups(level)}
    from mace_torch.train.optimizers import GROUP_MARKERS

    for name, parameter in engine.named_parameters():
        frozen = any(GROUP_MARKERS[g] in f".{name}." for g in frozen_groups(level))
        assert parameter.requires_grad is not frozen, name
    training = config.training.model_copy(
        update={
            "scheduler": config.training.scheduler.model_copy(
                update={"group_factors": factors}
            )
        }
    )
    built = {group["name"] for group in parameter_groups(engine, training)}
    assert not built & set(frozen_groups(level))


@fp64_only
def test_a_soft_freeze_is_the_factors_alone():
    """The frozen tree's `test_run_train_soft_freeze`: `--lr_params_factors`
    without `--freeze`. Everything still trains, one group slowly."""
    engine, config = model()
    training = config.training.model_copy(
        update={
            "scheduler": config.training.scheduler.model_copy(
                update={"group_factors": {"embedding": 0.1}}
            )
        }
    )
    groups = {group["name"]: group for group in parameter_groups(engine, training)}
    assert groups["embedding"]["lr"] == pytest.approx(0.1 * training.lr)
    assert groups["readouts"]["lr"] == pytest.approx(training.lr)
    assert all(parameter.requires_grad for parameter in engine.parameters())


def test_a_negative_level_is_refused():
    """Legacy documents -1 as freezing the last layer and does nothing with
    it: every threshold is a `>=` on a positive number."""
    with pytest.raises(ValueError):
        ResolvedConfig.model_validate(
            {
                "data": {"heads": {"a": {"train_file": "x.xyz"}}},
                "finetune": {"freeze": -1},
            }
        )
