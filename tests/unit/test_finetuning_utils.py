"""Unit tests for mace/tools/finetuning_utils.py.

Builds two small ScaleShiftMACE models in-process (no network, no
pretrained checkpoints): a "foundation" over elements [1, 6, 8] and a
target over the subset [1, 8]. Checks that load_foundations_elements
transfers weights (element-subselected embeddings, scale/shift,
avg_num_neighbors) and leaves a usable model, and that load_foundations
does a shape-matched state-dict copy that skips readouts by default.
"""

import math

import numpy as np
import pytest
import torch
import torch.nn.functional
from e3nn import o3

from mace import data, modules, tools
from mace.tools import torch_geometric
from mace.tools.finetuning_utils import load_foundations, load_foundations_elements

torch.set_default_dtype(torch.float64)

FOUNDATION_ZS = [1, 6, 8]
TARGET_ZS = [1, 8]
R_MAX = 5.0
MAX_L = 1  # matches hidden_irreps "16x0e + 16x1o"


def build_scale_shift_mace(
    zs, seed, r_max=R_MAX, scale=1.0, shift=0.0, avg_num_neighbors=None
):
    torch.manual_seed(seed)
    table = tools.AtomicNumberTable(zs)
    model_config = dict(
        r_max=r_max,
        num_bessel=8,
        num_polynomial_cutoff=6,
        max_ell=3,
        interaction_cls=modules.interaction_classes[
            "RealAgnosticResidualInteractionBlock"
        ],
        interaction_cls_first=modules.interaction_classes[
            "RealAgnosticResidualInteractionBlock"
        ],
        num_interactions=2,
        num_elements=len(zs),
        hidden_irreps=o3.Irreps("16x0e + 16x1o"),
        MLP_irreps=o3.Irreps("16x0e"),
        gate=torch.nn.functional.silu,
        atomic_energies=np.zeros(len(zs), dtype=float),
        # distinct per model by default, so the transfer is observable
        avg_num_neighbors=(
            avg_num_neighbors if avg_num_neighbors is not None else 3.0 + seed
        ),
        atomic_numbers=table.zs,
        correlation=3,
        radial_type="bessel",
        atomic_inter_scale=scale,
        atomic_inter_shift=shift,
        use_reduced_cg=False,
    )
    return modules.ScaleShiftMACE(**model_config), table


def water_batch(table, cutoff):
    config = data.Configuration(
        atomic_numbers=np.array([8, 1, 1]),
        positions=np.array(
            [
                [0.0, -2.0, 0.0],
                [1.0, 0.0, 0.0],
                [0.0, 1.0, 0.0],
            ]
        ),
        properties={"energy": -1.5, "forces": np.zeros((3, 3))},
        property_weights={"energy": 1.0, "forces": 1.0},
    )
    atomic_data = data.AtomicData.from_config(config, z_table=table, cutoff=cutoff)
    loader = torch_geometric.dataloader.DataLoader(
        dataset=[atomic_data], batch_size=1, shuffle=False, drop_last=False
    )
    return next(iter(loader))


@pytest.fixture(scope="module", name="loaded_pair")
def fixture_loaded_pair():
    """Foundation ([1,6,8]) + target ([1,8]) after load_foundations_elements."""
    foundation, _ = build_scale_shift_mace(FOUNDATION_ZS, seed=1, scale=0.5, shift=0.2)
    target, target_table = build_scale_shift_mace(TARGET_ZS, seed=2)
    target = load_foundations_elements(
        target,
        foundation,
        table=target_table,
        load_readout=True,
        use_shift=True,
        use_scale=True,
        max_L=MAX_L,
    )
    return foundation, target, target_table


def test_load_foundations_elements_node_embedding_rows(loaded_pair):
    foundation, target, _ = loaded_pair
    n_found = len(FOUNDATION_ZS)
    n_target = len(TARGET_ZS)
    fw = foundation.node_embedding.linear.weight.view(n_found, -1)
    tw = target.node_embedding.linear.weight.view(n_target, -1)
    # target zs [1, 8] map to foundation rows [0, 2] (foundation zs [1, 6, 8]),
    # rescaled by 1/sqrt(n_found / n_target)
    expected = fw[[0, 2], :] / math.sqrt(n_found / n_target)
    assert torch.allclose(tw, expected)
    # rows are genuinely element-selected: row for O differs from row for H
    assert not torch.allclose(tw[0], tw[1])


def test_load_foundations_elements_scale_shift_and_neighbors(loaded_pair):
    foundation, target, _ = loaded_pair
    assert torch.allclose(target.scale_shift.scale, foundation.scale_shift.scale)
    assert torch.allclose(target.scale_shift.shift, foundation.scale_shift.shift)
    for i in range(int(target.num_interactions)):
        assert (
            target.interactions[i].avg_num_neighbors
            == foundation.interactions[i].avg_num_neighbors
        )


def test_load_foundations_elements_hyperparams_coherent(loaded_pair):
    foundation, target, _ = loaded_pair
    assert float(target.r_max) == float(foundation.r_max)
    assert int(target.num_interactions) == int(foundation.num_interactions)
    # element tables stay those of the target subset
    assert target.atomic_numbers.tolist() == TARGET_ZS
    assert foundation.atomic_numbers.tolist() == FOUNDATION_ZS
    # shared-shape weights were copied verbatim
    assert torch.allclose(
        target.interactions[0].linear_up.weight,
        foundation.interactions[0].linear_up.weight,
    )
    assert torch.allclose(
        target.radial_embedding.bessel_fn.bessel_weights,
        foundation.radial_embedding.bessel_fn.bessel_weights,
    )


def test_load_foundations_elements_model_is_usable(loaded_pair):
    _, target, target_table = loaded_pair
    batch = water_batch(target_table, cutoff=R_MAX)
    out = target(batch.to_dict(), training=False, compute_force=True)
    assert out["energy"].shape == (1,)
    assert torch.isfinite(out["energy"]).all()
    assert out["forces"].shape == (3, 3)
    assert torch.isfinite(out["forces"]).all()


def test_load_foundations_elements_r_max_mismatch_raises():
    foundation, _ = build_scale_shift_mace(FOUNDATION_ZS, seed=1)
    target, target_table = build_scale_shift_mace(TARGET_ZS, seed=2, r_max=4.0)
    with pytest.raises(AssertionError):
        load_foundations_elements(target, foundation, table=target_table, max_L=MAX_L)


def test_load_foundations_copies_matching_shapes_and_skips_readouts():
    foundation, _ = build_scale_shift_mace(TARGET_ZS, seed=3)
    target, _ = build_scale_shift_mace(TARGET_ZS, seed=4)
    # models differ before loading
    assert not torch.allclose(
        target.node_embedding.linear.weight, foundation.node_embedding.linear.weight
    )
    readout_before = target.readouts[0].linear.weight.clone()
    target = load_foundations(target, foundation)  # include_readouts=False
    assert torch.allclose(
        target.node_embedding.linear.weight, foundation.node_embedding.linear.weight
    )
    assert torch.allclose(
        target.interactions[0].skip_tp.weight, foundation.interactions[0].skip_tp.weight
    )
    # readouts untouched by default
    assert torch.allclose(target.readouts[0].linear.weight, readout_before)
    assert not torch.allclose(
        target.readouts[0].linear.weight, foundation.readouts[0].linear.weight
    )


def test_load_foundations_include_readouts():
    foundation, _ = build_scale_shift_mace(TARGET_ZS, seed=5, avg_num_neighbors=3.0)
    target, target_table = build_scale_shift_mace(
        TARGET_ZS, seed=6, avg_num_neighbors=9.0
    )
    target = load_foundations(target, foundation, include_readouts=True)
    assert torch.allclose(
        target.readouts[0].linear.weight, foundation.readouts[0].linear.weight
    )
    for target_interaction, foundation_interaction in zip(
        target.interactions, foundation.interactions
    ):
        assert torch.equal(
            target_interaction.avg_num_neighbors,
            foundation_interaction.avg_num_neighbors,
        )
    # identical architecture + full copy => identical predictions
    batch = water_batch(target_table, cutoff=R_MAX)
    out_t = target(batch.to_dict(), training=False, compute_force=False)
    out_f = foundation(batch.to_dict(), training=False, compute_force=False)
    assert torch.allclose(out_t["energy"], out_f["energy"])


def test_avg_num_neighbors_is_checkpointed_and_cast():
    model, _ = build_scale_shift_mace(TARGET_ZS, seed=7, avg_num_neighbors=4.25)

    # Conversion utilities may copy this value from legacy models as a float.
    model.interactions[0].set_avg_num_neighbors(6.5)
    state_dict = model.state_dict()
    for index in range(int(model.num_interactions)):
        key = f"interactions.{index}.avg_num_neighbors"
        assert key in state_dict
        expected = 6.5 if index == 0 else 4.25
        assert state_dict[key].item() == pytest.approx(expected)

    model.float()
    for interaction in model.interactions:
        assert interaction.avg_num_neighbors.dtype == torch.float32


def test_legacy_state_dict_without_avg_num_neighbors_loads_strictly():
    source, _ = build_scale_shift_mace(TARGET_ZS, seed=8, avg_num_neighbors=3.0)
    legacy_state_dict = {
        key: value
        for key, value in source.state_dict().items()
        if not key.endswith("avg_num_neighbors")
    }
    target, _ = build_scale_shift_mace(TARGET_ZS, seed=9, avg_num_neighbors=7.0)

    incompatible = target.load_state_dict(legacy_state_dict, strict=True)

    assert incompatible.missing_keys == []
    assert incompatible.unexpected_keys == []
    for interaction in target.interactions:
        assert interaction.avg_num_neighbors.item() == pytest.approx(7.0)


# --- fine-tuning with joint embeddings (embedding_specs) -----------------------
#
# GenericJointEmbedding concatenates every spec's embedding in insertion order
# and feeds the result to project[0], so a spec's weight columns sit at the
# cumulative offset over ALL of that model's specs. The cases below pin down
# what load_foundations_elements must do when the two models' specs differ.


def spec_continuous(emb_dim):
    return {"type": "continuous", "per": "atom", "in_dim": 1, "emb_dim": emb_dim}


def build_mace_with_embeddings(zs, seed, embedding_specs, num_channels=16):
    torch.manual_seed(seed)
    table = tools.AtomicNumberTable(zs)
    model = modules.ScaleShiftMACE(
        r_max=R_MAX,
        num_bessel=8,
        num_polynomial_cutoff=6,
        max_ell=3,
        interaction_cls=modules.interaction_classes[
            "RealAgnosticResidualInteractionBlock"
        ],
        interaction_cls_first=modules.interaction_classes[
            "RealAgnosticResidualInteractionBlock"
        ],
        num_interactions=2,
        num_elements=len(zs),
        hidden_irreps=o3.Irreps(f"{num_channels}x0e + {num_channels}x1o"),
        MLP_irreps=o3.Irreps("16x0e"),
        gate=torch.nn.functional.silu,
        atomic_energies=np.zeros(len(zs), dtype=float),
        avg_num_neighbors=3.0,
        atomic_numbers=table.zs,
        correlation=3,
        radial_type="bessel",
        atomic_inter_scale=1.0,
        atomic_inter_shift=0.0,
        use_reduced_cg=False,
        embedding_specs=embedding_specs,
    )
    return model, table


def paint_head_columns(model, marker_per_spec):
    """Set each spec's block of project[0].weight to a distinct marker value."""
    start = 0
    weight = model.joint_embedding.project[0].weight.data
    for name, spec in model.joint_embedding.specs.items():
        dim = spec["emb_dim"]
        weight[:, start : start + dim] = marker_per_spec[name]
        start += dim


def head_columns_for(model, name):
    """The project[0] columns actually belonging to `name` in `model`."""
    start = 0
    for spec_name, spec in model.joint_embedding.specs.items():
        dim = spec["emb_dim"]
        if spec_name == name:
            return model.joint_embedding.project[0].weight.data[:, start : start + dim]
        start += dim
    raise KeyError(name)


def test_joint_embedding_head_columns_follow_the_model_layout():
    """A non-matching spec before a matching one must not shift the copy.

    The foundation supplies "shared"; the target's own "extra" sits before it,
    so "shared" lives at columns 8:12 of the target head, not 0:4.
    """
    shared = spec_continuous(4)
    foundation, _ = build_mace_with_embeddings(
        FOUNDATION_ZS, seed=1, embedding_specs={"dropped": spec_continuous(8), "shared": shared}
    )
    target, target_table = build_mace_with_embeddings(
        TARGET_ZS, seed=2, embedding_specs={"extra": spec_continuous(8), "shared": shared}
    )
    paint_head_columns(foundation, {"dropped": 1.0, "shared": 2.0})

    load_foundations_elements(
        target, foundation, table=target_table, max_L=MAX_L, load_readout=False
    )

    shared_cols = head_columns_for(target, "shared")
    assert torch.allclose(shared_cols, torch.full_like(shared_cols, 2.0)), (
        "the shared embedding's head columns did not receive the foundation's "
        "weights for that same embedding"
    )
    # the target-only embedding keeps its fresh init, never the foundation's
    extra_cols = head_columns_for(target, "extra")
    assert not torch.any(extra_cols == 1.0)
    assert not torch.any(extra_cols == 2.0)


def test_joint_embedding_matching_spec_weights_are_copied():
    """Embedder parameters are copied for matching specs and re-initialised otherwise."""
    shared = spec_continuous(4)
    foundation, _ = build_mace_with_embeddings(
        FOUNDATION_ZS, seed=3, embedding_specs={"shared": shared, "dropped": spec_continuous(8)}
    )
    target, target_table = build_mace_with_embeddings(
        TARGET_ZS, seed=4, embedding_specs={"shared": shared, "extra": spec_continuous(8)}
    )
    foundation_shared = {
        name: param.detach().clone()
        for name, param in foundation.joint_embedding.embedders["shared"].named_parameters()
    }
    target_extra_before = {
        name: param.detach().clone()
        for name, param in target.joint_embedding.embedders["extra"].named_parameters()
    }

    load_foundations_elements(
        target, foundation, table=target_table, max_L=MAX_L, load_readout=False
    )

    for name, param in target.joint_embedding.embedders["shared"].named_parameters():
        assert torch.allclose(param.data, foundation_shared[name])
    # "extra" has no counterpart, so it is re-initialised rather than left alone
    for name, param in target.joint_embedding.embedders["extra"].named_parameters():
        assert not torch.allclose(param.data, target_extra_before[name])


def test_joint_embedding_survives_the_blanket_state_dict_copy():
    """The spec-by-spec head transfer must not be undone at the end.

    load_foundations_elements finishes with a shape-matched state-dict copy.
    Both heads here are [16, 12], so an unguarded copy would overwrite the
    carefully sliced columns with the foundation's raw column order.
    """
    shared = spec_continuous(4)
    foundation, _ = build_mace_with_embeddings(
        FOUNDATION_ZS,
        seed=9,
        embedding_specs={"dropped": spec_continuous(8), "shared": shared},
    )
    target, target_table = build_mace_with_embeddings(
        TARGET_ZS,
        seed=10,
        embedding_specs={"extra": spec_continuous(8), "shared": shared},
    )
    assert (
        foundation.joint_embedding.project[0].weight.shape
        == target.joint_embedding.project[0].weight.shape
    ), "this test is only meaningful while the two heads have equal shapes"
    paint_head_columns(foundation, {"dropped": 1.0, "shared": 2.0})

    load_foundations_elements(
        target, foundation, table=target_table, max_L=MAX_L, load_readout=False
    )

    # "extra" has no counterpart in the foundation, so it must keep its own init
    # rather than inherit whatever sat in those column positions ("dropped").
    extra_cols = head_columns_for(target, "extra")
    assert not torch.any(extra_cols == 1.0)


def test_joint_embedding_foundation_without_embeddings_initialises():
    """A foundation with no joint embedding leaves the target freshly initialised."""
    foundation, _ = build_scale_shift_mace(FOUNDATION_ZS, seed=7)
    assert not hasattr(foundation, "joint_embedding")
    target, target_table = build_mace_with_embeddings(
        TARGET_ZS, seed=8, embedding_specs={"shared": spec_continuous(4)}
    )

    load_foundations_elements(
        target, foundation, table=target_table, max_L=MAX_L, load_readout=False
    )

    assert torch.all(torch.isfinite(target.joint_embedding.project[0].weight.data))
