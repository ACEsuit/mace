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
        load_foundations_elements(
            target, foundation, table=target_table, max_L=MAX_L
        )


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
    # same avg_num_neighbors: load_foundations only copies the state dict, so
    # non-state attributes must already match for predictions to coincide
    foundation, _ = build_scale_shift_mace(TARGET_ZS, seed=5, avg_num_neighbors=3.0)
    target, target_table = build_scale_shift_mace(
        TARGET_ZS, seed=6, avg_num_neighbors=3.0
    )
    target = load_foundations(target, foundation, include_readouts=True)
    assert torch.allclose(
        target.readouts[0].linear.weight, foundation.readouts[0].linear.weight
    )
    # identical architecture + full copy => identical predictions
    batch = water_batch(target_table, cutoff=R_MAX)
    out_t = target(batch.to_dict(), training=False, compute_force=False)
    out_f = foundation(batch.to_dict(), training=False, compute_force=False)
    assert torch.allclose(out_t["energy"], out_f["energy"])
