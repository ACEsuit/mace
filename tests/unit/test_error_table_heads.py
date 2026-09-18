"""The final error table, per head, and `--skip_evaluate_heads`.

`create_error_table` is what a training prints at the end, and multihead is the
normal case, so a row per loader is the product rather than a detail. Nothing
covered it: the only test that reached this function at all compared a
single-head table's numbers against a recorded reference, which says nothing
about which rows appear.

`--skip_evaluate_heads` decides which of those rows to leave out, and it defaults
to `pt_head` -- so on every multiheads-finetuning run the flag is already doing
something, unasked, with no test behind it. Evaluating the replay head is what it
saves: the pretraining set is large, and its final metrics are not what the run
is for.
"""

import logging

import numpy as np
import pytest
import torch
from ase import Atoms
from e3nn import o3

from mace import data, modules, tools
from mace.data import KeySpecification
from mace.tools import torch_geometric
from mace.tools.tables_utils import create_error_table
from mace.tools.torch_tools import default_dtype

TABLE = tools.AtomicNumberTable([1, 8])


def _model():
    return modules.ScaleShiftMACE(
        r_max=4.0,
        num_bessel=4,
        num_polynomial_cutoff=5,
        max_ell=2,
        interaction_cls=modules.interaction_classes[
            "RealAgnosticResidualInteractionBlock"
        ],
        interaction_cls_first=modules.interaction_classes[
            "RealAgnosticInteractionBlock"
        ],
        num_interactions=2,
        num_elements=2,
        hidden_irreps=o3.Irreps("8x0e"),
        MLP_irreps=o3.Irreps("4x0e"),
        gate=torch.nn.functional.silu,
        atomic_energies=np.array([-1.0, -5.0]),
        avg_num_neighbors=4.0,
        atomic_numbers=TABLE.zs,
        correlation=2,
        atomic_inter_scale=1.0,
        atomic_inter_shift=0.0,
    ).double()


def _loader(seed):
    """One batch of two labelled waters, enough for a row of metrics."""
    rng = np.random.default_rng(seed)
    keyspec = KeySpecification.from_defaults()
    keyspec.info_keys["energy"] = "REF_energy"
    keyspec.arrays_keys["forces"] = "REF_forces"
    items = []
    for _ in range(2):
        atoms = Atoms(
            "H2O",
            positions=[[0, 0, 0], [0.95, 0, 0], [-0.24, 0.93, 0]],
            cell=[8, 8, 8],
            pbc=True,
        )
        atoms.info["REF_energy"] = float(rng.normal())
        atoms.arrays["REF_forces"] = rng.normal(size=(3, 3))
        config = data.utils.config_from_atoms(atoms, key_specification=keyspec)
        items.append(data.AtomicData.from_config(config, z_table=TABLE, cutoff=4.0))
    return torch_geometric.dataloader.DataLoader(
        dataset=items, batch_size=2, shuffle=False, drop_last=False
    )


def table_rows(skip_heads, names=("train_solid", "valid_solid", "train_pt_head")):
    """The `config_type` column of the table, which is the row's loader name."""
    with default_dtype(torch.float64):
        table = create_error_table(
            table_type="PerAtomRMSE",
            all_data_loaders={
                name: _loader(seed) for seed, name in enumerate(names)
            },
            model=_model(),
            loss_fn=modules.WeightedEnergyForcesLoss(energy_weight=1.0, forces_weight=1.0),
            output_args={
                "energy": True,
                "forces": True,
                "virials": False,
                "stress": False,
                "dipoles": False,
            },
            log_wandb=False,
            device="cpu",
            skip_heads=skip_heads,
        )
    return [row[0] for row in table.rows]


# ---------------------------------------------------------------------------
# a row per head
# ---------------------------------------------------------------------------


def test_every_loader_becomes_its_own_row():
    """Per-head reporting: the table is keyed on the loader name, so a multihead
    run's heads stay separable in the one place a user reads metrics."""
    assert table_rows(skip_heads=None) == [
        "train_pt_head",
        "train_solid",
        "valid_solid",
    ]


def test_no_skip_list_evaluates_everything():
    """`skip_heads=None` and `skip_heads=[]` are the same request, and neither is
    the CLI default -- `--skip_evaluate_heads` defaults to `pt_head`."""
    assert table_rows(skip_heads=[]) == table_rows(skip_heads=None)


# ---------------------------------------------------------------------------
# --skip_evaluate_heads
# ---------------------------------------------------------------------------


def test_a_skipped_head_loses_its_row():
    """The flag's default, `pt_head`, on a run that has one."""
    assert table_rows(skip_heads=["pt_head"]) == ["train_solid", "valid_solid"]


def test_skipping_names_a_head_not_a_split():
    """The name is matched against the whole loader name, so a head is skipped in
    every split it appears in rather than only at test time."""
    assert table_rows(skip_heads=["solid"]) == ["train_pt_head"]


def test_a_skip_list_naming_nothing_present_removes_nothing():
    assert table_rows(skip_heads=["absent"]) == table_rows(skip_heads=None)


def test_several_heads_can_be_skipped_at_once():
    """`--skip_evaluate_heads` is comma-separated, and `run_train` splits it on
    the comma before this function sees it."""
    assert table_rows(skip_heads="pt_head,solid".split(",")) == []


def test_the_match_is_a_substring_and_not_the_head_name():
    """Recorded, not endorsed. The loader names are `train_<head>` and
    `valid_<head>` for the training splits but `<head>_<config_type>` for the
    test sets, so the head sits at either end and a substring test is what
    handles both. The cost is a false positive: a head whose name merely
    *contains* another's is skipped along with it, and since the flag defaults to
    `pt_head`, a head called `expt_head` is dropped from the final table of every
    finetuning run without asking for it.
    """
    assert "pt_head" in "train_expt_head"
    assert table_rows(
        skip_heads=["pt_head"], names=("train_expt_head", "valid_expt_head")
    ) == []


def test_a_skipped_head_says_so_in_the_log(caplog):
    """It is the only trace: a row that is simply absent from a table reads like
    a head that was never trained."""
    with caplog.at_level(logging.INFO):
        table_rows(skip_heads=["pt_head"])

    assert any(
        "train_pt_head" in record.message and "skip_heads" in record.message
        for record in caplog.records
    ), [record.message for record in caplog.records]
