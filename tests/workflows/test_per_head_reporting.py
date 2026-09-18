"""What a multihead run reports per head, and what it leaves out.

Multihead is the normal case, so per-head reporting is the product: the
validation lines during training, the rows of the final error table, and the
`head` field on every row of the results log a plot is drawn from. None of it was
covered, and the three do not agree with each other -- the replay head is
validated at every eval epoch and then absent from the final table, because
`--skip_evaluate_heads` defaults to `pt_head`. That is a reasonable default and an
invisible one: a head missing from a table looks like a head that was never
trained.

The run is the session-scoped fine-tuning fixture, so these cost a file read.
"""

import json

import pytest


@pytest.fixture(name="artifacts")
def fixture_artifacts(finetuned_multihead_model):
    """The three things the run writes that report per head."""
    work = finetuned_multihead_model.parent
    log = work / "mh_run-11.log"
    results = work / "mh_run-11_train.txt"
    assert log.exists() and results.exists(), sorted(p.name for p in work.iterdir())
    rows = [
        json.loads(line)
        for line in results.read_text(encoding="utf-8").splitlines()
        if line.strip().startswith("{")
    ]
    return log.read_text(encoding="utf-8"), rows


HEADS = ("pt_head", "Default")


# ---------------------------------------------------------------------------
# during training
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("head", HEADS)
def test_every_head_is_validated_at_every_eval_epoch(artifacts, head):
    """`--eval_interval=1` over two epochs, plus the pass before any training."""
    _, rows = artifacts

    # In the order written: the pre-training pass logs no epoch number at all.
    epochs = [
        row["epoch"] for row in rows if row["mode"] == "eval" and row["head"] == head
    ]
    assert epochs == [None, 0, 1], epochs


@pytest.mark.parametrize("head", HEADS)
def test_the_validation_line_names_the_head(artifacts, head):
    """Two heads' losses are two numbers on consecutive lines, and the head name
    is the only thing distinguishing them."""
    log, _ = artifacts

    named = [line for line in log.splitlines() if f"head: {head}," in line]
    assert len(named) == 3, named


def test_the_replay_head_is_validated_before_the_new_one(artifacts):
    """Order, because `train()` keeps `valid_loss` from the *last* head it
    evaluated and that single number drives the LR scheduler, the patience
    counter and which epoch is kept. Evaluating `pt_head` first is what makes the
    new head the one those decisions follow; reversing the loader order would
    silently hand them to the replay set.
    """
    log, _ = artifacts

    order = [
        head
        for line in log.splitlines()
        for head in HEADS
        if f"head: {head}," in line
    ]
    assert order == ["pt_head", "Default"] * 3, order


def test_the_results_log_tags_every_row_with_its_head(artifacts):
    """The field the plots group on. Optimiser rows carry no head -- a training
    step is not per head -- and eval rows carry one for each."""
    _, rows = artifacts

    assert {row["head"] for row in rows if row["mode"] == "eval"} == set(HEADS)
    assert {row.get("head") for row in rows if row["mode"] == "opt"} == {None}


# ---------------------------------------------------------------------------
# the final table
# ---------------------------------------------------------------------------


def test_the_final_table_has_a_row_per_split_of_the_new_head(artifacts):
    log, _ = artifacts

    assert "train_Default" in log
    assert "valid_Default" in log


def test_the_replay_head_is_left_out_of_the_final_table(artifacts):
    """`--skip_evaluate_heads` defaults to `pt_head`, so this happens on every
    multiheads-finetuning run without being asked for. What it saves is real: the
    replay set is the large one and its final metrics are not what the run is
    for."""
    log, _ = artifacts

    assert "| train_pt_head" not in log
    assert "| valid_pt_head" not in log


def test_the_skipped_head_is_named_in_the_log(artifacts):
    """Twice over: once for the flag's value, once per loader dropped. It is the
    only evidence, since the table cannot show what is not in it."""
    log, _ = artifacts

    assert "Skipping evaluation for heads: ['pt_head']" in log
    for split in ("train_pt_head", "valid_pt_head"):
        assert f"Skipping evaluation of {split} (in skip_heads list)" in log
