"""The fine-tuning contract, offline, from a committed anchor.

Separate file from ``test_finetuning_pseudolabels.py`` on purpose, and the
reason is a trap worth stating rather than working around: that file carries
a module-level ``pytest.mark.network``, so *every* test added to it inherits
it and is skipped on every pull-request job -- the contracts here would run
only in the nightly network job, which is precisely the "a skip is not a
pass" failure. These tests use a committed tiny anchor as their foundation
model and download nothing, so they belong in the required suite and are put
where they will land there.

The existing network file is not replaced. It pins something these cannot:
that real published foundation models reproduce their own labels. What is
pinned here is the *mechanics* -- that replay assembles, that a frozen
``pt_head`` trains beside the new head, that pseudolabels come from the model
rather than the file -- which needs no published model and must not be
hostage to a download.

Two of the behaviours below are **recorded DROPs**: the silent per-batch
pseudolabel fallback and the unlabelled-replay continuation. They are pinned
so the rewrite's decision to remove them is a declared change with a test to
delete, rather than something a user discovers.
"""

from __future__ import annotations

import json
from pathlib import Path

import ase.io
import numpy as np
import pytest
from ase.atoms import Atoms

from tests.helpers import REPO_ROOT, run_mace_train
from tests.workflows.conftest import split_regression_set

FINETUNING_SELECT = REPO_ROOT / "mace" / "cli" / "fine_tuning_select.py"


def finetuning_params(work: Path, finetune: Path, replay: Path, **overrides) -> dict:
    params = {
        "name": "ft",
        "train_file": str(finetune),
        "valid_fraction": 0.25,
        "E0s": "isolated",
        "loss": "weighted",
        "batch_size": 4,
        "valid_batch_size": 4,
        "max_num_epochs": 2,
        "eval_interval": 1,
        "device": "cpu",
        "default_dtype": "float64",
        "seed": 11,
        "multiheads_finetuning": True,
        "pt_train_file": str(replay),
        "force_mh_ft_lr": True,
        "lr": 0.005,
        "error_table": "PerAtomRMSE",
        "save_cpu": None,
        "model_dir": str(work),
        "checkpoints_dir": str(work),
        "results_dir": str(work),
        "log_dir": str(work),
    }
    params.update(overrides)
    return params


def eval_records(work: Path, name: str, seed: int, head: str) -> list:
    """Every validation record the run logged for one head, in order."""
    path = work / f"{name}_run-{seed}_train.txt"
    assert path.exists(), f"no record file at {path}"
    out = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        record = json.loads(line)
        if record.get("mode") == "eval" and record.get("head") == head:
            out.append(record)
    return out


@pytest.fixture(name="split")
def fixture_split(tmp_path):
    return split_regression_set(tmp_path)


# ---------------------------------------------------------------------------
# (a) - (c): the run completes, carries both heads, and adapts
# ---------------------------------------------------------------------------


@pytest.mark.timeout(900)
def test_multihead_replay_finetuning_completes_and_carries_both_heads(
    tmp_path, split, anchor_scaleshift
):
    """(a) and (b) of the fine-tuning contract, in one run.

    The heads are read off the saved artefact rather than off the log,
    because the log line is a message and the head list is the thing every
    downstream tool -- ``mace_select_head``, ``--head`` at evaluation, the
    LAMMPS export -- actually dispatches on.
    """
    import torch  # noqa: PLC0415

    finetune, replay = split
    run_mace_train(
        finetuning_params(
            tmp_path, finetune, replay, foundation_model=str(anchor_scaleshift)
        )
    )
    model_path = tmp_path / "ft.model"
    assert model_path.exists(), "the fine-tuning run wrote no model"

    model = torch.load(model_path, map_location="cpu", weights_only=False)
    assert set(model.heads) == {"Default", "pt_head"}, (
        f"a multi-head fine-tuning run produced heads {list(model.heads)}; the "
        f"replay head is what mitigates forgetting, so its absence is the "
        f"whole feature missing"
    )

    for head in ("Default", "pt_head"):
        assert eval_records(tmp_path, "ft", 11, head), (
            f"the {head!r} head was never evaluated, so nothing says it trained"
        )


@pytest.mark.timeout(900)
def test_finetuning_reduces_the_error_on_the_finetuning_set(
    tmp_path, split, anchor_scaleshift
):
    """(c): adaptation is measured against the unadapted model, not against zero.

    The fine-tuning half of the committed set is the molecules with their
    labels scaled, so it is a different level of theory from the one the
    anchor was trained on -- which is what makes "the error went down"
    meaningful rather than a restatement of the anchor's own training error.
    The comparison is the run's own initial evaluation (the foundation model
    before a single step) against its last, both from the record file.
    """
    finetune, replay = split
    run_mace_train(
        finetuning_params(
            tmp_path,
            finetune,
            replay,
            max_num_epochs=6,
            foundation_model=str(anchor_scaleshift),
        )
    )
    records = eval_records(tmp_path, "ft", 11, "Default")
    initial = next(r for r in records if r.get("epoch") is None)
    trained = [r for r in records if r.get("epoch") is not None]
    assert trained, "the fine-tuning head was never evaluated after an epoch"
    final = trained[-1]

    assert final["loss"] < initial["loss"], (
        f"fine-tuning did not adapt: the unadapted foundation model scored "
        f"{initial['loss']:.6g} and after {len(trained)} epochs it scores "
        f"{final['loss']:.6g}"
    )


# ---------------------------------------------------------------------------
# (d): pseudolabels come from the model
# ---------------------------------------------------------------------------


@pytest.mark.timeout(900)
def test_pseudolabel_replay_relabels_the_replay_set_from_the_foundation_model(
    tmp_path, split, anchor_scaleshift
):
    """(d), pinned as a contrast rather than as a threshold.

    With ``--pseudolabel_replay`` the replay labels are regenerated by the
    foundation model, so the foundation model's own error against them is
    zero to machine precision -- it is being scored against its own output.
    Without the flag the same replay file keeps its file labels and the error
    is finite. Two runs differing in one flag, and the initial ``pt_head``
    evaluation is the observable, so nothing depends on how well anything
    trains.
    """
    finetune, replay = split
    losses = {}
    for label, extra in (
        ("file_labels", {}),
        ("pseudolabels", {"pseudolabel_replay": True}),
    ):
        run_mace_train(
            finetuning_params(
                tmp_path,
                finetune,
                replay,
                name=label,
                max_num_epochs=0,
                foundation_model=str(anchor_scaleshift),
                **extra,
            )
        )
        initial = next(
            r for r in eval_records(tmp_path, label, 11, "pt_head")
            if r.get("epoch") is None
        )
        losses[label] = initial["loss"]

    assert losses["file_labels"] > 1e-6, (
        f"the replay set's own labels already agree with the foundation model "
        f"to {losses['file_labels']:.3g}, so this comparison cannot tell "
        f"pseudolabels from file labels; the replay half of the committed set "
        f"has to stay something the anchor does not already predict"
    )
    assert losses["pseudolabels"] < 1e-12, (
        f"--pseudolabel_replay left the pt_head with a loss of "
        f"{losses['pseudolabels']:.3g} against labels it is supposed to have "
        f"generated itself; the labels came from the file"
    )


# ---------------------------------------------------------------------------
# The two recorded DROPs
# ---------------------------------------------------------------------------


@pytest.mark.timeout(900)
def test_a_failing_pseudolabel_batch_keeps_the_file_labels_and_says_nothing(
    tmp_path, split, anchor_scaleshift
):
    """A batch that cannot be relabelled stops the stage instead of mixing.

    Pseudolabel generation used to catch every exception per batch and keep
    that batch's *original file* labels, so a replay set could end up holding
    foundation-model labels for most configurations and foreign ones for the
    rest, with only a log line to say which -- and the stage then reported
    success. It refuses now, naming the batch and the configurations in it,
    and the caller reports that it did not succeed.

    "continuing with original configurations" is the load-bearing part. It was
    false before: train was replaced as soon as it relabelled, so a failure on
    valid left the two splits on different label sources while that line
    claimed nothing had changed. Both splits are committed together now, so
    the message is accurate.

    Exhibiting it still needs a configuration the generation step chokes on,
    and the one reachable from the command line -- an empty configuration,
    which trips the aperiodic extent calculation -- also stops the run later in
    the ordinary loader. So this asserts what the pseudolabel stage says, not
    the exit code, which belongs to that unrelated second failure.
    """
    finetune, replay = split
    contaminated = tmp_path / "replay_with_empty.xyz"
    configs = ase.io.read(replay, index=":")
    empty = Atoms(numbers=[], positions=np.zeros((0, 3)), cell=[6.0] * 3, pbc=False)
    empty.info["REF_energy"] = 0.0
    ase.io.write(contaminated, configs + [empty])

    completed = run_mace_train(
        finetuning_params(
            tmp_path,
            finetune,
            contaminated,
            name="dropbatch",
            max_num_epochs=0,
            foundation_model=str(anchor_scaleshift),
            pseudolabel_replay=True,
            batch_size=2,
        ),
        check=False,
        capture_output=True,
        text=True,
    )
    printed = completed.stdout + completed.stderr

    assert "Pseudolabelling failed on batch" in printed, (
        "no batch failed, so this test is no longer exhibiting the refusal; "
        "find another way to make one batch fail before deleting it\n"
        + printed[-4000:]
    )
    assert "Pseudolabeling was not successful" in printed, (
        "a batch failed and the stage still declared success, which is the "
        "silent label mixing this refusal exists to prevent:\n"
        + printed[-4000:]
    )
    assert "Successfully applied pseudolabels to pt_head configurations" not in printed, (
        "the stage reported success for a run in which a batch failed:\n"
        + printed[-4000:]
    )


@pytest.mark.timeout(900)
def test_pseudolabel_replay_accepts_a_replay_set_with_no_labels_at_all(
    tmp_path, split, anchor_scaleshift
):
    """An unlabelled replay set is relabelled into something that trains.

    With pseudolabelling on, the replay reader is told that finding no labels is
    acceptable, on the grounds that they are about to be generated. That used to
    be worse than merely permissive: the generated labels arrived carrying the
    file's property *weights*, which are zero, so the ``pt_head`` reported a loss
    of exactly zero with every error metric absent -- the best number on the page,
    from a head contributing nothing to the gradient.

    The weights follow the labels now, so the head reports a real loss and real
    metrics. What is asserted is the presence of the metrics rather than their
    size: at ``max_num_epochs=0`` the labels come from the same model being
    evaluated, so the residual is tiny by construction, and a threshold here
    would be measuring that coincidence rather than the fix.

    The contrast run -- same file, flag off -- is still refused outright, naming
    the keys it could not find.
    """
    finetune, replay = split
    unlabelled = tmp_path / "replay_unlabelled.xyz"
    configs = ase.io.read(replay, index=":")
    for atoms in configs:
        for key in ("REF_energy", "REF_stress", "REF_dipole"):
            atoms.info.pop(key, None)
        for key in ("REF_forces", "REF_charges"):
            atoms.arrays.pop(key, None)
    ase.io.write(unlabelled, configs)

    run_mace_train(
        finetuning_params(
            tmp_path,
            finetune,
            unlabelled,
            name="nodataok",
            max_num_epochs=0,
            foundation_model=str(anchor_scaleshift),
            pseudolabel_replay=True,
        )
    )
    initial = next(
        r for r in eval_records(tmp_path, "nodataok", 11, "pt_head")
        if r.get("epoch") is None
    )
    assert initial.get("rmse_e_per_atom") is not None, (
        "the unlabelled replay head reported no energy error at all, which is "
        "what a head whose property weights are all zero does: the generated "
        "labels are present but contribute nothing"
    )
    assert initial.get("rmse_f") is not None, (
        "the unlabelled replay head reported no force error, so the generated "
        "forces carry no weight"
    )

    refused = run_mace_train(
        finetuning_params(
            tmp_path,
            finetune,
            unlabelled,
            name="nodatarefused",
            max_num_epochs=0,
            foundation_model=str(anchor_scaleshift),
        ),
        check=False,
        capture_output=True,
        text=True,
    )
    assert refused.returncode != 0, (
        "without --pseudolabel_replay an unlabelled replay set is accepted; "
        "the recorded behaviour is a refusal naming the missing keys"
    )
    assert "None of" in (refused.stdout + refused.stderr)


# ---------------------------------------------------------------------------
# Replay-set selection, through the command line
#
# The three `--*_pt` flags on `mace_run_train` are forwarded verbatim into the
# same selection settings `mace_finetuning_select` exposes as `--subselect`,
# `--filtering_type` and `--disallow_random_padding`. The `_pt` flags
# themselves only run against the four *downloaded* replay corpora, so the
# semantics are pinned here through the selection entry point, which needs no
# network. What a run does with the selected file is covered above.
# ---------------------------------------------------------------------------


def run_selection(cwd, **flags):
    """Drive ``mace_finetuning_select``, from a scratch directory.

    ``cwd`` is not optional and is not cosmetic: the selection CLI writes a
    ``<pool>_descriptors.npy`` cache **next to wherever it was started**, not
    next to its output, so a test run from the checkout leaves a stray array
    in the repository root. Discovered by finding one there.
    """
    params = {"device": "cpu", "default_dtype": "float64"}
    params.update(flags)
    return run_mace_train(
        params,
        script=FINETUNING_SELECT,
        check=False,
        capture_output=True,
        text=True,
        cwd=str(cwd),
    )


@pytest.fixture(name="selection_pool")
def fixture_selection_pool(tmp_path, regression_set):
    """A pool to select from, and a fine-tuning set that fixes the elements."""
    configs = ase.io.read(regression_set, index=":")
    pool = tmp_path / "pool.xyz"
    ase.io.write(pool, [a for a in configs if len(a) > 1])

    # A water molecule, so the element filter has something to cut on: the
    # committed pool is mostly carbon-bearing, and the handful of pure H/O
    # configurations are what an 'exclusive' filter keeps.
    target = tmp_path / "target.xyz"
    ase.io.write(
        target,
        [Atoms("OH2", positions=[[0, 0, 0.12], [0, 0.76, -0.47], [0, -0.76, -0.47]])],
    )
    return pool, target, len(ase.io.read(pool, index=":"))


@pytest.mark.timeout(300)
@pytest.mark.parametrize(
    "flag, bad_value, expected",
    [
        ("subselect_pt", "not_a_method", "fps"),
        ("filter_type_pt", "not_a_filter", "combinations"),
    ],
)
def test_the_replay_selection_flags_exist_on_the_training_cli(
    tmp_path, regression_set, flag, bad_value, expected
):
    """The `_pt` flags themselves, pinned as a surface rather than by effect.

    Their *behaviour* only runs against the four downloaded replay corpora,
    so the semantics are pinned through ``mace_finetuning_select`` below.
    What can be pinned offline is that the flag still exists on the training
    command line and still accepts the same vocabulary -- which is what a
    rename or a dropped choice would break, and what every replay recipe in
    the wild spells out.
    """
    completed = run_mace_train(
        {
            "name": "flags",
            "train_file": str(regression_set),
            "device": "cpu",
            "model_dir": str(tmp_path),
            flag: bad_value,
        },
        check=False,
        capture_output=True,
        text=True,
    )
    printed = completed.stdout + completed.stderr
    assert completed.returncode != 0, f"--{flag} accepted {bad_value!r}"
    assert f"--{flag}" in printed, (
        f"--{flag} is no longer a recognised argument:\n" + printed[-2000:]
    )
    assert expected in printed, (
        f"--{flag} no longer offers {expected!r}:\n" + printed[-2000:]
    )


@pytest.mark.timeout(300)
def test_disallow_random_padding_pt_is_a_bare_flag_on_the_training_cli(
    tmp_path, regression_set
):
    """The third selection flag, which has no choices to probe.

    ``--disallow_random_padding_pt`` is ``store_false`` onto
    ``allow_random_padding_pt``, so the only thing to pin offline is that it
    is still accepted with no argument -- the inverted name is exactly the
    kind of thing a rewrite renames without noticing.
    """
    completed = run_mace_train(
        {
            "name": "padflag",
            "train_file": str(regression_set),
            "device": "cpu",
            "model_dir": str(tmp_path),
            "disallow_random_padding_pt": None,
            "num_samples_pt": "not_an_int",
        },
        check=False,
        capture_output=True,
        text=True,
    )
    printed = completed.stdout + completed.stderr
    assert "--disallow_random_padding_pt" not in printed.split("error:")[-1], (
        "argparse rejected --disallow_random_padding_pt itself:\n"
        + printed[-2000:]
    )
    assert "--num_samples_pt" in printed, (
        "the run failed on something other than the deliberately bad "
        "--num_samples_pt, so this test is not showing that the padding flag "
        "parsed:\n" + printed[-2000:]
    )


@pytest.mark.timeout(600)
def test_subselect_random_returns_exactly_the_requested_number(
    tmp_path, selection_pool
):
    """``--subselect random`` is the offline half of the pair.

    It is also the branch replay runs on a machine with no descriptors: it
    loads no model at all, which is why the fine-tuning tests above can use
    it without a foundation model on disc.
    """
    pool, _target, pool_size = selection_pool
    output = tmp_path / "selected.xyz"
    completed = run_selection(
        tmp_path,
        configs_pt=str(pool),
        output=str(output),
        num_samples=4,
        subselect="random",
        filtering_type="none",
    )
    assert completed.returncode == 0, completed.stderr[-2000:]
    selected = ase.io.read(output, index=":")
    assert len(selected) == 4, f"asked for 4 of {pool_size}, got {len(selected)}"
    assert all(atoms.info.get("pretrained") for atoms in selected), (
        "the selected configurations are not marked as replay data, so the "
        "training run cannot tell them from the fine-tuning set"
    )


@pytest.mark.timeout(600)
def test_subselect_fps_uses_the_model_and_still_returns_the_requested_number(
    tmp_path, selection_pool, anchor_scaleshift
):
    """``--subselect fps`` is the default and the one that needs a model.

    Farthest-point sampling scores the pool with descriptors from the model
    it is given, so this is also the only selection path that would break if
    the descriptor surface changed shape. Given an anchor, it runs offline.
    """
    pool, target, _ = selection_pool
    output = tmp_path / "selected_fps.xyz"
    completed = run_selection(
        tmp_path,
        configs_pt=str(pool),
        configs_ft=str(target),
        output=str(output),
        num_samples=4,
        subselect="fps",
        filtering_type="none",
        model=str(anchor_scaleshift),
    )
    assert completed.returncode == 0, completed.stderr[-2000:]
    assert len(ase.io.read(output, index=":")) == 4


@pytest.mark.timeout(600)
def test_filtering_type_restricts_the_pool_to_the_target_elements(
    tmp_path, selection_pool
):
    """``--filtering_type`` decides which replay configurations are eligible.

    ``none`` keeps everything; ``exclusive`` keeps only configurations built
    entirely from the fine-tuning set's elements. The committed pool has both
    kinds, so the two settings must not return the same thing -- and a
    filtering type other than ``none`` with no elements to filter on has to
    be refused rather than silently degrade to ``none``.
    """
    pool, target, pool_size = selection_pool
    kept = {}
    for filtering in ("none", "exclusive"):
        output = tmp_path / f"filtered_{filtering}.xyz"
        completed = run_selection(
            tmp_path,
            configs_pt=str(pool),
            configs_ft=str(target),
            output=str(output),
            subselect="random",
            filtering_type=filtering,
        )
        assert completed.returncode == 0, completed.stderr[-2000:]
        kept[filtering] = len(ase.io.read(output, index=":"))

    assert kept["none"] == pool_size, kept
    assert 0 < kept["exclusive"] < kept["none"], (
        f"filtering kept {kept}; the committed pool contains both "
        f"hydrogen-oxygen-only configurations and carbon-bearing ones, so "
        f"'exclusive' against a water target has to keep some and drop some"
    )

    refused = run_selection(
        tmp_path,
        configs_pt=str(pool),
        output=str(tmp_path / "nofilter.xyz"),
        subselect="random",
        filtering_type="exclusive",
    )
    assert refused.returncode != 0, (
        "a filtering type was accepted with no elements to filter on; that "
        "silently selects the whole pool while looking filtered"
    )


@pytest.mark.timeout(600)
def test_random_padding_tops_up_a_short_pool_and_disallowing_it_does_not(
    tmp_path, selection_pool
):
    """``--disallow_random_padding`` decides what happens when the pool is short.

    Asking for more replay configurations than the filter left is the normal
    case for a narrow fine-tuning set. The default is to make up the
    difference with random configurations from the *rejected* remainder --
    which quietly reintroduces exactly the compositions the filter excluded,
    and is the behaviour worth knowing about. Disallowing it does **not**
    return the short set, as the flag's help suggests: the run is refused,
    naming the shortfall. Both halves are pinned, because a rewrite that read
    only the help text would implement the wrong one.
    """
    pool, target, _ = selection_pool
    wanted = 12  # more than the water-only subset, fewer than the whole pool

    padded_output = tmp_path / "padding_padded.xyz"
    padded = run_selection(
        tmp_path,
        configs_pt=str(pool),
        configs_ft=str(target),
        output=str(padded_output),
        num_samples=wanted,
        subselect="random",
        filtering_type="exclusive",
    )
    assert padded.returncode == 0, padded.stderr[-2000:]
    selected = ase.io.read(padded_output, index=":")
    assert len(selected) == wanted, (
        f"random padding did not top the selection up to {wanted}, it "
        f"returned {len(selected)}"
    )
    assert any("C" in atoms.get_chemical_symbols() for atoms in selected), (
        "the top-up drew nothing from the rejected remainder, so this test is "
        "not exercising the padding branch -- and the branch's whole point is "
        "that it reintroduces the compositions the filter excluded"
    )

    refused = run_selection(
        tmp_path,
        configs_pt=str(pool),
        configs_ft=str(target),
        output=str(tmp_path / "padding_short.xyz"),
        num_samples=wanted,
        subselect="random",
        filtering_type="exclusive",
        disallow_random_padding=None,
    )
    assert refused.returncode != 0, (
        "--disallow_random_padding returned a selection; the recorded "
        "behaviour is a refusal, so either the flag now degrades gracefully "
        "(an improvement worth recording here) or it stopped being read"
    )
    assert "than available" in (refused.stdout + refused.stderr), (
        "the refusal did not name the shortfall:\n"
        + (refused.stdout + refused.stderr)[-2000:]
    )
