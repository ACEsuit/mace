"""The CLI-observable contract of the legacy stack.

These tests are written once and re-used verbatim: the rewrite may change the
syntax of the command line, but the semantics pinned here are the user
contract it has to honour, and the live parity run re-executes this file with
the engine switched rather than porting it. That is only possible because of
one rule, and it is the rule to break the tests over:

    **every assertion is made on something a user can see.**

Return codes, the log the CLI prints, the JSON-lines record file it writes
into ``--results_dir``, the checkpoints and models it leaves on disc, the
extxyz it writes, and the results an ase calculator returns. Nothing here
imports ``mace.modules``, ``mace.data`` or ``mace.tools`` -- the two imports
of the package that do appear are ``mace.calculators.MACECalculator``, which
is itself one of the contracts under test (there is no console script for the
ase calculator, and the graph-padding arguments exist nowhere else), and
``torch.load`` on an artefact, which reads a file rather than reaching into
the implementation. Tolerances come from the one table in
``tests/golden/harness.py`` and are never restated as a literal here.

What is deliberately *not* here: every CLI flag (out of scope), fine-tuning
(``test_finetuning_contracts.py``), the LAMMPS export
(``tests/integrations/lammps/``), and the magnetic and long-range command
lines, which belong to their own families.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import ase.io
import numpy as np
import pytest

from tests.golden import harness
from tests.helpers import REPO_ROOT, run_mace_train

EVAL_CONFIGS = REPO_ROOT / "mace" / "cli" / "eval_configs.py"
SELECT_HEAD = REPO_ROOT / "mace" / "cli" / "select_head.py"

TOL = harness.FP64_CPU_REFERENCE


# ---------------------------------------------------------------------------
# Reading the artefacts the CLI leaves behind
# ---------------------------------------------------------------------------


def training_records(results_dir, name: str, seed: int) -> list:
    """The JSON-lines record file ``mace_run_train`` writes per run.

    This is a first-class user artefact -- ``--results_dir`` exists to produce
    it and ``mace_plot_train`` consumes it -- so reading it keeps the
    assertions black-box while giving them numbers instead of log scraping.
    """
    path = Path(results_dir) / f"{name}_run-{seed}_train.txt"
    assert path.exists(), (
        f"the training run wrote no record file at {path}; --results_dir is "
        f"part of the CLI contract"
    )
    records = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line:
            records.append(json.loads(line))
    return records


def validation_losses(records, head: str = "Default") -> dict:
    """epoch -> validation loss, for the epochs that were evaluated."""
    return {
        record["epoch"]: record["loss"]
        for record in records
        if record.get("mode") == "eval"
        and record.get("head") == head
        and record.get("epoch") is not None
    }


def optimiser_steps_per_epoch(records) -> dict:
    """epoch -> how many optimiser steps were logged for it.

    The number is the regime: the per-batch path logs one record per batch,
    the L-BFGS path logs exactly one per epoch because the whole epoch is one
    closure.
    """
    counts: dict = {}
    for record in records:
        if record.get("mode") == "opt":
            counts[record["epoch"]] = counts.get(record["epoch"], 0) + 1
    return counts


def base_training_params(tmp_path: Path, regression_set: Path, **overrides) -> dict:
    """A tiny, fast, deterministic training run on the committed set.

    Small enough that a workflow test is seconds rather than minutes, and
    large enough that the loss has somewhere to go: the labels come from a
    closed-form potential (tests/golden/make_regression_set.py), so a run that
    does not improve is a broken optimiser rather than an unlucky split.
    """
    params = {
        "name": "contract",
        "train_file": str(regression_set),
        "valid_fraction": 0.25,
        "E0s": "isolated",
        "energy_key": "REF_energy",
        "forces_key": "REF_forces",
        "stress_key": "REF_stress",
        "loss": "weighted",
        "energy_weight": 1.0,
        "forces_weight": 10.0,
        "r_max": 3.5,
        "max_ell": 2,
        "num_radial_basis": 6,
        "num_cutoff_basis": 5,
        "num_interactions": 2,
        "correlation": 2,
        "hidden_irreps": "8x0e + 8x1o",
        "MLP_irreps": "4x0e",
        "batch_size": 8,
        "valid_batch_size": 4,
        "max_num_epochs": 4,
        "eval_interval": 1,
        "lr": 0.01,
        "device": "cpu",
        "default_dtype": "float64",
        "seed": 7,
        "error_table": "PerAtomRMSE",
        "save_cpu": None,
        "model_dir": str(tmp_path),
        "checkpoints_dir": str(tmp_path),
        "results_dir": str(tmp_path),
        "log_dir": str(tmp_path),
    }
    params.update(overrides)
    return params


def run_eval(model: Path, configs: Path, output: Path, **flags) -> None:
    """Drive ``mace_eval_configs`` and insist it succeeded."""
    params = {
        "configs": str(configs),
        "model": str(model),
        "output": str(output),
        "device": "cpu",
        "default_dtype": "float64",
    }
    # run_mace_train renders a None value as a bare flag, which is what the
    # store_true options need, so the mapping is passed through untouched.
    params.update(flags)
    run_mace_train(params, script=EVAL_CONFIGS)
    assert output.exists(), f"mace_eval_configs wrote no {output}"


# ---------------------------------------------------------------------------
# 1. Train smoke
# ---------------------------------------------------------------------------


@pytest.mark.timeout(600)
def test_training_reduces_the_validation_loss_and_writes_a_model(
    tmp_path, regression_set
):
    """The floor of every training contract: it runs, it learns, it saves.

    Deliberately first->last rather than epoch-to-epoch monotone: a step that
    does not improve is a legitimate outcome of any optimiser, and pinning
    monotonicity would make the contract about the learning rate.
    """
    params = base_training_params(tmp_path, regression_set)
    run_mace_train(params)

    model_path = tmp_path / "contract.model"
    assert model_path.exists(), "the training run wrote no model"
    checkpoints = sorted(tmp_path.glob("contract_run-7_epoch-*.pt"))
    assert checkpoints, "the training run wrote no checkpoint"

    losses = validation_losses(training_records(tmp_path, "contract", 7))
    assert len(losses) >= 2, f"only {len(losses)} validation evaluations were logged"
    first, last = losses[min(losses)], losses[max(losses)]
    assert last < first, (
        f"the validation loss did not fall over training: epoch {min(losses)} "
        f"{first:.6g} -> epoch {max(losses)} {last:.6g}. On this dataset the "
        f"labels come from a closed-form potential, so this is the optimiser, "
        f"not the split."
    )


@pytest.mark.timeout(600)
def test_the_end_of_training_error_table_is_printed_and_parseable(
    tmp_path, regression_set
):
    """`--error_table` is a user-facing report, so its shape is a contract.

    Pinned: it names both subsets, it carries one column per trained
    property, and every cell is a number rather than the ``None`` the table
    prints for a property that was never fitted.
    """
    params = base_training_params(tmp_path, regression_set)
    completed = run_mace_train(params, capture_output=True, text=True)
    printed = completed.stdout + completed.stderr

    assert "Error-table on TRAIN and VALID" in printed, printed[-2000:]
    rows = {}
    for line in printed.splitlines():
        cells = [cell.strip() for cell in line.split("|")[1:-1]]
        if len(cells) >= 2 and cells[0] in ("train_Default", "valid_Default"):
            rows[cells[0]] = cells[1:]
    assert set(rows) == {"train_Default", "valid_Default"}, (
        f"the error table did not carry both subsets, only {sorted(rows)}"
    )
    for subset, cells in rows.items():
        assert len(cells) >= 3, f"{subset}: {cells}"
        for cell in cells[:3]:
            float(cell)  # raises, with the offending text, if it is not one


# ---------------------------------------------------------------------------
# 2. Resume
# ---------------------------------------------------------------------------


@pytest.mark.timeout(900)
def test_restart_latest_continues_from_the_checkpoint_epoch(tmp_path, regression_set):
    """A resumed run picks up where it stopped, in both senses.

    The epoch counter is the easy half. The half that actually matters is
    that the loss does not jump back to where an untrained model sits: a
    resume that silently reinitialised the weights would still advance the
    counter and still finish, and only the loss would say so.
    """
    first = base_training_params(tmp_path, regression_set, max_num_epochs=3)
    run_mace_train(first)
    before = validation_losses(training_records(tmp_path, "contract", 7))
    assert max(before) == 2, sorted(before)

    resumed = base_training_params(
        tmp_path, regression_set, max_num_epochs=6, restart_latest=None
    )
    run_mace_train(resumed)
    after = validation_losses(training_records(tmp_path, "contract", 7))

    new_epochs = sorted(epoch for epoch in after if epoch > max(before))
    assert new_epochs, (
        f"--restart_latest logged no epoch beyond {max(before)}; the counter "
        f"did not advance and the run repeated itself"
    )
    assert max(new_epochs) == 5, sorted(after)

    resumed_first = after[min(new_epochs)]
    assert resumed_first <= before[max(before)] * 1.5, (
        f"the resumed run started at a validation loss of {resumed_first:.6g}, "
        f"against {before[max(before)]:.6g} where the first run stopped. That "
        f"is the size of a reinitialised model, not of a resumed one."
    )


# ---------------------------------------------------------------------------
# 3. The L-BFGS regime
#
# A second optimiser regime that no test drove end to end. It is not just
# another optimiser: it flips `drop_last` off on every loader because the
# closure needs the whole epoch, and it routes the epoch through a single
# closure-driven step instead of one step per batch. Each of those three is a
# separately named test, because each can break on its own.
# ---------------------------------------------------------------------------


@pytest.mark.timeout(900)
def test_lbfgs_is_selectable_from_the_command_line_and_reduces_the_loss(
    tmp_path, regression_set
):
    params = base_training_params(
        tmp_path, regression_set, name="lbfgs", lbfgs=None, max_num_epochs=4
    )
    run_mace_train(params)

    assert (tmp_path / "lbfgs.model").exists(), "the L-BFGS run wrote no model"
    losses = validation_losses(training_records(tmp_path, "lbfgs", 7))
    assert len(losses) >= 2, sorted(losses)
    first, last = losses[min(losses)], losses[max(losses)]
    assert last < first, (
        f"the L-BFGS validation loss did not fall: {first:.6g} -> {last:.6g}"
    )


@pytest.mark.timeout(900)
def test_lbfgs_takes_one_step_per_epoch_and_the_other_regime_one_per_batch(
    tmp_path, regression_set
):
    """The regime is visible in the record file, which is how it is pinned.

    ``take_step_lbfgs`` accumulates the whole epoch into one closure and the
    optimiser steps once; the default path steps once per batch. So the count
    of ``mode: opt`` records per epoch separates the two without reading a
    single line of the implementation.
    """
    common = dict(batch_size=4, max_num_epochs=2)
    run_mace_train(
        base_training_params(tmp_path, regression_set, name="perbatch", **common)
    )
    run_mace_train(
        base_training_params(
            tmp_path, regression_set, name="whole", lbfgs=None, **common
        )
    )

    per_batch = optimiser_steps_per_epoch(training_records(tmp_path, "perbatch", 7))
    whole_epoch = optimiser_steps_per_epoch(training_records(tmp_path, "whole", 7))

    assert set(whole_epoch.values()) == {1}, (
        f"--lbfgs logged {whole_epoch} optimiser steps per epoch; the regime "
        f"is one closure over the whole epoch, so it must be exactly one"
    )
    assert min(per_batch.values()) > 1, (
        f"the default regime logged {per_batch} steps per epoch; it steps once "
        f"per batch, so more than one means the two regimes are still distinct"
    )


@pytest.mark.timeout(900)
def test_lbfgs_keeps_the_last_partial_batch_and_the_default_regime_drops_it(
    tmp_path, regression_set
):
    """``drop_last`` is flipped off by ``--lbfgs``, and here is what that costs.

    Not a cosmetic difference. With a batch size larger than the training set
    there is exactly one batch and it is partial, so dropping it leaves the
    loader empty -- and the default regime then dies before training starts,
    on an empty-tensor error from the average-neighbour statistic, while the
    identical command with ``--lbfgs`` trains normally. Characterisation, not
    approval: the failure is a poor one and a rewrite is free to improve it,
    but it must not silently start *training on a subset* instead.
    """
    oversized = dict(batch_size=64, valid_batch_size=4, max_num_epochs=2)

    run_mace_train(
        base_training_params(
            tmp_path, regression_set, name="keeps", lbfgs=None, **oversized
        )
    )
    losses = validation_losses(training_records(tmp_path, "keeps", 7))
    assert losses, "--lbfgs did not train at all with a batch larger than the set"

    dropped = run_mace_train(
        base_training_params(tmp_path, regression_set, name="drops", **oversized),
        check=False,
        capture_output=True,
        text=True,
    )
    assert dropped.returncode != 0, (
        "without --lbfgs a batch size larger than the training set leaves the "
        "loader empty, so the run cannot succeed; it did, which means "
        "drop_last is no longer the difference between the two regimes"
    )
    assert "non-empty list of Tensors" in (dropped.stdout + dropped.stderr), (
        "the default regime failed for some reason other than the empty "
        "loader that drop_last produces:\n" + (dropped.stdout + dropped.stderr)[-2000:]
    )


@pytest.mark.timeout(900)
def test_restart_latest_with_lbfgs_resumes_and_never_reaches_the_post_swap_reload(
    tmp_path, regression_set
):
    """The recorded state of the ``restart_lbfgs`` branch: it does not run.

    ``mace/cli/run_train.py`` carries a second, post-optimiser-swap reload for
    L-BFGS resumes, reached only when *both* ``load_latest`` attempts raise.
    They cannot both raise. The intended trigger -- a checkpoint whose L-BFGS
    optimiser state the freshly built Adam cannot accept -- raises a
    ``ValueError`` that the checkpoint loader itself catches and downgrades to
    a warning, so the load returns normally; and the two attempts partition
    the checkpoints into stage-two and stage-one, so a non-empty checkpoint
    set always satisfies one of them (an empty one returns ``None`` rather
    than raising).

    What is therefore pinned is the behaviour a user gets, in both checkpoint
    layouts: the run resumes at the checkpoint's epoch with a *fresh* L-BFGS
    optimiser, says so, and never loads a second time. The ordering assertion
    is the whole point -- a reload through the dead branch would print its
    ``Loading checkpoint`` line *after* the swap, and this test is what would
    notice if that branch ever came back to life.
    """
    for name, stage_two in (("lbstage1", {}), ("lbstage2", {"swa": None, "start_swa": 1})):
        first = base_training_params(
            tmp_path, regression_set, name=name, lbfgs=None, max_num_epochs=3,
            **stage_two,
        )
        run_mace_train(first)

        resumed = base_training_params(
            tmp_path, regression_set, name=name, lbfgs=None, max_num_epochs=5,
            restart_latest=None, **stage_two,
        )
        completed = run_mace_train(resumed, capture_output=True, text=True)
        printed = completed.stdout + completed.stderr

        swap = printed.index("Switching optimizer to LBFGS")
        loads = [
            index
            for index, line in enumerate(printed.splitlines())
            if "Loading checkpoint" in line
        ]
        assert loads, f"{name}: the resume loaded no checkpoint at all"
        setup = printed[:swap]
        assert "Loading checkpoint" in setup, (
            f"{name}: nothing was loaded before the optimiser swap, so the "
            f"resume did not go through the ordinary restart path"
        )
        # Everything after the swap and before training starts must contain no
        # second load: that region is exactly where the dead branch would act.
        started = printed.index("Started training")
        assert "Loading checkpoint" not in printed[swap:started], (
            f"{name}: a checkpoint was loaded between the L-BFGS swap and the "
            f"start of training. That is the restart_lbfgs branch, which this "
            f"contract records as unreachable -- if it now runs, the recorded "
            f"analysis is stale and the resume semantics changed."
        )

        after = validation_losses(training_records(tmp_path, name, 7))
        assert max(after) == 4, (
            f"{name}: the resumed L-BFGS run ended at epoch {max(after)}, not 4"
        )


# ---------------------------------------------------------------------------
# 4. The evaluation command line
# ---------------------------------------------------------------------------


class EvaluatedFile:
    """The eval CLI as something the golden harness can snapshot.

    The CLI is not calculator-shaped: it consumes a file of structures and
    writes its results back onto them under a prefix. So it is run once over
    all the fixtures and this object serves the per-structure results, which
    is also the honest thing to do -- ``--batch_size`` means the structures
    are not independent and evaluating them one at a time would test a
    different code path.
    """

    golden_surface = harness.SURFACE_EVAL

    def __init__(self, written: Path, prefix: str = "MACE_"):
        self.by_name = {
            atoms.info["golden_name"]: atoms
            for atoms in ase.io.read(written, index=":")
        }
        self.prefix = prefix

    def golden_outputs(self, atoms):
        return harness.collect_prefixed_outputs(
            self.by_name[atoms.info["golden_name"]], self.prefix
        )


#: These contracts drive the tiny ScaleShiftMACE anchor, which is an H/C/O
#: model. The golden manifest is shared with every other family, so a bare
#: load_fixtures() hands it whatever the next family commits -- and an
#: element it has no z-table row for fails inside the CLI subprocess as a
#: bare ValueError, which reads as a broken CLI rather than a wrong input.
ANCHOR_ELEMENTS = (1, 6, 8)


@pytest.fixture(name="fixture_structures", scope="module")
def fixture_fixture_structures():
    return harness.load_fixtures(elements=ANCHOR_ELEMENTS)


@pytest.fixture(name="fixture_file", scope="module")
def fixture_fixture_file(fixture_structures, tmp_path_factory):
    path = tmp_path_factory.mktemp("eval") / "fixtures.xyz"
    ase.io.write(path, list(fixture_structures.values()))
    return path


@pytest.mark.timeout(600)
def test_eval_configs_agrees_with_the_ase_calculator_on_every_fixture(
    tmp_path, anchor_scaleshift, fixture_structures, fixture_file
):
    """The two shipped inference routes must be the same numbers.

    Compared through the harness rather than by hand, so the comparison is
    channel-by-channel with units and shapes checked, and at the one fp64 row
    -- not at a number invented here.
    """
    from mace.calculators import MACECalculator  # noqa: PLC0415

    written = tmp_path / "evaluated.xyz"
    run_eval(
        anchor_scaleshift, fixture_file, written, compute_stress=None, batch_size=3
    )

    calculator = MACECalculator(
        model_paths=str(anchor_scaleshift), device="cpu", default_dtype="float64"
    )
    from_calculator = harness.snapshot_outputs(
        calculator, fixture_structures, channels=["energy", "forces"]
    )
    from_cli = harness.snapshot_outputs(
        EvaluatedFile(written), fixture_structures, channels=["energy", "forces"]
    )
    harness.compare_to_reference(from_cli, from_calculator, row=TOL.name)


@pytest.mark.timeout(600)
def test_eval_configs_reproduces_the_committed_anchor_reference(
    tmp_path, anchor_scaleshift, fixture_structures, fixture_file
):
    """The strongest form of the same claim: against the frozen numbers.

    The sibling test compares two live routes, which stays green if both move
    together. This compares the evaluation command line against the reference
    every other surface in the tree is asserted against, so the CLI cannot
    drift away from the calculator, the model forward and the LAMMPS export
    while remaining self-consistent with any one of them.
    """
    written = tmp_path / "against_reference.xyz"
    run_eval(anchor_scaleshift, fixture_file, written, compute_stress=None)

    snapshot = harness.snapshot_outputs(
        EvaluatedFile(written),
        fixture_structures,
        channels=["energy", "forces", "stress"],
    )
    reference = harness.load_reference(
        harness.REFERENCES_DIR / "tiny_scaleshift_e3nn_cpu_fp64.json"
    )
    harness.compare_to_reference(
        snapshot,
        reference,
        row=TOL.name,
        channels=["energy", "forces", "stress"],
    )


@pytest.mark.timeout(600)
def test_eval_batch_size_does_not_change_the_numbers(
    tmp_path, anchor_scaleshift, fixture_structures, fixture_file
):
    """Batching is a performance knob, so it is a bug if it is a physics knob."""
    snapshots = {}
    for batch_size in (1, 4, 64):
        written = tmp_path / f"evaluated_{batch_size}.xyz"
        run_eval(anchor_scaleshift, fixture_file, written, batch_size=batch_size)
        snapshots[batch_size] = harness.snapshot_outputs(
            EvaluatedFile(written), fixture_structures, channels=["energy", "forces"]
        )
    for batch_size in (4, 64):
        harness.compare_to_reference(
            snapshots[batch_size], snapshots[1], row=TOL.name
        )


@pytest.mark.timeout(600)
def test_eval_compute_stress_writes_a_stress_and_omitting_it_does_not(
    tmp_path, anchor_scaleshift, fixture_file
):
    with_stress = tmp_path / "with_stress.xyz"
    without_stress = tmp_path / "without_stress.xyz"
    run_eval(anchor_scaleshift, fixture_file, with_stress, compute_stress=None)
    run_eval(anchor_scaleshift, fixture_file, without_stress)

    for atoms in ase.io.read(with_stress, index=":"):
        assert "MACE_stress" in atoms.info, atoms.info["golden_name"]
        assert np.asarray(atoms.info["MACE_stress"]).shape == (3, 3)
    for atoms in ase.io.read(without_stress, index=":"):
        assert "MACE_stress" not in atoms.info, atoms.info["golden_name"]


@pytest.mark.timeout(600)
def test_eval_info_prefix_renames_every_written_key(
    tmp_path, anchor_scaleshift, fixture_file
):
    """`--info_prefix` is the only thing separating results from labels."""
    written = tmp_path / "prefixed.xyz"
    run_eval(
        anchor_scaleshift,
        fixture_file,
        written,
        compute_stress=None,
        info_prefix="pinned_",
    )
    for atoms in ase.io.read(written, index=":"):
        assert {"pinned_energy", "pinned_stress"} <= set(atoms.info)
        assert "pinned_forces" in atoms.arrays
        assert not [key for key in atoms.info if key.startswith("MACE_")]
        assert not [key for key in atoms.arrays if key.startswith("MACE_")]


@pytest.fixture(name="equal_size_file", scope="module")
def fixture_equal_size_file(tmp_path_factory):
    """Structures that all have the same number of atoms.

    Needed by ``--return_node_energies``, which cannot handle anything else
    (see the two tests below). Taken from the committed regression set so the
    file is reproducible.
    """
    from tests.workflows.conftest import REGRESSION_SET  # noqa: PLC0415

    configs = [
        atoms
        for atoms in ase.io.read(REGRESSION_SET, index=":")
        if len(atoms) == 6
    ]
    assert len(configs) >= 4, "the committed regression set lost its 6-atom cells"
    path = tmp_path_factory.mktemp("equal_size") / "equal_size.xyz"
    ase.io.write(path, configs)
    return path


@pytest.mark.timeout(600)
def test_eval_node_energies_sum_to_the_total_energy(
    tmp_path, anchor_scaleshift, equal_size_file
):
    """`--return_node_energies` asserted by value, not merely exercised.

    The per-atom energies are a decomposition of the total, isolated-atom
    reference included, so their sum is the energy the same run wrote. That
    is a real constraint: an off-by-one in the per-configuration split of a
    batch, or a per-atom array taken from the wrong graph, breaks it while
    still producing an array of the right shape. Two batch sizes, because the
    split is per batch and a one-batch run would not exercise it.
    """
    for batch_size in (1, 3):
        written = tmp_path / f"node_energies_{batch_size}.xyz"
        run_eval(
            anchor_scaleshift,
            equal_size_file,
            written,
            return_node_energies=None,
            batch_size=batch_size,
        )
        for index, atoms in enumerate(ase.io.read(written, index=":")):
            where = f"batch_size={batch_size}, config {index}"
            node_energies = np.asarray(atoms.arrays["MACE_node_energies"], dtype=float)
            assert node_energies.shape == (len(atoms),), where
            total = float(np.asarray(atoms.info["MACE_energy"]))
            assert node_energies.sum() == pytest.approx(total, abs=TOL.atol), (
                f"{where}: the per-atom energies sum to "
                f"{node_energies.sum():.12g} but the run reported a total of "
                f"{total:.12g}"
            )


@pytest.mark.timeout(600)
def test_eval_node_energies_cannot_handle_structures_of_different_sizes(
    tmp_path, anchor_scaleshift, fixture_file
):
    """RECORDED DEFECT: the flag works only on a set of equal-sized structures.

    The per-configuration arrays are collected into a list of lists and then
    handed to ``numpy.concatenate``, which builds a rectangular array -- so
    the moment two structures have different atom counts the run dies on
    "inhomogeneous shape", before writing anything. Every realistic
    evaluation set is ragged, so the flag is effectively unusable as shipped,
    which is why the sibling test above needs a purpose-built file.

    Pinned rather than fixed, because this ticket characterises the frozen
    stack. **If this test starts failing, the defect was fixed** -- delete it
    and point the sibling at the ordinary fixture set, which is the whole
    repair.
    """
    written = tmp_path / "ragged_node_energies.xyz"
    completed = run_mace_train(
        {
            "configs": str(fixture_file),
            "model": str(anchor_scaleshift),
            "output": str(written),
            "device": "cpu",
            "default_dtype": "float64",
            "return_node_energies": None,
        },
        script=EVAL_CONFIGS,
        check=False,
        capture_output=True,
        text=True,
    )
    assert completed.returncode != 0, (
        "--return_node_energies now handles structures of different sizes. "
        "That is the fix, not a regression: delete this test and widen "
        "test_eval_node_energies_sum_to_the_total_energy to the ragged "
        "fixture set."
    )
    assert "inhomogeneous shape" in (completed.stdout + completed.stderr), (
        "--return_node_energies failed on a ragged set for some reason other "
        "than the recorded one:\n" + (completed.stdout + completed.stderr)[-2000:]
    )
    assert not written.exists()


@pytest.mark.timeout(600)
def test_eval_contributions_sum_to_the_total_energy_on_the_plain_model(
    tmp_path, anchor_mace, fixture_file
):
    """`--return_contributions` asserted by value, on the class that has them.

    The per-body-order terms are a decomposition of the total energy, so they
    sum to it exactly. The extent of the array is not asserted: how many
    readouts a model has is architecture, not contract.
    """
    written = tmp_path / "contributions.xyz"
    run_eval(anchor_mace, fixture_file, written, return_contributions=None)

    for atoms in ase.io.read(written, index=":"):
        name = atoms.info["golden_name"]
        contributions = np.asarray(
            atoms.info["MACE_BO_contributions"], dtype=float
        ).ravel()
        assert contributions.size >= 2, name
        total = float(np.asarray(atoms.info["MACE_energy"]))
        assert contributions.sum() == pytest.approx(total, abs=TOL.atol), (
            f"{name}: the body-order contributions sum to "
            f"{contributions.sum():.12g} against a total of {total:.12g}"
        )


@pytest.mark.timeout(600)
def test_eval_contributions_are_refused_for_the_scale_shift_model(
    tmp_path, anchor_scaleshift, fixture_file
):
    """The other half of the flag's contract, and the half users hit.

    ``--return_contributions`` is documented as unsupported for
    ``ScaleShiftMACE`` -- which is the class the training CLI actually emits
    -- so the failure is the common case and it is pinned as one: the run
    must fail rather than write a file of something else.
    """
    written = tmp_path / "refused.xyz"
    completed = run_mace_train(
        {
            "configs": str(fixture_file),
            "model": str(anchor_scaleshift),
            "output": str(written),
            "device": "cpu",
            "default_dtype": "float64",
            "return_contributions": None,
        },
        script=EVAL_CONFIGS,
        check=False,
        capture_output=True,
        text=True,
    )
    assert completed.returncode != 0, (
        "--return_contributions succeeded on a ScaleShiftMACE; it is "
        "documented as unsupported there, so either the documentation or this "
        "contract is now wrong"
    )
    assert not written.exists(), "a refused evaluation still wrote an output file"


@pytest.mark.timeout(600)
def test_eval_descriptors_land_per_atom_and_the_aggregations_reduce_them(
    tmp_path, anchor_scaleshift, fixture_file
):
    """The three shapes ``--descriptor_aggregation_method`` produces.

    Pinned by relationship rather than by value: the two aggregations are
    means over the per-atom array, so asserting that they *are* those means
    pins the reduction without committing a second copy of the same numbers.
    """
    per_atom = tmp_path / "desc_per_atom.xyz"
    averaged = tmp_path / "desc_mean.xyz"
    run_eval(anchor_scaleshift, fixture_file, per_atom, return_descriptors=None)
    run_eval(
        anchor_scaleshift,
        fixture_file,
        averaged,
        return_descriptors=None,
        descriptor_aggregation_method="mean",
    )

    rows = {
        atoms.info["golden_name"]: np.asarray(atoms.arrays["MACE_descriptors"])
        for atoms in ase.io.read(per_atom, index=":")
    }
    for atoms in ase.io.read(averaged, index=":"):
        name = atoms.info["golden_name"]
        assert "MACE_descriptors" not in atoms.arrays, (
            f"{name}: an aggregated descriptor was still written per atom"
        )
        mean = np.asarray(atoms.info["MACE_descriptors"], dtype=float)
        assert rows[name].shape == (len(atoms), mean.size), name
        assert mean == pytest.approx(rows[name].mean(axis=0), abs=TOL.atol), name


@pytest.mark.timeout(600)
def test_eval_descriptor_layer_and_invariant_flags_change_the_width(
    tmp_path, anchor_scaleshift, fixture_file
):
    """`--descriptor_num_layers` / `--descriptor_invariants_only` do something.

    A width, not a value: what the columns *mean* is architecture, but a flag
    that silently returns the same block whatever it is set to is a flag that
    has stopped working, and that is what this catches.
    """
    widths = {}
    for label, flags in (
        ("all_layers_invariant", {}),
        ("one_layer_invariant", {"descriptor_num_layers": 1}),
        ("all_layers_equivariant", {"descriptor_invariants_only": ""}),
    ):
        written = tmp_path / f"desc_{label}.xyz"
        run_eval(
            anchor_scaleshift,
            fixture_file,
            written,
            return_descriptors=None,
            **flags,
        )
        atoms = ase.io.read(written, index=0)
        widths[label] = np.asarray(atoms.arrays["MACE_descriptors"]).shape[1]

    assert widths["one_layer_invariant"] < widths["all_layers_invariant"], widths
    assert widths["all_layers_equivariant"] > widths["all_layers_invariant"], widths


@pytest.mark.timeout(600)
def test_eval_at_float32_reproduces_float64_within_the_fp32_row(
    tmp_path, anchor_scaleshift, fixture_structures, fixture_file
):
    """`--default_dtype` is a precision knob, and the row that bounds it exists.

    The single-precision run is given a single-precision *checkpoint*: the
    flag sets the process dtype and does not touch the model, so the two have
    to be matched by the caller (the sibling test pins what happens when they
    are not).
    """
    import torch  # noqa: PLC0415

    float32_model = tmp_path / "anchor_float32.model"
    torch.save(
        torch.load(anchor_scaleshift, map_location="cpu", weights_only=False).float(),
        float32_model,
    )

    snapshots = {}
    for dtype, model in (
        ("float64", anchor_scaleshift),
        ("float32", float32_model),
    ):
        written = tmp_path / f"eval_{dtype}.xyz"
        run_mace_train(
            {
                "configs": str(fixture_file),
                "model": str(model),
                "output": str(written),
                "device": "cpu",
                "default_dtype": dtype,
            },
            script=EVAL_CONFIGS,
        )
        snapshots[dtype] = harness.snapshot_outputs(
            EvaluatedFile(written),
            fixture_structures,
            dtype=dtype,
            channels=["energy", "forces"],
        )
    harness.compare_to_reference(
        snapshots["float32"], snapshots["float64"], row=harness.FP32.name
    )


@pytest.mark.timeout(600)
def test_eval_refuses_a_dtype_that_disagrees_with_the_checkpoint(
    tmp_path, anchor_scaleshift, fixture_file
):
    """RECORDED DEFECT: the flag does not cast the model, and says so late.

    ``mace_eval_configs`` sets the process default dtype and then loads the
    checkpoint unchanged, so ``--default_dtype float32`` against a
    double-precision model builds single-precision graphs and feeds them to
    double-precision weights. The ase calculator does convert the model for
    exactly this reason; the CLI does not, and the failure surfaces as a
    dtype error from inside a scripted tensor product, several frames deep in
    somebody else's library, with nothing naming the flag that caused it.

    Pinned as it stands. A rewrite should either cast the model or refuse up
    front naming ``--default_dtype``; both are behaviour changes and both
    should come here and delete this test.
    """
    completed = run_mace_train(
        {
            "configs": str(fixture_file),
            "model": str(anchor_scaleshift),
            "output": str(tmp_path / "mismatched.xyz"),
            "device": "cpu",
            "default_dtype": "float32",
        },
        script=EVAL_CONFIGS,
        check=False,
        capture_output=True,
        text=True,
    )
    assert completed.returncode != 0, (
        "--default_dtype float32 now works on a float64 checkpoint. If the "
        "CLI casts the model, delete this test; if it silently produced "
        "numbers, find out in which precision before doing anything else."
    )
    assert "same dtype" in (completed.stdout + completed.stderr), (
        "the dtype mismatch failed differently than recorded:\n"
        + (completed.stdout + completed.stderr)[-2000:]
    )


@pytest.mark.timeout(600)
def test_eval_head_selects_a_head_and_refuses_one_the_model_does_not_have(
    tmp_path, anchor_scaleshift, fixture_file
):
    written = tmp_path / "head.xyz"
    run_eval(anchor_scaleshift, fixture_file, written, head="Default")
    assert written.exists()

    completed = run_mace_train(
        {
            "configs": str(fixture_file),
            "model": str(anchor_scaleshift),
            "output": str(tmp_path / "nohead.xyz"),
            "device": "cpu",
            "default_dtype": "float64",
            "head": "not_a_head",
        },
        script=EVAL_CONFIGS,
        check=False,
        capture_output=True,
        text=True,
    )
    assert completed.returncode != 0, (
        "--head accepted a head the model does not have, so a typo silently "
        "evaluates the wrong one"
    )


# ---------------------------------------------------------------------------
# 5. mace_select_head
#
# An installed entry point with no test at all before this file.
# ---------------------------------------------------------------------------


def run_select_head(model: Path, *argv) -> subprocess.CompletedProcess:
    env = os.environ.copy()
    env["PYTHONPATH"] = ":".join([str(REPO_ROOT)] + sys.path)
    return subprocess.run(
        [sys.executable, str(SELECT_HEAD), *[str(a) for a in argv], str(model)],
        env=env,
        check=False,
        capture_output=True,
        text=True,
    )


@pytest.mark.timeout(300)
def test_select_head_lists_the_heads_of_a_multihead_model(finetuned_multihead_model):
    completed = run_select_head(finetuned_multihead_model, "--list_heads")
    assert completed.returncode == 0, completed.stderr
    assert "Available heads:" in completed.stdout
    listed = {
        line.strip()
        for line in completed.stdout.splitlines()[1:]
        if line.strip()
    }
    assert {"pt_head", "Default"} <= listed, completed.stdout


@pytest.mark.timeout(300)
def test_select_head_writes_a_single_head_model_to_the_default_name(
    tmp_path, finetuned_multihead_model
):
    """No ``--output_file``: the name is derived, and the derivation is the
    contract a script that post-processes the artefact depends on."""
    working = tmp_path / "multihead.model"
    working.write_bytes(finetuned_multihead_model.read_bytes())

    completed = run_select_head(working, "--head_name", "Default")
    assert completed.returncode == 0, completed.stderr

    produced = tmp_path / "multihead.model.Default"
    assert produced.exists(), sorted(p.name for p in tmp_path.iterdir())

    import torch  # noqa: PLC0415

    selected = torch.load(produced, map_location="cpu", weights_only=False)
    assert list(selected.heads) == ["Default"], (
        f"the extracted model still carries {list(selected.heads)}"
    )


@pytest.mark.timeout(300)
def test_select_head_honours_output_file_and_target_device(
    tmp_path, finetuned_multihead_model
):
    working = tmp_path / "multihead.model"
    working.write_bytes(finetuned_multihead_model.read_bytes())
    destination = tmp_path / "just_default.model"

    completed = run_select_head(
        working,
        "--head_name",
        "Default",
        "--target_device",
        "cpu",
        "--output_file",
        destination,
    )
    assert completed.returncode == 0, completed.stderr
    assert destination.exists()
    assert not (tmp_path / "multihead.model.Default.cpu").exists(), (
        "--output_file was given and the derived name was written anyway"
    )

    import torch  # noqa: PLC0415

    selected = torch.load(destination, map_location="cpu", weights_only=False)
    assert list(selected.heads) == ["Default"]
    assert str(next(selected.parameters()).device) == "cpu"


@pytest.mark.timeout(300)
def test_select_head_and_the_multihead_model_agree_on_the_selected_head(
    tmp_path, finetuned_multihead_model, fixture_file
):
    """Extraction must not change the numbers of the head it extracted.

    The point of the entry point is to ship one head; if the extracted model
    predicts something else, everything downstream of it is wrong and nothing
    else in the tree would notice.
    """
    working = tmp_path / "multihead.model"
    working.write_bytes(finetuned_multihead_model.read_bytes())
    destination = tmp_path / "just_default.model"
    completed = run_select_head(
        working, "--head_name", "Default", "--output_file", destination
    )
    assert completed.returncode == 0, completed.stderr

    before = tmp_path / "before.xyz"
    after = tmp_path / "after.xyz"
    run_eval(working, fixture_file, before, head="Default")
    run_eval(destination, fixture_file, after)

    for original, extracted in zip(
        ase.io.read(before, index=":"), ase.io.read(after, index=":")
    ):
        name = original.info["golden_name"]
        assert float(np.asarray(extracted.info["MACE_energy"])) == pytest.approx(
            float(np.asarray(original.info["MACE_energy"])), abs=TOL.atol
        ), f"{name}: the extracted head does not reproduce the multihead model"
        assert np.abs(
            np.asarray(extracted.arrays["MACE_forces"])
            - np.asarray(original.arrays["MACE_forces"])
        ).max() <= TOL.atol, name


# ---------------------------------------------------------------------------
# 6. The ase calculator, and its graph padding
# ---------------------------------------------------------------------------


@pytest.mark.timeout(600)
def test_the_calculator_returns_the_contract_keys_with_the_right_shapes(
    anchor_scaleshift, fixture_structures
):
    from mace.calculators import MACECalculator  # noqa: PLC0415

    calculator = MACECalculator(
        model_paths=str(anchor_scaleshift), device="cpu", default_dtype="float64"
    )
    for name, atoms in fixture_structures.items():
        probe = atoms.copy()
        probe.calc = calculator
        energy = probe.get_potential_energy()
        forces = probe.get_forces()

        assert isinstance(energy, float), f"{name}: energy is {type(energy)}"
        assert forces.shape == (len(atoms), 3), name
        assert forces.dtype == np.float64, f"{name}: forces are {forces.dtype}"
        assert np.isfinite(forces).all(), name

        if harness.is_periodic(probe):
            stress = probe.get_stress(voigt=True)
            assert stress.shape == (6,), name
            assert stress.dtype == np.float64, name
            assert np.isfinite(stress).all(), (
                f"{name}: a periodic structure came back with a non-finite "
                f"stress, which is what a degenerate cell produces"
            )


@pytest.mark.timeout(600)
def test_the_calculator_agrees_with_eval_configs(
    tmp_path, anchor_scaleshift, fixture_structures, fixture_file
):
    """Stated separately from the eval-side test so a failure names the side.

    The same comparison as
    ``test_eval_configs_agrees_with_the_ase_calculator_on_every_fixture``
    extended to the stress, which only the periodic fixtures carry and which
    the two surfaces store in different layouts (Voigt-6 against 3x3) -- the
    harness reconciles that, which is the reason to route it through the
    harness rather than compare by hand.
    """
    from mace.calculators import MACECalculator  # noqa: PLC0415

    written = tmp_path / "evaluated_stress.xyz"
    run_eval(anchor_scaleshift, fixture_file, written, compute_stress=None)

    calculator = MACECalculator(
        model_paths=str(anchor_scaleshift), device="cpu", default_dtype="float64"
    )
    from_calculator = harness.snapshot_outputs(
        calculator, fixture_structures, channels=["energy", "forces", "stress"]
    )
    from_cli = harness.snapshot_outputs(
        EvaluatedFile(written),
        fixture_structures,
        channels=["energy", "forces", "stress"],
    )
    harness.compare_to_reference(from_cli, from_calculator, row=TOL.name)


def padded_versus_unpadded(anchor, fixtures, *, constructor=None, environment=None):
    """Snapshot the same structures with and without graph padding.

    Two calculators, built the same way except for the padding, and the
    padded one is the *second* so that a padding argument leaking into a
    class-level default would show up as the unpadded run changing rather
    than as agreement.
    """
    from mace.calculators import MACECalculator  # noqa: PLC0415

    plain = MACECalculator(
        model_paths=str(anchor), device="cpu", default_dtype="float64"
    )
    unpadded = harness.snapshot_outputs(
        plain, fixtures, channels=["energy", "forces", "stress"]
    )

    previous = {}
    try:
        for key, value in (environment or {}).items():
            previous[key] = os.environ.get(key)
            os.environ[key] = value
        padded_calculator = MACECalculator(
            model_paths=str(anchor),
            device="cpu",
            default_dtype="float64",
            **(constructor or {}),
        )
    finally:
        for key, value in previous.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value

    padded = harness.snapshot_outputs(
        padded_calculator, fixtures, channels=["energy", "forces", "stress"]
    )
    return padded, unpadded, padded_calculator


@pytest.mark.timeout(600)
def test_padding_through_the_constructor_does_not_change_a_single_number(
    anchor_scaleshift, fixture_structures
):
    """The invariant the whole padding feature rests on.

    Every padding edge is a self-loop on the last fake atom with a shift of
    twice the cutoff, so its length is at least ``r_max`` and the polynomial
    cutoff annihilates it exactly. If that ever stops being exact the padded
    calculator starts returning slightly different physics depending on how
    much padding was requested, which is the worst possible failure mode --
    it looks like noise. Asserted against an *unpadded* run, not against a
    second padded one.
    """
    padded, unpadded, calculator = padded_versus_unpadded(
        anchor_scaleshift,
        fixture_structures,
        constructor={"pad_num_atoms": 24, "pad_num_edges": 256},
    )
    assert calculator.pad_num_atoms == 24
    assert calculator.pad_num_edges == 256
    harness.compare_to_reference(padded, unpadded, row=TOL.name)


@pytest.mark.timeout(600)
def test_padding_through_the_two_environment_variables_behaves_identically(
    anchor_scaleshift, fixture_structures
):
    """The env fallback is a shipped surface and is pinned as one.

    It exists so a deployment can turn padding on without touching the code
    that builds the calculator, which means nothing in the calling program
    reveals that it is on -- so if the two routes ever diverge, the
    environment route is the one nobody would look at.
    """
    padded, unpadded, calculator = padded_versus_unpadded(
        anchor_scaleshift,
        fixture_structures,
        environment={
            "MACE_ASE_PAD_NUM_ATOMS": "24",
            "MACE_ASE_PAD_NUM_EDGES": "256",
        },
    )
    assert calculator.pad_num_atoms == 24, (
        "MACE_ASE_PAD_NUM_ATOMS did not reach the calculator"
    )
    assert calculator.pad_num_edges == 256, (
        "MACE_ASE_PAD_NUM_EDGES did not reach the calculator"
    )
    harness.compare_to_reference(padded, unpadded, row=TOL.name)


@pytest.mark.timeout(600)
def test_a_constructor_argument_wins_over_the_environment_variable(
    anchor_scaleshift, fixture_structures
):
    """Precedence, because the fallback only reads as a fallback if it loses.

    The implementation consults the environment only when the argument is
    non-positive, so an explicit request cannot be overridden by a variable
    somebody exported in a shell three layers up.
    """
    padded, unpadded, calculator = padded_versus_unpadded(
        anchor_scaleshift,
        fixture_structures,
        constructor={"pad_num_atoms": 12, "pad_num_edges": 128},
        environment={
            "MACE_ASE_PAD_NUM_ATOMS": "999",
            "MACE_ASE_PAD_NUM_EDGES": "9999",
        },
    )
    assert (calculator.pad_num_atoms, calculator.pad_num_edges) == (12, 128), (
        "the environment overrode an explicit constructor argument"
    )
    harness.compare_to_reference(padded, unpadded, row=TOL.name)


@pytest.mark.timeout(600)
def test_padded_per_atom_arrays_come_back_with_exactly_len_atoms_rows(
    anchor_scaleshift, fixture_structures
):
    """The slicing half of the contract, stated on its own.

    The equivalence tests above would also fail if the padding were left in
    the outputs -- but they would fail on a shape check inside the harness,
    which says the schema was disappointed rather than that the calculator
    handed a caller rows for atoms that do not exist.
    """
    from mace.calculators import MACECalculator  # noqa: PLC0415

    calculator = MACECalculator(
        model_paths=str(anchor_scaleshift),
        device="cpu",
        default_dtype="float64",
        pad_num_atoms=32,
        pad_num_edges=512,
        compute_atomic_stresses=True,
    )
    for name, atoms in fixture_structures.items():
        probe = atoms.copy()
        probe.calc = calculator
        probe.get_potential_energy()
        probe.get_forces()
        for key, value in calculator.results.items():
            array = np.asarray(value)
            if array.ndim >= 1 and array.shape[0] not in (1, 3, 6, 9):
                assert array.shape[0] == len(atoms), (
                    f"{name}: the padded calculator returned {array.shape[0]} "
                    f"rows of {key!r} for {len(atoms)} atoms"
                )
        assert np.asarray(calculator.results["forces"]).shape == (len(atoms), 3), name


@pytest.mark.timeout(600)
def test_an_edge_budget_smaller_than_the_structure_is_raised_not_truncated(
    anchor_scaleshift, fixture_structures
):
    """A padding budget is a floor, never a ceiling.

    The one way padding could silently change physics is by dropping real
    edges to fit a budget. It does not -- the budget is raised and the run
    continues -- and this pins that, because "the numbers moved a little on
    the largest structure" is not a failure anybody would trace back here.
    """
    padded, unpadded, calculator = padded_versus_unpadded(
        anchor_scaleshift,
        fixture_structures,
        constructor={"pad_num_atoms": 1, "pad_num_edges": 1},
    )
    harness.compare_to_reference(padded, unpadded, row=TOL.name)
    assert calculator.pad_num_edges > 1, (
        "the edge budget was never raised, so either no fixture exceeded one "
        "edge or the budget is being enforced by dropping edges"
    )
