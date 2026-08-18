"""Train the ``ScaleShiftMACE`` parity anchor.

Unlike its plain-``MACE`` sibling this anchor *is* produced by the training
CLI, because ``ScaleShiftMACE`` is exactly what the CLI emits and pinning the
class a user actually gets is the point. The command is built here as an
explicit argv list and recorded verbatim in the sidecar next to the
checkpoint, so the recipe is a fact about the committed file rather than a
paragraph that can drift away from it.

The training set is the committed ``fixtures/tiny_train.xyz``: three isolated
atoms carrying the reference energies for H, C and O, plus 24 rattled copies
of a small periodic cell with seeded synthetic labels. It is not physical and
does not need to be -- what it has to be is stable, small, and enough to move
the weights off their initialisation.
"""

from __future__ import annotations

import json
import os
import re
import shlex
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Dict, List

GOLDEN_ROOT = Path(__file__).resolve().parent
REPO_ROOT = GOLDEN_ROOT.parent.parent
MODELS_DIR = GOLDEN_ROOT / "models"
FIXTURES_DIR = GOLDEN_ROOT / "fixtures"
REFERENCES_DIR = GOLDEN_ROOT / "references"

MODEL_PATH = MODELS_DIR / "tiny_scaleshift.model"
SIDECAR_PATH = MODELS_DIR / "tiny_scaleshift.build.json"
ERRORS_PATH = REFERENCES_DIR / "tiny_scaleshift_training_errors.json"

NAME = "tiny_scaleshift"
SEED = 20260810

#: The training flags, in the order they are passed. `pair_repulsion` is a
#: bare flag (value None); everything else is `--key=value`.
TRAIN_ARGS: Dict[str, object] = {
    "name": NAME,
    "model": "MACE",
    "train_file": "tests/golden/fixtures/tiny_train.xyz",
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
    "num_radial_basis": 8,
    "num_cutoff_basis": 5,
    "num_interactions": 2,
    "correlation": 3,
    "hidden_irreps": "16x0e + 16x1o",
    "MLP_irreps": "8x0e",
    "pair_repulsion": None,
    # See build_mace_anchor.py: the default is True and degrades silently to
    # False when cuequivariance is absent, which would make the committed
    # weights depend on what is installed on the machine that trained them.
    "use_reduced_cg": False,
    "batch_size": 4,
    "valid_batch_size": 4,
    "max_num_epochs": 8,
    "eval_interval": 1,
    "lr": 0.01,
    "device": "cpu",
    "default_dtype": "float64",
    "seed": SEED,
    "error_table": "PerAtomRMSE",
    "save_cpu": None,
}


def build_argv(work_dir: Path) -> List[str]:
    """The exact command, as an argv list, that produces the anchor."""
    argv = [sys.executable, str(REPO_ROOT / "mace" / "cli" / "run_train.py")]
    for key, value in TRAIN_ARGS.items():
        argv.append(f"--{key}" if value is None else f"--{key}={value}")
    for directory in ("model_dir", "checkpoints_dir", "results_dir", "log_dir"):
        argv.append(f"--{directory}={work_dir}")
    return argv


_TABLE_ROW = re.compile(r"^\|(?P<cells>.*)\|\s*$")


def parse_error_table(log: str) -> Dict[str, Dict[str, float]]:
    """Extract the final train/valid error table from the training log.

    The legacy stack renders this table and nothing else: there is no
    machine-readable copy of the per-set errors anywhere on disk. Parsing the
    rendered table is therefore not a shortcut around a structured artifact,
    it is the only way to record the thing GATE-3 compares against.
    """
    marker = "Error-table on TRAIN and VALID:"
    if marker not in log:
        raise RuntimeError(
            "the training log does not contain the final error table; the run "
            "did not reach the evaluation stage"
        )
    block = log.split(marker, 1)[1]
    header: List[str] = []
    rows: Dict[str, Dict[str, float]] = {}
    for line in block.splitlines():
        match = _TABLE_ROW.match(line.strip())
        if match is None:
            if rows:
                break
            continue
        cells = [cell.strip() for cell in match.group("cells").split("|")]
        if not header:
            header = cells
            continue
        label = cells[0]
        values: Dict[str, float] = {}
        for name, cell in zip(header[1:], cells[1:]):
            try:
                values[name] = float(cell)
            except ValueError:
                values[name] = cell
        rows[label] = values
    if not rows:
        raise RuntimeError("found the error-table marker but no parseable rows")
    return rows


def read_final_metrics(work_dir: Path) -> Dict[str, object]:
    """The last evaluation record the metrics logger wrote (final loss & co)."""
    candidates = sorted(work_dir.glob("*_train.txt"))
    if not candidates:
        return {}
    records = []
    for line in candidates[-1].read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            records.append(json.loads(line))
        except json.JSONDecodeError:
            continue
    evals = [r for r in records if r.get("mode") == "eval"]
    if not evals:
        return {}
    # `time` is a wall-clock measurement, not a metric: keeping it would make
    # every regeneration produce a different file and turn a genuine golden
    # diff into noise nobody reads.
    return {k: v for k, v in evals[-1].items() if k != "time"}


def train_anchor(model_path: Path = MODEL_PATH) -> Path:
    """Run the training, install the checkpoint and write both sidecars."""
    with tempfile.TemporaryDirectory(prefix="golden_anchor_") as tmp:
        work_dir = Path(tmp)
        argv = build_argv(work_dir)
        env = os.environ.copy()
        env["PYTHONPATH"] = os.pathsep.join(
            [str(REPO_ROOT), env.get("PYTHONPATH", "")]
        ).strip(os.pathsep)
        completed = subprocess.run(
            argv,
            cwd=REPO_ROOT,
            env=env,
            check=True,
            capture_output=True,
            text=True,
        )
        log = completed.stdout + completed.stderr
        produced = work_dir / f"{NAME}.model"
        if not produced.exists():
            raise RuntimeError(f"training did not write {produced}\n{log[-4000:]}")
        model_path.parent.mkdir(parents=True, exist_ok=True)
        model_path.write_bytes(produced.read_bytes())

        error_table = parse_error_table(log)
        final_metrics = read_final_metrics(work_dir)

    # The recorded command must be reproducible, so the four output
    # directories come back as a placeholder: their real value is a
    # per-invocation temporary path, and leaving it in would make the sidecar
    # differ on every regeneration for no reason anyone could act on.
    printable = [
        "python" if index == 0 else "mace/cli/run_train.py" if index == 1 else arg
        for index, arg in enumerate(argv)
    ]
    printable = [
        f"{arg.split('=', 1)[0]}=<work_dir>"
        if arg.startswith(("--model_dir=", "--checkpoints_dir=", "--results_dir=", "--log_dir="))
        else arg
        for arg in printable
    ]
    sidecar = {
        "model": model_path.name,
        "class": "ScaleShiftMACE",
        "recipe": "tests/golden/train_anchor.py",
        # shell-quoted, because two of the flags carry spaces and a command
        # that cannot be pasted is not a recipe
        "command": " ".join(shlex.quote(arg) for arg in printable),
        "argv": printable,
        "regenerate_with": (
            "python tests/golden/regenerate.py --target anchors "
            "--i-know-what-i-am-doing"
        ),
        "built_by": "mace/cli/run_train.py (the CLI class for --model MACE)",
        "note": (
            "--model MACE yields a ScaleShiftMACE with atomic_inter_scale "
            "set from the dataset std and the shift zeroed "
            "(mace/tools/model_script_utils.py:279-296)."
        ),
        "seed": SEED,
        "dtype": "float64",
        "train_file": "tests/golden/fixtures/tiny_train.xyz",
        "args": {k: ("<flag>" if v is None else v) for k, v in TRAIN_ARGS.items()},
    }
    with SIDECAR_PATH.open("w", encoding="utf-8") as handle:
        json.dump(sidecar, handle, indent=2, sort_keys=True)
        handle.write("\n")

    errors = {
        "schema_version": 1,
        "description": (
            "Final train/valid error table of the ScaleShiftMACE anchor "
            "training, as rendered by mace/tools/tables_utils.create_error_table, "
            "plus the last evaluation record written by the metrics logger. "
            "Committed so a rewrite's training run can be compared against "
            "the legacy stack's own numbers on the same tiny set."
        ),
        "recipe": "tests/golden/train_anchor.py",
        "model": model_path.name,
        "error_table_type": TRAIN_ARGS["error_table"],
        "units": {
            "energy": "meV (per atom, as the table renders it)",
            "forces": "meV/Ang",
        },
        "error_table": error_table,
        "final_eval_record": final_metrics,
    }
    ERRORS_PATH.parent.mkdir(parents=True, exist_ok=True)
    with ERRORS_PATH.open("w", encoding="utf-8") as handle:
        json.dump(errors, handle, indent=2, sort_keys=True)
        handle.write("\n")
    return model_path


if __name__ == "__main__":  # pragma: no cover - manual invocation
    print(train_anchor())
