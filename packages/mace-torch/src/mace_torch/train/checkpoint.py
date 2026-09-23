"""Writing a run down so it can be picked up again, with no pickle anywhere.

Two artifacts, and they are different things.

The **model** is what every downstream consumer loads: canonical weights and
the model record, written by :mod:`mace_torch.serialization`. When the run
averages its weights, the model holds the average, as the frozen tree's does.

The **run checkpoint** is what an interrupted run continues from: the raw
canonical weights, never the average; the optimizer's state, the schedule's and
the average's own shadow; and the counters, which are the next epoch, the best
loss and its epoch, how many evaluations have passed since, and the stage. The
tensors go in safetensors and everything else in a JSON sidecar, so reading one
back executes nothing.

**A checkpoint is found by what it says, not by its name.** Each run checkpoint
is two files named for the run and its epoch, but the newest is the one whose
sidecar records the highest epoch; nothing parses a filename. The sidecar is
written last and renamed into place, so an interrupted write leaves either the
previous checkpoint or none, never half of one.

**A resume is complete or it is an error**, with one exception that is reported
rather than hidden: when the optimizer's parameter groups no longer match the
ones the checkpoint was written with, the weights load and the optimizer and
schedule start afresh, and :class:`ResumeResult` says so and why. A checkpoint
that cannot be read is refused, never skipped in favour of an older one.
"""

from __future__ import annotations

import json
import os
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

from mace_core.metadata import ModelMetadata
from safetensors import SafetensorError
from safetensors.torch import load_file, save_file
from torch import Tensor, nn
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LRScheduler

from mace_torch.serialization import (
    CheckpointError,
    canonical_state,
    load_canonical_state,
    save_checkpoint,
)
from mace_torch.train.ema import ExponentialMovingAverage

__all__ = [
    "RUN_FORMAT",
    "ResumeResult",
    "RunState",
    "latest_run_checkpoint",
    "read_run_checkpoint",
    "read_run_state",
    "retain_run_checkpoints",
    "write_model",
    "write_run_checkpoint",
]

#: What a run checkpoint's sidecar says it is.
RUN_FORMAT = "mace-v1-run-checkpoint"

#: Bumped when a field's meaning changes. A reader refuses any other.
RUN_VERSION = 1

_SEPARATOR = "::"


@dataclass(frozen=True)
class RunState:
    """Where a run had got to.

    Attributes:
        epoch: The next epoch to run, so a resume starts here rather than
            repeating the one that was written.
        best_valid_loss: The lowest validation loss seen, or ``None`` if the
            run had not evaluated yet.
        best_epoch: Which epoch that was.
        since_best: Evaluations since the best one, which is what patience
            counts. Restarting it at zero would give a resumed run more
            patience than the one it continues.
        stage: The stage the last epoch trained in.
        improved: Whether the epoch that wrote this checkpoint improved on the
            best loss, which is what the retention rules read.
    """

    epoch: int
    best_valid_loss: float | None = None
    best_epoch: int | None = None
    since_best: int = 0
    stage: str = ""
    improved: bool = False


@dataclass(frozen=True)
class ResumeResult:
    """What a resume restored, and what it could not.

    Attributes:
        state: The counters, as the checkpoint recorded them.
        path: The sidecar it was read from.
        optimizer_state: ``"restored"``, or ``"reinitialized"`` when the
            optimizer's parameter groups no longer match the checkpoint's. The
            schedule follows the optimizer: its state is meaningless for an
            optimizer that starts afresh.
        reason: Why the optimizer was reinitialized, or ``None``.
    """

    state: RunState
    path: Path
    optimizer_state: Literal["restored", "reinitialized"]
    reason: str | None = None


def write_model(path: str | Path, model: nn.Module, metadata: ModelMetadata) -> Path:
    """The weights and the record, as one checkpoint.

    The whole resolved configuration goes in the sidecar, because a checkpoint
    that carried only weights would need its run's command line to be rebuilt
    and that is the thing least likely to still exist.
    """
    return save_checkpoint(path, model, metadata.model_dump(mode="json"))


# ---------------------------------------------------------------------------
# Nested state as JSON plus tensors
# ---------------------------------------------------------------------------


def _encode(value: Any, key: str, tensors: dict[str, Tensor]) -> Any:
    """``value`` as JSON, with each tensor moved into ``tensors`` under a key.

    Optimizer and schedule states are nested dicts of numbers, lists and
    tensors, keyed partly by integers. Each shape is written so it reads back
    as itself: a tuple stays a tuple and an integer key stays an integer.

    Raises:
        CheckpointError: On anything else, such as a function a schedule holds.
            Pickling it is the alternative, and that is what this replaces.
    """
    if isinstance(value, Tensor):
        tensors[key] = value.detach().cpu().contiguous()
        return {"tensor": key}
    if value is None or isinstance(value, (bool, int, str)):
        return value
    if isinstance(value, float):
        # JSON has no infinity, and a schedule's best loss starts at one.
        return (
            {"float": repr(value)}
            if value != value or abs(value) == float("inf")
            else value
        )
    if isinstance(value, (list, tuple)):
        items = [
            _encode(item, f"{key}.{index}", tensors) for index, item in enumerate(value)
        ]
        return {"tuple": items} if isinstance(value, tuple) else items
    if isinstance(value, Mapping):
        return {
            "mapping": [
                [
                    _encode(name, f"{key}.key", tensors),
                    _encode(item, f"{key}.{name}", tensors),
                ]
                for name, item in value.items()
            ]
        }
    raise CheckpointError(
        f"the run's state holds a {type(value).__name__} at {key}, which has no "
        f"representation but a pickle. A schedule or optimizer built around a "
        f"function cannot be checkpointed this way."
    )


def _decode(value: Any, tensors: Mapping[str, Tensor]) -> Any:
    if isinstance(value, list):
        return [_decode(item, tensors) for item in value]
    if isinstance(value, dict):
        if "tensor" in value:
            return tensors[value["tensor"]]
        if "float" in value:
            return float(value["float"])
        if "tuple" in value:
            return tuple(_decode(item, tensors) for item in value["tuple"])
        if "mapping" in value:
            return {
                _decode(name, tensors): _decode(item, tensors)
                for name, item in value["mapping"]
            }
        raise CheckpointError(f"a run checkpoint holds an unknown entry {value}.")
    return value


def _topology(optimizer: Optimizer) -> list[list[list[int]]]:
    """The shape of every parameter in every group, which is what an
    optimizer's saved state is indexed against."""
    return [
        [list(parameter.shape) for parameter in group["params"]]
        for group in optimizer.param_groups
    ]


# ---------------------------------------------------------------------------
# Writing and finding
# ---------------------------------------------------------------------------


def _files(directory: Path, stem: str) -> tuple[Path, Path]:
    """The tensors and the sidecar of one checkpoint. Joined as strings: a
    stem like ``model.run-000003`` has a dot in it, and treating what follows
    as a suffix would write every epoch over the same file."""
    return directory / f"{stem}.safetensors", directory / f"{stem}.json"


def _tensors_beside(sidecar: Path) -> Path:
    return sidecar.with_name(sidecar.name.removesuffix(".json") + ".safetensors")


def write_run_checkpoint(
    directory: str | Path,
    name: str,
    state: RunState,
    *,
    model: nn.Module,
    optimizer: Optimizer,
    scheduler: LRScheduler | None = None,
    ema: ExponentialMovingAverage | None = None,
    metadata: ModelMetadata | None = None,
) -> Path:
    """Write the run, raw weights included, and return its sidecar.

    Called with the raw weights in place, never inside the average's context:
    the average travels as its own state, and a checkpoint holding averaged
    weights as the model would resume training from parameters the optimizer
    never stepped.
    """
    root = Path(directory)
    root.mkdir(parents=True, exist_ok=True)
    tensors: dict[str, Tensor] = {}
    for module, values in canonical_state(model).items():
        for tensor_name, value in values.items():
            tensors[f"model{_SEPARATOR}{module}{_SEPARATOR}{tensor_name}"] = (
                value.detach().cpu().contiguous()
            )
    document: dict[str, Any] = {
        "format": RUN_FORMAT,
        "version": RUN_VERSION,
        "name": name,
        "state": {
            "epoch": state.epoch,
            "best_valid_loss": _encode(state.best_valid_loss, "state", tensors),
            "best_epoch": state.best_epoch,
            "since_best": state.since_best,
            "stage": state.stage,
            "improved": state.improved,
        },
        "model_tensors": sorted(key for key in tensors if key.startswith("model")),
        "optimizer": {
            "topology": _topology(optimizer),
            "state": _encode(optimizer.state_dict(), "optimizer", tensors),
        },
        "scheduler": None
        if scheduler is None
        else _encode(scheduler.state_dict(), "scheduler", tensors),
        "ema": None if ema is None else _encode(ema.state_dict(), "ema", tensors),
        "config": None if metadata is None else metadata.model_dump(mode="json"),
    }
    stem = f"{name}.run-{state.epoch:06d}"
    weights, sidecar = _files(root, stem)
    staged_weights, staged_sidecar = _files(root, f"{stem}.partial")
    save_file(tensors, str(staged_weights))
    staged_sidecar.write_text(json.dumps(document, indent=1, sort_keys=True))
    # Tensors first, sidecar last: the sidecar is what makes a checkpoint
    # visible, so until it is in place the previous one is the newest.
    os.replace(staged_weights, weights)
    os.replace(staged_sidecar, sidecar)
    return sidecar


def _read_sidecar(path: Path) -> dict[str, Any]:
    try:
        document = json.loads(path.read_text())
    except (OSError, ValueError) as error:
        raise CheckpointError(
            f"{path} cannot be read as a run checkpoint ({error}). It is refused "
            f"rather than skipped: resuming from an older one instead would "
            f"silently repeat the epochs between them."
        ) from error
    if document.get("format") != RUN_FORMAT or document.get("version") != RUN_VERSION:
        raise CheckpointError(
            f"{path} is {document.get('format')!r} version "
            f"{document.get('version')!r}, and this reads {RUN_FORMAT!r} version "
            f"{RUN_VERSION}."
        )
    return document


def _candidates(directory: Path, name: str) -> list[tuple[int, Path, dict[str, Any]]]:
    found = []
    for path in sorted(directory.glob(f"{name}.run-*.json")):
        if path.name.endswith(".partial.json"):
            continue
        document = _read_sidecar(path)
        if document.get("name") != name:
            continue
        found.append((int(document["state"]["epoch"]), path, document))
    return found


def latest_run_checkpoint(directory: str | Path, name: str) -> Path | None:
    """The newest run checkpoint for ``name``, by the epoch its sidecar records.

    Raises:
        CheckpointError: If any of the run's checkpoints cannot be read.
    """
    found = _candidates(Path(directory), name)
    return max(found, key=lambda entry: entry[0])[1] if found else None


def read_run_state(path: str | Path) -> RunState:
    """The counters a run checkpoint records, without touching any model."""
    recorded = _read_sidecar(Path(path))["state"]
    return RunState(
        epoch=int(recorded["epoch"]),
        best_valid_loss=_decode(recorded["best_valid_loss"], {}),
        best_epoch=recorded["best_epoch"],
        since_best=int(recorded["since_best"]),
        stage=str(recorded["stage"]),
        improved=bool(recorded["improved"]),
    )


def read_run_checkpoint(
    path: str | Path,
    *,
    model: nn.Module,
    optimizer: Optimizer,
    scheduler: LRScheduler | None = None,
    ema: ExponentialMovingAverage | None = None,
) -> ResumeResult:
    """Put a run back: weights, optimizer, schedule, average and counters.

    The optimizer and schedule given are the ones the resumed run will step,
    already built for the stage it resumes in, since a stage that names its
    own optimizer builds a new one and would otherwise replace what this
    restores.

    Raises:
        CheckpointError: If the files disagree, the model and the checkpoint
            hold different operators, or the saved run and this one disagree
            about averaging.
    """
    sidecar = Path(path)
    document = _read_sidecar(sidecar)
    try:
        tensors = load_file(str(_tensors_beside(sidecar)))
    except (OSError, ValueError, RuntimeError, SafetensorError) as error:
        raise CheckpointError(
            f"{_tensors_beside(sidecar)} cannot be read ({error})."
        ) from error
    missing = [key for key in document["model_tensors"] if key not in tensors]
    if missing:
        raise CheckpointError(
            f"{sidecar} declares model tensors the file lacks: {missing[:5]}."
        )

    weights: dict[str, dict[str, Tensor]] = {}
    for key in document["model_tensors"]:
        _, module, tensor_name = key.split(_SEPARATOR)
        weights.setdefault(module, {})[tensor_name] = tensors[key]
    load_canonical_state(model, weights)

    if (ema is None) != (document["ema"] is None):
        raise CheckpointError(
            "the saved run and this one disagree about whether the weights are "
            "averaged. Resuming either way continues from parameters the other "
            "half of the run never used."
        )
    if ema is not None:
        ema.load_state_dict(_decode(document["ema"], tensors))

    saved = document["optimizer"]["topology"]
    current = _topology(optimizer)
    outcome: Literal["restored", "reinitialized"] = "restored"
    reason = None
    if saved != current:
        outcome = "reinitialized"
        reason = (
            f"the optimizer's parameter groups changed since the checkpoint "
            f"was written ({len(saved)} group(s) of sizes "
            f"{[len(group) for group in saved]} then, {len(current)} of sizes "
            f"{[len(group) for group in current]} now), so the weights were "
            f"loaded and the optimizer and schedule start afresh"
        )
    else:
        optimizer.load_state_dict(_decode(document["optimizer"]["state"], tensors))
        if scheduler is not None and document["scheduler"] is not None:
            scheduler.load_state_dict(_decode(document["scheduler"], tensors))
    return ResumeResult(
        state=read_run_state(sidecar),
        path=sidecar,
        optimizer_state=outcome,
        reason=reason,
    )


def retain_run_checkpoints(
    directory: str | Path, name: str, *, keep_improving: bool, keep_all: bool
) -> list[Path]:
    """Delete the run checkpoints the retention rules do not keep.

    The newest is always kept. ``keep_all`` keeps every one; ``keep_improving``
    keeps those written by an epoch that improved, which is what the frozen
    tree's keep flag keeps, since it writes a checkpoint only then; neither
    keeps the newest alone.

    Returns:
        The sidecars deleted.
    """
    if keep_all:
        return []
    found = sorted(_candidates(Path(directory), name), key=lambda entry: entry[0])
    deleted = []
    for _, path, document in found[:-1]:
        if keep_improving and document["state"]["improved"]:
            continue
        # Sidecar first: without it the tensors are invisible, so an
        # interruption here leaves an orphan file and never a half checkpoint.
        path.unlink()
        _tensors_beside(path).unlink(missing_ok=True)
        deleted.append(path)
    return deleted
