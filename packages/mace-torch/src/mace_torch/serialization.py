"""Reading and writing a model, without pickle and without a state dict.

A checkpoint is two files side by side: the tensors in safetensors, and a JSON
sidecar naming what they are. Both are the point.

**No pickle.** A legacy checkpoint is a pickled module, which is why the frozen
tree has to disable PyTorch's own safety flag to load one. Reading one means
executing whatever it contains, so a model file is code. Here the tensors are a
format a reader can validate and the sidecar is data, so a checkpoint from a
stranger is a file rather than a program.

**No state dict either.** A ``state_dict`` is a picture of one backend's module
tree: rename a submodule and it stops loading, swap the backend and it never
did. What is written instead is each operator's **canonical** form, which the
backend defines and every backend agrees on, so a model trained with one set of
kernels loads into another.

**The sidecar also carries the layout.** Tensors alone say how many weights
there are, never which path each one multiplies, and the path order of the
symmetric contraction is a free choice that two implementations can make
differently. So each operator that has one writes its own layout description
beside the tensors, and a load compares it against the model being loaded into.
A disagreement then names the paths it disagreed about instead of producing a
model that runs and means something else.

The sidecar's ``config`` is opaque here on purpose. What this module owns is
the format: the versioning, the tensor naming and the dispatch to each
operator's own loader. What the configuration means belongs to the schema that
defines it, and a format that also knew the schema would have to change
whenever the schema did.
"""

from __future__ import annotations

import json
from collections.abc import Callable, Mapping
from pathlib import Path
from typing import Any

import torch
from safetensors.torch import load_file, save_file
from torch import Tensor, nn

__all__ = [
    "FORMAT",
    "VERSION",
    "CheckpointError",
    "canonical_metadata",
    "canonical_state",
    "check_layout",
    "load_canonical_state",
    "load_checkpoint",
    "save_checkpoint",
]

#: What the sidecar says it is, so a reader can refuse someone else's file
#: rather than misreading it.
FORMAT = "mace-v1-checkpoint"

#: Bumped when the meaning of an existing field changes. A reader refuses a
#: version it was not written for, because guessing at a format is how a model
#: loads and computes something else. Version 2 added ``layout``, and a version
#: 1 file carries no record of which basis path each weight multiplies, so it
#: cannot be validated rather than merely lacking a field.
VERSION = 2

#: Separates a module's path from the name its own canonical form gives a
#: tensor. A dot would be ambiguous against the module path itself.
_SEPARATOR = "::"


class CheckpointError(RuntimeError):
    """Raised instead of loading something that would be wrong."""


def canonical_state(model: nn.Module) -> dict[str, dict[str, Tensor]]:
    """Every operator's canonical tensors, keyed by where it sits.

    Args:
        model: Anything holding operators that define ``to_canonical``.

    Returns:
        ``module path -> name -> tensor``. Modules without a canonical form
        contribute nothing, which is how a module that holds only derived
        buffers stays out of the file.
    """
    state: dict[str, dict[str, Tensor]] = {}
    for path, module in model.named_modules():
        writer = getattr(module, "to_canonical", None)
        if writer is None:
            continue
        state[path] = {name: value.detach() for name, value in writer().items()}
    return state


def load_canonical_state(
    model: nn.Module, state: Mapping[str, Mapping[str, Tensor]]
) -> None:
    """Put canonical tensors back, dispatching to each operator's own loader.

    Raises:
        CheckpointError: If the model and the file disagree about which
            operators exist. Loading the intersection would give a model that
            runs with some of its weights left at their initial values.
    """
    expected = set(canonical_state(model))
    found = set(state)
    if expected != found:
        missing = sorted(expected - found)
        extra = sorted(found - expected)
        raise CheckpointError(
            f"the checkpoint and this model do not hold the same operators. "
            f"Missing from the file: {missing or 'none'}. Present in the file "
            f"and not in the model: {extra or 'none'}. A partial load leaves "
            f"weights at their initial values and still runs."
        )
    for path, module in model.named_modules():
        reader = getattr(module, "load_canonical", None)
        if reader is None:
            continue
        reader(dict(state[path]))


def canonical_metadata(model: nn.Module) -> dict[str, dict[str, Any]]:
    """Every operator's description of its own layout, keyed by where it sits.

    An operator opts in by defining ``canonical_metadata``. One that does not
    contributes nothing, so an operator whose tensors carry their meaning in
    their shape alone costs no sidecar entry.

    Args:
        model: Anything holding operators.

    Returns:
        ``module path -> a JSON-serializable description``.
    """
    described: dict[str, dict[str, Any]] = {}
    for path, module in model.named_modules():
        writer = getattr(module, "canonical_metadata", None)
        if writer is None:
            continue
        described[path] = dict(writer())
    return described


def check_layout(model: nn.Module, recorded: Mapping[str, Mapping[str, Any]]) -> None:
    """Refuse a file whose layout is not the one this model reads.

    Args:
        model: The model about to be loaded into.
        recorded: What the sidecar's ``layout`` holds.

    Only operators the two have in common are compared. A model and a file that
    hold different operators at all is
    :func:`load_canonical_state`'s refusal, and it runs before any weight is
    written, so stating it twice here would only give the same fault two
    different messages.

    Raises:
        CheckpointError: If a shared operator's description differs. The
            message names the first few paths that differ rather than only
            reporting that something did, because the whole reason the layout
            is recorded is that a path count can match while the paths do not.
    """
    expected = canonical_metadata(model)
    for path in sorted(set(expected) & set(recorded)):
        description = expected[path]
        stored = dict(recorded[path])
        if stored == description:
            continue
        for key, value in description.items():
            if stored.get(key) == value:
                continue
            if isinstance(value, list) and isinstance(stored.get(key), list):
                differing = [
                    f"position {position}: file {theirs!r}, model {mine!r}"
                    for position, (theirs, mine) in enumerate(
                        zip(stored[key], value, strict=False)
                    )
                    if theirs != mine
                ]
                detail = "; ".join(differing[:3]) or (
                    f"the file holds {len(stored[key])} and the model {len(value)}"
                )
            else:
                detail = f"file {stored.get(key)!r}, model {value!r}"
            raise CheckpointError(
                f"{path} was written in a different layout: {key} disagrees "
                f"({detail}). The weights would load and multiply the wrong "
                f"basis paths."
            )


def save_checkpoint(
    path: str | Path, model: nn.Module, config: Mapping[str, Any]
) -> Path:
    """Write the tensors and the sidecar.

    Args:
        path: Where the tensors go. The sidecar is written beside it with a
            ``.json`` suffix.
        model: The model to write.
        config: Whatever a reader needs to rebuild the model before the tensors
            can be put back. Stored verbatim; this module does not interpret it.

    Returns:
        The sidecar's path, since that is the half a reader has to find first.
    """
    tensors = Path(path).with_suffix(".safetensors")
    sidecar = tensors.with_suffix(".json")
    state = canonical_state(model)

    flat: dict[str, Tensor] = {}
    entries = []
    for module_path in sorted(state):
        names = sorted(state[module_path])
        for name in names:
            value = state[module_path][name].contiguous()
            flat[f"{module_path}{_SEPARATOR}{name}"] = value
            entries.append(
                {
                    "module": module_path,
                    "name": name,
                    "shape": list(value.shape),
                    "dtype": str(value.dtype).removeprefix("torch."),
                }
            )

    tensors.parent.mkdir(parents=True, exist_ok=True)
    save_file(flat, str(tensors))
    sidecar.write_text(
        json.dumps(
            {
                "format": FORMAT,
                "version": VERSION,
                "config": dict(config),
                "layout": canonical_metadata(model),
                "tensors": entries,
            },
            indent=2,
            sort_keys=True,
        )
    )
    return sidecar


def read_sidecar(path: str | Path) -> dict[str, Any]:
    """The sidecar, checked for being one of ours.

    Raises:
        CheckpointError: If the file is not this format, or is a version this
            reader was not written for.
    """
    sidecar = Path(path).with_suffix(".json")
    if not sidecar.exists():
        raise CheckpointError(
            f"no sidecar at {sidecar}. A checkpoint is two files: the tensors "
            f"and the JSON that says what they are, and the tensors alone do "
            f"not say how to rebuild the model."
        )
    document = json.loads(sidecar.read_text())
    if document.get("format") != FORMAT:
        raise CheckpointError(
            f"{sidecar} says its format is {document.get('format')!r} and this "
            f"reader only reads {FORMAT!r}."
        )
    if document.get("version") != VERSION:
        raise CheckpointError(
            f"{sidecar} is version {document.get('version')!r} and this reader "
            f"was written for {VERSION}. Reading it anyway would guess at what "
            f"changed."
        )
    return document


def load_checkpoint(
    path: str | Path, build: Callable[[dict[str, Any]], nn.Module]
) -> nn.Module:
    """Rebuild a model and put its weights back.

    Args:
        path: Either half of the pair; the suffix is replaced as needed.
        build: Turns the sidecar's ``config`` into an empty model. The format
            does not know what a configuration means, so the caller says.

    Returns:
        The built model, with every operator's weights loaded.

    Raises:
        CheckpointError: If the file is not this format, if the tensors and the
            sidecar disagree, or if the model and the file hold different
            operators.
    """
    source = Path(path)
    if source.suffix in {".pt", ".pth", ".ckpt"}:
        raise CheckpointError(
            f"{source} looks like a pickled checkpoint. Reading one executes "
            f"whatever it contains, which is why it is not a format this reads. "
            f"Convert it first."
        )
    document = read_sidecar(source)
    flat = load_file(str(source.with_suffix(".safetensors")))

    declared = {
        f"{entry['module']}{_SEPARATOR}{entry['name']}" for entry in document["tensors"]
    }
    if declared != set(flat):
        raise CheckpointError(
            f"the sidecar declares {len(declared)} tensor(s) and the file holds "
            f"{len(flat)}. Undeclared: {sorted(set(flat) - declared)}. Declared "
            f"and absent: {sorted(declared - set(flat))}."
        )

    state: dict[str, dict[str, Tensor]] = {}
    for entry in document["tensors"]:
        value = flat[f"{entry['module']}{_SEPARATOR}{entry['name']}"]
        wanted = getattr(torch, entry["dtype"])
        if value.dtype != wanted:
            raise CheckpointError(
                f"{entry['module']}.{entry['name']} is {value.dtype} in the "
                f"file and the sidecar says {wanted}."
            )
        state.setdefault(entry["module"], {})[entry["name"]] = value

    model = build(dict(document["config"]))
    check_layout(model, document.get("layout", {}))
    load_canonical_state(model, state)
    return model
