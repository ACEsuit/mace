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
from collections.abc import Mapping
from pathlib import Path
from typing import Any, TypeVar

from mace_core.metadata import ModelMetadata
from mace_core.weights import CHECKPOINT_FORMAT as FORMAT
from mace_core.weights import CHECKPOINT_VERSION as VERSION
from mace_core.weights.checkpoint import (
    CheckpointError,
    CheckpointManifest,
    read_checkpoint_manifest,
    validate_layout,
    validate_tensors,
)
from safetensors.torch import load_file, save_file
from torch import Tensor, nn

from mace_torch.canonical import canonical_modules

#: Keep the concrete module type when loading weights into an existing model.
Model = TypeVar("Model", bound=nn.Module)


__all__ = [
    "FORMAT",
    "VERSION",
    "CheckpointError",
    "canonical_metadata",
    "canonical_state",
    "check_layout",
    "load_canonical_state",
    "load_checkpoint",
    "read_canonical_state",
    "save_checkpoint",
    "write_model",
]

# Raw linear/skip weights, canonical layout and logical component paths are
# part of the initial format. Unpublished intermediate formats are not read.

#: Separates a module's path from the name its own canonical form gives a
#: tensor. A dot would be ambiguous against the module path itself.
_SEPARATOR = "::"


def canonical_state(model: nn.Module) -> dict[str, dict[str, Tensor]]:
    """Every operator's canonical tensors, keyed by where it sits.

    Args:
        model: Anything holding operators that define ``to_canonical``.

    Returns:
        ``logical operator identity -> name -> tensor``. Modules without a
        canonical form contribute nothing, which is how a module with only derived
        buffers stays out of the file.
    """
    state: dict[str, dict[str, Tensor]] = {}
    for path, module in canonical_modules(model).items():
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
    for path, module in canonical_modules(model).items():
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
        ``logical operator identity -> a JSON-serializable description``.
    """
    described: dict[str, dict[str, Any]] = {}
    for path, module in canonical_modules(model).items():
        writer = getattr(module, "canonical_metadata", None)
        if writer is None:
            continue
        described[path] = dict(writer())
    return described


def check_layout(model: nn.Module, recorded: Mapping[str, Mapping[str, Any]]) -> None:
    """Require the complete canonical layout before installing any weights."""
    validate_layout(canonical_metadata(model), recorded)


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
    # Configuration mapping order assigns head indices to weight rows.
    # Sorting JSON keys would silently relabel a multi-head model on reload.
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
        )
    )
    return sidecar


def load_checkpoint(
    path: str | Path,
    model: Model,
    *,
    manifest: CheckpointManifest | None = None,
    tensors_path: str | Path | None = None,
) -> Model:
    """Validate and restore weights into an already constructed model.

    ``manifest`` reuses a sidecar already read to construct the model. The
    layout and every tensor are checked before weights are installed.
    ``tensors_path`` supports cached pairs whose URL digests differ.
    """
    document = manifest or read_checkpoint_manifest(_refuse_pickle(path))
    check_layout(model, document.layout)
    load_canonical_state(
        model,
        read_canonical_state(path, manifest=document, tensors_path=tensors_path),
    )
    return model


def _refuse_pickle(path: str | Path) -> Path:
    source = Path(path)
    if source.suffix in {".pt", ".pth", ".ckpt"}:
        raise CheckpointError(
            f"{source} looks like a pickled checkpoint. Reading one executes "
            f"whatever it contains, which is why it is not a format this reads. "
            f"Convert it first."
        )
    return source


def read_canonical_state(
    path: str | Path,
    *,
    manifest: CheckpointManifest | None = None,
    tensors_path: str | Path | None = None,
) -> dict[str, dict[str, Tensor]]:
    """The tensors of a checkpoint, by operator, checked against the sidecar.

    For a caller that already holds a model and wants a written state back in
    it, which is what a run does to end on its best epoch.

    Raises:
        CheckpointError: If the file is not this format, or the tensors and
            the sidecar disagree.
    """
    source = _refuse_pickle(path)
    document = manifest or read_checkpoint_manifest(source)
    flat = load_file(str(tensors_path or source.with_suffix(".safetensors")))

    validate_tensors(
        document,
        {
            name: (value.shape, str(value.dtype).removeprefix("torch."))
            for name, value in flat.items()
        },
    )
    state: dict[str, dict[str, Tensor]] = {}
    for entry in document.tensors:
        state.setdefault(entry.module, {})[entry.name] = flat[entry.key]

    return state


def write_model(path: str | Path, model: nn.Module, metadata: ModelMetadata) -> Path:
    """The weights and the record, as one checkpoint.

    The whole resolved configuration goes in the sidecar, because a checkpoint
    that carried only weights would need its run's command line to be rebuilt
    and that is the thing least likely to still exist.
    """
    from mace_torch.physics.fixed_point import FixedPointDriver

    if isinstance(model, FixedPointDriver):
        if metadata.config.resolved.get("fixed_point") != model.spec.model_dump(
            mode="json"
        ):
            raise ValueError(
                "the checkpoint configuration must record "
                "the fixed-point driver being saved"
            )
        underlying = model.engine
        if not isinstance(underlying, nn.Module):
            raise TypeError("a saved fixed-point engine must be a torch module")
        model = underlying
    return save_checkpoint(path, model, metadata.model_dump(mode="json"))
