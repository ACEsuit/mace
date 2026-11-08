"""The native checkpoint manifest shared by framework-specific tensor readers."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Annotated, Any, Literal

from pydantic import BaseModel, ConfigDict, Field, ValidationError, model_validator


class CheckpointError(RuntimeError):
    """A checkpoint cannot be loaded without guessing what its tensors mean."""


class TensorEntry(BaseModel):
    """The name, dimensions and storage dtype of one canonical tensor."""

    model_config = ConfigDict(extra="forbid", frozen=True)
    module: str
    name: Annotated[str, Field(min_length=1)]
    shape: tuple[Annotated[int, Field(strict=True, ge=0)], ...]
    dtype: Annotated[str, Field(min_length=1)]

    @property
    def key(self) -> str:
        return f"{self.module}::{self.name}"


class CheckpointManifest(BaseModel):
    """Required native format fields; the model configuration is opaque here."""

    model_config = ConfigDict(extra="forbid", frozen=True)
    format: Literal["mace-v1-checkpoint"]
    version: Literal["1.0"]
    config: dict[str, Any]
    layout: dict[str, dict[str, Any]]
    tensors: tuple[TensorEntry, ...]

    @model_validator(mode="after")
    def _unique_tensors(self) -> CheckpointManifest:
        seen = set()
        for entry in self.tensors:
            if entry.key in seen:
                raise ValueError(f"duplicate canonical tensor {entry.key}")
            seen.add(entry.key)
        return self


def read_checkpoint_manifest(path: str | Path) -> CheckpointManifest:
    """Read either checkpoint stem's JSON, refusing missing or malformed fields."""
    sidecar = Path(path).with_suffix(".json")
    try:
        contents = sidecar.read_text()
    except OSError as error:
        raise CheckpointError(
            f"no readable sidecar at {sidecar}: {error}. A checkpoint needs two files: "
            "the tensors and their JSON manifest."
        ) from error
    try:
        return CheckpointManifest.model_validate_json(contents)
    except ValidationError as error:
        raise CheckpointError(
            f"{sidecar}: invalid checkpoint manifest: {error}"
        ) from error


def validate_tensors(
    manifest: CheckpointManifest,
    tensors: Mapping[str, tuple[Sequence[int], str]],
) -> None:
    """Check actual tensor shapes/dtype names before a framework installs weights."""
    declared = {entry.key for entry in manifest.tensors}
    if declared != set(tensors):
        raise CheckpointError(
            f"Undeclared canonical tensors: {sorted(set(tensors) - declared)}. "
            f"Missing canonical tensors: {sorted(declared - set(tensors))}."
        )
    for entry in manifest.tensors:
        shape, dtype = tensors[entry.key]
        if tuple(shape) != entry.shape or dtype != entry.dtype:
            raise CheckpointError(
                f"{entry.key}: dtype or shape disagrees with the sidecar; "
                f"file {tuple(shape)} {dtype}, manifest {entry.shape} {entry.dtype}"
            )


def validate_layout(
    expected: Mapping[str, Mapping[str, Any]],
    recorded: Mapping[str, Mapping[str, Any]],
) -> None:
    """Require every layout and every field, including operators without weights."""
    if set(expected) != set(recorded):
        raise CheckpointError(
            "canonical layout modules disagree with the architecture; "
            f"missing {sorted(set(expected) - set(recorded))}, "
            f"unsupported {sorted(set(recorded) - set(expected))}"
        )
    for path, description in expected.items():
        stored = recorded[path]
        if stored != description:
            keys = sorted(
                key
                for key in set(stored) | set(description)
                if key not in stored
                or key not in description
                or stored[key] != description[key]
            )
            raise CheckpointError(
                f"{path} was written in a different layout: {', '.join(keys)} disagrees"
            )
