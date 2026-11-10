"""Foundation identities, metadata and alias migration, without a framework.

The roster records the legacy source artifacts. This registry records native
release candidates and publication separately: a missing artifact reference is
an explicit publication dependency, never a guessed URL or a pickle fallback.
"""

from __future__ import annotations

import json
import logging
import re
import warnings
from functools import lru_cache
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, model_validator

from mace_core.artifacts import ArtifactRef
from mace_core.config.model import ModelConfig
from mace_core.metadata import ModelMetadata

logger = logging.getLogger(__name__)
FoundationFamily = Literal["mace_mp", "mace_off", "mace_omol", "mace_mdp", "mace_polar"]
# These published names have no size tier. The FM-3 naming exception preserves
# them explicitly instead of assigning an invented medium/large classification.
_UNSIZED_IDS = frozenset({"mace-mh-0", "mace-mh-1", "mace-mdp"})

ASL_NOTICE = (
    "Using model under Academic Software License (ASL) license, see "
    "https://github.com/gabor1/ASL \n To use this model you accept the "
    "terms of the license."
)


class Recommendation(BaseModel):
    """The declared chemistry and configuration scope, not a support badge."""

    model_config = ConfigDict(extra="forbid", frozen=True)
    systems: str
    level_of_theory: str
    limitations: str
    source: str


class ModelLicense(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    identifier: Literal["MIT", "ASL"]
    source: str

    def notice(self) -> str | None:
        return ASL_NOTICE if self.identifier == "ASL" else None


class RegistryEntry(BaseModel):
    """A native foundation candidate; model metadata uses the shared schema.

    ``artifact_ref`` and the model DOI remain absent until publication. Paper
    DOIs belong to ``metadata.citations`` and never stand in for an artifact DOI.
    Hyperparameters are the resolved model config in ``metadata``. E0 tensors
    and training data are not duplicated in this catalogue.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)
    model_id: str
    family: FoundationFamily
    aliases: tuple[str, ...]
    default: bool = False
    default_dtype: Literal["float32", "float64"]
    artifact_ref: ArtifactRef | None = None
    license: ModelLicense
    status: Literal["recommended", "deprecated"]
    recommendation: Recommendation
    metadata: ModelMetadata
    golden_refs: tuple[str, ...]
    badge_ref: str

    @model_validator(mode="after")
    def _validate(self) -> RegistryEntry:
        if self.model_id not in _UNSIZED_IDS and not re.fullmatch(
            r"mace-(?:[a-z0-9]+-)+(?:small|medium|large|extra-large)", self.model_id
        ):
            raise ValueError(
                f"{self.model_id!r}: use mace-<data>-<size> canonical naming"
            )
        if not self.aliases or len(set(self.aliases)) != len(self.aliases):
            raise ValueError(f"{self.model_id}: aliases must be nonempty and unique")
        ModelConfig.model_validate(self.metadata.config.resolved["model"])
        if any(head.e0 is not None for head in self.metadata.heads.values()):
            raise ValueError(
                f"{self.model_id}: registry summaries must not duplicate E0s"
            )
        return self

    def hyperparameters(self) -> ModelConfig:
        return ModelConfig.model_validate(self.metadata.config.resolved["model"])


class UnpublishedModelError(ValueError):
    """A known candidate has no published native checkpoint yet."""


class RetiredModelError(ValueError):
    """An old model has an explicit replacement instead of a loader."""


def retired_models() -> dict[str, str]:
    """The two released ANI-CC spellings and their retained replacement."""
    return {"mace_anicc": "mace-off23-medium", "mace-anicc": "mace-off23-medium"}


@lru_cache(maxsize=1)
def _entries() -> tuple[RegistryEntry, ...]:
    data = json.loads(Path(__file__).with_name("model_registry.json").read_text())
    entries = tuple(RegistryEntry.model_validate(row) for row in data)
    ids = [entry.model_id for entry in entries]
    aliases = [(entry.family, alias) for entry in entries for alias in entry.aliases]
    if len(set(ids)) != len(ids) or len(set(aliases)) != len(aliases):
        raise ValueError("foundation registry has duplicate canonical IDs or aliases")
    for family in ("mace_mp", "mace_off", "mace_omol", "mace_mdp", "mace_polar"):
        defaults = [
            entry for entry in entries if entry.family == family and entry.default
        ]
        if len(defaults) != (0 if family == "mace_polar" else 1):
            raise ValueError(f"{family}: wrong number of default registry models")
    return entries


def list_models() -> tuple[RegistryEntry, ...]:
    """Independent records, so a caller cannot mutate another caller's metadata."""
    return tuple(entry.model_copy(deep=True) for entry in _entries())


def family_default_dtype(family: FoundationFamily) -> str:
    values = {entry.default_dtype for entry in _entries() if entry.family == family}
    if len(values) != 1:
        raise ValueError(f"{family}: registry must declare one default dtype")
    return values.pop()


def get_model(model_id: str) -> RegistryEntry:
    if model_id in retired_models():
        raise RetiredModelError(
            f"{model_id} is retired; use {retired_models()[model_id]}"
        )
    for entry in _entries():
        if entry.model_id == model_id:
            return entry.model_copy(deep=True)
    raise KeyError(f"unknown foundation model {model_id!r}; use list_models()")


def resolve_model(model: str | None, *, family: FoundationFamily) -> RegistryEntry:
    """Resolve canonical names or legacy aliases; announce defaults and aliases."""
    if model in retired_models():
        raise RetiredModelError(f"{model} is retired; use {retired_models()[model]}")
    candidates = [entry for entry in _entries() if entry.family == family]
    for entry in candidates:
        if (
            model == entry.model_id
            or model in entry.aliases
            or (model is None and entry.default)
        ):
            if model is None:
                logger.info("%s selected default model %s", family, entry.model_id)
            elif model != entry.model_id:
                warnings.warn(
                    f"{family} alias {model!r} is deprecated; use {entry.model_id!r}",
                    RuntimeWarning,
                    stacklevel=2,
                )
            if entry.status == "deprecated":
                warnings.warn(
                    f"{entry.model_id} is deprecated: "
                    f"{entry.recommendation.limitations}",
                    RuntimeWarning,
                    stacklevel=2,
                )
            return entry.model_copy(deep=True)
    if model is None:
        raise ValueError(f"{family} requires an explicit model")
    raise KeyError(
        f"{model!r} is not a {family} model; use a canonical ID, a "
        f"native checkpoint path or an https URL"
    )


def published_reference(entry: RegistryEntry) -> ArtifactRef:
    if entry.artifact_ref is None:
        raise UnpublishedModelError(
            f"{entry.model_id}: native artifact publication is pending; "
            "pass a converted local .json/.safetensors checkpoint"
        )
    return entry.artifact_ref


def export_registry(path: str | Path) -> Path:
    """Export the same records and labels used at runtime for docs/website use."""
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(
            {
                "models": [entry.model_dump(mode="json") for entry in list_models()],
                "retired": retired_models(),
                "support": (
                    "A recommendation is not a support badge. Publication, "
                    "a committed golden and green conformance are required."
                ),
            },
            indent=2,
        )
        + "\n"
    )
    return output
