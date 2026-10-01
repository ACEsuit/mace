"""`ModelMetadata`: the record every trained v1 model carries about how it was
made, stored as JSON beside the weights."""

from __future__ import annotations

import json
from collections.abc import Iterable
from typing import Any, Final

from pydantic import BaseModel, ConfigDict, Field, model_validator

from mace_core.config import BaseConfig

__all__ = [
    "SCHEMA_VERSION",
    "Citation",
    "ConfigRecord",
    "DataSourceSummary",
    "DataSummary",
    "HeadSummary",
    "MetadataSchemaError",
    "ModelMetadata",
    "Provenance",
    "format_citations",
]

#: Bump when a field is added, removed or changes meaning.
SCHEMA_VERSION: Final = 1


class MetadataSchemaError(ValueError):
    """Metadata this code cannot read back: an unknown schema version, or a
    value JSON would change."""


class _Record(BaseModel):
    """Common ground: unknown keys are errors, so a typo cannot be stored."""

    # inf/nan are written as JSON constants (Infinity, NaN) rather than
    # pydantic's default null, which would turn a value into a different one.
    model_config = ConfigDict(extra="forbid", ser_json_inf_nan="constants")


class ConfigRecord(_Record):
    """Record of the training config, as written and as resolved."""

    user: dict[str, Any] = Field(default_factory=dict)
    resolved: dict[str, Any] = Field(default_factory=dict)

    @classmethod
    def from_config(cls, config: BaseConfig) -> ConfigRecord:
        return cls(user=config.to_user_dict(), resolved=config.to_resolved_dict())


class Provenance(_Record):
    """Which code produced the model."""

    #: Version per distribution used, `{"mace-core": "1.0.2", "mace-torch": "1.1.0"}`
    versions: dict[str, str]
    #: Full hash of the commit the code was run from; None when not in a checkout.
    git_commit: str | None = None


class DataSourceSummary(_Record):
    """Summary of one data source."""

    #: The data source's name in the config.
    name: str
    num_configurations: int | None = None
    num_atoms: int | None = None
    #: Atomic numbers of every element present.
    elements: list[int] = Field(default_factory=list)
    #: The reference quantities provided, each prefixed by the method that
    #: produced it: `pbe_energy`, `pbe_forces`, `r2scan_energy`.
    reference_keys: list[str] = Field(default_factory=list)


class DataSummary(_Record):
    """One summary per data source, each source once."""

    sources: list[DataSourceSummary] = Field(default_factory=list)


class HeadSummary(_Record):
    """What one head was fitted on."""

    #: Names from `DataSummary.sources`.
    sources: list[str] = Field(default_factory=list)


class Citation(_Record):
    """One work users of the model are asked to cite."""

    title: str
    authors: list[str] = Field(default_factory=list)
    venue: str | None = None
    year: int | None = None
    doi: str | None = None
    url: str | None = None


class ModelMetadata(_Record):
    """How a model was made: its config, the code, the data and what to cite."""

    schema_version: int = SCHEMA_VERSION
    config: ConfigRecord
    provenance: Provenance
    data: DataSummary = Field(default_factory=DataSummary)
    #: Keyed by head name, as in the config; a single-head model has one entry.
    heads: dict[str, HeadSummary] = Field(default_factory=dict)
    #: DOI of the model itself, not of the papers describing it.
    doi: str | None = None
    citations: list[Citation] = Field(default_factory=list)
    notes: str = ""

    @model_validator(mode="after")
    def _heads_name_known_sources(self) -> ModelMetadata:
        names = [source.name for source in self.data.sources]
        if len(set(names)) != len(names):
            raise ValueError(f"data.sources names a source twice: {sorted(names)}")
        for head, summary in self.heads.items():
            for name in summary.sources:
                if name not in names:
                    raise ValueError(
                        f"heads.{head}.sources names {name!r}, which is not in "
                        f"data.sources; add its summary or drop the name"
                    )
        return self

    def to_json(self, indent: int | None = 2) -> str:
        """Serialise to JSON, checking that it reads back as the same record."""
        text = self.model_dump_json(indent=indent)
        if self.from_json(text) != self:
            raise MetadataSchemaError(
                "model metadata does not survive a JSON round trip: a field holds "
                "a value JSON reads back differently, such as a tuple (a list "
                "on the way back) or NaN (never equal to itself)"
            )
        return text

    @classmethod
    def from_json(cls, text: str) -> ModelMetadata:
        """Read a record written by `to_json`."""
        document = json.loads(text)
        # Checked before validation: a newer record may hold fields this code
        # does not know, and the version is the error worth reporting.
        version = document.get("schema_version")
        if version != SCHEMA_VERSION:
            raise MetadataSchemaError(
                f"model metadata has schema_version {version!r}, but this "
                f"mace-core reads schema_version {SCHEMA_VERSION}; a newer "
                "record needs an upgrade of mace-core"
            )
        return cls.model_validate(document)


def format_citations(citations: Iterable[Citation]) -> str:
    """Render citations as a numbered block, one line each; unset fields are
    left out."""
    lines = []
    for number, citation in enumerate(citations, start=1):
        parts = []
        if citation.authors:
            parts.append(", ".join(citation.authors))
        parts.append(citation.title)
        if citation.venue and citation.year:
            parts.append(f"{citation.venue} ({citation.year})")
        elif citation.venue:
            parts.append(citation.venue)
        elif citation.year:
            parts.append(str(citation.year))
        if citation.doi:
            parts.append(f"https://doi.org/{citation.doi}")
        elif citation.url:
            parts.append(citation.url)
        lines.append(f"[{number}] " + ". ".join(parts))
    return "\n".join(lines)
