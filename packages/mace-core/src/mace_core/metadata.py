"""`ModelMetadata`: the record every trained v1 model carries about how it was
made, stored as JSON beside the weights."""

from __future__ import annotations

import re
from collections.abc import Iterable
from typing import Annotated, Any, Final, Literal

from pydantic import BaseModel, ConfigDict, Field, ValidationInfo, model_validator

from mace_core.config import BaseConfig
from mace_core.kernels.descriptors import ClebschGordanBasis

__all__ = [
    "SCHEMA_VERSION",
    "Citation",
    "ClebschGordanConversion",
    "ConfigRecord",
    "ConversionRecord",
    "DataSourceSummary",
    "DataSummary",
    "HeadSummary",
    "MetadataSchemaError",
    "ModelMetadata",
    "ObservableErrors",
    "ParityGate",
    "Provenance",
    "StageSummary",
    "TrainingSummary",
    "format_citations",
]

#: The schema version this code writes, `"<major>.<minor>"`. A minor bump only
#: adds fields, each with a default; renaming or removing a field, or changing
#: what one means, is a major bump. A record is read when it has this major and
#: this minor or an older one.
SCHEMA_VERSION: Final = "1.0"

_SCHEMA_VERSION_PATTERN: Final = re.compile(r"(0|[1-9][0-9]*)\.(0|[1-9][0-9]*)")

#: Validation context key set by `ModelMetadata.from_json`, the route that
#: reads a record rather than builds one, and so requires its version.
_READING: Final = "mace_core.metadata.reading"


class MetadataSchemaError(ValueError):
    """Metadata that would not read back from its JSON as the same record."""


def _parse_schema_version(version: object) -> tuple[int, int] | None:
    """`(major, minor)` of a version string, or None when it is not one."""
    if not isinstance(version, str):
        return None
    match = _SCHEMA_VERSION_PATTERN.fullmatch(version)
    if match is None:
        return None
    return int(match[1]), int(match[2])


def _check_schema_version(version: object, *, present: bool = True) -> None:
    """Raise `ValueError` unless this code reads a record of `version`.

    `present` is False when the record has no `schema_version` key at all.
    Every message names the record's version, the version this code writes and
    the installed mace-core release."""
    # Here rather than at the top: `mace_core` imports this module.
    from mace_core import __version__

    major, minor = (int(part) for part in SCHEMA_VERSION.split("."))
    reads = (
        f"mace-core {__version__} reads schema_version {SCHEMA_VERSION} and the "
        f"older minors of major {major}"
    )
    if not present:
        raise ValueError(
            f"model metadata has no schema_version; {reads}. A record written "
            "by to_json always carries one, so this one was not written by "
            "mace-core or was edited"
        )
    record = _parse_schema_version(version)
    if record is None:
        raise ValueError(
            f"model metadata has schema_version {version!r}, which is not a "
            f"'<major>.<minor>' string; {reads}"
        )
    if record[0] < major:
        raise ValueError(
            f"model metadata has schema_version {version!r}, of an older major "
            f"than this code; {reads}. Records are not migrated across majors: "
            "reconvert the model with this mace-core"
        )
    if record[0] > major or record[1] > minor:
        raise ValueError(
            f"model metadata has schema_version {version!r}, newer than this "
            f"code; {reads}. Upgrade mace-core to a release that reads "
            f"schema_version {version}"
        )


class _Record(BaseModel):
    """Common ground: unknown keys are errors, so a typo cannot be stored."""

    # inf/nan are written as JSON constants (Infinity, NaN) rather than
    # pydantic's default null, which would turn a value into a different one.
    model_config = ConfigDict(extra="forbid", ser_json_inf_nan="constants")


#: A name or a version string. An empty one names nothing, so it is refused.
_Name = Annotated[str, Field(min_length=1)]

#: Version per distribution, `{"mace-core": "1.0.2", "mace-torch": "1.1.0"}`.
_DistributionVersions = dict[_Name, _Name]

#: A deviation or an error: a magnitude, so never negative, and finite, since an
#: infinite one is a failure that should not have been recorded as a result.
_Magnitude = Annotated[float, Field(ge=0, allow_inf_nan=False)]


class ConfigRecord(_Record):
    """Record of the training config, as written and as resolved."""

    user: dict[str, Any] = Field(default_factory=dict)
    resolved: dict[str, Any] = Field(default_factory=dict)

    @classmethod
    def from_config(cls, config: BaseConfig) -> ConfigRecord:
        return cls(user=config.to_user_dict(), resolved=config.to_resolved_dict())


class Provenance(_Record):
    """Which code produced the model."""

    #: Version per distribution used.
    versions: _DistributionVersions
    #: Full hash of the commit the code was run from; None when not in a checkout.
    git_commit: str | None = None


class DataSourceSummary(_Record):
    """Summary of one data source."""

    #: The data source's name in the config.
    name: _Name
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


class ClebschGordanConversion(_Record):
    """The Clebsch-Gordan basis a conversion read its weights in and the one it
    wrote them in."""

    read: ClebschGordanBasis
    written: ClebschGordanBasis

    @model_validator(mode="after")
    def _converts_something(self) -> ClebschGordanConversion:
        if self.read == self.written:
            raise ValueError(
                f"the Clebsch-Gordan basis was read and written as {self.read!r}, "
                "which converts nothing; record clebsch_gordan_conversion as None"
            )
        return self


class ParityGate(_Record):
    """The parity check a converter ran between the source and the converted
    model."""

    #: Name of the tolerance row the deviations were held to.
    tolerance_row: _Name
    #: Largest absolute difference between the two models, per quantity
    #: compared, in that quantity's units.
    max_absolute_deviation: dict[_Name, _Magnitude] = Field(min_length=1)


class ConversionRecord(_Record):
    """How a model converted from a legacy checkpoint was converted."""

    #: What the file read was, such as `"mace-torch-pickle"`.
    source_format: _Name
    #: Hex SHA-256 of the file read.
    source_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    #: Distribution versions found in the source, `mace-torch` among them.
    source_versions: _DistributionVersions
    #: Distribution versions that ran the conversion. They stay with the model
    #: when later work replaces `provenance`, such as a fine-tuning run.
    converter_versions: _DistributionVersions = Field(min_length=1)
    #: True when the source had no head names and the converter assigned one.
    headless: bool
    #: None when the weights kept the basis they were read in.
    clebsch_gordan_conversion: ClebschGordanConversion | None = None
    parity_gate: ParityGate

    @model_validator(mode="after")
    def _source_names_mace_torch(self) -> ConversionRecord:
        if "mace-torch" not in self.source_versions:
            raise ValueError(
                "conversion.source_versions has no 'mace-torch' entry; a legacy "
                "checkpoint was written by mace-torch, so record its version"
            )
        return self


class StageSummary(_Record):
    """How one training stage ended."""

    #: The stage's name in the config.
    name: _Name
    #: Epochs actually run, fewer than the config allows when the stage
    #: stopped early.
    epochs: int = Field(ge=0)
    stop_reason: Literal["max_epochs", "patience", "converged", "interrupted"]


class ObservableErrors(_Record):
    """The final validation errors of one observable, in its declared units."""

    rmse: _Magnitude
    mae: _Magnitude


class TrainingSummary(_Record):
    """How the run that produced the model ended.

    What the run was asked to do is the config's, in `config.resolved`; this
    holds only what happened. The units of an error are those of its
    observable's declaration there."""

    #: One entry per stage, in the order they ran.
    stages: list[StageSummary] = Field(min_length=1)
    #: Keyed by head name, then by observable name.
    validation_errors: dict[_Name, dict[_Name, ObservableErrors]] = Field(
        default_factory=dict
    )

    @model_validator(mode="after")
    def _stages_named_once(self) -> TrainingSummary:
        names = [stage.name for stage in self.stages]
        if len(set(names)) != len(names):
            raise ValueError(f"training.stages names a stage twice: {names}")
        return self


class ModelMetadata(_Record):
    """How a model was made: its config, the code, the data and what to cite."""

    #: Always `SCHEMA_VERSION` once read: a record of an older minor is read
    #: into this schema, its missing fields set to their defaults.
    schema_version: str = SCHEMA_VERSION
    config: ConfigRecord
    provenance: Provenance
    data: DataSummary = Field(default_factory=DataSummary)
    #: Keyed by head name, as in the config; a single-head model has one entry.
    heads: dict[_Name, HeadSummary] = Field(default_factory=dict)
    #: DOI of the model itself, not of the papers describing it.
    doi: str | None = None
    citations: list[Citation] = Field(default_factory=list)
    notes: str = ""
    #: Set only on a model converted from a legacy checkpoint.
    conversion: ConversionRecord | None = None
    #: Set only on a model produced by a training run.
    training: TrainingSummary | None = None

    @model_validator(mode="before")
    @classmethod
    def _reads_a_schema_version_it_knows(cls, data: Any, info: ValidationInfo) -> Any:
        # Before the fields are read: a newer record may hold fields this code
        # does not know, and the version is the error worth reporting. A record
        # read from JSON must carry its version; one built in code without it
        # is being written, and takes the current one.
        if isinstance(data, dict):
            reading = bool(info.context and info.context.get(_READING))
            present = "schema_version" in data
            if present or reading:
                _check_schema_version(data.get("schema_version"), present=present)
            data = {**data, "schema_version": SCHEMA_VERSION}
        return data

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

    @model_validator(mode="after")
    def _training_names_known_heads(self) -> ModelMetadata:
        if self.training is not None:
            for head in self.training.validation_errors:
                if head not in self.heads:
                    raise ValueError(
                        f"training.validation_errors names head {head!r}, which "
                        "is not in heads; add its summary or drop the errors"
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
        """Read a record written by `to_json`. Unlike building one, reading
        requires `schema_version`."""
        return cls.model_validate_json(text, context={_READING: True})


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
