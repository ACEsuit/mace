"""`ModelMetadata`: the record every trained v1 model carries about how it was
made, stored as JSON beside the weights."""

from __future__ import annotations

from collections.abc import Iterable
from typing import Annotated, Any, Final, Literal

from pydantic import BaseModel, ConfigDict, Field, ValidationInfo, model_validator

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
    "ObservableErrors",
    "Provenance",
    "StageSummary",
    "TrainingSummary",
    "format_citations",
]

#: Only this schema is read and written while V1 is under development.
SCHEMA_VERSION: Final = "1.2"

#: Validation context key set by `ModelMetadata.from_json`, the route that
#: reads a record rather than builds one, and so requires its version.
_READING: Final = "mace_core.metadata.reading"


class MetadataSchemaError(ValueError):
    """Metadata that would not read back from its JSON as the same record."""


def _check_schema_version(version: object, *, present: bool = True) -> None:
    """Require the current schema before validating the rest of a record."""
    from mace_core import __version__

    reads = f"mace-core {__version__} reads schema_version {SCHEMA_VERSION}"
    if not present:
        raise ValueError(f"model metadata has no schema_version; {reads}")
    if version != SCHEMA_VERSION:
        raise ValueError(
            f"model metadata has unsupported schema_version {version!r}; {reads}"
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

#: An error: a magnitude, so never negative, and finite, since an infinite one
#: is a failure that should not have been recorded as a result.
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


class E0Details(_Record):
    """How one head's per-element reference energies (the E0s) were obtained.

    `values` maps chemical symbol to E0 in the model's energy unit. Symbols
    rather than atomic numbers, because JSON keys are strings and an integer
    key would not survive the round trip.
    """

    #: "explicit" when the E0s were given; "estimated" when fitted from the data.
    source: Literal["explicit", "estimated"]
    #: The estimation method (e.g. "average", "least_squares"); None when explicit.
    method: str | None = None
    #: Parameters of the method or reference, e.g. which data the fit used.
    parameters: dict[str, Any] = Field(default_factory=dict)
    values: dict[str, float] = Field(default_factory=dict)


class ElectrostaticsRecord(_Record):
    """The long-range solver a model computes with, as model state.

    A solver that does not reproduce the reference bit for bit gives other
    numbers, so which one a model was trained with is part of the model. It
    is recorded apart from the configuration because loading reads it before
    any model is built, and because a converted artifact has no training run
    whose configuration would say.
    """

    #: The name it is registered under.
    solver: str
    #: Whether it declared bit parity with the reference, which decides whether
    #: another solver may stand in for it at load time.
    bit_parity: bool
    #: The solve, as :func:`mace_core.electrostatics.descriptor_record` writes
    #: it. A rebuilt model's solve is compared against it.
    descriptor: dict[str, Any] = Field(default_factory=dict)


class ElementExtensionRecord(_Record):
    """The elements a model was given beyond its parent's, and how.

    The rows those elements have in every per-element tensor were never
    trained, so what they hold is the initialization that made them, and that
    is recorded rather than left to be guessed from the weights.
    """

    #: The added elements, as chemical symbols.
    added: list[str]
    #: The initialization spec the new rows were made with, by name.
    initialization: str
    #: The seed it drew them with, for an initialization that draws.
    seed: int | None = None
    #: For an initialization that copies, the element each added one copied.
    donors: dict[str, str] = Field(default_factory=dict)


class HeadSummary(_Record):
    """What one head was fitted on."""

    e0: E0Details | None = None

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


class BasisConversionRecord(_Record):
    """An explicit artifact rewrite, never a load-time basis reinterpretation."""

    from_basis: Literal["full"] = "full"
    to_basis: Literal["reduced"] = "reduced"
    projection_verified: bool
    reference: str | None = None
    golden_gate_passed: bool | None = None


class ModelMetadata(_Record):
    """How a model was made: its config, the code, the data and what to cite."""

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
    parents: list[ParentModel] = Field(default_factory=list)
    electrostatics: ElectrostaticsRecord | None = None
    element_extension: ElementExtensionRecord | None = None
    elements: list[int] = Field(default_factory=list)
    #: Set only on a model produced by a training run.
    training: TrainingSummary | None = None
    conversion: BasisConversionRecord | None = None

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


class ParentModel(_Record):
    """A model this one was built from."""

    #: What the parent contributed: its weights as the starting point (fine-tuning
    #: or continued training), or its predictions as distillation targets.
    role: Literal["initial_weights", "teacher"]
    #: How the config named it: a path or a registry name such as "mace-mp-0b3".
    name: str
    #: The parent's own record; None for a legacy checkpoint that carries none.
    metadata: ModelMetadata | None = None


ModelMetadata.model_rebuild()


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


__all__ += [
    "E0Details",
    "ElectrostaticsRecord",
    "ElementExtensionRecord",
    "ParentModel",
]
