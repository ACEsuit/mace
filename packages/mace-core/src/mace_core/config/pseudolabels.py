"""Relabelling a replay head's structures with the foundation model.

A fine-tune that replays the data a foundation model was trained on can replay
the model's own labels instead of the file's, so the replay head reproduces the
model rather than a level of theory the new head may not share. Generating them
is a step of the run, and what it writes is an artifact of the run: the labels,
and a record of everything they depend on.

**There is no cache across runs.** Labels are generated once per run, on the
first process, and written into the run directory. They depend on the
foundation model's weights, the structures, the device, the precision and the
torch build, and on a GPU they are not reproducible bit for bit, so a cache
keyed on the inputs would record a promise where the run records a fact. A
later run reuses them only by naming them, and only after the record says they
were made from the same model and the same structures.
"""

from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator

from mace_core.config.section import FrozenSection

__all__ = [
    "PSEUDOLABEL_PROPERTIES",
    "PseudolabelConfig",
    "PseudolabelProperty",
    "PseudolabelProvenance",
]

#: What a foundation model can label a structure with.
PseudolabelProperty = Literal[
    "energy", "forces", "stress", "virials", "dipole", "charges"
]

PSEUDOLABEL_PROPERTIES: tuple[str, ...] = (
    "energy",
    "forces",
    "stress",
    "virials",
    "dipole",
    "charges",
)


class PseudolabelConfig(FrozenSection):
    """Replay labels regenerated from the foundation model, or read back.

    Args:
        enabled: Generate them during this run.
        labels_from: Read them from a previous run's artifact instead, a
            directory holding one subdirectory per head as a run writes it.
            Mutually exclusive with ``enabled``.
        heads: The heads whose structures are relabelled. Empty, it is every
            head that reads a published replay dataset.
        properties: What the foundation model labels them with, each once.
        stress_if_missing: Label a stress on a structure that had none. Off, a
            structure gets a stress only if its file gave it one, and keeps
            that stress's weight; on, a stress it did not have is weighted one.
        batch_size: How many structures are labelled at once. ``None`` takes
            the training batch size.
    """

    enabled: bool = False
    labels_from: Path | None = None
    heads: tuple[str, ...] = ()
    properties: tuple[PseudolabelProperty, ...] = ("energy", "forces")
    stress_if_missing: bool = False
    batch_size: int | None = Field(default=None, ge=1)

    @field_validator("properties")
    @classmethod
    def _each_once(cls, value: tuple[str, ...]) -> tuple[str, ...]:
        repeated = sorted({name for name in value if value.count(name) > 1})
        if repeated:
            raise ValueError(
                f"finetune.pseudolabels.properties names {repeated} more than "
                f"once. Each is one label."
            )
        if not value:
            raise ValueError(
                "finetune.pseudolabels.properties is empty, so nothing would "
                "be relabelled. Name at least the energy."
            )
        return value


class PseudolabelProvenance(BaseModel):
    """Everything a set of generated labels depends on.

    Written beside the labels. A later run that reuses them compares it with
    its own foundation model and structures, and refuses a mismatch in either.

    Attributes:
        foundation_model: How the run named the foundation model.
        foundation_fingerprint: The sha256 of its weights.
        dataset_fingerprint: The digest of the structures that were labelled,
            of their geometry and not of their labels.
        head: The fine-tune head the structures belong to.
        foundation_head: The foundation head that labelled them.
        spec: The settings they were generated with.
        device: Where they were generated.
        dtype: What they were computed in.
        torch_version: The torch build.
        kernel_backend: The backend that computed them.
        generated_at: When, in UTC.
        n_configs: How many structures.
        deterministic: Whether generating them again gives the same bits. A
            CPU run does; a GPU run's reductions are not ordered.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    foundation_model: str
    foundation_fingerprint: str
    dataset_fingerprint: str
    head: str
    foundation_head: str
    spec: PseudolabelConfig
    device: str
    dtype: str
    torch_version: str
    kernel_backend: str
    generated_at: datetime
    n_configs: int
    deterministic: bool
