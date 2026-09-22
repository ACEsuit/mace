"""What passes between the three stages of a training run.

A run is ``ResolvedConfig -> DataStage -> DataBundle -> ModelStage ->
BuiltModel -> TrainStage -> TrainedModel``, and these are the three objects in
the middle. They are here rather than in :mod:`mace_torch` because the seam is
the contract, not the implementation: a jax training entry point produces the
same bundle, and a tool that inspects a finished run reads the same record.

**The statistics are computed once and carried, never recomputed.** The mean
and spread the model's scale and shift are set from have to be the ones taken
against the isolated-atom energies that become the model's own buffer, and a
second pass can differ from the first for reasons nothing reports: a different
split, a rounded cutoff, a head filtered out. Recomputing also costs a second
pass over the whole dataset for a number already in hand. So
:class:`DataBundle` owns the one :class:`~mace_core.data.backend.DatasetStatistics`
and :class:`BuiltModel` reads it through the bundle it carries.

**Each stage takes the previous object, so the chain is linear.**
:class:`BuiltModel` holds its :class:`DataBundle` rather than the training
stage receiving two arguments. A stage that needed both would let a caller pass
a bundle the model was not built from, and nothing downstream could notice.

The framework objects are type parameters. A dataloader and a model are the two
things this package cannot name, and leaving them as ``Any`` would give every
consumer an untyped attribute at the exact point the stage contract exists to
type.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Generic, TypeVar

from mace_core.data.backend import DatasetStatistics
from mace_core.data.e0_resolution import E0Provenance
from mace_core.elements import AtomicNumberTable, ResolvedE0s
from mace_core.metadata import ModelMetadata
from mace_core.observables import ObservableSpec

__all__ = [
    "BuiltModel",
    "DataBundle",
    "EpochRecord",
    "TrainedModel",
]

#: A batched dataloader, whatever the framework calls one.
Loader = TypeVar("Loader")

#: A built model, whatever the framework calls one.
Model = TypeVar("Model")


@dataclass(frozen=True)
class DataBundle(Generic[Loader]):
    """Everything the data stage resolved, and nothing the model stage decides.

    Attributes:
        z_table: The element table, ascending. Every element a head's E0s were
            resolved for is in it, and the model's one-hot width is its length.
        heads: The head names, in the order their rows appear in ``e0s``.
        e0s: Isolated-atom energies per head, already resolved. The one place
            they are settled for the run.
        e0_provenance: How each head's energies were obtained, keyed by head.
            Recorded because ``average`` and a table read from a file produce
            the same numbers and mean different things.
        statistics: Average neighbour count, and the mean and spread of the
            interaction energy, taken against ``e0s``.
        train_loader: The batches the loop steps on.
        valid_loader: The batches it evaluates on.
        test_loader: Structures evaluated once after training, when the
            configuration named any.
    """

    z_table: AtomicNumberTable
    heads: tuple[str, ...]
    e0s: ResolvedE0s
    e0_provenance: Mapping[str, E0Provenance]
    statistics: DatasetStatistics
    train_loader: Loader
    valid_loader: Loader
    test_loader: Loader | None = None


@dataclass(frozen=True)
class BuiltModel(Generic[Model, Loader]):
    """A model, its data, and the record of how both were arrived at.

    Attributes:
        model: The model itself, on its device and in its precision.
        observables: What it reads out, as declared. The loss terms and the
            error tables are derived from this rather than from a model class
            name.
        data: The bundle it was built from. Carried so the training stage takes
            one argument and cannot be handed a bundle from another run.
        metadata: The record that is written beside the weights. Complete at
            build time except for what only training can know.
    """

    model: Model
    observables: tuple[ObservableSpec, ...]
    data: DataBundle[Loader]
    metadata: ModelMetadata

    @property
    def statistics(self) -> DatasetStatistics:
        """The bundle's, never a second computation."""
        return self.data.statistics


@dataclass(frozen=True)
class EpochRecord:
    """One epoch as the loop saw it.

    Attributes:
        epoch: Counted from zero.
        train_loss: Mean over the epoch's batches.
        valid_loss: ``None`` on an epoch the loop did not evaluate, which is
            every epoch that is not a multiple of the evaluation interval.
        learning_rate: What the optimizer used, after any schedule step.
        stage: ``"one"`` before the stage-two switch and ``"two"`` after it.
        evaluated_with_ema: Whether the validation loss above was measured
            through the averaged weights. It is not a detail: with EMA on, the
            reported number comes from parameters the optimizer never saw.
    """

    epoch: int
    train_loss: float
    valid_loss: float | None = None
    learning_rate: float = 0.0
    stage: str = "one"
    evaluated_with_ema: bool = False


@dataclass(frozen=True)
class TrainedModel(Generic[Model]):
    """What a finished run leaves behind.

    Attributes:
        model: The trained model. With EMA on it carries the averaged weights,
            because those are the ones the best checkpoint was chosen by.
        metadata: The build record, plus what training added to it.
        history: One entry per epoch, oldest first.
        best_epoch: The epoch whose validation loss was lowest, or ``None``
            when nothing was ever evaluated.
        checkpoint_path: Where the best checkpoint was written, or ``None``
            when the run was asked not to write one.
    """

    model: Model
    metadata: ModelMetadata
    history: tuple[EpochRecord, ...] = ()
    best_epoch: int | None = None
    checkpoint_path: Path | None = None

    @property
    def best(self) -> EpochRecord | None:
        """The best epoch's record, by validation loss."""
        if self.best_epoch is None:
            return None
        for record in self.history:
            if record.epoch == self.best_epoch:
                return record
        return None
