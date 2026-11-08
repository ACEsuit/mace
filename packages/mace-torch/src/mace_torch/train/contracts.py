"""Typed boundaries for ``DataStage -> ModelStage -> TrainStage``.

Each stage carries its predecessor: ``BuiltModel`` holds the ``DataBundle``
whose E0s and statistics it used. Statistics are computed once. Evaluation
loaders are per head; the training loader combines heads with configured
sampling. The stages use Torch models and data loaders.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path

from mace_core.data.backend import DatasetStatistics
from mace_core.data.e0_resolution import E0Provenance
from mace_core.data.normalization import NormalizationValues
from mace_core.elements import AtomicNumberTable, ResolvedE0s
from mace_core.metadata import ModelMetadata
from mace_core.observables import RequestedOutputs
from torch import nn
from torch.utils.data import DataLoader

from mace_torch.train.loaders import TrainingLoader

__all__ = [
    "BuiltModel",
    "DataBundle",
    "EpochRecord",
    "TrainedModel",
]


@dataclass(frozen=True)
class DataBundle:
    """Resolved data shared by model construction and training.

    Attributes:
        z_table: Ascending element table; its length is the one-hot width.
        heads: Head names in the order of the rows in ``e0s``.
        e0s: Resolved isolated-atom energies in eV, by head and element.
        e0_provenance: How each head's E0s were obtained.
        statistics: Neighbour count and interaction-energy statistics against ``e0s``.
        normalization: Optional measured values by observable and dataset head.
        train_loader: Training batches with heads combined for sampling.
        valid_loaders: Per-head validation passes for evaluation and error tables.
        train_eval_loaders: Per-head training-set passes used only for error tables.
        test_loaders: Post-training evaluation sets, keyed by configured name.
    """

    z_table: AtomicNumberTable
    heads: tuple[str, ...]
    e0s: ResolvedE0s
    e0_provenance: Mapping[str, E0Provenance]
    statistics: DatasetStatistics
    train_loader: TrainingLoader
    valid_loaders: Mapping[str, DataLoader]
    train_eval_loaders: Mapping[str, DataLoader] = field(default_factory=dict)
    test_loaders: Mapping[str, DataLoader] = field(default_factory=dict)
    normalization: Mapping[str, Mapping[str, NormalizationValues]] = field(
        default_factory=dict
    )

    def reported_loaders(self) -> dict[str, DataLoader]:
        """Every loader an error table has a row for, in the legacy order.

        Keyed ``train_<head>`` and ``valid_<head>``, which is the naming the
        frozen tree's table sorts on and the plot subcommand reads back.
        """
        rows: dict[str, DataLoader] = {}
        for head, loader in self.train_eval_loaders.items():
            rows[f"train_{head}"] = loader
        for head, loader in self.valid_loaders.items():
            rows[f"valid_{head}"] = loader
        return rows


@dataclass(frozen=True)
class BuiltModel:
    """Built model with its data and checkpoint metadata.

    Attributes:
        model: nn.Module on the configured device and precision.
        outputs: Resolved observables and requested derivatives, used by losses
            and error tables.
        data: The data bundle used to build this model.
        metadata: Checkpoint record, complete except for training results.
    """

    model: nn.Module
    outputs: RequestedOutputs
    data: DataBundle
    metadata: ModelMetadata

    @property
    def statistics(self) -> DatasetStatistics:
        """The bundle's, never a second computation."""
        return self.data.statistics


@dataclass(frozen=True)
class EpochRecord:
    """Metrics for one epoch.

    Attributes:
        epoch: Zero-based epoch index.
        train_loss: Mean over training batches.
        valid_loss: Validation loss, or ``None`` when evaluation was not scheduled.
        learning_rate: Optimizer rate after the schedule step.
        stage: ``"one"`` before the stage-two switch, ``"two"`` after it.
        evaluated_with_ema: Whether validation used averaged weights.
    """

    epoch: int
    train_loss: float
    valid_loss: float | None = None
    learning_rate: float = 0.0
    stage: str = "one"
    evaluated_with_ema: bool = False


@dataclass(frozen=True)
class TrainedModel:
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

    model: nn.Module
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
