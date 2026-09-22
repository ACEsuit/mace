"""The three objects that pass between the stages of a run.

What is pinned here is the shape of the seam rather than any number: every
follow-up ticket builds against these fields, so a field that quietly changes
meaning is a breakage nothing else would report. The two properties worth
testing are that the objects are immutable and that the statistics exist once.
"""

from __future__ import annotations

import dataclasses

import pytest
from mace_core import BuiltModel, DataBundle, EpochRecord, TrainedModel
from mace_core.data.backend import DatasetStatistics
from mace_core.data.e0_resolution import E0Provenance
from mace_core.elements import AtomicNumberTable, ResolvedE0s
from mace_core.metadata import ConfigRecord, ModelMetadata, Provenance

Z_TABLE = AtomicNumberTable([1, 8])

STATISTICS = DatasetStatistics(
    atomic_energies={1: -13.6, 8: -2040.0},
    avg_num_neighbors=6.0,
    mean=-0.5,
    std=0.25,
    atomic_numbers=[1, 8],
    r_max=5.0,
)


def metadata() -> ModelMetadata:
    return ModelMetadata(
        config=ConfigRecord(), provenance=Provenance(code_version="0.0.0")
    )


def bundle() -> DataBundle[list[int]]:
    return DataBundle(
        z_table=Z_TABLE,
        heads=("default",),
        e0s=ResolvedE0s({"default": {1: -13.6, 8: -2040.0}}),
        e0_provenance={"default": E0Provenance(kind="table")},
        statistics=STATISTICS,
        train_loader=[1, 2],
        valid_loader=[3],
    )


def test_the_bundle_carries_what_the_model_stage_needs_and_no_more():
    """Every field is read by the model stage or by the loop.

    The list is asserted rather than described, because a field added here
    without a reader is a decision nobody made.
    """
    names = {field.name for field in dataclasses.fields(DataBundle)}
    assert names == {
        "z_table",
        "heads",
        "e0s",
        "e0_provenance",
        "statistics",
        "train_loader",
        "valid_loader",
        "test_loader",
    }


def test_the_boundary_objects_are_immutable():
    """A stage that could edit its input would make the seam decorative."""
    with pytest.raises(dataclasses.FrozenInstanceError):
        bundle().heads = ("other",)  # ty: ignore[invalid-assignment]
    with pytest.raises(dataclasses.FrozenInstanceError):
        EpochRecord(epoch=0, train_loss=1.0).epoch = 1  # ty: ignore[invalid-assignment]
    trained = TrainedModel(model=object(), metadata=metadata())
    with pytest.raises(dataclasses.FrozenInstanceError):
        trained.best_epoch = 3  # ty: ignore[invalid-assignment]


def test_there_is_one_statistics_object_and_the_model_reads_the_bundle_s():
    """Recomputing is what this identity check exists to forbid."""
    data = bundle()
    built = BuiltModel(model=object(), observables=(), data=data, metadata=metadata())
    assert built.statistics is data.statistics


def test_the_bundle_travels_with_the_model_rather_than_beside_it():
    """The training stage takes one argument, so it cannot be given a bundle
    the model was not built from."""
    built = BuiltModel(
        model=object(), observables=(), data=bundle(), metadata=metadata()
    )
    assert isinstance(built.data, DataBundle)


def test_an_epoch_records_whether_its_validation_number_came_through_ema():
    """Two numbers with the same name and different meanings otherwise."""
    plain = EpochRecord(epoch=0, train_loss=1.0, valid_loss=0.9)
    averaged = EpochRecord(
        epoch=0, train_loss=1.0, valid_loss=0.9, evaluated_with_ema=True
    )
    assert not plain.evaluated_with_ema
    assert averaged.evaluated_with_ema


def test_the_best_epoch_is_found_by_its_number_not_its_position():
    """The history can start after a resume, so the index is not the epoch."""
    history = (
        EpochRecord(epoch=7, train_loss=1.0, valid_loss=0.9),
        EpochRecord(epoch=8, train_loss=0.8, valid_loss=0.7),
    )
    trained = TrainedModel(
        model=object(), metadata=metadata(), history=history, best_epoch=8
    )
    assert trained.best is not None
    assert trained.best.train_loss == 0.8


def test_a_run_that_never_evaluated_has_no_best_epoch():
    trained = TrainedModel(model=object(), metadata=metadata())
    assert trained.best is None
