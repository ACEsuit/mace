"""Splitting a dataset, and grouping it for reporting.

Both are pure functions over sequences and both belong beside the boundary
object they operate on. Put either one behind a framework and the type that
crosses every layer of the data stack acquires a second home.
"""

from __future__ import annotations

import logging
from collections.abc import Sequence
from pathlib import Path
from typing import TypeVar

import numpy as np

from mace_core.data.configuration import Configuration

__all__ = ["group_by_config_type", "random_train_valid_split"]

logger = logging.getLogger(__name__)

ItemT = TypeVar("ItemT")

#: Above this many validation items the chosen indices go to a file instead of
#: into the log. The threshold exists because the indices are what makes a run
#: reproducible after the fact, and a log line long enough to hold a thousand
#: of them is a log line nobody reads.
_INDEX_FILE_THRESHOLD = 10


def random_train_valid_split(
    items: Sequence[ItemT],
    valid_fraction: float,
    seed: int,
    work_dir: str | Path,
    prefix: str | None = None,
) -> tuple[list[ItemT], list[ItemT]]:
    """Split a dataset into training and validation parts, reproducibly.

    The validation set always gets at least one item, even when the fraction
    rounds down to nothing, because a tiny fitting database with an empty
    validation set fails much later and much less clearly.

    The chosen validation indices are recorded as a side effect, and that is
    part of what this function is for rather than a debug aid: it is what lets
    a finished run be re-split identically. Fewer than ten of them are logged
    inline; from ten upwards they are written to
    ``<prefix>_valid_indices_<seed>.txt`` in the work directory.

    Args:
        items: The dataset to split.
        valid_fraction: Fraction held out, strictly between 0 and 1.
        seed: Seeds the shuffle. The same seed and the same length give the
            same split.
        work_dir: Where the index file is written.
        prefix: Prepended to the index file name, to keep concurrent runs in
            one directory from overwriting each other's.

    Returns:
        ``(train, valid)``, each a new list.
    """
    if not 0.0 < valid_fraction < 1.0:
        raise ValueError(
            f"valid_fraction is {valid_fraction}, which holds out either "
            f"nothing or everything. It must be strictly between 0 and 1."
        )
    size = len(items)
    if size < 2:
        raise ValueError(
            f"cannot split {size} item(s) into a training and a validation "
            f"set; at least 2 are needed."
        )

    train_size = min(size - int(valid_fraction * size), size - 1)

    indices = list(range(size))
    np.random.default_rng(seed).shuffle(indices)
    valid_indices = indices[train_size:]

    if len(valid_indices) < _INDEX_FILE_THRESHOLD:
        logger.info(
            "Using a random %.0f%% of the training set for validation, indices %s",
            100 * valid_fraction,
            valid_indices,
        )
    else:
        name = f"valid_indices_{seed}.txt"
        if prefix:
            name = f"{prefix}_{name}"
        path = Path(work_dir) / name
        path.write_text(
            "".join(f"{index}\n" for index in valid_indices), encoding="utf-8"
        )
        logger.info(
            "Using a random %.0f%% of the training set for validation, indices "
            "saved in %s",
            100 * valid_fraction,
            path,
        )

    return [items[i] for i in indices[:train_size]], [items[i] for i in valid_indices]


def group_by_config_type(
    configurations: Sequence[Configuration],
) -> list[tuple[str, list[Configuration]]]:
    """Group configurations by ``"<config_type>_<head>"``, in first-seen order.

    This is what a per-config-type error table is built on, so the group name
    carries the head as well: the same config type evaluated under two heads is
    two rows, not one averaged row.

    An empty head reads as the empty string, giving names like ``"bulk_"``.
    Unlike the legacy grouping this does not write that normalisation back onto
    the configuration: a function that only reports should not edit what it is
    reporting on.
    """
    groups: dict[str, list[Configuration]] = {}
    for configuration in configurations:
        name = f"{configuration.config_type}_{configuration.head or ''}"
        groups.setdefault(name, []).append(configuration)
    return list(groups.items())
