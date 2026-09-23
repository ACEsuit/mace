"""Starting from a trained model: replay data, adapters, freezing.

A fine-tune is a training run whose model starts from a foundation model's
weights and whose heads may include a replay head. Nothing here makes a replay
head a kind of its own: it is a head whose structures come from a published
dataset rather than from a file, and every head goes through the same stages.
"""

from mace_torch.finetune.subselect import (
    FILTERINGS,
    METHODS,
    SelectionError,
    farthest_point_indices,
    passes_filter,
    select,
    split_by_filter,
)

__all__ = [
    "FILTERINGS",
    "METHODS",
    "SelectionError",
    "farthest_point_indices",
    "passes_filter",
    "select",
    "split_by_filter",
]
