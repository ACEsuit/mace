"""One batch: the model's input, and what its outputs are compared against.

The joining itself is :func:`mace_core.graph.collate`, in numpy, so the torch
batch and the jax one are the same arrangement of the same numbers. What is
added here is the binding to tensors and the two fields the model reads that
are derived rather than stored: the element index and the graph count.

They are derived at the last moment on purpose. ``element_index`` depends on
the element table the model was built with, so storing it in the dataset would
tie a cached dataset to one model; ``num_graphs`` is ``len(ptr) - 1`` and
reading it from ``batch.max()`` is both a host synchronisation and wrong for a
batch whose last graph has no nodes.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass, field

import numpy as np
import torch
from mace_core.data.configuration import Configuration
from mace_core.elements import AtomicNumberTable
from mace_core.graph import collate as collate_numpy
from torch import Tensor
from torch.utils.data import DataLoader, Dataset

from mace_torch.data.graphs import (
    TargetSpec,
    graph_from_configuration,
    targets_from_configuration,
)
from mace_torch.graph import DTYPES, to_tensors

__all__ = [
    "GraphDataset",
    "Item",
    "TrainingBatch",
    "collate_training",
    "make_collate",
    "make_loader",
]


@dataclass(frozen=True)
class TrainingBatch:
    """What one optimizer step reads.

    Attributes:
        graph: The model's input, schema fields plus the derived two. The
            graph count is a plain ``int``, which is what makes it stay
            symbolic under tracing rather than becoming a host read.
        targets: Reference values by name, joined along the first axis.
        property_weights: How much each structure's value of each property
            counts, ``[n_graphs]`` per name. Zero where a structure carries no
            such value, which is what makes a partially labelled dataset train
            without a mask anyone has to remember.
    """

    graph: dict[str, Tensor | int]
    targets: dict[str, Tensor]
    property_weights: dict[str, Tensor] = field(default_factory=dict)

    def to(self, device: str | torch.device) -> TrainingBatch:
        """The same batch on another device."""
        return TrainingBatch(
            graph={
                name: value.to(device) if isinstance(value, Tensor) else value
                for name, value in self.graph.items()
            },
            targets={name: value.to(device) for name, value in self.targets.items()},
            property_weights={
                name: value.to(device) for name, value in self.property_weights.items()
            },
        )


#: One structure as a dataset hands it over: its graph, its reference values,
#: and how much each of those counts. Named because three modules pass it
#: around and the tuple written out three times is the thing that drifts.
Item = tuple[Mapping[str, np.ndarray], Mapping[str, np.ndarray], Mapping[str, float]]


def collate_training(
    items: Sequence[Item],
    *,
    z_table: AtomicNumberTable,
    float_dtype: str = "float64",
) -> TrainingBatch:
    """Join structures into one batch.

    Args:
        items: Triples of graph, targets and per-property weights, as
            :mod:`mace_torch.data.graphs` builds them.
        z_table: The model's element table, used for the element index.
        float_dtype: What the floating fields are bound at.
    """
    tensors = to_tensors(
        collate_numpy([item[0] for item in items]), float_dtype=float_dtype
    )
    lookup = torch.full((max(z_table.zs) + 1,), -1, dtype=torch.long)
    for position, number in enumerate(z_table.zs):
        lookup[number] = position
    graph: dict[str, Tensor | int] = dict(tensors)
    graph["element_index"] = lookup[tensors["atomic_numbers"]]
    # A plain int rather than a tensor: it indexes the reductions, and as a
    # tensor it would be a host read in every one of them.
    graph["num_graphs"] = int(tensors["ptr"].numel() - 1)

    names = list(items[0][1]) if items else []
    targets = {
        name: torch.as_tensor(
            np.concatenate([np.asarray(item[1][name]) for item in items], axis=0),
            dtype=DTYPES[float_dtype],
        )
        for name in names
    }
    weights = {
        name: torch.as_tensor(
            [float(item[2][name]) for item in items], dtype=DTYPES[float_dtype]
        )
        for name in names
    }
    return TrainingBatch(graph=graph, targets=targets, property_weights=weights)


class GraphDataset(Dataset):
    """Structures, built into graphs on the way out.

    Building lazily rather than up front is what lets a dataset larger than
    memory use the same loader, and the cost is one neighbour search per epoch
    per structure. The frozen tree builds them all first, which is why its
    memory scales with the dataset rather than with the batch.
    """

    def __init__(
        self,
        configurations: Sequence[Configuration],
        *,
        cutoff: float,
        z_table: AtomicNumberTable,
        targets: Sequence[TargetSpec],
        heads: Sequence[str] = ("default",),
        graph_inputs: Sequence[str] = (),
        augmentations: Sequence[Callable[[Configuration], Configuration]] = (),
    ) -> None:
        self.configurations = list(configurations)
        # Drawn again on every read. A dataset that is evaluated is built
        # without them.
        self.augmentations = tuple(augmentations)
        self.graph_inputs = tuple(graph_inputs)
        self.cutoff = cutoff
        self.z_table = z_table
        self.targets = tuple(targets)
        self.head_index = {name: position for position, name in enumerate(heads)}

    def __len__(self) -> int:
        return len(self.configurations)

    def __getitem__(self, index: int) -> Item:
        configuration = self.configurations[index]
        for augment in self.augmentations:
            configuration = augment(configuration)
        head = self.head_index.get(configuration.head, 0)
        graph = graph_from_configuration(
            configuration,
            cutoff=self.cutoff,
            z_table=self.z_table,
            head=head,
            weight=float(configuration.weight),
            graph_inputs=self.graph_inputs,
        )
        targets, weights = targets_from_configuration(
            configuration, self.targets, len(configuration.atomic_numbers)
        )
        return graph, targets, weights


def make_collate(
    z_table: AtomicNumberTable, float_dtype: str = "float64"
) -> Callable[[Iterable[Item]], TrainingBatch]:
    """The collate function a loader over this element table needs.

    Separate from :func:`make_loader` because a loader over several heads\'
    datasets has no single dataset to read the table off, and the table is the
    model\'s rather than any one dataset\'s.
    """

    def collate(items: Iterable[Item]) -> TrainingBatch:
        return collate_training(list(items), z_table=z_table, float_dtype=float_dtype)

    return collate


def make_loader(
    dataset: GraphDataset,
    *,
    batch_size: int,
    shuffle: bool,
    float_dtype: str = "float64",
    num_workers: int = 0,
    pin_memory: bool = False,
    drop_last: bool = False,
    seed: int | None = None,
) -> DataLoader:
    """A loader whose batches are :class:`TrainingBatch`."""
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        # Its own generator, seeded by the run, so the order is the run's to
        # reproduce and not whatever the global one has drawn before. The
        # frozen tree seeds its training loader the same way
        # (`mace/cli/run_train.py:778`).
        generator=torch.Generator().manual_seed(seed) if seed is not None else None,
        collate_fn=make_collate(dataset.z_table, float_dtype),
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=drop_last,
    )
