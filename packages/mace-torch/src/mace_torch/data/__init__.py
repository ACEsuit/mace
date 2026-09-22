"""Structures in, batches out.

The parsing and the neighbour search are :mod:`mace_core`'s; what is here is
the binding to torch and the dataset and loader that a training run steps
through.
"""

from mace_torch.data.batch import (
    GraphDataset,
    TrainingBatch,
    collate_training,
    make_loader,
)
from mace_torch.data.graphs import (
    MissingTargetError,
    TargetSpec,
    graph_from_configuration,
    target_specs,
    targets_from_configuration,
)
from mace_torch.data.transforms import (
    TRANSFORM_REGISTRY,
    Transform,
    TransformError,
    UnknownTransformError,
    apply_transforms,
    register_transform,
)

__all__ = [
    "TRANSFORM_REGISTRY",
    "GraphDataset",
    "MissingTargetError",
    "TargetSpec",
    "TrainingBatch",
    "Transform",
    "TransformError",
    "UnknownTransformError",
    "apply_transforms",
    "collate_training",
    "graph_from_configuration",
    "make_loader",
    "register_transform",
    "target_specs",
    "targets_from_configuration",
]
