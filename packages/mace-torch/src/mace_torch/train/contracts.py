"""The stage objects with their type parameters bound, once.

:class:`~mace_core.stages.DataBundle` and
:class:`~mace_core.stages.BuiltModel` are generic over three things this
framework does name: the model, the loader the loop steps on, and the plain
loader an evaluation runs over. Binding them here rather than in each signature
keeps one spelling, so adding a fourth parameter later is one edit and not six.
"""

from __future__ import annotations

from mace_core.stages import BuiltModel, DataBundle
from torch import nn
from torch.utils.data import DataLoader

from mace_torch.train.loaders import TrainingLoader

__all__ = ["TorchBuiltModel", "TorchDataBundle"]

#: What the data stage produces.
TorchDataBundle = DataBundle[TrainingLoader, DataLoader]

#: What the model stage produces, and the training stage consumes.
TorchBuiltModel = BuiltModel[nn.Module, TrainingLoader, DataLoader]
