"""The three stages of a run, and the loop the last of them runs.

Imported from here, not from the model or the layer packages: nothing under
``nn`` or ``models`` imports ``train``, and no package ``__init__`` imports it
eagerly. That is the rule that keeps the training driver from becoming the
module everything else reaches into, which is what the frozen tree's
``run_train`` is.
"""

from mace_torch.train.checkpoint import (
    RunState,
    read_run_state,
    write_model,
    write_run_state,
)
from mace_torch.train.data_stage import (
    DEFAULT_PRECISION,
    DataStageError,
    run_data_stage,
)
from mace_torch.train.ema import ExponentialMovingAverage
from mace_torch.train.loop import evaluate, run_train_stage, train_one_epoch
from mace_torch.train.loss import (
    LossTerm,
    UnsupportedLossError,
    WeightedLoss,
    weighted_loss,
)
from mace_torch.train.model_stage import ModelStageError, run_model_stage
from mace_torch.train.optimizers import (
    UnsupportedOptimizerError,
    build_optimizer,
    build_scheduler,
    parameter_groups,
)

__all__ = [
    "DEFAULT_PRECISION",
    "DataStageError",
    "ExponentialMovingAverage",
    "LossTerm",
    "ModelStageError",
    "RunState",
    "UnsupportedLossError",
    "UnsupportedOptimizerError",
    "WeightedLoss",
    "build_optimizer",
    "build_scheduler",
    "evaluate",
    "parameter_groups",
    "read_run_state",
    "run_data_stage",
    "run_model_stage",
    "run_train_stage",
    "train_one_epoch",
    "weighted_loss",
    "write_model",
    "write_run_state",
]
