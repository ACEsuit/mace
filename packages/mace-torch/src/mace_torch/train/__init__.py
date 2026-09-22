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
from mace_torch.train.loaders import (
    BalancedLoader,
    IndexSharder,
    ProportionalLoader,
    TrainingLoader,
    build_training_loader,
)
from mace_torch.train.logs import LOG_FORMAT, setup_logging
from mace_torch.train.loop import (
    evaluate,
    evaluate_heads,
    report_errors,
    run_train_stage,
    selection_loss,
    train_one_epoch,
)
from mace_torch.train.loss import (
    LOSS_REGISTRY,
    GeneratedLoss,
    LossTerm,
    UnknownLossError,
    build_loss,
    reduce_loss,
    register_loss,
    terms_for,
)
from mace_torch.train.metrics import MetricSpec, RunningMetrics, metric_specs
from mace_torch.train.model_stage import ModelStageError, run_model_stage
from mace_torch.train.optimizers import (
    UnsupportedOptimizerError,
    build_optimizer,
    build_scheduler,
    parameter_groups,
)
from mace_torch.train.tracking import (
    NullTracker,
    Tracker,
    WandbTracker,
    epoch_values,
    open_tracker,
)

__all__ = [
    "DEFAULT_PRECISION",
    "LOG_FORMAT",
    "LOSS_REGISTRY",
    "BalancedLoader",
    "DataStageError",
    "ExponentialMovingAverage",
    "GeneratedLoss",
    "IndexSharder",
    "LossTerm",
    "MetricSpec",
    "ModelStageError",
    "NullTracker",
    "ProportionalLoader",
    "RunState",
    "RunningMetrics",
    "Tracker",
    "TrainingLoader",
    "UnknownLossError",
    "UnsupportedOptimizerError",
    "WandbTracker",
    "build_loss",
    "build_optimizer",
    "build_scheduler",
    "build_training_loader",
    "epoch_values",
    "evaluate",
    "evaluate_heads",
    "metric_specs",
    "open_tracker",
    "parameter_groups",
    "read_run_state",
    "reduce_loss",
    "register_loss",
    "report_errors",
    "run_data_stage",
    "run_model_stage",
    "run_train_stage",
    "selection_loss",
    "setup_logging",
    "terms_for",
    "train_one_epoch",
    "write_model",
    "write_run_state",
]
