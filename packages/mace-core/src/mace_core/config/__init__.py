"""Configuration schemas for MACE v1.

The base machinery lives in `base`; the training schema sections (model, data,
training, ...) arrive with their own tickets and are re-exported from here.
"""

from mace_core.config.base import (
    ConfigError,
    ConfigSection,
    BaseConfig,
    read_config_file,
)
from mace_core.config.data import DataConfig, GraphInputKeys, HeadDataConfig
from mace_core.config.e0s import (
    E0sAverage,
    E0sEstimated,
    E0sFromFoundation,
    E0sIsolatedAtoms,
    E0Spec,
    E0sTable,
)
from mace_core.config.loss import LossConfig
from mace_core.config.model import ModelConfig, ReadoutConfig
from mace_core.config.resolved import FinetuneConfig, PseudolabelConfig, ResolvedConfig
from mace_core.config.runtime import RuntimeConfig
from mace_core.config.section import FrozenSection
from mace_core.config.training import (
    EMAConfig,
    SchedulerConfig,
    StageTwoConfig,
    TrainingConfig,
)

__all__ = [
    "ConfigError",
    "ConfigSection",
    "DataConfig",
    "E0Spec",
    "E0sAverage",
    "E0sEstimated",
    "E0sFromFoundation",
    "E0sIsolatedAtoms",
    "E0sTable",
    "EMAConfig",
    "FinetuneConfig",
    "FrozenSection",
    "GraphInputKeys",
    "HeadDataConfig",
    "LossConfig",
    "ModelConfig",
    "PseudolabelConfig",
    "ReadoutConfig",
    "BaseConfig",
    "ResolvedConfig",
    "RuntimeConfig",
    "SchedulerConfig",
    "StageTwoConfig",
    "TrainingConfig",
    "read_config_file",
]
