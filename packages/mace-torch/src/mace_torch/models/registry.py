"""Build-time registration of additional energy model families."""

from collections.abc import Callable
from importlib import import_module
from typing import Protocol, TypeVar, runtime_checkable

from mace_core.config.electrostatics import ElectrostaticsConfig
from mace_core.config.model import ModelConfig
from mace_core.electrostatics.descriptor import ElectrostaticsSolverDescriptor
from mace_core.kernels.precision import Precision
from mace_core.observables import InputSpec

from mace_torch.models.base import MACEModel

ModelType = TypeVar("ModelType", bound=type[MACEModel])
MODEL_REGISTRY: dict[str, type[MACEModel]] = {}


@runtime_checkable
class DeclaresModelInputs(Protocol):
    """Additional physical inputs a registered model requires in its graphs."""

    @staticmethod
    def declared_inputs(config: ModelConfig) -> tuple[InputSpec, ...]: ...


@runtime_checkable
class ElectrostaticModelFactory(Protocol):
    """The solver specification supplied by an electrostatic model family."""

    @staticmethod
    def solver_descriptor(
        config: ModelConfig, electrostatics: ElectrostaticsConfig, precision: Precision
    ) -> ElectrostaticsSolverDescriptor: ...


def registered_models() -> dict[str, type[MACEModel]]:
    """Load built-in registrations before constructing a dataset or model."""
    import_module("mace_torch.models.electrostatics")
    return MODEL_REGISTRY


def register_model(name: str) -> Callable[[ModelType], ModelType]:
    """Register a model constructor; conflicting definitions fail immediately."""

    def register(model: ModelType) -> ModelType:
        if name in MODEL_REGISTRY:
            raise ValueError(f"model {name!r} is already registered")
        if model.requires_electrostatics and not isinstance(
            model, ElectrostaticModelFactory
        ):
            raise TypeError(
                f"electrostatic model {name!r} must define solver_descriptor"
            )
        MODEL_REGISTRY[name] = model
        return model

    return register
