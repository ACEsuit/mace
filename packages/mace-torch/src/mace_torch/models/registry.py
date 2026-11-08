"""Build-time registration of additional energy model families."""

from collections.abc import Callable
from importlib import import_module
from typing import TypeVar

from mace_torch.models.base import MACEModel

ModelType = TypeVar("ModelType", bound=type[MACEModel])
MODEL_REGISTRY: dict[str, type[MACEModel]] = {}


def registered_models() -> dict[str, type[MACEModel]]:
    """Load built-in registrations before constructing a dataset or model."""
    import_module("mace_torch.models.electrostatics")
    return MODEL_REGISTRY


def register_model(name: str) -> Callable[[ModelType], ModelType]:
    """Register a model constructor; conflicting definitions fail immediately."""

    def register(model: ModelType) -> ModelType:
        if name in MODEL_REGISTRY:
            raise ValueError(f"model {name!r} is already registered")
        MODEL_REGISTRY[name] = model
        return model

    return register
