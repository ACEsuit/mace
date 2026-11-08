"""Build-time registry of built-in and additional model families."""

from collections.abc import Callable
from dataclasses import dataclass
from typing import Protocol, TypeVar, runtime_checkable

from mace_core.config.electrostatics import ElectrostaticsConfig
from mace_core.config.model import ModelConfig
from mace_core.electrostatics.descriptor import ElectrostaticsSolverDescriptor
from mace_core.kernels.precision import Precision
from mace_core.observables import InputSpec

from mace_torch.models.base import MACEModel
from mace_torch.models.magnetic import MagneticModel

ModelType = TypeVar("ModelType", bound=type[MACEModel])


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


@dataclass(frozen=True)
class ModelFamily:
    """Constructor and physical energy policies for one named model family."""

    model: type[MACEModel | MagneticModel]
    zbl_in_scale_shift: bool = True
    widen_node_energy: bool = True
    electrostatics: bool = False


def _builtins() -> dict[str, ModelFamily]:
    from mace_torch.models.dipoles import DipoleModel
    from mace_torch.models.electrostatics import (
        FixedPointSCFModel,
        LocalChargeModel,
        LocalSplitChargeModel,
        PolarModel,
        QEqModel,
    )
    from mace_torch.models.les import LatentEwaldModel

    return {
        "scale_shift": ModelFamily(MACEModel),
        "plain": ModelFamily(
            MACEModel, zbl_in_scale_shift=False, widen_node_energy=False
        ),
        "les": ModelFamily(LatentEwaldModel),
        "polar": ModelFamily(PolarModel, electrostatics=True),
        "magnetic": ModelFamily(MagneticModel, widen_node_energy=False),
        "dipole": ModelFamily(DipoleModel),
        "dielectric": ModelFamily(DipoleModel),
        **{
            name: ModelFamily(
                model,
                zbl_in_scale_shift=False,
                widen_node_energy=False,
                electrostatics=True,
            )
            for name, model in (
                ("lc_mace", LocalChargeModel),
                ("lsc_mace", LocalSplitChargeModel),
                ("qeq", QEqModel),
                ("fixed_point_scf", FixedPointSCFModel),
            )
        },
    }


MODEL_REGISTRY: dict[str, ModelFamily] | None = None


def registered_models() -> dict[str, ModelFamily]:
    """One registry for built-in and explicitly registered model families."""
    global MODEL_REGISTRY
    if MODEL_REGISTRY is None:
        MODEL_REGISTRY = _builtins()
    return MODEL_REGISTRY


def model_family(name: str) -> ModelFamily:
    families = registered_models()
    if name not in families:
        raise ValueError(
            f"model.model {name!r} is not a registered model; "
            f"choose from {sorted(families)}"
        )
    return families[name]


def register_model(
    name: str,
    *,
    zbl_in_scale_shift: bool = True,
    widen_node_energy: bool = True,
    electrostatics: bool = False,
) -> Callable[[ModelType], ModelType]:
    """Register a constructor and its energy policies, rejecting name collisions."""

    def register(model: ModelType) -> ModelType:
        families = registered_models()
        if name in families:
            raise ValueError(f"model {name!r} is already registered")
        if electrostatics and not isinstance(model, ElectrostaticModelFactory):
            raise TypeError(
                f"electrostatic model {name!r} must define solver_descriptor"
            )
        families[name] = ModelFamily(
            model, zbl_in_scale_shift, widen_node_energy, electrostatics
        )
        return model

    return register
