"""Build-time registry of built-in and additional model architectures."""

from collections.abc import Callable
from dataclasses import dataclass
from typing import Protocol, TypeVar, runtime_checkable

from mace_core.config.electrostatics import ElectrostaticsConfig
from mace_core.config.model import ModelConfig
from mace_core.electrostatics.spec import ElectrostaticsSpec
from mace_core.kernels.precision import Precision
from mace_core.observables import InputSpec

from mace_torch.models.base import MACEModel
from mace_torch.models.inputs import with_embedded_inputs
from mace_torch.models.magnetic import MagneticModel

ModelType = TypeVar("ModelType", bound=type[MACEModel])


@runtime_checkable
class ElectrostaticModelFactory(Protocol):
    """The solver specification supplied by an electrostatic model architecture."""

    @staticmethod
    def make_solver_spec(
        config: ModelConfig, electrostatics: ElectrostaticsConfig, precision: Precision
    ) -> ElectrostaticsSpec: ...


@dataclass(frozen=True)
class ModelArchitecture:
    """Constructor and physical energy policies for one named model architecture."""

    model: type[MACEModel | MagneticModel]
    zbl_in_scale_shift: bool = True
    widen_node_energy: bool = True
    electrostatics: bool = False


def _builtins() -> dict[str, ModelArchitecture]:
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
        "scale_shift": ModelArchitecture(MACEModel),
        "plain": ModelArchitecture(
            MACEModel, zbl_in_scale_shift=False, widen_node_energy=False
        ),
        "les": ModelArchitecture(LatentEwaldModel),
        "polar": ModelArchitecture(PolarModel, electrostatics=True),
        "magnetic": ModelArchitecture(MagneticModel, widen_node_energy=False),
        "dipole": ModelArchitecture(DipoleModel),
        "dielectric": ModelArchitecture(DipoleModel),
        **{
            name: ModelArchitecture(
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


MODEL_ARCHITECTURES: dict[str, ModelArchitecture] | None = None


def registered_architectures() -> dict[str, ModelArchitecture]:
    """One registry for built-in and explicitly registered model architectures."""
    global MODEL_ARCHITECTURES
    if MODEL_ARCHITECTURES is None:
        MODEL_ARCHITECTURES = _builtins()
    return MODEL_ARCHITECTURES


def model_architecture(name: str) -> ModelArchitecture:
    architectures = registered_architectures()
    if name not in architectures:
        raise ValueError(
            f"model.model {name!r} is not a registered model; "
            f"choose from {sorted(architectures)}"
        )
    return architectures[name]


def model_inputs(config: ModelConfig) -> tuple[InputSpec, ...]:
    """Resolve all model inputs without constructing its operators or weights."""
    return with_embedded_inputs(
        config, model_architecture(config.model).model.declared_inputs(config)
    )


def register_model(
    name: str,
    *,
    zbl_in_scale_shift: bool = True,
    widen_node_energy: bool = True,
    electrostatics: bool = False,
) -> Callable[[ModelType], ModelType]:
    """Register a constructor and its energy policies, rejecting name collisions."""

    def register(model: ModelType) -> ModelType:
        architectures = registered_architectures()
        if name in architectures:
            raise ValueError(f"model {name!r} is already registered")
        if electrostatics and not isinstance(model, ElectrostaticModelFactory):
            raise TypeError(
                f"electrostatic model {name!r} must define make_solver_spec"
            )
        architectures[name] = ModelArchitecture(
            model, zbl_in_scale_shift, widen_node_energy, electrostatics
        )
        return model

    return register
