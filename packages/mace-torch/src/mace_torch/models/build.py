"""Build a model from its configuration and resolved physical inputs.

Shared by training, checkpoint loading and deployment. Dataset loading,
weight transfer and run metadata belong to the calling stage.
"""

from __future__ import annotations

import logging
from collections.abc import Mapping
from typing import Any, cast

import torch
from mace_core.config.precision import DEFAULT_PRECISION, PrecisionConfig
from mace_core.config.resolved import ResolvedConfig
from mace_core.data.backend import DatasetStatistics
from mace_core.data.normalization import NormalizationValues
from mace_core.elements import AtomicNumberTable, ResolvedE0s
from mace_core.kernels.descriptors import RadialBasisDescriptor
from mace_core.kernels.precision import widest
from mace_core.kernels.protocol import KernelBackend
from mace_core.observables import (
    ObservableCatalogue,
    RequestedOutputs,
    resolve_requested,
)

from mace_torch.backends import (
    CompositeBackend,
    resolve_backend,
)
from mace_torch.backends.precision import PrecisionBackend, degradation_report
from mace_torch.kernels import initialize_model_weights
from mace_torch.models import EnergyOutputHead, MACEModel, ScaleShiftSpec
from mace_torch.models.dipoles import DipoleModel, DipoleSettings
from mace_torch.models.electrostatics import PolarModel, PolarSettings
from mace_torch.models.magnetic import MagneticModel
from mace_torch.models.normalization import ObservableNormalization
from mace_torch.models.registry import ElectrostaticModelFactory, model_family
from mace_torch.physics import DerivativeEngine

logger = logging.getLogger(__name__)

__all__ = ["DEFAULT_HIDDEN_IRREPS", "ModelBuildError", "build_model"]

#: What one channel carries when the configuration does not say. The frozen
#: tree spells the same choice as `--max_L 1`, which it then expands into a
#: multiplicity per irrep; here the multiplicity is `num_channels` and this is
#: the shape of one channel.
DEFAULT_HIDDEN_IRREPS = "0e+1o"

#: The derivative only a model that reads moments has.
_MAGNETIC_DERIVATIVE = "magforces"

#: What the magnetic model's first and later interactions are called in the
#: frozen tree. The model builds that pair whatever the two fields say, so
#: these and the defaults are the spellings it accepts.
_MAGNETIC_INTERACTIONS = {
    "interaction": "MagneticRealAgnosticResidueSpinOrbitCoupledDensityInteractionBlock",
    "interaction_first": "MagneticRealAgnosticSpinOrbitCoupledDensityInteractionBlock",
}

_BUILT_ONE_WAY: dict[str, tuple[object, ...]] = {
    "interaction": ("plain", "residual", "density", "density_residual", "nonlinear"),
    "interaction_first": (
        "plain",
        "residual",
        "density",
        "density_residual",
        "nonlinear",
    ),
    "distance_transform.kind": ("none", "agnesi", "soft"),
    "clebsch_gordan_basis": ("reduced", "full"),
}


class ModelBuildError(RuntimeError):
    """The configuration does not describe a model that can be built."""


def build_model(
    config: ResolvedConfig,
    catalogue: ObservableCatalogue,
    *,
    z_table: AtomicNumberTable,
    heads: tuple[str, ...],
    e0s: ResolvedE0s,
    statistics: DatasetStatistics,
    normalization: Mapping[str, Mapping[str, NormalizationValues]] | None = None,
    precision: PrecisionConfig = DEFAULT_PRECISION,
    supports_float64: bool = True,
    initialize: bool = True,
) -> tuple[DerivativeEngine, RequestedOutputs]:
    """The model a configuration describes, over an element table and heads.

    Training supplies measured statistics. Checkpoint loading supplies the
    recorded element and head tables, then restores the canonical buffers.

    Returns:
        The model wrapped in its derivative engine, and what it reads out.
    """
    if config.catalogue is not None and catalogue != config.catalogue:
        raise ModelBuildError("catalogue disagrees with config.catalogue")
    requested = resolve_requested(config.model.observables, catalogue)
    if not requested.observables:
        raise ModelBuildError(
            "the configuration declares no observable, so the model would "
            "read out nothing. Declare at least one under `model.observables`."
        )
    try:
        family = model_family(config.model.model)
    except ValueError as error:
        raise ModelBuildError(str(error)) from error
    model_type = family.model
    if (
        _MAGNETIC_DERIVATIVE in requested.derivatives
        and config.model.model != "magnetic"
    ):
        raise ModelBuildError(
            f"model.observables asks for {_MAGNETIC_DERIVATIVE!r}, the energy's "
            f"derivative against the moments, and a {config.model.model!r} model "
            f"reads no moments, so it would be zero. Set `model.model` to "
            f"'magnetic', or drop it."
        )
    _refuse_unbuilt(config)
    _check_electrostatics(config)
    backend = _backend(config, precision, supports_float64)
    if model_type is DipoleModel:
        energy = None
        responses = requested.observables
        model = _response_model(
            config,
            requested,
            backend=backend,
            z_table=z_table,
            statistics=statistics,
            precision=precision,
            supports_float64=supports_float64,
        )
    else:
        energy = next(
            (spec for spec in requested.observables if spec.name == "energy"), None
        )
        energy_head = None
        if energy is None and (
            model_type is not MACEModel or config.model.pair_repulsion
        ):
            raise ModelBuildError(
                "this model's physical energy terms require an `energy` observable"
            )
        if energy is not None:
            energy_head = EnergyOutputHead(
                e0s,
                heads,
                z_table,
                _scale_shift(config, statistics, len(heads)),
                precision,
                zbl_in_scale_shift=family.zbl_in_scale_shift,
                supports_float64=supports_float64,
                widen_node_energy=family.widen_node_energy,
            )
            if (report := energy_head.build_report()) is not None:
                logger.warning(report)
        responses = (
            ()
            if config.model.model in {"magnetic", "polar"}
            else tuple(spec for spec in requested.observables if spec.name != "energy")
        )
        if model_type is MagneticModel:
            assert energy_head is not None
            model = _magnetic_model(
                config,
                requested,
                energy_head,
                backend=backend,
                z_table=z_table,
                heads=heads,
                precision=precision,
                supports_float64=supports_float64,
            )
        elif config.model.model == "polar":
            assert energy_head is not None
            model = _polar_model(
                config,
                requested,
                energy_head,
                backend=backend,
                z_table=z_table,
                heads=heads,
                statistics=statistics,
                precision=precision,
                supports_float64=supports_float64,
                trains_derivatives=initialize and bool(requested.derivatives),
            )
        else:
            model_options: dict[str, Any] = {}
            if family.electrostatics:
                assert energy_head is not None
                assert isinstance(model_type, ElectrostaticModelFactory)
                model_options.update(
                    settings=model_type.solver_descriptor(
                        config.model,
                        config.electrostatics,
                        widest(precision.model, energy_head.accumulate),
                    ),
                    solver=config.electrostatics.solver,
                    trains_derivatives=initialize and bool(requested.derivatives),
                )
            model = cast(type[MACEModel], model_type)(
                backend,
                config=config.model,
                atomic_numbers=list(z_table.zs),
                observables=requested.observables,
                energy_head=energy_head,
                avg_num_neighbors=statistics.avg_num_neighbors,
                radial_precision=precision.for_device(supports_float64)[0]
                .policy_for(RadialBasisDescriptor())
                .accumulate_dtype,
                precision=precision.model,
                num_heads=len(heads),
                **model_options,
            )
    _configure_normalization(model, config, heads, normalization, precision)
    _report_backend(backend)
    if initialize:
        # Keep the walk over the model, before the derivative wrapper, so the
        # same seed addresses the same canonical operators on every path.
        initialize_model_weights(model, config.runtime.seed)
    return DerivativeEngine(
        model, energy, inputs=catalogue.inputs, responses=responses
    ), requested


def _configure_normalization(
    model: MACEModel | MagneticModel,
    config: ResolvedConfig,
    heads: tuple[str, ...],
    measured: Mapping[str, Mapping[str, NormalizationValues]] | None,
    precision: PrecisionConfig,
) -> None:
    """Build optional output buffers, including identity buffers for reloads."""
    specs = {spec.name: spec for spec in model.outputs.specs}
    for name, recipes in config.model.normalization.items():
        if name not in specs:
            raise ModelBuildError(
                f"model.normalization.{name}: this model computes that quantity "
                "from physical terms rather than a declarative readout; "
                "normalizing those terms independently is not supported"
            )
        spec = specs[name]
        identity = NormalizationValues(1.0, (0.0,) * spec.dimension)
        values = [
            measured[name][head]
            if measured is not None and head in recipes
            else identity
            for head in heads
        ]
        if name == "energy":
            energy = model.outputs.energy_head
            assert energy is not None
            for index, head in enumerate(heads):
                if head in recipes:
                    energy.scale[index] = values[index].scale
                    energy.shift[index] = values[index].shift[0]
        else:
            model.outputs.normalization[name] = ObservableNormalization(
                spec, values, getattr(torch, precision.model)
            )


def _report_backend(backend: KernelBackend) -> None:
    """Say which backend built which ops, when more than one did."""
    if isinstance(backend, PrecisionBackend):
        backend = backend.backend
    if isinstance(backend, CompositeBackend):
        logger.info("Kernel backends:\n%s", backend.report())


def _polar_model(
    config: ResolvedConfig,
    requested: RequestedOutputs,
    energy_head: EnergyOutputHead,
    *,
    backend: KernelBackend,
    z_table: AtomicNumberTable,
    heads: tuple[str, ...],
    statistics: DatasetStatistics,
    precision: PrecisionConfig,
    supports_float64: bool = True,
    trains_derivatives: bool,
) -> PolarModel:
    """The charge-aware model, with its long-range ops from the named solver.

    Its dipole is the density's, so a declared ``dipole`` observable is read
    off the model rather than given a head.
    """
    polar = config.model.polar
    electrostatics = config.electrostatics
    settings = PolarSettings(
        multipole_max_l=polar.multipole_max_l,
        multipole_width=polar.multipole_width,
        feature_max_l=polar.feature_max_l,
        feature_widths=tuple(polar.feature_widths),
        feature_norms=None
        if polar.feature_norms is None
        else tuple(polar.feature_norms),
        num_recursion_steps=polar.num_recursion_steps,
        kspace_cutoff_factor=electrostatics.kspace_cutoff_factor,
        feature_self_interaction=polar.feature_self_interaction,
        energy_self_interaction=polar.energy_self_interaction,
        add_local_electron_energy=polar.add_local_electron_energy,
        quadrupole_feature_corrections=polar.quadrupole_feature_corrections,
        fukui_hidden=_readout_hidden(config),
        periodicity_profile=electrostatics.periodicity_profile,
        slab_normal=electrostatics.slab_normal,
    )
    return PolarModel(
        backend,
        config=config.model,
        atomic_numbers=list(z_table.zs),
        observables=[
            spec
            for spec in requested.observables
            if spec.name not in PolarModel.PRODUCED
        ],
        energy_head=energy_head,
        settings=settings,
        solver=electrostatics.solver,
        trains_derivatives=trains_derivatives,
        avg_num_neighbors=statistics.avg_num_neighbors,
        radial_precision=precision.for_device(supports_float64)[0]
        .policy_for(RadialBasisDescriptor())
        .accumulate_dtype,
        precision=precision.model,
        num_heads=len(heads),
    )


def _response_model(
    config: ResolvedConfig,
    requested: RequestedOutputs,
    *,
    backend: KernelBackend,
    z_table: AtomicNumberTable,
    statistics: DatasetStatistics,
    precision: PrecisionConfig,
    supports_float64: bool = True,
) -> DipoleModel:
    """A dipole or dielectric model, which reads out no energy.

    Raises:
        ModelBuildError: If the declared observables are not ones the model
            produces, or leave out the dipole it is built around.
    """
    settings = DipoleSettings(
        charges="predicted" if config.model.model == "dielectric" else "fixed",
        polarizability=config.model.model == "dielectric",
    )
    produced = {"dipole", "polarizability"} if settings.polarizability else {"dipole"}
    declared = {spec.name for spec in requested.observables}
    if "dipole" not in declared or not declared <= produced:
        raise ModelBuildError(
            f"model.model is {config.model.model!r}, which produces "
            f"{sorted(produced)} and nothing else, and the observables declared "
            f"are {sorted(declared)}. Declare the dipole, and of the rest only "
            f"what the model produces."
        )
    return DipoleModel(
        backend,
        config=config.model,
        atomic_numbers=list(z_table.zs),
        settings=settings,
        avg_num_neighbors=statistics.avg_num_neighbors,
        radial_precision=precision.for_device(supports_float64)[0]
        .policy_for(RadialBasisDescriptor())
        .accumulate_dtype,
        precision=precision.model,
    )


def _magnetic_model(
    config: ResolvedConfig,
    requested: RequestedOutputs,
    energy_head: EnergyOutputHead,
    *,
    backend: KernelBackend,
    z_table: AtomicNumberTable,
    heads: tuple[str, ...],
    precision: PrecisionConfig,
    supports_float64: bool = True,
) -> MagneticModel:
    """The magnetic model, which reads out an energy and nothing else."""
    others = [spec.name for spec in requested.observables if spec.name != "energy"]
    if others:
        raise ModelBuildError(
            f"model.observables declares {others} beside the energy, and the "
            f"magnetic model reads out the energy alone. Its moments are an "
            f"input, and `magforces` is the energy's derivative against them."
        )
    settings = config.model.magnetic
    model = MagneticModel(
        backend,
        config.model,
        atomic_numbers=list(z_table.zs),
        observables=requested.observables,
        energy_head=energy_head,
        saturation=settings.saturation_for(z_table.zs),
        num_heads=len(heads),
        precision=precision.model,
    )
    if model.one_body is not None and not settings.train_one_body:
        model.one_body.coefficients.requires_grad_(False)
    return model


def _backend(
    config: ResolvedConfig, precision: PrecisionConfig, supports_float64: bool
) -> KernelBackend:
    """The configured backend, building every op at the precision given it."""
    if (report := degradation_report(precision, supports_float64)) is not None:
        logger.warning(report)
    return resolve_backend(config.model.backend, precision, supports_float64)


def _unbuilt_graph_features(config: ResolvedConfig) -> list[str]:
    """Embedded inputs the data path could not supply, and a readout of an
    embedding that has nothing embedded in it."""
    model = config.model
    refused = []
    if model.readout.from_embedding and not model.graph_features:
        refused.append(
            "model.readout.from_embedding is True with no model.graph_features; "
            "the frozen tree builds the embedding readout only beside them"
        )
    if model.model == "polar":
        for path, value in (("graph_features", bool(model.graph_features)),):
            if value:
                refused.append(
                    f"model.{path} is set for the 'polar' model, and it is "
                    f"built for the energy models"
                )
    return refused


def _unbuilt_convolution_irreps(config: ResolvedConfig) -> list[str]:
    """The edge irreps a layer would be asked to honour and could not.

    The frozen tree hands them to every block. Only the nonlinear one is
    built to convolve at another width than its node features, which is where
    every published model that sets them sets them.
    """
    model = config.model
    if model.edge_irreps is None:
        return []
    refused = []
    if model.num_interactions > 1 and model.interaction != "nonlinear":
        refused.append(
            f"model.edge_irreps is {model.edge_irreps!r} with model.interaction "
            f"{model.interaction!r}; it is built for "
            f"'RealAgnosticResidualNonLinearInteractionBlock' only"
        )
    if model.use_edge_irreps_first and model.interaction_first != "nonlinear":
        refused.append(
            f"model.use_edge_irreps_first is True with model.interaction_first "
            f"{model.interaction_first!r}; it is built for "
            f"'RealAgnosticResidualNonLinearInteractionBlock' only"
        )
    return refused


def _refuse_unbuilt(config: ResolvedConfig) -> None:
    """Refuse every model setting this stage would not build as written.

    Raises:
        ModelBuildError: Naming each such setting, its value, and what can be
            built instead.
    """
    unbuilt = []
    built = dict(_BUILT_ONE_WAY)
    if config.model.model != "magnetic":
        built["magnetic.time_reversal"] = ("none",)
    if config.model.model == "polar":
        # The frozen tree computes the repulsion of this model and never adds
        # it, so the only faithful build is the one without it.
        built["pair_repulsion"] = (False,)
    if model_family(config.model.model).model is DipoleModel:
        # A response model has no energy to add a repulsion to.
        built["pair_repulsion"] = (False,)
    if config.model.model == "magnetic":
        for field, name in _MAGNETIC_INTERACTIONS.items():
            built[field] = (*built[field], name)
        # Its radial networks read the Bessel basis of the edge length beside
        # the moment's own basis, and it has no other.
        built["radial_basis.kind"] = ("bessel",)
    for path, choices in built.items():
        value: object = config.model
        for name in path.split("."):
            value = getattr(value, name)
        if value not in choices:
            allowed = " or ".join(repr(choice) for choice in choices)
            unbuilt.append(f"model.{path} is {value!r}, and only {allowed} is built")
    unbuilt += _unbuilt_convolution_irreps(config)
    unbuilt += _unbuilt_graph_features(config)
    if unbuilt:
        raise ModelBuildError(
            "the configuration asks for a model this stage cannot build, and "
            "building the default instead would train another architecture: "
            + "; ".join(unbuilt)
            + "."
        )


def _check_electrostatics(config: ResolvedConfig) -> None:
    """The long-range section and the model have to agree.

    The solver is resolved first, so one that is not registered or did not
    import is named as such rather than hidden behind a mismatch.

    Raises:
        SolverNotAvailableError: If the solver cannot be delivered.
        ModelBuildError: If the section is enabled for a model that carries no
            long-range term, which would train one without it, or the polar
            model is asked for with the section off, which would leave its
            solver and its systems unsaid.
    """
    settings = config.electrostatics
    polar = model_family(config.model.model).electrostatics
    if settings.enabled:
        from mace_core.electrostatics import get_solver

        get_solver(settings.solver)
    if settings.enabled and not polar:
        raise ModelBuildError(
            f"electrostatics.enabled is set and model.model is "
            f"{config.model.model!r}, which carries no long-range term, so the "
            f"run would train one without it. Use a model with electrostatics, "
            f"or leave "
            f"the section out."
        )
    if polar and not settings.enabled:
        raise ModelBuildError(
            f"model.model is {config.model.model!r}, whose long-range term uses "
            "solver the electrostatics section names. Set "
            "electrostatics.enabled, with the solver and the systems it is set "
            "up for."
        )


def _scale_shift(
    config: ResolvedConfig, statistics: DatasetStatistics, num_heads: int
) -> ScaleShiftSpec:
    """Legacy aggregate defaults, before explicit per-head recipes override them."""
    if config.model.scaling == "none":
        return ScaleShiftSpec("none", (1.0,) * num_heads, (0.0,) * num_heads)
    return ScaleShiftSpec(
        config.model.scaling,
        # Keep a constant target trainable, as the per-head recipes do. The
        # measured spread remains zero in the dataset statistics.
        (statistics.std or 1.0,) * num_heads,
        (statistics.mean,) * num_heads,
    )


def _readout_hidden(config: ResolvedConfig) -> int:
    """The width of the last readout's middle, from its irreps string."""
    from mace_core.observables import irreps_dimension

    return irreps_dimension(config.model.readout.mlp_irreps)
