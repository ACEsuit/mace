"""The second stage: a configuration and its data in, a model out.

Everything the model is comes from the configuration and from the statistics
the data stage measured. Nothing is decided by a class name: the frozen tree
branches on one to set six booleans that say which quantities exist, and here
those quantities are the declared observables.

The backend is resolved once, here, and the model is built with it. The frozen
tree builds an e3nn model and then converts it in place, which means the layout
depends on a flag read after construction and a conversion that has to be
undone to save.
"""

from __future__ import annotations

import logging
from collections.abc import Mapping
from typing import Any

import torch
from ase.data import chemical_symbols
from mace_core.config.model import ModelConfig
from mace_core.config.precision import PrecisionConfig
from mace_core.config.provenance import e0_details
from mace_core.config.resolved import ResolvedConfig
from mace_core.data.backend import DatasetStatistics
from mace_core.data.normalization import NormalizationValues
from mace_core.elements import AtomicNumberTable, ResolvedE0s
from mace_core.kernels.descriptors import RadialBasisDescriptor
from mace_core.kernels.precision import widest
from mace_core.metadata import (
    ConfigRecord,
    E0Details,
    ElectrostaticsRecord,
    HeadSummary,
    ModelMetadata,
    ParentModel,
    Provenance,
)
from mace_core.observables import (
    DEFAULT_CATALOGUE,
    ObservableCatalogue,
    RequestedOutputs,
    resolve_requested,
)
from mace_core.stages import BuiltModel

from mace_torch import __version__
from mace_torch.backends import (
    CompositeBackend,
    require_double_backward,
    resolve_backend,
)
from mace_torch.backends.precision import PrecisionBackend, degradation_report
from mace_torch.finetune.foundation import Foundation
from mace_torch.finetune.transfer import readout_sources, transfer_foundation
from mace_torch.kernels import initialize_model_weights
from mace_torch.models import EnergyOutputHead, MACEModel, ScaleShiftSpec
from mace_torch.models.dipoles import DipoleModel, DipoleSettings
from mace_torch.models.electrostatics import PolarModel, PolarSettings
from mace_torch.models.les import LatentEwaldModel
from mace_torch.models.magnetic import MagneticModel
from mace_torch.models.normalization import ObservableNormalization
from mace_torch.models.registry import MODEL_REGISTRY
from mace_torch.physics import DerivativeEngine
from mace_torch.train.contracts import TorchBuiltModel, TorchDataBundle
from mace_torch.train.data_stage import DEFAULT_PRECISION

logger = logging.getLogger(__name__)

__all__ = [
    "DEFAULT_HIDDEN_IRREPS",
    "ModelStageError",
    "build_model",
    "run_model_stage",
    "with_foundation_architecture",
]

#: What one channel carries when the configuration does not say. The frozen
#: tree spells the same choice as `--max_L 1`, which it then expands into a
#: multiplicity per irrep; here the multiplicity is `num_channels` and this is
#: the shape of one channel.
DEFAULT_HIDDEN_IRREPS = "0e+1o"

logger = logging.getLogger(__name__)

#: Which model spelling puts the short-range repulsion inside the scaled sum.
#: The two differ by about 77 eV on a probe geometry, so it is a fact about a
#: trained model rather than a preference.
_ZBL_INSIDE = {"scale_shift": True, "les": True, "plain": False}

#: Which energy models hand out per-atom energies in the accumulation dtype
#: rather than their own. The frozen tree widens them in its scale-shift, polar
#: and LES models and not in its plain and magnetic ones.
_WIDENS_NODE_ENERGY = {
    "scale_shift": True,
    "plain": False,
    "polar": True,
    "les": True,
    "magnetic": False,
}

#: The two response models, which read out a dipole and no energy: fixed
#: charges and a dipole alone, or predicted charges, a dipole and a
#: polarizability.
_RESPONSE_MODELS = {
    "dipole": DipoleSettings(charges="fixed"),
    "dielectric": DipoleSettings(charges="predicted", polarizability=True),
}

#: Every registered model name. The charge-aware one scales and shifts its
#: local energy as the scale-shift model does.
_MODELS = frozenset({*_ZBL_INSIDE, "polar", "magnetic", *_RESPONSE_MODELS})

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


class ModelStageError(RuntimeError):
    """The configuration does not describe a model that can be built."""


def run_model_stage(
    config: ResolvedConfig,
    data: TorchDataBundle,
    catalogue: ObservableCatalogue,
    *,
    precision: PrecisionConfig | None = None,
    supports_float64: bool = True,
    initialize: bool = True,
    foundation: Foundation | None = None,
) -> TorchBuiltModel:
    """Build the model the configuration describes, scaled by the data.

    Args:
        config: The resolved configuration.
        data: What the data stage resolved. Its statistics are read, never
            recomputed.
        catalogue: The observable declarations.
        precision: Optional override of config.precision, recorded in metadata.
        supports_float64: Whether the target device has float64 at all. A
            device that does not degrades the accumulation and says so.
        initialize: Draw a fresh set of weights. ``False`` leaves every
            weighted op at zero, which is what a model about to be loaded from
            a checkpoint wants and what a model about to be trained must not
            have: zeros multiply to zeros and so does the gradient.
        foundation: The model a fine-tune starts from. Its architecture is
            the model's, its weights are transferred into it, and it is
            recorded as the model's parent.

    Returns:
        The model wrapped in its derivative engine, with the bundle it was
        built from and the record that travels with the weights.
    """
    if config.fixed_point is not None:
        raise ModelStageError(
            "fixed_point configures inference relaxation; "
            "train the underlying model at fixed inputs"
        )
    if foundation is not None:
        config = with_foundation_architecture(config, foundation, catalogue)
    precision = precision or config.precision
    if precision != config.precision:
        config = config.model_copy(update={"precision": precision})
    engine, requested = build_model(
        config,
        catalogue,
        z_table=data.z_table,
        heads=data.heads,
        e0s=data.e0s,
        statistics=data.statistics,
        normalization=data.normalization if foundation is None else None,
        precision=precision,
        supports_float64=supports_float64,
        initialize=initialize,
    )
    if requested.derivatives:
        # Training on a derivative differentiates through it once more.
        require_double_backward(
            config.model.backend,
            "training on " + ", ".join(sorted(requested.derivatives)),
        )
    if catalogue != DEFAULT_CATALOGUE:
        config = config.model_copy(update={"catalogue": catalogue})
    metadata = _metadata(config, data)
    model = engine.get_submodule("backbone")
    solver_record = getattr(model, "solver_record", None)
    if callable(solver_record):
        metadata = metadata.model_copy(
            update={"electrostatics": ElectrostaticsRecord(**solver_record())}
        )
    if foundation is not None:
        transfer_foundation(
            foundation.model,
            engine.get_submodule("backbone"),
            foundation_elements=foundation.z_table.zs,
            elements=data.z_table.zs,
            foundation_heads=foundation.heads,
            heads=data.heads,
            readout_from=readout_sources(
                data.heads,
                {name: config.data.heads[name].readout_from for name in data.heads},
                foundation.heads,
            ),
            transfer_readout=config.finetune.transfer_readout,
        )
        metadata = metadata.model_copy(
            update={
                "parents": [
                    ParentModel(
                        role="initial_weights",
                        name=foundation.name,
                        metadata=foundation.metadata,
                    )
                ]
            }
        )
    from mace_core.citations import required_citations

    metadata = metadata.model_copy(update={"citations": required_citations(metadata)})
    return BuiltModel(
        model=engine,
        outputs=requested,
        data=data,
        metadata=metadata,
    )


#: The model settings that are the run's own even when it starts from a
#: foundation model. Everything else is architecture, and the foundation's.
_RUN_OWNED_MODEL_FIELDS = frozenset({"observables", "backend"})


def with_foundation_architecture(
    config: ResolvedConfig,
    foundation: Foundation,
    catalogue: ObservableCatalogue = DEFAULT_CATALOGUE,
) -> ResolvedConfig:
    """The configuration with the foundation model's architecture in it.

    A fine-tune does not choose its architecture: its weights are the
    foundation's, and they fit only the model they were trained in. So the
    model section is the foundation's, except for what the run reads out and
    which backend computes it. A setting the run wrote down that disagrees
    with the foundation is refused rather than overridden, because a run that
    asked for a cutoff of five and trained at six would never be told.

    Raises:
        ModelStageError: Naming each setting the run set that the foundation
            model contradicts, and each observable it cannot read out.
    """
    theirs = _saturations_by_element(foundation.config.model, foundation)
    ours = config.model
    if ours.model == "les" and theirs.model == "scale_shift":
        # Keep the donor backbone and add fresh LES readouts. The LES family
        # itself keeps all final irreps; its new vector channels start fresh.
        theirs = theirs.model_copy(update={"model": "les", "les": ours.les})
    conflicts = sorted(
        name
        for name in ours.model_fields_set - _RUN_OWNED_MODEL_FIELDS - {"normalization"}
        if getattr(ours, name) != getattr(theirs, name)
    )
    if conflicts:
        details = ", ".join(
            f"{name}: {getattr(ours, name)!r} here, {getattr(theirs, name)!r} "
            f"in the foundation model"
            for name in conflicts
        )
        raise ModelStageError(
            f"the run sets model settings its foundation model was not built "
            f"with ({details}). A fine-tune takes the foundation's "
            f"architecture, so drop them or start from a model built that way."
        )
    # Derivatives reuse their parent's trained head. A foundation that was
    # trained without stress labels can still differentiate its energy.
    requested = resolve_requested(ours.observables, config.catalogue or catalogue)
    available = resolve_requested(
        theirs.observables, foundation.config.catalogue or catalogue
    )
    unread = sorted(
        {spec.name for spec in requested.observables}
        - {spec.name for spec in available.observables}
    )
    if unread:
        raise ModelStageError(
            f"the run reads out {unread}, which the foundation model does not; "
            f"it reads out {list(theirs.observables)}. A new observable needs a "
            f"head nobody trained, which a fine-tune does not start from."
        )
    model = theirs.model_copy(
        update={name: getattr(ours, name) for name in _RUN_OWNED_MODEL_FIELDS}
    )
    if theirs.normalization:
        sources = readout_sources(
            tuple(config.data.heads),
            {name: head.readout_from for name, head in config.data.heads.items()},
            foundation.heads,
        )
        model = model.model_copy(
            update={"normalization": theirs.normalization_for_heads(sources)}
        )
    if (
        "normalization" in ours.model_fields_set
        and ours.normalization != model.normalization
    ):
        raise ModelStageError(
            "model.normalization must preserve the foundation head recipes"
        )
    return config.model_copy(update={"model": model})


def _saturations_by_element(model: ModelConfig, foundation: Foundation) -> ModelConfig:
    """The foundation's moment saturations by atomic number.

    Given one per element of its table in its order, they cannot be read by a
    fine-tune over fewer elements; by atomic number they can, and they are the
    same numbers.
    """
    saturation = model.magnetic.saturation
    if not isinstance(saturation, tuple):
        return model
    by_element = dict(zip(foundation.z_table.zs, saturation, strict=True))
    return model.model_copy(
        update={
            "magnetic": model.magnetic.model_copy(update={"saturation": by_element})
        }
    )


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

    Apart from :func:`run_model_stage` because rebuilding a model from its
    checkpoint needs it without any data: the element table, the heads and the
    energies come from the record, and every constant the statistics would
    have set is put back from the tensors afterwards.

    Returns:
        The model wrapped in its derivative engine, and what it reads out.
    """
    if config.catalogue is not None and catalogue != config.catalogue:
        raise ModelStageError("catalogue disagrees with config.catalogue")
    requested = resolve_requested(config.model.observables, catalogue)
    if not requested.observables:
        raise ModelStageError(
            "the configuration declares no observable, so the model would "
            "read out nothing. Declare at least one under `model.observables`."
        )
    registered = MODEL_REGISTRY.get(config.model.model)
    if config.model.model not in _MODELS and registered is None:
        raise ModelStageError(
            f"{config.model.model!r} is not a registered model. The choices are "
            f"{sorted(_MODELS | MODEL_REGISTRY.keys())}: the two energy models "
            f"differ in where the "
            f"short-range repulsion is added, `polar` carries a "
            f"self-consistent density and a long-range term, `magnetic` reads a "
            f"moment on every atom, and `dipole` and `dielectric` read out a "
            f"dipole and no energy."
        )
    if (
        _MAGNETIC_DERIVATIVE in requested.derivatives
        and config.model.model != "magnetic"
    ):
        raise ModelStageError(
            f"model.observables asks for {_MAGNETIC_DERIVATIVE!r}, the energy's "
            f"derivative against the moments, and a {config.model.model!r} model "
            f"reads no moments, so it would be zero. Set `model.model` to "
            f"'magnetic', or drop it."
        )
    _refuse_unbuilt(config)
    _check_electrostatics(config)
    backend = _backend(config, precision, supports_float64)
    if config.model.model in _RESPONSE_MODELS:
        model = _response_model(
            config,
            requested,
            backend=backend,
            z_table=z_table,
            statistics=statistics,
            precision=precision,
            supports_float64=supports_float64,
        )
        _configure_normalization(model, config, heads, normalization, precision)
        _report_backend(backend)
        if initialize:
            initialize_model_weights(model, config.runtime.seed)
        engine = DerivativeEngine(
            model, None, inputs=catalogue.inputs, responses=requested.observables
        )
        return engine, requested

    energy = next(
        (spec for spec in requested.observables if spec.name == "energy"), None
    )
    energy_head = None
    if energy is None and (
        config.model.model in {"les", "polar", "magnetic"}
        or registered is not None
        or config.model.pair_repulsion
    ):
        raise ModelStageError(
            "this model's physical energy terms require an `energy` observable"
        )
    if energy is not None:
        energy_head = EnergyOutputHead(
            e0s,
            heads,
            z_table,
            _scale_shift(config, statistics, len(heads)),
            precision,
            zbl_in_scale_shift=getattr(
                registered,
                "zbl_in_scale_shift",
                _ZBL_INSIDE.get(config.model.model, True),
            ),
            supports_float64=supports_float64,
            widen_node_energy=getattr(
                registered,
                "widen_node_energy",
                _WIDENS_NODE_ENERGY.get(config.model.model, True),
            ),
        )
        if (report := energy_head.build_report()) is not None:
            logger.warning(report)
    if config.model.model == "magnetic":
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
        _configure_normalization(model, config, heads, normalization, precision)
        _report_backend(backend)
        if initialize:
            initialize_model_weights(model, config.runtime.seed)
        return DerivativeEngine(model, energy, None, inputs=catalogue.inputs), requested
    if config.model.model == "polar":
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
        _configure_normalization(model, config, heads, normalization, precision)
        _report_backend(backend)
        if initialize:
            initialize_model_weights(model, config.runtime.seed)
        return DerivativeEngine(model, energy, None, inputs=catalogue.inputs), requested
    model_type = registered or (
        LatentEwaldModel if config.model.model == "les" else MACEModel
    )
    model_options: dict[str, Any] = {}
    if getattr(model_type, "requires_electrostatics", False):
        assert energy_head is not None
        solver_descriptor = getattr(model_type, "solver_descriptor", None)
        assert callable(solver_descriptor), "electrostatic models declare their solve"
        model_options.update(
            settings=solver_descriptor(
                config.model,
                config.electrostatics,
                widest(precision.model, energy_head.accumulate),
            ),
            solver=config.electrostatics.solver,
            trains_derivatives=initialize and bool(requested.derivatives),
        )
    model = model_type(
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
        # Seeded from the run, so the same configuration and the same seed
        # rebuild the same model. The walk is over the model rather than the
        # engine, so wrapping it in one more layer later cannot change a
        # weight.
        initialize_model_weights(model, config.runtime.seed)
    return DerivativeEngine(
        model,
        energy,
        inputs=catalogue.inputs,
        responses=[spec for spec in requested.observables if spec.name != "energy"],
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
            raise ModelStageError(
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


def _report_backend(backend: Any) -> None:
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
    backend: Any,
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
    backend: Any,
    z_table: AtomicNumberTable,
    statistics: DatasetStatistics,
    precision: PrecisionConfig,
    supports_float64: bool = True,
) -> DipoleModel:
    """A dipole or dielectric model, which reads out no energy.

    Raises:
        ModelStageError: If the declared observables are not ones the model
            produces, or leave out the dipole it is built around.
    """
    settings = _RESPONSE_MODELS[config.model.model]
    produced = {"dipole", "polarizability"} if settings.polarizability else {"dipole"}
    declared = {spec.name for spec in requested.observables}
    if "dipole" not in declared or not declared <= produced:
        raise ModelStageError(
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
    backend: Any,
    z_table: AtomicNumberTable,
    heads: tuple[str, ...],
    precision: PrecisionConfig,
    supports_float64: bool = True,
) -> MagneticModel:
    """The magnetic model, which reads out an energy and nothing else."""
    others = [spec.name for spec in requested.observables if spec.name != "energy"]
    if others:
        raise ModelStageError(
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
) -> Any:
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
        ModelStageError: Naming each such setting, its value, and what can be
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
    if config.model.model in _RESPONSE_MODELS:
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
        raise ModelStageError(
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
        ModelStageError: If the section is enabled for a model that carries no
            long-range term, which would train one without it, or the polar
            model is asked for with the section off, which would leave its
            solver and its systems unsaid.
    """
    settings = config.electrostatics
    polar = config.model.model == "polar" or getattr(
        MODEL_REGISTRY.get(config.model.model), "requires_electrostatics", False
    )
    if settings.enabled:
        from mace_core.electrostatics import get_solver

        get_solver(settings.solver)
    if settings.enabled and not polar:
        raise ModelStageError(
            f"electrostatics.enabled is set and model.model is "
            f"{config.model.model!r}, which carries no long-range term, so the "
            f"run would train one without it. Use a model with electrostatics, "
            f"or leave "
            f"the section out."
        )
    if polar and not settings.enabled:
        raise ModelStageError(
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


def _metadata(config: ResolvedConfig, data: TorchDataBundle) -> ModelMetadata:
    """The record written beside the weights.

    The resolved configuration goes in whole. A checkpoint that carried only
    the weights would need its run's command line to be rebuilt, and that is
    the thing least likely to still exist.

    Each head's isolated-atom energies go in too, with how they were obtained.
    Two reasons, and the second is not a nicety: ``average`` and a table read
    from a file produce the same numbers and mean different things, and a
    fine-tune that copies a foundation model's energies reads them from here
    rather than out of a loaded module's buffer, which says what some model
    was built with and nothing about which head it belonged to. A model that
    reads out no energy has none, and its heads record an empty table.
    """
    reads_energy = "energy" in config.model.observables
    return ModelMetadata(
        config=ConfigRecord(resolved=config.model_dump(mode="json")),
        provenance=Provenance(versions={"mace-torch-v1": __version__}),
        elements=[int(z) for z in data.z_table.zs],
        heads={
            head: HeadSummary(
                e0=E0Details(source="explicit")
                if not reads_energy
                else e0_details(
                    config.data.heads[head].e0s,
                    {
                        chemical_symbols[number]: float(energy)
                        for number, energy in data.e0s.values[head].items()
                    },
                    from_foundation=[
                        chemical_symbols[number]
                        for number in data.e0_provenance[head].from_foundation
                    ]
                    if head in data.e0_provenance
                    else (),
                )
            )
            for head in data.heads
        },
    )
