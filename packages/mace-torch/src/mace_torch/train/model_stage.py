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

from typing import Literal, NamedTuple

from ase.data import chemical_symbols
from mace_core.config.provenance import e0_details
from mace_core.config.resolved import ResolvedConfig
from mace_core.data.backend import DatasetStatistics
from mace_core.elements import AtomicNumberTable, ResolvedE0s
from mace_core.graph import GRAPH_INPUT_DEFAULTS
from mace_core.kernels.precision import PrecisionConfig
from mace_core.kernels.registry import get_backend
from mace_core.metadata import (
    ConfigRecord,
    ElectrostaticsRecord,
    HeadSummary,
    ModelMetadata,
    ParentModel,
    Provenance,
)
from mace_core.observables import (
    ObservableCatalogue,
    RequestedOutputs,
    resolve_requested,
)
from mace_core.stages import BuiltModel

from mace_torch import __version__
from mace_torch.finetune.foundation import Foundation
from mace_torch.finetune.transfer import readout_sources, transfer_foundation
from mace_torch.kernels import initialize_model_weights
from mace_torch.models import EnergyOutputHead, MACEModel, ScaleShiftSpec
from mace_torch.models.electrostatics import PolarModel, PolarSettings
from mace_torch.nn.graph_features import FeatureSpec
from mace_torch.physics import DerivativeEngine
from mace_torch.train.contracts import TorchBuiltModel, TorchDataBundle
from mace_torch.train.data_stage import DEFAULT_PRECISION

__all__ = [
    "DEFAULT_HIDDEN_IRREPS",
    "ModelStageError",
    "build_model",
    "run_model_stage",
]

#: What one channel carries when the configuration does not say. The frozen
#: tree spells the same choice as `--max_L 1`, which it then expands into a
#: multiplicity per irrep; here the multiplicity is `num_channels` and this is
#: the shape of one channel.
DEFAULT_HIDDEN_IRREPS = "0e+1o"

#: Which model spelling puts the short-range repulsion inside the scaled sum.
#: The two differ by about 77 eV on a probe geometry, so it is a fact about a
#: trained model rather than a preference.
_ZBL_INSIDE = {"scale_shift": True, "plain": False}

#: Every registered model name. The charge-aware one scales and shifts its
#: local energy as the scale-shift model does.
_MODELS = frozenset({*_ZBL_INSIDE, "polar"})

#: The model settings this stage builds one way only, and what that way is.
#: The configuration can name others, since the schema covers every setting a
#: legacy model has, and a value the model cannot build is refused rather than
#: replaced: building the default instead trains another architecture under
#: the name of the one asked for.
#:
#: What each spelling of an interaction builds: whether the layer carries its
#: skip onward for the product basis to add, rather than applying it to the
#: message; whether it normalizes by a density learned per atom rather than by
#: the average neighbour count; and whether it is the nonlinear block, which
#: conditions its convolution on both elements and gates its output. Distinct
#: models, not names for one: a residual first layer takes its skip from the
#: element embedding.
_FIRST_INTERACTIONS: dict[str, tuple[bool, bool, bool]] = {
    "RealAgnosticInteractionBlock": (False, False, False),
    "RealAgnosticResidualInteractionBlock": (True, False, False),
    "RealAgnosticDensityInteractionBlock": (False, True, False),
    "RealAgnosticDensityResidualInteractionBlock": (True, True, False),
    "RealAgnosticResidualNonLinearInteractionBlock": (True, False, True),
}

#: The same for every later layer, which always carries its skip.
_INTERACTIONS: dict[str, tuple[bool, bool]] = {
    "RealAgnosticResidualInteractionBlock": (False, False),
    "RealAgnosticDensityResidualInteractionBlock": (True, False),
    "RealAgnosticResidualNonLinearInteractionBlock": (False, True),
}

#: The configuration's spelling of a distance transform, and the backbone's.
_DISTANCE_TRANSFORMS: dict[str, Literal["none", "agnesi", "soft"]] = {
    "None": "none",
    "Agnesi": "agnesi",
    "Soft": "soft",
}

_BUILT_ONE_WAY: dict[str, tuple[object, ...]] = {
    "interaction": tuple(_INTERACTIONS),
    "interaction_first": tuple(_FIRST_INTERACTIONS),
    "radial_mlp": ((64, 64, 64),),
    "distance_transform": tuple(_DISTANCE_TRANSFORMS),
    "clebsch_gordan_basis": ("reduced",),
    "readout.gate": ("silu",),
}


class ModelStageError(RuntimeError):
    """The configuration does not describe a model that can be built."""


def run_model_stage(
    config: ResolvedConfig,
    data: TorchDataBundle,
    catalogue: ObservableCatalogue,
    *,
    precision: PrecisionConfig = DEFAULT_PRECISION,
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
        precision: What the model computes and accumulates in.
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
    if foundation is not None:
        config = _with_foundation_architecture(config, foundation)
    engine, requested = build_model(
        config,
        catalogue,
        z_table=data.z_table,
        heads=data.heads,
        e0s=data.e0s,
        statistics=data.statistics,
        precision=precision,
        supports_float64=supports_float64,
        initialize=initialize,
    )
    metadata = _metadata(config, data)
    model = engine.get_submodule("backbone")
    if isinstance(model, PolarModel):
        metadata = metadata.model_copy(
            update={"electrostatics": ElectrostaticsRecord(**model.solver_record())}
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
    return BuiltModel(
        model=engine,
        outputs=requested,
        data=data,
        metadata=metadata,
    )


#: The model settings that are the run's own even when it starts from a
#: foundation model. Everything else is architecture, and the foundation's.
_RUN_OWNED_MODEL_FIELDS = frozenset({"observables", "backend"})


def _with_foundation_architecture(
    config: ResolvedConfig, foundation: Foundation
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
    theirs = foundation.config.model
    ours = config.model
    conflicts = sorted(
        name
        for name in ours.model_fields_set - _RUN_OWNED_MODEL_FIELDS
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
    unread = sorted(set(ours.observables) - set(theirs.observables))
    if unread:
        raise ModelStageError(
            f"the run reads out {unread}, which the foundation model does not; "
            f"it reads out {list(theirs.observables)}. A new observable needs a "
            f"head nobody trained, which a fine-tune does not start from."
        )
    model = theirs.model_copy(
        update={name: getattr(ours, name) for name in _RUN_OWNED_MODEL_FIELDS}
    )
    return config.model_copy(update={"model": model})


def build_model(
    config: ResolvedConfig,
    catalogue: ObservableCatalogue,
    *,
    z_table: AtomicNumberTable,
    heads: tuple[str, ...],
    e0s: ResolvedE0s,
    statistics: DatasetStatistics,
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
    requested = resolve_requested(config.model.observables, catalogue)
    if not requested.observables:
        raise ModelStageError(
            "the configuration declares no observable, so the model would "
            "read out nothing. Declare at least one under `model.observables`."
        )
    if config.model.model not in _MODELS:
        raise ModelStageError(
            f"{config.model.model!r} is not a registered model. The choices are "
            f"{sorted(_MODELS)}: the two energy models differ in where the "
            f"short-range repulsion is added, and `polar` carries a "
            f"self-consistent density and a long-range term."
        )
    _refuse_unbuilt(config)
    _check_electrostatics(config)

    energy = next(
        (spec for spec in requested.observables if spec.name == "energy"), None
    )
    if energy is None:
        raise ModelStageError(
            "no `energy` observable is declared, and the derivative engine "
            "differentiates an energy. A model without one is an inference "
            "path this stage does not build yet."
        )
    energy_head = EnergyOutputHead(
        e0s,
        heads,
        z_table,
        _scale_shift(config, statistics, len(heads)),
        precision,
        zbl_in_scale_shift=_ZBL_INSIDE.get(config.model.model, True),
        supports_float64=supports_float64,
    )
    if config.model.model == "polar":
        model = _polar_model(
            config,
            requested,
            energy_head,
            z_table=z_table,
            heads=heads,
            statistics=statistics,
            precision=precision,
            trains_derivatives=initialize and bool(requested.derivatives),
        )
        if initialize:
            initialize_model_weights(model, config.runtime.seed)
        return DerivativeEngine(model, energy, None, inputs=catalogue.inputs), requested
    model = MACEModel(
        get_backend(config.model.backend),
        atomic_numbers=list(z_table.zs),
        observables=requested.observables,
        energy_head=energy_head,
        num_layers=config.model.num_interactions,
        num_features=config.model.num_channels,
        lmax=config.model.max_ell,
        hidden_irreps=config.model.hidden_irreps or DEFAULT_HIDDEN_IRREPS,
        num_radial=config.model.num_radial_basis,
        cutoff=config.model.r_max,
        correlation=config.model.correlation,
        avg_num_neighbors=statistics.avg_num_neighbors,
        radial_kind=config.model.radial_type,
        precision=precision.model,
        pair_repulsion=config.model.pair_repulsion,
        cutoff_order=config.model.num_cutoff_basis,
        distance_transform=_DISTANCE_TRANSFORMS[config.model.distance_transform],
        apply_cutoff=config.model.apply_cutoff,
        readout_hidden=_readout_hidden(config),
        graph_features=_graph_features(config),
        readout_last_only=config.model.readout.last_only,
        readout_biased=config.model.readout.bias,
        readout_from_embedding=config.model.readout.from_embedding,
        # One readout per head, so a head that is a different level of theory
        # has weights of its own to fit it with.
        num_heads=len(heads),
        **_interaction_settings(config)._asdict(),
        element_agnostic_product=config.model.use_agnostic_product,
    )
    if initialize:
        # Seeded from the run, so the same configuration and the same seed
        # rebuild the same model. The walk is over the model rather than the
        # engine, so wrapping it in one more layer later cannot change a
        # weight.
        initialize_model_weights(model, config.runtime.seed)
    return DerivativeEngine(model, energy, None, inputs=catalogue.inputs), requested


def _polar_model(
    config: ResolvedConfig,
    requested: RequestedOutputs,
    energy_head: EnergyOutputHead,
    *,
    z_table: AtomicNumberTable,
    heads: tuple[str, ...],
    statistics: DatasetStatistics,
    precision: PrecisionConfig,
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
        get_backend(config.model.backend),
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
        num_layers=config.model.num_interactions,
        num_features=config.model.num_channels,
        lmax=config.model.max_ell,
        hidden_irreps=config.model.hidden_irreps or DEFAULT_HIDDEN_IRREPS,
        num_radial=config.model.num_radial_basis,
        cutoff=config.model.r_max,
        correlation=config.model.correlation,
        avg_num_neighbors=statistics.avg_num_neighbors,
        radial_kind=config.model.radial_type,
        precision=precision.model,
        cutoff_order=config.model.num_cutoff_basis,
        distance_transform=_DISTANCE_TRANSFORMS[config.model.distance_transform],
        apply_cutoff=config.model.apply_cutoff,
        readout_hidden=_readout_hidden(config),
        readout_biased=config.model.readout.bias,
        readout_last_only=config.model.readout.last_only,
        num_heads=len(heads),
        **_interaction_settings(config)._asdict(),
        element_agnostic_product=config.model.use_agnostic_product,
    )


class _InteractionSettings(NamedTuple):
    """What the configured interactions build, as the backbone's settings."""

    residual_first_layer: bool
    learned_density_first_layer: bool
    nonlinear_first_layer: bool
    learned_density: bool
    nonlinear: bool
    convolution_irreps: str | None
    narrow_first_convolution: bool


def _interaction_settings(config: ResolvedConfig) -> _InteractionSettings:
    model = config.model
    return _InteractionSettings(
        *_FIRST_INTERACTIONS[model.interaction_first],
        *_INTERACTIONS[model.interaction],
        convolution_irreps=model.edge_irreps,
        narrow_first_convolution=model.use_edge_irreps_first,
    )


def _graph_features(config: ResolvedConfig) -> tuple[FeatureSpec, ...]:
    return tuple(
        FeatureSpec(
            name=feature.name,
            kind=feature.kind,
            embedding_dim=feature.embedding_dim,
            num_classes=feature.num_classes,
            input_dim=feature.input_dim,
            level="graph" if feature.per == "graph" else "node",
            offset=feature.offset,
            use_bias=feature.use_bias,
        )
        for feature in config.model.graph_features
    )


def _unbuilt_graph_features(config: ResolvedConfig) -> list[str]:
    """Embedded inputs the data path could not supply, and a readout of an
    embedding that has nothing embedded in it."""
    model = config.model
    refused = []
    for feature in model.graph_features:
        name = feature.name
        if name not in GRAPH_INPUT_DEFAULTS:
            refused.append(
                f"model.graph_features.{name} is not a per-structure input the "
                f"graphs carry; they carry {sorted(GRAPH_INPUT_DEFAULTS)}"
            )
        elif feature.per != "graph":
            refused.append(
                f"model.graph_features.{name} is per {feature.per}, and "
                f"{name} is one value per structure"
            )
        elif feature.kind == "continuous" and feature.input_dim != len(
            GRAPH_INPUT_DEFAULTS[name]
        ):
            refused.append(
                f"model.graph_features.{name} has input_dim {feature.input_dim}, "
                f"and {name} has {len(GRAPH_INPUT_DEFAULTS[name])} component(s)"
            )
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
    if model.num_interactions > 1 and not _INTERACTIONS[model.interaction][1]:
        refused.append(
            f"model.edge_irreps is {model.edge_irreps!r} with model.interaction "
            f"{model.interaction!r}; it is built for "
            f"'RealAgnosticResidualNonLinearInteractionBlock' only"
        )
    if (
        model.use_edge_irreps_first
        and not _FIRST_INTERACTIONS[model.interaction_first][2]
    ):
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
    if config.model.model == "polar":
        # The frozen tree computes the repulsion of this model and never adds
        # it, so the only faithful build is the one without it.
        built["pair_repulsion"] = (False,)
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
    polar = config.model.model == "polar"
    if settings.enabled:
        from mace_core.electrostatics import get_solver

        get_solver(settings.solver)
    if settings.enabled and not polar:
        raise ModelStageError(
            f"electrostatics.enabled is set and model.model is "
            f"{config.model.model!r}, which carries no long-range term, so the "
            f"run would train one without it. Use the `polar` model, or leave "
            f"the section out."
        )
    if polar and not settings.enabled:
        raise ModelStageError(
            "model.model is `polar`, whose long-range term is computed by the "
            "solver the electrostatics section names. Set "
            "electrostatics.enabled, with the solver and the systems it is set "
            "up for."
        )


def _scale_shift(
    config: ResolvedConfig, statistics: DatasetStatistics, num_heads: int
) -> ScaleShiftSpec:
    """The per-head scale and shift, from the one measurement.

    Every head gets the same pair because there is one statistics object for
    the run. Per-head statistics are what multi-dataset balancing brings, and
    inventing them here would put two answers in the tree for the same
    question.
    """
    if config.model.scaling == "none":
        return ScaleShiftSpec("none", (1.0,) * num_heads, (0.0,) * num_heads)
    return ScaleShiftSpec(
        config.model.scaling,
        (statistics.std,) * num_heads,
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
    was built with and nothing about which head it belonged to.
    """
    return ModelMetadata(
        config=ConfigRecord(resolved=config.model_dump(mode="json")),
        provenance=Provenance(code_version=__version__),
        heads={
            head: HeadSummary(
                e0=e0_details(
                    config.data.heads[head].e0s,
                    {
                        chemical_symbols[number]: float(energy)
                        for number, energy in data.e0s.values[head].items()
                    },
                )
            )
            for head in data.heads
        },
    )
