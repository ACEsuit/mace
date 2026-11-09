"""Bind a constructed model to training data, transferred weights and metadata."""

from __future__ import annotations

import logging

from ase.data import chemical_symbols
from mace_core.config.model import ModelConfig
from mace_core.config.precision import PrecisionConfig
from mace_core.config.provenance import e0_details
from mace_core.config.resolved import ResolvedConfig
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
    resolve_requested,
)

from mace_torch import __version__
from mace_torch.backends import (
    require_double_backward,
)
from mace_torch.finetune.foundation import Foundation
from mace_torch.finetune.transfer import readout_sources, transfer_foundation
from mace_torch.models.build import ModelBuildError, build_model
from mace_torch.train.contracts import BuiltModel, DataBundle

logger = logging.getLogger(__name__)

__all__ = [
    "run_model_stage",
    "with_foundation_architecture",
]


def run_model_stage(
    config: ResolvedConfig,
    data: DataBundle,
    catalogue: ObservableCatalogue,
    *,
    precision: PrecisionConfig | None = None,
    supports_float64: bool = True,
    initialize: bool = True,
    foundation: Foundation | None = None,
) -> BuiltModel:
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
        raise ModelBuildError(
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
    model = engine.model
    solver_record = getattr(model, "solver_record", None)
    if callable(solver_record):
        metadata = metadata.model_copy(
            update={"electrostatics": ElectrostaticsRecord(**solver_record())}
        )
    if foundation is not None:
        transfer_foundation(
            foundation.model,
            engine.model,
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
        engine=engine,
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
        ModelBuildError: Naming each setting the run set that the foundation
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
        raise ModelBuildError(
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
        raise ModelBuildError(
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
        raise ModelBuildError(
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


def _metadata(config: ResolvedConfig, data: DataBundle) -> ModelMetadata:
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
