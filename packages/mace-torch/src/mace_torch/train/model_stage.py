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

from ase.data import chemical_symbols
from mace_core.config.provenance import e0_details
from mace_core.config.resolved import ResolvedConfig
from mace_core.data.backend import DatasetStatistics
from mace_core.kernels.precision import PrecisionConfig
from mace_core.kernels.registry import get_backend
from mace_core.metadata import ConfigRecord, HeadSummary, ModelMetadata, Provenance
from mace_core.observables import ObservableCatalogue, resolve_requested
from mace_core.stages import BuiltModel

from mace_torch import __version__
from mace_torch.kernels import initialize_model_weights
from mace_torch.models import EnergyOutputHead, MACEModel, ScaleShiftSpec
from mace_torch.physics import DerivativeEngine
from mace_torch.train.contracts import TorchBuiltModel, TorchDataBundle
from mace_torch.train.data_stage import DEFAULT_PRECISION

__all__ = ["DEFAULT_HIDDEN_IRREPS", "ModelStageError", "run_model_stage"]

#: What one channel carries when the configuration does not say. The frozen
#: tree spells the same choice as `--max_L 1`, which it then expands into a
#: multiplicity per irrep; here the multiplicity is `num_channels` and this is
#: the shape of one channel.
DEFAULT_HIDDEN_IRREPS = "0e+1o"

#: Which model spelling puts the short-range repulsion inside the scaled sum.
#: The two differ by about 77 eV on a probe geometry, so it is a fact about a
#: trained model rather than a preference.
_ZBL_INSIDE = {"scale_shift": True, "plain": False}

#: The model settings this stage builds one way only, and what that way is.
#: The configuration can name others, since the schema covers every setting a
#: legacy model has, and a value the model cannot build is refused rather than
#: replaced: building the default instead trains another architecture under
#: the name of the one asked for.
#:
#: Both spellings of the first interaction build the same block. The first
#: layer has no incoming features to skip from, and the frozen tree builds the
#: plain block there whichever of the two it was given.
_BUILT_ONE_WAY: dict[str, tuple[object, ...]] = {
    "interaction": ("RealAgnosticResidualInteractionBlock",),
    "interaction_first": (
        "RealAgnosticInteractionBlock",
        "RealAgnosticResidualInteractionBlock",
    ),
    "radial_mlp": ((64, 64, 64),),
    "distance_transform": ("None",),
    "apply_cutoff": (True,),
    "use_agnostic_product": (False,),
    "edge_irreps": (None,),
    "use_edge_irreps_first": (False,),
    "clebsch_gordan_basis": ("reduced",),
    "readout.gate": ("silu",),
    "readout.last_only": (False,),
    "readout.from_embedding": (False,),
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

    Returns:
        The model wrapped in its derivative engine, with the bundle it was
        built from and the record that travels with the weights.
    """
    requested = resolve_requested(config.model.observables, catalogue)
    if not requested.observables:
        raise ModelStageError(
            "the configuration declares no observable, so the model would "
            "read out nothing. Declare at least one under `model.observables`."
        )
    if config.model.model not in _ZBL_INSIDE:
        raise ModelStageError(
            f"{config.model.model!r} is not a model spelling. The choices are "
            f"{sorted(_ZBL_INSIDE)}, and they differ in where the short-range "
            f"repulsion is added."
        )
    _refuse_unbuilt(config)

    energy = next(
        (spec for spec in requested.observables if spec.name == "energy"), None
    )
    energy_head = (
        None
        if energy is None
        else EnergyOutputHead(
            data.e0s,
            data.heads,
            data.z_table,
            _scale_shift(config, data.statistics, len(data.heads)),
            precision,
            zbl_in_scale_shift=_ZBL_INSIDE[config.model.model],
            supports_float64=supports_float64,
        )
    )
    model = MACEModel(
        get_backend(config.model.backend),
        atomic_numbers=list(data.z_table.zs),
        observables=requested.observables,
        energy_head=energy_head,
        num_layers=config.model.num_interactions,
        num_features=config.model.num_channels,
        lmax=config.model.max_ell,
        hidden_irreps=config.model.hidden_irreps or DEFAULT_HIDDEN_IRREPS,
        num_radial=config.model.num_radial_basis,
        cutoff=config.model.r_max,
        correlation=config.model.correlation,
        avg_num_neighbors=data.statistics.avg_num_neighbors,
        radial_kind=config.model.radial_type,
        precision=precision.model,
        pair_repulsion=config.model.pair_repulsion,
        cutoff_order=config.model.num_cutoff_basis,
        readout_hidden=_readout_hidden(config),
        # One readout per head, so a head that is a different level of theory
        # has weights of its own to fit it with.
        num_heads=len(data.heads),
    )
    if energy is None:
        raise ModelStageError(
            "no `energy` observable is declared, and the derivative engine "
            "differentiates an energy. A model without one is an inference "
            "path this stage does not build yet."
        )
    if initialize:
        # Seeded from the run, so the same configuration and the same seed
        # rebuild the same model. The walk is over the model rather than the
        # engine, so wrapping it in one more layer later cannot change a
        # weight.
        initialize_model_weights(model, config.runtime.seed)
    engine = DerivativeEngine(model, energy, None, inputs=catalogue.inputs)
    return BuiltModel(
        model=engine,
        outputs=requested,
        data=data,
        metadata=_metadata(config, data),
    )


def _refuse_unbuilt(config: ResolvedConfig) -> None:
    """Refuse every model setting this stage would not build as written.

    Raises:
        ModelStageError: Naming each such setting, its value, and what can be
            built instead.
    """
    unbuilt = []
    for path, choices in _BUILT_ONE_WAY.items():
        value: object = config.model
        for name in path.split("."):
            value = getattr(value, name)
        if value not in choices:
            allowed = " or ".join(repr(choice) for choice in choices)
            unbuilt.append(f"model.{path} is {value!r}, and only {allowed} is built")
    if unbuilt:
        raise ModelStageError(
            "the configuration asks for a model this stage cannot build, and "
            "building the default instead would train another architecture: "
            + "; ".join(unbuilt)
            + "."
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
