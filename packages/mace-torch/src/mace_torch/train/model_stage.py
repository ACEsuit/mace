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

from mace_core.config.resolved import ResolvedConfig
from mace_core.data.backend import DatasetStatistics
from mace_core.kernels.precision import PrecisionConfig
from mace_core.kernels.registry import get_backend
from mace_core.metadata import ConfigRecord, ModelMetadata, Provenance
from mace_core.observables import ObservableCatalogue, resolve_requested
from mace_core.stages import BuiltModel, DataBundle
from torch import nn
from torch.utils.data import DataLoader

from mace_torch import __version__
from mace_torch.models import EnergyOutputHead, MACEModel, ScaleShiftSpec
from mace_torch.physics import DerivativeEngine
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


class ModelStageError(RuntimeError):
    """The configuration does not describe a model that can be built."""


def run_model_stage(
    config: ResolvedConfig,
    data: DataBundle[DataLoader],
    catalogue: ObservableCatalogue,
    *,
    precision: PrecisionConfig = DEFAULT_PRECISION,
    supports_float64: bool = True,
) -> BuiltModel[nn.Module, DataLoader]:
    """Build the model the configuration describes, scaled by the data.

    Args:
        config: The resolved configuration.
        data: What the data stage resolved. Its statistics are read, never
            recomputed.
        catalogue: The observable declarations.
        precision: What the model computes and accumulates in.
        supports_float64: Whether the target device has float64 at all. A
            device that does not degrades the accumulation and says so.

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
    )
    if energy is None:
        raise ModelStageError(
            "no `energy` observable is declared, and the derivative engine "
            "differentiates an energy. A model without one is an inference "
            "path this stage does not build yet."
        )
    engine = DerivativeEngine(model, energy, None, inputs=catalogue.inputs)
    return BuiltModel(
        model=engine,
        outputs=requested,
        data=data,
        metadata=_metadata(config, data),
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


def _metadata(config: ResolvedConfig, data: DataBundle[DataLoader]) -> ModelMetadata:
    """The record written beside the weights.

    The resolved configuration goes in whole. A checkpoint that carried only
    the weights would need its run's command line to be rebuilt, and that is
    the thing least likely to still exist.
    """
    return ModelMetadata(
        config=ConfigRecord(resolved=config.model_dump(mode="json")),
        provenance=Provenance(code_version=__version__),
    )
