"""Training datasets and E0 sources, grouped by head in ``HeadDataConfig``.

Label keys follow the data key convention. Graph-level inputs such as electronic
temperature and total charge have configurable file keys.
"""

from __future__ import annotations

from pathlib import Path
from typing import Annotated, Any, Literal

from pydantic import Field, model_validator

from mace_core.config.base import ConfigSection
from mace_core.config.e0s import E0sIsolatedAtoms, E0Spec
from mace_core.config.loss import LossWeights

__all__ = [
    "CURATED_DATASETS",
    "AugmentationSpec",
    "CuratedDataset",
    "DataConfig",
    "DatasetSource",
    "GraphInputKeys",
    "HeadDataConfig",
    "RatioGuardConfig",
    "SubselectConfig",
    "TransformSpec",
]

#: The replay datasets published beside the foundation models, by the name a
#: head selects them with. Where each one is downloaded from is the loader's
#: business; which names exist is the schema's, so a misspelled one is refused
#: when the run is configured rather than when the download fails.
CuratedDataset = Literal["mp", "omat", "matpes_pbe", "matpes_r2scan"]

#: The same names, for a caller that enumerates them.
CURATED_DATASETS: tuple[str, ...] = ("mp", "omat", "matpes_pbe", "matpes_r2scan")

# A scalar keeps existing configs and metadata stable; sequences preserve order.
DatasetSource = Path | Annotated[tuple[Path, ...], Field(min_length=1)]


class GraphInputKeys(ConfigSection):
    """File keys of the graph-level inputs, when a dataset spells them its way.

    Only the inputs. A label's key is the data contract's and is not settable
    per run; these are declared features, and a feature nobody declared
    is not read.

    Args:
        elec_temp: Electronic temperature, per structure.
        total_spin: Total spin, per structure.
        total_charge: Total charge, per structure.
        external_field: The applied electric field, per structure.
    """

    elec_temp: str = "elec_temp"
    total_spin: str = "total_spin"
    total_charge: str = "total_charge"
    external_field: str = "external_field"


class TransformSpec(ConfigSection):
    """One data transform, by the name it is registered under.

    A name and a settings mapping rather than a kind written as a key, because
    the transforms are an open registry: a package this schema never heard of
    registers one, and a closed union could not name it.

    Args:
        name: The registered name.
        settings: What to pass the transform's factory. Validated by the
            factory when the run is configured, not when the data is read.
    """

    name: str
    settings: dict[str, Any] = Field(default_factory=dict)


class AugmentationSpec(ConfigSection):
    """One training-data augmentation, by the name it is registered under.

    Drawn afresh every time a training structure is read, where a transform
    rewrites the data once. Named and configured as a transform is.

    Args:
        name: The registered name.
        settings: What to pass the augmentation's factory.
    """

    name: str
    settings: dict[str, Any] = Field(default_factory=dict)


class SubselectConfig(ConfigSection):
    """Keeping a representative part of a head's structures.

    A replay set is large, and training on all of it would make the fine-tune
    about the replay set. So a head can keep a subset, chosen either at random
    or by farthest-point sampling over a foundation model's descriptors, after
    a filter on which elements the structures contain.

    The defaults are the frozen tree's **in-run** ones, which are not its
    standalone selection script's: ``--subselect_pt`` defaults to ``random``
    and ``--filter_type_pt`` to ``none`` (``mace/tools/arg_parser.py:603-613``),
    while the script defaults to farthest-point sampling and ``combinations``.
    The library function carries the script's.

    Args:
        num_samples: How many structures to keep. ``None`` keeps every one
            that passes the filter.
        method: ``random``, or ``fps`` for farthest-point sampling.
        structure_descriptors_file: Numeric ``.npy`` matrix
            ``[structures, features]``
            for FPS, in source order before filtering. Rows must cover every
            structure across all source files. Otherwise the foundation model
            computes them. Object arrays and pickle files are not accepted.
        filtering: Which structures may be kept at all, by the elements in
            them: ``none``; ``combinations``, only elements of the set;
            ``exclusive``, exactly the set; ``inclusive``, at least the set.
        elements: The atomic numbers the filter reads. Required by every
            filter but ``none``.
        allow_random_padding: When fewer structures pass the filter than
            ``num_samples``, make up the rest at random from those that did
            not. A positive setting: legacy spells it as a flag that turns it
            off (``--disallow_random_padding_pt``), which reads backwards.
    """

    num_samples: int | None = 10000
    method: Literal["random", "fps"] = "random"
    structure_descriptors_file: Path | None = None
    filtering: Literal["none", "combinations", "exclusive", "inclusive"] = "none"
    elements: tuple[int, ...] = ()
    allow_random_padding: bool = True

    @model_validator(mode="after")
    def _check(self) -> SubselectConfig:
        if self.structure_descriptors_file is not None and self.method != "fps":
            raise ValueError(
                "subselect.structure_descriptors_file requires method='fps'"
            )
        if self.num_samples is not None and self.num_samples < 1:
            raise ValueError(
                f"subselect.num_samples is {self.num_samples}. Keep at least "
                f"one structure, or set it to null to keep them all."
            )
        if self.filtering != "none" and not self.elements:
            raise ValueError(
                f"subselect.filtering is {self.filtering!r} and names no "
                f"elements to filter by. Give them as subselect.elements, or "
                f"set filtering to 'none'."
            )
        return self


class RatioGuardConfig(ConfigSection):
    """Repeating the other heads when one head outnumbers them all.

    The frozen tree's guard for a replay set that dwarfs the fine-tuning data
    (``mace/cli/run_train.py:443-457``): when the other heads' structures,
    together, number fewer than ``threshold`` times this head's, each of them
    is repeated. Stated against a named head rather than against the replay
    head, so nothing reads which head is the replay one; the configuration
    says which is the reference, and that is all.

    Args:
        reference: The head the others are measured against.
        threshold: The ratio below which the others are repeated. Legacy's
            ``--real_pt_data_ratio_threshold``.
    """

    reference: str
    threshold: float = 0.1


class HeadDataConfig(ConfigSection):
    """One level of theory: its structures, and where its E0s come from.

    Args:
        train_file: One dataset path or an ordered, nonempty list of paths.
            Files and dataset directories may use different registered formats.
        valid_file: Its validation structures. ``None`` splits them out of
            ``train_file`` by ``valid_fraction``.
        test_file: Structures evaluated after training. Like ``train_file`` and
            ``valid_file``, a list is concatenated in its declared order.
        e0s: Where the isolated-atom energies come from. Per head because two
            levels of theory do not share them, which is the whole reason a
            head exists.
        config_type_weights: Per ``config_type`` weight in the loss. A type
            absent from the mapping weighs ``1.0``.
        keep_isolated_atoms: Whether the ``IsolatedAtom`` structures stay in
            the training set after their energies have been read out of it.
        curated: A published replay dataset to read instead of ``train_file``.
            A head names one source or the other, and a replay head differs
            from any other head in nothing but this.
        subselect: Keep only part of the head's structures.
        weight: Multiplies every structure's weight in this head, which is how
            a replay head is made to count less than the data the fine-tune is
            for. Legacy's ``--weight_pt_head``.
        readout_from: The foundation model head this head's readout starts
            from. ``None`` takes the foundation model's only head, and is an
            error when it has several.
    """

    train_file: DatasetSource | None = None
    valid_file: DatasetSource | None = None
    test_file: DatasetSource | None = None
    e0s: E0Spec = E0sIsolatedAtoms()
    config_type_weights: LossWeights = Field(default_factory=dict)
    keep_isolated_atoms: bool = False
    curated: CuratedDataset | None = None
    subselect: SubselectConfig | None = None
    weight: float = Field(default=1.0, ge=0, allow_inf_nan=False)
    readout_from: str | None = None

    @model_validator(mode="after")
    def _one_source(self) -> HeadDataConfig:
        if self.curated is not None and self.train_file is not None:
            raise ValueError(
                f"a head names both train_file ({self.train_file}) and the "
                f"curated dataset {self.curated!r}. A head reads one source; "
                f"drop one of the two."
            )
        if self.weight < 0:
            raise ValueError(
                f"a head's weight is {self.weight}. A negative weight trains "
                f"the model away from the head's labels."
            )
        return self


class DataConfig(ConfigSection):
    """The datasets, the heads, and how they are loaded.

    Args:
        heads: Level of theory to its data. A single-head run writes one
            entry; there is no separate single-head shape, so nothing has to
            be rewritten to add a second head.
        valid_fraction: Fraction of a head's training set held out when it
            declares no ``valid_file``.
        test_dir: A directory of test datasets. Each non-hidden child is opened
            through the backend registry, in sorted order, for every head.
            Files and shard directories produce separate ``<head>_<stem>`` rows
            (directories keep their full name). Duplicate row names raise.
        statistics_file: A precomputed statistics document. Statistics
            normally travel with the prepared shards; this is the path for a
            legacy `statistics.json` and for a dataset prepared elsewhere.
            A recorded normalization method must match the scale the run uses.
        graph_input_keys: File keys of the declared graph-level inputs.
        embedding_specs: User-declared input features, by name. Declaring one
            is what makes a quantity that is neither a position nor a label
            readable, and therefore what the derivative grammar can be asked
            for a derivative against.
        num_workers: Loader worker processes.
        pin_memory: Whether the loader pins its batches.
        transforms: The data transforms, in the order they apply. Order is
            part of the meaning: shifting energies and then masking on a
            threshold is not the same run as masking and then shifting.
        augmentations: The training-data augmentations, in the order they
            apply, drawn again on every read of a training structure.
        ratio_guard: Repeat the other heads when one outnumbers them all.
        skip_evaluate_heads: Heads left out of the evaluation tables, for a
            replay head whose errors are not the run's subject. Matched against
            the head and not against the row's name, so a head whose name is
            part of another's does not take it with it.
    """

    heads: dict[str, HeadDataConfig] = Field(default_factory=dict)
    valid_fraction: float = Field(default=0.1, ge=0, lt=1, allow_inf_nan=False)
    test_dir: Path | None = None
    statistics_file: Path | None = None
    graph_input_keys: GraphInputKeys = GraphInputKeys()
    embedding_specs: dict[str, dict[str, str]] = Field(default_factory=dict)
    num_workers: int = Field(default=0, ge=0)
    pin_memory: bool = True
    skip_evaluate_heads: tuple[str, ...] = ()
    transforms: tuple[TransformSpec, ...] = ()
    augmentations: tuple[AugmentationSpec, ...] = ()
    ratio_guard: RatioGuardConfig | None = None

    @model_validator(mode="after")
    def _guard_names_a_head(self) -> DataConfig:
        if (
            self.ratio_guard is not None
            and self.ratio_guard.reference not in self.heads
        ):
            raise ValueError(
                f"data.ratio_guard.reference is {self.ratio_guard.reference!r}, "
                f"which is not a head. The heads are {sorted(self.heads)}."
            )
        return self
