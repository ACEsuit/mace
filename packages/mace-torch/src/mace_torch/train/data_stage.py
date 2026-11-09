"""The first stage: a configuration in, a bundle of resolved data out.

What this stage owns is everything that has to be decided before a model can be
built and must not be decided twice: which elements exist, what each head's
isolated-atom energies are, and what the dataset's scale is. The frozen tree
interleaves all three with the model build in one scope, and the consequence is
not untidiness: a statistic is taken against one set of E0s and the model's
buffer is filled from another, and nothing reports it.

Every fallback here is a refusal. An element with no resolvable E0, a head
whose file carries none of the declared properties, a dataset too small to
split: each of them is a run that would otherwise finish and report a number.
"""

from __future__ import annotations

import dataclasses
import json
import logging
import math
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path

from mace_core.config.data import DataConfig, DatasetSource, HeadDataConfig
from mace_core.config.e0s import E0sTable
from mace_core.config.precision import DEFAULT_PRECISION, PrecisionConfig
from mace_core.config.resolved import ResolvedConfig
from mace_core.data import (
    Configuration,
    DatasetStatistics,
    InMemoryBackend,
    KeySpecification,
    compute_statistics,
    open_dataset,
    random_train_valid_split,
    resolve_e0s,
)
from mace_core.data.backends.xyz import SUFFIXES as XYZ_SUFFIXES
from mace_core.data.e0_resolution import E0Provenance, EnergyPredictor
from mace_core.data.normalization import compute_normalization
from mace_core.data.xyz import ISOLATED_ATOM_CONFIG_TYPE
from mace_core.elements import AtomicNumberTable, ResolvedE0s
from mace_core.observables import (
    ObservableCatalogue,
    resolve_requested,
)
from torch.utils.data import DataLoader

from mace_torch.data import GraphDataset, make_loader, target_specs
from mace_torch.data.augmentation import build_augmentations
from mace_torch.data.inputs import key_specification
from mace_torch.data.transforms import apply_transforms
from mace_torch.finetune.foundation import FoundationContext, FoundationError
from mace_torch.finetune.ratio import RatioGuardError, repeat_count
from mace_torch.finetune.replay import read_curated
from mace_torch.finetune.subselect import (
    SelectionError,
    passes_filter,
    read_structure_descriptors,
    select,
    split_by_filter,
)
from mace_torch.train.contracts import DataBundle
from mace_torch.train.loaders import build_training_loader

__all__ = [
    "DEFAULT_PRECISION",
    "DataStageError",
    "run_data_stage",
    "selected_structures",
    "validation_structures",
]

logger = logging.getLogger(__name__)


class DataStageError(RuntimeError):
    """The data cannot be turned into a bundle a model could be built from."""


def run_data_stage(
    config: ResolvedConfig,
    catalogue: ObservableCatalogue,
    *,
    precision: PrecisionConfig | None = None,
    predict_energy: EnergyPredictor | None = None,
    foundation: FoundationContext | None = None,
    pseudolabel: Callable[[str, list[Configuration]], list[Configuration]]
    | None = None,
) -> DataBundle:
    """Read the data, resolve the E0s once, and measure the dataset.

    Args:
        config: The resolved configuration.
        catalogue: The observable declarations the requested names resolve
            against.
        precision: Optional override for batch binding. Otherwise the
            configuration's per-op precision is used.
        predict_energy: A foundation model's energies, needed only by the
            ``estimated`` E0 kind. Injected rather than imported: fitting an E0
            is arithmetic and the model that corrects it is torch.
        foundation: What the foundation model a fine-tune starts from
            contributes: its element table, which the model is built over,
            its energies per head, for the E0 kinds that copy them, and
            structure_descriptors for farthest-point sampling.
        pseudolabel: The seam a fine-tune uses to relabel replay structures
            before they are measured, called with each head's name and its
            structures. It runs before the statistics, because a statistic
            taken over labels that are about to be replaced describes a dataset
            that never trains, and before the head's weight, so what it writes
            does not depend on it.

    Raises:
        DataStageError: On anything that would otherwise train a model against
            less data than the configuration asked for.
    """
    precision = precision or config.precision
    heads = _heads(config.data)
    requested = resolve_requested(config.model.observables, catalogue)
    key_spec = key_specification(config.data, requested, model=config.model)

    train: list[Configuration] = []
    valid: list[Configuration] = []
    test: list[Configuration] = []
    e0s: dict[str, dict[int, float]] = {}
    provenance: dict[str, E0Provenance] = {}

    for name, head in heads.items():
        head_train, head_valid, head_test = _read_head(
            name, head, config, key_spec, pseudolabel, foundation
        )
        train.extend(_type_weighted(head_train, head.config_type_weights))
        valid.extend(_type_weighted(head_valid, head.config_type_weights))
        test.extend(_type_weighted(head_test, head.config_type_weights))

    test_sets = _test_sets(test, config, key_spec)

    present = {int(z) for item in train for z in item.atomic_numbers}
    if not present:
        raise DataStageError(
            "the training set holds no structures, so there is no element "
            "table to build a model from."
        )
    if foundation is not None:
        # The elements the data holds, which have to be among the foundation
        # model's: a fine-tune is built over them and takes the foundation's
        # weights for each. An element outside its table would need an
        # embedding row nobody trained, which is new-species initialization
        # rather than a fine-tune of what is there.
        outside = sorted(present - {int(z) for z in foundation.z_table.zs})
        if outside:
            raise DataStageError(
                f"the data holds elements {outside} the foundation model was "
                f"not fitted for; its elements are {list(foundation.z_table.zs)}."
            )
    z_table = AtomicNumberTable(sorted(present))
    # A fine-tune keeps the foundation model's whole element table unless the
    # run asks for the data's, so a model fine-tuned on two elements can be
    # fine-tuned again on a third. The energies are resolved over the elements
    # each head's data holds. Other elements take the energy of the foundation
    # head its readout starts from: no training structure in this head holds
    # them, and the model stays ready for them.
    keeps_foundation_table = (
        foundation is not None and config.finetune.element_table == "foundation"
    )
    if keeps_foundation_table:
        assert foundation is not None
        z_table = foundation.z_table

    # A model that reads out no energy has no isolated-atom energies to be
    # shifted by, and a declaration of them is refused when the configuration
    # is validated. Its heads carry none.
    reads_energy = any(spec.name == "energy" for spec in requested.observables)

    for name, head in heads.items():
        if not reads_energy:
            e0s[name] = {}
            provenance[name] = E0Provenance(kind="none")
            continue
        head_train = [item for item in train if item.head == name]
        # Another head's elements do not require references in this head's
        # dataset. Unseen elements keep this head's foundation references.
        energy_elements = (
            AtomicNumberTable(
                sorted({int(z) for item in head_train for z in item.atomic_numbers})
            )
            if foundation is not None
            else z_table
        )
        values, record = resolve_e0s(
            head.e0s,
            energy_elements,
            head_train,
            foundation_e0s=_foundation_e0s(name, head, foundation),
            predict_energy=predict_energy,
        )
        if foundation is not None:
            values, record = _with_foundation_energies(
                name, head, values, record, foundation, z_table
            )
        e0s[name] = values
        provenance[name] = record

    # After the energies have been read off them, and only for the heads that
    # did not ask to keep them. An isolated atom left in the training set is a
    # structure with no neighbours whose energy the model is then asked to
    # reproduce on top of the reference it just became.
    train = [
        item
        for item in train
        if heads[item.head].keep_isolated_atoms
        or item.config_type != ISOLATED_ATOM_CONFIG_TYPE
    ]

    if config.data.ratio_guard is not None:
        train = _guard_ratio(train, config, heads)

    resolved = ResolvedE0s(
        {name: dict(values) for name, values in e0s.items()},
    )
    # `none` is a decision about the model, not about the dataset: the spread
    # is still measured, and the model stage is what ignores it and uses one.
    # Measuring nothing instead would leave a checkpoint unable to say what the
    # data looked like.
    measured = "std" if config.model.scaling == "none" else config.model.scaling
    if not reads_energy:
        # The one spread a model without an energy has data for.
        measured = "rms_dipoles" if "dipole" in config.model.observables else "none"
    elif set(config.model.normalization.get("energy", {})) == set(heads):
        # Every energy head has its own recipe; the aggregate is used only
        # for neighbour counts and must not require unrequested forces.
        measured = "none"
    energies = _average_e0s(e0s, z_table) if reads_energy else {}
    if config.data.statistics_file is None:
        statistics = compute_statistics(
            InMemoryBackend(train),
            list(z_table.zs),
            config.model.r_max,
            energies,
            scaling=measured,
        )
    else:
        from mace_core.data.backends.hdf5_legacy import read_legacy_statistics

        path = config.data.statistics_file
        try:
            text = Path(path).read_text()
            statistics = (
                DatasetStatistics.from_json(text)
                if "normalization_scale" in json.loads(text)
                else read_legacy_statistics(path)
            )
        except Exception as failure:
            raise DataStageError(f"data.statistics_file {path}: {failure}") from failure
        expected = dict(
            r_max=config.model.r_max,
            atomic_numbers=list(z_table.zs),
            atomic_energies=energies,
        )
        for name, value in expected.items():
            if getattr(statistics, name) != value:
                raise DataStageError(
                    f"data.statistics_file {path}: {name} disagrees with the "
                    "resolved data/model; regenerate statistics using these inputs"
                )
        for name in ("mean", "normalization_scale", "avg_num_neighbors"):
            value = getattr(statistics, name)
            if not math.isfinite(value) or (name != "mean" and value < 0):
                raise DataStageError(
                    f"data.statistics_file {path}: {name} must be finite"
                    + (" and nonnegative" if name != "mean" else "")
                )

    specs = target_specs(requested)
    head_names = tuple(heads)

    augmentations = build_augmentations(
        [(spec.name, spec.settings) for spec in config.data.augmentations]
    )

    def build(items: Sequence[Configuration], augment: bool = False) -> GraphDataset:
        return GraphDataset(
            list(items),
            cutoff=config.model.r_max,
            z_table=z_table,
            targets=specs,
            heads=head_names,
            features=config.model.feature_storages,
            augmentations=augmentations if augment else (),
        )

    # Per head, because that is what the balancing decides between and what an
    # error table has a row for. The concatenation the frozen tree trains on is
    # one of the two modes, rebuilt from these rather than being the only
    # shape the stage can produce.
    train_sets = {name: build(_of_head(train, name)) for name in head_names}
    train_loader = build_training_loader(
        {name: build(_of_head(train, name), augment=True) for name in head_names},
        mode=config.training.head_balancing,
        batch_size=config.training.batch_size,
        seed=config.runtime.seed,
        float_dtype=precision.model,
        num_workers=config.data.num_workers,
        pin_memory=config.data.pin_memory,
    )
    return DataBundle(
        z_table=z_table,
        heads=head_names,
        e0s=resolved,
        e0_provenance=provenance,
        statistics=statistics,
        normalization=compute_normalization(
            train, requested.observables, config.model.normalization, resolved
        )
        if foundation is None
        else {},
        train_loader=train_loader,
        valid_loaders={
            name: _evaluation_loader(build(_of_head(valid, name)), config, precision)
            for name in head_names
        },
        train_eval_loaders={
            name: _evaluation_loader(train_sets[name], config, precision)
            for name in head_names
        },
        test_loaders={
            name: _evaluation_loader(build(items), config, precision)
            for name, items in test_sets.items()
        },
    )


def _of_head(items: Sequence[Configuration], head: str) -> list[Configuration]:
    return [item for item in items if item.head == head]


def _heads(data: DataConfig) -> dict[str, HeadDataConfig]:
    """The heads, refusing a configuration that names none."""
    if not data.heads:
        raise DataStageError(
            "no head is configured, so there is nothing to train on. A "
            "single-dataset run still has one head; give it a `train_file`."
        )
    return dict(data.heads)


def _read_head(
    name: str,
    head: HeadDataConfig,
    config: ResolvedConfig,
    key_spec: KeySpecification,
    pseudolabel: Callable[[str, list[Configuration]], list[Configuration]] | None,
    foundation: FoundationContext | None = None,
) -> tuple[list[Configuration], list[Configuration], list[Configuration]]:
    """One head's training, validation and test structures.

    A head reading a published replay dataset and one reading a file go
    through every step here alike; the source is the one line that differs.
    """
    train = selected_structures(name, head, config, key_spec, foundation)
    valid = validation_structures(name, head, config, key_spec)
    if pseudolabel is not None:
        train_count = len(train)
        labelled = list(pseudolabel(name, [*train, *valid]))
        train, valid = labelled[:train_count], labelled[train_count:]
    if head.weight != 1.0:
        train = [
            dataclasses.replace(item, weight=item.weight * head.weight)
            for item in train
        ]
    # Before the split and before the statistics. A scale computed from
    # energies that are about to be shifted is wrong by the shift, and a
    # validation set taken before the transform would be scored on a different
    # target than the one the model trains on.
    train = apply_transforms(
        train,
        [(spec.name, spec.settings) for spec in config.data.transforms],
    )

    # The isolated atoms are references, not structures to fit, so they stay
    # out of the split and are never validated on. Split with them in, one
    # lands in the validation set often enough to matter, and the E0 read
    # from the training set then has no energy for that element: measured,
    # four seeds in ten on a file of eight waters and two isolated atoms. The
    # frozen tree takes them out before it splits too, so what is split is
    # the same set it splits.
    references = [
        item for item in train if item.config_type == ISOLATED_ATOM_CONFIG_TYPE
    ]
    structures = [
        item for item in train if item.config_type != ISOLATED_ATOM_CONFIG_TYPE
    ]
    if head.valid_file is not None:
        train = structures
    else:
        train, valid = random_train_valid_split(
            structures,
            config.data.valid_fraction,
            config.runtime.seed,
            config.runtime.work_dir,
            prefix=name,
        )
    train = references + train
    test = _read(head.test_file, name, key_spec) if head.test_file is not None else []
    return train, valid, test


def selected_structures(
    name: str,
    head: HeadDataConfig,
    config: ResolvedConfig,
    key_spec: KeySpecification,
    foundation: FoundationContext | None = None,
) -> list[Configuration]:
    """A head's structures as its source holds them, after its subselection.

    Before its weight, its pseudolabels, its transforms and its split, which is
    what makes it a boundary a fine-tune can stop at and start again from: a
    head reading the written structures with no subselection goes through the
    rest exactly as this one would.
    """
    if name in config.pseudolabelled_heads:
        key_spec = dataclasses.replace(key_spec, label_names=())
    if head.curated is not None:
        structures = read_curated(head.curated, head=name)
        source = f"the {head.curated!r} replay dataset"
    elif head.train_file is not None:
        # The reference structures stay in: the isolated-atom E0 kind reads
        # them, and a backend that dropped them first would hand over whatever
        # its own extraction does with an unlabelled one, which is a zero.
        structures = _read(head.train_file, name, key_spec)
        source = str(head.train_file)
    else:
        raise DataStageError(
            f"head {name!r} names neither a `train_file` nor a `curated` "
            f"dataset. A head with no structures contributes nothing and would "
            f"train a readout against nothing."
        )
    if not structures:
        raise DataStageError(f"head {name!r} read {source} and it is empty.")
    if head.subselect is not None:
        structures = _subselect(name, head, structures, config, foundation)
    return structures


def validation_structures(
    name: str,
    head: HeadDataConfig,
    config: ResolvedConfig,
    key_spec: KeySpecification,
) -> list[Configuration]:
    """Explicit validation structures, before applying their label source."""
    if head.valid_file is None:
        return []
    if name in config.pseudolabelled_heads:
        key_spec = dataclasses.replace(key_spec, label_names=())
    return _read(head.valid_file, name, key_spec)


def _subselect(
    name: str,
    head: HeadDataConfig,
    structures: list[Configuration],
    config: ResolvedConfig,
    foundation: FoundationContext | None,
) -> list[Configuration]:
    """The part of a head's structures its subselection keeps.

    Before the split, as the frozen tree does it: the kept structures are then
    divided into training and validation like any other head's.
    """
    settings = head.subselect
    assert settings is not None
    structure_descriptors = None
    passed, _ = split_by_filter(structures, settings.elements, settings.filtering)
    if settings.structure_descriptors_file is not None:
        try:
            structure_descriptors = read_structure_descriptors(
                settings.structure_descriptors_file, len(structures)
            )
        except SelectionError as failure:
            raise DataStageError(f"head {name!r}: {failure}") from failure
        structure_descriptors = structure_descriptors[
            [
                passes_filter(
                    item.atomic_numbers, settings.elements, settings.filtering
                )
                for item in structures
            ]
        ]
    elif (
        settings.method == "fps"
        and settings.num_samples is not None
        and 0 < settings.num_samples < len(passed)
    ):
        if foundation is None or foundation.structure_descriptors is None:
            raise DataStageError(
                f"head {name!r} asks for farthest-point sampling, which reads "
                f"a foundation model's structure descriptors, and the run has none."
            )
        structure_descriptors = foundation.structure_descriptors(passed)
    try:
        return select(
            structures,
            num_samples=settings.num_samples,
            method=settings.method,
            filtering=settings.filtering,
            elements=settings.elements,
            allow_random_padding=settings.allow_random_padding,
            seed=config.runtime.seed,
            structure_descriptors=structure_descriptors,
        )
    except SelectionError as failure:
        raise DataStageError(f"head {name!r}: {failure}") from failure


def _with_foundation_energies(
    name: str,
    head: HeadDataConfig,
    values: Mapping[int, float],
    record: E0Provenance,
    foundation: FoundationContext,
    z_table: AtomicNumberTable,
) -> tuple[dict[int, float], E0Provenance]:
    """A head's energies over the fine-tune's selected element table.

    The elements the data holds keep what the head's declaration resolved to.
    Every other one takes the energy of the foundation head the head's readout
    starts from, or the value a table declaration gives it outright.
    """
    try:
        source = foundation.e0_table(head.readout_from)
    except FoundationError as failure:
        raise DataStageError(f"head {name!r}: {failure}") from failure
    given = head.e0s.values if isinstance(head.e0s, E0sTable) else {}
    filled = dict(values)
    taken = []
    for z in z_table.zs:
        if z in filled:
            continue
        if z in given:
            filled[z] = float(given[z])
        else:
            filled[z] = float(source[z])
            taken.append(z)
    return filled, dataclasses.replace(record, from_foundation=tuple(taken))


def _foundation_e0s(
    name: str, head: HeadDataConfig, foundation: FoundationContext | None
):
    """The foundation table a head's E0 kind copies, when it copies one."""
    if foundation is None or head.e0s.kind not in {"foundation", "estimated"}:
        return None
    try:
        return foundation.e0_table(getattr(head.e0s, "head", None))
    except FoundationError as failure:
        raise DataStageError(f"head {name!r}: {failure}") from failure


def _reads_in_memory(head: HeadDataConfig) -> bool:
    """Whether a head's structures are all held as a list, rather than streamed.

    The frozen tree applies the ratio guard only when every head is an
    ase-readable file, and skips it silently otherwise
    (``mace/cli/run_train.py:436-461``): repeating a streamed database's
    entries would mean copying the database. The same rule, stated.
    """
    if head.curated is not None:
        return True
    source = head.train_file
    if source is None:
        return False
    paths = source if isinstance(source, tuple) else (source,)
    return all(path.suffix.lower() in XYZ_SUFFIXES for path in paths)


def _guard_ratio(
    train: list[Configuration],
    config: ResolvedConfig,
    heads: dict[str, HeadDataConfig],
) -> list[Configuration]:
    """Repeat the other heads when the reference head outnumbers them.

    After the isolated atoms are gone, since those are references rather than
    structures to fit, and the frozen tree counts its training collections
    without them.
    """
    guard = config.data.ratio_guard
    assert guard is not None
    streamed = sorted(
        name for name, head in heads.items() if not _reads_in_memory(head)
    )
    if streamed:
        logger.info(
            "The ratio guard is skipped: heads %s are streamed databases, and "
            "repeating their entries would mean copying the database.",
            streamed,
        )
        return train
    reference = sum(1 for item in train if item.head == guard.reference)
    others = len(train) - reference
    try:
        copies = repeat_count(reference, others, guard.threshold)
    except RatioGuardError as failure:
        raise DataStageError(str(failure)) from failure
    if copies == 1:
        return train
    logger.warning(
        "The heads other than %r hold %d structures against its %d, below the "
        "ratio %s; each of them is repeated %d times.",
        guard.reference,
        others,
        reference,
        guard.threshold,
        copies,
    )
    kept = [item for item in train if item.head == guard.reference]
    repeated = [item for item in train if item.head != guard.reference]
    return kept + repeated * copies


def _read(
    source: DatasetSource, head: str, key_spec: KeySpecification
) -> list[Configuration]:
    """Concatenate requested datasets, retaining their order and source errors."""
    items = []
    for path in source if isinstance(source, tuple) else (source,):
        try:
            dataset = open_dataset(
                path, key_spec=key_spec, head=head, keep_isolated_atoms=True
            )
            if not len(dataset):
                raise ValueError("the dataset holds no structures")
            items.extend(dataset.iter_range())
        except Exception as failure:
            raise DataStageError(
                f"head {head!r} could not read {path}: {failure}"
            ) from failure
    return items


def _type_weighted(
    configurations: Sequence[Configuration], weights: Mapping[str, float]
) -> list[Configuration]:
    """Apply run-specific multipliers once, after source selection/relabelling.

    Dataset weights may already include preparation-time weighting. These
    explicit run settings multiply the stored weights for every data format.
    """
    return [
        dataclasses.replace(item, weight=item.weight * weights[item.config_type])
        if item.config_type in weights
        else item
        for item in configurations
    ]


def _test_sets(
    test: Sequence[Configuration], config: ResolvedConfig, key_spec: KeySpecification
) -> dict[str, list[Configuration]]:
    sets = {
        name: items for name in config.data.heads if (items := _of_head(test, name))
    }
    directory = config.data.test_dir
    if directory is None:
        return sets
    try:
        paths = sorted(
            path for path in directory.iterdir() if not path.name.startswith(".")
        )
    except OSError as failure:
        raise DataStageError(f"data.test_dir {directory}: {failure}") from failure
    if not paths:
        raise DataStageError(f"data.test_dir {directory} holds no test datasets")
    reserved = {
        f"{split}_{head}" for head in config.data.heads for split in ("train", "valid")
    }
    for head in config.data.heads:
        for path in paths:
            name = f"{head}_{path.name if path.is_dir() else path.stem}"
            if name in sets or name in reserved:
                raise DataStageError(
                    f"test set {path} produces duplicate row {name!r}; "
                    "rename the dataset or head"
                )
            try:
                sets[name] = _type_weighted(
                    _read(path, head, key_spec),
                    config.data.heads[head].config_type_weights,
                )
            except DataStageError as failure:
                raise DataStageError(
                    f"data.test_dir {directory}: {failure}"
                ) from failure
    return sets


def _average_e0s(
    e0s: dict[str, dict[int, float]], z_table: AtomicNumberTable
) -> dict[int, float]:
    """One reference per element for the statistics, averaged over the heads.

    The statistics are a property of the training set as a whole, and with
    several heads the set has several references per element. Averaging them is
    the only choice that does not privilege one head's level of theory; with
    one head, which is every single-dataset run, it is that head's own values.
    """
    return {
        int(number): sum(values[int(number)] for values in e0s.values()) / len(e0s)
        for number in z_table.zs
    }


def _evaluation_loader(
    dataset: GraphDataset, config: ResolvedConfig, precision: PrecisionConfig
) -> DataLoader:
    """One plain pass over a dataset, in file order.

    No shuffling and no balancing: an evaluation visits every structure once,
    and the order it visits them in cannot change a mean.
    """
    return make_loader(
        dataset,
        batch_size=config.training.valid_batch_size,
        shuffle=False,
        float_dtype=precision.model,
        num_workers=config.data.num_workers,
        pin_memory=config.data.pin_memory,
    )
