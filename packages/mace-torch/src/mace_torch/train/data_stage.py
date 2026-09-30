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
import logging
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path

from mace_core.config.data import DataConfig, HeadDataConfig
from mace_core.config.e0s import E0sTable
from mace_core.config.model import ModelConfig
from mace_core.config.precision import PrecisionConfig
from mace_core.config.resolved import ResolvedConfig
from mace_core.data import (
    Configuration,
    InMemoryBackend,
    KeySpecification,
    compute_statistics,
    open_dataset,
    random_train_valid_split,
    resolve_e0s,
)
from mace_core.data.backends.xyz import SUFFIXES as XYZ_SUFFIXES
from mace_core.data.e0_resolution import E0Provenance, EnergyPredictor
from mace_core.data.xyz import ISOLATED_ATOM_CONFIG_TYPE
from mace_core.elements import AtomicNumberTable, ResolvedE0s
from mace_core.observables import ObservableCatalogue, resolve_requested
from mace_core.stages import DataBundle
from torch.utils.data import DataLoader

from mace_torch.data import GraphDataset, make_loader, target_specs
from mace_torch.data.augmentation import build_augmentations
from mace_torch.data.transforms import apply_transforms
from mace_torch.finetune.foundation import FoundationContext, FoundationError
from mace_torch.finetune.ratio import RatioGuardError, repeat_count
from mace_torch.finetune.replay import read_curated
from mace_torch.finetune.subselect import SelectionError, select, split_by_filter
from mace_torch.train.contracts import TorchDataBundle
from mace_torch.train.loaders import build_training_loader

__all__ = [
    "DEFAULT_PRECISION",
    "DataStageError",
    "graph_inputs_of",
    "key_specification",
    "run_data_stage",
    "selected_structures",
]

#: What a run computes in until a precision section exists to say otherwise.
#: It is frozen, so one instance serves as every caller's default.
DEFAULT_PRECISION = PrecisionConfig()


logger = logging.getLogger(__name__)


class DataStageError(RuntimeError):
    """The data cannot be turned into a bundle a model could be built from."""


def run_data_stage(
    config: ResolvedConfig,
    catalogue: ObservableCatalogue,
    *,
    precision: PrecisionConfig = DEFAULT_PRECISION,
    predict_energy: EnergyPredictor | None = None,
    foundation: FoundationContext | None = None,
    pseudolabel: Callable[[str, list[Configuration]], list[Configuration]]
    | None = None,
) -> TorchDataBundle:
    """Read the data, resolve the E0s once, and measure the dataset.

    Args:
        config: The resolved configuration.
        catalogue: The observable declarations the requested names resolve
            against.
        precision: What the batches are bound at. An argument rather than a
            configuration section because there is no precision section yet:
            the legacy `default_dtype` flag is reserved for one, and inventing
            it here would put the schema in two places.
        predict_energy: A foundation model's energies, needed only by the
            ``estimated`` E0 kind. Injected rather than imported: fitting an E0
            is arithmetic and the model that corrects it is torch.
        foundation: What the foundation model a fine-tune starts from
            contributes: its element table, which the model is built over,
            its energies per head, for the E0 kinds that copy them, and
            descriptors for farthest-point sampling.
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
    heads = _heads(config.data)
    key_spec = key_specification(config.data)
    requested = resolve_requested(config.model.observables, catalogue)

    train: list[Configuration] = []
    valid: list[Configuration] = []
    test: list[Configuration] = []
    e0s: dict[str, dict[int, float]] = {}
    provenance: dict[str, E0Provenance] = {}

    for name, head in heads.items():
        head_train, head_valid, head_test = _read_head(
            name, head, config, key_spec, pseudolabel, foundation
        )
        train.extend(head_train)
        valid.extend(head_valid)
        test.extend(head_test)

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
    # the data holds, as they would be without a foundation model, and every
    # other element takes the energy of the foundation head its readout starts
    # from: no structure holds it, so it trains nothing, and the model stays
    # ready for it.
    keeps_foundation_table = (
        foundation is not None and config.finetune.element_table == "foundation"
    )

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
        values, record = resolve_e0s(
            head.e0s,
            z_table,
            head_train,
            foundation_e0s=_foundation_e0s(name, head, foundation),
            predict_energy=predict_energy,
        )
        if keeps_foundation_table:
            assert foundation is not None
            values, record = _with_foundation_energies(
                name, head, values, record, foundation
            )
        e0s[name] = values
        provenance[name] = record
    if keeps_foundation_table:
        assert foundation is not None
        z_table = foundation.z_table

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
        measured = "rms_dipoles"
    statistics = compute_statistics(
        InMemoryBackend(train),
        list(z_table.zs),
        config.model.r_max,
        _average_e0s(e0s, z_table) if reads_energy else {},
        scaling=measured,
    )

    specs = target_specs(requested)
    head_names = tuple(heads)
    graph_inputs = graph_inputs_of(config.model)

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
            graph_inputs=graph_inputs,
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
        z_table=z_table,
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
            name: _evaluation_loader(build(_of_head(test, name)), config, precision)
            for name in head_names
            if _of_head(test, name)
        },
    )


def graph_inputs_of(model: ModelConfig) -> tuple[str, ...]:
    """The inputs a model reads, per structure or per atom.

    The charge-aware, dipole and magnetic models', and the ones declared as
    embedded features. Written into every graph built for it, from the file or
    from their defaults.
    """
    inputs: tuple[str, ...] = ()
    if model.model == "polar":
        from mace_torch.models.electrostatics import POLAR_GRAPH_INPUTS

        inputs = POLAR_GRAPH_INPUTS
    elif model.model in ("dipole", "dielectric"):
        from mace_torch.models.dipoles import DIPOLE_GRAPH_INPUTS

        inputs = DIPOLE_GRAPH_INPUTS[
            "fixed" if model.model == "dipole" else "predicted"
        ]
    elif model.model == "magnetic":
        from mace_torch.models.magnetic import MAGNETIC_GRAPH_INPUTS

        inputs = MAGNETIC_GRAPH_INPUTS
    return inputs + tuple(
        feature.name for feature in model.graph_features if feature.name not in inputs
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


def key_specification(data: DataConfig) -> KeySpecification:
    """The default property keys, plus the graph inputs this run renamed."""
    spec = KeySpecification.from_defaults()
    keys = data.graph_input_keys
    return spec.update(
        graph_keys={
            "elec_temp": keys.elec_temp,
            "total_spin": keys.total_spin,
            "total_charge": keys.total_charge,
            "external_field": keys.external_field,
        }
    )


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
    if pseudolabel is not None:
        train = list(pseudolabel(name, train))
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
        valid = list(_open(head.valid_file, name, key_spec).iter_range())
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
    test = (
        list(_open(head.test_file, name, key_spec).iter_range())
        if head.test_file is not None
        else []
    )
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
    if head.curated is not None:
        structures = read_curated(head.curated, head=name)
        source = f"the {head.curated!r} replay dataset"
    elif head.train_file is not None:
        # The reference structures stay in: the isolated-atom E0 kind reads
        # them, and a backend that dropped them first would hand over whatever
        # its own extraction does with an unlabelled one, which is a zero.
        structures = list(_open(head.train_file, name, key_spec).iter_range())
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
    descriptors = None
    if settings.method == "fps":
        if foundation is None or foundation.describe is None:
            raise DataStageError(
                f"head {name!r} asks for farthest-point sampling, which reads "
                f"a foundation model's descriptors, and the run has none."
            )
        passed, _ = split_by_filter(structures, settings.elements, settings.filtering)
        descriptors = foundation.describe(passed)
    try:
        return select(
            structures,
            num_samples=settings.num_samples,
            method=settings.method,
            filtering=settings.filtering,
            elements=settings.elements,
            allow_random_padding=settings.allow_random_padding,
            seed=config.runtime.seed,
            descriptors=descriptors,
        )
    except SelectionError as failure:
        raise DataStageError(f"head {name!r}: {failure}") from failure


def _with_foundation_energies(
    name: str,
    head: HeadDataConfig,
    values: Mapping[int, float],
    record: E0Provenance,
    foundation: FoundationContext,
) -> tuple[dict[int, float], E0Provenance]:
    """A head's energies over the foundation model's whole element table.

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
    for z in foundation.z_table.zs:
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
    return (
        head.train_file is not None and head.train_file.suffix.lower() in XYZ_SUFFIXES
    )


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


def _open(path: Path, head: str, key_spec: KeySpecification):
    """Open one file, saying which head asked for it when it cannot be read."""
    try:
        return open_dataset(
            path, key_spec=key_spec, head=head, keep_isolated_atoms=True
        )
    except Exception as failure:
        raise DataStageError(f"head {head!r} could not read {path}: {failure}") from (
            failure
        )


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
