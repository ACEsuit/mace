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
from collections.abc import Callable, Sequence
from pathlib import Path

from mace_core.config.data import DataConfig, HeadDataConfig
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
from mace_core.kernels.precision import PrecisionConfig
from mace_core.observables import ObservableCatalogue, resolve_requested
from mace_core.stages import DataBundle
from torch.utils.data import DataLoader

from mace_torch.data import GraphDataset, make_loader, target_specs
from mace_torch.data.transforms import apply_transforms
from mace_torch.finetune.foundation import FoundationContext, FoundationError
from mace_torch.finetune.ratio import RatioGuardError, repeat_count
from mace_torch.finetune.replay import read_curated
from mace_torch.finetune.subselect import SelectionError, select, split_by_filter
from mace_torch.train.contracts import TorchDataBundle
from mace_torch.train.loaders import build_training_loader

__all__ = ["DEFAULT_PRECISION", "DataStageError", "run_data_stage"]

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
    pseudolabel: Callable[[Sequence[Configuration]], Sequence[Configuration]]
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
            before they are measured. It runs before the statistics, because a
            statistic taken over labels that are about to be replaced describes
            a dataset that never trains.

    Raises:
        DataStageError: On anything that would otherwise train a model against
            less data than the configuration asked for.
    """
    heads = _heads(config.data)
    key_spec = _key_spec(config.data)
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
        # rather than a fine-tune of what is there. Building over the whole of
        # its table instead is a different mode, the frozen tree's
        # `--foundation_model_elements`, and it obliges every head to have an
        # energy for every one of those elements.
        outside = sorted(present - {int(z) for z in foundation.z_table.zs})
        if outside:
            raise DataStageError(
                f"the data holds elements {outside} the foundation model was "
                f"not fitted for; its elements are {list(foundation.z_table.zs)}."
            )
    z_table = AtomicNumberTable(sorted(present))

    for name, head in heads.items():
        head_train = [item for item in train if item.head == name]
        values, record = resolve_e0s(
            head.e0s,
            z_table,
            head_train,
            foundation_e0s=_foundation_e0s(name, head, foundation),
            predict_energy=predict_energy,
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
    statistics = compute_statistics(
        InMemoryBackend(train),
        list(z_table.zs),
        config.model.r_max,
        _average_e0s(e0s, z_table),
        scaling=measured,
    )

    specs = target_specs(requested)
    head_names = tuple(heads)

    def build(items: Sequence[Configuration]) -> GraphDataset:
        return GraphDataset(
            list(items),
            cutoff=config.model.r_max,
            z_table=z_table,
            targets=specs,
            heads=head_names,
        )

    # Per head, because that is what the balancing decides between and what an
    # error table has a row for. The concatenation the frozen tree trains on is
    # one of the two modes, rebuilt from these rather than being the only
    # shape the stage can produce.
    train_sets = {name: build(_of_head(train, name)) for name in head_names}
    train_loader = build_training_loader(
        train_sets,
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


def _key_spec(data: DataConfig) -> KeySpecification:
    """The default property keys, plus the graph inputs this run renamed."""
    spec = KeySpecification.from_defaults()
    keys = data.graph_input_keys
    return spec.update(
        graph_keys={
            "elec_temp": keys.elec_temp,
            "total_spin": keys.total_spin,
            "total_charge": keys.total_charge,
        }
    )


def _read_head(
    name: str,
    head: HeadDataConfig,
    config: ResolvedConfig,
    key_spec: KeySpecification,
    pseudolabel: Callable[[Sequence[Configuration]], Sequence[Configuration]] | None,
    foundation: FoundationContext | None = None,
) -> tuple[list[Configuration], list[Configuration], list[Configuration]]:
    """One head's training, validation and test structures.

    A head reading a published replay dataset and one reading a file go
    through every step here alike; the source is the one line that differs.
    """
    if head.curated is not None:
        train = read_curated(head.curated, head=name)
        source = f"the {head.curated!r} replay dataset"
    elif head.train_file is not None:
        # The reference structures stay in: the isolated-atom E0 kind reads
        # them, and a backend that dropped them first would hand over whatever
        # its own extraction does with an unlabelled one, which is a zero.
        train = list(_open(head.train_file, name, key_spec).iter_range())
        source = str(head.train_file)
    else:
        raise DataStageError(
            f"head {name!r} names neither a `train_file` nor a `curated` "
            f"dataset. A head with no structures contributes nothing and would "
            f"train a readout against nothing."
        )
    if not train:
        raise DataStageError(f"head {name!r} read {source} and it is empty.")
    if head.subselect is not None:
        train = _subselect(name, head, train, config, foundation)
    if head.weight != 1.0:
        train = [
            dataclasses.replace(item, weight=item.weight * head.weight)
            for item in train
        ]
    if pseudolabel is not None:
        train = list(pseudolabel(train))
    # Before the split and before the statistics. A scale computed from
    # energies that are about to be shifted is wrong by the shift, and a
    # validation set taken before the transform would be scored on a different
    # target than the one the model trains on.
    train = apply_transforms(
        train,
        [(spec.name, spec.settings) for spec in config.data.transforms],
    )

    if head.valid_file is not None:
        valid = list(_open(head.valid_file, name, key_spec).iter_range())
    else:
        train, valid = random_train_valid_split(
            train,
            config.data.valid_fraction,
            config.runtime.seed,
            config.runtime.work_dir,
            prefix=name,
        )
    test = (
        list(_open(head.test_file, name, key_spec).iter_range())
        if head.test_file is not None
        else []
    )
    return train, valid, test


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
