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
from mace_core.data.e0_resolution import E0Provenance, EnergyPredictor
from mace_core.data.xyz import ISOLATED_ATOM_CONFIG_TYPE
from mace_core.elements import AtomicNumberTable, ResolvedE0s
from mace_core.kernels.precision import PrecisionConfig
from mace_core.observables import ObservableCatalogue, resolve_requested
from mace_core.stages import DataBundle
from torch.utils.data import DataLoader

from mace_torch.data import GraphDataset, make_loader, target_specs
from mace_torch.data.transforms import apply_transforms

__all__ = ["DEFAULT_PRECISION", "DataStageError", "run_data_stage"]

#: What a run computes in until a precision section exists to say otherwise.
#: It is frozen, so one instance serves as every caller's default.
DEFAULT_PRECISION = PrecisionConfig()


class DataStageError(RuntimeError):
    """The data cannot be turned into a bundle a model could be built from."""


def run_data_stage(
    config: ResolvedConfig,
    catalogue: ObservableCatalogue,
    *,
    precision: PrecisionConfig = DEFAULT_PRECISION,
    predict_energy: EnergyPredictor | None = None,
    foundation_e0s: dict[int, float] | None = None,
    pseudolabel: Callable[[Sequence[Configuration]], Sequence[Configuration]]
    | None = None,
) -> DataBundle[DataLoader]:
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
        foundation_e0s: A foundation model's table, for the kinds that copy it.
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
            name, head, config, key_spec, pseudolabel
        )
        train.extend(head_train)
        valid.extend(head_valid)
        test.extend(head_test)

    z_table = AtomicNumberTable(
        sorted({int(z) for item in train for z in item.atomic_numbers})
    )
    if not z_table.zs:
        raise DataStageError(
            "the training set holds no structures, so there is no element "
            "table to build a model from."
        )

    for name, head in heads.items():
        head_train = [item for item in train if item.head == name]
        values, record = resolve_e0s(
            head.e0s,
            z_table,
            head_train,
            foundation_e0s=foundation_e0s,
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
    loaders = {
        "train": _loader(
            train, config, z_table, specs, head_names, precision, shuffle=True
        ),
        "valid": _loader(
            valid, config, z_table, specs, head_names, precision, shuffle=False
        ),
    }
    test_loader = (
        _loader(test, config, z_table, specs, head_names, precision, shuffle=False)
        if test
        else None
    )
    return DataBundle(
        z_table=z_table,
        heads=head_names,
        e0s=resolved,
        e0_provenance=provenance,
        statistics=statistics,
        train_loader=loaders["train"],
        valid_loader=loaders["valid"],
        test_loader=test_loader,
    )


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
) -> tuple[list[Configuration], list[Configuration], list[Configuration]]:
    """One head's training, validation and test structures."""
    if head.train_file is None:
        raise DataStageError(
            f"head {name!r} names no `train_file`. A head with no structures "
            f"contributes nothing and would train a readout against nothing."
        )
    # The reference structures stay in: the isolated-atom E0 kind reads
    # them, and a backend that dropped them first would hand over whatever
    # its own extraction does with an unlabelled one, which is a zero.
    train = list(_open(head.train_file, name, key_spec).iter_range())
    if not train:
        raise DataStageError(f"head {name!r} read {head.train_file} and it is empty.")
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


def _loader(
    configurations: Sequence[Configuration],
    config: ResolvedConfig,
    z_table: AtomicNumberTable,
    specs,
    heads: tuple[str, ...],
    precision: PrecisionConfig,
    *,
    shuffle: bool,
) -> DataLoader:
    dataset = GraphDataset(
        configurations,
        cutoff=config.model.r_max,
        z_table=z_table,
        targets=specs,
        heads=heads,
    )
    return make_loader(
        dataset,
        batch_size=(
            config.training.batch_size if shuffle else config.training.valid_batch_size
        ),
        shuffle=shuffle,
        float_dtype=precision.model,
        num_workers=config.data.num_workers,
        pin_memory=config.data.pin_memory,
        seed=config.runtime.seed if shuffle else None,
    )
