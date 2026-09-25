"""The v2 shard format: HDF5 files of configurations and a JSON manifest.

This module is the format, written down once. It says where each field of a
:class:`~mace_core.data.configuration.Configuration` lives inside a shard, how
a missing value and a piece of text are told apart from an array, and what the
manifest beside the shards holds. The backend in
:mod:`mace_core.data.backends.hdf5` reads it and :func:`write_shards` writes
it, and neither makes any of these decisions itself.

A dataset is a directory::

    dataset.json            the manifest, a DatasetManifest as JSON
    <prefix>_0.h5           one shard per file, in the order the manifest lists
    <prefix>_1.h5
    ...

and one shard is::

    /                       attrs: format, schema_version
    configs/{i}             one group per configuration, i counting from 0
                            within the shard
        atomic_numbers      int64 [n_atoms]
        positions           float64 [n_atoms, 3], Angstrom
        cell                float64 [3, 3], Angstrom; absent if there is none
        pbc                 bool [3]; absent if there is none
        properties/{name}   one dataset per labelled property, keyed by
                            convention name, never by file key
        property_weights    group whose attrs hold one float per property
        attrs: weight, config_type

Three things about the encoding are deliberate.

**Absence is recorded, not spelled.** A declared property with no value is
listed in the ``absent`` attribute of ``properties`` and has no dataset. The
frozen tree writes the string ``"None"`` in its place and turns any stored
value whose text is ``None`` back into ``None`` on the way in
(``mace/data/utils.py:613`` against ``mace/data/hdf5_dataset.py:95``), so a
``config_type`` of ``"None"`` reads back as a missing one.

**Every property dataset says what it holds.** Its ``kind`` attribute is
``array``, ``scalar`` or ``text``, so an energy comes back a float, a stress
comes back an array, and a text label comes back a string, whatever HDF5 would
otherwise have handed back.

**The head is not stored.** Which head a dataset trains is decided when it is
opened, the same as for an XYZ file, so one set of shards can serve any head.

The length of a dataset is the sum of the manifest's per-shard counts. The
frozen tree multiplies the number of groups by the size of the first one
(``mace/data/hdf5_dataset.py:19-21``), which counts phantom structures whenever
the last group is short.
"""

from __future__ import annotations

import json
from collections.abc import Sequence
from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict
from pathlib import Path
from typing import Any

import h5py
import numpy as np

from mace_core.data.backend import DataBackendError, DatasetManifest, DatasetStatistics
from mace_core.data.configuration import DEFAULT_HEAD, Configuration
from mace_core.data.keys import KeySpecification

__all__ = [
    "MANIFEST_NAME",
    "SCHEMA_VERSION",
    "SHARD_FORMAT",
    "read_configuration",
    "read_manifest",
    "write_configuration",
    "write_shards",
]

#: The manifest's file name, inside the dataset directory.
MANIFEST_NAME = "dataset.json"

#: The version of both the shard layout and the manifest. Anything that
#: changes either shape bumps it, and a reader refuses a version it does not
#: know rather than guessing at a missing key.
SCHEMA_VERSION = 1

#: What a shard's root ``format`` attribute says, so a shard is recognisable
#: without its manifest.
SHARD_FORMAT = "mace-v1-shard"

_KINDS = ("array", "scalar", "text")


def write_configuration(group: h5py.Group, configuration: Configuration) -> None:
    """Write one configuration into an empty group, in the layout above."""
    group["atomic_numbers"] = np.asarray(configuration.atomic_numbers, dtype=np.int64)
    group["positions"] = np.asarray(configuration.positions, dtype=np.float64)
    if configuration.cell is not None:
        group["cell"] = np.asarray(configuration.cell, dtype=np.float64)
    if configuration.pbc is not None:
        group["pbc"] = np.asarray(configuration.pbc, dtype=bool)
    group.attrs["weight"] = float(configuration.weight)
    group.attrs["config_type"] = str(configuration.config_type)

    properties = group.create_group("properties")
    absent: list[str] = []
    for name, value in configuration.properties.items():
        if value is None:
            absent.append(name)
            continue
        if isinstance(value, str):
            dataset = properties.create_dataset(name, data=value)
            dataset.attrs["kind"] = "text"
        elif np.ndim(value) == 0 and not isinstance(value, np.ndarray):
            dataset = properties.create_dataset(name, data=value)
            dataset.attrs["kind"] = "scalar"
        else:
            dataset = properties.create_dataset(name, data=np.asarray(value))
            dataset.attrs["kind"] = "array"
    properties.attrs["absent"] = absent

    weights = group.create_group("property_weights")
    for name, weight in configuration.property_weights.items():
        weights.attrs[name] = float(weight)


def _flags(values: Any) -> tuple[bool, bool, bool]:
    x, y, z = (bool(flag) for flag in values)
    return x, y, z


def read_configuration(group: h5py.Group, *, head: str = DEFAULT_HEAD) -> Configuration:
    """Read one configuration back out of its group.

    Raises:
        KeyError: A required dataset or attribute is missing.
        ValueError: A property says it holds a kind this version does not know.
    """
    properties: dict[str, Any] = {}
    for name, dataset in group["properties"].items():
        kind = dataset.attrs["kind"]
        if kind == "text":
            properties[name] = dataset.asstr()[()]
        elif kind == "scalar":
            properties[name] = dataset[()].item()
        elif kind == "array":
            properties[name] = dataset[()]
        else:
            raise ValueError(
                f"property {name!r} is stored as {kind!r}; this reader knows "
                f"{list(_KINDS)}"
            )
    for name in group["properties"].attrs["absent"]:
        properties[str(name)] = None

    return Configuration(
        atomic_numbers=group["atomic_numbers"][()],
        positions=group["positions"][()],
        properties=properties,
        property_weights={
            name: float(value)
            for name, value in group["property_weights"].attrs.items()
        },
        cell=group["cell"][()] if "cell" in group else None,
        pbc=_flags(group["pbc"][()]) if "pbc" in group else None,
        weight=float(group.attrs["weight"]),
        config_type=str(group.attrs["config_type"]),
        head=head,
    )


def _write_shard(path: Path, configurations: Sequence[Configuration]) -> int:
    with h5py.File(path, "w") as shard:
        shard.attrs["format"] = SHARD_FORMAT
        shard.attrs["schema_version"] = SCHEMA_VERSION
        configs = shard.create_group("configs")
        for index, configuration in enumerate(configurations):
            write_configuration(configs.create_group(str(index)), configuration)
    return len(configurations)


def write_shards(
    configurations: Sequence[Configuration],
    out_dir: str | Path,
    *,
    shard_size: int,
    key_spec: KeySpecification,
    prefix: str = "shard",
    statistics: DatasetStatistics | None = None,
    atomic_numbers: Sequence[int] | None = None,
    r_max: float | None = None,
    e0_provenance: str = "",
    processes: int = 1,
) -> DatasetManifest:
    """Write configurations as v2 shards and the manifest that describes them.

    Args:
        configurations: The structures, in the order they will be read back.
        out_dir: The dataset directory. Created if missing; a directory that
            already holds a manifest is refused rather than overwritten.
        shard_size: At most this many configurations per shard. The last shard
            holds the remainder, and the manifest records every shard's count.
        key_spec: The key specification the configurations were parsed with,
            recorded in the manifest.
        prefix: Shard files are named ``<prefix>_<index>.h5``.
        statistics: Embedded in the manifest, so a reader gets them back
            without recomputing. Their element table and cutoff are the
            manifest's unless given explicitly.
        atomic_numbers: The element table. Defaults to the statistics' table,
            else to the elements present.
        r_max: The cutoff in Angstrom. Defaults to the statistics' cutoff.
        e0_provenance: Where the isolated-atom energies came from, in words.
        processes: How many processes write shards at once. Each shard is
            written by exactly one of them.

    Returns:
        The manifest, as written.

    Raises:
        ValueError: ``shard_size`` or ``processes`` is not positive, or there
            is nothing to write.
        FileExistsError: ``out_dir`` already holds a manifest.
    """
    if shard_size < 1:
        raise ValueError(f"shard_size must be at least 1, got {shard_size}")
    if processes < 1:
        raise ValueError(f"processes must be at least 1, got {processes}")
    if not configurations:
        raise ValueError("there are no configurations to write")
    directory = Path(out_dir)
    if (directory / MANIFEST_NAME).exists():
        raise FileExistsError(
            f"{directory / MANIFEST_NAME} already exists. Write to an empty "
            f"directory; a dataset is not appended to in place."
        )
    directory.mkdir(parents=True, exist_ok=True)

    runs = [
        list(configurations[start : start + shard_size])
        for start in range(0, len(configurations), shard_size)
    ]
    names = [f"{prefix}_{index}.h5" for index in range(len(runs))]
    paths = [directory / name for name in names]
    if processes == 1:
        counts = [
            _write_shard(path, run) for path, run in zip(paths, runs, strict=True)
        ]
    else:
        with ProcessPoolExecutor(max_workers=processes) as pool:
            counts = list(pool.map(_write_shard, paths, runs))

    if atomic_numbers is not None:
        table = sorted(int(z) for z in atomic_numbers)
    elif statistics is not None and statistics.atomic_numbers:
        table = list(statistics.atomic_numbers)
    else:
        table = sorted({int(z) for item in configurations for z in item.atomic_numbers})
    if r_max is None:
        r_max = statistics.r_max if statistics is not None else 0.0

    manifest = DatasetManifest(
        schema_version=SCHEMA_VERSION,
        shards=names,
        shard_counts=counts,
        key_specification=asdict(key_spec),
        atomic_numbers=table,
        r_max=float(r_max),
        e0_provenance=e0_provenance,
        statistics=statistics,
    )
    (directory / MANIFEST_NAME).write_text(manifest.to_json() + "\n")
    return manifest


_MANIFEST_FIELDS = {
    "schema_version",
    "shards",
    "shard_counts",
    "key_specification",
    "atomic_numbers",
    "r_max",
    "e0_provenance",
    "statistics",
}


def read_manifest(directory: str | Path) -> DatasetManifest:
    """Read and check a dataset's manifest.

    Raises:
        DataBackendError: No manifest, a schema version this reader does not
            know, a field missing or unknown, or shard lists that disagree.
            Each names the file and what is wrong with it.
    """
    path = Path(directory) / MANIFEST_NAME
    source = str(path)
    try:
        payload = json.loads(path.read_text())
    except FileNotFoundError:
        raise DataBackendError(source, reason="no manifest") from None
    except (OSError, ValueError) as failure:
        raise DataBackendError(source, reason=f"unreadable: {failure}") from failure
    if not isinstance(payload, dict):
        raise DataBackendError(source, reason="the manifest is not a JSON object")

    version = payload.get("schema_version")
    if version != SCHEMA_VERSION:
        raise DataBackendError(
            source,
            reason=(
                f"schema_version {version!r}; this reader knows version "
                f"{SCHEMA_VERSION}. Re-prepare the dataset with this version, "
                f"or read it with the version that wrote it."
            ),
        )
    missing = sorted(_MANIFEST_FIELDS - set(payload))
    unknown = sorted(set(payload) - _MANIFEST_FIELDS)
    if missing or unknown:
        raise DataBackendError(
            source,
            reason=f"fields missing {missing} and unknown {unknown} for version 1",
        )
    manifest = DatasetManifest.from_json(json.dumps(payload))
    if len(manifest.shards) != len(manifest.shard_counts):
        raise DataBackendError(
            source,
            reason=(
                f"{len(manifest.shards)} shards are listed with "
                f"{len(manifest.shard_counts)} counts"
            ),
        )
    return manifest
