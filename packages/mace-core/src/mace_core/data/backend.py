"""Framework-independent random-access datasets of ``Configuration`` objects.

Backends provide an exact length and raise on unreadable items. Graph building
and collation happen downstream. Samplers own sharding; ``iter_range`` lets
backends serve contiguous reads efficiently. Iterable-only sources are unsupported.
"""

from __future__ import annotations

import json
from collections.abc import Iterator
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Protocol, runtime_checkable

from mace_core.data.configuration import Configuration
from mace_core.data.keys import KeySpecification

__all__ = [
    "DATASET_SCHEMA_VERSION",
    "DataBackend",
    "DataBackendError",
    "DatasetManifest",
    "DatasetStatistics",
]

DATASET_SCHEMA_VERSION = "1.0"


class DataBackendError(RuntimeError):
    """A dataset could not be read.

    Carries the source and, where it applies, the index.
    """

    def __init__(self, source: str, index: int | None = None, reason: str = "") -> None:
        where = f"{source!r}" if index is None else f"{source!r} at index {index}"
        super().__init__(
            f"could not read {where}: {reason}" if reason else f"could not read {where}"
        )
        self.source = source
        self.index = index
        self.reason = reason


@dataclass
class DatasetStatistics:
    """The numbers a model needs from its data before it can be built.

    The field set is a compatibility surface rather than a fresh design: it is
    exactly what the frozen tree reads back out of its ``statistics.json``
    (``mace/cli/run_train.py:319-334``).

    Attributes:
        atomic_energies: Isolated-atom reference energies, ``{Z: eV}``. Passed
            in already resolved, so the interaction-energy mean and standard
            deviation are always taken against the same values that become the
            model buffer, whatever their origin.
        avg_num_neighbors: Mean edge count per atom at ``r_max``.
        mean: Mean interaction energy per atom, in eV.
        std: Its standard deviation, or the force RMS when that is what the
            scaling asks for.
        atomic_numbers: Every element present, ascending.
        r_max: The cutoff the neighbour count was taken at, in Angstrom.
    """

    atomic_energies: dict[int, float] = field(default_factory=dict)
    avg_num_neighbors: float = 0.0
    mean: float = 0.0
    std: float = 1.0
    atomic_numbers: list[int] = field(default_factory=list)
    r_max: float = 0.0

    def to_json(self) -> str:
        """Real JSON, with real types.

        The frozen tree writes these as stringified Python dicts and reads them
        back with ``ast.literal_eval`` (``mace/cli/preprocess_data.py:250-260``
        against ``run_train.py:313-334``), which makes the file unreadable by
        anything that is not Python and silently accepts whatever eval returns.
        """
        payload = asdict(self)
        payload["atomic_energies"] = {
            str(number): value for number, value in self.atomic_energies.items()
        }
        return json.dumps(payload, indent=2, sort_keys=True)

    @classmethod
    def from_json(cls, text: str) -> DatasetStatistics:
        payload = json.loads(text)
        payload["atomic_energies"] = {
            int(number): float(value)
            for number, value in payload.get("atomic_energies", {}).items()
        }
        payload["atomic_numbers"] = [int(z) for z in payload.get("atomic_numbers", [])]
        return cls(**payload)


@dataclass
class DatasetManifest:
    """The typed description a persisted dataset carries beside its data.

    Attributes:
        schema_version: The initial V1 schema; other versions are refused.
        shards: The shard files, relative to the manifest, in reading order.
            Listed rather than globbed, so a missing shard is an error and the
            order does not depend on how a directory happens to sort.
        shard_counts: How many configurations each shard holds, one entry per
            file in ``shards``. Stored per shard rather than as one group size,
            so an exact length never has to assume the shards are uniform.
        key_specification: The specification used at write time. A dataset read
            back under a different one is a different dataset.
        atomic_numbers: The element table.
        r_max: The cutoff, in Angstrom.
        e0_provenance: Where the isolated-atom energies came from, in words.
        statistics: Embedded statistics, when the format carries them.
    """

    schema_version: str = DATASET_SCHEMA_VERSION
    shards: list[str] = field(default_factory=list)
    shard_counts: list[int] = field(default_factory=list)
    key_specification: dict[str, Any] = field(default_factory=dict)
    atomic_numbers: list[int] = field(default_factory=list)
    r_max: float = 0.0
    e0_provenance: str = ""
    statistics: DatasetStatistics | None = None

    def __post_init__(self) -> None:
        if self.schema_version != DATASET_SCHEMA_VERSION:
            raise ValueError(
                f"unsupported dataset schema_version {self.schema_version!r}; "
                f"expected {DATASET_SCHEMA_VERSION!r}"
            )

    @property
    def length(self) -> int:
        """The exact number of configurations, summed over the shards."""
        return sum(self.shard_counts)

    def to_json(self) -> str:
        payload = asdict(self)
        if self.statistics is not None:
            payload["statistics"] = json.loads(self.statistics.to_json())
        return json.dumps(payload, indent=2, sort_keys=True)

    @classmethod
    def from_json(cls, text: str) -> DatasetManifest:
        payload = json.loads(text)
        statistics = payload.pop("statistics", None)
        manifest = cls(**payload)
        if statistics is not None:
            manifest.statistics = DatasetStatistics.from_json(json.dumps(statistics))
        return manifest


@runtime_checkable
class DataBackend(Protocol):
    """Map-style, lazy and picklable.

    Picklable matters for a concrete reason: dataloader workers and distributed
    training send the dataset to another process. A backend holding an open file
    handle drops it in ``__getstate__`` and reopens on demand.
    """

    name: str

    @classmethod
    def sniff(cls, source: str) -> bool:
        """Whether this backend claims ``source``. Cheap: an extension or magic
        bytes, never a full parse.
        """
        ...

    @classmethod
    def open(
        cls,
        source: str | Path,
        *,
        key_spec: KeySpecification,
        head: str = "Default",
        keep_isolated_atoms: bool = False,
    ) -> DataBackend:
        """Open a dataset. Raises :class:`DataBackendError` if it cannot.

        ``keep_isolated_atoms`` leaves the single-atom reference structures in
        the dataset. They are dropped by default because they are references
        rather than training structures, and a caller that resolves the
        isolated-atom energies itself needs them present: reading them off a
        table the backend built instead would take whatever that backend does
        with an unlabelled one.
        """
        ...

    def __len__(self) -> int:
        """Exact, and cheap once open."""
        ...

    def __getitem__(self, index: int) -> Configuration:
        """One configuration. Raises :class:`DataBackendError` on a bad read."""
        ...

    def iter_range(
        self, start: int = 0, stop: int | None = None, step: int = 1
    ) -> Iterator[Configuration]:
        """A sequential run. Shard formats override this; the default loops."""
        ...

    def statistics(self) -> DatasetStatistics | None:
        """Precomputed statistics if the format carries them, else ``None``."""
        ...

    def metadata(self) -> DatasetManifest | None:
        """The manifest if the format carries one, else ``None``."""
        ...
