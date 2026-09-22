"""Which structures a run trains on, and per head where its E0s come from.

A head is a level of theory, so everything that differs between two of them is
a field of :class:`HeadDataConfig` rather than a parallel list somewhere:
legacy keeps the heads in one dict and their E0s, files and weights in flags
that apply to all of them at once, which is why a multi-head run has to be
written as a YAML document the flags cannot express.

**The ten label keys are not fields here.** They are the data contract every
labelled dataset on disk was written against, so they belong to the key
convention rather than to a run. The three that *are* fields are the
graph-level inputs: an electronic temperature or a total charge is something a
particular dataset carries under a particular name, and declaring it is what
makes it readable at all. That split is the disposition table's, not this
module's.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from pydantic import Field

from mace_core.config.e0s import E0sIsolatedAtoms, E0Spec
from mace_core.config.section import FrozenSection

__all__ = ["DataConfig", "GraphInputKeys", "HeadDataConfig", "TransformSpec"]


class GraphInputKeys(FrozenSection):
    """File keys of the graph-level inputs, when a dataset spells them its way.

    Only the inputs. A label's key is the data contract's and is not settable
    per run; these three are declared features, and a feature nobody declared
    is not read.

    Args:
        elec_temp: Electronic temperature, per structure.
        total_spin: Total spin, per structure.
        total_charge: Total charge, per structure.
    """

    elec_temp: str = "elec_temp"
    total_spin: str = "total_spin"
    total_charge: str = "total_charge"


class TransformSpec(FrozenSection):
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


class HeadDataConfig(FrozenSection):
    """One level of theory: its structures, and where its E0s come from.

    Args:
        train_file: The structures this head trains on.
        valid_file: Its validation structures. ``None`` splits them out of
            ``train_file`` by ``valid_fraction``.
        test_file: Structures evaluated after training.
        e0s: Where the isolated-atom energies come from. Per head because two
            levels of theory do not share them, which is the whole reason a
            head exists.
        config_type_weights: Per ``config_type`` weight in the loss. A type
            absent from the mapping weighs ``1.0``.
        keep_isolated_atoms: Whether the ``IsolatedAtom`` structures stay in
            the training set after their energies have been read out of it.
    """

    train_file: Path | None = None
    valid_file: Path | None = None
    test_file: Path | None = None
    e0s: E0Spec = E0sIsolatedAtoms()
    config_type_weights: dict[str, float] = Field(default_factory=dict)
    keep_isolated_atoms: bool = False


class DataConfig(FrozenSection):
    """The datasets, the heads, and how they are loaded.

    Args:
        heads: Level of theory to its data. A single-head run writes one
            entry; there is no separate single-head shape, so nothing has to
            be rewritten to add a second head.
        valid_fraction: Fraction of a head's training set held out when it
            declares no ``valid_file``.
        test_dir: A directory of test sets, each file evaluated separately.
        statistics_file: A precomputed statistics document. Statistics
            normally travel with the prepared shards; this is the path for a
            legacy `statistics.json` and for a dataset prepared elsewhere.
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
        skip_evaluate_heads: Heads left out of the evaluation tables, for a
            replay head whose errors are not the run's subject.
    """

    heads: dict[str, HeadDataConfig] = Field(default_factory=dict)
    valid_fraction: float = 0.1
    test_dir: Path | None = None
    statistics_file: Path | None = None
    graph_input_keys: GraphInputKeys = GraphInputKeys()
    embedding_specs: dict[str, dict[str, str]] = Field(default_factory=dict)
    num_workers: int = 0
    pin_memory: bool = True
    skip_evaluate_heads: tuple[str, ...] = ()
    transforms: tuple[TransformSpec, ...] = ()
