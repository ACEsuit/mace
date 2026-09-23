"""What a run reads from the foundation model it starts from.

The data stage needs three things from a foundation model and no more: its
element table, which the fine-tune's model is built over; its isolated-atom
energies per head, which a head declaring ``foundation`` E0s copies; and, for
farthest-point sampling, a descriptor per structure. They are gathered here
into one object so the data stage takes one argument and never a model.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from mace_core.config.resolved import ResolvedConfig
from mace_core.data.configuration import Configuration
from mace_core.elements import AtomicNumberTable
from mace_core.metadata import ModelMetadata
from mace_core.observables import ObservableCatalogue
from torch import nn

__all__ = [
    "Describe",
    "Foundation",
    "FoundationContext",
    "FoundationError",
    "read_foundation",
]

#: One descriptor row per structure, in the order given.
Describe = Callable[[Sequence[Configuration]], np.ndarray]


class FoundationError(ValueError):
    """A request the foundation model cannot answer."""


@dataclass(frozen=True)
class FoundationContext:
    """A foundation model, as the data stage sees it.

    Attributes:
        z_table: Its element table. A fine-tune's model is built over it, so a
            structure with an element outside it is refused rather than given
            an embedding row nobody trained.
        heads: Its heads, in order.
        e0s: Its isolated-atom energies, ``head -> atomic number -> eV``, as
            its checkpoint records them.
        describe: Descriptors for farthest-point sampling, or ``None`` when the
            run does not sample that way.
    """

    z_table: AtomicNumberTable
    heads: tuple[str, ...]
    e0s: Mapping[str, Mapping[int, float]]
    describe: Describe | None = None

    def e0_table(self, head: str | None) -> Mapping[int, float]:
        """One head's energies.

        Raises:
            FoundationError: If ``head`` names none of its heads, or is
                ``None`` while it has several. The frozen tree takes the first
                in that case and only logs which it took.
        """
        if head is None:
            if len(self.heads) != 1:
                raise FoundationError(
                    f"the foundation model has heads {list(self.heads)}, and "
                    f"copying its energies needs one named. Set `head` on the "
                    f"E0 declaration."
                )
            head = self.heads[0]
        if head not in self.e0s:
            raise FoundationError(
                f"the foundation model has no head {head!r}. Its heads are "
                f"{list(self.heads)}."
            )
        return self.e0s[head]


@dataclass(frozen=True)
class Foundation:
    """A foundation model, rebuilt from its checkpoint.

    Attributes:
        engine: The model in its derivative engine, with every weight and
            every constant the checkpoint carried.
        config: The configuration it was trained with, as its record holds it.
        metadata: The record itself, which a fine-tune names as its parent.
        z_table: Its element table.
        heads: Its heads, in the order they were declared.
        e0s: Its isolated-atom energies per head, from the record.
        name: How the run named it, a path.
    """

    engine: nn.Module
    config: ResolvedConfig
    metadata: ModelMetadata
    z_table: AtomicNumberTable
    heads: tuple[str, ...]
    e0s: Mapping[str, Mapping[int, float]]
    name: str

    @property
    def model(self) -> nn.Module:
        """The model inside the engine, where the canonical state lives."""
        return self.engine.get_submodule("backbone")

    def context(self, describe: bool = False) -> FoundationContext:
        """What the data stage reads from it.

        Args:
            describe: Attach a descriptor function, which farthest-point
                sampling needs and nothing else does. It runs the model over
                every structure it is given, so it is not attached by default.
        """
        return FoundationContext(
            z_table=self.z_table,
            heads=self.heads,
            e0s=self.e0s,
            describe=self.describe if describe else None,
        )

    def describe(self, configurations: Sequence[Configuration]) -> np.ndarray:
        """One descriptor row per structure, for farthest-point sampling.

        Each structure's invariant node features, averaged per element, one
        block per element of the table. An element the structure lacks gets
        :data:`~mace_torch.finetune.subselect.ABSENT` in its block, as the
        frozen tree fills it, which puts two structures with different
        elements far apart.
        """
        from mace_torch.data import GraphDataset, collate_training
        from mace_torch.finetune.subselect import ABSENT
        from mace_torch.nn import MACEBackbone

        backbone = self.model.get_submodule("backbone")
        assert isinstance(backbone, MACEBackbone)
        dataset = GraphDataset(
            list(configurations),
            cutoff=self.config.model.r_max,
            z_table=self.z_table,
            targets=(),
            heads=self.heads,
        )
        rows = []
        with torch.no_grad():
            for index in range(len(dataset)):
                batch = collate_training([dataset[index]], z_table=self.z_table)
                block = backbone.descriptors(
                    batch.graph, aggregation="per_element_mean"
                ).clone()
                present = {int(z) for z in configurations[index].atomic_numbers}
                for position, number in enumerate(self.z_table.zs):
                    if int(number) not in present:
                        block[position] = ABSENT
                rows.append(block.reshape(-1).to(torch.float64).numpy())
        return np.stack(rows)


def read_foundation(
    path: str | Path, catalogue: ObservableCatalogue, precision=None
) -> Foundation:
    """Rebuild a foundation model from its checkpoint.

    The record says what the model was: its configuration, its heads and their
    energies, from which its element table follows. The tensors put back every
    weight and every constant, including the ones a rebuild could only have
    guessed, so the model is built with placeholders and then loaded.

    Raises:
        FoundationError: If the record carries no heads, which is a checkpoint
            written before heads were recorded, or heads that disagree about
            the elements.
    """
    from ase.data import atomic_numbers as numbers_of
    from mace_core.data.backend import DatasetStatistics
    from mace_core.elements import ResolvedE0s

    from mace_torch.serialization import load_checkpoint, read_sidecar
    from mace_torch.train.model_stage import DEFAULT_PRECISION, build_model

    document = read_sidecar(path)
    metadata = ModelMetadata.model_validate(document["config"])
    config = ResolvedConfig.model_validate(metadata.config.resolved)
    heads = tuple(config.data.heads)
    missing = [head for head in heads if head not in metadata.heads]
    if not heads or missing:
        raise FoundationError(
            f"{path} records no isolated-atom energies for heads "
            f"{missing or list(heads)}, so the model cannot be rebuilt: its "
            f"element table and its energies both come from that record."
        )
    e0s = {
        head: {
            int(numbers_of[symbol]): float(energy)
            for symbol, energy in metadata.heads[head].e0.values.items()
        }
        for head in heads
    }
    tables = {tuple(sorted(values)) for values in e0s.values()}
    if len(tables) != 1:
        raise FoundationError(
            f"{path} records energies over different elements per head: "
            f"{sorted(tables)}. One model has one element table."
        )
    z_table = AtomicNumberTable(list(tables.pop()))

    def build(_: object) -> nn.Module:
        engine, _ = build_model(
            config,
            catalogue,
            z_table=z_table,
            heads=heads,
            e0s=ResolvedE0s(e0s),
            statistics=DatasetStatistics(),
            precision=precision or DEFAULT_PRECISION,
            initialize=False,
        )
        return engine

    return Foundation(
        engine=load_checkpoint(path, build),
        config=config,
        metadata=metadata,
        z_table=z_table,
        heads=heads,
        e0s=e0s,
        name=str(path),
    )
