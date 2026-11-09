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
from mace_core.config.precision import PrecisionConfig
from mace_core.config.resolved import ResolvedConfig
from mace_core.data.configuration import Configuration
from mace_core.elements import AtomicNumberTable
from mace_core.metadata import ModelMetadata
from mace_core.observables import ObservableCatalogue
from torch import nn

__all__ = [
    "Foundation",
    "FoundationContext",
    "FoundationError",
    "StructureDescriptors",
    "read_foundation",
]

#: One descriptor row per structure, in the order given.
StructureDescriptors = Callable[[Sequence[Configuration]], np.ndarray]


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
        structure_descriptors: Descriptors for farthest-point sampling, or
            ``None`` when the
            run does not sample that way.
    """

    z_table: AtomicNumberTable
    heads: tuple[str, ...]
    e0s: Mapping[str, Mapping[int, float]]
    structure_descriptors: StructureDescriptors | None = None

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
        return self.engine.get_submodule("model")

    def context(self, include_structure_descriptors: bool = False) -> FoundationContext:
        """What the data stage reads from it.

        Args:
            include_structure_descriptors: Attach the descriptor function that FPS
                sampling needs and nothing else does. It runs the model over
                every structure it is given, so it is not attached by default.
        """
        return FoundationContext(
            z_table=self.z_table,
            heads=self.heads,
            e0s=self.e0s,
            structure_descriptors=self.structure_descriptors
            if include_structure_descriptors
            else None,
        )

    def structure_descriptors(
        self, configurations: Sequence[Configuration]
    ) -> np.ndarray:
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
        from mace_torch.nn.backbone import aggregate_atomic_descriptors

        backbone = self.model.get_submodule("backbone")
        assert isinstance(backbone, MACEBackbone)
        dataset = GraphDataset(
            list(configurations),
            cutoff=self.config.model.r_max,
            z_table=self.z_table,
            targets=(),
            heads=self.heads,
            features=self.config.model.feature_storages,
        )
        parameter = next(backbone.parameters())
        dtype = str(parameter.dtype).removeprefix("torch.")
        rows = []
        with torch.no_grad():
            for index in range(len(dataset)):
                batch = collate_training(
                    [dataset[index]],
                    float_dtype=dtype,
                    features=dataset.features,
                ).to(parameter.device)
                atomic_descriptors = backbone.atomic_descriptors(backbone(batch.graph))
                means = aggregate_atomic_descriptors(
                    atomic_descriptors,
                    batch.graph["batch"],
                    batch.graph["ptr"].shape[0] - 1,
                    "per_element_mean",
                    element=batch.graph["node_attrs"].argmax(-1),
                    atomic_numbers=self.z_table.zs,
                )
                assert isinstance(means, list)
                block = atomic_descriptors.new_full(
                    (len(self.z_table.zs), atomic_descriptors.shape[1]), ABSENT
                )
                for position, number in enumerate(self.z_table.zs):
                    if int(number) in means[0]:
                        block[position] = means[0][int(number)]
                rows.append(
                    block.reshape(-1).to(dtype=torch.float64, device="cpu").numpy()
                )
        return np.stack(rows)


def read_foundation(
    path: str | Path,
    catalogue: ObservableCatalogue,
    precision: PrecisionConfig | None = None,
    *,
    device: str = "cpu",
) -> Foundation:
    """Rebuild a foundation model from its checkpoint.

    The record says what the model was: its configuration, its heads and their
    energies, from which its element table follows. The tensors put back every
    weight and every constant, including the ones a rebuild could only have
    guessed, so the model is built with placeholders and then loaded.

    Raises:
        FoundationError: If the record lacks head energies or its heads
            disagree about the elements.
    """
    from mace_torch.deploy.loader import DeployError, load_deployed

    try:
        deployed = load_deployed(
            path, catalogue=catalogue, precision=precision, device=device
        )
    except DeployError as error:
        raise FoundationError(str(error)) from error
    return Foundation(
        engine=deployed.derivative_engine,
        config=deployed.config,
        metadata=deployed.metadata,
        z_table=deployed.z_table,
        heads=deployed.heads,
        e0s=deployed.e0s,
        name=str(path),
    )
