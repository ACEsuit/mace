"""A trained v1 model, rebuilt from its checkpoint for evaluation.

One loader for every consumer that evaluates a model outside training: the ASE
calculator, the foundation model a fine-tune starts from, and whatever reads a
model next. A checkpoint is the canonical weights and a JSON record; the record
carries the resolved configuration and the isolated-atom energies per head, and
those two say everything a rebuild needs. The tensors then put back every
weight and every constant, so the model is built with placeholders and loaded.

The model is self-contained: its isolated-atom energies are inside it, so a
consumer never adds them back.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from mace_core.config.resolved import ResolvedConfig
from mace_core.elements import AtomicNumberTable
from mace_core.metadata import ModelMetadata
from mace_core.observables import (
    ObservableCatalogue,
    RequestedOutputs,
    load_default_catalogue,
)
from mace_core.outputs import MACEOutput
from torch import Tensor, nn

__all__ = ["DeployError", "DeployedModel", "load_deployed"]


class DeployError(ValueError):
    """A checkpoint that cannot be rebuilt into a model."""


@dataclass(frozen=True)
class DeployedModel:
    """A model ready to evaluate, with what its checkpoint says about it.

    Attributes:
        engine: The model inside its derivative engine, which is what computes
            forces and stress around it.
        config: The configuration it was trained with.
        metadata: The checkpoint's record.
        z_table: Its element table.
        heads: Its heads, in the order the model indexes them.
        e0s: Its isolated-atom energies per head, ``head -> Z -> eV``, as the
            record holds them. For reading, never for adding back.
        outputs: What it was built to produce, observables and derivatives.
        path: Where it was read from.
    """

    engine: nn.Module
    config: ResolvedConfig
    metadata: ModelMetadata
    z_table: AtomicNumberTable
    heads: tuple[str, ...]
    e0s: Mapping[str, Mapping[int, float]]
    outputs: RequestedOutputs
    path: Path

    @property
    def r_max(self) -> float:
        """The cutoff, in Angstrom."""
        return float(self.config.model.r_max)

    @property
    def model(self) -> nn.Module:
        """The model inside the engine, where the canonical state lives."""
        return self.engine.get_submodule("backbone")

    def compute(
        self, graph: Mapping[str, Any], compute: Iterable[str] = ("forces",)
    ) -> MACEOutput[Tensor]:
        """Evaluate one batch, with whichever derivatives are asked for."""
        return self.engine(dict(graph), compute=tuple(compute), training=False)


def load_deployed(
    path: str | Path,
    *,
    device: str = "cpu",
    catalogue: ObservableCatalogue | None = None,
    precision: Any = None,
) -> DeployedModel:
    """Rebuild a model from a v1 checkpoint.

    Args:
        path: Either file of the checkpoint pair, weights or record.
        device: Where the model is put.
        catalogue: The observables it may declare. The default catalogue
            unless a model declares others.
        precision: What it computes in. The default precision unless given.

    Raises:
        DeployError: If the record carries no isolated-atom energies for its
            heads, which is a checkpoint written before heads were recorded,
            or heads that disagree about the elements.
        CheckpointError: If the files are not a v1 checkpoint, or the weights
            and the record disagree.
    """
    from ase.data import atomic_numbers as numbers_of
    from mace_core.data.backend import DatasetStatistics
    from mace_core.elements import ResolvedE0s

    from mace_torch.serialization import load_checkpoint, read_sidecar
    from mace_torch.train.model_stage import DEFAULT_PRECISION, build_model

    catalogue = catalogue or load_default_catalogue()
    document = read_sidecar(path)
    metadata = ModelMetadata.model_validate(document["config"])
    config = ResolvedConfig.model_validate(metadata.config.resolved)
    heads = tuple(config.data.heads)
    missing = [head for head in heads if head not in metadata.heads]
    if not heads or missing:
        raise DeployError(
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
        raise DeployError(
            f"{path} records energies over different elements per head: "
            f"{sorted(tables)}. One model has one element table."
        )
    z_table = AtomicNumberTable(list(tables.pop()))

    built: list[RequestedOutputs] = []

    def build(_: object) -> nn.Module:
        engine, outputs = build_model(
            config,
            catalogue,
            z_table=z_table,
            heads=heads,
            e0s=ResolvedE0s(e0s),
            statistics=DatasetStatistics(),
            precision=precision or DEFAULT_PRECISION,
            initialize=False,
        )
        built.append(outputs)
        return engine

    engine = load_checkpoint(path, build).to(device)
    engine.eval()
    return DeployedModel(
        engine=engine,
        config=config,
        metadata=metadata,
        z_table=z_table,
        heads=heads,
        e0s=e0s,
        outputs=built[0],
        path=Path(path),
    )
