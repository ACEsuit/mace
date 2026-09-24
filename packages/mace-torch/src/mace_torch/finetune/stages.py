"""A fine-tune as separate steps, each one callable on its own.

``mace train`` runs a fine-tune in one go. A power user can run it step by step
instead, stopping after any step with something written to disk and starting
the next step from it, possibly in another process on another day:

1. **Select** a head's structures, for a replay head subselected from a large
   dataset: :func:`select_structures` writes the kept ones to a structure file,
   and :func:`reading_selected` points the head at it.
2. **Extend** the foundation model with elements it lacks:
   :func:`~mace_torch.finetune.extend.extend_elements` writes a new checkpoint,
   and the configuration's ``finetune.foundation_model`` names it.
3. **Build** the fine-tune's model over its data: :func:`build`.
4. **Train** it: :func:`train`, which writes the checkpoint.

Every boundary is a typed value or a file, never an argument namespace. Run
step by step from the written intermediates, a fine-tune produces the model the
one-go run does, with the same seeds.

Generating pseudolabels for a replay head is a step as well, and it lands with
its own implementation; until then a head's labels are the ones its file holds.
"""

from __future__ import annotations

from pathlib import Path

from mace_core.config.resolved import ResolvedConfig
from mace_core.data.xyz import write_configurations
from mace_core.observables import DEFAULT_CATALOGUE, ObservableCatalogue
from mace_core.stages import TrainedModel

from mace_torch.finetune.extend import NewSpeciesInit, extend_elements
from mace_torch.finetune.foundation import Foundation, read_foundation
from mace_torch.train.contracts import TorchBuiltModel
from mace_torch.train.data_stage import (
    DataStageError,
    key_specification,
    run_data_stage,
    selected_structures,
)
from mace_torch.train.loop import run_train_stage
from mace_torch.train.model_stage import run_model_stage

__all__ = [
    "NewSpeciesInit",
    "build",
    "extend_elements",
    "reading_selected",
    "select_structures",
    "train",
]


def select_structures(
    config: ResolvedConfig,
    head: str,
    output: str | Path,
    foundation: Foundation | None = None,
    catalogue: ObservableCatalogue = DEFAULT_CATALOGUE,
) -> Path:
    """Write the structures a head's subselection keeps.

    Args:
        config: The fine-tune's configuration.
        head: The head whose structures are selected.
        output: The structure file to write.
        foundation: The foundation model, for farthest-point sampling, which
            reads its descriptors. Read from the configuration when not given
            and the subselection needs it.
        catalogue: The observables the foundation model may declare.

    Returns:
        The written file.

    Raises:
        DataStageError: If the configuration has no such head, or the head
            cannot be read or selected from.
    """
    settings = _head(config, head)
    describe = settings.subselect is not None and settings.subselect.method == "fps"
    if describe and foundation is None:
        foundation = _foundation(config, catalogue)
    structures = selected_structures(
        head,
        settings,
        config,
        key_specification(config.data),
        foundation.context(describe=describe) if foundation is not None else None,
    )
    return write_configurations(output, structures, key_specification(config.data))


def reading_selected(
    config: ResolvedConfig, head: str, path: str | Path
) -> ResolvedConfig:
    """The configuration with ``head`` reading written structures as they are.

    The head keeps everything else it declares, its weight and its energies
    included; only its source changes, and its subselection, which the file
    already applied.
    """
    settings = _head(config, head).model_copy(
        update={"train_file": Path(path), "curated": None, "subselect": None}
    )
    heads = {**config.data.heads, head: settings}
    return config.model_copy(
        update={"data": config.data.model_copy(update={"heads": heads})}
    )


def build(
    config: ResolvedConfig,
    catalogue: ObservableCatalogue = DEFAULT_CATALOGUE,
    foundation: Foundation | None = None,
) -> TorchBuiltModel:
    """Read the data and build the fine-tune's model, its weights transferred.

    Args:
        config: The fine-tune's configuration.
        catalogue: The observable declarations.
        foundation: The foundation model, read from the configuration when not
            given.
    """
    if foundation is None and config.finetune.foundation_model is not None:
        foundation = _foundation(config, catalogue)
    describe = any(
        head.subselect is not None and head.subselect.method == "fps"
        for head in config.data.heads.values()
    )
    data = run_data_stage(
        config,
        catalogue,
        foundation=(
            foundation.context(describe=describe) if foundation is not None else None
        ),
    )
    return run_model_stage(config, data, catalogue, foundation=foundation)


def train(
    config: ResolvedConfig,
    built: TorchBuiltModel,
    checkpoint_path: str | Path | None = None,
    **options,
) -> TrainedModel:
    """Train a built fine-tune, writing its checkpoint where asked.

    ``options`` go to :func:`~mace_torch.train.run_train_stage` unchanged.
    """
    return run_train_stage(
        config,
        built,
        checkpoint_path=None if checkpoint_path is None else Path(checkpoint_path),
        **options,
    )


def _head(config: ResolvedConfig, head: str):
    if head not in config.data.heads:
        raise DataStageError(
            f"the configuration has no head {head!r}; its heads are "
            f"{sorted(config.data.heads)}."
        )
    return config.data.heads[head]


def _foundation(config: ResolvedConfig, catalogue: ObservableCatalogue) -> Foundation:
    if config.finetune.foundation_model is None:
        raise DataStageError(
            "the step needs the foundation model and the configuration names "
            "none. Set `finetune.foundation_model`."
        )
    return read_foundation(config.finetune.foundation_model, catalogue)
