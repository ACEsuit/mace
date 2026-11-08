"""Add declared observables while retaining every trained canonical tensor."""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

from mace_core.config.resolved import ResolvedConfig
from mace_core.data.backend import DatasetStatistics
from mace_core.elements import ResolvedE0s
from mace_core.metadata import ConfigRecord, ParentModel, Provenance
from mace_core.observables import DEFAULT_CATALOGUE, ObservableCatalogue, ObservableSpec
from mace_core.weights.checkpoint import validate_layout

from mace_torch import __version__
from mace_torch.deploy.loader import DeployedModel
from mace_torch.serialization import (
    canonical_metadata,
    canonical_state,
    load_canonical_state,
)


def extend_readouts(
    parent: DeployedModel,
    observables: Sequence[ObservableSpec],
    output: str | Path,
    *,
    seed: int = 0,
) -> Path:
    """Write a checkpoint with freshly initialized heads for new observables.

    The existing backbone must already support the new readouts. A declaration
    that changes an existing operator's shape is refused before writing output.
    New heads use the normal model initializer and the explicit seed; old
    tensors, E0s and normalization are copied without modification. Predictions
    of new heads require training, and descendants inherit no support badge.
    """
    from mace_torch.models.build import build_model
    from mace_torch.serialization import write_model

    if not observables:
        raise ValueError("give at least one new observable")
    catalogue = parent.config.catalogue or DEFAULT_CATALOGUE
    specs = {spec.name: spec for spec in catalogue.observables}
    names = [spec.name for spec in observables]
    if len(names) != len(set(names)) or set(names) & set(
        parent.config.model.observables
    ):
        raise ValueError(
            "new readout names must be unique and must not replace an "
            "existing observable"
        )
    for spec in observables:
        if spec.name in specs and specs[spec.name] != spec:
            raise ValueError(
                f"{spec.name}: the readout disagrees with the existing "
                f"catalogue declaration"
            )
        specs[spec.name] = spec
    expanded = ObservableCatalogue(
        inputs=catalogue.inputs, observables=tuple(specs.values())
    )
    document = parent.config.to_resolved_dict()
    document["model"]["observables"].extend(names)
    document["runtime"]["seed"] = seed
    document["catalogue"] = expanded.model_dump(mode="json")
    config = ResolvedConfig.model_validate(document)
    engine, _ = build_model(
        config,
        expanded,
        z_table=parent.z_table,
        heads=parent.heads,
        e0s=ResolvedE0s(parent.e0s),
        statistics=DatasetStatistics(),
        initialize=True,
    )
    original_layout = canonical_metadata(parent.derivative_engine)
    expanded_layout = canonical_metadata(engine)
    validate_layout(
        original_layout,
        {
            name: expanded_layout[name]
            for name in original_layout
            if name in expanded_layout
        },
    )
    before = canonical_state(parent.derivative_engine)
    after = canonical_state(engine)
    for operator, tensors in before.items():
        if operator not in after or set(tensors) != set(after[operator]):
            raise ValueError(
                f"new readouts would replace trained operator "
                f"{operator}; use a compatible observable"
            )
        for name, value in tensors.items():
            if value.shape != after[operator][name].shape:
                raise ValueError(
                    f"new readouts change {operator}.{name} from "
                    f"{tuple(value.shape)} to {tuple(after[operator][name].shape)}; "
                    "the backbone cannot be preserved"
                )
        after[operator] = tensors
    load_canonical_state(engine, after)
    metadata = parent.metadata.model_copy(
        update={
            "config": ConfigRecord(
                user=parent.metadata.config.user, resolved=config.to_resolved_dict()
            ),
            "provenance": Provenance(versions={"mace-torch-v1": __version__}),
            "parents": [
                ParentModel(
                    role="initial_weights",
                    name=str(parent.path),
                    metadata=parent.metadata,
                )
            ],
            "training": None,
            "notes": parent.metadata.notes
            + (
                f"\nAdded untrained readouts {names} with seed {seed}; "
                "expected to work, not a supported registry artifact."
            ),
        }
    )
    return write_model(output, engine, metadata)
