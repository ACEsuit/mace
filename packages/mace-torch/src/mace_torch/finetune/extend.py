"""Giving a model elements it was not trained on.

An artifact goes in and an artifact comes out; nothing is trained. The model is
rebuilt over the larger element table and every canonical tensor is carried
across: those that do not depend on the elements unchanged, and the per-element
ones row by row, each existing element's row copied bit for bit and each new
element's row made by the initialization spec. A fine-tune then starts from the
result like from any other foundation model.

**Which tensors are per element is read off the two models, not listed.** Both
tables are built and their canonical shapes compared; a tensor whose shape
differs is per element, and it must be one of the kinds below or the extension
refuses, naming it. A per-element tensor nobody wrote a rule for would
otherwise keep whatever the rebuild drew, for the old elements too.

* the node embedding, a linear map out of the element one-hot, one column of
  weights per element;
* the skip connection, one linear map per element, ``[Z, ...]``;
* the symmetric contraction's weights, ``[Z, A, mul]``;
* the energy head's isolated-atom energies, ``[heads, Z]``, which take the
  energies given for the new elements.

**The old elements compute exactly what they did.** Per-element parameters are
gathered by element, so a structure holding only old elements reads only old
rows, and every other tensor is the parent's. Its energies and forces are the
parent's to the last bit.

**The initialization spec.** Two, each recorded in the artifact with the
elements it added:

* ``fresh``: each new row is what a freshly initialized model over the new
  table draws there, with the spec's seed. The same spec, seed and table give
  the same rows.
* ``copy``: each new element takes the rows of a donor element the parent
  already has, typically a chemical neighbour, so it starts from something
  trained rather than from noise.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

from ase.data import chemical_symbols
from mace_core.data.backend import DatasetStatistics
from mace_core.elements import AtomicNumberTable, ResolvedE0s
from mace_core.metadata import (
    E0Details,
    ElementExtensionRecord,
    HeadSummary,
    ModelMetadata,
    ParentModel,
    Provenance,
)
from mace_core.observables import DEFAULT_CATALOGUE, ObservableCatalogue
from torch import Tensor

from mace_torch import __version__
from mace_torch.serialization import canonical_state, load_canonical_state

__all__ = ["ElementExtensionError", "NewSpeciesInit", "extend_elements"]


class ElementExtensionError(ValueError):
    """An extension that cannot be made as asked."""


@dataclass(frozen=True)
class NewSpeciesInit:
    """How the rows of an added element are made.

    Attributes:
        kind: ``"fresh"`` draws them as a new model would, ``"copy"`` takes a
            donor element's.
        seed: The seed ``fresh`` draws with.
        donors: For ``copy``, the donor of each added element, by atomic
            number. Every added element needs one.
    """

    kind: Literal["fresh", "copy"] = "fresh"
    seed: int = 0
    donors: Mapping[int, int] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.kind not in ("fresh", "copy"):
            raise ElementExtensionError(
                f"the initialization is {self.kind!r}; it is 'fresh', drawn as "
                f"a new model would draw it, or 'copy', from a donor element."
            )


def extend_elements(
    path: str | Path,
    output: str | Path,
    e0s: Mapping[str, Mapping[int, float]],
    init: NewSpeciesInit | None = None,
    catalogue: ObservableCatalogue = DEFAULT_CATALOGUE,
) -> Path:
    """Write a copy of a checkpoint that also covers the given elements.

    Args:
        path: The checkpoint to extend.
        output: Where the extended one is written, as for ``write_model``.
        e0s: The isolated-atom energy of each added element, per head, in eV.
            Which elements are added is read from here, and every head has to
            give the same ones: an energy is a property of a head's level of
            theory, so it is given rather than invented.
        init: How the added elements' rows are made; ``fresh`` with seed 0
            when not given.
        catalogue: The observables the checkpoint may declare.

    Returns:
        The path of the written checkpoint's record.

    Raises:
        ElementExtensionError: If an element is already in the table, the heads
            give different elements or miss one, a donor is missing or unknown,
            or a per-element tensor has no rule here.
    """
    from mace_torch.deploy.loader import load_deployed
    from mace_torch.train.checkpoint import write_model
    from mace_torch.train.model_stage import build_model

    init = init or NewSpeciesInit()
    parent = load_deployed(path, catalogue=catalogue)
    added = _added_elements(e0s, parent.heads, parent.z_table)
    if init.kind == "copy":
        _check_donors(init, added, parent.z_table)
    table = AtomicNumberTable(sorted({*parent.z_table.zs, *added}))
    energies = {
        head: {**parent.e0s[head], **{z: float(e0s[head][z]) for z in added}}
        for head in parent.heads
    }
    config = parent.config.model_copy(
        update={"runtime": parent.config.runtime.model_copy(update={"seed": init.seed})}
    )
    engine, _ = build_model(
        config,
        catalogue,
        z_table=table,
        heads=parent.heads,
        e0s=ResolvedE0s(energies),
        statistics=DatasetStatistics(avg_num_neighbors=1.0),
        initialize=True,
    )
    model = engine.get_submodule("backbone")
    state = _extended_state(
        canonical_state(parent.model),
        canonical_state(model),
        list(parent.z_table.zs),
        list(table.zs),
        init,
    )
    load_canonical_state(model, state)
    return write_model(output, engine, _metadata(parent, energies, added, init, path))


def _added_elements(
    e0s: Mapping[str, Mapping[int, float]],
    heads: tuple[str, ...],
    table: AtomicNumberTable,
) -> list[int]:
    missing_heads = sorted(set(heads) - set(e0s))
    extra_heads = sorted(set(e0s) - set(heads))
    if missing_heads or extra_heads:
        raise ElementExtensionError(
            f"the energies are given for heads {sorted(e0s)} and the model has "
            f"{list(heads)}; every head needs the added elements' energies."
        )
    sets = {tuple(sorted(int(z) for z in e0s[head])) for head in heads}
    if len(sets) != 1:
        raise ElementExtensionError(
            f"the heads give energies for different elements, {sorted(sets)}. "
            f"An element added to the model is added for every head."
        )
    added = list(sets.pop())
    if not added:
        raise ElementExtensionError("no element is given, so nothing is added.")
    present = sorted(set(added) & {int(z) for z in table.zs})
    if present:
        raise ElementExtensionError(
            f"elements {present} are already in the model's table "
            f"{list(table.zs)}; their rows are trained and are not replaced."
        )
    return added


def _check_donors(
    init: NewSpeciesInit, added: list[int], table: AtomicNumberTable
) -> None:
    without = sorted(set(added) - {int(z) for z in init.donors})
    if without:
        raise ElementExtensionError(
            f"the 'copy' initialization needs a donor for every added element, "
            f"and {without} have none."
        )
    unknown = sorted(
        {int(z) for z in init.donors.values()} - {int(z) for z in table.zs}
    )
    if unknown:
        raise ElementExtensionError(
            f"donors {unknown} are not in the model's table {list(table.zs)}, "
            f"so they have no trained rows to copy."
        )


def _extended_state(
    source: Mapping[str, Mapping[str, Tensor]],
    target: Mapping[str, Mapping[str, Tensor]],
    old: list[int],
    new: list[int],
    init: NewSpeciesInit,
) -> dict[str, dict[str, Tensor]]:
    """The parent's canonical state over the new table.

    ``target`` is a freshly built model's over the new table, which is where
    ``fresh`` rows come from and what every shape is checked against.
    """
    if set(source) != set(target):
        raise ElementExtensionError(
            f"the rebuilt model holds different operators from its parent: "
            f"{sorted(set(source) ^ set(target))}."
        )
    position = {z: index for index, z in enumerate(old)}
    # Each row of the new table: an old element's own row in the parent, and
    # where its weights are read from, which for an added element is its
    # donor's row under `copy` and the rebuild's draw under `fresh`.
    own: list[int | None] = [position.get(z) for z in new]
    sources: list[int | None] = [
        position[int(init.donors[z])]
        if z not in position and init.kind == "copy"
        else position.get(z)
        for z in new
    ]

    state: dict[str, dict[str, Tensor]] = {}
    for path, tensors in target.items():
        found = source[path]
        state[path] = {}
        for name, value in tensors.items():
            parent_value = found[name]
            if parent_value.shape == value.shape:
                state[path][name] = parent_value.clone()
                continue
            state[path][name] = _per_element(
                path, name, parent_value, value, own, sources, len(old), len(new)
            )
    return state


def _per_element(
    path: str,
    name: str,
    parent: Tensor,
    fresh: Tensor,
    own: list[int | None],
    sources: list[int | None],
    old_count: int,
    new_count: int,
) -> Tensor:
    """One per-element tensor over the new table, by the rule for its kind."""
    if path.endswith("energy_head") and name == "e0_table":
        # The rebuild holds the given energies for the added elements, which a
        # copied element keeps: its energy is its own, never its donor's. The
        # old elements' are copied from the parent, to keep their bits.
        result = fresh.clone()
        for column, origin in enumerate(own):
            if origin is not None:
                result[:, column] = parent[:, origin]
        return result
    if path.endswith("node_embedding") and name == "weight":
        # Output copies outermost, so each channel's weights over the elements
        # are one contiguous run, as the transfer into a smaller table reads.
        grid = parent.reshape(-1, old_count)
        result = fresh.reshape(-1, new_count).clone()
        for column, origin in enumerate(sources):
            if origin is not None:
                result[:, column] = grid[:, origin]
        return result.reshape(-1)
    if (path.endswith(".skip") or path.endswith(".contraction")) and name == "weight":
        result = fresh.clone()
        for row, origin in enumerate(sources):
            if origin is not None:
                result[row] = parent[origin]
        return result
    raise ElementExtensionError(
        f"{path}:{name} changes shape with the element table, "
        f"{tuple(parent.shape)} to {tuple(fresh.shape)}, and no rule here says "
        f"how its rows are carried. It is refused rather than left at the "
        f"rebuild's draw for the old elements too."
    )


def _metadata(
    parent,
    energies: Mapping[str, Mapping[int, float]],
    added: list[int],
    init: NewSpeciesInit,
    path: str | Path,
) -> ModelMetadata:
    """The extended model's record, with the parent's record untouched inside."""
    heads = {}
    for head in parent.heads:
        before: E0Details = parent.metadata.heads[head].e0
        heads[head] = HeadSummary(
            e0=E0Details(
                source=before.source,
                method=before.method,
                parameters={
                    **before.parameters,
                    "added_explicitly": sorted(chemical_symbols[z] for z in added),
                },
                values={
                    chemical_symbols[z]: float(energy)
                    for z, energy in sorted(energies[head].items())
                },
            ),
            sources=list(parent.metadata.heads[head].sources),
        )
    return parent.metadata.model_copy(
        update={
            "provenance": Provenance(code_version=__version__),
            "heads": heads,
            "parents": [
                ParentModel(
                    role="initial_weights", name=str(path), metadata=parent.metadata
                )
            ],
            "element_extension": ElementExtensionRecord(
                added=[chemical_symbols[z] for z in added],
                initialization=init.kind,
                seed=init.seed if init.kind == "fresh" else None,
                donors={
                    chemical_symbols[int(z)]: chemical_symbols[int(donor)]
                    for z, donor in init.donors.items()
                }
                if init.kind == "copy"
                else {},
            ),
        }
    )
