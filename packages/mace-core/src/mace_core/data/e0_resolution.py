"""Turning an E0 declaration into one energy per element.

Called once, before the model is built, because an isolated-atom energy is a
per-species constant that every energy the head trains on is shifted by. Get it
wrong and the model absorbs the difference into its readout, the fit looks
merely poor, and nothing anywhere says a number was missing.

**Every element of the table is resolved or this raises.** That is the whole
contract, and it replaces four separate places where the frozen tree keeps
going instead:

* an ``IsolatedAtom`` structure whose energy key is absent warns and
  contributes ``0.0`` (``mace/data/utils.py:333-337``);
* a singular least-squares system logs an error and zeroes **every** element
  (``:381-387``);
* an element the foundation model does not cover is padded with
  ``head_energies.get(z, 0.0)`` (``mace/cli/run_train.py:578-586``);
* and the assertion that a foundation model is configured is made for one of
  the two kinds that need one and not the other (``:527``).

A zero is not a neutral default here. It is a claim that an isolated atom of
that species has exactly zero energy, which is indistinguishable in the fit
from a real reference and wrong by however much the real one is.

No torch, no jax. ``estimated`` is the one kind that needs a model, and it
takes it as an injected callable rather than importing one.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any, Protocol

import numpy as np

from mace_core.config.e0s import (
    E0sAverage,
    E0sEstimated,
    E0sFromFoundation,
    E0sIsolatedAtoms,
    E0Spec,
    E0sTable,
)
from mace_core.data.configuration import Configuration
from mace_core.data.xyz import ISOLATED_ATOM_CONFIG_TYPE
from mace_core.elements.number_table import AtomicNumberTable

__all__ = ["E0Provenance", "E0ResolutionError", "EnergyPredictor", "resolve_e0s"]


class E0ResolutionError(ValueError):
    """An E0 request that cannot be answered for every element of the table."""


class EnergyPredictor(Protocol):
    """What ``estimated`` needs from a foundation model, and nothing more.

    A callable rather than a model, so this module stays free of a framework
    and so the caller decides what is being predicted with.
    """

    def __call__(self, configurations: Sequence[Configuration], /) -> Sequence[float]:
        """The total energy the model predicts for each configuration, in eV.

        Positional-only: a protocol that named its parameter would oblige every
        caller to name it the same, and the ordinary implementation of this is
        a small function written at the call site.
        """


@dataclass(frozen=True)
class E0Provenance:
    """How the energies were obtained, recorded beside them.

    Two runs of one method against different data or a different foundation
    model produce different numbers, so the method alone does not say where a
    table came from.

    Attributes:
        kind: The requested kind.
        solver: What produced the numbers, when a method was run.
        dataset_fingerprint: A stable digest of the structures the fit used,
            so a table can be tied back to the data it came from.
        num_configurations: How many structures fed a fit.
        rank: The rank of the least-squares system, and the reason a singular
            one raises rather than defaulting: a rank below the element count
            means the composition does not determine the energies.
        foundation_model: Which artifact a copied or corrected table came from.
        missing_filled: Elements the source did not cover, filled by the
            declared policy. Empty under the default, which is to refuse.
        from_foundation: Elements no structure holds, whose energies were
            taken from the foundation model a fine-tune keeps the element
            table of.
    """

    kind: str
    solver: str | None = None
    dataset_fingerprint: str | None = None
    num_configurations: int | None = None
    rank: int | None = None
    foundation_model: str | None = None
    missing_filled: tuple[int, ...] = field(default_factory=tuple)
    from_foundation: tuple[int, ...] = field(default_factory=tuple)


def resolve_e0s(
    spec: E0Spec,
    z_table: AtomicNumberTable,
    train_configurations: Sequence[Configuration] | None = None,
    *,
    foundation_e0s: Mapping[int, float] | None = None,
    foundation_model: str | None = None,
    predict_energy: EnergyPredictor | None = None,
) -> tuple[dict[int, float], E0Provenance]:
    """One isolated-atom energy per element of ``z_table``, and where it came from.

    Args:
        spec: What the configuration asked for.
        z_table: The elements that have to be covered. Every one of them, or
            this raises.
        train_configurations: The head's training structures, needed by the
            three kinds that read or fit against data.
        foundation_e0s: The foundation model's own table, for the two kinds
            that read one.
        foundation_model: What to record as its identity.
        predict_energy: Required by ``estimated`` and ignored otherwise.

    Returns:
        ``(atomic number -> energy in eV, provenance)``.

    Raises:
        E0ResolutionError: If the request cannot be answered for every element,
            naming the elements and why. This is the only outcome other than a
            complete table; there is no partial answer and no default.
    """
    resolvers: dict[type, Callable[..., tuple[dict[int, float], E0Provenance]]] = {
        E0sTable: _from_table,
        E0sIsolatedAtoms: _from_isolated_atoms,
        E0sAverage: _from_least_squares,
        E0sFromFoundation: _from_foundation,
        E0sEstimated: _from_corrected_least_squares,
    }
    resolve = resolvers[type(spec)]
    values, provenance = resolve(
        spec,
        z_table,
        train_configurations,
        foundation_e0s=foundation_e0s,
        foundation_model=foundation_model,
        predict_energy=predict_energy,
    )
    _check_complete(values, z_table, provenance.kind)
    return values, provenance


def _check_complete(
    values: Mapping[int, float], z_table: AtomicNumberTable, kind: str
) -> None:
    missing = sorted(z for z in z_table.zs if z not in values)
    if missing:
        raise E0ResolutionError(
            f"the {kind!r} isolated-atom energies cover "
            f"{sorted(values)} and the model is built for {sorted(z_table.zs)}, "
            f"so {missing} would train against energies short by a constant "
            f"per atom of those species. Give them explicitly, or declare a "
            f"policy for elements the source does not cover."
        )


def _needs(value: Any, what: str, kind: str) -> Any:
    if value is None:
        raise E0ResolutionError(
            f"the {kind!r} isolated-atom energies need {what}, and none was given."
        )
    return value


def _from_table(spec, z_table, configurations, **_: Any):
    return dict(spec.values), E0Provenance(kind=spec.kind)


def _from_isolated_atoms(spec, z_table, configurations, **_: Any):
    configurations = _needs(configurations, "the head's training structures", spec.kind)
    values: dict[int, float] = {}
    unlabelled: list[int] = []
    for configuration in configurations:
        if configuration.config_type != ISOLATED_ATOM_CONFIG_TYPE:
            continue
        if len(configuration.atomic_numbers) != 1:
            raise E0ResolutionError(
                f"a structure marked {ISOLATED_ATOM_CONFIG_TYPE!r} holds "
                f"{len(configuration.atomic_numbers)} atoms. An isolated atom "
                f"is one atom, and reading an energy off anything else would "
                f"attribute a whole structure's energy to one species."
            )
        number = int(configuration.atomic_numbers[0])
        energy = configuration.properties.get("energy")
        if energy is None:
            unlabelled.append(number)
            continue
        values[number] = float(energy)
    if unlabelled and spec.on_missing_energy == "error":
        raise E0ResolutionError(
            f"the isolated atoms for {sorted(set(unlabelled))} carry no "
            f"energy. Legacy warns and uses 0.0, which is a claim that those "
            f"atoms have exactly zero energy; set on_missing_energy: zero to "
            f"ask for it deliberately."
        )
    for number in unlabelled:
        values.setdefault(number, 0.0)
    return values, E0Provenance(
        kind=spec.kind,
        dataset_fingerprint=_fingerprint(configurations),
        num_configurations=len(configurations),
        missing_filled=tuple(sorted(set(unlabelled))),
    )


def _least_squares(
    configurations: Sequence[Configuration],
    z_table: AtomicNumberTable,
    energies: Sequence[float],
    kind: str,
) -> tuple[dict[int, float], int]:
    """Fit one energy per element to the given total energies.

    Raises:
        E0ResolutionError: If the system is singular. Legacy logs and zeroes
            every element; a rank below the element count means the
            composition simply does not determine the energies, and answering
            with zeros answers a different question.
    """
    counts = np.zeros((len(configurations), len(z_table.zs)), dtype=np.float64)
    for row, configuration in enumerate(configurations):
        for number in configuration.atomic_numbers:
            counts[row, z_table.z_to_index(int(number))] += 1.0
    targets = np.asarray(energies, dtype=np.float64)
    solution, _residuals, rank, _singular = np.linalg.lstsq(counts, targets, rcond=None)
    if rank < len(z_table.zs):
        raise E0ResolutionError(
            f"the {kind!r} fit is singular: {len(configurations)} structures "
            f"over {len(z_table.zs)} elements give rank {rank}, so the "
            f"composition does not determine the energies. Legacy zeroes every "
            f"element here. Add structures that vary the composition, or give "
            f"the energies explicitly."
        )
    return {z: float(solution[z_table.z_to_index(z)]) for z in z_table.zs}, int(rank)


def _labelled_energies(
    configurations: Sequence[Configuration], kind: str
) -> tuple[list[Configuration], list[float]]:
    kept, energies = [], []
    for configuration in configurations:
        energy = configuration.properties.get("energy")
        if energy is None:
            continue
        kept.append(configuration)
        energies.append(float(energy))
    if not kept:
        raise E0ResolutionError(
            f"the {kind!r} isolated-atom energies are fitted to the training "
            f"energies and none of the {len(configurations)} structures "
            f"carries one."
        )
    return kept, energies


def _from_least_squares(spec, z_table, configurations, **_: Any):
    configurations = _needs(configurations, "the head's training structures", spec.kind)
    kept, energies = _labelled_energies(configurations, spec.kind)
    values, rank = _least_squares(kept, z_table, energies, spec.kind)
    return values, E0Provenance(
        kind=spec.kind,
        solver="least_squares",
        dataset_fingerprint=_fingerprint(kept),
        num_configurations=len(kept),
        rank=rank,
    )


def _from_foundation(spec, z_table, configurations, **kwargs: Any):
    table = _needs(
        kwargs.get("foundation_e0s"), "the foundation model's own table", spec.kind
    )
    values = {z: float(table[z]) for z in z_table.zs if z in table}
    missing = [z for z in z_table.zs if z not in table]
    values, filled = _fill_missing(
        values, missing, spec.missing, spec.kind, configurations
    )
    return values, E0Provenance(
        kind=spec.kind,
        foundation_model=kwargs.get("foundation_model"),
        missing_filled=filled,
    )


def _from_corrected_least_squares(spec, z_table, configurations, **kwargs: Any):
    configurations = _needs(configurations, "the head's training structures", spec.kind)
    predict = _needs(
        kwargs.get("predict_energy"), "a foundation model to correct against", spec.kind
    )
    kept, energies = _labelled_energies(configurations, spec.kind)
    predicted = list(predict(kept))
    if len(predicted) != len(kept):
        raise E0ResolutionError(
            f"the foundation model returned {len(predicted)} energies for "
            f"{len(kept)} structures. The correction is per structure, so the "
            f"two have to line up."
        )
    residuals = [
        label - float(guess) for label, guess in zip(energies, predicted, strict=True)
    ]
    values, rank = _least_squares(kept, z_table, residuals, spec.kind)
    missing = [z for z in z_table.zs if z not in values]
    values, filled = _fill_missing(values, missing, spec.missing, spec.kind, kept)
    return values, E0Provenance(
        kind=spec.kind,
        solver="foundation_corrected_least_squares",
        dataset_fingerprint=_fingerprint(kept),
        num_configurations=len(kept),
        rank=rank,
        foundation_model=kwargs.get("foundation_model"),
        missing_filled=filled,
    )


def _fill_missing(
    values: dict[int, float],
    missing: Sequence[int],
    policy: str,
    kind: str,
    configurations: Sequence[Configuration] | None,
) -> tuple[dict[int, float], tuple[int, ...]]:
    """Apply the declared policy for elements the source did not cover.

    ``average`` fits the uncovered elements to the head's training energies
    with the covered ones held at the source's values: each structure's energy
    less its covered atoms' energies is a sum over its uncovered atoms, and
    that is the only system whose answer is an energy per uncovered element.
    ``zero`` is a padding, and a padding is only harmless where nothing trains
    against it, so an element it fills that a structure holds is refused.
    """
    if not missing:
        return values, ()
    if policy == "error":
        raise E0ResolutionError(
            f"the {kind!r} source covers {sorted(values)} and the model is "
            f"built for elements including {sorted(missing)}. Legacy pads "
            f"those with 0.0. Declare missing: average to fit them to the "
            f"training energies with the covered ones held fixed, or missing: "
            f"zero if no structure holds them."
        )
    held = {int(z) for item in configurations or () for z in item.atomic_numbers}
    if policy == "zero":
        trained = sorted(held & set(missing))
        if trained:
            raise E0ResolutionError(
                f"missing: zero pads {trained} with 0.0, and the training data "
                f"holds them, so the head would train against a reference "
                f"energy of zero. Use missing: average, or give their energies."
            )
        return {**values, **dict.fromkeys(missing, 0.0)}, tuple(sorted(missing))
    return {**values, **_fit_uncovered(values, missing, configurations, kind)}, tuple(
        sorted(missing)
    )


def _fit_uncovered(
    values: Mapping[int, float],
    missing: Sequence[int],
    configurations: Sequence[Configuration] | None,
    kind: str,
) -> dict[int, float]:
    """The uncovered elements' energies, the covered ones held fixed."""
    kept, energies = _labelled_energies(
        _needs(configurations, "the head's training structures", kind), kind
    )
    order = {int(z): column for column, z in enumerate(sorted(missing))}
    counts = np.zeros((len(kept), len(order)), dtype=np.float64)
    residuals = np.asarray(energies, dtype=np.float64)
    for row, configuration in enumerate(kept):
        for number in configuration.atomic_numbers:
            number = int(number)
            if number in order:
                counts[row, order[number]] += 1.0
            else:
                residuals[row] -= values[number]
    solution, _residuals, rank, _singular = np.linalg.lstsq(
        counts, residuals, rcond=None
    )
    if rank < len(order):
        raise E0ResolutionError(
            f"missing: average fits {sorted(missing)} to the training energies "
            f"with the covered elements held fixed, and the structures that "
            f"hold them give rank {rank} for {len(order)} unknowns. Add "
            f"structures that hold them, or give their energies."
        )
    return {z: float(solution[column]) for z, column in order.items()}


def _fingerprint(configurations: Sequence[Configuration]) -> str:
    """A stable digest of the structures a fit used.

    Composition and energy per structure, which is what the fit actually reads.
    Positions are deliberately left out: two datasets differing only in
    geometry give the same E0s, and a fingerprint that moved with them would
    report a change that did not happen.
    """
    import hashlib

    digest = hashlib.sha256()
    for configuration in configurations:
        counts = np.bincount(np.asarray(configuration.atomic_numbers, dtype=np.int64))
        digest.update(counts.tobytes())
        energy = configuration.properties.get("energy")
        digest.update(repr(None if energy is None else float(energy)).encode())
    return digest.hexdigest()[:16]
