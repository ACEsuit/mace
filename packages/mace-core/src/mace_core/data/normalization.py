"""Training-set normalization in spherical coordinates, independently per head."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass

import numpy as np

from mace_core.config.model import ScaleShiftSpec
from mace_core.data.configuration import Configuration
from mace_core.data.statistics import root_mean_square
from mace_core.elements import ResolvedE0s
from mace_core.observables import ObservableSpec
from mace_core.observables.grammar import parse_irreps


@dataclass(frozen=True)
class NormalizationValues:
    """One invariant scale and a spherical shift vector for one dataset head."""

    scale: float
    shift: tuple[float, ...]


def compute_normalization(
    configurations: Sequence[Configuration],
    observables: Sequence[ObservableSpec],
    recipes: Mapping[str, Mapping[str, ScaleShiftSpec]],
    e0s: ResolvedE0s,
) -> dict[str, dict[str, NormalizationValues]]:
    """Measure only labelled training examples, after splitting and transforms.

    Extensive graph labels are measured per atom; energy first subtracts the
    owning head's E0s. Only even scalar components may have a nonzero shift.
    One scale over all components commutes with rotations and inversion.
    A constant target uses scale one so its readout remains trainable.
    """
    specs = {spec.name: spec for spec in observables}
    result: dict[str, dict[str, NormalizationValues]] = {}
    for name, by_head in recipes.items():
        spec = specs[name]
        result[name] = {}
        scalar = np.array(
            [
                term.degree == 0 and term.parity == "e"
                for term in parse_irreps(spec.irreps, observable=name)
                for _ in range(term.multiplicity * (2 * term.degree + 1))
            ]
        )
        for head, recipe in by_head.items():
            shift = np.zeros(spec.dimension)
            if recipe.kind == "none":
                result[name][head] = NormalizationValues(1.0, tuple(shift))
                continue
            samples, targets = [], []
            target = {"rms_forces": "forces", "rms_dipoles": "dipole"}.get(
                recipe.kind, name
            )
            for configuration in configurations:
                if configuration.head != head:
                    continue
                value = configuration.properties.get(name)
                if value is not None:
                    sample = np.asarray(value, dtype=float).reshape(-1, spec.dimension)
                    if name == "energy":
                        reference = sum(
                            e0s.values[head][int(z)]
                            for z in configuration.atomic_numbers
                        )
                        sample = sample - reference
                    if not spec.per_atom and (spec.extensive or name == "energy"):
                        sample = sample / len(configuration.atomic_numbers)
                    samples.append(sample)
                value = configuration.properties.get(target)
                if value is not None:
                    targets.append(np.asarray(value, dtype=float).reshape(-1))
            if not samples or (recipe.kind != "std" and not targets):
                missing = name if not samples else target
                raise ValueError(
                    f"model.normalization.{name}.{head}: no training labels for "
                    f"{missing!r}; supply that property or choose another recipe"
                )
            values = np.concatenate(samples)
            if not np.all(np.isfinite(values)):
                raise ValueError(f"model.normalization.{name}.{head}: nonfinite labels")
            shift[scalar] = values.mean(axis=0)[scalar]
            spread = root_mean_square(
                values - shift if recipe.kind == "std" else np.concatenate(targets)
            )
            if not np.isfinite(spread) or not np.all(np.isfinite(shift)):
                raise ValueError(f"model.normalization.{name}.{head}: nonfinite labels")
            result[name][head] = NormalizationValues(spread or 1.0, tuple(shift))
    return result
