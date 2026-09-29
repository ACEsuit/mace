"""Changing what a run trains against, without changing what a loss is.

A transform rewrites the structures between reading them and measuring them.
That is where a change of target belongs: an energy the model should reproduce
is a property of the data, and moving it into the loss would mean a loss that
knows which of its inputs were rewritten.

**A transform is a registered function, not a branch.** The decorator is the
registration, so a scheme this package never heard of is written in another
package and named in a configuration. The three shipped here are presets, not
a closed set, and each is small enough to read as an example of writing one.

They run over parsed structures, before the statistics. That is deliberate: a
statistic taken over energies that are about to be shifted describes a dataset
that never trains, and a scale computed from it is wrong by the shift.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import replace
from typing import Any

import numpy as np
from mace_core.data.configuration import Configuration

__all__ = [
    "TRANSFORM_REGISTRY",
    "Transform",
    "TransformError",
    "UnknownTransformError",
    "apply_transforms",
    "register_transform",
]

#: A transform takes the structures and returns them, changed.
Transform = Callable[[Sequence[Configuration]], list[Configuration]]

#: Every registered transform, by the name a configuration selects it with.
TRANSFORM_REGISTRY: dict[str, Callable[..., Transform]] = {}


class TransformError(ValueError):
    """A transform cannot be applied to the data it was given."""


class UnknownTransformError(KeyError):
    """A configured transform nobody registered."""


def register_transform(name: str) -> Callable[[Callable[..., Transform]], Any]:
    """Register a transform factory under ``name``.

    The factory takes the settings a configuration gives it and returns the
    transform. Splitting the two is what lets the settings be validated when
    the run is configured rather than when the data is read.

    Raises:
        ValueError: If the name is taken. Two transforms under one name means
            the data is rewritten by whichever module was imported last.
    """

    def decorate(factory: Callable[..., Transform]) -> Callable[..., Transform]:
        if name in TRANSFORM_REGISTRY:
            raise ValueError(
                f"{name!r} is already a registered transform, from "
                f"{TRANSFORM_REGISTRY[name]!r}. Two under one name means the "
                f"data is rewritten by whichever module was imported last."
            )
        TRANSFORM_REGISTRY[name] = factory
        return factory

    return decorate


def apply_transforms(
    configurations: Sequence[Configuration],
    specifications: Sequence[tuple[str, Mapping[str, Any]]],
) -> list[Configuration]:
    """Run the named transforms in order.

    Args:
        configurations: The parsed structures.
        specifications: ``(name, settings)`` pairs, in the order they apply.
            Order is part of the meaning: shifting energies and then masking on
            a threshold is not the same run as masking and then shifting.

    Raises:
        UnknownTransformError: Naming the value and listing what is registered.
    """
    result = list(configurations)
    for name, settings in specifications:
        if name not in TRANSFORM_REGISTRY:
            raise UnknownTransformError(
                f"{name!r} is not a registered transform. The registered names "
                f"are {sorted(TRANSFORM_REGISTRY)}. Register yours with "
                f"@register_transform."
            )
        result = list(TRANSFORM_REGISTRY[name](**dict(settings))(result))
    return result


# ---------------------------------------------------------------------------
# The presets
# ---------------------------------------------------------------------------


@register_transform("relative_energy")
def relative_energy(
    group_by: str = "config_type", energy_key: str = "energy"
) -> Transform:
    """Train on energies measured from the lowest structure of each group.

    A fitting database often holds several sets whose absolute energies are on
    different scales, because they came from different calculations. Fitting
    the absolute numbers then spends the model's capacity on the offsets
    between the sets. Subtracting each group's own minimum leaves the
    differences, which is what the model is for.

    The shift is a constant per group, so it changes no force and no stress.

    Args:
        group_by: Which attribute names the group. ``config_type`` by default;
            any attribute of a structure or key of its properties works.
        energy_key: Which property is shifted.

    Raises:
        TransformError: If a structure carries no energy. The shift is defined
            by a group's minimum and a structure with no energy has no place
            in that minimum, so it is refused rather than left at its absolute
            value beside shifted neighbours.
    """

    def transform(configurations: Sequence[Configuration]) -> list[Configuration]:
        groups: dict[Any, float] = {}
        for configuration in configurations:
            energy = _energy_of(configuration, energy_key, "relative_energy")
            key = _group_key(configuration, group_by)
            groups[key] = min(groups.get(key, energy), energy)
        shifted = []
        for configuration in configurations:
            key = _group_key(configuration, group_by)
            energy = _energy_of(configuration, energy_key, "relative_energy")
            properties = dict(configuration.properties)
            properties[energy_key] = energy - groups[key]
            shifted.append(replace(configuration, properties=properties))
        return shifted

    return transform


@register_transform("subtract_property")
def subtract_property(
    subtract: str, from_property: str = "energy", drop: bool = True
) -> Transform:
    """Train on what is left after taking one property out of another.

    The inter/intra decomposition in its general form: a file that carries the
    energy of the fragments alongside the total energy leaves the interaction
    energy when one is taken from the other. The fragments are identified by
    whoever wrote the file, which is the only place that can do it.

    Args:
        subtract: The property to take away, per structure.
        from_property: What to take it from.
        drop: Whether to remove the subtracted property afterwards. Leaving it
            in means a model that declares it would train on a quantity it has
            just been told to ignore.

    Raises:
        TransformError: If a structure carries neither property. There is no
            sensible default: zero would claim the fragments have no energy.
    """

    def transform(configurations: Sequence[Configuration]) -> list[Configuration]:
        result = []
        for configuration in configurations:
            total = _energy_of(configuration, from_property, "subtract_property")
            part = _energy_of(configuration, subtract, "subtract_property")
            properties = dict(configuration.properties)
            properties[from_property] = total - part
            if drop:
                properties.pop(subtract, None)
            result.append(replace(configuration, properties=properties))
        return result

    return transform


@register_transform("mask_above")
def mask_above(property_name: str, threshold: float) -> Transform:
    """Stop a structure counting for one property when its values are extreme.

    An outlier in a fitting database is usually a failed calculation rather
    than physics, and one structure with forces of a thousand eV per Angstrom
    dominates the term it appears in. Masking sets the structure's weight for
    that property to zero, which is the same mechanism a missing value uses, so
    nothing downstream needs to know the difference.

    The structure is kept, and its other properties keep counting. Dropping the
    structure would throw away a good energy because a force was wrong.

    Args:
        property_name: Which property to judge and mask.
        threshold: The largest absolute value that still counts.
    """

    def transform(configurations: Sequence[Configuration]) -> list[Configuration]:
        result = []
        for configuration in configurations:
            value = configuration.properties.get(property_name)
            if value is None or np.abs(np.asarray(value, dtype=float)).max() <= (
                threshold
            ):
                result.append(configuration)
                continue
            weights = dict(configuration.property_weights)
            weights[property_name] = 0.0
            result.append(replace(configuration, property_weights=weights))
        return result

    return transform


def _group_key(configuration: Configuration, group_by: str) -> Any:
    """The group a structure belongs to, from an attribute or a property."""
    if hasattr(configuration, group_by):
        return getattr(configuration, group_by)
    return configuration.properties.get(group_by)


def _energy_of(configuration: Configuration, key: str, transform: str) -> float:
    value = configuration.properties.get(key)
    if value is None:
        raise TransformError(
            f"the {transform!r} transform needs {key!r} on every structure and "
            f"this one carries {sorted(configuration.properties)}."
        )
    return float(value)
