"""Drawing a new version of a training structure every time it is read.

An augmentation is not a transform. A transform rewrites the data once, before
the statistics, and every epoch trains on the same rewritten structures. An
augmentation draws afresh on every read, so across epochs a structure is seen
under many draws of a symmetry the model does not have exactly, and learns it.
It applies to the training structures only, never to what is evaluated.

**A symmetry the model already has is not worth augmenting.** Turning a
structure and its moments together changes nothing an equivariant model
computes. What the magnetic models do not have built in is the spin-space
symmetry: without spin-orbit coupling the energy does not change when the
moments alone are turned, and at zero field it does not change when they are
all reversed.

Like the transforms, an augmentation is a registered function, selected from a
configuration by name, and a package this one never heard of can register its
own.

Each read draws from torch's generator, as the frozen tree's does, so a loader
worker draws from the stream torch seeds it with and two workers do not repeat
each other.
"""

from __future__ import annotations

import math
from collections.abc import Callable, Mapping, Sequence
from dataclasses import replace
from typing import Any, Literal

import numpy as np
import torch
from mace_core.data.configuration import Configuration

__all__ = [
    "AUGMENTATION_REGISTRY",
    "Augmentation",
    "UnknownAugmentationError",
    "build_augmentations",
    "random_rotation",
    "register_augmentation",
]

#: An augmentation takes one structure and returns a new draw of it.
Augmentation = Callable[[Configuration], Configuration]

#: Every registered augmentation, by the name a configuration selects it with.
AUGMENTATION_REGISTRY: dict[str, Callable[..., Augmentation]] = {}


class UnknownAugmentationError(KeyError):
    """A configured augmentation nobody registered."""


def register_augmentation(name: str) -> Callable[[Callable[..., Augmentation]], Any]:
    """Register an augmentation factory under ``name``.

    Raises:
        ValueError: If the name is taken.
    """

    def decorate(
        factory: Callable[..., Augmentation],
    ) -> Callable[..., Augmentation]:
        if name in AUGMENTATION_REGISTRY:
            raise ValueError(
                f"{name!r} is already a registered augmentation, from "
                f"{AUGMENTATION_REGISTRY[name]!r}."
            )
        AUGMENTATION_REGISTRY[name] = factory
        return factory

    return decorate


def build_augmentations(
    specifications: Sequence[tuple[str, Mapping[str, Any]]],
) -> tuple[Augmentation, ...]:
    """The named augmentations, built from their settings, in order.

    Built when the run is set up, so a bad setting fails before any data is
    read.

    Raises:
        UnknownAugmentationError: Naming the value and listing what is
            registered.
    """
    built = []
    for name, settings in specifications:
        if name not in AUGMENTATION_REGISTRY:
            raise UnknownAugmentationError(
                f"{name!r} is not a registered augmentation. The registered "
                f"names are {sorted(AUGMENTATION_REGISTRY)}; register another "
                f"with @register_augmentation."
            )
        built.append(AUGMENTATION_REGISTRY[name](**dict(settings)))
    return tuple(built)


def random_rotation(dtype: torch.dtype = torch.float64) -> np.ndarray:
    """A rotation drawn uniformly, from a uniform unit quaternion.

    Shoemake's construction: three uniform numbers give a quaternion uniform on
    the sphere, and so a rotation uniform in the Haar measure. Written out
    because it is short and is the frozen tree's.
    """
    u1, u2, u3 = torch.rand(3, dtype=dtype).tolist()
    q1 = math.sqrt(1 - u1) * math.sin(2 * math.pi * u2)
    q2 = math.sqrt(1 - u1) * math.cos(2 * math.pi * u2)
    q3 = math.sqrt(u1) * math.sin(2 * math.pi * u3)
    q4 = math.sqrt(u1) * math.cos(2 * math.pi * u3)
    return np.array(
        [
            [1 - 2 * (q3**2 + q4**2), 2 * (q2 * q3 - q1 * q4), 2 * (q2 * q4 + q1 * q3)],
            [2 * (q2 * q3 + q1 * q4), 1 - 2 * (q2**2 + q4**2), 2 * (q3 * q4 - q1 * q2)],
            [2 * (q2 * q4 - q1 * q3), 2 * (q3 * q4 + q1 * q2), 1 - 2 * (q2**2 + q3**2)],
        ]
    )


#: The per-atom vectors that turn with the moments: the moments themselves,
#: and their energy derivative, which is a covector of the same space.
MOMENT_VECTORS = ("magmom", "magforces")


@register_augmentation("magnetic_moments")
def magnetic_moments(mode: Literal["non-soc", "soc"] = "non-soc") -> Augmentation:
    """Turn every moment of a structure by one draw of a spin symmetry.

    ``non-soc`` draws from the whole of O(3) in spin space: a uniform rotation
    and, half the time, a reversal. Without spin-orbit coupling both are
    symmetries of the energy. ``soc`` draws the reversal alone: with the
    coupling, how the moments point relative to the lattice matters, and
    teaching a model otherwise washes out what it was meant to learn.

    The positions are never turned. One draw per structure, so the moments keep
    their relative orientation, and the magnetic forces turn with them.

    Raises:
        ValueError: For a mode that is neither.
    """
    if mode not in ("non-soc", "soc"):
        raise ValueError(
            f"the magnetic_moments augmentation's mode is {mode!r}; it is "
            f"'non-soc', the whole spin O(3), or 'soc', the reversal alone."
        )

    def augment(configuration: Configuration) -> Configuration:
        present = {
            name: configuration.properties.get(name)
            for name in MOMENT_VECTORS
            if configuration.properties.get(name) is not None
        }
        if "magmom" not in present:
            return configuration
        rotation = random_rotation() if mode == "non-soc" else np.eye(3)
        if float(torch.rand(())) < 0.5:
            rotation = -rotation
        turned = {
            name: np.asarray(value, dtype=float) @ rotation.T
            for name, value in present.items()
        }
        return replace(configuration, properties={**configuration.properties, **turned})

    return augment
