"""Keeping a representative part of a dataset.

A replay set is large, and training on all of it makes the fine-tune about the
replay set. So a head keeps a subset: first the structures a filter on their
elements allows, then either a random draw or farthest-point sampling over a
foundation model's descriptors, padded at random from the rest when the filter
let too few through.

The frozen tree's version is a 575-line command-line script whose selection
logic is reached from the training run by writing a file and reading it back.
Here it is a function over configurations.

**Two differences, both deliberate.** The frozen tree falls back to a random
draw whenever farthest-point sampling raises for any reason, including the
optional ``fpsample`` package not being installed, and says so only in the log
(``mace/cli/fine_tuning_select.py:421-429``). A run asked for farthest points
and got random ones. Here the fallback for a missing ``fpsample`` is a plain
farthest-point sampler, slower and the same method, and nothing else falls
back. And the random draws come from a generator of their own, seeded by the
caller, rather than from the global one the script seeds and then shares with
everything else in the process. The stream is the same one: a
``RandomState(seed)`` draws what ``np.random.seed(seed)`` followed by the
global draw does, so a random selection is the frozen tree's, index for index.
"""

from __future__ import annotations

import logging
from collections.abc import Sequence
from typing import Literal

import numpy as np
from mace_core.data.configuration import Configuration

__all__ = [
    "FILTERINGS",
    "METHODS",
    "Filtering",
    "Method",
    "SelectionError",
    "farthest_point_indices",
    "passes_filter",
    "select",
    "split_by_filter",
]

logger = logging.getLogger(__name__)

Filtering = Literal["none", "combinations", "exclusive", "inclusive"]
Method = Literal["random", "fps"]

#: The filters, for a caller that enumerates them.
FILTERINGS: tuple[str, ...] = ("none", "combinations", "exclusive", "inclusive")

#: The two ways a subset is drawn.
METHODS: tuple[str, ...] = ("random", "fps")

#: What a descriptor row holds for an element the structure lacks. Far enough
#: from every real descriptor that two structures with different elements are
#: never each other's nearest point, which is the frozen tree's choice and what
#: makes farthest-point sampling spread over element combinations first.
ABSENT = 10e10


class SelectionError(ValueError):
    """A selection that cannot be made as asked."""


def passes_filter(
    atomic_numbers: np.ndarray | Sequence[int],
    elements: Sequence[int],
    filtering: Filtering,
) -> bool:
    """Whether a structure's elements are what the filter allows.

    Args:
        atomic_numbers: The structure's atoms.
        elements: The set the filter reads.
        filtering: ``none`` allows everything; ``combinations`` only structures
            made of elements of the set, not necessarily all of them;
            ``exclusive`` exactly the set; ``inclusive`` at least the set.
    """
    present = {int(number) for number in atomic_numbers}
    wanted = {int(number) for number in elements}
    if filtering == "none":
        return True
    if filtering == "combinations":
        return present <= wanted
    if filtering == "exclusive":
        return present == wanted
    if filtering == "inclusive":
        return wanted <= present
    raise SelectionError(
        f"{filtering!r} is not a filter. The filters are {list(FILTERINGS)}."
    )


def split_by_filter(
    configurations: Sequence[Configuration],
    elements: Sequence[int],
    filtering: Filtering,
) -> tuple[list[Configuration], list[Configuration]]:
    """The structures the filter allows, and the rest, both in order."""
    if filtering != "none" and not elements:
        raise SelectionError(
            f"the {filtering!r} filter was given no elements, so it has "
            f"nothing to compare a structure's elements with."
        )
    passed, rest = [], []
    for configuration in configurations:
        allowed = passes_filter(configuration.atomic_numbers, elements, filtering)
        (passed if allowed else rest).append(configuration)
    return passed, rest


def _random_indices(count: int, available: int, generator: np.random.RandomState):
    if count > available:
        raise SelectionError(
            f"{count} structures were asked for and {available} are available "
            f"to draw from."
        )
    return generator.choice(list(range(available)), count, replace=False).tolist()


def farthest_point_indices(
    points: np.ndarray, count: int, generator: np.random.RandomState
) -> list[int]:
    """``count`` rows of ``points``, each the farthest from those already taken.

    Uses ``fpsample`` when it is installed, which is what the frozen tree calls,
    and an exact sampler otherwise. The two agree on the method and not always
    on the choice: ``fpsample``'s tree-based sampler approximates the distances
    it compares, so on near ties it can pick differently.

    Args:
        points: ``[n, d]``.
        count: How many to take, at most ``n``.
        generator: Draws the first point.
    """
    total = points.shape[0]
    if count > total:
        raise SelectionError(f"{count} farthest points were asked for out of {total}.")
    start = int(generator.randint(0, total))
    try:
        import fpsample  # ty: ignore[unresolved-import]
    except ImportError:
        return _exact_farthest_points(points, count, start)
    return [
        int(index)
        for index in fpsample.fps_npdu_kdtree_sampling(
            points.astype(np.float32), count, start_idx=start
        )
    ]


def _exact_farthest_points(points: np.ndarray, count: int, start: int) -> list[int]:
    """Farthest-point sampling by keeping every point's distance to the set."""
    values = points.astype(np.float64)
    chosen = [start]
    nearest = np.linalg.norm(values - values[start], axis=1)
    for _ in range(count - 1):
        candidate = int(np.argmax(nearest))
        chosen.append(candidate)
        nearest = np.minimum(
            nearest, np.linalg.norm(values - values[candidate], axis=1)
        )
    return chosen


def select(
    configurations: Sequence[Configuration],
    *,
    num_samples: int | None = None,
    method: Method = "fps",
    filtering: Filtering = "combinations",
    elements: Sequence[int] = (),
    allow_random_padding: bool = True,
    seed: int = 42,
    descriptors: np.ndarray | None = None,
) -> list[Configuration]:
    """The part of a dataset a head keeps.

    The defaults are the frozen tree's standalone script's: farthest points,
    ``combinations``, seed 42. The in-run ones differ and live on the
    configuration section, which is why neither is inferred from the other.

    Args:
        configurations: The whole dataset, in order.
        num_samples: How many to keep. ``None`` keeps all that pass the filter.
        method: ``random`` or ``fps``.
        filtering: See :func:`passes_filter`.
        elements: What the filter reads.
        allow_random_padding: Make up a shortfall at random from the structures
            the filter refused.
        seed: Seeds every random draw here, and nothing else in the process.
        descriptors: ``[n, d]`` over the structures the filter allows, in
            their order. Required for ``fps`` and read by nothing else.

    Raises:
        SelectionError: On a request that cannot be met, naming what fell
            short. The frozen tree raises from inside its random helper for the
            same cases, with a message about the remaining set.
    """
    if num_samples is not None and num_samples < 1:
        raise SelectionError(f"num_samples is {num_samples}; keep at least one.")
    generator = np.random.RandomState(seed)
    passed, rest = split_by_filter(configurations, elements, filtering)
    if num_samples is None or num_samples == len(passed):
        return passed
    if num_samples > len(passed):
        if not allow_random_padding:
            raise SelectionError(
                f"{len(passed)} structures pass the {filtering!r} filter and "
                f"{num_samples} were asked for, with random padding turned off. "
                f"Ask for fewer, loosen the filter, or allow padding."
            )
        shortfall = num_samples - len(passed)
        logger.info(
            "%d structures pass the %r filter; padding with %d drawn at random "
            "from the rest.",
            len(passed),
            filtering,
            shortfall,
        )
        return passed + [
            rest[index] for index in _random_indices(shortfall, len(rest), generator)
        ]
    if method == "random":
        return [
            passed[index]
            for index in _random_indices(num_samples, len(passed), generator)
        ]
    if method == "fps":
        if descriptors is None:
            raise SelectionError(
                "farthest-point sampling needs a descriptor per structure, and "
                "none were given."
            )
        if descriptors.shape[0] != len(passed):
            raise SelectionError(
                f"{descriptors.shape[0]} descriptors were given for "
                f"{len(passed)} structures."
            )
        return [
            passed[index]
            for index in farthest_point_indices(descriptors, num_samples, generator)
        ]
    raise SelectionError(f"{method!r} is not a method. They are {list(METHODS)}.")
