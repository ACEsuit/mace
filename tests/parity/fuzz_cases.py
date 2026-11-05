"""PAR-2's six input classes. Coordinates and cells are in Angstrom.

No rejection sampling: each draw retains the pathology its class promises.
The serialized case contains the structure, not just a random seed.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np
from hypothesis import strategies as st

INPUT_CLASSES = (
    "triclinic_mixed_pbc",
    "periodic_self_images",
    "effective_cells",
    "empty_neighborhoods",
    "high_lmax",
    "mixed_precision",
)
CUTOFF = 3.0
MODEL_SEED = 20261105


@dataclass(frozen=True)
class Case:
    input_class: str
    geometry_seed: int
    positions: list[list[float]]
    numbers: list[int]
    cell: list[list[float]]
    pbc: list[bool]
    lmax: int = 2
    dtype: str = "float64"
    constants_dtype: str = "float64"
    regime: str = "periodic"

    def record(self):
        return asdict(self)


def _case(input_class, seed, *, count=3, **settings):
    rng = np.random.default_rng(seed)
    # Separated sites, with continuous jitter; no singular zero-length bonds.
    positions = np.arange(count)[:, None] * np.array([0.7, 0.3, 0.2])
    positions += rng.uniform(-0.1, 0.1, (count, 3))
    defaults = dict(
        positions=positions.tolist(),
        numbers=rng.choice([1, 8], count).tolist(),
        cell=(np.eye(3) * 5).tolist(),
        pbc=[True] * 3,
    )
    defaults.update(settings)
    return Case(input_class, seed, **defaults)


SEEDS = st.integers(min_value=0, max_value=2**32 - 1)


@st.composite
def triclinic_mixed_pbc(draw):
    seed = draw(SEEDS)
    lengths = draw(st.lists(st.floats(3.5, 6, allow_nan=False), min_size=3, max_size=3))
    shear = draw(st.floats(0.1, 0.8, allow_nan=False))
    cell = np.diag(lengths)
    cell[1, 0], cell[2, :2] = shear, [-shear / 2, shear / 3]
    mask = draw(st.integers(1, 6))  # always both periodic and open axes
    return _case(
        INPUT_CLASSES[0],
        seed,
        count=draw(st.integers(2, 5)),
        cell=cell.tolist(),
        pbc=[bool(mask & (1 << axis)) for axis in range(3)],
    )


@st.composite
def periodic_self_images(draw):
    side = draw(st.floats(1.9, 2.1, allow_nan=False))
    return _case(
        INPUT_CLASSES[1],
        draw(SEEDS),
        count=1,
        cell=(np.eye(3) * side).tolist(),
    )


@st.composite
def effective_cells(draw):
    regime = draw(st.sampled_from(("aperiodic", "physical", "zero_slab")))
    cell = np.eye(3) * draw(st.floats(4, 7, allow_nan=False))
    pbc = [True, True, False]
    if regime == "aperiodic":
        pbc = [False] * 3
        cell[:] = 0
    elif regime == "zero_slab":
        cell[2] = 0
    return _case(
        INPUT_CLASSES[2],
        draw(SEEDS),
        cell=cell.tolist(),
        pbc=pbc,
        regime=regime,
    )


@st.composite
def empty_neighborhoods(draw):
    count = draw(st.integers(1, 5))
    positions = np.zeros((count, 3))
    positions[:, 0] = np.arange(count) * 7
    # Also exercise connected nodes beside an isolated node.
    if count >= 3 and draw(st.booleans()):
        positions[1] = [1, 0.2, 0.1]
    return _case(
        INPUT_CLASSES[3],
        draw(SEEDS),
        count=count,
        positions=positions.tolist(),
        pbc=[False] * 3,
    )


@st.composite
def high_lmax(draw, *, max_lmax):
    # The reference declares 0 = unbounded. Four is a resource bound for this
    # CPU campaign, not a fabricated backend capability. Finite caps use their
    # two highest supported degrees; a low-cap backend cannot silently skip.
    upper = max_lmax or 4
    if upper < 2:
        raise ValueError("PAR-2 high_lmax requires a backend supporting lmax >= 2")
    return _case(
        INPUT_CLASSES[4],
        draw(SEEDS),
        lmax=draw(st.integers(max(2, upper - 1), upper)),
    )


@st.composite
def mixed_precision(draw):
    dtype, constants = draw(
        st.sampled_from(
            (
                ("float32", "float32"),
                ("float32", "float64"),
                ("float64", "float64"),
            )
        )
    )
    return _case(
        INPUT_CLASSES[5],
        draw(SEEDS),
        count=draw(st.integers(2, 5)),
        dtype=dtype,
        constants_dtype=constants,
    )


def strategies(max_lmax):
    return dict(
        zip(
            INPUT_CLASSES,
            (
                triclinic_mixed_pbc(),
                periodic_self_images(),
                effective_cells(),
                empty_neighborhoods(),
                high_lmax(max_lmax=max_lmax),
                mixed_precision(),
            ),
            strict=True,
        )
    )
