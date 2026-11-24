"""Shared fixtures and the one comparison helper of the mace-torch test suite."""

import numpy as np
import pytest
import torch
from mace_core.clebsch_gordan.irreps import Irreps
from mace_core.kernels.precision import Precision
from mace_torch.backends.harmonics import spherical_harmonics

#: Tolerance per dtype, `(atol, rtol)`. Each mirrors a row of
#: `tests/golden/harness.py`, which this package cannot import: the packages
#: are installed and tested without the repository's `tests/` tree on the
#: path. A change here is a tolerance change and goes through its own reviewed
#: PR.
#:
#: * float64 mirrors `closed_form_fp64`: an implementation against the closed
#:   form it implements, in one process, where the two sides differ only in
#:   the order of the arithmetic.
#: * float32 mirrors `fp32`: the same fp64 literals evaluated in fp32.
#:   Relative, with an absolute floor for values legitimately near zero.
TOLERANCES = {
    torch.float64: (1e-12, 1e-12),
    torch.float32: (5e-5, 1e-3),
}


@pytest.fixture(
    params=[torch.float32, torch.float64], ids=["float32", "float64"], autouse=True
)
def dtype(request):
    """Modules read `torch.get_default_dtype()` at construction, so every test
    in this suite runs once per supported dtype."""
    previous = torch.get_default_dtype()
    torch.set_default_dtype(request.param)
    try:
        yield request.param
    finally:
        torch.set_default_dtype(previous)


#: Restricts a test to fp64 by overriding the `dtype` parametrization, so the
#: test generates one case rather than a skip. For claims whose *method* needs
#: fp64: `gradcheck` and finite-difference slopes compare against difference
#: quotients that are rounding noise at fp32.
fp64_only = pytest.mark.parametrize(
    "dtype", [torch.float64], ids=["float64"], indirect=True
)

#: Restricts a test to fp32, for claims about a model that is not float64.
fp32_only = pytest.mark.parametrize(
    "dtype", [torch.float32], ids=["float32"], indirect=True
)

#: The precision name of each dtype the suite runs at.
PRECISIONS: dict[torch.dtype, Precision] = {
    torch.float64: "float64",
    torch.float32: "float32",
}


def precision() -> Precision:
    """The precision name of the active default dtype."""
    return PRECISIONS[torch.get_default_dtype()]


def assert_close(actual, expected, what: str = "") -> None:
    """`|actual - expected| <= atol + rtol * |expected|` at the row of the
    active default dtype."""
    atol, rtol = TOLERANCES[torch.get_default_dtype()]
    actual = np.asarray(
        actual.detach().cpu().numpy() if torch.is_tensor(actual) else actual,
        dtype=float,
    )
    expected = np.asarray(expected, dtype=float)
    assert actual.shape == expected.shape, f"{what}: {actual.shape} vs {expected.shape}"
    deviation = np.abs(actual - expected)
    bound = atol + rtol * np.abs(expected)
    worst = int(np.argmax(deviation - bound)) if deviation.size else 0
    assert np.all(deviation <= bound), (
        f"{what}: worst deviation {deviation.ravel()[worst]:.3e} at index {worst} "
        f"exceeds the {torch.get_default_dtype()} bound"
    )


@pytest.fixture
def no_gradients():
    """For a module whose tests read values and differentiate nothing:
    ``pytestmark = pytest.mark.usefixtures("no_gradients")``."""
    with torch.no_grad():
        yield


# ---------------------------------------------------------------------------
# Graphs and rotations
# ---------------------------------------------------------------------------

#: The elements of the test structures.
ELEMENTS = (1, 6, 8)

#: The cutoff the test graphs are built at.
R_MAX = 3.0


def graph_of(structures) -> dict[str, torch.Tensor]:
    """Aperiodic structures batched into the flat graph dict.

    Every ordered pair of atoms of one structure closer than the cutoff is an
    edge, which is what any neighbour list returns for a cluster with no
    cell, so the shifts are zero.
    """
    dtype = torch.get_default_dtype()
    positions, numbers, senders, receivers, batch, ptr = [], [], [], [], [], [0]
    for index, (atomic_numbers, coordinates) in enumerate(structures):
        offset = ptr[-1]
        coordinates = np.asarray(coordinates, dtype=float)
        distances = np.linalg.norm(coordinates[:, None] - coordinates[None], axis=-1)
        sender, receiver = np.nonzero((distances < R_MAX) & (distances > 0.0))
        senders.append(sender + offset)
        receivers.append(receiver + offset)
        positions.append(coordinates)
        numbers.extend(atomic_numbers)
        batch.extend([index] * len(atomic_numbers))
        ptr.append(offset + len(atomic_numbers))
    atomic_numbers = torch.tensor(numbers, dtype=torch.int64)
    edge_index = torch.tensor(
        np.stack([np.concatenate(senders), np.concatenate(receivers)]),
        dtype=torch.int64,
    )
    return {
        "positions": torch.tensor(np.concatenate(positions), dtype=dtype),
        "node_atomic_numbers": atomic_numbers,
        "edge_index": edge_index,
        "shifts": torch.zeros(edge_index.shape[1], 3, dtype=dtype),
        "batch": torch.tensor(batch, dtype=torch.int64),
        "ptr": torch.tensor(ptr, dtype=torch.int64),
    }


def clusters(*seeds: int, n_atoms: int = 6) -> dict[str, torch.Tensor]:
    """Small aperiodic clusters batched, one per seed, every pair within the
    cutoff an edge."""
    structures = []
    for seed in seeds:
        positions = np.random.default_rng(seed).uniform(-1.2, 1.2, size=(n_atoms, 3))
        structures.append(([8, 1, 1, 6, 1, 8, 1, 6][:n_atoms], positions))
    return graph_of(structures)


def cluster(seed: int = 0, n_atoms: int = 6) -> dict[str, torch.Tensor]:
    """One small aperiodic cluster, every pair within the cutoff an edge."""
    return clusters(seed, n_atoms=n_atoms)


def wigner_d(degree: int, rotation: torch.Tensor) -> torch.Tensor:
    """``D^l(R)`` in this project's real basis, read off the harmonics.

    ``Y(R x) = D(R) Y(x)`` for every direction, solved over enough random
    directions to fix it. The harmonics define the basis the features are in,
    so this is the matrix the features must transform by.
    """
    points = torch.randn(
        64, 3, generator=torch.Generator().manual_seed(3), dtype=torch.float64
    )
    start, stop = degree**2, (degree + 1) ** 2
    original = spherical_harmonics(points, degree)[:, start:stop]
    rotated = spherical_harmonics(points @ rotation.T, degree)[:, start:stop]
    return torch.linalg.lstsq(original, rotated).solution.T


def transform_features(features: torch.Tensor, irreps: str, rotation, parity: int):
    """Apply ``D(R)`` and the parity to grouped features."""
    pieces, start = [], 0
    for mul, irrep in Irreps.parse(irreps).terms:
        width = mul * irrep.dimension
        block = features[:, start : start + width].reshape(-1, mul, irrep.dimension)
        matrix = wigner_d(irrep.degree, rotation).to(features.dtype)
        sign = irrep.parity**1 if parity == -1 else 1
        pieces.append((sign * block @ matrix.T).reshape(-1, width))
        start += width
    return torch.cat(pieces, dim=-1)


def rotation_matrix(seed: int) -> torch.Tensor:
    generator = torch.Generator().manual_seed(seed)
    rotation, _ = torch.linalg.qr(
        torch.randn(3, 3, generator=generator, dtype=torch.float64)
    )
    return rotation if torch.det(rotation) > 0 else -rotation
