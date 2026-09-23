"""Write the reference values the in-tree long-range solver is checked against.

Run with the original ``graph_longrange`` importable, the version the in-tree
copy came from, and ``e3nn`` beside it, which that version needs::

    python make_electrostatics_reference.py

It writes ``electrostatics_reference.json`` beside itself: for each
periodicity mode, the inputs, the energy, and its gradients against the
positions, the multipoles and the cell. Nothing in the test suite imports the
original; the file is what carries its numbers.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import torch

OUTPUT = Path(__file__).with_name("electrostatics_reference.json")

MAX_L, SIGMA, KSPACE_FACTOR = 1, 1.0, 1.5

BOX = [[7.0, 0.3, 0.0], [0.0, 7.5, 0.2], [0.1, 0.0, 8.0]]
SLAB = [[7.0, 0.0, 0.0], [0.5, 7.0, 0.0], [0.0, 0.0, 25.0]]
OPEN = [[12.0, 0.0, 0.0], [0.0, 12.0, 0.0], [0.0, 0.0, 12.0]]

#: Mode, then per structure its periodicity, cell and atom count.
CASES = {
    "pbc": ("pbc", [((True, True, True), BOX, 5)]),
    "slab": ("slab", [((True, True, False), SLAB, 6)]),
    "realspace": ("realspace", [((False, False, False), OPEN, 4)]),
    "mixed_periodic": (
        "mixed_periodic",
        [((True, True, True), BOX, 5), ((False, False, False), OPEN, 4)],
    ),
}


def inputs(structures, seed: int) -> dict[str, list]:
    generator = torch.Generator().manual_seed(seed)
    positions, batch = [], []
    for index, (_, cell, count) in enumerate(structures):
        fractional = torch.rand(count, 3, generator=generator, dtype=torch.float64)
        positions.append(fractional @ torch.tensor(cell, dtype=torch.float64) * 0.8)
        batch += [index] * count
    total = sum(count for *_, count in structures)
    return {
        "positions": torch.cat(positions).tolist(),
        "batch": batch,
        "cell": [cell for _, cell, _ in structures],
        "pbc": [list(pbc) for pbc, _, _ in structures],
        "source_feats": torch.randn(
            total, (MAX_L + 1) ** 2, generator=generator, dtype=torch.float64
        ).tolist(),
    }


def evaluate(case: dict, mode: Any) -> dict[str, Any]:
    from graph_longrange.energy import (  # ty: ignore[unresolved-import]
        GTOElectrostaticEnergy,
    )
    from graph_longrange.gto_utils import (  # ty: ignore[unresolved-import]
        gto_basis_kspace_cutoff,
    )
    from graph_longrange.kspace import (  # ty: ignore[unresolved-import]
        compute_k_vectors_flat,
    )

    positions = torch.tensor(case["positions"], requires_grad=True)
    feats = torch.tensor(case["source_feats"], requires_grad=True)
    cell = torch.tensor(case["cell"], dtype=torch.float64, requires_grad=True)
    cutoff = KSPACE_FACTOR * gto_basis_kspace_cutoff([SIGMA], MAX_L)
    solver = GTOElectrostaticEnergy(
        density_max_l=MAX_L,
        density_smearing_width=SIGMA,
        kspace_cutoff=cutoff,
        pbc_handling=mode,
    )
    rcell = 2 * torch.pi * torch.linalg.inv(cell.transpose(-1, -2))
    volume = torch.linalg.det(cell).abs()
    k_vectors, k_norm2, k_batch, k0_mask = compute_k_vectors_flat(cutoff, cell, rcell)
    energy = solver(
        k_vectors=k_vectors,
        k_norm2=k_norm2,
        k_vector_batch=k_batch,
        k0_mask=k0_mask,
        source_feats=feats,
        node_positions=positions,
        batch=torch.tensor(case["batch"]),
        volume=volume,
        pbc=torch.tensor(case["pbc"]),
    )
    gradients = torch.autograd.grad(
        energy.sum(), [positions, feats, cell], allow_unused=True
    )
    names = ("d_positions", "d_source_feats", "d_cell")
    leaves = (positions, feats, cell)
    return {
        "kspace_cutoff": cutoff,
        "energy": energy.detach().tolist(),
        **{
            name: (torch.zeros_like(leaf) if value is None else value).tolist()
            for name, value, leaf in zip(names, gradients, leaves, strict=True)
        },
    }


def main() -> None:
    torch.set_default_dtype(torch.float64)
    import graph_longrange  # ty: ignore[unresolved-import]

    reference = {
        "source": "graph_longrange",
        "version": getattr(graph_longrange, "__version__", "unknown"),
        "max_l": MAX_L,
        "sigma": SIGMA,
        "cases": {},
    }
    for seed, (name, (mode, structures)) in enumerate(CASES.items()):
        case = inputs(structures, seed)
        reference["cases"][name] = {"mode": mode, **case, **evaluate(case, mode)}
    OUTPUT.write_text(json.dumps(reference, indent=1))


if __name__ == "__main__":
    main()
