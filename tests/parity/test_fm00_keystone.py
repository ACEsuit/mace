"""The keystone: a trained legacy model, run through the whole v1 stack.

Every piece has been checked on its own by now. This is the first time a real
trained model goes in one end and energies and forces come out the other, and
it is the only test that can catch a mistake in how the pieces fit rather than
in the pieces.

It compares against the **live** frozen model in the same process rather than
against a stored number, so a disagreement is between two implementations
evaluated on the same inputs with nothing in between.
"""

from __future__ import annotations

import os
from pathlib import Path

import ase.io
import numpy as np
import pytest
import torch
from fm00_convert import build_config, energy_constants_to_canonical, transfer_weights
from mace_core.elements import AtomicNumberTable, ResolvedE0s
from mace_core.kernels.precision import PrecisionConfig
from mace_core.neighbors import get_neighborhood
from mace_core.observables import ObservableSpec
from mace_torch.backends.reference import ReferenceBackend
from mace_torch.models import EnergyOutputHead, MACEModel, ScaleShiftSpec
from mace_torch.physics import DerivativeEngine
from mace_torch.serialization import load_checkpoint, save_checkpoint

GOLDEN = Path(__file__).resolve().parents[1] / "golden"
ENERGY = ObservableSpec(
    name="energy", irreps="0e", per_atom=False, units="eV", normalization="none"
)

#: The fp64 row of the golden tolerance table. Stated here because this package
#: cannot import that harness, and a change to it is its own reviewed PR.
ENERGY_TOLERANCE = 1e-6
FORCE_TOLERANCE = 1e-6


def load_anchor(name: str):
    previous = os.environ.get("TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD")
    os.environ["TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD"] = "1"
    try:
        return torch.load(
            GOLDEN / "models" / name, map_location="cpu", weights_only=False
        )
    finally:
        if previous is None:
            os.environ.pop("TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD", None)
        else:
            os.environ["TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD"] = previous


def convert(legacy):
    """The whole conversion, config and weights."""
    numbers = [int(z) for z in legacy.atomic_numbers.tolist()]
    config = build_config(legacy)
    values, scale, shift = energy_constants_to_canonical(legacy)
    head = EnergyOutputHead(
        ResolvedE0s(values),
        ["default"],
        AtomicNumberTable(numbers),
        ScaleShiftSpec("std", scale, shift),
        PrecisionConfig(),
        zbl_in_scale_shift=getattr(legacy, "scale_shift", None) is not None,
    )
    model = MACEModel(
        ReferenceBackend(), observables=[ENERGY], energy_head=head, **config
    )
    transfer_weights(legacy, model, config["correlation"])
    return model, config


def legacy_batch(legacy, atoms):
    from mace import data as legacy_data
    from mace.tools import AtomicNumberTable as LegacyTable
    from mace.tools import torch_geometric

    numbers = [int(z) for z in legacy.atomic_numbers.tolist()]
    item = legacy_data.AtomicData.from_config(
        legacy_data.config_from_atoms(atoms),
        z_table=LegacyTable(numbers),
        cutoff=float(legacy.r_max),
    )
    loader = torch_geometric.DataLoader(dataset=[item], batch_size=1, shuffle=False)
    return next(iter(loader))


def v1_graph(atoms, numbers, cutoff):
    positions = atoms.get_positions()
    periodic = tuple(bool(flag) for flag in atoms.pbc)
    cell = np.array(atoms.cell) if any(periodic) else None
    neighborhood = get_neighborhood(positions, cutoff, periodic, cell)
    index = {z: i for i, z in enumerate(numbers)}
    return {
        "positions": torch.tensor(positions),
        "atomic_numbers": torch.tensor(atoms.get_atomic_numbers()),
        "element_index": torch.tensor(
            [index[int(z)] for z in atoms.get_atomic_numbers()]
        ),
        "edge_index": torch.tensor(neighborhood.edge_index),
        "shifts": torch.tensor(neighborhood.shifts),
        "unit_shifts": torch.tensor(neighborhood.unit_shifts),
        "cell": torch.tensor(np.asarray(neighborhood.cell, dtype=float)).reshape(
            1, 3, 3
        ),
        "pbc": torch.tensor([list(periodic)]),
        "batch": torch.zeros(len(atoms), dtype=torch.long),
        "num_graphs": 1,
        "head": torch.zeros(1, dtype=torch.long),
    }


@pytest.mark.parametrize("anchor", ["tiny_scaleshift.model", "tiny_mace.model"])
def test_a_converted_anchor_matches_the_live_legacy_model(fp64, anchor):
    """Energy and forces, on every structure of the training fixture.

    Both model classes, so the two places the frozen tree puts the pair
    repulsion are exercised: inside the scaled sum in one and outside it in the
    other.
    """
    legacy = load_anchor(anchor)
    model, config = convert(legacy)
    engine = DerivativeEngine(model)
    numbers = [int(z) for z in legacy.atomic_numbers.tolist()]

    structures = ase.io.read(GOLDEN / "fixtures/tiny_train.xyz", index=":")
    interesting = [atoms for atoms in structures if len(atoms) > 1]
    assert len(interesting) >= 5, "the fixture has too little in it to say much"

    worst_energy = worst_force = 0.0
    for atoms in interesting:
        reference = legacy(
            legacy_batch(legacy, atoms).to_dict(), training=False, compute_force=True
        )
        result = engine(v1_graph(atoms, numbers, config["cutoff"]), compute=("forces",))
        worst_energy = max(
            worst_energy,
            abs(
                float(reference["energy"].detach())
                - float(result.total_energy.detach())
            ),
        )
        worst_force = max(
            worst_force,
            float((reference["forces"].detach() - result.forces.detach()).abs().max()),
        )

    assert worst_energy < ENERGY_TOLERANCE, (
        f"the converted model's energy differs by {worst_energy:.3e} eV"
    )
    assert worst_force < FORCE_TOLERANCE, (
        f"the converted model's forces differ by {worst_force:.3e} eV/A"
    )
    # Far below the gate rather than just inside it: the only inexact step in
    # the whole conversion is the basis projection, at 1e-14.
    assert worst_energy < 1e-12 and worst_force < 1e-12, (
        f"the conversion is inside the gate at {worst_energy:.3e} eV and "
        f"{worst_force:.3e} eV/A, but far enough above rounding that something "
        f"is approximating where nothing should be"
    )


def test_a_converted_anchor_survives_the_checkpoint(fp64, tmp_path):
    """Converted, written, read back through the ordinary loader, still right.

    The loader has no converter-specific path: if the neutral format could not
    carry a converted model, this is where it would show.
    """
    legacy = load_anchor("tiny_scaleshift.model")
    model, config = convert(legacy)
    numbers = [int(z) for z in legacy.atomic_numbers.tolist()]
    values, scale, shift = energy_constants_to_canonical(legacy)

    save_checkpoint(tmp_path / "converted", model, config)

    def rebuild(stored):
        head = EnergyOutputHead(
            ResolvedE0s(values),
            ["default"],
            AtomicNumberTable(numbers),
            ScaleShiftSpec("std", scale, shift),
            PrecisionConfig(),
        )
        return MACEModel(
            ReferenceBackend(), observables=[ENERGY], energy_head=head, **stored
        )

    restored = load_checkpoint(tmp_path / "converted", rebuild)

    atoms = ase.io.read(GOLDEN / "fixtures/tiny_train.xyz", index=3)
    reference = legacy(
        legacy_batch(legacy, atoms).to_dict(), training=False, compute_force=True
    )
    graph = v1_graph(atoms, numbers, config["cutoff"])
    result = DerivativeEngine(restored)(graph, compute=("forces",))

    assert (
        abs(float(reference["energy"].detach()) - float(result.total_energy.detach()))
        < 1e-12
    )
