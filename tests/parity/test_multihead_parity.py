"""A two-head legacy model, converted, against itself, head by head.

A multi-head model in the frozen tree shares its backbone and gives every head
its own readout, its own isolated-atom energies and its own scale and shift.
The rewrite has to hold exactly that, or a multi-head foundation model cannot
be converted at all: mh-0 has seven heads.

There is no multi-head anchor, so one is built from the trained single-head
one: the backbone is the anchor's, untouched, and the readouts, the energies
and the scale are replaced by two-head versions made with the same classes and
arguments the frozen tree's own constructor uses
(`mace/modules/models.py:216,259-271`). Every per-head constant is made to
differ between the heads, so a conversion that mixed them up, or read one
head's copy for the other, cannot agree by accident.
"""

from __future__ import annotations

import copy

import ase.io
import pytest
import torch
from mace_core.elements import AtomicNumberTable, ResolvedE0s
from mace_core.kernels.precision import PrecisionConfig
from mace_torch.backends.reference import ReferenceBackend
from mace_torch.models import EnergyOutputHead, MACEModel, ScaleShiftSpec
from mace_torch.physics import DerivativeEngine

from tests.parity.fm00_convert import (
    build_config,
    energy_constants_to_canonical,
    transfer_weights,
)
from tests.parity.test_fm00_keystone import ENERGY, GOLDEN, load_anchor, v1_graph

HEADS = ("replay", "target")


def two_headed(anchor):
    """The anchor, with every per-head part rebuilt for two heads."""
    from e3nn import o3

    from mace.modules.blocks import (
        AtomicEnergiesBlock,
        LinearReadoutBlock,
        NonLinearReadoutBlock,
        ScaleShiftBlock,
    )

    legacy = copy.deepcopy(anchor)
    legacy.heads = list(HEADS)
    count = len(HEADS)
    torch.manual_seed(20260923)
    width = legacy.readouts[1].hidden_irreps
    legacy.readouts[0] = LinearReadoutBlock(
        legacy.readouts[0].linear.irreps_in, o3.Irreps(f"{count}x0e")
    )
    legacy.readouts[1] = NonLinearReadoutBlock(
        legacy.readouts[1].linear_1.irreps_in,
        (count * width).simplify(),
        torch.nn.functional.silu,
        o3.Irreps(f"{count}x0e"),
        count,
    )
    energies = anchor.atomic_energies_fn.atomic_energies.detach().reshape(-1)
    legacy.atomic_energies_fn = AtomicEnergiesBlock(
        torch.stack([energies, energies + torch.tensor([0.3, -0.2, 0.1])])
    )
    scale = float(anchor.scale_shift.scale)
    shift = float(anchor.scale_shift.shift)
    legacy.scale_shift = ScaleShiftBlock(
        scale=[scale, 1.7 * scale], shift=[shift, shift - 0.4]
    )
    return legacy.to(torch.float64)


def convert(legacy):
    numbers = [int(z) for z in legacy.atomic_numbers.tolist()]
    config = build_config(legacy)
    values, scale, shift = energy_constants_to_canonical(legacy, heads=HEADS)
    head = EnergyOutputHead(
        ResolvedE0s(values),
        list(HEADS),
        AtomicNumberTable(numbers),
        ScaleShiftSpec("std", scale, shift),
        PrecisionConfig(),
        zbl_in_scale_shift=True,
    )
    model = MACEModel(
        ReferenceBackend(), observables=[ENERGY], energy_head=head, **config
    )
    transfer_weights(legacy, model, config["correlation"])
    return model, config


def legacy_batch(legacy, atoms, head: str):
    from mace import data as legacy_data
    from mace.tools import AtomicNumberTable as LegacyTable
    from mace.tools import torch_geometric

    numbers = [int(z) for z in legacy.atomic_numbers.tolist()]
    item = legacy_data.AtomicData.from_config(
        legacy_data.config_from_atoms(atoms, head_name=head),
        z_table=LegacyTable(numbers),
        cutoff=float(legacy.r_max),
        heads=list(HEADS),
    )
    loader = torch_geometric.DataLoader(dataset=[item], batch_size=1, shuffle=False)
    return next(iter(loader))


def structures():
    every = ase.io.read(GOLDEN / "fixtures/tiny_train.xyz", index=":")
    return [atoms for atoms in every if len(atoms) > 1]


@pytest.fixture
def pair(fp64):
    legacy = two_headed(load_anchor("tiny_scaleshift.model"))
    model, config = convert(legacy)
    return legacy, model, config


@pytest.mark.parametrize("head", range(len(HEADS)))
def test_every_head_of_the_converted_model_matches_the_live_one(pair, head):
    """Energy and forces, per head, on every structure of the fixture."""
    legacy, model, config = pair
    engine = DerivativeEngine(model, ENERGY)
    numbers = [int(z) for z in legacy.atomic_numbers.tolist()]
    worst_energy = worst_force = 0.0
    for atoms in structures():
        reference = legacy(
            legacy_batch(legacy, atoms, HEADS[head]).to_dict(),
            training=False,
            compute_force=True,
        )
        graph = v1_graph(atoms, numbers, config["cutoff"])
        graph["head"] = torch.tensor([head])
        result = engine(graph, compute=("forces",))
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
    assert worst_energy < 1e-12, f"head {HEADS[head]}: energy off by {worst_energy:.3e}"
    assert worst_force < 1e-12, f"head {HEADS[head]}: forces off by {worst_force:.3e}"


def test_the_two_heads_really_disagree(pair):
    """The guard: with identical heads, the test above could pass on a
    conversion that read head zero's copy for both."""
    legacy, _, _ = pair
    atoms = structures()[0]
    energies = [
        float(
            legacy(
                legacy_batch(legacy, atoms, name).to_dict(),
                training=False,
                compute_force=False,
            )["energy"].detach()
        )
        for name in HEADS
    ]
    assert abs(energies[0] - energies[1]) > 1e-2


def test_the_converted_model_holds_one_readout_per_head(pair):
    _, model, config = pair
    assert config["num_heads"] == len(HEADS)
    assert model.outputs.heads["energy"].num_heads == len(HEADS)
