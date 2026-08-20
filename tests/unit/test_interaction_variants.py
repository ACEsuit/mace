"""The attention-residual interaction, which `--interaction` offers and nothing built.

`RealAgnosticAttResidualInteractionBlock` is in `interaction_classes` and in the
`--interaction` choices, so a user can train with it today. Nothing in the tree
constructed it, which the functionality inventory surfaced as a gap: a block
reachable from the command line that no test has ever run.

It does work. This pins that, and pins it in the two ways that can fail
separately -- the layers really are that class (a wiring mistake would silently
give the default residual block), and the forward produces finite energies and
forces of the right shape.
"""

import numpy as np
import pytest
import torch
from ase import Atoms
from e3nn import o3

from mace import data, modules
from mace.tools import AtomicNumberTable, torch_geometric

ATT_RESIDUAL = "RealAgnosticAttResidualInteractionBlock"
PLAIN_RESIDUAL = "RealAgnosticResidualInteractionBlock"


@pytest.fixture(name="water_batch")
def water_batch_fixture():
    table = AtomicNumberTable([1, 8])
    atoms = Atoms("OH2", positions=[[0.0, 0.0, 0.0], [0.95, 0.0, 0.0], [-0.3, 0.9, 0.0]])
    config = data.Configuration(
        atomic_numbers=atoms.numbers,
        positions=atoms.positions,
        properties={"energy": 0.0, "forces": np.zeros((len(atoms), 3))},
        property_weights={"energy": 1.0, "forces": 1.0},
    )
    atomic_data = data.AtomicData.from_config(config, z_table=table, cutoff=5.0)
    loader = torch_geometric.dataloader.DataLoader([atomic_data], batch_size=1)
    return next(iter(loader)), table, len(atoms)


def _model(name, table):
    torch.manual_seed(0)
    return modules.MACE(
        r_max=5,
        num_bessel=8,
        num_polynomial_cutoff=6,
        max_ell=2,
        interaction_cls=modules.interaction_classes[name],
        interaction_cls_first=modules.interaction_classes[name],
        num_interactions=2,
        num_elements=2,
        hidden_irreps=o3.Irreps("16x0e + 16x1o"),
        MLP_irreps=o3.Irreps("8x0e"),
        gate=torch.nn.functional.silu,
        atomic_energies=np.array([1.0, 3.0]),
        avg_num_neighbors=8,
        atomic_numbers=table.zs,
        correlation=3,
        radial_type="bessel",
    )


def test_the_attention_residual_interaction_builds_and_runs(water_batch):
    batch, table, num_atoms = water_batch
    model = _model(ATT_RESIDUAL, table)

    wanted = modules.interaction_classes[ATT_RESIDUAL]
    assert [type(layer) for layer in model.interactions] == [wanted, wanted], (
        "the model was built with something other than the block under test, so "
        "this test would pass for the default residual block too"
    )

    out = model(batch.to_dict(), training=False)
    energy = out["energy"]
    forces = out["forces"]
    assert energy.shape == (1,)
    assert forces.shape == (num_atoms, 3)
    assert torch.isfinite(energy).all() and torch.isfinite(forces).all()


def test_the_attention_residual_interaction_is_not_the_plain_one(water_batch):
    """Same seed, same shapes, different numbers.

    Both blocks are residual and both accept the same arguments, so a mix-up
    between them changes no shape and raises nothing. The energies have to
    differ, or `--interaction` is offering two names for one model.
    """
    batch, table, _ = water_batch
    att = _model(ATT_RESIDUAL, table)(batch.to_dict(), training=False)["energy"]
    plain = _model(PLAIN_RESIDUAL, table)(batch.to_dict(), training=False)["energy"]
    assert not torch.allclose(att, plain), (
        f"the attention-residual block reproduced the plain residual energy "
        f"({float(att[0]):.12f}), so it is not doing anything of its own"
    )
