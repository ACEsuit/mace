"""The residual SOC interaction, which `--interaction` offers and nothing built.

`MagneticRealAgnosticResidueSpinOrbitCoupledDensityInteractionBlock` is in
`interaction_classes` and in the `--interaction` choices, and the magnetic tests
all use its non-residual sibling
(`MagneticRealAgnosticSpinOrbitCoupledDensityInteractionBlock`). So the residual
variant was trainable from the command line with no test ever constructing it,
which the functionality inventory surfaced as a gap.

Note where it has to be built: in plain `MACE` it fails with
`TypeError: 'NoneType' object is not iterable`, because plain MACE passes no
magmom arguments. That reads as a broken block and is not one -- it belongs in
the magnetic model, which is what this builds.
"""

import numpy as np
import pytest
import torch
from e3nn import o3

from mace import data
from mace.modules import interaction_classes
from mace.modules.extensions import MagneticScaleShiftMACE
from mace.tools import AtomicNumberTable, torch_geometric

RESIDUAL_SOC = "MagneticRealAgnosticResidueSpinOrbitCoupledDensityInteractionBlock"
PLAIN_SOC = "MagneticRealAgnosticSpinOrbitCoupledDensityInteractionBlock"


@pytest.fixture(name="iron_batch")
def iron_batch_fixture():
    table = AtomicNumberTable([26])
    config = data.Configuration(
        atomic_numbers=np.array([26, 26]),
        positions=np.array([[0.0, 0.0, 0.0], [2.3, 0.0, 0.0]]),
        properties={"magmom": np.tile([[0.0, 0.0, 2.2]], (2, 1))},
        property_weights={"magmom": 1.0},
    )
    atomic_data = data.AtomicData.from_config(config, z_table=table, cutoff=3.5)
    loader = torch_geometric.dataloader.DataLoader([atomic_data], batch_size=1)
    return next(iter(loader)), table, 2


def _model(name, table):
    torch.manual_seed(0)
    return MagneticScaleShiftMACE(
        r_max=3.5,
        num_bessel=4,
        num_polynomial_cutoff=4,
        max_ell=2,
        interaction_cls=interaction_classes[name],
        interaction_cls_first=interaction_classes[name],
        num_interactions=1,
        num_elements=1,
        hidden_irreps=o3.Irreps("8x0e"),
        MLP_irreps=o3.Irreps("4x0e"),
        atomic_energies=np.zeros(len(table.zs)),
        avg_num_neighbors=1.0,
        atomic_numbers=table.zs,
        correlation=[1],
        gate=torch.nn.functional.silu,
        atomic_inter_shift=0.0,
        atomic_inter_scale=1.0,
        m_max=[3.0],
        num_mag_radial_basis=8,
        num_mag_radial_basis_one_body=10,
        max_m_ell=1,
        use_magmom_one_body=False,
    )


def test_the_residual_soc_interaction_builds_and_runs(iron_batch):
    batch, table, num_atoms = iron_batch
    model = _model(RESIDUAL_SOC, table)

    wanted = interaction_classes[RESIDUAL_SOC]
    assert [type(layer) for layer in model.interactions] == [wanted], (
        "the model was built with something other than the block under test, so "
        "this test would pass for the non-residual variant too"
    )

    out = model(batch.to_dict(), training=False)
    energy, magforces = out["energy"], out["magforces"]
    assert energy.shape == (1,)
    assert magforces.shape == (num_atoms, 3)
    assert torch.isfinite(energy).all() and torch.isfinite(magforces).all()


def test_the_residual_soc_interaction_is_not_the_plain_one(iron_batch):
    """Same seed, same shapes, different numbers.

    The two blocks take the same arguments and produce the same shapes, so
    confusing them raises nothing. The energies have to differ, or
    `--interaction` is offering two names for one model.
    """
    batch, table, _ = iron_batch
    residual = _model(RESIDUAL_SOC, table)(batch.to_dict(), training=False)["energy"]
    plain = _model(PLAIN_SOC, table)(batch.to_dict(), training=False)["energy"]
    assert not torch.allclose(residual, plain), (
        f"the residual SOC block reproduced the non-residual energy "
        f"({float(residual[0]):.12f}), so it is not doing anything of its own"
    )
