"""The magnetic energy is continuous as a neighbour crosses r_max.

Both magnetic interactions build their radial weights from the edge basis
concatenated with the sender's moment features. The edge basis carries the
cutoff envelope and vanishes at r_max; the moment features do not. Without the
envelope on the weights, a neighbour's message stays finite up to r_max and
disappears when the neighbour leaves the neighbour list, so the energy steps.

An Fe dimer is the smallest structure where one neighbour crosses r_max: just
inside, there is one edge each way; just outside, there are none.
"""

import numpy as np
import pytest
import torch
from e3nn import o3

from mace import data
from mace.modules import interaction_classes
from mace.modules.extensions import MagneticScaleShiftMACE
from mace.tools import AtomicNumberTable, torch_geometric
from mace.tools.torch_tools import default_dtype

R_MAX = 3.5
TABLE = AtomicNumberTable([26])

SOC_BLOCKS = [
    "MagneticRealAgnosticSpinOrbitCoupledDensityInteractionBlock",
    "MagneticRealAgnosticResidueSpinOrbitCoupledDensityInteractionBlock",
]


def _model(name):
    torch.manual_seed(0)
    return MagneticScaleShiftMACE(
        r_max=R_MAX,
        num_bessel=8,
        num_polynomial_cutoff=5,
        max_ell=2,
        interaction_cls=interaction_classes[name],
        interaction_cls_first=interaction_classes[name],
        num_interactions=2,
        num_elements=1,
        hidden_irreps=o3.Irreps("8x0e+8x1o"),
        MLP_irreps=o3.Irreps("8x0e"),
        atomic_energies=np.zeros(len(TABLE.zs)),
        avg_num_neighbors=1.0,
        atomic_numbers=TABLE.zs,
        correlation=[2, 2],
        gate=torch.nn.functional.silu,
        atomic_inter_shift=0.0,
        atomic_inter_scale=1.0,
        m_max=[3.0],
        num_mag_radial_basis=8,
        num_mag_radial_basis_one_body=10,
        max_m_ell=2,
        use_magmom_one_body=False,
        use_reduced_cg=False,
    )


def _dimer(distance, magmom):
    config = data.Configuration(
        atomic_numbers=np.array([26, 26]),
        positions=np.array([[0.0, 0.0, 0.0], [distance, 0.0, 0.0]]),
        properties={"magmom": np.array(magmom)},
        property_weights={"magmom": 1.0},
    )
    atomic_data = data.AtomicData.from_config(config, z_table=TABLE, cutoff=R_MAX)
    loader = torch_geometric.dataloader.DataLoader([atomic_data], batch_size=1)
    return next(iter(loader)).to_dict()


@pytest.mark.parametrize("name", SOC_BLOCKS)
@pytest.mark.parametrize(
    "magmom",
    [
        [[0.0, 0.0, 2.2], [0.0, 0.0, 2.2]],
        [[0.0, 0.0, 2.2], [0.0, 0.0, -2.2]],
        [[0.0, 0.0, 0.0], [0.0, 0.0, 0.0]],
    ],
    ids=["fm", "afm", "zero"],
)
def test_energy_and_magforces_are_continuous_at_r_max(name, magmom):
    with default_dtype(torch.float64):
        model = _model(name)
        inside = model(_dimer(R_MAX - 1e-6, magmom), training=False)
        outside = model(_dimer(R_MAX + 1e-6, magmom), training=False)

    # The uncorrected blocks step by 1e-2 to 1 eV here.
    step = float(inside["energy"][0] - outside["energy"][0])
    assert abs(step) < 1e-9, f"energy steps by {step:.3e} eV at r_max"
    np.testing.assert_allclose(
        inside["magforces"].detach().numpy(),
        outside["magforces"].detach().numpy(),
        rtol=0.0,
        atol=1e-9,
    )
    np.testing.assert_allclose(
        inside["forces"].detach().numpy(),
        outside["forces"].detach().numpy(),
        rtol=0.0,
        atol=1e-9,
    )
