from pathlib import Path
import numpy as np
import pytest
import torch
import torch.nn.functional
from e3nn import o3
from e3nn.util import jit
from scipy.spatial.transform import Rotation as R

from mace import data, modules, tools
from ase.build import molecule
from mace.tools import torch_geometric
from mace.tools.utils import load_foundations

torch.set_default_dtype(torch.float32)
config = data.Configuration(
    atomic_numbers=molecule("H2COH").numbers,
    positions=molecule("H2COH").positions,
    forces=molecule("H2COH").positions,
    energy=-1.5,
    charges=molecule("H2COH").numbers,
    dipole=np.array([-1.5, 1.5, 2.0]),
)
# Created the rotated environment
rot = R.from_euler("z", 60, degrees=True).as_matrix()
positions_rotated = np.array(rot @ config.positions.T).T
config_rotated = data.Configuration(
    atomic_numbers=molecule("H2COH").numbers,
    positions=positions_rotated,
    forces=molecule("H2COH").positions,
    energy=-1.5,
    charges=molecule("H2COH").numbers,
    dipole=np.array([-1.5, 1.5, 2.0]),
)
table = tools.AtomicNumberTable([1, 6, 8])
atomic_energies = np.array([0.0, 0.0, 0.0], dtype=float)


@pytest.skip("Problem with the float type", allow_module_level=True)
def test_foundations():
    # Create MACE model
    model_config = dict(
        r_max=6,
        num_bessel=10,
        num_polynomial_cutoff=10,
        max_ell=3,
        interaction_cls=modules.interaction_classes[
            "RealAgnosticResidualInteractionBlock"
        ],
        interaction_cls_first=modules.interaction_classes[
            "RealAgnosticResidualInteractionBlock"
        ],
        num_interactions=2,
        num_elements=3,
        hidden_irreps=o3.Irreps("64x0e + 64x1o + 64x2e"),
        MLP_irreps=o3.Irreps("64x0e"),
        gate=torch.nn.functional.silu,
        atomic_energies=atomic_energies,
        avg_num_neighbors=3,
        atomic_numbers=table.zs,
        correlation=3,
        radial_type="bessel",
        atomic_inter_scale=0.1,
        atomic_inter_shift=0.0,
    )
    model = modules.ScaleShiftMACE(**model_config)
    foundation_path = (
        Path(__file__).parent.parent
        / "mace"
        / "calculators"
        / "foundations_models"
        / "2023-08-14-mace-universal.model"
    )
    model_foundations = torch.load(foundation_path)
    model_loaded = load_foundations(
        model, model_foundations, table=table, load_readout=True, use_shift=False
    )
    atomic_data = data.AtomicData.from_config(config, z_table=table, cutoff=6.0)
    atomic_data2 = data.AtomicData.from_config(
        config_rotated, z_table=table, cutoff=6.0
    )

    data_loader = torch_geometric.dataloader.DataLoader(
        dataset=[atomic_data, atomic_data2],
        batch_size=2,
        shuffle=True,
        drop_last=False,
    )
    batch = next(iter(data_loader))
    forces_loaded = model_loaded(batch)["forces"]
    forces = model(batch)["forces"]
    assert torch.allclose(forces, forces_loaded)
