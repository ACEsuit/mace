from pathlib import Path

import numpy as np
import pytest
import torch
import torch.nn.functional
from ase.build import molecule
from e3nn import o3
from scipy.spatial.transform import Rotation as R

from mace import data, modules, tools
from mace.calculators import mace_mp, mace_off
from mace.tools import torch_geometric
from mace.tools.finetuning_utils import load_foundations_elements
from mace.tools.scripts_utils import extract_config_mace_model
from mace.tools.utils import AtomicNumberTable

MODEL_PATH = (
    Path(__file__).parent.parent
    / "mace"
    / "calculators"
    / "foundations_models"
    / "2023-12-03-mace-mp.model"
)

torch.set_default_dtype(torch.float64)
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


# @pytest.skip("Problem with the float type", allow_module_level=True)
def test_foundations():
    # Create MACE model
    model_config = dict(
        r_max=6,
        num_bessel=10,
        num_polynomial_cutoff=5,
        max_ell=3,
        interaction_cls=modules.interaction_classes[
            "RealAgnosticResidualInteractionBlock"
        ],
        interaction_cls_first=modules.interaction_classes[
            "RealAgnosticResidualInteractionBlock"
        ],
        num_interactions=2,
        num_elements=3,
        hidden_irreps=o3.Irreps("128x0e"),
        MLP_irreps=o3.Irreps("16x0e"),
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
    calc = mace_mp(
        model="small",
        device="cpu",
        default_dtype="float64",
    )
    model_foundations = calc.models[0]
    model_loaded = load_foundations_elements(
        model,
        model_foundations,
        table=table,
        load_readout=True,
        use_shift=False,
        max_L=0,
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


def test_multi_reference():
    config_multi = data.Configuration(
        atomic_numbers=molecule("H2COH").numbers,
        positions=molecule("H2COH").positions,
        forces=molecule("H2COH").positions,
        energy=-1.5,
        charges=molecule("H2COH").numbers,
        dipole=np.array([-1.5, 1.5, 2.0]),
        head="MP2",
    )
    table_multi = tools.AtomicNumberTable([1, 6, 8])
    atomic_energies_multi = np.array([[0.0, 0.0, 0.0], [0.0, 0.0, 0.0]], dtype=float)

    # Create MACE model
    model_config = dict(
        r_max=6,
        num_bessel=10,
        num_polynomial_cutoff=5,
        max_ell=3,
        interaction_cls=modules.interaction_classes[
            "RealAgnosticResidualInteractionBlock"
        ],
        interaction_cls_first=modules.interaction_classes[
            "RealAgnosticResidualInteractionBlock"
        ],
        num_interactions=2,
        num_elements=3,
        hidden_irreps=o3.Irreps("128x0e + 128x1o"),
        MLP_irreps=o3.Irreps("16x0e"),
        gate=torch.nn.functional.silu,
        atomic_energies=atomic_energies_multi,
        avg_num_neighbors=61,
        atomic_numbers=table.zs,
        correlation=3,
        radial_type="bessel",
        atomic_inter_scale=[1.0, 1.0],
        atomic_inter_shift=[0.0, 0.0],
        heads=["MP2", "DFT"],
    )
    model = modules.ScaleShiftMACE(**model_config)
    calc_foundation = mace_mp(device="cpu", default_dtype="float64")
    model_loaded = load_foundations_elements(
        model,
        calc_foundation.models[0],
        table=table,
        load_readout=True,
        use_shift=False,
        max_L=1,
    )
    atomic_data = data.AtomicData.from_config(
        config_multi, z_table=table_multi, cutoff=6.0, heads=["MP2", "DFT"]
    )
    data_loader = torch_geometric.dataloader.DataLoader(
        dataset=[atomic_data, atomic_data],
        batch_size=2,
        shuffle=True,
        drop_last=False,
    )
    batch = next(iter(data_loader))
    forces_loaded = model_loaded(batch)["forces"]
    calc_foundation = mace_mp(device="cpu", default_dtype="float64")
    atoms = molecule("H2COH")
    atoms.info["head"] = "MP2"
    atoms.calc = calc_foundation
    forces = atoms.get_forces()
    assert np.allclose(
        forces, forces_loaded.detach().numpy()[:5, :], atol=1e-5, rtol=1e-5
    )


@pytest.mark.parametrize(
    "model",
    [
        mace_mp(model="small", device="cpu", default_dtype="float64").models[0],
        mace_mp(model="medium", device="cpu", default_dtype="float64").models[0],
        mace_mp(model="large", device="cpu", default_dtype="float64").models[0],
        mace_mp(model=MODEL_PATH, device="cpu", default_dtype="float64").models[0],
        mace_off(model="small", device="cpu", default_dtype="float64").models[0],
        mace_off(model="medium", device="cpu", default_dtype="float64").models[0],
        mace_off(model="large", device="cpu", default_dtype="float64").models[0],
        mace_off(model=MODEL_PATH, device="cpu", default_dtype="float64").models[0],
    ],
)
def test_extract_config(model):
    assert isinstance(model, modules.ScaleShiftMACE)
    model_copy = modules.ScaleShiftMACE(**extract_config_mace_model(model))
    model_copy.load_state_dict(model.state_dict())
    z_table = AtomicNumberTable([int(z) for z in model.atomic_numbers])
    atomic_data = data.AtomicData.from_config(config, z_table=z_table, cutoff=6.0)
    data_loader = torch_geometric.dataloader.DataLoader(
        dataset=[atomic_data, atomic_data],
        batch_size=2,
        shuffle=True,
        drop_last=False,
    )
    batch = next(iter(data_loader))
    output = model(batch)
    output_copy = model_copy(batch)
    # assert all items of the output dicts are equal
    for key in output.keys():
        if isinstance(output[key], torch.Tensor):
            assert torch.allclose(output[key], output_copy[key], atol=1e-5)
