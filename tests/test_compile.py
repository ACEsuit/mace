from functools import wraps
from typing import Callable

import numpy as np
import pandas as pd
import pytest
import torch
import torch.nn.functional
from e3nn import o3, set_optimization_defaults
from scipy.spatial.transform import Rotation as R

from mace import data, modules, tools
from mace.tools import torch_geometric

torch.set_default_dtype(torch.float64)
config = data.Configuration(
    atomic_numbers=np.array([8, 1, 1]),
    positions=np.array(
        [
            [0.0, -2.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
        ]
    ),
    forces=np.array(
        [
            [0.0, -1.3, 0.0],
            [1.0, 0.2, 0.0],
            [0.0, 1.1, 0.3],
        ]
    ),
    energy=-1.5,
    charges=np.array([-2.0, 1.0, 1.0]),
    dipole=np.array([-1.5, 1.5, 2.0]),
)
# Created the rotated environment
rot = R.from_euler("z", 60, degrees=True).as_matrix()
positions_rotated = np.array(rot @ config.positions.T).T
config_rotated = data.Configuration(
    atomic_numbers=np.array([8, 1, 1]),
    positions=positions_rotated,
    forces=np.array(
        [
            [0.0, -1.3, 0.0],
            [1.0, 0.2, 0.0],
            [0.0, 1.1, 0.3],
        ]
    ),
    energy=-1.5,
    charges=np.array([-2.0, 1.0, 1.0]),
    dipole=np.array([-1.5, 1.5, 2.0]),
)
table = tools.AtomicNumberTable([1, 8])
atomic_energies = np.array([1.0, 3.0], dtype=float)


def create_mace(device: str, seed: int = 1702):
    torch_geometric.seed_everything(seed)

    model_config = {
        "r_max": 5,
        "num_bessel": 8,
        "num_polynomial_cutoff": 6,
        "max_ell": 2,
        "interaction_cls": modules.interaction_classes[
            "RealAgnosticResidualInteractionBlock"
        ],
        "interaction_cls_first": modules.interaction_classes[
            "RealAgnosticResidualInteractionBlock"
        ],
        "num_interactions": 2,
        "num_elements": 2,
        "hidden_irreps": o3.Irreps("32x0e + 32x1o"),
        "MLP_irreps": o3.Irreps("16x0e"),
        "gate": torch.nn.functional.silu,
        "atomic_energies": atomic_energies,
        "avg_num_neighbors": 8,
        "atomic_numbers": table.zs,
        "correlation": 3,
        "radial_type": "bessel",
    }
    model = modules.MACE(**model_config)
    return model.to(device)


def create_batch(device: str):
    atomic_data = data.AtomicData.from_config(config, z_table=table, cutoff=3.0)
    atomic_data2 = data.AtomicData.from_config(
        config_rotated, z_table=table, cutoff=3.0
    )

    data_loader = torch_geometric.dataloader.DataLoader(
        dataset=[atomic_data, atomic_data2],
        batch_size=2,
        shuffle=True,
        drop_last=False,
    )
    batch = next(iter(data_loader))
    batch = batch.to(device)
    batch = batch.to_dict()
    return batch


def time_func(func: Callable):
    @wraps(func)
    def wrapper(*args, **kwargs):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        results = func(*args, **kwargs)
        end.record()
        torch.cuda.synchronize()
        return results, start.elapsed_time(end) / 1000

    return wrapper


@pytest.mark.parametrize("device", ["cpu", "cuda"])
def test_mace(device):
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip(reason="cuda is not available")

    model = create_mace(device)

    # Disable CodeGenMixin compilation to TorchScript module used in e3nn.o3.Linear
    set_optimization_defaults(jit_script_fx=False)
    model_compiled = torch.compile(create_mace(device), mode="default")

    batch = create_batch(device)
    output1 = model(batch, training=True)
    output2 = model_compiled(batch, training=True)
    assert torch.allclose(output1["energy"][0], output2["energy"][0])
    assert torch.allclose(output2["energy"][0], output2["energy"][1])


@pytest.mark.skipif(not torch.cuda.is_available(), reason="cuda is not available")
def test_inference_speedup():
    # PyTorch eager Baseline
    batch = create_batch("cuda")
    model = create_mace("cuda")
    model = time_func(model)

    # Disable CodeGenMixin compilation to TorchScript module used in e3nn.o3.Linear
    set_optimization_defaults(jit_script_fx=False)
    compiled = torch.compile(create_mace("cuda"), mode="default")
    compiled = time_func(compiled)

    nruns = 10
    t_eager = np.array([model(batch, training=False)[1] for _ in range(nruns)])
    t_compiled = np.array([compiled(batch, training=False)[1] for _ in range(nruns)])
    df = pd.DataFrame(
        {"eager": t_eager, "compiled": t_compiled, "speedup": t_eager / t_compiled}
    )
    print(f"\n\n{df.to_string(index=False)}\n\n")

    assert np.median(df["speedup"][-4:]) > 1, "Median compile speedup is less than 1"
