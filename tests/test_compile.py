from functools import wraps
from typing import Callable

import numpy as np
import pandas as pd
import pytest
import torch
import torch.nn.functional as F
from e3nn import o3
from scipy.spatial.transform import Rotation as R

from mace import data, modules, tools
from mace.tools import torch_geometric, compile

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
        "max_ell": 3,
        "interaction_cls": modules.interaction_classes[
            "RealAgnosticResidualInteractionBlock"
        ],
        "interaction_cls_first": modules.interaction_classes[
            "RealAgnosticResidualInteractionBlock"
        ],
        "num_interactions": 2,
        "num_elements": 2,
        "hidden_irreps": o3.Irreps("128x0e + 128x1o"),
        "MLP_irreps": o3.Irreps("16x0e"),
        "gate": F.silu,
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
        torch._inductor.cudagraph_mark_step_begin()
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        func(*args, **kwargs)
        end.record()
        torch.cuda.synchronize()
        return start.elapsed_time(end) / 1000

    return wrapper


@pytest.mark.parametrize("device", ["cpu", "cuda"])
def test_mace(device):
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip(reason="cuda is not available")

    model_defaults = create_mace(device)
    tmp_model = compile.prepare(create_mace)(device)
    model_compiled = torch.compile(tmp_model, mode="default")

    batch = create_batch(device)
    output1 = model_defaults(batch, training=True)
    output2 = model_compiled(batch, training=True)
    assert torch.allclose(output1["energy"][0], output2["energy"][0])
    assert torch.allclose(output2["energy"][0], output2["energy"][1])


@pytest.mark.skipif(not torch.cuda.is_available(), reason="cuda is not available")
@pytest.mark.parametrize("compile_mode", ["default", "reduce-overhead", "max-autotune"])
@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_inference_speedup(compile_mode, dtype):
    torch.set_default_dtype(dtype)

    # PyTorch eager Baseline
    nruns = 16
    batch = create_batch("cuda")
    model = create_mace("cuda")
    model = time_func(model)
    t_eager = np.array([model(batch, training=False) for _ in range(nruns)])

    print(f'Compiling using mode="{compile_mode}"')
    torch.compiler.reset()
    model = compile.prepare(create_mace)("cuda")
    compiled = torch.compile(model, mode=compile_mode, fullgraph=True)
    compiled = time_func(compiled)
    t_compiled = np.array([compiled(batch, training=True) for _ in range(nruns)])

    df = pd.DataFrame(
        {
            "eager": t_eager,
            f"compile mode={compile_mode}": t_compiled,
            "speedup": t_eager / t_compiled,
        }
    )
    print(f"\n\n{df.to_string(index=False)}\n\n")

    assert np.median(df["speedup"][-4:]) > 1, "Median compile speedup is less than 1"


@pytest.mark.skipif(not torch.cuda.is_available(), reason="cuda is not available")
def test_graph_breaks():
    # Ideally we would have as few graph breaks as possible...
    import torch._dynamo as dynamo

    batch = create_batch("cuda")
    model = compile.prepare(create_mace)("cuda")
    explanation = dynamo.explain(model)(batch, training=False)
    print(explanation.break_reasons)
    assert explanation.graph_break_count == 2
