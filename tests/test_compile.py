from functools import wraps
from typing import Callable
import os

import numpy as np
import pytest
import torch
import torch.nn.functional as F
from e3nn import o3
from torch.testing import assert_close

from mace import data, modules, tools
from mace.tools import compile as mace_compile
from mace.tools import torch_geometric

table = tools.AtomicNumberTable([6])
atomic_energies = np.array([1.0], dtype=float)
cutoff = 5.0


def create_mace(device: str, seed: int = 1702):
    torch_geometric.seed_everything(seed)

    model_config = {
        "r_max": cutoff,
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
        "num_elements": 1,
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
    from ase import build

    size = 2
    atoms = build.bulk("C", "diamond", a=3.567, cubic=True)
    atoms_list = [atoms.repeat((size, size, size))]
    print("Number of atoms", len(atoms_list[0]))

    configs = [data.config_from_atoms(atoms) for atoms in atoms_list]
    data_loader = torch_geometric.dataloader.DataLoader(
        dataset=[
            data.AtomicData.from_config(config, z_table=table, cutoff=cutoff)
            for config in configs
        ],
        batch_size=1,
        shuffle=False,
        drop_last=False,
    )
    batch = next(iter(data_loader))
    batch = batch.to(device)
    batch = batch.to_dict()
    return batch


def time_func(func: Callable):
    @wraps(func)
    def wrapper(*args, **kwargs):
        torch._inductor.cudagraph_mark_step_begin()  # pylint: disable=W0212
        outputs = func(*args, **kwargs)
        torch.cuda.synchronize()
        return outputs

    return wrapper


@pytest.fixture(params=[torch.float32, torch.float64], ids=["fp32", "fp64"])
def default_dtype(request):
    with tools.torch_tools.default_dtype(request.param):
        yield torch.get_default_dtype()


# skip if on windows
@pytest.mark.skipif(os.name == "nt", reason="Not supported on Windows")
@pytest.mark.parametrize("device", ["cpu", "cuda"])
def test_mace(device, default_dtype):  # pylint: disable=W0621
    print(f"using default dtype = {default_dtype}")
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip(reason="cuda is not available")

    model_defaults = create_mace(device)
    tmp_model = mace_compile.prepare(create_mace)(device)
    model_compiled = torch.compile(tmp_model, mode="default")

    batch = create_batch(device)
    output1 = model_defaults(batch, training=True)
    output2 = model_compiled(batch, training=True)
    assert_close(output1["energy"], output2["energy"])
    assert_close(output1["forces"], output2["forces"])


@pytest.mark.skipif(os.name == "nt", reason="Not supported on Windows")
@pytest.mark.skipif(not torch.cuda.is_available(), reason="cuda is not available")
def test_eager_benchmark(benchmark, default_dtype):  # pylint: disable=W0621
    print(f"using default dtype = {default_dtype}")
    batch = create_batch("cuda")
    model = create_mace("cuda")
    model = time_func(model)
    benchmark(model, batch, training=True)


@pytest.mark.skipif(os.name == "nt", reason="Not supported on Windows")
@pytest.mark.skipif(not torch.cuda.is_available(), reason="cuda is not available")
@pytest.mark.parametrize("compile_mode", ["default", "reduce-overhead", "max-autotune"])
@pytest.mark.parametrize("enable_amp", [False, True], ids=["fp32", "mixed"])
def test_compile_benchmark(benchmark, compile_mode, enable_amp):
    with tools.torch_tools.default_dtype(torch.float32):
        batch = create_batch("cuda")
        torch.compiler.reset()
        model = mace_compile.prepare(create_mace)("cuda")
        model = torch.compile(model, mode=compile_mode, fullgraph=True)
        model = time_func(model)

        with torch.autocast("cuda", enabled=enable_amp):
            benchmark(model, batch, training=True)


@pytest.mark.skipif(os.name == "nt", reason="Not supported on Windows")
@pytest.mark.skipif(not torch.cuda.is_available(), reason="cuda is not available")
def test_graph_breaks():
    import torch._dynamo as dynamo

    batch = create_batch("cuda")
    model = mace_compile.prepare(create_mace)("cuda")
    explanation = dynamo.explain(model)(batch, training=False)

    # these clutter the output but might be useful for investigating graph breaks
    explanation.ops_per_graph = None
    explanation.out_guards = None
    print(explanation)
    assert explanation.graph_break_count == 0
