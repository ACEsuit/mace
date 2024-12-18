import os
from typing import Optional

import pandas as pd
import json
import pytest
import torch
from ase import build

from mace import data
from mace.calculators.foundations_models import mace_mp
from mace.tools import AtomicNumberTable, torch_geometric, torch_tools


def is_mace_full_bench():
    return os.environ.get("MACE_FULL_BENCH", "0") == "1"


@pytest.mark.skipif(not torch.cuda.is_available(), reason="cuda is not available")
@pytest.mark.benchmark(warmup=True, warmup_iterations=4, min_rounds=8)
@pytest.mark.parametrize("size", (3, 5, 7, 9))
@pytest.mark.parametrize("dtype", ["float32", "float64"])
@pytest.mark.parametrize("compile_mode", [None, "default"])
def test_inference(
    benchmark, size: int, dtype: str, compile_mode: Optional[str], device: str = "cuda"
):
    if not is_mace_full_bench() and compile_mode is not None:
        pytest.skip("Skipping long running benchmark, set MACE_FULL_BENCH=1 to execute")

    with torch_tools.default_dtype(dtype):
        model = load_mace_mp_medium(dtype, compile_mode, device)
        batch = create_batch(size, model, device)
        log_bench_info(benchmark, dtype, compile_mode, batch)

        def func():
            torch.cuda.synchronize()
            model(batch, training=compile_mode is not None, compute_force=True)

        torch.cuda.empty_cache()
        benchmark(func)


def load_mace_mp_medium(dtype, compile_mode, device):
    calc = mace_mp(
        model="medium",
        default_dtype=dtype,
        device=device,
        compile_mode=compile_mode,
        fullgraph=False,
    )
    model = calc.models[0].to(device)
    return model


def create_batch(size: int, model: torch.nn.Module, device: str) -> dict:
    cutoff = model.r_max.item()
    z_table = AtomicNumberTable([int(z) for z in model.atomic_numbers])
    atoms = build.bulk("C", "diamond", a=3.567, cubic=True)
    atoms = atoms.repeat((size, size, size))
    config = data.config_from_atoms(atoms)
    dataset = [data.AtomicData.from_config(config, z_table=z_table, cutoff=cutoff)]
    data_loader = torch_geometric.dataloader.DataLoader(
        dataset=dataset,
        batch_size=1,
        shuffle=False,
        drop_last=False,
    )
    batch = next(iter(data_loader))
    batch.to(device)
    return batch.to_dict()


def log_bench_info(benchmark, dtype, compile_mode, batch):
    benchmark.extra_info["num_atoms"] = int(batch["positions"].shape[0])
    benchmark.extra_info["num_edges"] = int(batch["edge_index"].shape[1])
    benchmark.extra_info["dtype"] = dtype
    benchmark.extra_info["is_compiled"] = compile_mode is not None
    benchmark.extra_info["device_name"] = torch.cuda.get_device_name()


def read_bench_results(files: list[str]) -> pd.DataFrame:
    def read(file):
        with open(file, "r") as f:
            data = json.load(f)

        records = []
        for bench in data["benchmarks"]:
            record = {**bench["extra_info"], **bench["stats"]}
            records.append(record)

        df = pd.DataFrame(records)
        df["ns/day (1 fs/step)"] = 0.086400 / df["median"]
        df["Steps per day"] = df["ops"] * 86400
        columns = [
            "num_atoms",
            "num_edges",
            "dtype",
            "is_compiled",
            "device_name",
            "median",
            "Steps per day",
            "ns/day (1 fs/step)",
        ]
        return df[columns]

    return pd.concat([read(f) for f in files])


if __name__ == "__main__":
    # Print to stdout a csv of the benchmark metrics
    import subprocess

    result = subprocess.run(
        ["pytest-benchmark", "list"], capture_output=True, text=True
    )

    if result.returncode != 0:
        raise RuntimeError(f"Command failed with return code {result.returncode}")

    files = result.stdout.strip().split("\n")
    df = read_bench_results(files)
    print(df.to_csv(index=False))
