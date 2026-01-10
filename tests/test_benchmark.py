import json
import os
from pathlib import Path

import pandas as pd
import pytest
import torch
from ase import build

from mace import data as mace_data
from mace.calculators.foundations_models import mace_mp
from mace.tools import AtomicNumberTable, torch_geometric
from mace.tools.torch_tools import dtype_dict


def is_mace_full_bench():
    return os.environ.get("MACE_FULL_BENCH", "0") == "1"

def load_mace_mp_medium(dtype, compile_mode, device):
    calc = mace_mp(
        model="medium",
        default_dtype=dtype,
        device=device,
        compile_mode=compile_mode,
        fullgraph=False,
    )
    return calc.models[0].to(device=device, dtype=dtype_dict[dtype])


def create_batch(size: int, model: torch.nn.Module, device: str, dtype: torch.dtype) -> dict:
    cutoff = model.r_max.item()
    z_table = AtomicNumberTable([int(z) for z in model.atomic_numbers])
    atoms = build.bulk("C", "diamond", a=3.567, cubic=True)
    atoms = atoms.repeat((size, size, size))
    config = mace_data.config_from_atoms(atoms)
    dataset = [mace_data.AtomicData.from_config(config, z_table=z_table, cutoff=cutoff, dtype=dtype)]
    data_loader = torch_geometric.dataloader.DataLoader(
        dataset=dataset,
        batch_size=1,
        shuffle=False,
        drop_last=False,
    )
    batch = next(iter(data_loader))
    batch.to(device)
    return batch.to_dict()


def log_bench_info(benchmark, dtype, compile_mode, batch) -> None:
    benchmark.extra_info["num_atoms"] = int(batch["positions"].shape[0])
    benchmark.extra_info["num_edges"] = int(batch["edge_index"].shape[1])
    benchmark.extra_info["dtype"] = dtype
    benchmark.extra_info["is_compiled"] = compile_mode is not None
    benchmark.extra_info["device_name"] = torch.cuda.get_device_name()


@pytest.mark.skipif(not torch.cuda.is_available(), reason="cuda is not available")
@pytest.mark.benchmark(warmup=True, warmup_iterations=4, min_rounds=8)
@pytest.mark.parametrize("size", [3, 5, 7, 9])
@pytest.mark.parametrize("default_dtype_str", ["float32", "float64"])
@pytest.mark.parametrize("compile_mode", [None, "default"])
def test_inference(
    benchmark, size: int, default_dtype_str: str, compile_mode: str | None,
) -> None:
    device = "cuda"
    if not is_mace_full_bench() and compile_mode is not None:
        pytest.skip("Skipping long running benchmark, set MACE_FULL_BENCH=1 to execute")

    model = load_mace_mp_medium(default_dtype_str, compile_mode, device)
    batch = create_batch(size, model, device, dtype_dict[default_dtype_str])
    log_bench_info(benchmark, default_dtype_str, compile_mode, batch)

    def func() -> None:
        torch.cuda.synchronize()
        model(batch, training=compile_mode is not None, compute_force=True)

    torch.cuda.empty_cache()
    benchmark(func)


if __name__ == "__main__":
    # Print to stdout a csv of the benchmark metrics
    import subprocess

    def process_benchmark_file(bench_file: Path) -> pd.DataFrame:
        with open(bench_file, encoding="utf-8") as f:
            bench_data = json.load(f)

        records = []
        for bench in bench_data["benchmarks"]:
            record = {**bench["extra_info"], **bench["stats"]}
            records.append(record)

        result_df = pd.DataFrame(records)
        result_df["ns/day (1 fs/step)"] = 0.086400 / result_df["median"]
        result_df["Steps per day"] = result_df["ops"] * 86400
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
        return result_df[columns]

    def read_bench_results(result_files: list[str]) -> pd.DataFrame:
        return pd.concat([process_benchmark_file(Path(f)) for f in result_files])

    result = subprocess.run(
        ["pytest-benchmark", "list"], capture_output=True, text=True, check=True,
    )

    bench_files = result.stdout.strip().split("\n")
    bench_results = read_bench_results(bench_files)
