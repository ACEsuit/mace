"""Benchmark full cuEq PolarMACE compilation with NVIDIA electrostatics."""

import gc
import os
import statistics
import time
from dataclasses import dataclass
from typing import Callable, Dict

os.environ.setdefault("TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD", "1")
os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

import torch
from ase import build

from mace import data as mace_data
from mace.calculators.foundations_models import download_mace_polar_checkpoint
from mace.cli.convert_e3nn_cueq import run as run_e3nn_to_cueq
from mace.modules.polar_backends import NVALCHEMIOPS_BACKEND
from mace.tools import AtomicNumberTable, torch_geometric
from mace.tools.compile import configure_autograd_for_compile, simplify

MODEL_NAME = "polar-1-s"
SUPERCELL_SIDES = (2, 3, 4, 5)


@dataclass
class Timing:
    median_ms: float
    min_ms: float
    max_ms: float
    peak_gib: float


def _clear_cuda() -> None:
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.synchronize()


def _clone_batch(batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    cloned = {key: value.detach().clone() for key, value in batch.items()}
    cloned["positions"].requires_grad_(True)
    return cloned


def _load_model(device: torch.device) -> torch.nn.Module:
    model_path = download_mace_polar_checkpoint(MODEL_NAME)
    source_model = torch.load(
        model_path, map_location=device, weights_only=False
    ).eval()
    dtype = next(source_model.parameters()).dtype
    torch.set_default_dtype(dtype)
    model = run_e3nn_to_cueq(source_model, device=device.type, layout="ir_mul").eval()
    del source_model

    model.set_electrostatics_backend(NVALCHEMIOPS_BACKEND)
    model = simplify(model).to(device=device, dtype=dtype).eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)

    if model.electrostatics_backend != NVALCHEMIOPS_BACKEND:
        raise RuntimeError("The benchmark must use NVIDIA electrostatics")
    return model


def _build_batch(
    model: torch.nn.Module, side: int, device: torch.device
) -> Dict[str, torch.Tensor]:
    atoms = build.bulk("C", "diamond", a=3.567, cubic=True).repeat((side, side, side))
    z_table = AtomicNumberTable([int(z) for z in model.atomic_numbers])
    config = mace_data.config_from_atoms(atoms, head_name="Default")
    atomic_data = mace_data.AtomicData.from_config(
        config,
        z_table=z_table,
        cutoff=float(model.r_max),
        heads=model.heads,
    )
    loader = torch_geometric.dataloader.DataLoader(
        dataset=[atomic_data],
        batch_size=1,
        shuffle=False,
        drop_last=False,
    )
    batch = next(iter(loader)).to(device).to_dict()
    dtype = next(model.parameters()).dtype
    for key, value in batch.items():
        if value.dtype.is_floating_point:
            batch[key] = value.to(dtype=dtype)
    batch["positions"].requires_grad_(True)
    if not bool(torch.all(batch["pbc"]).item()):
        raise RuntimeError("The benchmark must not exercise the molecular fallback")
    return batch


def _call_model(
    model: torch.nn.Module, batch: Dict[str, torch.Tensor]
) -> Dict[str, torch.Tensor]:
    return model(
        batch,
        training=False,
        compute_force=True,
        compute_virials=False,
        compute_stress=False,
    )


def _snapshot(
    model: torch.nn.Module, batch: Dict[str, torch.Tensor]
) -> Dict[str, torch.Tensor]:
    output = _call_model(model, batch)
    torch.cuda.synchronize()
    selected = {
        key: output[key].detach().clone()
        for key in (
            "energy",
            "forces",
            "electrostatic_energy",
            "density_coefficients",
        )
    }
    del output
    return selected


def _time_cuda(fn: Callable[[], Dict[str, torch.Tensor]], repeats: int) -> Timing:
    for _ in range(5):
        output = fn()
        torch.cuda.synchronize()
        del output
    torch.cuda.reset_peak_memory_stats()

    samples = []
    for _ in range(repeats):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        output = fn()
        end.record()
        torch.cuda.synchronize()
        samples.append(start.elapsed_time(end))
        del output
    return Timing(
        median_ms=statistics.median(samples),
        min_ms=min(samples),
        max_ms=max(samples),
        peak_gib=torch.cuda.max_memory_allocated() / 1024**3,
    )


def _max_abs(left: torch.Tensor, right: torch.Tensor) -> float:
    return float(torch.max(torch.abs(left - right)).item())


def _assert_parity(
    eager: Dict[str, torch.Tensor], compiled: Dict[str, torch.Tensor]
) -> Dict[str, float]:
    tolerances = {
        "energy": (1.0e-5, 5.0e-3),
        "forces": (5.0e-4, 5.0e-4),
        "electrostatic_energy": (5.0e-5, 5.0e-5),
        "density_coefficients": (5.0e-4, 5.0e-4),
    }
    differences = {}
    for key, (rtol, atol) in tolerances.items():
        torch.testing.assert_close(compiled[key], eager[key], rtol=rtol, atol=atol)
        differences[key] = _max_abs(compiled[key], eager[key])
    return differences


def _counter_value(group: str, key: str) -> int:
    from torch._dynamo.utils import counters

    return int(counters[group].get(key, 0))


def _print_graph_breaks(num_atoms: int) -> None:
    from torch._dynamo.utils import counters

    graph_breaks = sum(int(value) for value in counters["graph_break"].values())
    reasons = sorted(
        counters["graph_break"].items(), key=lambda item: item[1], reverse=True
    )[:3]
    detail = " | ".join(
        f"{count}x {' '.join(str(reason).split())[:180]}" for reason, count in reasons
    )
    print(
        "POLAR_NVAL_COMPILE_GRAPHS "
        f"N={num_atoms} unique_graphs={_counter_value('stats', 'unique_graphs')} "
        f"calls_captured={_counter_value('stats', 'calls_captured')} "
        f"graph_breaks={graph_breaks} detail={detail}",
        flush=True,
    )


def _run_case(model: torch.nn.Module, side: int, device: torch.device) -> None:
    from torch._dynamo.utils import counters

    _clear_cuda()
    batch = _build_batch(model, side, device)
    num_atoms = int(batch["positions"].shape[0])
    num_edges = int(batch["edge_index"].shape[1])
    repeats = 20 if num_atoms <= 216 else 10
    print(
        "POLAR_NVAL_COMPILE_CASE "
        f"N={num_atoms} edges={num_edges} side={side} repeats={repeats}",
        flush=True,
    )

    eager_batch = _clone_batch(batch)
    eager_reference = _snapshot(model, eager_batch)
    eager_timing = _time_cuda(lambda: _call_model(model, eager_batch), repeats=repeats)
    print(
        "POLAR_NVAL_COMPILE_TIMING "
        f"N={num_atoms} variant=cueq_nval_eager "
        f"median_ms={eager_timing.median_ms:.3f} "
        f"min_ms={eager_timing.min_ms:.3f} max_ms={eager_timing.max_ms:.3f} "
        f"peak_GiB={eager_timing.peak_gib:.3f}",
        flush=True,
    )

    torch.compiler.reset()
    counters.clear()
    configure_autograd_for_compile(allow_autograd=True)
    compiled_model = torch.compile(model, mode="default", fullgraph=False)
    compiled_batch = _clone_batch(batch)
    try:
        torch.cuda.synchronize()
        start = time.perf_counter()
        compiled_reference = _snapshot(compiled_model, compiled_batch)
        first_call_ms = (time.perf_counter() - start) * 1000.0
        differences = _assert_parity(eager_reference, compiled_reference)
        print(
            "POLAR_NVAL_COMPILE_PARITY "
            f"N={num_atoms} energy_max_abs={differences['energy']:.3e} "
            f"forces_max_abs={differences['forces']:.3e} "
            f"electrostatic_max_abs={differences['electrostatic_energy']:.3e} "
            f"density_max_abs={differences['density_coefficients']:.3e}",
            flush=True,
        )
        _print_graph_breaks(num_atoms)

        compiled_timing = _time_cuda(
            lambda: _call_model(compiled_model, compiled_batch), repeats=repeats
        )
        speedup = eager_timing.median_ms / compiled_timing.median_ms
        compile_overhead_ms = max(0.0, first_call_ms - compiled_timing.median_ms)
        saved_ms = eager_timing.median_ms - compiled_timing.median_ms
        break_even = compile_overhead_ms / saved_ms if saved_ms > 0 else float("inf")
        print(
            "POLAR_NVAL_COMPILE_TIMING "
            f"N={num_atoms} variant=cueq_nval_compiled "
            f"median_ms={compiled_timing.median_ms:.3f} "
            f"min_ms={compiled_timing.min_ms:.3f} "
            f"max_ms={compiled_timing.max_ms:.3f} "
            f"peak_GiB={compiled_timing.peak_gib:.3f}",
            flush=True,
        )
        print(
            "POLAR_NVAL_COMPILE_SUMMARY "
            f"N={num_atoms} speedup={speedup:.3f} "
            f"first_call_ms={first_call_ms:.3f} "
            f"compile_overhead_ms={compile_overhead_ms:.3f} "
            f"break_even_calls={break_even:.1f}",
            flush=True,
        )
    except Exception as exc:  # pylint: disable=broad-exception-caught
        detail = " ".join(str(exc).split())[:1500]
        print(
            "POLAR_NVAL_COMPILE_FAILURE "
            f"N={num_atoms} type={type(exc).__name__} detail={detail}",
            flush=True,
        )
    finally:
        del compiled_model
        torch.compiler.reset()
        _clear_cuda()


def main() -> None:
    if not torch.cuda.is_available():
        raise SystemExit("CUDA is required")
    device = torch.device("cuda")
    model = _load_model(device)
    print(
        "POLAR_NVAL_COMPILE_DEVICE "
        f"name={torch.cuda.get_device_name(0)} torch={torch.__version__} "
        f"dtype={next(model.parameters()).dtype} model={MODEL_NAME} "
        f"recursions={model.num_recursion_steps} backend={model.electrostatics_backend}",
        flush=True,
    )
    for side in SUPERCELL_SIDES:
        _run_case(model, side, device)


if __name__ == "__main__":
    main()
