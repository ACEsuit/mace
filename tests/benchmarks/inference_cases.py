import platform
import torch

SYSTEM_SIZES = {"subdomain216": (3, 216), "subdomain512": (4, 512), "kernel1728": (6, 1728)}
DTYPES = ("float32", "float64")

def _device_label(device: str) -> str:
    if device == "cuda":
        return torch.cuda.get_device_name()
    # platform.processor() is empty on many Linux builds -- which is exactly
    # where the nightly runs. pytest-benchmark's own machine_info carries the
    # full detail; this is the human-readable label next to the number.
    return platform.processor() or platform.machine() or "cpu"

def _record_throughput(benchmark, num_atoms: int) -> None:
    """Latency comes from pytest-benchmark; throughput has to be derived.

    Guarded rather than asserted: ``benchmark.stats`` is only populated once a
    timing actually ran, and is absent under ``--benchmark-disable``, which is
    how a plain correctness run of this directory would execute the body.
    """
    stats = getattr(getattr(benchmark, "stats", None), "stats", None)
    median = getattr(stats, "median", None)
    if median:
        benchmark.extra_info["median_seconds"] = median
        benchmark.extra_info["atom_steps_per_second"] = num_atoms / median
