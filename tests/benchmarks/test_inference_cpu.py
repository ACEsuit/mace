"""Inference baselines recorded while legacy is still the live stack.

"The rewrite is not slower" is only falsifiable against an *old* number, so
these cases exist to be measured now and frozen, not to gate anything. They
publish through the nightly ``benchmarks`` job's ``--benchmark-json``
artifact; the downstream comparisons read that history, including after
legacy retires -- the new-architecture benchmark harness compares against it,
and the deployment-path retirement trigger reads two fp32/NVIDIA cases from
it. That is why every case is a fixed structure at a fixed size with its
dtype, device, torch version and backend recorded next to the timing: a
one-off number with an unrecorded denominator is not a baseline.

Two size regimes, deliberately, because the downstream comparison is judged
in both:

* ``subdomain`` (216 and 512 atoms) -- the per-rank size of a
  domain-decomposed MD run, where host-side cost (dispatch, allocation, the
  Python between the kernels) is a material fraction of the step. This is
  where a rewrite that is faster per kernel can still be slower per step.
* ``kernel`` (1728 atoms) -- large enough that the tensor-product and
  symmetric-contraction kernels dominate and host overhead amortises away.

The regimes are usually described in terms of a GPU step being a few
milliseconds, and on a CPU runner that is simply not the wall-clock: the
anchor's fp32 forward measured 31 / 60 / 162 ms at the three sizes on an
8-core arm64 laptop. What survives the move to CPU is the thing the split is
actually for -- the *shape* of the scaling -- and it shows in the derived
throughput rather than the latency. Same run, atom-steps per second: 6897 at
216 atoms, 8588 at 512, 10658 at 1728. Per-atom cost falls monotonically with
size, and the small case runs at 65% of the large one's rate: about a third of
its per-atom cost is work that does not scale with the system, which is
exactly the overhead a faster-per-kernel rewrite can still lose to. Note what
this does *not* show -- the curve has not flattened by 1728 atoms, so
`kernel1728` is the most kernel-dominated case here rather than a saturated
one, and reading it as an asymptote would overstate what these three points
support.

Absolute latencies are a property of the runner and are not comparable across
machines, which is why every case records its own metadata. Timings taken
while other work shares the host differ substantially -- a contended run of
this same set read 4326 / 5590 / 5936 -- so trend comparisons must come from
the nightly's own history, on its own runner, not from a number pasted out of
a docstring.

REACHABILITY -- the CI decision this file records
-------------------------------------------------
``tests/benchmarks`` is auto-marked ``benchmark`` + ``slow``
(``tests/conftest.py``), and exactly one CI job runs it: the nightly
``benchmarks`` job, on a CPU runner, with ``numprocs: "0"``.

The GPU pipeline (``.github/gitlab/ci.yml``) appends ``and not benchmark`` to
its marker expression, and that stays. It runs ``-n 2`` under a two-hour cap
on a *contended* shared runner, so a timing taken inside it would be noise
wearing a number, and it is a correctness gate that must not grow a
perf-shaped flake.

The consequence is now explicit instead of accidental: the ``gpu``-marked
cases below -- and the pre-existing ``test_inference`` in
``test_benchmark.py``, all sixteen of whose parametrizations are ``gpu`` *and*
``network`` -- run in **zero** CI jobs. That is why the nightly artifact was
reproducibly a 0-byte file before this file existed: the job was green, and
published nothing. The cases here are what fill it, and the nightly job now
asserts the artifact's *contents* rather than its existence, because
pytest-benchmark writes the json path either way. The GPU cases stay as the
reproduction recipe for the frozen fp32/NVIDIA numbers, run by hand on a
quiet GPU host:

    MACE_CI_ALLOW_NETWORK=1 pytest tests/benchmarks -m benchmark \\
        -p no:randomly --benchmark-json=benchmark.json

``tests/unit/test_ci_gates.py`` is what keeps the CPU half from regressing
into the same hole: it fails, in a PR-gating job, if any declared size stops
having a case a CPU-only nightly can run.
"""

from __future__ import annotations

import platform
from typing import Dict, Iterator, List, Tuple

import pytest
import torch
from ase import build

from mace import data as mace_data
from mace.tools import AtomicNumberTable, torch_geometric, torch_tools
from tests.golden import harness

# ---------------------------------------------------------------------------
# The cases, kept as module-level data so the reachability guard in
# tests/unit/test_ci_gates.py can reason about them without running them.
# ---------------------------------------------------------------------------

#: regime -> (cubic-diamond repeat factor, resulting atom count). Diamond
#: carbon at the experimental lattice constant: element 6 is the one species
#: every model here shares (the committed tiny anchor knows H/C/O only), and
#: a homogeneous bulk keeps the edge count a pure function of the size.
SYSTEM_SIZES: Dict[str, Tuple[int, int]] = {
    "subdomain216": (3, 216),
    "subdomain512": (4, 512),
    "kernel1728": (6, 1728),
}

#: fp64 is MACE's default and what the "not slower" comparison is judged in;
#: fp32 is what the deployment-path retirement trigger freezes.
DTYPES: Tuple[str, ...] = ("float32", "float64")

#: (model, regime). The tiny anchor is committed in-repo, so it is measurable
#: on any runner with no network at all, and it covers every size. MP-small is
#: the production-shaped counterpart, has to be downloaded (hence the
#: `network` mark), and deliberately stops at 512 atoms. Its cutoff is 6.0 A
#: against the anchor's 3.5, which is 158 edges per atom rather than 28: the
#: 512-atom case is already 80896 edges, measured at 5.25 GB peak RSS and
#: ~7 s per fp64 forward locally, and a 1728-atom cell would be roughly three
#: times that in both. That is more than a shared nightly runner should be
#: asked for by a job that gates nothing. The kernel-bound regime is covered
#: by the anchor, which is the case the rewrite comparison is anchored on
#: anyway. The whole CPU set (10 cases) measured ~160 s locally.
CASES: List[Tuple[str, str]] = [
    ("anchor", "subdomain216"),
    ("anchor", "subdomain512"),
    ("anchor", "kernel1728"),
    ("mp_small", "subdomain216"),
    ("mp_small", "subdomain512"),
]

NETWORK_MODELS: Tuple[str, ...] = ("mp_small",)


def case_params() -> List:
    return [
        pytest.param(
            model,
            regime,
            marks=[pytest.mark.network] if model in NETWORK_MODELS else [],
            id=f"{model}-{regime}",
        )
        for model, regime in CASES
    ]


def backend_params() -> List:
    # e3nn is the only backend that exists without a GPU. cueq and oeq carry
    # both the hardware and the library capability, so the skip reason names
    # whichever is actually missing rather than blaming the GPU for a missing
    # wheel.
    return [
        pytest.param("e3nn", id="e3nn"),
        pytest.param("cueq", marks=[pytest.mark.gpu, pytest.mark.cueq], id="cueq"),
        pytest.param("oeq", marks=[pytest.mark.gpu, pytest.mark.oeq], id="oeq"),
    ]


def _load_model(name: str, dtype: str, device: str) -> torch.nn.Module:
    if name == "anchor":
        torch_dtype = torch.float32 if dtype == "float32" else torch.float64
        model = torch.load(
            harness.MODELS_DIR / "tiny_scaleshift.model",
            weights_only=False,
            map_location="cpu",
        )
        return model.to(torch_dtype).to(device)
    if name == "mp_small":
        # Imported here so a CPU-only collection never touches the download
        # machinery at import time.
        from mace.calculators.foundations_models import mace_mp  # noqa: PLC0415

        calc = mace_mp(model="small", default_dtype=dtype, device=device)
        return calc.models[0].to(device)
    raise AssertionError(f"unknown benchmark model {name!r}")


def _apply_backend(model: torch.nn.Module, backend: str, device: str):
    if backend == "e3nn":
        return model
    if backend == "cueq":
        from mace.cli.convert_e3nn_cueq import run as to_cueq  # noqa: PLC0415

        return to_cueq(model, device=device, return_model=True).to(device)
    if backend == "oeq":
        from mace.cli.convert_e3nn_oeq import run as to_oeq  # noqa: PLC0415

        return to_oeq(model, device=device, return_model=True).to(device)
    raise AssertionError(f"unknown backend {backend!r}")


def _batch(model: torch.nn.Module, repeat: int, dtype: str, device: str) -> dict:
    """One diamond supercell as the dict ``model.forward`` eats.

    Built inside a ``default_dtype`` scope for the reason spelled out in
    ``tests/golden/test_tiny_anchors.py``: ``AtomicData`` reads the
    process-wide default dtype, which is float32 under pytest, and building in
    float32 then casting up rounds the positions first. Here that would not
    change a verdict -- nothing is compared -- but it would change the edge
    count at a cutoff boundary, and two nights whose structures differ are not
    a baseline.
    """
    z_table = AtomicNumberTable([int(z) for z in model.atomic_numbers])
    atoms = build.bulk("C", "diamond", a=3.567, cubic=True).repeat((repeat,) * 3)
    with torch_tools.default_dtype(dtype):
        graph = mace_data.AtomicData.from_config(
            mace_data.config_from_atoms(atoms),
            z_table=z_table,
            cutoff=float(model.r_max),
        )
        loader = torch_geometric.dataloader.DataLoader(
            dataset=[graph], batch_size=1, shuffle=False, drop_last=False
        )
        batch = next(iter(loader))
    return batch.to(device).to_dict()


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


@pytest.mark.benchmark(warmup=True, warmup_iterations=2, min_rounds=5)
@pytest.mark.parametrize(("model_name", "regime"), case_params())
@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("backend", backend_params())
def test_inference_latency(
    benchmark, backend: str, dtype: str, model_name: str, regime: str
) -> None:
    """One energy+forces forward at a fixed size, as a frozen reference."""
    device = "cpu" if backend == "e3nn" else "cuda"
    repeat, expected_atoms = SYSTEM_SIZES[regime]

    with torch_tools.default_dtype(dtype):
        model = _apply_backend(_load_model(model_name, dtype, device), backend, device)
        batch = _batch(model, repeat, dtype, device)
        num_atoms = int(batch["positions"].shape[0])
        # The regime names carry the atom count, and a silent change to the
        # structure would make two nights' numbers incomparable while looking
        # like a performance result.
        assert num_atoms == expected_atoms

        benchmark.extra_info.update(
            model=model_name,
            backend=backend,
            regime=regime,
            num_atoms=num_atoms,
            num_edges=int(batch["edge_index"].shape[1]),
            dtype=dtype,
            device=device,
            device_name=_device_label(device),
            torch_version=torch.__version__,
            r_max=float(model.r_max),
        )

        def func():
            if device == "cuda":
                torch.cuda.synchronize()
            model(batch, training=False, compute_force=True)
            if device == "cuda":
                torch.cuda.synchronize()

        benchmark(func)
        _record_throughput(benchmark, num_atoms)


def iter_case_marks() -> Iterator[Tuple[str, frozenset]]:
    """(regime, marker names) for every case this module declares.

    Exposed for ``tests/unit/test_ci_gates.py``, which needs to know which
    cases a CPU-only nightly can run without importing pytest's collection
    machinery or running a single timing.
    """
    module_marks = {mark.name for mark in globals().get("pytestmark", [])}
    for backend in backend_params():
        backend_marks = {mark.name for mark in backend.marks}
        for case in case_params():
            case_marks = {mark.name for mark in case.marks}
            _model, regime = case.values
            yield regime, frozenset(module_marks | backend_marks | case_marks)
