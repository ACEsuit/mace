"""Steady-state inference over the certified roster with prebuilt neighbour lists."""

from __future__ import annotations

import argparse
import json
import os
import statistics
from pathlib import Path
from time import perf_counter

import jax
from ase.build import molecule
from mace_jax.deploy.neutral_io import load_model
from mace_jax.graph import graph_from_atoms
from mace_jax.inference import JITPipeline
from mace_jax.models.registry import SUPPORTED_MODELS


def elapsed(evaluate, graph):
    start = perf_counter()
    out = evaluate(graph)
    jax.block_until_ready(
        (out.total_energy, out.node_energies, out.forces, out.stress, out.virials)
    )
    return 1000 * (perf_counter() - start)


def benchmark(reference, dtype, steps):
    apply, params = load_model(reference, dtype=dtype)
    # Rebuilt lists for three small configurations, largest first to establish capacity.
    graphs = [
        graph_from_atoms(
            molecule(name),
            elements=apply.elements,
            cutoff=apply.config.model.r_max,
            dtype=dtype,
            head=next(iter(apply.config.data.heads)),
        )
        for name in ("CH4", "H2O", "H2")
    ]

    def eager(graph):
        return apply(params, graph)

    compiled = JITPipeline(apply, params)
    warmup = {
        name: [elapsed(evaluate, g) for g in graphs]
        for name, evaluate in (("eager", eager), ("jit", compiled))
    }
    before = (compiled.energy_traces, compiled.output_traces)
    eager_ms, jit_ms = [], []
    for _ in range(steps):
        for graph in graphs:
            eager_ms.append(elapsed(eager, graph))
            jit_ms.append(elapsed(compiled, graph))
    after = (compiled.energy_traces, compiled.output_traces)
    if before != (1, 1) or after != before:
        raise RuntimeError(f"JAX retraced: warmup={before}, steady state={after}")
    return {
        "structures": ["CH4", "H2O", "H2"],
        "calls_per_path": len(jit_ms),
        "warmup_ms": warmup,
        "eager_median_ms": statistics.median(eager_ms),
        "jit_median_ms": statistics.median(jit_ms),
        "energy_traces": after[0],
        "output_traces": after[1],
        "steady_state_retraces": 0,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--dtype", choices=("float32", "float64"), default="float64")
    parser.add_argument(
        "--artifacts",
        type=Path,
        default=os.environ.get("MACE_JAX_MODEL_ARTIFACTS"),
        help="JSON of registry IDs to local checkpoints until registry publication",
    )
    parser.add_argument(
        "--steps", type=int, default=5, help="Measured passes over the three structures"
    )
    args = parser.parse_args()
    if args.steps < 1:
        parser.error("--steps must be positive")
    references = json.loads(args.artifacts.read_text()) if args.artifacts else {}
    device = jax.devices("gpu" if args.device == "cuda" else args.device)[0]
    with jax.default_device(device):
        for model_id in SUPPORTED_MODELS:
            row = benchmark(references.get(model_id, model_id), args.dtype, args.steps)
            print(
                json.dumps(
                    dict(
                        model=model_id,
                        dtype=args.dtype,
                        device=device.device_kind,
                        jax=jax.__version__,
                        **row,
                    )
                ),
                flush=True,
            )
            jax.clear_caches()


if __name__ == "__main__":
    main()
