"""Measure whole MD steps separately from isolated graph/model diagnostics.

Use the same converted MACE-MP-0 small checkpoint and reproducible silicon
geometries as the native LAMMPS campaign. Compilation is outside timings.
"""

import argparse
import json
import os
import re
import statistics
import subprocess
import time
from pathlib import Path

import numpy as np
import torch
from ase.build import bulk
from tests.integrations.gromacs._harness import prepare, command
from tests.integrations.libmace._harness import geometry_text


def samples(call, *, count=10):
    for _ in range(3):
        call()
    values = []
    for _ in range(count):
        torch.cuda.synchronize()
        start = time.perf_counter()
        call()
        torch.cuda.synchronize()
        values.append(1000 * (time.perf_counter() - start))
    return {"median_ms": statistics.median(values), "samples_ms": values}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--repeat", type=int, default=4)
    parser.add_argument(
        "--mode", choices=["gromacs", "torchsim", "openmm"], required=True
    )
    parser.add_argument("--native", type=Path)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(1)
    torch.set_default_dtype(torch.float32)
    atoms = bulk("Si", "diamond", a=5.431, cubic=True).repeat((args.repeat,) * 3)
    atoms.rattle(stdev=0.025, seed=4811)
    atoms.wrap()
    record = {
        "atoms": len(atoms),
        "mode": args.mode,
        "gpu": torch.cuda.get_device_name(),
        "torch": torch.__version__,
        "checkpoint": str(args.checkpoint),
        "precision": "float32",
        "backend": "cueq",
    }
    if args.mode == "torchsim":
        from mace_torch.deploy.torchsim import MaceTorchSimModel
        from torch_sim.io import atoms_to_state
        from torch_sim.integrators.nve import nve_init, nve_step
        from mace_core.config.precision import PrecisionConfig

        model = MaceTorchSimModel(
            args.checkpoint,
            device="cuda",
            backend="cueq",
            precision=PrecisionConfig.float32(),
            compile_mode="default",
            compute_stress=False,
        )
        state = atoms_to_state(atoms, device=torch.device("cuda"), dtype=torch.float32)
        state.rng = 9043
        md = nve_init(state, model, kT=0.02585)

        def step():
            nonlocal md
            md = nve_step(md, model, dt=0.001)

        record["md"] = samples(step)
        record["graph_with_host_transfers"] = samples(lambda: model._graph(state))
        graph, _ = model._graph(state)
        record["model_only"] = samples(
            lambda: model.deployed.compute(graph, compute=("forces",))
        )
        reference = MaceTorchSimModel(
            args.checkpoint,
            device="cuda",
            backend="cueq",
            precision=PrecisionConfig.float32(),
            compute_stress=False,
        )
        expected = reference(md)
        actual = model(md)
        from tests.golden.harness import FP32

        errors = {}
        for key in ("energy", "forces"):
            np.testing.assert_allclose(
                actual[key].cpu(), expected[key].cpu(), atol=FP32.atol, rtol=FP32.rtol
            )
            errors[key] = float((actual[key] - expected[key]).abs().max())
        record["compiled_vs_eager_max_error"] = errors
    elif args.mode == "gromacs":
        if not args.native:
            parser.error("--native is required")
        record["ranks"] = {}
        for ranks in (1, 2):
            directory = args.output / f"ranks-{ranks}"
            prepare(directory, atoms, args.native, steps=60)
            mdp = directory / "run.mdp"
            mdp.write_text(
                mdp.read_text()
                .replace("comm-mode = none", "comm-mode = Linear")
                .replace("nstxout = 1", "nstxout = 0")
                .replace("nstvout = 1", "nstvout = 0")
                .replace("nstfout = 1", "nstfout = 0")
                .replace("nstenergy = 1", "nstenergy = 60")
                .replace("nstcalcenergy = 1", "nstcalcenergy = 1")
            )
            command(
                directory,
                [
                    "grompp",
                    "-f",
                    "run.mdp",
                    "-c",
                    "input.g96",
                    "-p",
                    "system.top",
                    "-o",
                    "run.tpr",
                ],
            )
            timings = []
            for trial in range(3):
                argv = [
                    "mdrun",
                    "-s",
                    "run.tpr",
                    "-deffnm",
                    f"measure-{trial}",
                    "-nb",
                    "cpu",
                    "-pin",
                    "off",
                    "-resetstep",
                    "10",
                ]
                if ranks > 1:
                    argv += ["-dd", "2", "1", "1"]
                result = command(directory, argv, ranks=ranks)
                log = (directory / f"measure-{trial}.log").read_text()
                match = re.search(r"Performance:\s+[0-9.]+\s+[0-9.]+\s+([0-9.]+)", log)
                assert match, log[-3000:]
                # GROMACS reports ms/step directly. Inverting the rounded
                # ns/day column loses precision for these short timesteps.
                timings.append(float(match[1]))
            record["ranks"][ranks] = {
                "step_ms": timings,
                "median_ms": statistics.median(timings),
            }
            if os.environ.get("MACE_PROFILE"):
                profiles = []
                for line in result.stderr.splitlines():
                    if line.startswith("MACE_PROFILE "):
                        profiles.append([float(x) for x in line.split()[2:]])
                assert len(profiles) == 61, "expected one native profile per force call"
                names = (
                    "neighbors",
                    "prepare_edges",
                    "h2d",
                    "model_synchronized",
                    "d2h",
                    "force_assembly",
                )
                record["ranks"][ranks]["diagnostic_mean_ms"] = dict(
                    zip(names, np.mean(profiles[10:], axis=0).tolist(), strict=True)
                )
    elif args.mode == "openmm":
        if not args.native:
            parser.error("--native is required")
        env = os.environ.copy()
        record["runs"] = []
        for _ in range(3):
            result = subprocess.run(
                [
                    env["MACE_OPENMM_DRIVER"],
                    str(args.native),
                    "CUDA",
                    "60",
                    "benchmark",
                ],
                input=geometry_text([atoms]),
                text=True,
                capture_output=True,
                env=env,
                check=True,
                timeout=240,
            )
            elapsed = [
                float(x) for x in re.findall(r"BENCHMARK_MS (\S+)", result.stderr)
            ]
            assert len(elapsed) == 1 and elapsed[0] > 0, result.stderr[-3000:]
            record["runs"].append({"step_ms": elapsed, "stderr": result.stderr[-1000:]})
    path = args.output / "result.json"
    path.write_text(json.dumps(record, indent=2) + "\n")
    print(path.read_text(), flush=True)


if __name__ == "__main__":
    main()
