"""Polar medium / cuEq: fixed-cell NVE across ASE and three native engines.

Each mode runs in a fresh process. Timings exclude export, loading and warmup;
include neighbour construction, transfers, integration and force evaluation.
The native energy/force/virial gate precedes each timed engine run.
"""

import argparse
import gc
import hashlib
import json
import os
import re
import socket
import statistics
import subprocess
import time
from importlib.metadata import version
from pathlib import Path

import mace_core
import mace_torch
import numpy as np
import torch
from ase import units
from ase.io import read, write
from ase.md.verlet import VelocityVerlet
from ase.neighborlist import neighbor_list
from mace_core.config.precision import PrecisionConfig
from mace_torch.calculators import MACECalculator, NativeMACECalculator
from mace_torch.deploy.bundle import export_lammps
from mace_torch.deploy.loader import load_deployed

from tests.golden.harness import FP32
from tests.integrations.gromacs import _harness as gmx
from tests.integrations.libmace._harness import geometry_text, run_driver, unpack

REPEATS = {
    24: (1, 1, 1),
    96: (2, 2, 1),
    192: (2, 2, 2),
    384: (4, 2, 2),
    768: (4, 4, 2),
    1536: (4, 4, 4),
    3072: (8, 4, 4),
}


def provenance(checkpoint):
    digest = hashlib.sha256()
    for module in (mace_core, mace_torch):
        root = Path(module.__file__).parent
        for path in sorted(root.rglob("*.py")):
            digest.update(str(path.relative_to(root)).encode())
            digest.update(path.read_bytes())
    native_digest = hashlib.sha256()
    native_root = Path(__file__).resolve().parents[3] / "packages/mace-torch"
    for directory in ("native", "libmace", "lammps", "openmm", "gromacs"):
        for path in sorted((native_root / directory).rglob("*")):
            if path.is_file() and path.suffix in {
                ".cpp",
                ".cu",
                ".h",
                ".py",
                ".cmake",
                ".txt",
                ".patch",
            }:
                native_digest.update(str(path.relative_to(native_root)).encode())
                native_digest.update(path.read_bytes())
    return {
        "python_source_sha256": digest.hexdigest(),
        "native_source_sha256": native_digest.hexdigest(),
        "checkpoint_sha256": {
            suffix: hashlib.sha256(
                checkpoint.with_suffix(suffix).read_bytes()
            ).hexdigest()
            for suffix in (".json", ".safetensors")
        },
        "cuequivariance_torch": version("cuequivariance-torch"),
        "compute_capability": torch.cuda.get_device_capability(),
    }


def calculator(checkpoint, backend="cueq"):
    deployed = load_deployed(
        checkpoint,
        device="cuda",
        backend=backend,
        precision=PrecisionConfig(model="float32"),
    )
    return MACECalculator(models=deployed, device="cuda")


def check(values, frames, checkpoint):
    calc = calculator(checkpoint)
    errors = {}
    for actual, atoms in zip(values, frames, strict=True):
        calc.calculate(atoms)
        expected = {
            **calc.results,
            "virial": -atoms.get_volume() * calc.results["stress"][[0, 1, 2, 5, 4, 3]],
        }
        for key, result in actual.items():
            np.testing.assert_allclose(
                result, expected[key], atol=FP32.atol, rtol=FP32.rtol, err_msg=key
            )
            errors[key] = max(
                errors.get(key, 0), float(np.max(np.abs(result - expected[key])))
            )
    del calc
    gc.collect()
    torch.cuda.empty_cache()
    return errors


def execute(argv, directory, *, stdin=None, label="run"):
    result = subprocess.run(
        list(map(str, argv)),
        cwd=directory,
        input=stdin,
        text=True,
        capture_output=True,
        timeout=600,
    )
    (directory / f"{label}.stdout").write_text(result.stdout)
    (directory / f"{label}.stderr").write_text(result.stderr)
    if result.returncode:
        raise RuntimeError(
            f"{argv[0]} failed ({result.returncode}): "
            f"{result.stderr[-5000:]} {result.stdout[-1500:]}"
        )
    return result


def lammps(directory, atoms, package, *, steps, benchmark=False):
    directory.mkdir(parents=True, exist_ok=True)
    write(
        directory / "water.data",
        atoms,
        format="lammps-data",
        atom_style="atomic",
        specorder=["H", "O"],
        masses=True,
    )
    preamble = f"""units metal
atom_style atomic
boundary p p p
read_data water.data
plugin load {os.environ["MACE_NATIVE_PLUGIN"]}
pair_style mace/native/kk {package}
pair_coeff * * H O
neighbor 0.3 bin
neigh_modify every 1 delay 0 check no
compute p all pressure NULL pair
thermo_style custom step pe vol c_p[1] c_p[2] c_p[3] c_p[4] c_p[5] c_p[6]
thermo_modify format float %.17g
fix integrate all nve
timestep 0.0001
"""
    if benchmark:
        body = f"thermo {steps}\nrun 10\nrun {steps} pre no\n"
    else:
        body = (
            "thermo 1\n"
            "dump states all custom 1 frames.dump id type x y z fx fy fz\n"
            "dump_modify states sort id format float %.17g\n"
            f"run {steps}\n"
        )
    (directory / "run.in").write_text(preamble + body)
    result = execute(
        [
            os.environ["MACE_NATIVE_LMP"],
            "-k",
            "on",
            "g",
            "1",
            "-sf",
            "kk",
            "-pk",
            "kokkos",
            "neigh",
            "full",
            "newton",
            "off",
            "-in",
            "run.in",
        ],
        directory,
    )
    if "kokkos_device" not in result.stdout:
        raise RuntimeError("LAMMPS did not report a Kokkos device neighbour list")
    if benchmark:
        match = re.findall(
            r"Loop time of ([0-9.eE+-]+) on \d+ procs for (\d+) steps", result.stdout
        )
        assert int(match[-1][1]) == steps
        return 1000 * float(match[-1][0]) / steps
    thermo = []
    for line in result.stdout.splitlines():
        tokens = line.split()
        if len(tokens) == 9 and tokens[0].isdigit():
            thermo.append(list(map(float, tokens)))
    frames, values = [], []
    for item, row in zip(
        (directory / "frames.dump").read_text().split("ITEM: TIMESTEP\n")[1:],
        thermo,
        strict=True,
    ):
        body = item.split("ITEM: ATOMS id type x y z fx fy fz\n")[1]
        data = np.loadtxt(body.splitlines(), ndmin=2)
        assert np.array_equal(data[:, 0], np.arange(1, len(atoms) + 1))
        frame = atoms.copy()
        frame.positions = data[:, 2:5]
        frames.append(frame)
        values.append(
            {
                "energy": row[1],
                "forces": data[:, 5:8],
                "virial": np.asarray(row[3:]) * row[2] * units.bar,
            }
        )
    return values, frames


def measure(args, atoms, package, directory):
    result = {}
    timings = []
    if args.engine == "ase":
        # Establish cuEq parity independently of the native exporter.
        expected = calculator(args.checkpoint, backend="reference")
        expected.calculate(atoms)
        baseline = {
            key: np.array(expected.results[key])
            for key in ("energy", "forces", "stress")
        }
        del expected
        gc.collect()
        torch.cuda.empty_cache()
        result["backend_max_errors_vs_reference"] = check(
            [baseline], [atoms], args.checkpoint
        )
        result["max_errors_vs_reference"] = {}
        for _ in range(args.trials):
            frame = atoms.copy()
            frame.calc = NativeMACECalculator(
                package, neighbor_library=os.environ["MACE_CUDA_NEIGHBORS_LIBRARY"]
            )
            frame.calc.calculate(frame)
            for key, value in baseline.items():
                np.testing.assert_allclose(
                    frame.calc.results[key], value, atol=FP32.atol, rtol=FP32.rtol
                )
                result["max_errors_vs_reference"][key] = max(
                    result["max_errors_vs_reference"].get(key, 0),
                    float(np.max(np.abs(frame.calc.results[key] - value))),
                )
            result["neighbor_backend"] = "native_cuda"
            result["neighbor_device"] = "cuda"
            md = VelocityVerlet(frame, timestep=0.1 * units.fs, logfile=None)
            md.run(10)
            torch.cuda.synchronize()
            start = time.perf_counter()
            md.run(args.steps)
            torch.cuda.synchronize()
            timings.append(1000 * (time.perf_counter() - start) / args.steps)
            del md, frame
            gc.collect()
            torch.cuda.empty_cache()
    else:
        native = run_driver(package, [atoms], directory / "native-gate")
        native_values = unpack(native[0][0], atoms)
        result["native_max_errors"] = check([native_values], [atoms], args.checkpoint)
        if args.engine == "lammps":
            values, frames = lammps(directory / "parity", atoms, package, steps=3)
            result["max_errors"] = check(values, frames, args.checkpoint)
            for trial in range(args.trials):
                timings.append(
                    lammps(
                        directory / f"trial-{trial}",
                        atoms,
                        package,
                        steps=args.steps,
                        benchmark=True,
                    )
                )
        elif args.engine == "openmm":
            argv = [os.environ["MACE_OPENMM_DRIVER"], package, "CUDA"]
            observed = execute(
                [*argv, "3"], directory, stdin=geometry_text([atoms]), label="parity"
            )
            assert "MACE_OPENMM_DEVICE_IO cuda" in observed.stderr
            values, frames = [], []
            for line in observed.stdout.splitlines():
                if line.startswith("RESULT "):
                    row = np.fromstring(line[7:], sep=" ")
                    frame = atoms.copy()
                    frame.positions = row[1 : 1 + 3 * len(atoms)].reshape(-1, 3)
                    frames.append(frame)
                    values.append(
                        {
                            "energy": row[0],
                            "forces": row[1 + 3 * len(atoms) :].reshape(-1, 3),
                        }
                    )
            assert len(frames) == 4
            result["max_errors"] = check(values, frames, args.checkpoint)
            for trial in range(args.trials):
                observed = execute(
                    [*argv, args.steps, "benchmark"],
                    directory,
                    stdin=geometry_text([atoms]),
                    label=f"trial-{trial}",
                )
                timings.append(
                    float(re.search(r"BENCHMARK_MS ([0-9.eE+-]+)", observed.stderr)[1])
                )
        elif args.engine == "gromacs":
            parity = directory / "parity"
            gmx.prepare(parity, atoms, package, steps=3)
            observed = gmx.run(parity)
            values, frames = [], []
            for i, positions in enumerate(observed["positions"]):
                frame = atoms.copy()
                frame.positions = positions
                frame.set_cell(observed["cell"][i])
                frames.append(frame)
                values.append(
                    {key: observed[key][i] for key in ("energy", "forces", "virial")}
                )
            result["max_errors"] = check(values, frames, args.checkpoint)
            for trial in range(args.trials):
                target = directory / f"trial-{trial}"
                gmx.prepare(target, atoms, package, steps=args.steps + 10)
                mdp = target / "run.mdp"
                mdp.write_text(
                    mdp.read_text()
                    .replace("comm-mode = none", "comm-mode = Linear")
                    .replace("nstxout = 1", "nstxout = 0")
                    .replace("nstvout = 1", "nstvout = 0")
                    .replace("nstfout = 1", "nstfout = 0")
                    .replace("nstenergy = 1", f"nstenergy = {args.steps + 10}")
                )
                gmx.command(
                    target,
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
                gmx.command(
                    target,
                    [
                        "mdrun",
                        "-s",
                        "run.tpr",
                        "-deffnm",
                        "measure",
                        "-nb",
                        "cpu",
                        "-pin",
                        "off",
                        "-resetstep",
                        "10",
                    ],
                )
                match = re.search(
                    r"Performance:\s+[0-9.]+\s+[0-9.]+\s+([0-9.]+)",
                    (target / "measure.log").read_text(),
                )
                assert match
                timings.append(float(match[1]))
    result.update(samples_ms=timings, median_ms=statistics.median(timings))
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("base_water", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--atoms", type=int, required=True, choices=REPEATS)
    parser.add_argument(
        "--engine",
        choices=["export", "ase", "lammps", "openmm", "gromacs"],
        required=True,
    )
    parser.add_argument("--steps", type=int, default=60)
    parser.add_argument("--trials", type=int, default=3)
    args = parser.parse_args()
    torch.set_num_threads(1)
    torch.set_default_dtype(torch.float32)
    directory = args.output.resolve() / str(args.atoms)
    directory.mkdir(parents=True, exist_ok=True)
    atoms = read(args.base_water).repeat(REPEATS[args.atoms])
    assert len(atoms) == args.atoms
    atoms.wrap()
    atoms.set_momenta(np.zeros((len(atoms), 3)))
    structure = directory / "water.xyz"
    if not structure.exists():
        write(structure, atoms)
    package = directory / "native" / "native.pt2"
    if args.engine == "export":
        if package.exists():
            return
        deployed = load_deployed(args.checkpoint)
        live_edges = len(neighbor_list("i", atoms, deployed.r_max))
        edges = 2 ** int(np.ceil(np.log2(live_edges * 1.25)))
        del deployed
        print(
            f"Exporting {args.atoms} atoms, {live_edges} edges, capacity {edges}",
            flush=True,
        )
        export_lammps(
            args.checkpoint,
            package.parent,
            halo_exchange=False,
            dtype="float32",
            backend="cueq",
            device="cuda",
            native_capacity=(len(atoms), edges),
            structure=structure,
        )
        return
    directory = directory / args.engine
    directory.mkdir(exist_ok=True)
    record = {
        **provenance(args.checkpoint),
        "atoms": len(atoms),
        "engine": args.engine,
        "host": socket.gethostname(),
        "gpu": torch.cuda.get_device_name(),
        "torch": torch.__version__,
        "backend": "cueq",
        "execution": "native_aot",
        "native_package_sha256": hashlib.sha256(package.read_bytes()).hexdigest(),
        "neighbor_backend": "kokkos" if args.engine == "lammps" else "native_cuda",
        "neighbor_device": "cuda",
        "native_host_staging": args.engine in {"ase", "gromacs"},
        "neighbor_rebuild": "every_step",
        "model_precision": "float32",
        "integrator_precision": "float64",
        "timestep_fs": 0.1,
        "warmup_steps": 10,
        "measured_steps": args.steps,
        "trials": args.trials,
        "initial_velocities": "zero",
        "cell_angstrom": atoms.cell.array.tolist(),
    }
    record.update(measure(args, atoms, package, directory))
    (directory / "result.json").write_text(json.dumps(record, indent=2) + "\n")
    print(json.dumps(record), flush=True)


if __name__ == "__main__":
    main()
