"""Validate the measured native packages against the cached frozen MP-0 small.

Also measure MPI Allreduce elapsed time by rank. Non-root waiting includes the
root's model evaluation, so these times are not isolated wire bandwidth.
"""

import argparse
import hashlib
import json
import os
import re
import subprocess
from pathlib import Path

import numpy as np
from ase.build import bulk
from tests.golden.harness import FP32
from tests.integrations.libmace._harness import run_driver, unpack
from tests.integrations.gromacs._harness import prepare, command

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("output", type=Path)
parser.add_argument(
    "--case",
    action="append",
    nargs=2,
    required=True,
    metavar=("REPEAT", "NATIVE_PACKAGE"),
)
args = parser.parse_args()
args.output.mkdir(parents=True, exist_ok=True)
source = Path.home() / ".cache/mace/20231210mace128L0_energy_epoch249model"
assert (
    hashlib.sha256(source.read_bytes()).hexdigest()
    == "2ddb079cee0e131eaaf6912ba581b394551ead283e95c99cfe78c605d10b5736"
)
from mace.calculators import MACECalculator

oracle = MACECalculator(model_paths=source, device="cpu", default_dtype="float64")
rows = []
for repeats, native in args.case:
    repeat = int(repeats)
    atoms = bulk("Si", "diamond", a=5.431, cubic=True).repeat((repeat,) * 3)
    atoms.rattle(stdev=0.025, seed=4811)
    atoms.wrap()
    package = Path(native)
    directory = args.output / str(len(atoms))
    directory.mkdir(exist_ok=True)
    result = unpack(run_driver(package, [atoms], directory)[0][0], atoms)
    oracle.calculate(atoms)
    errors = {}
    for key in ("energy", "energies", "forces"):
        actual = result[key]
        expected = oracle.results[key]
        np.testing.assert_allclose(actual, expected, atol=FP32.atol, rtol=FP32.rtol)
        errors[key] = float(np.max(np.abs(actual - expected)))
    record = {"atoms": len(atoms), "native_vs_legacy_max_errors": errors}
    prepare(directory, atoms, package, steps=30)
    mdp = directory / "run.mdp"
    mdp.write_text(mdp.read_text().replace("comm-mode = none", "comm-mode = Linear"))
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
    os.environ["MACE_PROFILE_ATOMS"] = str(len(atoms))
    os.environ["LD_PRELOAD"] = str(args.output / "mpi_profile.so")
    run = command(
        directory,
        [
            "mdrun",
            "-s",
            "run.tpr",
            "-deffnm",
            "mpi",
            "-nb",
            "cpu",
            "-pin",
            "off",
            "-dd",
            "2",
            "1",
            "1",
        ],
        ranks=2,
    )
    os.environ.pop("LD_PRELOAD")
    profiles = {str(rank): [] for rank in (0, 1)}
    for rank, count, elapsed in re.findall(
        r"MACE_MPI (\d+) (\d+) ([\d.]+)", run.stderr
    ):
        assert int(count) == 3 * len(atoms)
        profiles[rank].append(float(elapsed))
    # The pinned NNPot provider gathers positions, then our adapter distributes
    # forces. Both are 3*N double Allreduces, in that order, once per force call.
    assert all(len(v) == 2 * 31 for v in profiles.values()), profiles
    record["allreduce_3n_double_ms_by_rank"] = profiles
    record["position_allreduce_ms_by_rank"] = {
        rank: values[::2] for rank, values in profiles.items()
    }
    record["force_allreduce_ms_by_rank"] = {
        rank: values[1::2] for rank, values in profiles.items()
    }
    rows.append(record)
    (args.output / "result.json").write_text(json.dumps(rows, indent=2) + "\n")
    print(json.dumps(record), flush=True)
