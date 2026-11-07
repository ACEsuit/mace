"""Native test drivers use libmace's row-vector C / column-vector Fortran ABI."""

import os
import re
import subprocess
import sys
from pathlib import Path

import numpy as np


def geometry_text(frames):
    lines = []
    for index, atoms in enumerate(frames):
        # Re-load within one process, and alternate requested/disabled virials.
        lines += [
            f"{len(atoms)} {int(index % 3 != 2)} {int(index == 1)}",
            " ".join(map(str, np.asarray(atoms.cell).ravel())),
            " ".join(str(int(value)) for value in atoms.pbc),
            " ".join(map(str, atoms.numbers)),
            " ".join(map(str, atoms.positions.ravel())),
        ]
    return "\n".join(lines) + "\n"


def run_driver(package, frames, directory, *, interface="cpp", ranks=1):
    directory.mkdir(parents=True, exist_ok=True)
    geometry = geometry_text(frames)
    if interface == "fortran":
        driver = os.environ.get("MACE_LIBMACE_FORTRAN")
        assert driver and Path(driver).is_file(), "set MACE_LIBMACE_FORTRAN"
        source = directory / "frames.txt"
        source.write_text(geometry)
        command = [
            os.environ.get("MPIEXEC", "mpiexec"),
            "-n",
            str(ranks),
            driver,
            str(package),
            str(source),
        ]
    else:
        driver = os.environ["MACE_LIBMACE_DRIVER"]
        command = [driver, str(package), *(["c"] if interface == "c" else [])]
    env = os.environ.copy()
    if sys.platform == "darwin":
        env["DYLD_PRINT_LIBRARIES"] = "1"
    else:
        env["LD_DEBUG"] = "libs"
        env["LD_DEBUG_OUTPUT"] = str(directory / "libs")
    completed = subprocess.run(
        command, input=geometry, text=True, capture_output=True, env=env, timeout=180
    )
    assert completed.returncode == 0, (
        completed.stdout[-2000:] + completed.stderr[-4000:]
    )
    libraries = completed.stderr + "".join(
        path.read_text(errors="replace") for path in directory.glob("libs.*")
    )
    assert "libtorch_cpu" in libraries
    assert not re.search(
        r"/(?:libpython[0-9]|libtorch_python|Python\.framework/.*/Python\b)", libraries
    )
    results = {rank: [] for rank in range(ranks)}
    for line in completed.stdout.splitlines():
        if line.startswith("RESULT "):
            results[0].append(np.fromstring(line[7:], sep=" "))
        elif line.startswith("RANK "):
            _, rank, values = line.split(maxsplit=2)
            results[int(rank)].append(np.fromstring(values, sep=" "))
    assert all(len(values) == len(frames) for values in results.values())
    return results


def unpack(result, atoms):
    count = len(atoms)
    assert result.shape == (1 + 4 * count + 6,)
    return {
        "energy": result[0],
        "energies": result[1 : 1 + count],
        "forces": result[1 + count : 1 + 4 * count].reshape(count, 3),
        "virial": result[-6:],
    }
