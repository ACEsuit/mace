"""Run the GROMACS native NNP interface; lengths nm, energies kJ/mol."""

import os
import re
import subprocess
import sys
from pathlib import Path

import numpy as np

# GROMACS units.h: exact 2018 SI electron charge and Avogadro constant.
EV_TO_KJ_MOL = 1.602176634e-19 * 6.02214076e23 / 1000


def write_g96(path, frames):
    lines = ["TITLE", "MACE native parity", "END"]
    for step, atoms in enumerate(frames):
        lines += ["TIMESTEP", f"{step:15d}{step * 0.001:15.9f}", "END", "POSITION"]
        for index, (symbol, position) in enumerate(
            zip(atoms.symbols, atoms.positions / 10), 1
        ):
            lines.append(
                f"{1:5d} {'SYS':<5} {symbol:<5}{index:7d}"
                + "".join(f"{x:15.12f}" for x in position)
            )
        cell = np.asarray(atoms.cell) / 10
        values = [
            cell[i, j]
            for i, j in (
                (0, 0),
                (1, 1),
                (2, 2),
                (0, 1),
                (0, 2),
                (1, 0),
                (1, 2),
                (2, 0),
                (2, 1),
            )
        ]
        lines += ["END", "BOX", "".join(f"{x:15.12f}" for x in values), "END"]
    path.write_text("\n".join(lines) + "\n")


def prepare(directory, atoms, package, *, steps=4):
    directory.mkdir(parents=True, exist_ok=True)
    write_g96(directory / "input.g96", [atoms])
    types = []
    for number in sorted(set(atoms.numbers)):
        index = list(atoms.numbers).index(number)
        types.append(
            f"{atoms[index].symbol} {number} {atoms.get_masses()[index]:.10f} 0 A 0 0"
        )
    topology = [
        "[ defaults ]",
        "1 2 no 1 1",
        "[ atomtypes ]",
        *types,
        "[ moleculetype ]",
        "SYS 0",
        "[ atoms ]",
    ]
    topology += [
        f"{i} {a.symbol} 1 SYS {a.symbol} {i} 0 {a.mass:.10f}"
        for i, a in enumerate(atoms, 1)
    ]
    topology += ["[ system ]", "MACE parity", "[ molecules ]", "SYS 1"]
    (directory / "system.top").write_text("\n".join(topology) + "\n")
    (directory / "run.mdp").write_text(f"""integrator = md
nsteps = {steps}
dt = 0.0001
comm-mode = none
constraints = none
cutoff-scheme = Verlet
nstlist = 1
verlet-buffer-tolerance = -1
rlist = 0.12
coulombtype = Cut-off
rcoulomb = 0.1
rvdw = 0.1
pbc = xyz
tcoupl = no
pcoupl = no
gen-vel = no
nstxout = 1
nstvout = 1
nstfout = 1
nstenergy = 1
nstcalcenergy = 1
nnpot-active = true
nnpot-modelfile = {package}
nnpot-input-group = System
nnpot-model-input1 = atom-positions
nnpot-model-input2 = atom-numbers
nnpot-model-input3 = box
nnpot-model-input4 = pbc
""")


def command(directory, args, *, input=None, trace=False, ranks=1):
    env = os.environ.copy()
    env.update(GMX_MAXBACKUP="-1", OMP_NUM_THREADS="1")
    if trace:
        if sys.platform == "darwin":
            env["DYLD_PRINT_LIBRARIES"] = "1"
        else:
            env["LD_DEBUG"] = "libs"
            env["LD_DEBUG_OUTPUT"] = str(directory / "libs")
    argv = [os.environ["MACE_GROMACS_BINARY"], *args]
    if ranks > 1 and os.environ.get("MACE_GROMACS_MPIEXEC"):
        argv = [os.environ["MACE_GROMACS_MPIEXEC"], "-n", str(ranks), *argv]
    completed = subprocess.run(
        argv,
        cwd=directory,
        input=input,
        text=True,
        capture_output=True,
        env=env,
        timeout=180,
    )
    (directory / f"{args[0]}.log").write_text(completed.stdout + completed.stderr)
    assert completed.returncode == 0, completed.stderr[-7000:]
    if trace:
        libraries = completed.stderr + "".join(
            p.read_text(errors="replace") for p in directory.glob("libs.*")
        )
        assert "libmace" in libraries and "libtorch_cpu" in libraries
        assert not re.search(
            r"/(?:libpython[0-9]|libtorch_python|Python\.framework/.*/Python\b)",
            libraries,
        )
    return completed


def table(path):
    return np.atleast_2d(
        np.loadtxt(
            [
                line
                for line in path.read_text().splitlines()
                if line.strip() and not line.startswith(("#", "@"))
            ]
        )
    )


def run(directory, *, rerun=None, ranks=1):
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
    args = ["mdrun", "-s", "run.tpr", "-deffnm", "result", "-nb", "cpu", "-pin", "off"]
    if not os.environ.get("MACE_GROMACS_MPIEXEC"):
        args += ["-ntmpi", str(ranks)]
    if ranks > 1:
        args += ["-dd", str(ranks), "1", "1"]
    if rerun is not None:
        args += ["-rerun", str(rerun)]
    command(directory, args, trace=True, ranks=ranks)
    if ranks > 1:
        log = (directory / "result.log").read_text()
        grid = re.search(r"Domain decomposition grid (\d+) x (\d+) x (\d+)", log)
        assert grid and np.prod([int(x) for x in grid.groups()]) == ranks
        if rerun is not None:
            distributions = re.findall(r"Atom distribution over \d+ domains:.*", log)
            assert len(set(distributions)) > 1, "The atoms did not change MPI ownership"
    command(
        directory,
        [
            "traj",
            "-f",
            "result.trr",
            "-s",
            "run.tpr",
            "-of",
            "forces.xvg",
            "-ox",
            "positions.xvg",
            "-ob",
            "box.xvg",
            "-fp",
            "-xvg",
            "none",
        ],
        input="System\n",
    )
    command(
        directory,
        ["energy", "-f", "result.edr", "-o", "energy.xvg", "-dp", "-xvg", "xmgrace"],
        input="Potential\n0\n"
        if rerun is not None
        else "Potential\nVir-XX\nVir-YY\nVir-ZZ\nVir-XY\nVir-XZ\nVir-YZ\n0\n",
    )
    energy_file = directory / "energy.xvg"
    energies = table(energy_file)
    labels = {
        label: int(index) + 1
        for index, label in re.findall(
            r'@\s+s(\d+)\s+legend\s+"([^"]+)"', energy_file.read_text()
        )
    }
    output = {
        "energy": energies[:, labels["Potential"]] / EV_TO_KJ_MOL,
        "forces": table(directory / "forces.xvg")[:, 1:].reshape(len(energies), -1, 3)
        / (10 * EV_TO_KJ_MOL),
        "positions": table(directory / "positions.xvg")[:, 1:].reshape(
            len(energies), -1, 3
        )
        * 10,
    }
    # GROMACS deliberately omits virials from rerun output. MD verifies them.
    if rerun is None:
        columns = [
            labels[key]
            for key in ("Vir-XX", "Vir-YY", "Vir-ZZ", "Vir-XY", "Vir-XZ", "Vir-YZ")
        ]
        output["virial"] = -2 * energies[:, columns] / EV_TO_KJ_MOL
    boxes = table(directory / "box.xvg")[:, 1:] * 10
    output["cell"] = np.zeros((len(energies), 3, 3))
    for column, (i, j) in enumerate(((0, 0), (1, 1), (2, 2), (1, 0), (2, 0), (2, 1))):
        output["cell"][:, i, j] = boxes[:, column]
    return output
