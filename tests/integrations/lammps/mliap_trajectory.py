"""MPI subprocess for the shared neighbor-change and atom-migration trajectory."""

import argparse
import faulthandler
from pathlib import Path

from lammps import lammps
from lammps.mliap import activate_mliappy_kokkos, load_unified_kokkos
from mace_torch.deploy.mliap import MACEMLIAPUnified


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("bundle")
    parser.add_argument("output", type=Path)
    parser.add_argument("--ranks", type=int, required=True)
    parser.add_argument("--gpus", type=int, required=True)
    args = parser.parse_args()
    faulthandler.dump_traceback_later(90, repeat=True)
    adapter = MACEMLIAPUnified(args.bundle)
    with lammps(
        cmdargs=[
            "-k",
            "on",
            "g",
            str(args.gpus),
            "-sf",
            "kk",
            "-pk",
            "kokkos",
            "newton",
            "on",
            "neigh",
            "half",
            "comm",
            "device",
            "gpu/aware",
            "on",
            "-log",
            str(args.output / "log"),
            "-screen",
            "none",
        ]
    ) as lmp:
        activate_mliappy_kokkos(lmp)
        load_unified_kokkos(adapter)
        lmp.command(f"variable ranks equal {args.ranks}")
        lmp.command(f"variable out string {args.output}")
        for command in (
            Path(__file__).with_name("native_trajectory.in").read_text().splitlines()
        ):
            if command.startswith("plugin load"):
                continue
            if command.startswith("pair_style"):
                command = "pair_style mliap/kk unified EXISTS 0"
            lmp.command(command)
    # This driver owns MPI/KOKKOS. Python owns the calling interpreter.
    # LAMMPS 22Jul2025's broad finalize() also calls Py_Finalize from ctypes.
    lmp.lib.lammps_kokkos_finalize()
    lmp.lib.lammps_mpi_finalize()
    faulthandler.cancel_dump_traceback_later()


if __name__ == "__main__":
    main()
