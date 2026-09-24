"""Writing a charge-aware model's density as cube files, from a command line.

``mace_polar_density_cube --engine v1``: one structure is evaluated with a v1
charge-aware model, and the density of its charge, of its spin, or of either
spin channel is sampled on a grid and written as a Gaussian cube, with an
optional potential and a JSON report of how faithful the grid is.

The flags are the frozen tree's, all eighteen. ``--backend`` chooses the
*interpolation*, reciprocal or real space, and has nothing to do with which
electrostatics solver the model computes with, which is the model's own.
``--model`` is a v1 checkpoint, and it is required: the frozen tree's default
named a published model that has no v1 conversion yet.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import ase.io
import numpy as np
from ase import Atoms

__all__ = ["build_parser", "load_calculator", "main", "run"]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="mace_polar_density_cube --engine v1",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        description=(
            "Write a charge-aware model's charge or spin density as cube files."
        ),
    )
    parser.add_argument("--configs", required=True, help="input XYZ/EXTXYZ file")
    parser.add_argument("--model", required=True, help="a v1 charge-aware checkpoint")
    parser.add_argument("--output", required=True, help="output cube path or prefix")
    parser.add_argument("--index", type=int, default=0, help="configuration index")
    parser.add_argument(
        "--quantity",
        choices=["charge", "spin", "alpha", "beta", "all"],
        default="spin",
        help="density to write; all writes the four with a suffix each",
    )
    parser.add_argument(
        "--grid",
        nargs=3,
        type=int,
        metavar=("NX", "NY", "NZ"),
        default=(80, 80, 160),
        help="cube grid dimensions",
    )
    parser.add_argument("--device", default="cpu", choices=["cpu", "cuda"])
    parser.add_argument(
        "--default_dtype", default="float32", choices=["float32", "float64"]
    )
    parser.add_argument(
        "--sigma", type=float, default=None, help="density width; the model's if unset"
    )
    parser.add_argument(
        "--kspace_cutoff",
        type=float,
        default=None,
        help="reciprocal-space cutoff; the model's if unset",
    )
    parser.add_argument(
        "--backend",
        choices=["auto", "fourier", "realspace"],
        default="auto",
        help="interpolation; auto uses real space where reciprocal does not apply",
    )
    parser.add_argument(
        "--realspace_cutoff_factor",
        type=float,
        default=6.0,
        help="real-space image cutoff in units of sigma",
    )
    parser.add_argument(
        "--chunk_size",
        type=int,
        default=65536,
        help="grid points per real-space evaluation chunk",
    )
    parser.add_argument(
        "--subtract_total_charge",
        action="store_true",
        help="spread the net charge over the atoms and remove it",
    )
    parser.add_argument(
        "--external_field",
        nargs=3,
        type=float,
        default=None,
        help="applied field for the corrected potential; atoms.info or zero if unset",
    )
    parser.add_argument(
        "--fermi_level",
        type=float,
        default=None,
        help="offset of the potential output; atoms.info or zero if unset",
    )
    parser.add_argument(
        "--write_potential",
        action="store_true",
        help="also write the potential and the corrected potential",
    )
    parser.add_argument(
        "--quality_report", default=None, help="a JSON path for the grid metrics"
    )
    return parser


def load_calculator(model: str, device: str, default_dtype: str):
    """The calculator and the density settings of a charge-aware checkpoint.

    Returns:
        The calculator, and the model's density width, multipole order and
        reciprocal-space cutoff.

    Raises:
        ValueError: If the checkpoint is not a charge-aware model.
    """
    from mace_core.kernels.precision import PrecisionConfig

    from mace_torch.calculators import MACECalculator
    from mace_torch.deploy.loader import load_deployed
    from mace_torch.models.electrostatics import PolarModel

    deployed = load_deployed(
        model,
        device=device,
        precision=PrecisionConfig(model=default_dtype),  # ty: ignore[invalid-argument-type]
    )
    polar = deployed.model
    if not isinstance(polar, PolarModel):
        raise ValueError(
            f"{model} is a {type(polar).__name__}, which has no density to "
            f"write. This takes a charge-aware model."
        )
    descriptor = polar.descriptor
    settings = (
        float(descriptor.smearing_width),
        int(descriptor.multipole_max_l),
        float(descriptor.kspace_cutoff),
    )
    return MACECalculator(models=deployed, device=device), settings


def _vector(values: Sequence[float] | None, default: Any) -> np.ndarray:
    array = np.asarray(default if values is None else values, dtype=float)
    if array.shape != (3,):
        raise ValueError(f"{list(array.ravel())} is not one vector of three.")
    return array


def run(args: argparse.Namespace) -> list[Path]:
    """Evaluate, sample, and write. Returns every path written, in order."""
    from mace_torch.electrostatics.density_cube import (
        QUANTITIES,
        FourierDensity,
        RealSpaceDensity,
        cube_quality_metrics,
        make_grid,
        select_backend,
        select_multipoles,
        write_cube_file,
    )

    atoms = ase.io.read(args.configs, index=args.index)
    if not isinstance(atoms, Atoms):
        raise ValueError(f"--index {args.index} names more than one structure.")
    calculator, (sigma, max_l, kspace_cutoff) = load_calculator(
        args.model, args.device, args.default_dtype
    )
    atoms.calc = calculator
    atoms.get_potential_energy()
    sigma = sigma if args.sigma is None else float(args.sigma)
    kspace_cutoff = (
        kspace_cutoff if args.kspace_cutoff is None else float(args.kspace_cutoff)
    )

    backend = select_backend(atoms, args.backend)
    if backend != "fourier" and args.write_potential:
        raise NotImplementedError(
            "--write_potential needs the potential, which only the "
            "reciprocal-space interpolation gives: use --backend fourier."
        )
    fourier = (
        FourierDensity(
            sigma=sigma,
            multipoles_max_l=max_l,
            kspace_cutoff=kspace_cutoff,
            device=args.device,
            subtract_total_charge=args.subtract_total_charge,
        )
        if backend == "fourier"
        else None
    )
    direct = (
        RealSpaceDensity(
            sigma=sigma,
            multipoles_max_l=max_l,
            device=args.device,
            cutoff_factor=args.realspace_cutoff_factor,
            chunk_size=args.chunk_size,
        )
        if fourier is None
        else None
    )
    coords = make_grid(atoms, args.grid)
    field = _vector(args.external_field, atoms.info.get("external_field", np.zeros(3)))
    fermi_level = float(
        atoms.info.get("fermi_level", 0.0)
        if args.fermi_level is None
        else args.fermi_level
    )

    quantities = list(QUANTITIES) if args.quantity == "all" else [args.quantity]
    output = Path(args.output)
    written: list[Path] = []
    reports: dict[str, Any] = {}
    for quantity in quantities:
        multipoles = select_multipoles(calculator.results, quantity)
        if fourier is not None:
            density, potential, corrected = fourier(
                atoms, multipoles, field, fermi_level, coords
            )
        else:
            assert direct is not None
            density, potential, corrected = direct(atoms, multipoles, coords)
        path = (
            output.with_name(f"{output.stem}_{quantity}.cube")
            if args.quantity == "all"
            else output
        )
        write_cube_file(path, atoms, density, f"MACE-Polar {quantity} density")
        written.append(path)
        reports[quantity] = {
            key: value.tolist() if isinstance(value, np.ndarray) else value
            for key, value in cube_quality_metrics(
                atoms, density, coords, multipoles
            ).items()
        }
        if args.write_potential:
            for suffix, data, label in (
                ("potential", potential, "potential"),
                ("potential_corrected", corrected, "corrected potential"),
            ):
                assert data is not None
                target = path.with_name(f"{path.stem}_{suffix}.cube")
                write_cube_file(target, atoms, data, f"MACE-Polar {quantity} {label}")
                written.append(target)
    if args.quality_report is not None:
        report = Path(args.quality_report)
        report.write_text(
            json.dumps(
                {"backend": backend, "grid": list(args.grid), "quantities": reports},
                indent=2,
            ),
            encoding="utf-8",
        )
        written.append(report)
    return written


def main(argv: Sequence[str] | None = None) -> int:
    for path in run(build_parser().parse_args(argv)):
        print(path)
    return 0


if __name__ == "__main__":
    sys.exit(main())
