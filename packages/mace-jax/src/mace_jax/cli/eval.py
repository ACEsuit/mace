"""Evaluate XYZ frames through the compiled ASE calculator."""

from __future__ import annotations

import argparse

import ase.io
import numpy as np

from mace_jax.calculators import MACECalculator


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(prog="mace-jax")
    parser.add_argument("--model", required=True)
    parser.add_argument("--configs", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--dtype", choices=("float64", "float32"), default="float64")
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--backend", default="reference")
    args = parser.parse_args(argv)
    calculator = MACECalculator(
        args.model, dtype=args.dtype, device=args.device, backend=args.backend
    )
    structures = ase.io.read(args.configs, index=":")
    for atoms in structures:
        atoms.calc = calculator
        # XYZ predictions are raw model values even on constrained ASE frames.
        atoms.info["MACE_energy"] = atoms.get_potential_energy(apply_constraint=False)
        atoms.arrays["MACE_forces"] = atoms.get_forces(apply_constraint=False)
        # The XYZ contract includes a zero stress matrix for molecular frames.
        atoms.info["MACE_stress"] = (
            atoms.get_stress(voigt=False, apply_constraint=False)
            if atoms.pbc.any()
            else np.zeros((3, 3))
        )
        atoms.calc = None
    ase.io.write(args.output, structures, format="extxyz")


if __name__ == "__main__":
    main()
