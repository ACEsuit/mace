"""Inference-only XYZ evaluation; no training or calculator API."""

from __future__ import annotations

import argparse

import ase.io
import jax
import numpy as np

from mace_jax.deploy.neutral_io import load_model
from mace_jax.graph import graph_from_atoms


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(prog="mace-jax")
    parser.add_argument("--model", required=True)
    parser.add_argument("--configs", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--dtype", choices=("float64", "float32"), default="float64")
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--backend", default="reference")
    args = parser.parse_args(argv)
    device = jax.devices("gpu" if args.device == "cuda" else "cpu")[0]
    with jax.default_device(device):
        apply, params = load_model(args.model, dtype=args.dtype, backend=args.backend)
        structures = ase.io.read(args.configs, index=":")
        for atoms in structures:
            graph = graph_from_atoms(
                atoms,
                elements=apply.elements,
                cutoff=apply.config.model.r_max,
                dtype=args.dtype,
                head=next(iter(apply.config.data.heads)),
            )
            output = apply(params, graph)
            assert output.total_energy is not None and output.stress is not None
            assert output.forces is not None
            atoms.info["MACE_energy"] = float(output.total_energy[0])
            atoms.arrays["MACE_forces"] = np.asarray(output.forces)
            atoms.info["MACE_stress"] = np.asarray(output.stress[0])
            atoms.calc = None
        ase.io.write(args.output, structures, format="extxyz")


if __name__ == "__main__":
    main()
