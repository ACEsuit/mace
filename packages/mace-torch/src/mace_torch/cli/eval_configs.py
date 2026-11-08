"""Batch evaluation and the extxyz output names of the public CLI.

The file boundary keeps the historical names; model outputs retain their
canonical names. Energies are eV, forces eV/Angstrom and stress eV/Angstrom^3.
"""

from __future__ import annotations

import argparse
import logging
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class OutputColumn:
    """One extxyz column, its model quantity and the ticket owning it."""

    quantity: str
    per_atom: bool
    owner: str = "CLI-1"


OUTPUT_COLUMNS = {
    "energy": OutputColumn("energy", False),
    "forces": OutputColumn("forces", True),
    "stress": OutputColumn("stress", False),
    "BO_contributions": OutputColumn("contributions", False),
    "descriptors": OutputColumn("descriptors", True),
    "node_energies": OutputColumn("node_energies", True),
    "magforces": OutputColumn("magforces", True, "MAG-1"),
    "BEC": OutputColumn("BEC", True, "ELEC-4"),
    **{
        f"latent_{name}": OutputColumn(f"latent_{name}", True, "ELEC-4")
        for name in ("charges", "dipoles", "kappas", "alphas", "quads")
    },
}


def boolean(value: str) -> bool:
    """A boolean value, including an explicit false on a command line."""
    if value.lower() in {"true", "1", "yes"}:
        return True
    if value.lower() in {"false", "0", "no"}:
        return False
    raise argparse.ArgumentTypeError(f"{value!r}: expected true or false")


def add_arguments(parser: argparse.ArgumentParser) -> None:
    for name in ("configs", "model", "output"):
        parser.add_argument(f"--{name}", type=Path)
    parser.add_argument("--device", default="cpu")
    parser.add_argument(
        "--default_dtype", choices=("float32", "float64"), default="float64"
    )
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--head")
    parser.add_argument("--info_prefix", default="MACE_")
    parser.add_argument("--magmom_key", default="REF_magmom")
    for name in (
        "compute_stress",
        "compute_bec",
        "return_contributions",
        "return_descriptors",
        "return_node_energies",
        "return_magforces",
        "enable_cueq",
    ):
        parser.add_argument(f"--{name}", action="store_true")
    parser.add_argument("--descriptor_num_layers", type=int, default=-1)
    parser.add_argument("--descriptor_invariants_only", type=boolean, default=True)
    parser.add_argument(
        "--descriptor_aggregation_method", choices=("mean", "per_element_mean")
    )


def run(args: argparse.Namespace) -> int:
    import numpy as np
    import torch
    from ase.io import read, write
    from mace_core.config.precision import PrecisionConfig
    from mace_core.data.xyz import configuration_from_atoms

    from mace_torch.calculators.ase_calculator import produced_outputs
    from mace_torch.cli.reporting import print_citations
    from mace_torch.data.graphs import graph_from_configuration
    from mace_torch.data.inputs import graph_inputs_of, key_specification
    from mace_torch.deploy.loader import load_deployed
    from mace_torch.graph import collate
    from mace_torch.nn import MACEBackbone

    for name in ("configs", "model", "output"):
        if getattr(args, name) is None:
            raise ValueError(f"mace eval requires --{name}")
    if args.batch_size < 1:
        raise ValueError("--batch_size must be positive")
    if args.enable_cueq:
        raise ValueError(
            "--enable_cueq is replaced by the checkpoint's backend "
            "selection; canonical weights need no conversion"
        )
    deployed = load_deployed(
        args.model,
        device=args.device,
        precision=PrecisionConfig(model=args.default_dtype),
    )
    if args.default_dtype != deployed.config.precision.model:
        logging.getLogger(__name__).warning(
            "Requested dtype %s does not match model dtype %s; evaluating with %s.",
            args.default_dtype,
            deployed.config.precision.model,
            args.default_dtype,
        )
    head = deployed.select_head(args.head)
    available = produced_outputs(
        [spec.name for spec in deployed.outputs.observables],
        getattr(deployed.model, "extra_rows", {}),
        produced=getattr(deployed.model, "produced_observables", ()),
        derivatives=deployed.outputs.derivatives,
    )
    compute = [name for name in ("forces",) if name in available]
    for flag, name in (
        (args.compute_stress, "stress"),
        (args.compute_bec, "BEC"),
        (args.return_magforces, "magforces"),
    ):
        if flag:
            if name not in available:
                raise ValueError(
                    f"the model does not declare {name!r} "
                    f"(model={deployed.config.model.model!r})"
                )
            compute.append(name)
    if args.return_contributions and deployed.config.model.model != "plain":
        raise ValueError("--return_contributions requires a plain energy model")
    structures = list(read(args.configs, index=":"))
    inputs = graph_inputs_of(deployed.config.model)
    keys = key_specification(deployed.config.data, model=deployed.config.model)
    features = deployed.config.model.feature_storages
    columns = dict(OUTPUT_COLUMNS)
    for spec in deployed.outputs.observables:
        columns.setdefault(spec.name, OutputColumn(spec.name, spec.per_atom))
    if "magmom" in inputs:
        keys.update(atom_keys={"magmom": args.magmom_key})
    for start in range(0, len(structures), args.batch_size):
        batch_atoms = structures[start : start + args.batch_size]
        if "magmom" in inputs and any(
            args.magmom_key not in atoms.arrays for atoms in batch_atoms
        ):
            raise ValueError(f"the model requires atoms.arrays[{args.magmom_key!r}]")
        samples = [
            graph_from_configuration(
                configuration_from_atoms(atoms, keys),
                cutoff=deployed.r_max,
                z_table=deployed.z_table,
                head=deployed.heads.index(head),
                graph_inputs=inputs,
                features=features,
            )
            for atoms in batch_atoms
        ]
        graph = {
            name: value.to(args.device) if isinstance(value, torch.Tensor) else value
            for name, value in collate(
                samples, dtype=getattr(torch, args.default_dtype), features=features
            ).items()
        }
        output = deployed.compute(graph, compute=compute)
        if args.return_descriptors:
            backbone = deployed.model.get_submodule("backbone")
            assert isinstance(backbone, MACEBackbone)
            with torch.no_grad():
                output.extras["descriptors"] = backbone.descriptors(
                    backbone(graph),
                    num_layers=args.descriptor_num_layers,
                    invariants_only=args.descriptor_invariants_only,
                )
        enabled = {
            "energy",
            "forces",
            *(spec.name for spec in deployed.outputs.observables),
        }
        enabled.update(
            name
            for flag, name in (
                (args.compute_stress, "stress"),
                (args.return_contributions, "BO_contributions"),
                (args.return_descriptors, "descriptors"),
                (args.return_node_energies, "node_energies"),
                (args.return_magforces, "magforces"),
                (args.compute_bec, "BEC"),
            )
            if flag
        )
        enabled.update(name for name in OUTPUT_COLUMNS if name.startswith("latent_"))
        offset = 0
        for index, atoms in enumerate(batch_atoms):
            atoms.calc = None
            for name, column in columns.items():
                value = output.get(column.quantity)
                if name not in enabled or value is None:
                    continue
                values = value.detach().cpu().numpy()
                row = (
                    values[offset : offset + len(atoms)]
                    if column.per_atom
                    else values[index]
                )
                target = atoms.arrays if column.per_atom else atoms.info
                if name == "descriptors" and args.descriptor_aggregation_method:
                    target = atoms.info
                    row = (
                        row.mean(axis=0)
                        if args.descriptor_aggregation_method == "mean"
                        else np.concatenate(
                            [
                                row[atoms.numbers == number].mean(axis=0)
                                if (atoms.numbers == number).any()
                                else np.zeros(row.shape[1])
                                for number in deployed.z_table.zs
                            ]
                        )
                    )
                elif column.per_atom and row.ndim > 2:
                    row = row.reshape(len(atoms), -1)
                target[args.info_prefix + name] = row
            offset += len(atoms)
    write(args.output, structures, format="extxyz")
    print_citations(deployed.metadata)
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="mace eval")
    add_arguments(parser)
    return run(parser.parse_args(argv))
