###########################################################################################
# Script for evaluating configurations contained in an xyz file with a trained model
# Authors: Ilyes Batatia, Gregor Simm
# This program is distributed under the MIT License (see MIT.md)
###########################################################################################

import argparse
import os 

os.environ['TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD'] = '1'

import ase.data
import ase.io
import numpy as np
import torch

from mace import data
from mace.tools import torch_geometric, torch_tools, utils


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--configs", help="path to XYZ configurations", required=True)
    parser.add_argument("--model", help="path to model", required=True)
    parser.add_argument("--output", help="output path", required=True)
    parser.add_argument(
        "--device",
        help="select device",
        type=str,
        choices=["cpu", "cuda"],
        default="cpu",
    )
    parser.add_argument(
        "--default_dtype",
        help="set default dtype",
        type=str,
        choices=["float32", "float64"],
        default="float64",
    )
    parser.add_argument("--batch_size", help="batch size", type=int, default=64)
    parser.add_argument(
        "--compute_energy",
        help="compute energy",
        action="store_true",
        default=False,
    )
    parser.add_argument(
        "--compute_forces",
        help="compute forces",
        action="store_true",
        default=False,
    )
    parser.add_argument(
        "--compute_stress",
        help="compute stress",
        action="store_true",
        default=False,
    )
    parser.add_argument(
        "--compute_polarisation",
        help="compute polarisation",
        action="store_true",
        default=False,
    )
    parser.add_argument(
        "--compute_becs",
        help="compute becs",
        action="store_true",
        default=False,
    )
    parser.add_argument(
        "--compute_polarisability",
        help="compute polarisability",
        action="store_true",
        default=False,
    )
    parser.add_argument(
        "--return_contributions",
        help="model outputs energy contributions for each body order, only supported for MACE, not ScaleShiftMACE",
        action="store_true",
        default=False,
    )
    parser.add_argument(
        "--info_prefix",
        help="prefix for energy, forces and stress keys",
        type=str,
        default="MACE_",
    )
    parser.add_argument(
        "--head",
        help="Model head used for evaluation",
        type=str,
        required=False,
        default=None,
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    run(args)


def run(args: argparse.Namespace) -> None:
    torch_tools.set_default_dtype(args.default_dtype)
    device = torch_tools.init_device(args.device)

    # Load model
    model = torch.load(f=args.model, map_location=args.device)
    model = model.to(
        args.device
    )  # shouldn't be necessary but seems to help with CUDA problems

    for param in model.parameters():
        param.requires_grad = False

    # Load data and prepare input
    atoms_list = ase.io.read(args.configs, index=":")
    if args.head is not None:
        for atoms in atoms_list:
            atoms.info["head"] = args.head
    configs = [data.config_from_atoms(atoms) for atoms in atoms_list]

    z_table = utils.AtomicNumberTable([int(z) for z in model.atomic_numbers])

    try:
        heads = model.heads
    except AttributeError:
        heads = None

    data_loader = torch_geometric.dataloader.DataLoader(
        dataset=[
            data.AtomicData.from_config(
                config, z_table=z_table, cutoff=float(model.r_max), heads=heads
            )
            for config in configs
        ],
        batch_size=1,
        shuffle=False,
        drop_last=False,
    )

    # Collect data
    energies_list = []
    contributions_list = []
    stresses_list = []
    forces_collection = []
    polarisations_list = []
    becs_collection = []
    polarisabilities_list = []

    for batch in data_loader:
        batch = batch.to(device)
        output = model(batch.to_dict(), compute_forces=args.compute_forces, compute_stress=args.compute_stress, compute_field=True)

        if args.compute_energy:
            energies_list.append(torch_tools.to_numpy(output["energy"]))

        if args.compute_stress:
            stresses_list.append(torch_tools.to_numpy(output["stress"]))

        if args.return_contributions:
            contributions_list.append(torch_tools.to_numpy(output["contributions"]))

        if args.compute_polarisation:
            polarisations_list.append(torch_tools.to_numpy(output["polarisation"]).reshape(3))
        
        if args.compute_polarisability:
            polarisabilities_list.append(torch_tools.to_numpy(output["polarisability"]).reshape(9))

        if args.compute_becs:
            becs = np.split(
                torch_tools.to_numpy(output["becs"]),
                indices_or_sections=batch.ptr[1:],
                axis=0,
            )
            becs = [becs.reshape(-1, 9) for becs in becs[:-1]]  # drop last as its empty
            becs_collection.append(becs)

        if args.compute_forces:
            forces = np.split(
                torch_tools.to_numpy(output["forces"]),
                indices_or_sections=batch.ptr[1:],
                axis=0,
            )
            forces_collection.append(forces[:-1])  # drop last as its empty

    if args.compute_energy:
        energies = np.concatenate(energies_list, axis=0)

    if args.compute_forces:
        forces_list = [
            forces for forces_list in forces_collection for forces in forces_list
        ]
        assert len(atoms_list) == len(energies) == len(forces_list)

    if args.compute_stress:
        stresses = np.concatenate(stresses_list, axis=0)
        assert len(atoms_list) == stresses.shape[0]

    if args.return_contributions:
        contributions = np.concatenate(contributions_list, axis=0)
        assert len(atoms_list) == contributions.shape[0]

    if args.compute_polarisation:
        polarisations = np.stack(polarisations_list, axis=0)

    if args.compute_polarisability:
        polarisabilities = np.stack(polarisabilities_list, axis=0)

    if args.compute_becs:
        becs_list = [
            becs for becs_list in becs_collection for becs in becs_list
        ]

    # Store data in atoms objects

    if args.compute_energy:
        for (atoms, energy) in zip(atoms_list, energies):
            atoms.calc = None  # crucial
            atoms.info[args.info_prefix + "energy"] = energy

    if args.compute_forces:
        for (atoms, forces) in zip(atoms_list, forces_list):
            atoms.calc = None  # crucial
            atoms.arrays[args.info_prefix + "forces"] = forces

    if args.compute_becs:
        for (atoms, becs) in zip(atoms_list, becs_list):
            atoms.calc = None  # crucial
            atoms.arrays[args.info_prefix + "becs"] = becs


    for i, atoms in enumerate(atoms_list):
        atoms.calc = None  # crucial

        if args.compute_stress:
            atoms.info[args.info_prefix + "stress"] = stresses[i]
        if args.return_contributions:
            atoms.info[args.info_prefix + "BO_contributions"] = contributions[i]
        if args.compute_polarisation:
            atoms.info[args.info_prefix + "polarisation"] = polarisations[i,:]
        if args.compute_polarisability:
            atoms.info[args.info_prefix + "polarisability"] = polarisabilities[i,:]

    # Write atoms to output path
    ase.io.write(args.output, images=atoms_list, format="extxyz")


if __name__ == "__main__":
    main()
