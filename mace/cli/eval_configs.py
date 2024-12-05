###########################################################################################
# Script for evaluating configurations contained in an xyz file with a trained model
# Authors: Ilyes Batatia, Gregor Simm
# This program is distributed under the MIT License (see MIT.md)
###########################################################################################

import argparse

import ase.data
import ase.io
import numpy as np
import torch

from mace import data
from mace.tools import torch_geometric, torch_tools, utils


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
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
        "--compute_stress",
        help="compute stress",
        action="store_true",
        default=False,
    )
    parser.add_argument(
        "--return_contributions",
        help="model outputs energy contributions for each body order, only supported for MACE, not ScaleShiftMACE",
        action="store_true",
        default=False,
    )
    # parser.add_argument(
    #     "--return_features",
    #     help="Return features of layers in `features.npy",
    #     action="store_true",
    #     default=False,
    # )
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
        default=None
    )
    parser.add_argument(
        "--predict_committee",
        help="Combine all multiheads to a committee for prediction",
        action="store_true",
        default=False,
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
        batch_size=args.batch_size,
        shuffle=False,
        drop_last=False,
    )

    # Collect data
    energies_list = []
    contributions_list = []
    # features_list = []
    stresses_list = []
    forces_list = []
    energy_stds = []
    forces_stds = []

    for batch in data_loader:
        batch = batch.to(device)
        output = model(
            batch.to_dict(),
            compute_stress=args.compute_stress,
            predict_committee=args.predict_committee
        )
        energies_list.append(torch_tools.to_numpy(output["energy"]))
        if args.compute_stress:
            stresses_list.append(torch_tools.to_numpy(output["stress"]))

        if args.return_contributions:
            contributions = np.split(
                torch_tools.to_numpy(output["contributions"]),
                indices_or_sections=batch.ptr[1:],
                axis=0,
            )
            contributions_list += contributions[:-1]

        # if args.return_features:
        #     features = np.split(
        #         torch_tools.to_numpy(output["node_feats"]),
        #         indices_or_sections=batch.ptr[1:],
        #         axis=0,
        #     )
        #     features_list += features[:-1]

        forces = np.split(
            torch_tools.to_numpy(output["forces"]),
            indices_or_sections=batch.ptr[1:],
            axis=0,
        )
        forces_list += forces[:-1]  # drop last as its empty

        if args.predict_committee:
            energy_stds.append(torch_tools.to_numpy(output["stds"]["energy"]))
            forces_std = np.split(
                torch_tools.to_numpy(output["stds"]["forces"]),
                indices_or_sections=batch.ptr[1:],
                axis=0,
            )
            forces_stds.append(forces_std[:-1])

    energies = np.concatenate(energies_list, axis=0)
    assert len(atoms_list) == len(energies) == len(forces_list)
    if args.compute_stress:
        stresses = np.concatenate(stresses_list, axis=0)
        assert len(atoms_list) == stresses.shape[0]

    if args.return_contributions:
        assert len(atoms_list) == len(contributions_list)

    # if args.return_features:
    #     assert len(atoms_list) == len(features_list)

    if args.predict_committee:
        energy_stds = np.concatenate(energy_stds, axis=0)
        forces_stds = [
            stds for std_batch in forces_stds for stds in std_batch
        ]

    # Store data in atoms objects
    for i, (atoms, energy, forces) in enumerate(zip(atoms_list, energies, forces_list)):
        atoms.calc = None  # crucial
        atoms.info[args.info_prefix + "energy"] = energy
        atoms.arrays[args.info_prefix + "forces"] = forces

        if args.compute_stress:
            atoms.info[args.info_prefix + "stress"] = stresses[i]

        if args.return_contributions:
            atoms.arrays[args.info_prefix + "BO_contributions"] = contributions_list[i]

        if args.predict_committee:
            atoms.info[args.info_prefix + "energy_std"] = energy_stds[i]
            atoms.arrays[args.info_prefix + "forces_std"] = forces_stds[i]

    # Write atoms to output path
    ase.io.write(args.output, images=atoms_list, format="extxyz")

    # Write features file, if requested
    # if args.return_features:
    #     np.save("features.npy", features)


if __name__ == "__main__":
    main()
