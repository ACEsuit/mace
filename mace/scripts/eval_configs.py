import argparse

import ase.data
import ase.io
import numpy as np
import torch

from mace import data, tools
from mace.tools import torch_geometric


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
        "--no_contributions",
        help="model does not output energy contributions ",
        action="store_true",
        default=True,
    )
    return parser.parse_args()


def main():
    args = parse_args()
    tools.set_default_dtype(args.default_dtype)
    device = tools.init_device(args.device)

    # Load model
    model = torch.load(f=args.model, map_location=device)

    # Load data and prepare input
    atoms_list = ase.io.read(args.configs, format="extxyz", index=":")
    configs = [data.config_from_atoms(atoms) for atoms in atoms_list]

    z_table = tools.AtomicNumberTable([int(z) for z in model.atomic_numbers])

    data_loader = torch_geometric.dataloader.DataLoader(
        dataset=[
            data.AtomicData.from_config(config, z_table=z_table, cutoff=model.r_max)
            for config in configs
        ],
        batch_size=args.batch_size,
        shuffle=False,
        drop_last=False,
    )

    # Collect data
    energies_list = []
    contributions_list = []
    forces_collection = []

    for batch in data_loader:
        batch = batch.to(device)
        output = model(batch, training=False)
        energies_list.append(tools.to_numpy(output["energy"]))

        if not args.no_contributions:
            contributions_list.append(tools.to_numpy(output["contributions"]))

        forces = np.split(
            tools.to_numpy(output["forces"]), indices_or_sections=batch.ptr[1:], axis=0
        )
        forces_collection.append(forces[:-1])  # drop last as its emtpy

    energies = np.concatenate(energies_list, axis=0)
    forces_list = [
        forces for forces_list in forces_collection for forces in forces_list
    ]
    assert len(atoms_list) == len(energies) == len(forces_list)

    if not args.no_contributions:
        contributions = np.concatenate(contributions_list, axis=0)
        assert len(atoms_list) == contributions.shape[0]

    # Store data in atoms objects
    for i, (atoms, energy, forces) in enumerate(zip(atoms_list, energies, forces_list)):
        atoms.calc = None  # crucial
        atoms.info["energy"] = energy
        atoms.arrays["forces"] = forces

        if not args.no_contributions:
            atoms.info["contributions"] = contributions[i]

    # Write atoms to output path
    ase.io.write(args.output, images=atoms_list, format="extxyz")


if __name__ == "__main__":
    main()
