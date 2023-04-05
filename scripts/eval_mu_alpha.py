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
        "--compute_dielectric_derivatives",
        help="compute derivatives of mu and alpha",
        action="store_true",
        default=False,
    )
    parser.add_argument(
        "--return_contributions",
        help="model outputs energy contributions for each body order, only suppported for MACE, not ScaleShiftMACE",
        action="store_true",
        default=False,
    )
    parser.add_argument(
        "--info_prefix",
        help="prefix for energy, forces and stress keys",
        type=str,
        default="MACE_",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    torch_tools.set_default_dtype(args.default_dtype)
    device = torch_tools.init_device(args.device)

    # Load model
    model = torch.load(f=args.model, map_location=args.device)
    model.to(device)

    for param in model.parameters():
        param.requires_grad = False

    # Load data and prepare input
    atoms_list = ase.io.read(args.configs, index=":")
    configs = [data.config_from_atoms(atoms) for atoms in atoms_list]

    z_table = utils.AtomicNumberTable([int(z) for z in model.atomic_numbers])

    data_loader = torch_geometric.dataloader.DataLoader(
        dataset=[
            data.AtomicData.from_config(
                config, z_table=z_table, cutoff=float(model.r_max)
            )
            for config in configs
        ],
        batch_size=args.batch_size,
        shuffle=False,
        drop_last=False,
    )

    # Collect data
    mus_list = []
    contributions_list = []
    alphas_list = []
    alphas_sh_list = []
    dmu_dr_collection = []
    dalpha_dr_collection = []
    for batch in data_loader:
        batch = batch.to(device)
        output = model(batch.to_dict(), 
                    training=False,
                    compute_stress=False, 
                    compute_dielectric_derivatives=args.compute_dielectric_derivatives,)
        mus_list.append(torch_tools.to_numpy(output["dipole"]))
        alphas_list.append(torch_tools.to_numpy(output["polarizability"]))
        alphas_sh_list.append(torch_tools.to_numpy(output["polarizability_sh"]))
        dmu_dr = np.split(
            torch_tools.to_numpy(output["dmu_dr"]),
            indices_or_sections=batch.ptr[1:],
            axis=0,
        )
        dmu_dr_collection.append(dmu_dr[:-1])
        dalpha_dr = np.split(
            torch_tools.to_numpy(output["dalpha_dr"]),
            indices_or_sections=batch.ptr[1:],
            axis=0,
        )
        dalpha_dr_collection.append(dalpha_dr[:-1])


        # if args.return_contributions:
        #     contributions_list.append(torch_tools.to_numpy(output["contributions"]))

    mus = np.concatenate(mus_list, axis=0)
    alphas = np.concatenate(alphas_list, axis=0)
    alphas_sh = np.concatenate(alphas_sh_list, axis=0)
    dmu_dr_list = [
        dmu_dr for dmu_dr_list in dmu_dr_collection for dmu_dr in dmu_dr_list
    ]
    dalpha_dr_list = [
        dalpha_dr for dalpha_dr_list in dalpha_dr_collection for dalpha_dr in dalpha_dr_list
    ]
    
    assert len(atoms_list) == mus.shape[0] == alphas.shape[0] == alphas_sh.shape[0]

    # if args.return_contributions:
    #     contributions = np.concatenate(contributions_list, axis=0)
    #     assert len(atoms_list) == contributions.shape[0]

    # Store data in atoms objects
    for i, (atoms, mu, alpha, dmu_dr) in enumerate(zip(atoms_list, mus, alphas, dmu_dr_list)):
        atoms.calc = None  # crucial
        atoms.info[args.info_prefix + "mu"] = mu
        atoms.info[args.info_prefix + "alpha"] = alpha
        atoms.info[args.info_prefix + "alpha_sh"] = alphas_sh[i]
        atoms.arrays[args.info_prefix +"dmu_dr"] = dmu_dr.reshape(dmu_dr.shape[0],9)
        atoms.arrays[args.info_prefix +"dalpha_dr"] = dalpha_dr_list[i].reshape(dalpha_dr_list[i].shape[0],27)

        # if args.return_contributions:
        #     atoms.info[args.info_prefix + "BO_contributions"] = contributions[i]

    # Write atoms to output path
    ase.io.write(args.output, images=atoms_list, format="extxyz")


if __name__ == "__main__":
    main()
