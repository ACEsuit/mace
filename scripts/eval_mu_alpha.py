###########################################################################################
# Script for evaluating configurations contained in an xyz file with a trained model
# Authors: Ilyes Batatia, Gregor Simm
# This program is distributed under the MIT License (see MIT.md)
###########################################################################################

import argparse, ast

import ase.data
import ase.io
import numpy as np
import torch

from mace import data
from mace.tools import torch_geometric, torch_tools, utils, tools
from mace.data import HDF5Dataset


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--configs", 
        help="path to XYZ or .h5 configurations", 
        required=True
    )
    parser.add_argument("--model", 
        help="path to model", 
        required=True
    )
    parser.add_argument("--output", 
        help="output path", 
        required=True
    )
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
    parser.add_argument("--batch_size", 
        help="batch size", 
        type=int, 
        default=64
    )
    parser.add_argument(
        "--compute_dielectric_derivatives",
        help="compute derivatives of mu and alpha",
        action="store_true",
        default=False,
    )
    parser.add_argument(
        "--output_SFG",
        help="output only tensors necessary for SFG spectra",
        action="store_true",
        default=False,
    )
    parser.add_argument(
        "--velocity_key",
        help="key for velocity array in ASE atoms object",
        type=str,
        default="velocities",
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
    if args.configs.endswith(".xyz"):
        atoms_list = ase.io.read(args.configs, index=":")
        if args.output_SFG:
            velocities = np.array([atoms.arrays[args.velocity_key] for atoms in atoms_list])
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
    elif args.configs.endswith(".h5"):
        assert args.atomic_numbers is not None
        zs_list = ast.literal_eval(args.atomic_numbers)
        assert isinstance(zs_list, list)
        z_table = tools.get_atomic_number_table_from_zs(zs_list)
        configs_preprocessed = HDF5Dataset(
            args.train_file, r_max=args.r_max, z_table=z_table
        )
        data_loader = torch_geometric.dataloader.DataLoader(
            configs_preprocessed,
            batch_size=args.batch_size,
            shuffle=False,
            drop_last=False,
            num_workers=args.num_workers,
            pin_memory=args.pin_memory,
        )
    else:
        raise ValueError("configs must be either .xyz or .h5")

    if args.output_SFG or args.compute_dielectric_derivatives:
        compute_dielectrics = True
    else:
        compute_dielectrics = False

    # Collect data
    mus_list = []
    # contributions_list = []
    alphas_list = []
    alphas_sh_list = []
    dmu_dr_collection = []
    dalpha_dr_collection = []
    with open("MACE_mu.txt", "w") as f_mu, open("MACE_alpha.txt", "w") as f_alpha, open(
        "MACE_alpha_sh.txt", "w"
    ) as f_alpha_sh:
        if args.output_SFG:
            f_mu_dot = open("MACE_mu_dot.txt", "w")
            f_alpha_dot = open("MACE_alpha_dot.txt", "w")
        for i, batch in enumerate(data_loader):
            batch = batch.to(device)
            output = model(
                batch.to_dict(),
                training=False,
                compute_stress=False,
                compute_dielectric_derivatives=compute_dielectrics,
            )
            np.savetxt(f_mu, torch_tools.to_numpy(output["dipole"]))
            np.savetxt(
                f_alpha, torch_tools.to_numpy(output["polarizability"].flatten(-2))
            )
            np.savetxt(f_alpha_sh, torch_tools.to_numpy(output["polarizability_sh"]))
            if compute_dielectrics:
                if args.compute_dielectric_derivatives:
                    dmu_dr = np.split(
                        torch_tools.to_numpy(output["dmu_dr"]),
                        indices_or_sections=batch.ptr[1:],
                        axis=0,
                    )[:-1]
                    dalpha_dr = np.split(
                        torch_tools.to_numpy(output["dalpha_dr"]),
                        indices_or_sections=batch.ptr[1:],
                        axis=0,
                    )[:-1]
                    dmu_dr_collection.append(dmu_dr)
                    dalpha_dr_collection.append(dalpha_dr)
                if args.output_SFG:
                    batch_size = batch.ptr[1:].shape[0]
                    dmu_dr = torch_tools.to_numpy(output["dmu_dr"])
                    dalpha_dr = torch_tools.to_numpy(output["dalpha_dr"])
                    nat = batch.positions.shape[0]
                    ndim = 3
                    z = torch_tools.to_numpy(batch.positions[:, 2])
                    q = torch_tools.to_numpy(batch.positions)
                    dmu_dr = np.array(dmu_dr).reshape(nat, ndim, ndim)
                    mu_x = dmu_dr[:, 0, :]
                    mu_y = dmu_dr[:, 1, :]
                    mu_z = dmu_dr[:, 2, :]
                    dalpha_dr = np.array(dalpha_dr).reshape(nat, ndim**2, ndim)
                    alpha_xx = dalpha_dr[:, 0, :]
                    alpha_yy = dalpha_dr[:, 4, :]
                    alpha_zz = dalpha_dr[:, 8, :]
                    alpha_xy = dalpha_dr[:, 1, :]
                    alpha_yz = dalpha_dr[:, 5, :]
                    alpha_zx = dalpha_dr[:, 2, :]

                    # split the tensors by graphs
                    z = np.split(z, batch.ptr[1:], axis=0)[:-1]
                    q = np.split(q, batch.ptr[1:], axis=0)[:-1]
                    mu_x = np.split(mu_x, batch.ptr[1:], axis=0)[:-1]
                    mu_y = np.split(mu_y, batch.ptr[1:], axis=0)[:-1]
                    mu_z = np.split(mu_z, batch.ptr[1:], axis=0)[:-1]
                    alpha_xx = np.split(alpha_xx, batch.ptr[1:], axis=0)[:-1]
                    alpha_yy = np.split(alpha_yy, batch.ptr[1:], axis=0)[:-1]
                    alpha_zz = np.split(alpha_zz, batch.ptr[1:], axis=0)[:-1]
                    alpha_xy = np.split(alpha_xy, batch.ptr[1:], axis=0)[:-1]
                    alpha_yz = np.split(alpha_yz, batch.ptr[1:], axis=0)[:-1]
                    alpha_zx = np.split(alpha_zx, batch.ptr[1:], axis=0)[:-1]

                    z = np.asarray(z)
                    z = z - np.mean(z)
                    z = np.sign(z)
                    q = np.asarray(q)
                    mu_x = np.asarray(mu_x)
                    mu_y = np.asarray(mu_y)
                    mu_z = np.asarray(mu_z)
                    alpha_xx = np.asarray(alpha_xx)
                    alpha_yy = np.asarray(alpha_yy)
                    alpha_zz = np.asarray(alpha_zz)
                    alpha_xy = np.asarray(alpha_xy)
                    alpha_yz = np.asarray(alpha_yz)
                    alpha_zx = np.asarray(alpha_zx)

                    vs = velocities[i * batch_size : (i + 1) * batch_size]

                    mu_x_dot = np.einsum(
                        "ijk, ijk -> i", mu_x * z[..., np.newaxis], vs
                    )  # nbatch, nat, ndim=3
                    mu_y_dot = np.einsum(
                        "ijk, ijk -> i", mu_y * z[..., np.newaxis], vs
                    )  # nbatch, nat, ndim=3
                    mu_z_dot = np.einsum(
                        "ijk, ijk -> i", mu_z * z[..., np.newaxis], vs
                    )  # nbatch, nat, ndim=3

                    alpha_xx_dot = np.einsum("ijk, ijk -> i", alpha_xx, vs)
                    alpha_yy_dot = np.einsum("ijk, ijk -> i", alpha_yy, vs)
                    alpha_zz_dot = np.einsum("ijk, ijk -> i", alpha_zz, vs)
                    alpha_xy_dot = np.einsum("ijk, ijk -> i", alpha_xy, vs)
                    alpha_yz_dot = np.einsum("ijk, ijk -> i", alpha_yz, vs)
                    alpha_zx_dot = np.einsum("ijk, ijk -> i", alpha_zx, vs)

                    np.savetxt(f_mu_dot, np.c_[mu_x_dot, mu_y_dot, mu_z_dot])
                    np.savetxt(
                        f_alpha_dot,
                        np.c_[
                            alpha_xx_dot,
                            alpha_xy_dot,
                            alpha_zx_dot,
                            alpha_xy_dot,
                            alpha_yy_dot,
                            alpha_yz_dot,
                            alpha_zx_dot,
                            alpha_yz_dot,
                            alpha_zz_dot,
                        ],
                    )
        # if args.return_contributions:
        #     contributions_list.append(torch_tools.to_numpy(output["contributions"]))

    if args.compute_dielectric_derivatives:
        dmu_dr_list = [
            dmu_dr for dmu_dr_list in dmu_dr_collection for dmu_dr in dmu_dr_list
        ]
        dalpha_dr_list = [
            dalpha_dr
            for dalpha_dr_list in dalpha_dr_collection
            for dalpha_dr in dalpha_dr_list
        ]

        # if args.return_contributions:
        #     contributions = np.concatenate(contributions_list, axis=0)
        #     assert len(atoms_list) == contributions.shape[0]

        # Store data in atoms objects
        if args.configs.endswith(".xyz"):
            for i, (atoms, dmu_dr, dalpha_dr) in enumerate(
                zip(atoms_list, dmu_dr_list, dalpha_dr_list)
            ):
                atoms.calc = None  # crucial
                if args.velocity_key in atoms.arrays.keys():
                    atoms.arrays.pop(args.velocity_key)
                atoms.arrays[args.info_prefix + "dmu_dr"] = dmu_dr.reshape(
                    dmu_dr.shape[0], 9
                )
                atoms.arrays[args.info_prefix + "dalpha_dr"] = dalpha_dr_list[i].reshape(
                    dalpha_dr_list[i].shape[0], 27
                )
        else:
            dmu_dr_array = np.array(dmu_dr_list)
            dalpha_dr_array = np.array(dalpha_dr_list)

            dmu_dr_array = dmu_dr_array.reshape(dmu_dr_array.shape[0], -1)
            dalpha_dr_array = dalpha_dr_array.reshape(dalpha_dr_array.shape[0], -1)

            np.savez_compressed("dmu_dr.npz", dmu_dr_array)
            np.savez_compressed("dalpha_dr.npz", dalpha_dr_array)


        # if args.return_contributions:
        #     atoms.info[args.info_prefix + "BO_contributions"] = contributions[i]

        # Write atoms to output path
        ase.io.write(args.output, images=atoms_list, format="extxyz")


if __name__ == "__main__":
    main()
