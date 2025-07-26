###########################################################################################
# Script for evaluating configurations contained in an xyz file with a trained model
# Authors: Ilyes Batatia, Gregor Simm
# This program is distributed under the MIT License (see MIT.md)
###########################################################################################

import argparse
from typing import Dict

import ase.data
import ase.io
import numpy as np
import torch
from e3nn import o3

from mace import data
from mace.cli.convert_e3nn_cueq import run as run_e3nn_to_cueq
from mace.modules.utils import extract_invariant
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
        "--enable_cueq",
        help="enable cuequivariance acceleration",
        action="store_true",
        default=False,
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
        "--compute_bec",
        help="compute BEC",
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
        "--return_descriptors",
        help="model outputs MACE descriptors",
        action="store_true",
        default=False,
    )
    parser.add_argument(
        "--descriptor_num_layers",
        help="number of layers to take descriptors from",
        type=int,
        default=-1,
    )
    parser.add_argument(
        "--descriptor_aggregation_method",
        help="method for aggregating node features. None saves descriptors for each atom.",
        choices=["mean", "per_element_mean", None],
        default=None,
    )
    parser.add_argument(
        "--descriptor_invariants_only",
        help="save invariant (l=0) descriptors only",
        type=bool,
        default=True,
    )
    parser.add_argument(
        "--return_node_energies",
        help="model outputs MACE node energies",
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


def get_model_output(
    model: torch.nn.Module,
    batch: Dict[str, torch.Tensor],
    compute_stress: bool,
    compute_bec: bool,
) -> Dict[str, torch.Tensor]:
    forward_args = {
        "compute_stress": compute_stress,
    }
    if compute_bec:
        # Only add `compute_bec` if it is requested
        # We check if the model is MACELES at the start of the run function
        forward_args["compute_bec"] = compute_bec
    return model(batch, **forward_args)


def main() -> None:
    args = parse_args()
    run(args)


def run(args: argparse.Namespace) -> None:
    torch_tools.set_default_dtype(args.default_dtype)
    device = torch_tools.init_device(args.device)

    # Load model
    model = torch.load(f=args.model, map_location=args.device)
    if model.__class__.__name__ != "MACELES" and args.compute_bec:
        raise ValueError("BEC can only be computed with MACELES model. ")
    if args.enable_cueq:
        print("Converting models to CuEq for acceleration")
        model = run_e3nn_to_cueq(model, device=device)
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
    descriptors_list = []
    node_energies_list = []
    stresses_list = []
    bec_list = []
    qs_list = []
    forces_collection = []

    for batch in data_loader:
        batch = batch.to(device)
        output = get_model_output(
            model, batch.to_dict(), args.compute_stress, args.compute_bec
        )
        energies_list.append(torch_tools.to_numpy(output["energy"]))
        if args.compute_stress:
            stresses_list.append(torch_tools.to_numpy(output["stress"]))
        if args.compute_bec:
            becs = np.split(
                torch_tools.to_numpy(output["BEC"]),
                indices_or_sections=batch.ptr[1:],
                axis=0,
            )
            bec_list.append(becs[:-1])  # drop last as its empty

            qs = np.split(
                torch_tools.to_numpy(output["latent_charges"]),
                indices_or_sections=batch.ptr[1:],
                axis=0,
            )
            qs_list.append(qs[:-1])  # drop last as its empty

        if args.return_contributions:
            contributions_list.append(torch_tools.to_numpy(output["contributions"]))

        if args.return_descriptors:
            num_layers = args.descriptor_num_layers
            if num_layers == -1:
                num_layers = int(model.num_interactions)
            irreps_out = o3.Irreps(str(model.products[0].linear.irreps_out))
            l_max = irreps_out.lmax
            num_invariant_features = irreps_out.dim // (l_max + 1) ** 2
            per_layer_features = [
                irreps_out.dim for _ in range(int(model.num_interactions))
            ]
            per_layer_features[-1] = (
                num_invariant_features  # Equivariant features not created for the last layer
            )

            descriptors = output["node_feats"]

            if args.descriptor_invariants_only:
                descriptors = extract_invariant(
                    descriptors,
                    num_layers=num_layers,
                    num_features=num_invariant_features,
                    l_max=l_max,
                )

            to_keep = np.sum(per_layer_features[:num_layers])
            descriptors = descriptors[:, :to_keep].detach().cpu().numpy()

            descriptors = np.split(
                descriptors,
                indices_or_sections=batch.ptr[1:],
                axis=0,
            )
            descriptors_list.extend(descriptors[:-1])  # drop last as its empty

        if args.return_node_energies:
            node_energies_list.append(
                np.split(
                    torch_tools.to_numpy(output["node_energy"]),
                    indices_or_sections=batch.ptr[1:],
                    axis=0,
                )[
                    :-1
                ]  # drop last as its empty
            )

        forces = np.split(
            torch_tools.to_numpy(output["forces"]),
            indices_or_sections=batch.ptr[1:],
            axis=0,
        )
        forces_collection.append(forces[:-1])  # drop last as its empty

    energies = np.concatenate(energies_list, axis=0)
    forces_list = [
        forces for forces_list in forces_collection for forces in forces_list
    ]
    assert len(atoms_list) == len(energies) == len(forces_list)
    if args.compute_stress:
        stresses = np.concatenate(stresses_list, axis=0)
        assert len(atoms_list) == stresses.shape[0]

    if args.compute_bec:
        bec_list = [becs for sublist in bec_list for becs in sublist]
        qs_list = [qs for sublist in qs_list for qs in sublist]

    if args.return_contributions:
        contributions = np.concatenate(contributions_list, axis=0)
        assert len(atoms_list) == contributions.shape[0]

    if args.return_descriptors:
        # no concatentation  - elements of descriptors_list have non-uniform shapes
        assert len(atoms_list) == len(descriptors_list)

    if args.return_node_energies:
        node_energies = np.concatenate(node_energies_list, axis=0)
        assert len(atoms_list) == node_energies.shape[0]

    # Store data in atoms objects
    for i, (atoms, energy, forces) in enumerate(zip(atoms_list, energies, forces_list)):
        atoms.calc = None  # crucial
        atoms.info[args.info_prefix + "energy"] = energy
        atoms.arrays[args.info_prefix + "forces"] = forces

        if args.compute_stress:
            atoms.info[args.info_prefix + "stress"] = stresses[i]

        if args.compute_bec:
            atoms.arrays[args.info_prefix + "BEC"] = bec_list[i].reshape(-1, 9)
            atoms.arrays[args.info_prefix + "latent_charges"] = qs_list[i]

        if args.return_contributions:
            atoms.info[args.info_prefix + "BO_contributions"] = contributions[i]

        if args.return_descriptors:
            descriptors = descriptors_list[i]
            if args.descriptor_aggregation_method:
                if args.descriptor_aggregation_method == "mean":
                    descriptors = np.mean(descriptors, axis=0)
                elif args.descriptor_aggregation_method == "per_element_mean":
                    descriptors = {
                        element: np.mean(
                            descriptors[atoms.symbols == element], axis=0
                        ).tolist()
                        for element in np.unique(atoms.symbols)
                    }
                atoms.info[args.info_prefix + "descriptors"] = descriptors
            else:  # args.descriptor_aggregation_method is None
                # Save descriptors for each atom (default behavior)
                atoms.arrays[args.info_prefix + "descriptors"] = np.array(descriptors)

        if args.return_node_energies:
            atoms.arrays[args.info_prefix + "node_energies"] = node_energies[i]

    # Write atoms to output path
    ase.io.write(args.output, images=atoms_list, format="extxyz")


if __name__ == "__main__":
    main()
