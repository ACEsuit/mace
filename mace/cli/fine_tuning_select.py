###########################################################################################
# This program is distributed under the MIT License (see MIT.md)
###########################################################################################

import argparse
import logging
import typing as t


import ase.data
import ase.io
import numpy as np
import torch

from mace.calculators import MACECalculator, mace_mp
from tqdm import tqdm

from mace import data
import pandas as pd
from mace.tools import torch_geometric, torch_tools, utils


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--configs_pt",
        help="path to XYZ configurations for the pretraining",
        required=True,
    )
    parser.add_argument(
        "--configs_ft",
        help="path to XYZ configurations for the finetuning",
        required=True,
    )
    parser.add_argument(
        "--num_samples",
        help="number of samples to select for the pretraining",
        type=int,
    )
    parser.add_argument("--model", help="path to model", required=True)
    parser.add_argument("--output", help="output path", required=True)
    parser.add_argument(
        "--descriptors", help="path to descriptors", required=False, default=None
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
    parser.add_argument(
        "--info_prefix",
        help="prefix for energy, forces and stress keys",
        type=str,
        default="MACE_",
    )
    parser.add_argument(
        "--theory_pt",
        help="level of theory for the pretraining set",
        type=str,
        default=None,
    )
    parser.add_argument(
        "--theory_ft",
        help="level of theory for the finetuning set",
        type=str,
        default=None,
    )
    return parser.parse_args()


def calculate_descriptors(
    atoms: t.List[ase.Atoms | ase.Atom], calc: MACECalculator, cutoffs: None | dict
) -> None:
    print("Calculating descriptors")
    for mol in tqdm(atoms):
        descriptors = calc.get_descriptors(mol.copy(), invariants_only=True)
        # average descriptors over atoms for each element
        descriptors_dict = {
            element: np.mean(descriptors[mol.symbols == element], axis=0)
            for element in np.unique(mol.symbols)
        }
        mol.info["mace_descriptors"] = descriptors_dict


def filter_atoms(
    atoms: ase.Atoms, element_subset: list[str], filtering_type: str
) -> bool:
    """
    Filters atoms based on the provided filtering type and element subset.

    Parameters:
    atoms (ase.Atoms): The atoms object to filter.
    element_subset (list): The list of elements to consider during filtering.
    filtering_type (str): The type of filtering to apply. Can be 'none', 'exclusive', or 'inclusive'.
        'none' - No filtering is applied.
        'combinations' - Return true if `atoms` is composed of combinations of elements in the subset, false otherwise. I.e. does not require all of the specified elements to be present.
        'exclusive' - Return true if `atoms` contains *only* elements in the subset, false otherwise.
        'inclusive' - Return true if `atoms` contains all elements in the subset, false otherwise. I.e. allows additional elements.

    Returns:
    bool: True if the atoms pass the filter, False otherwise.
    """
    if filtering_type == "none":
        return True
    elif filtering_type == "combinations":
        atom_symbols = np.unique(atoms.symbols)
        return all(
            [x in element_subset for x in atom_symbols]
        )  # atoms must *only* contain elements in the subset
    elif filtering_type == "exclusive":
        atom_symbols = set([x for x in atoms.symbols])
        return atom_symbols == set(element_subset)
    elif filtering_type == "inclusive":
        atom_symbols = np.unique(atoms.symbols)
        return all(
            [x in atom_symbols for x in element_subset]
        )  # atoms must *at least* contain elements in the subset
    else:
        raise ValueError(
            f"Filtering type {filtering_type} not recognised. Must be one of 'none', 'exclusive', or 'inclusive'."
        )


class FPS:
    def __init__(self, atoms_list: list[ase.Atoms], n_samples: int):
        self.n_samples = n_samples
        self.atoms_list = atoms_list
        # start from a random configuration
        self.list_index = [np.random.randint(0, len(atoms_list))]

    def run(self) -> list[int]:
        """
        Run the farthest point sampling algorithm.
        """
        for _ in range(max(self.n_samples, len(self.atoms_list)) - 1):
            self.update()
        return self.list_index

    def update(self) -> list[int]:
        """
        Compute the farthest point sampling for the index-th configuration.
        """
        distance_matrix = self.compute_distance(self.list_index[-1])
        index_next = np.argmax(np.mean(distance_matrix, axis=1))
        self.list_index.append(index_next)

    def compute_distance(self, index: int) -> np.ndarray:
        """
        Compute the distance matrix between the descriptor of the index-th configuration and all the other configurations.
        """
        descriptors_filtered = self.filter_species(index)
        # compute the distance matrix
        distance_matrix = np.zeros((len(self.atoms_list), len(descriptors_filtered)))
        descriptors_atoms_index = self.atoms_list[index].info["mace_descriptors"]
        for zi, z in enumerate(descriptors_filtered):
            distance_matrix[:, zi] = np.nan_to_num(
                np.linalg.norm(
                    descriptors_filtered[z] - descriptors_atoms_index[z],
                    axis=1,
                )
            )
            # put inf to zeros
        return distance_matrix

    def filter_species(self, index: int) -> list[ase.Atoms]:
        """
        Filter the configurations based on the species of the index-th configuration.
        """
        species_index = np.unique(self.atoms_list[index].symbols)
        descriptors_species = {z: [] for z in species_index}
        descriptors_index = self.atoms_list[index].info["mace_descriptors"]
        for i, atoms in enumerate(self.atoms_list):
            descriptors_atoms = atoms.info["mace_descriptors"]
            for z in species_index:
                descriptors_species[z].append(
                    descriptors_atoms[z]
                    if z in descriptors_atoms
                    else np.full_like(descriptors_index[z], np.nan)
                )
        for z in species_index:
            descriptors_species[z] = np.array(descriptors_species[z])
        return descriptors_species


def main():
    args = parse_args()
    if args.model in ["small", "medium", "large"]:
        calc = mace_mp(args.model, device=args.device, default_dtype=args.default_dtype)
    else:
        calc = MACECalculator(
            model_paths=args.model, device=args.device, default_dtype=args.default_dtype
        )
    atoms_list_ft = ase.io.read(args.configs_ft, index=":")
    all_species_ft = np.unique([x.symbol for atoms in atoms_list_ft for x in atoms])
    print(
        "Filtering configurations based on the finetuning set,"
        f"filtering type: combinations, elements: {all_species_ft}"
    )

    if args.descriptors is not None:
        print("Loading descriptors")
        descriptors = np.load(args.descriptors)
        atoms_list_pt = ase.io.read(args.configs_pt, index=":")
        for i, atoms in enumerate(atoms_list_pt):
            atoms.arrays["mace_descriptors"] = descriptors[i]
        print(
            "Filtering configurations based on the finetuning set,"
            f"filtering type: combinations, elements: {all_species_ft}"
        )
        atoms_list_pt = [
            x for x in atoms_list_pt if filter_atoms(x, all_species_ft, "combinations")
        ]

    else:
        print("Calculating descriptors for the pretraining set")
        atoms_list_pt = ase.io.read(args.configs_pt, index=":")
        atoms_list_pt = [
            x for x in atoms_list_pt if filter_atoms(x, all_species_ft, "combinations")
        ]
        calculate_descriptors(atoms_list_pt, calc, None)
    if args.num_samples < len(atoms_list_pt):
        print("Selecting configurations using Farthest Point Sampling")
        fps_pt = FPS(atoms_list_pt, args.num_samples)
        idx_pt = fps_pt.run()
        atoms_list_pt = [atoms_list_pt[i] for i in idx_pt]
    for atoms in atoms_list_pt:
        # del atoms.info["mace_descriptors"]
        atoms.info["pretrained"] = True
        if args.theory_pt is not None:
            atoms.info["theory"] = args.theory_pt

    print("Saving the selected configurations")
    ase.io.write(args.output, atoms_list_pt, format="extxyz")
    print("Saving a combined XYZ file")
    for atoms in atoms_list_ft:
        if args.theory_ft is not None:
            atoms.info["theory"] = args.theory_ft
    atoms_fps_pt_ft = atoms_list_pt + atoms_list_ft
    ase.io.write(
        args.output.replace(".xyz", "_combined.xyz"), atoms_fps_pt_ft, format="extxyz"
    )


if __name__ == "__main__":
    main()
