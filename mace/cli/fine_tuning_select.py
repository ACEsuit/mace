###########################################################################################
# This program is distributed under the MIT License (see MIT.md)
###########################################################################################

import argparse
import logging
from typing import List


import ase.data
import ase.io
import numpy as np
import torch

from mace.calculators import MACECalculator, mace_mp
from tqdm import tqdm

from mace import data
import pandas as pd
from mace.tools import torch_geometric, torch_tools, utils
import fpsample


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
        required=False,
        default=None,
    )
    parser.add_argument(
        "--model", help="path to model", default="small", required=False
    )
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
        "--head_pt",
        help="level of head for the pretraining set",
        type=str,
        default=None,
    )
    parser.add_argument(
        "--head_ft",
        help="level of head for the finetuning set",
        type=str,
        default=None,
    )
    parser.add_argument(
        "--filtering_type",
        help="filtering type",
        type=str,
        choices=["none", "combinations", "exclusive", "inclusive", "any"],
        default=None,
    )
    parser.add_argument(
        "--weight_ft",
        help="weight for the finetuning set",
        type=float,
        default=1.0,
    )
    parser.add_argument(
        "--weight_pt",
        help="weight for the pretraining set",
        type=float,
        default=1.0,
    )
    parser.add_argument("--seed", help="random seed", type=int, default=42)
    return parser.parse_args()


def calculate_descriptors(
    atoms: List[ase.Atoms], calc: MACECalculator, cutoffs: None
) -> None:
    logging.info("Calculating descriptors")
    for mol in atoms:
        descriptors = calc.get_descriptors(mol.copy(), invariants_only=True)
        # average descriptors over atoms for each element
        descriptors_dict = {
            element: np.mean(descriptors[mol.symbols == element], axis=0)
            for element in np.unique(mol.symbols)
        }
        mol.info["mace_descriptors"] = descriptors_dict


def filter_atoms(
    atoms: ase.Atoms, element_subset: set[str], filtering_type: str
) -> bool:
    """
    Filters atoms based on the provided filtering type and element subset.

    Parameters:
    atoms (ase.Atoms): The atoms object to filter.
    element_subset (list): The list of elements to consider during filtering.
    filtering_type (str): The type of filtering to apply. Can be 'none', 'combination', 'exclusive', 'inclusive', 'any'.
        'none' - No filtering is applied.
        'combinations' - Return true if `atoms` is composed of combinations of elements in the subset, false otherwise. I.e. does not require all of the specified elements to be present.
        'exclusive' - Return true if `atoms` contains *only* elements in the subset, false otherwise.
        'inclusive' - Return true if `atoms` contains all elements in the subset, false otherwise. I.e. allows additional elements.
        'any' - Return true if any of the element_subset is in the configuration

    Returns:
    bool: True if the atoms pass the filter, False otherwise
    """
    if filtering_type == "none":
        return True
    elif filtering_type == "combinations":
        return set(atoms.symbols).issubset(element_subset)
    elif filtering_type == "exclusive":
        return set(atoms.symbols) == element_subset
    elif filtering_type == "inclusive":
        return element_subset.issubset(atoms.symbols)
    elif filtering_type == "any":
        return len(element_subset & set(atoms.symbols)) > 0
    else:
        raise ValueError(
            f"Filtering type {filtering_type} not recognised. Must be one of 'none', 'combinations', 'exclusive', 'inclusive', 'any'."
        )


class FPS:
    def __init__(self, atoms_list: list[ase.Atoms], n_samples: int):
        self.n_samples = n_samples
        self.atoms_list = atoms_list
        self.species = np.unique([x.symbol for atoms in atoms_list for x in atoms])
        self.species_dict = {x: i for i, x in enumerate(self.species)}
        # start from a random configuration
        self.list_index = [np.random.randint(0, len(atoms_list))]
        self.assemble_descriptors()

    def run(
        self,
    ) -> list[int]:
        """
        Run the farthest point sampling algorithm.
        """
        logging.info(f"fps descriptor shape {self.descriptors_dataset.reshape((len(self.atoms_list), -1)).shape}")
        logging.info(f"fps n_samples {self.n_samples}")
        self.list_index = fpsample.fps_npdu_kdtree_sampling(
            self.descriptors_dataset.reshape((len(self.atoms_list), -1)), self.n_samples
        )
        return self.list_index

    def assemble_descriptors(self) -> np.ndarray:
        """
        Assemble the descriptors for all the configurations.
        """
        self.descriptors_dataset = np.float32(
            10e10
            * np.ones(
                (
                    len(self.atoms_list),
                    len(self.species),
                    len(list(self.atoms_list[0].info["mace_descriptors"].values())[0]),
                ),
                dtype=np.float32,
            )
        )
        for i, atoms in enumerate(self.atoms_list):
            descriptors = atoms.info["mace_descriptors"]
            for z in descriptors:
                self.descriptors_dataset[i, self.species_dict[z]] = np.float32(
                    descriptors[z]
                )


def select_samples(
    args: argparse.Namespace,
) -> None:
    # setup
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    # read finetuning set
    atoms_list_ft = ase.io.read(args.configs_ft, index=":")

    # read pretrained set
    atoms_list_pt = ase.io.read(args.configs_pt, index=":")

    # do filtering by elements
    indices_pt_filtered = []
    atoms_list_pt_filtered = []
    if args.filtering_type != "none":
        all_species_ft = {atom.symbol for atoms in atoms_list_ft for atom in atoms}
        logging.info(
            "Filtering configurations based on the finetuning set, "
            f"filtering type: {args.filtering_type}, elements: {all_species_ft}"
        )

        # get filter weights based on elemental composition
        pt_filter = [filter_atoms(atoms, all_species_ft, args.filtering_type) for atoms in atoms_list_pt]
        if sum(pt_filter) <= args.num_samples:
            # few enough to include all
            logging.info(f"Found few enough to include all {sum(pt_filter)} filtered by elements")
            indices_pt_filtered = np.where(pt_filter)[0]
        else:
            # too many, select by weight
            # [NB allow setting of exponential base, currently 10.0 ?]]
            logging.info(f"Found too many filtered by elements {sum(pt_filter)}, choosing randomly with weights")
            # try increasingly generous matching strategies
            indices_pt_filtered_orig = set(np.where(pt_filter)[0])
            indices_pt_filtered = set()
            for strategy in ("exclusive", "combinations", "any"):
                strategy_filter = [filter_atoms(atoms, all_species_ft, strategy) for atoms in atoms_list_pt]
                if sum(strategy_filter) == 0:
                    logging.info(f"Nothing selected by {strategy}")
                    continue
                indices_pt_strategy = set(np.where(strategy_filter)[0]) & indices_pt_filtered_orig
                indices_pt_strategy -= indices_pt_filtered
                if len(indices_pt_filtered) + len(indices_pt_strategy) <= args.num_samples:
                    # can include all of these
                    indices_pt_filtered |= indices_pt_strategy
                    logging.info(f"Adding all {len(indices_pt_strategy)} selected by {strategy}")
                else:
                    # pick a subset with weights, penalizing missing and extra elements
                    # for exclusive distances should all be 0
                    # for combinations should only have missing elements, no extra (first term)
                    # for any could have either/both, add them up (both terms)
                    indices_pt_strategy = list(indices_pt_strategy)
                    d = np.asarray([len(all_species_ft - set(atoms_list_pt[ind].symbols)) +
                                    len(set(atoms_list_pt[ind].symbols) - all_species_ft) for ind in indices_pt_strategy])
                    p = 10.0 ** (-d)
                    p /= np.sum(p)
                    inds = np.random.choice(len(indices_pt_strategy), args.num_samples - len(indices_pt_filtered), replace=False, p=p)
                    logging.info(f"Adding subset len {len(inds)} selected by {strategy}")
                    indices_pt_filtered |= {indices_pt_strategy[ind] for ind in inds}
                    break

    # read or calculate _ALL_ descriptors if needed
    # [NB: should we support random instead of FPS below, and not do this if that's being used?]
    if len(indices_pt_filtered) < args.num_samples:
        if args.descriptors is not None:
            logging.info(f"Loading descriptors from {args.descriptors}")
            descriptors = np.load(args.descriptors, allow_pickle=True)
            for descriptor, atoms in zip(descriptors, atoms_list_pt):
                atoms.info["mace_descriptors"] = descriptor
        else:
            logging.info("Calculating descriptors for the pretraining set")
            if args.model in ["small", "medium", "large"]:
                calc = mace_mp(args.model, device=args.device, default_dtype=args.default_dtype)
            else:
                calc = MACECalculator(
                    model_paths=args.model, device=args.device, default_dtype=args.default_dtype
                )
            calculate_descriptors(atoms_list_pt, calc, None)
            descriptors_list = [
                atoms.info["mace_descriptors"] for atoms in atoms_list_pt
            ]
            logging.info(
                "Saving descriptors at {}".format(
                    args.output.replace(".xyz", "descriptors.npy")
                )
            )
            np.save(args.output.replace(".xyz", "descriptors.npy"), descriptors_list)

    # actually do filtering
    atoms_list_pt_filtered = [atoms_list_pt[ind] for ind in indices_pt_filtered]

    # get additional configs from across DB
    # [NB: should support both random and FPS?]
    # [NB: should we be able to control this separately from size set chosen by filtering?]
    atoms_list_pt_extra = []
    if len(atoms_list_pt_filtered) < args.num_samples:
        logging.info(
            f"Number of configurations after filtering {len(atoms_list_pt_filtered)} "
            f"is less than the number of samples {args.num_samples}, "
            "selecting the rest with FPS."
        )

        indices_pt_avail = set(list(range(len(atoms_list_pt)))) - set(indices_pt_filtered)
        atoms_list_pt_avail = [atoms_list_pt[ind] for ind in indices_pt_avail]

        logging.info("Selecting configurations using Farthest Point Sampling")
        fps_pt = FPS(atoms_list_pt_avail, args.num_samples - len(atoms_list_pt_filtered))
        idx_pt = fps_pt.run()
        logging.info(f"Selected {len(idx_pt)} configurations")
        atoms_list_pt_extra = [atoms_list_pt_avail[i] for i in idx_pt]

    atoms_list_pt = atoms_list_pt_filtered + atoms_list_pt_extra

    for atoms in atoms_list_pt:
        # del atoms.info["mace_descriptors"]
        atoms.info["pretrained"] = True
        atoms.info["config_weight"] = args.weight_pt
        atoms.info["mace_descriptors"] = None
        if args.head_pt is not None:
            atoms.info["head"] = args.head_pt

    logging.info("Saving the selected configurations")
    ase.io.write(args.output, atoms_list_pt, format="extxyz")
    logging.info("Saving a combined XYZ file")
    for atoms in atoms_list_ft:
        atoms.info["pretrained"] = False
        atoms.info["config_weight"] = args.weight_ft
        atoms.info["mace_descriptors"] = None
        if args.head_ft is not None:
            atoms.info["head"] = args.head_ft
    atoms_fps_pt_ft = atoms_list_pt + atoms_list_ft
    ase.io.write(
        args.output.replace(".xyz", "_combined.xyz"), atoms_fps_pt_ft, format="extxyz"
    )


def main():
    args = parse_args()
    select_samples(args)


if __name__ == "__main__":
    main()
