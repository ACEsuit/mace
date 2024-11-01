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

try:
    import fpsample
except ImportError:
    pass


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--configs_pt",
        help="path to XYZ configurations for the pretraining",
        required=True,
    )
    parser.add_argument(
        "--configs_ft",
        help="path or list of paths to XYZ configurations for the finetuning",
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
        "--subselect",
        help="method to subselect the configurations of the pretraining set",
        type=str,
        choices=["fps", "random"],
        default="fps",
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
        choices=[None, "combinations", "exclusive", "inclusive"],
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


def calculate_descriptors(atoms: List[ase.Atoms], calc: MACECalculator) -> None:
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
    atoms: ase.Atoms, element_subset: List[str], filtering_type: str
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
    if filtering_type == "combinations":
        atom_symbols = np.unique(atoms.symbols)
        return all(
            x in element_subset for x in atom_symbols
        )  # atoms must *only* contain elements in the subset
    if filtering_type == "exclusive":
        atom_symbols = set(list(atoms.symbols))
        return atom_symbols == set(element_subset)
    if filtering_type == "inclusive":
        atom_symbols = np.unique(atoms.symbols)
        return all(
            x in atom_symbols for x in element_subset
        )  # atoms must *at least* contain elements in the subset
    raise ValueError(
        f"Filtering type {filtering_type} not recognised. Must be one of 'none', 'exclusive', or 'inclusive'."
    )


class FPS:
    def __init__(self, atoms_list: List[ase.Atoms], n_samples: int):
        self.n_samples = n_samples
        self.atoms_list = atoms_list
        self.species = np.unique([x.symbol for atoms in atoms_list for x in atoms])
        self.species_dict = {x: i for i, x in enumerate(self.species)}
        # start from a random configuration
        self.list_index = [np.random.randint(0, len(atoms_list))]
        self.assemble_descriptors()

    def run(
        self,
    ) -> List[int]:
        """
        Run the farthest point sampling algorithm.
        """
        descriptor_dataset_reshaped = (
            self.descriptors_dataset.reshape(  # pylint: disable=E1121
                (len(self.atoms_list), -1)
            )
        )
        logging.info(f"{descriptor_dataset_reshaped.shape}")
        logging.info(f"n_samples: {self.n_samples}")
        self.list_index = fpsample.fps_npdu_kdtree_sampling(
            descriptor_dataset_reshaped,
            self.n_samples,
        )
        return self.list_index

    def assemble_descriptors(self) -> np.ndarray:
        """
        Assemble the descriptors for all the configurations.
        """
        self.descriptors_dataset: np.ndarray = 10e10 * np.ones(
            (
                len(self.atoms_list),
                len(self.species),
                len(list(self.atoms_list[0].info["mace_descriptors"].values())[0]),
            ),
            dtype=np.float32,
        ).astype(np.float32)

        for i, atoms in enumerate(self.atoms_list):
            descriptors = atoms.info["mace_descriptors"]
            for z in descriptors:
                self.descriptors_dataset[i, self.species_dict[z]] = np.array(
                    descriptors[z]
                ).astype(np.float32)


def select_samples(
    args: argparse.Namespace,
) -> None:
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.model in ["small", "medium", "large"]:
        calc = mace_mp(args.model, device=args.device, default_dtype=args.default_dtype)
    else:
        calc = MACECalculator(
            model_paths=args.model, device=args.device, default_dtype=args.default_dtype
        )
    if isinstance(args.configs_ft, str):
        atoms_list_ft = ase.io.read(args.configs_ft, index=":")
    else:
        atoms_list_ft = []
        for path in args.configs_ft:
            atoms_list_ft += ase.io.read(path, index=":")

    if args.filtering_type is not None:
        all_species_ft = np.unique([x.symbol for atoms in atoms_list_ft for x in atoms])
        logging.info(
            "Filtering configurations based on the finetuning set, "
            f"filtering type: combinations, elements: {all_species_ft}"
        )
        if args.subselect != "random":
            if args.descriptors is not None:
                logging.info("Loading descriptors")
                descriptors = np.load(args.descriptors, allow_pickle=True)
                atoms_list_pt = ase.io.read(args.configs_pt, index=":")
                for i, atoms in enumerate(atoms_list_pt):
                    atoms.info["mace_descriptors"] = descriptors[i]
                atoms_list_pt_filtered = [
                    x
                    for x in atoms_list_pt
                    if filter_atoms(x, all_species_ft, "combinations")
                ]
            else:
                atoms_list_pt = ase.io.read(args.configs_pt, index=":")
                atoms_list_pt_filtered = [
                    x
                    for x in atoms_list_pt
                    if filter_atoms(x, all_species_ft, "combinations")
                ]
        else:
            atoms_list_pt = ase.io.read(args.configs_pt, index=":")
            atoms_list_pt_filtered = [
                x
                for x in atoms_list_pt
                if filter_atoms(x, all_species_ft, "combinations")
            ]
        if len(atoms_list_pt_filtered) <= args.num_samples:
            logging.info(
                f"Number of configurations after filtering {len(atoms_list_pt_filtered)} "
                f"is less than the number of samples {args.num_samples}, "
                "selecting random configurations for the rest."
            )
            atoms_list_pt_minus_filtered = [
                x for x in atoms_list_pt if x not in atoms_list_pt_filtered
            ]
            atoms_list_pt_random_inds = np.random.choice(
                list(range(len(atoms_list_pt_minus_filtered))),
                args.num_samples - len(atoms_list_pt_filtered),
                replace=False,
            )
            atoms_list_pt = atoms_list_pt_filtered + [
                atoms_list_pt_minus_filtered[ind] for ind in atoms_list_pt_random_inds
            ]
        else:
            atoms_list_pt = atoms_list_pt_filtered

    else:
        atoms_list_pt = ase.io.read(args.configs_pt, index=":")
        if args.descriptors is not None:
            logging.info(
                f"Loading descriptors for the pretraining set from {args.descriptors}"
            )
            descriptors = np.load(args.descriptors, allow_pickle=True)
            for i, atoms in enumerate(atoms_list_pt):
                atoms.info["mace_descriptors"] = descriptors[i]

    if args.num_samples is not None and args.num_samples < len(atoms_list_pt):
        if args.subselect == "fps":
            if args.descriptors is None:
                logging.info("Calculating descriptors for the pretraining set")
                calculate_descriptors(atoms_list_pt, calc)
                descriptors_list = [
                    atoms.info["mace_descriptors"] for atoms in atoms_list_pt
                ]
                logging.info(
                    f"Saving descriptors at {args.output.replace('.xyz', '_descriptors.npy')}"
                )
                np.save(
                    args.output.replace(".xyz", "_descriptors.npy"), descriptors_list
                )
            logging.info("Selecting configurations using Farthest Point Sampling")
            try:
                fps_pt = FPS(atoms_list_pt, args.num_samples)
                idx_pt = fps_pt.run()
                logging.info(f"Selected {len(idx_pt)} configurations")
            except Exception as e:  # pylint: disable=W0703
                logging.error(
                    f"FPS failed, selecting random configurations instead: {e}"
                )
                idx_pt = np.random.choice(
                    list(range(len(atoms_list_pt))), args.num_samples, replace=False
                )
            atoms_list_pt = [atoms_list_pt[i] for i in idx_pt]
        else:
            logging.info("Selecting random configurations")
            idx_pt = np.random.choice(
                list(range(len(atoms_list_pt))), args.num_samples, replace=False
            )
            atoms_list_pt = [atoms_list_pt[i] for i in idx_pt]
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
