###########################################################################################
# This program is distributed under the MIT License (see MIT.md)
###########################################################################################
from __future__ import annotations

import argparse
import ast
import logging
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import List, Tuple, Union

import ase.data
import ase.io
import numpy as np
import torch

from mace.calculators import MACECalculator, mace_mp

try:
    import fpsample  # type: ignore
except ImportError:
    pass


class FilteringType(Enum):
    NONE = "none"
    COMBINATIONS = "combinations"
    EXCLUSIVE = "exclusive"
    INCLUSIVE = "inclusive"


class SubselectType(Enum):
    FPS = "fps"
    RANDOM = "random"


@dataclass
class SelectionSettings:
    configs_pt: str
    output: str
    configs_ft: str | None = None
    atomic_numbers: List[int] | None = None
    num_samples: int | None = None
    subselect: SubselectType = SubselectType.FPS
    model: str = "small"
    descriptors: str | None = None
    device: str = "cpu"
    default_dtype: str = "float64"
    head_pt: str | None = None
    head_ft: str | None = None
    filtering_type: FilteringType = FilteringType.COMBINATIONS
    weight_ft: float = 1.0
    weight_pt: float = 1.0
    allow_random_padding: bool = True
    seed: int = 42


def str_to_list(s: str) -> List[int]:
    assert isinstance(s, str), "Input must be a string"
    return ast.literal_eval(s)


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
        required=False,
        default=None,
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
        type=SubselectType,
        choices=list(SubselectType),
        default=SubselectType.FPS,
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
        type=FilteringType,
        choices=list(FilteringType),
        default=FilteringType.NONE,
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
    parser.add_argument(
        "--atomic_numbers",
        help="list of atomic numbers to filter the configurations",
        type=str_to_list,
        default=None,
    )
    parser.add_argument(
        "--disallow_random_padding",
        help="do not allow random padding of the configurations to match the number of samples",
        action="store_false",
        dest="allow_random_padding",
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
    atoms: ase.Atoms,
    element_subset: List[str],
    filtering_type: FilteringType = FilteringType.COMBINATIONS,
) -> bool:
    """
    Filters atoms based on the provided filtering type and element subset.

    Parameters:
    atoms (ase.Atoms): The atoms object to filter.
    element_subset (list): The list of elements to consider during filtering.
    filtering_type (FilteringType): The type of filtering to apply.
        Can be one of the following `FilteringType` enum members:
          - `FilteringType.NONE`: No filtering is applied.
          - `FilteringType.COMBINATIONS`: Return true if `atoms` is composed of combinations of elements in the subset, false otherwise. I.e. does not require all of the specified elements to be present.
          - `FilteringType.EXCLUSIVE`: Return true if `atoms` contains *only* elements in the subset, false otherwise.
          - `FilteringType.INCLUSIVE`: Return true if `atoms` contains all elements in the subset, false otherwise. I.e. allows additional elements.

    Returns:
    bool: True if the atoms pass the filter, False otherwise.
    """
    if filtering_type == FilteringType.NONE:
        return True
    if filtering_type == FilteringType.COMBINATIONS:
        atom_symbols = np.unique(atoms.symbols)
        return all(
            x in element_subset for x in atom_symbols
        )  # atoms must *only* contain elements in the subset
    if filtering_type == FilteringType.EXCLUSIVE:
        atom_symbols = set(list(atoms.symbols))
        return atom_symbols == set(element_subset)
    if filtering_type == FilteringType.INCLUSIVE:
        atom_symbols = np.unique(atoms.symbols)
        return all(
            x in atom_symbols for x in element_subset
        )  # atoms must *at least* contain elements in the subset
    raise ValueError(
        f"Filtering type {filtering_type} not recognised. Must be one of {list(FilteringType)}."
    )


class FPS:
    def __init__(self, atoms_list: List[ase.Atoms], n_samples: int):
        self.n_samples = n_samples
        self.atoms_list = atoms_list
        self.species = np.unique([x.symbol for atoms in atoms_list for x in atoms])  # type: ignore
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

    def assemble_descriptors(self) -> None:
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


def _load_calc(
    model: str, device: str, default_dtype: str, subselect: SubselectType
) -> Union[MACECalculator, None]:
    if subselect == SubselectType.RANDOM:
        return None
    if model in ["small", "medium", "large"]:
        calc = mace_mp(model, device=device, default_dtype=default_dtype)
    else:
        calc = MACECalculator(
            model_paths=model,
            device=device,
            default_dtype=default_dtype,
        )
    return calc


def _get_finetuning_elements(
    atoms: List[ase.Atoms], atomic_numbers: List[int] | None
) -> List[str]:
    if atoms:
        logging.debug(
            "Using elements from the finetuning configurations for filtering."
        )
        species = np.unique([x.symbol for atoms in atoms for x in atoms]).tolist()  # type: ignore
    elif atomic_numbers is not None and atomic_numbers:
        logging.debug("Using the supplied atomic numbers for filtering.")
        species = [ase.data.chemical_symbols[z] for z in atomic_numbers]
    else:
        species = []
    return species


def _read_finetuning_configs(
    configs_ft: Union[str, list[str], None],
) -> List[ase.Atoms]:
    if isinstance(configs_ft, str):
        path = configs_ft
        return ase.io.read(path, index=":")  # type: ignore
    if isinstance(configs_ft, list):
        assert all(isinstance(x, str) for x in configs_ft)
        atoms_list_ft = []
        for path in configs_ft:
            atoms_list_ft += ase.io.read(path, index=":")
        return atoms_list_ft
    if configs_ft is None:
        return []
    raise ValueError(f"Invalid type for configs_ft: {type(configs_ft)}")


def _filter_pretraining_data(
    atoms: list[ase.Atoms],
    filtering_type: FilteringType,
    all_species_ft: List[str],
) -> Tuple[List[ase.Atoms], List[ase.Atoms], list[bool]]:
    logging.info(
        "Filtering configurations based on the finetuning set, "
        f"filtering type: {filtering_type}, elements: {all_species_ft}"
    )
    passes_filter = [filter_atoms(x, all_species_ft, filtering_type) for x in atoms]
    assert len(passes_filter) == len(atoms), "Filtering failed"
    filtered_atoms = [x for x, passes in zip(atoms, passes_filter) if passes]
    remaining_atoms = [x for x, passes in zip(atoms, passes_filter) if not passes]
    return filtered_atoms, remaining_atoms, passes_filter


def _get_random_configs(
    num_samples: int,
    atoms: List[ase.Atoms],
) -> list[ase.Atoms]:
    if num_samples > len(atoms):
        raise ValueError(
            f"Requested more samples ({num_samples}) than available in the remaining set ({len(atoms)})"
        )
    indices = np.random.choice(list(range(len(atoms))), num_samples, replace=False)
    return [atoms[i] for i in indices]


def _load_descriptors(
    atoms: List[ase.Atoms],
    passes_filter: List[bool],
    descriptors_path: str | None,
    calc: MACECalculator | None,
    full_data_length: int,
) -> None:
    if descriptors_path is not None:
        logging.info(f"Loading descriptors from {descriptors_path}")
        descriptors = np.load(descriptors_path, allow_pickle=True)
        assert sum(passes_filter) == len(atoms)
        if len(descriptors) != full_data_length:
            raise ValueError(
                f"Length of the descriptors ({len(descriptors)}) does not match the length of the data ({full_data_length})"
                "Please provide descriptors for all configurations"
            )
        required_descriptors = [
            descriptors[i] for i, passes in enumerate(passes_filter) if passes
        ]
        for i, atoms_ in enumerate(atoms):
            atoms_.info["mace_descriptors"] = required_descriptors[i]
    else:
        logging.info("Calculating descriptors")
        if calc is None:
            raise ValueError("MACECalculator must be provided to calculate descriptors")
        calculate_descriptors(atoms, calc)


def _maybe_save_descriptors(
    atoms: List[ase.Atoms],
    output_path: str,
) -> None:
    """
    Save the descriptors if they are present in the atoms objects.
    Also, delete the descriptors from the atoms objects.
    """
    if all("mace_descriptors" in x.info for x in atoms):
        output_path = Path(output_path)
        descriptor_save_path = output_path.parent / (
            output_path.stem + "_descriptors.npy"
        )
        logging.info(f"Saving descriptors at {descriptor_save_path}")
        descriptors_list = [x.info["mace_descriptors"] for x in atoms]
        np.save(descriptor_save_path, descriptors_list, allow_pickle=True)
        for x in atoms:
            del x.info["mace_descriptors"]


def _maybe_fps(atoms: List[ase.Atoms], num_samples: int) -> List[ase.Atoms]:
    try:
        fps_pt = FPS(atoms, num_samples)
        idx_pt = fps_pt.run()
        logging.info(f"Selected {len(idx_pt)} configurations")
        return [atoms[i] for i in idx_pt]
    except Exception as e:  # pylint: disable=W0703
        logging.error(f"FPS failed, selecting random configurations instead: {e}")
        return _get_random_configs(num_samples, atoms)


def _subsample_data(
    filtered_atoms: List[ase.Atoms],
    remaining_atoms: List[ase.Atoms],
    passes_filter: List[bool],
    num_samples: int | None,
    subselect: SubselectType,
    descriptors_path: str | None,
    allow_random_padding: bool,
    calc: MACECalculator | None,
) -> List[ase.Atoms]:
    if num_samples is None or num_samples == len(filtered_atoms):
        logging.info(
            f"No subsampling, keeping all {len(filtered_atoms)} filtered configurations"
        )
        return filtered_atoms
    if num_samples > len(filtered_atoms) and allow_random_padding:
        num_sample_randomly = num_samples - len(filtered_atoms)
        logging.info(
            f"Number of configurations after filtering {len(filtered_atoms)} "
            f"is less than the number of samples {num_samples}, "
            f"selecting {num_sample_randomly} random configurations for the rest."
        )
        return filtered_atoms + _get_random_configs(
            num_sample_randomly, remaining_atoms
        )
    if num_samples == 0:
        raise ValueError("Number of samples must be greater than 0")
    if subselect == SubselectType.FPS:
        _load_descriptors(
            filtered_atoms,
            passes_filter,
            descriptors_path,
            calc,
            full_data_length=len(filtered_atoms) + len(remaining_atoms),
        )
        logging.info(
            f"Selecting {num_samples} configurations out of {len(filtered_atoms)} using Farthest Point Sampling"
        )
        return _maybe_fps(filtered_atoms, num_samples)
    if subselect == SubselectType.RANDOM:
        logging.info(
            f"Subselecting {num_samples} from filtered {len(filtered_atoms)} using random sampling"
        )
        return _get_random_configs(num_samples, filtered_atoms)
    raise ValueError(f"Invalid subselect type: {subselect}")


def _write_metadata(
    atoms: list[ase.Atoms], pretrained: bool, config_weight: float, head: str | None
) -> None:
    for a in atoms:
        a.info["pretrained"] = pretrained
        a.info["config_weight"] = config_weight
        if head is not None:
            a.info["head"] = head


def select_samples(
    settings: SelectionSettings,
) -> None:
    np.random.seed(settings.seed)
    torch.manual_seed(settings.seed)
    calc = _load_calc(
        settings.model, settings.device, settings.default_dtype, settings.subselect
    )
    atoms_list_ft = _read_finetuning_configs(settings.configs_ft)
    all_species_ft = _get_finetuning_elements(atoms_list_ft, settings.atomic_numbers)

    if settings.filtering_type is not FilteringType.NONE and not all_species_ft:
        raise ValueError(
            "Filtering types other than NONE require elements for filtering. They can be specified via the `--atomic_numbers` flag."
        )

    atoms_list_pt: list[ase.Atoms] = ase.io.read(settings.configs_pt, index=":")  # type: ignore
    filtered_pt_atoms, remaining_atoms, passes_filter = _filter_pretraining_data(
        atoms_list_pt, settings.filtering_type, all_species_ft
    )

    subsampled_atoms = _subsample_data(
        filtered_pt_atoms,
        remaining_atoms,
        passes_filter,
        settings.num_samples,
        settings.subselect,
        settings.descriptors,
        settings.allow_random_padding,
        calc,
    )
    if ase.io.formats.filetype(settings.output, read=False) != "extxyz":
        raise ValueError(
            f"filename '{settings.output}' does no have "
            "suffix compatible with extxyz format"
        )
    _maybe_save_descriptors(subsampled_atoms, settings.output)

    _write_metadata(
        subsampled_atoms,
        pretrained=True,
        config_weight=settings.weight_pt,
        head=settings.head_pt,
    )
    _write_metadata(
        atoms_list_ft,
        pretrained=False,
        config_weight=settings.weight_ft,
        head=settings.head_ft,
    )

    logging.info("Saving the selected configurations")
    ase.io.write(settings.output, subsampled_atoms)

    logging.info("Saving a combined XYZ file")
    atoms_fps_pt_ft = subsampled_atoms + atoms_list_ft

    output = Path(settings.output)
    ase.io.write(
        output.parent / (output.stem + "_combined" + output.suffix),
        atoms_fps_pt_ft,
    )


def main():
    args = parse_args()
    settings = SelectionSettings(**vars(args))
    select_samples(settings)


if __name__ == "__main__":
    main()
