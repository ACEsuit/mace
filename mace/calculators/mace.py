###########################################################################################
# The ASE Calculator for MACE (based on https://github.com/mir-group/nequip)
# Authors: Ilyes Batatia, David Kovacs
# This program is distributed under the MIT License (see MIT.md)
###########################################################################################


from glob import glob
from typing import Union
from pathlib import Path

import numpy as np
import torch
from ase.calculators.calculator import Calculator, all_changes
from ase.stress import full_3x3_to_voigt_6_stress

from mace import data
from mace.tools import torch_geometric, torch_tools, utils


class MACECalculator(Calculator):
    """MACE ASE Calculator
    args:
        model_paths: str, path to model or models if a committee is produced
                to make a committee use a wild card notation like mace_*.model
        device: str, device to run on (cuda or cpu)
        energy_units_to_eV: float, conversion factor from model energy units to eV
        length_units_to_A: float, conversion factor from model length units to Angstroms
        default_dtype: str, default dtype of model
        model_type: str, type of model to load
                    Options: [MACE, DipoleMACE, EnergyDipoleMACE]
    """

    def __init__(
        self,
        model_paths: Union[list, str],
        device: str,
        energy_units_to_eV: float = 1.0,
        length_units_to_A: float = 1.0,
        default_dtype="float64",
        model_type="MACE",
        **kwargs,
    ):
        Calculator.__init__(self, **kwargs)
        self.results = {}

        self.model_type = model_type

        if model_type == 'MACE':
            self.implemented_properties = ["energy", "free_energy", "forces", "stress", "energies"]
        elif model_type == 'DipoleMACE':
            self.implemented_properties = ["dipole"]
        elif model_type == 'EnergyDipoleMACE':
            self.implemented_properties = ["energy", "free_energy", "forces", "stress", "energies", "dipole"]
        else:
            raise ValueError(f"Give a valid model_type: [MACE, DipoleMACE, EnergyDipoleMACE], {model_type} not supported")

        if isinstance(model_paths, str):
            # Find all models that staisfy the wildcard (e.g. mace_model_*.pt)
            model_paths_glob = glob(model_paths)
            if len(model_paths_glob) == 0:
                raise ValueError(f"Couldn't find MACE model files: {model_paths}")
            model_paths = model_paths_glob
        elif isinstance(model_paths, Path):
            model_paths = [model_paths]
        if len(model_paths) == 0:
            raise ValueError("No mace file neames supplied")
        self.num_models = len(model_paths)
        if len(model_paths) > 1:
            print(f"Running committee mace with {len(model_paths)} models")

        self.models = [
            torch.load(f=model_path, map_location=device) for model_path in model_paths
        ]
        r_maxs = [model.r_max.cpu() for model in self.models]
        r_maxs = np.array(r_maxs)
        assert np.all(
            r_maxs == r_maxs[0]
        ), "committee r_max are not all the same {' '.join(r_maxs)}"
        self.r_max = r_maxs[0]

        self.device = torch_tools.init_device(device)
        self.energy_units_to_eV = energy_units_to_eV
        self.length_units_to_A = length_units_to_A
        self.z_table = utils.AtomicNumberTable(
            [int(z) for z in self.models[0].atomic_numbers]
        )

        torch_tools.set_default_dtype(default_dtype)
        for model in self.models:
            for param in model.parameters():
                param.requires_grad = False

    # pylint: disable=dangerous-default-value
    def calculate(self, atoms=None, properties=None, system_changes=all_changes):
        """
        Calculate properties.
        :param atoms: ase.Atoms object
        :param properties: [str], properties to be computed, used by ASE internally
        :param system_changes: [str], system changes since last calculation, used by ASE internally
        :return:
        """
        # call to base-class to set atoms attribute
        Calculator.calculate(self, atoms)

        # prepare data
        config = data.config_from_atoms(atoms)
        data_loader = torch_geometric.dataloader.DataLoader(
            dataset=[
                data.AtomicData.from_config(
                    config, z_table=self.z_table, cutoff=self.r_max
                )
            ],
            batch_size=1,
            shuffle=False,
            drop_last=False,
        )

        # predict + extract data
        outputs = {}
        for prop in self.implemented_properties:
            outputs[prop] = []
        
        if self.model_type == 'MACE' or self.model_type == 'EnergyDipoleMACE':
            batch = next(iter(data_loader)).to(self.device)
            node_e0 = self.models[0].atomic_energies_fn(batch["node_attrs"])
            outputs["node_energies"] = []
            compute_stress = True
        else:
            compute_stress = False
        
        for model in self.models:
            batch = next(iter(data_loader)).to(self.device)

            out = model(batch.to_dict(), compute_stress=compute_stress)
            if self.model_type == 'MACE' or self.model_type == 'EnergyDipoleMACE':
                outputs["energy"].append(out["energy"].detach().cpu().item())
                outputs["free_energy"].append(out["energy"].detach().cpu().item())
                outputs["forces"].append(out["forces"].detach().cpu().numpy())
                outputs["node_energies"].append((out["node_energy"] - node_e0).detach().cpu().numpy())
                if out["stress"] is not None:
                    outputs["stress"].append(out["stress"].detach().cpu().numpy())
            if self.model_type == 'DipoleMACE' or self.model_type == 'EnergyDipoleMACE':
                outputs["dipole"].append(out["dipole"].detach().cpu().numpy()[0])

        self.results = {}
        # convert units
        if self.model_type == 'MACE' or self.model_type == 'EnergyDipoleMACE':
            energies = np.array(outputs["energy"]) * self.energy_units_to_eV
            self.results["energy"] = np.mean(energies)
            self.results["free_energy"] = self.results["energy"]
            forces = np.array(outputs["forces"]) * (self.energy_units_to_eV / self.length_units_to_A)
            self.results["forces"] = np.mean(forces, axis=0)
            self.results["node_energy"] = np.mean(np.array(outputs["node_energies"]), axis=0) * self.energy_units_to_eV
            if len(outputs["stress"]) > 0:
                stress = np.mean(np.array(outputs["stress"]), axis=0) * (self.energy_units_to_eV / self.length_units_to_A**3)
                self.results["stress"] = full_3x3_to_voigt_6_stress(stress)[0]
            if self.num_models > 1:
                self.results["energies"] = energies
                self.results["energy_var"] = np.var(energies)
                self.results["forces_var"] = np.var(forces, axis=0)
        if self.model_type == 'DipoleMACE' or self.model_type == 'EnergyDipoleMACE':
            self.results["dipole"] = np.mean(np.array(outputs["dipole"]), axis=0)
            if self.num_models > 1: 
                self.results["dipole_var"] = np.var(np.array(outputs["dipole"]), axis=0)


class DipoleMACECalculator(Calculator):
    """MACE ASE Calculator for predicting dipoles"""

    implemented_properties = [
        "dipole",
    ]

    def __init__(
        self,
        model_path: str,
        device: str,
        length_units_to_A: float = 1.0,
        default_dtype="float64",
        charges_key="Qs",
        **kwargs,
    ):
        """
        :param charges_key: str, Array field of atoms object where atomic charges are stored
        """
        Calculator.__init__(self, **kwargs)
        self.results = {}

        self.model = torch.load(f=model_path, map_location=device)
        self.r_max = self.model.r_max
        self.device = torch_tools.init_device(device)
        self.length_units_to_A = length_units_to_A
        self.z_table = utils.AtomicNumberTable(
            [int(z) for z in self.model.atomic_numbers]
        )
        self.charges_key = charges_key

        torch_tools.set_default_dtype(default_dtype)

    # pylint: disable=dangerous-default-value
    def calculate(self, atoms=None, properties=None, system_changes=all_changes):
        """
        Calculate properties.
        :param atoms: ase.Atoms object
        :param properties: [str], properties to be computed, used by ASE internally
        :param system_changes: [str], system changes since last calculation, used by ASE internally
        :return:
        """
        # call to base-class to set atoms attribute
        Calculator.calculate(self, atoms)

        # prepare data
        config = data.config_from_atoms(atoms, charges_key=self.charges_key)
        data_loader = torch_geometric.dataloader.DataLoader(
            dataset=[
                data.AtomicData.from_config(
                    config, z_table=self.z_table, cutoff=self.r_max
                )
            ],
            batch_size=1,
            shuffle=False,
            drop_last=False,
        )
        batch = next(iter(data_loader)).to(self.device)

        # predict + extract data
        out = self.model(batch)
        dipole = out["dipole"].detach().cpu().numpy()

        # store results
        self.results = {
            "dipole": dipole,
        }


class EnergyDipoleMACECalculator(Calculator):
    """MACE ASE Calculator for predicting energies, forces and dipoles"""

    implemented_properties = [
        "energy",
        "free_energy",
        "forces",
        "stress",
        "dipole",
    ]

    def __init__(
        self,
        model_path: str,
        device: str,
        energy_units_to_eV: float = 1.0,
        length_units_to_A: float = 1.0,
        default_dtype="float64",
        charges_key="Qs",
        **kwargs,
    ):
        """
        :param charges_key: str, Array field of atoms object where atomic charges are stored
        """
        Calculator.__init__(self, **kwargs)
        self.results = {}

        self.model = torch.load(f=model_path, map_location=device)
        self.r_max = self.model.r_max
        self.device = torch_tools.init_device(device)
        self.energy_units_to_eV = energy_units_to_eV
        self.length_units_to_A = length_units_to_A
        self.z_table = utils.AtomicNumberTable(
            [int(z) for z in self.model.atomic_numbers]
        )
        self.charges_key = charges_key

        torch_tools.set_default_dtype(default_dtype)

    # pylint: disable=dangerous-default-value
    def calculate(self, atoms=None, properties=None, system_changes=all_changes):
        """
        Calculate properties.
        :param atoms: ase.Atoms object
        :param properties: [str], properties to be computed, used by ASE internally
        :param system_changes: [str], system changes since last calculation, used by ASE internally
        :return:
        """
        # call to base-class to set atoms attribute
        Calculator.calculate(self, atoms)

        # prepare data
        config = data.config_from_atoms(atoms, charges_key=self.charges_key)
        data_loader = torch_geometric.dataloader.DataLoader(
            dataset=[
                data.AtomicData.from_config(
                    config, z_table=self.z_table, cutoff=self.r_max
                )
            ],
            batch_size=1,
            shuffle=False,
            drop_last=False,
        )
        batch = next(iter(data_loader)).to(self.device)

        # predict + extract data
        out = self.model(batch, compute_stress=True)
        energy = out["energy"].detach().cpu().item()
        forces = out["forces"].detach().cpu().numpy()
        dipole = out["dipole"].detach().cpu().numpy()

        # store results
        E = energy * self.energy_units_to_eV
        self.results = {
            "energy": E,
            "free_energy": E,
            # force has units eng / len:
            "forces": forces * (self.energy_units_to_eV / self.length_units_to_A),
            # stress has units eng / len:
            "dipole": dipole,
        }

        # even though compute_stress is True, stress can be none if pbc is False
        # not sure if correct ASE thing is to have no dict key, or dict key with value None
        if out["stress"] is not None:
            stress = out["stress"].detach().cpu().numpy()
            self.results["stress"] = (
                stress * (self.energy_units_to_eV / self.length_units_to_A**3)
            )[0]


class MACECommitteeCalculator(Calculator):
    """MACE ASE Committee Calculator
    This calculator can be used to obtain energy and force errors from a committee of
    MACE models. To load multiple committees, either pass a list of file names or a
    single string with a wildcard to the model_paths argument. This calculator can also
    be used to load single models.
    """

    implemented_properties = ["energy", "free_energy", "forces", "stress"]

    def __init__(
        self,
        model_paths: Union[list, str],
        device: str,
        energy_units_to_eV: float = 1.0,
        length_units_to_A: float = 1.0,
        default_dtype="float64",
        **kwargs,
    ):
        Calculator.__init__(self, **kwargs)
        self.results = {}

        if isinstance(model_paths, str):
            # Find all models that staisfy the wildcard (e.g. mace_model_*.pt)
            model_paths_glob = glob(model_paths)

            if len(model_paths_glob) == 0:
                raise ValueError(f"Couldn't find MACE model files: {model_paths}")
            model_paths = model_paths_glob
        if len(model_paths) == 0:
            raise ValueError("No mace file neames supplied")
        if len(model_paths) > 1:
            print(f"Running committee mace with {len(model_paths)} models")

        # Load models
        self.models = [
            torch.load(f=model_path, map_location=device) for model_path in model_paths
        ]

        r_maxs = [model.r_max.cpu() for model in self.models]
        r_maxs = np.array(r_maxs)
        assert np.all(
            r_maxs == r_maxs[0]
        ), "committee r_max are not all the same {' '.join(r_maxs)}"
        self.r_max = r_maxs[0]

        self.device = torch_tools.init_device(device)
        self.energy_units_to_eV = energy_units_to_eV
        self.length_units_to_A = length_units_to_A
        self.z_table = utils.AtomicNumberTable(
            [int(z) for z in self.models[0].atomic_numbers]
        )

        torch_tools.set_default_dtype(default_dtype)

    # pylint: disable=dangerous-default-value
    def calculate(self, atoms=None, properties=None, system_changes=all_changes):
        """
        Calculate properties.
        :param atoms: ase.Atoms object
        :param properties: [str], properties to be computed, used by ASE internally
        :param system_changes: [str], system changes since last calculation, used by
        ASE internally
        :return:
        """
        # call to base-class to set atoms attribute
        Calculator.calculate(self, atoms)

        # prepare data
        config = data.config_from_atoms(atoms)
        data_loader = torch_geometric.dataloader.DataLoader(
            dataset=[
                data.AtomicData.from_config(
                    config, z_table=self.z_table, cutoff=self.r_max
                )
            ],
            batch_size=1,
            shuffle=False,
            drop_last=False,
        )
        batch = next(iter(data_loader)).to(self.device)

        # predict + extract data
        energies, forces = [], []
        for _, model in enumerate(self.models):
            # Otherwise: RuntimeError: you can only change requires_grad flags of leaf variables.
            batch = next(iter(data_loader)).to(self.device)
            out = model(batch, compute_stress=True)
            energies.append(out["energy"].detach().cpu().item())
            forces.append(out["forces"].detach().cpu().numpy())

        # convert_units
        energies = np.array(energies) * self.energy_units_to_eV
        # force has units eng / len:
        forces = np.array(forces) * self.energy_units_to_eV / self.length_units_to_A
        # store results
        self.results = {
            "energies": energies,
            "forcess": forces,
            "energy": np.mean(energies),
            "free_energy": np.mean(energies),
            "energy_var": np.var(energies),
            "forces": np.mean(forces, axis=0),
            "forces_var": np.var(forces, axis=0),
        }
