###########################################################################################
# The ASE Calculator for MACE (based on https://github.com/mir-group/nequip)
# Authors: Ilyes Batatia, David Kovacs
# This program is distributed under the MIT License (see MIT.md)
###########################################################################################


import torch
from ase.calculators.calculator import Calculator, all_changes
from ase.stress import full_3x3_to_voigt_6_stress

from mace import data
from mace.tools import torch_geometric, torch_tools, utils


class MACECalculator(Calculator):
    """MACE ASE Calculator"""

    implemented_properties = ["energy", "free_energy", "forces", "stress"]

    def __init__(
        self,
        model_path: str,
        device: str,
        energy_units_to_eV: float = 1.0,
        length_units_to_A: float = 1.0,
        default_dtype="float64",
        **kwargs
    ):
        Calculator.__init__(self, **kwargs)
        self.results = {}

        self.model = torch.load(f=model_path, map_location=device)
        self.r_max = float(self.model.r_max)
        self.device = torch_tools.init_device(device)
        self.energy_units_to_eV = energy_units_to_eV
        self.length_units_to_A = length_units_to_A
        self.z_table = utils.AtomicNumberTable(
            [int(z) for z in self.model.atomic_numbers]
        )

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
        out = self.model(batch.to_dict(), compute_stress=True)
        energy = out["energy"].detach().cpu().item()
        forces = out["forces"].detach().cpu().numpy()

        # store results
        E = energy * self.energy_units_to_eV
        self.results = {
            "energy": E,
            "free_energy": E,
            # force has units eng / len:
            "forces": forces * (self.energy_units_to_eV / self.length_units_to_A),
        }

        # even though compute_stress is True, stress can be none if pbc is False
        # not sure if correct ASE thing is to have no dict key, or dict key with value None
        if out["stress"] is not None:
            stress = out["stress"].detach().cpu().numpy()
            # stress has units eng / len^3:
            self.results["stress"] = (
                stress * (self.energy_units_to_eV / self.length_units_to_A**3)
            )[0]
            self.results["stress"] = full_3x3_to_voigt_6_stress(self.results["stress"])


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
        **kwargs
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
        **kwargs
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
