from typing import Union
import torch

from ase.calculators.calculator import Calculator, all_changes

import os
import ase
import numpy as np

from LieACE import data

from ase import units
from ase.calculators.calculator import Calculator, all_changes
from ase.io import read, write
from ase.io.trajectory import Trajectory
from ase.io.xyz import read_xyz, write_xyz
from ase.md import VelocityVerlet, Langevin, MDLogger
from ase.md.velocitydistribution import (
    MaxwellBoltzmannDistribution,
    Stationary,
    ZeroRotation,
)
from ase.optimize import QuasiNewton
from ase.vibrations import Vibrations


class MultiACECalculator(Calculator):
    """NequIP ASE Calculator."""

    implemented_properties = ["energy", "forces"]

    def __init__(self,
                 model: torch.jit.ScriptModule,
                 r_max: float,
                 device: Union[str, torch.device],
                 atomic_numbers: float,
                 energy_units_to_eV: float = 1.0,
                 length_units_to_A: float = 1.0,
                 **kwargs):
        Calculator.__init__(self, **kwargs)
        self.results = {}

        self.model = model
        self.r_max = r_max
        self.device = device
        self.energy_units_to_eV = energy_units_to_eV
        self.length_units_to_A = length_units_to_A
        self.z_table = data.AtomicNumberTable([int(z) for z in atomic_numbers])

    def calculate(self, atoms=None, properties=["energy"], system_changes=all_changes):
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
        configs = config_from_atoms(atoms)
        #data = data.to(self.device)

        # predict + extract data
        out = self.model(data.AtomicData.from_config(configs, z_table=self.z_table, cutoff=self.r_max))
        forces = out['forces'].detach().cpu().numpy()
        energy = out['energy'].detach().cpu().item()

        # store results
        self.results = {
            "energy": energy * self.energy_units_to_eV,
            # force has units eng / len:
            "forces": forces * (self.energy_units_to_eV / self.length_units_to_A),
        }


def config_from_atoms(atoms: ase.Atoms) -> data.Configuration:
    atomic_numbers = np.array([ase.data.atomic_numbers[symbol] for symbol in atoms.symbols])
    return data.Configuration(atomic_numbers=atomic_numbers, positions=atoms.positions)


class AseInterface:
    """
    Interface for ASE calculations (optimization and molecular dynamics)
    Args:
        molecule_path (str): Path to initial geometry
        ml_model (object): Trained model
        working_dir (str): Path to directory where files should be stored
        device (str): cpu or cuda
    """
    def __init__(self,
                 molecule_path,
                 ml_model,
                 working_dir,
                 device="cpu",
                 energy="Energy",
                 forces="Forces",
                 energy_units="eV",
                 forces_units="eV/Angstrom",
                 r_max=4,
                 rescale=None):
        # Setup directory
        self.working_dir = working_dir
        if not os.path.exists(self.working_dir):
            os.makedirs(self.working_dir)

        # Load the molecule
        self.molecule = None
        self._load_molecule(molecule_path)
        self.r_max = r_max

        # Set up calculator
        calculator = MultiACECalculator(
            model=ml_model,
            r_max=self.r_max,
            device=device,
        )

        self.molecule.set_calculator(calculator)

        # Unless initialized, set dynamics to False
        self.dynamics = False

    def _load_molecule(self, molecule_path):
        """
        Load molecule from file (can handle all ase formats).
        Args:
            molecule_path (str): Path to molecular geometry
        """
        file_format = os.path.splitext(molecule_path)[-1]
        if file_format == "xyz":
            self.molecule = read_xyz(molecule_path)
        else:
            self.molecule = read(molecule_path)

    def save_molecule(self, name, file_format="xyz", append=False):
        """
        Save the current molecular geometry.
        Args:
            name (str): Name of save-file.
            file_format (str): Format to store geometry (default xyz).
            append (bool): If set to true, geometry is added to end of file
                (default False).
        """
        molecule_path = os.path.join(self.working_dir, "%s.%s" % (name, file_format))
        write(molecule_path, self.molecule, format=file_format, append=append)

    def calculate_single_point(self):
        """
        Perform a single point computation of the energies and forces and
        store them to the working directory. The format used is the extended
        xyz format. This functionality is mainly intended to be used for
        interfaces.
        """
        energy = self.molecule.get_potential_energy()
        forces = self.molecule.get_forces()
        self.molecule.energy = energy
        self.molecule.forces = forces

        self.save_molecule("single_point", file_format="extxyz")

    def init_md(
        self,
        name,
        time_step=0.5,
        temp_init=300,
        temp_bath=None,
        reset=False,
        interval=1,
    ):
        """
        Initialize an ase molecular dynamics trajectory. The logfile needs to
        be specifies, so that old trajectories are not overwritten. This
        functionality can be used to subsequently carry out equilibration and
        production.
        Args:
            name (str): Basic name of logfile and trajectory
            time_step (float): Time step in fs (default=0.5)
            temp_init (float): Initial temperature of the system in K
                (default is 300)
            temp_bath (float): Carry out Langevin NVT dynamics at the specified
                temperature. If set to None, NVE dynamics are performed
                instead (default=None)
            reset (bool): Whether dynamics should be restarted with new initial
                conditions (default=False)
            interval (int): Data is stored every interval steps (default=1)
        """

        # If a previous dynamics run has been performed, don't reinitialize
        # velocities unless explicitly requested via restart=True
        if not self.dynamics or reset:
            self._init_velocities(temp_init=temp_init)

        # Set up dynamics
        if temp_bath is None:
            self.dynamics = VelocityVerlet(self.molecule, time_step * units.fs)
        else:
            self.dynamics = Langevin(
                self.molecule,
                time_step * units.fs,
                temp_bath * units.kB,
                1.0 / (100.0 * units.fs),
            )

        # Create monitors for logfile and a trajectory file
        logfile = os.path.join(self.working_dir, "%s.log" % name)
        trajfile = os.path.join(self.working_dir, "%s.traj" % name)
        logger = MDLogger(
            self.dynamics,
            self.molecule,
            logfile,
            stress=False,
            peratom=False,
            header=True,
            mode="a",
        )
        trajectory = Trajectory(trajfile, "w", self.molecule)

        # Attach monitors to trajectory
        self.dynamics.attach(logger, interval=interval)
        self.dynamics.attach(trajectory.write, interval=interval)

    def _init_velocities(self, temp_init=300, remove_translation=True, remove_rotation=True):
        """
        Initialize velocities for molecular dynamics
        Args:
            temp_init (float): Initial temperature in Kelvin (default 300)
            remove_translation (bool): Remove translation components of
                velocity (default True)
            remove_rotation (bool): Remove rotation components of velocity
                (default True)
        """
        MaxwellBoltzmannDistribution(self.molecule, temp_init * units.kB)
        if remove_translation:
            Stationary(self.molecule)
        if remove_rotation:
            ZeroRotation(self.molecule)

    def run_md(self, steps):
        """
        Perform a molecular dynamics simulation using the settings specified
        upon initializing the class.
        Args:
            steps (int): Number of simulation steps performed
        """
        if not self.dynamics:
            raise AttributeError("Dynamics need to be initialized using the" " 'setup_md' function")

        self.dynamics.run(steps)

    def optimize(self, fmax=1.0e-2, steps=1000):
        """
        Optimize a molecular geometry using the Quasi Newton optimizer in ase
        (BFGS + line search)
        Args:
            fmax (float): Maximum residual force change (default 1.e-2)
            steps (int): Maximum number of steps (default 1000)
        """
        name = "optimization"
        optimize_file = os.path.join(self.working_dir, name)
        optimizer = QuasiNewton(
            self.molecule,
            trajectory="%s.traj" % optimize_file,
            restart="%s.pkl" % optimize_file,
        )
        optimizer.run(fmax, steps)

        # Save final geometry in xyz format
        self.save_molecule(name)

    def compute_normal_modes(self, write_jmol=True, delta=0.01):
        """
        Use ase calculator to compute numerical frequencies for the molecule
        Args:
            write_jmol (bool): Write frequencies to input file for
                visualization in jmol (default=True)
        """
        freq_file = os.path.join(self.working_dir, "normal_modes")

        # Compute frequencies
        frequencies = Vibrations(self.molecule, name=freq_file, delta=delta)
        frequencies.run()

        # Print a summary
        frequencies.summary()
        frequencies.combine()

        # Write jmol file if requested
        if write_jmol:
            frequencies.write_jmol()

        return frequencies