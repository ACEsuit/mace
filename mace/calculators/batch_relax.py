"""
Module for doing batch relaxation
"""
from typing import List, Dict, Any
from ase import Atoms
from ase.calculators.singlepoint import SinglePointCalculator
from ase.stress import full_3x3_to_voigt_6_stress
from ase.filters import FrechetCellFilter
from ase.optimize import FIRE
from ase.optimize.optimize import Optimizer

from logging import getLogger

from mace import data
from mace.tools import torch_geometric, torch_tools, utils
logger = getLogger(__name__)

class RelaxBatch:
    """
    A collection of atoms to be relaxed with a certain optimizers

    The energies and forces are computed as a single batch to increase the efficiency
    """

    def __init__(self, atoms_list, calc, optimizer=FIRE, fmax=0.01, filter=None, max_n_steps=500):
        self.optimizer = optimizer
        if filter:
            self.opt_list = [optimizer(filter(atoms)) for atoms in atoms_list]
        else:
            self.opt_list = [optimizer(atoms) for atoms in atoms_list]
        for opt in self.opt_list:
            opt.fmax = fmax
        self.fmax = fmax
        self.opt_flags = [True] * len(atoms_list)
        self.filter = filter
        assert len(calc.models) == 1, "Committee models are not supported"
        self.model = calc.models[0]
        self.calc = calc
        self.max_n_steps=500

    def insert(self, atoms):
        """Insert an atoms object to the batch"""
        atoms._calc_bak = atoms.calc
        atoms.calc = SinglePointCalculator(atoms)
        if not self.filter:
            opt_instance = self.optimizer(atoms)
        else:
            opt_instance = self.optimizer(self.filter(atoms))
        opt_instance.fmax = self.fmax
        self.opt_list.append(opt_instance)
        self.opt_flags.append(True)


    def pop_relaxed(self) -> List[Atoms]:
        """Pop the relaxed atoms"""
        relaxed = [opt.atoms for opt, flag in zip(self.opt_list, self.opt_flags) if flag is False]
        if self.filter is not None:
            relaxed = [atoms.atoms for atoms in relaxed]
        # Remove the relaxed atoms
        self.opt_list = [opt for opt, flag in zip(self.opt_list, self.opt_flags) if flag is True]
        self.opt_flags = [True] * len(self.opt_list)
        return relaxed

    def get_active_atoms(self) -> List[Atoms]:
        idx = self.get_activate_opt_index()
        if self.filter is not None:
            return [self.opt_list[i].atoms.atoms for i in idx]
        return [self.opt_list[i].atoms for i in idx]

    def get_active_opts(self) -> List[Optimizer]:
        idx = self.get_activate_opt_index()
        return [self.opt_list[i] for i in idx]

    def compute(self) -> Dict[str, Any]:
        """Compute the energy, forces and stress for the active atoms"""
        active_index = self.get_activate_opt_index()
        to_calc = self.get_active_atoms()

        configs = [ data.config_from_atoms(atoms, charges_key=self.calc.charges_key) for atoms in to_calc]
        data_loader = torch_geometric.dataloader.DataLoader(
            dataset=[
                data.AtomicData.from_config(
                    config, z_table=self.calc.z_table, cutoff=self.calc.r_max, heads=self.calc.heads
                )
                for config in configs
            ],
            batch_size=len(active_index),
            shuffle=False,
            drop_last=False,
        )
        batch_base = next(iter(data_loader)).to(self.calc.device)

        if self.calc.model_type in ["MACE", "EnergyDipoleMACE"]:
            compute_stress = not self.calc.use_compile
        else:
            compute_stress = False

        # Compute the batch
        batch = self._clone_batch(batch_base)
        out = self.model(
            batch.to_dict(),
            compute_stress=compute_stress,
            training=self.calc.use_compile
        )
        results = {}
        results['energies'] = out["energy"].detach().cpu().numpy()
        results['stresses'] = out["stress"].detach().cpu().numpy()  # [n_graph, 3, 3]
        node_forces = out["forces"].detach().cpu().numpy() # [n_nodes, 3]
        # Break the forces per node into a list of force arrays for each atoms
        results['forces'] = []
        pointer = 0
        for atoms in to_calc:
            results['forces'].append(node_forces[pointer:pointer + len(atoms), :])
            pointer += len(atoms)
        # TODO: process dipoles
        return results

        

    def _clone_batch(self, batch):
        batch_clone = batch.clone()
        if self.calc.use_compile:
            batch_clone["node_attrs"].requires_grad_(True)
            batch_clone["positions"].requires_grad_(True)
        return batch_clone


    def get_activate_opt_index(self) -> List[int]:
        """Return the index of active optimizers"""
        return [i for i, flag in enumerate(self.opt_flags) if flag is True]
 

    def step_batch(self):
        """
        Compute the energy and forces and take step for the optimizers
        """
        results = self.compute()
        self.finished = True
        for i, (opt, atoms) in enumerate(zip(self.get_active_opts(), self.get_active_atoms())):
            sp_calc = SinglePointCalculator(atoms, 
                                            energy=results['energies'][i] * self.calc.energy_units_to_eV,
                                            forces=results['forces'][i] * self.calc.energy_units_to_eV / self.calc.length_units_to_A, 
                                            stress=full_3x3_to_voigt_6_stress(results['stresses'][i] * self.calc.energy_units_to_eV / 
                                                                              self.calc.length_units_to_A ** 3)
                                            )
            if self.filter is None:
                opt.atoms.calc = sp_calc
            else:
                opt.atoms.atoms.calc = sp_calc
            if opt.converged() or get_nstep(opt) >= self.max_n_steps:
                self.opt_flags[i] = False
                continue
            else:
                self.finished = False
            # Step the optimizer
            opt.step()

def get_nstep(opt):
    """
    Get the current step number of the optimizer object
    """
    try:
        return getattr(opt, 'Nsteps') 
    except AttributeError:
        return getattr(opt, 'nsteps')

class BatchRelaxer:
    """
    Relax a collection of atoms in batch with increased efficiency
    """

    def __init__(self, calculator, optimizer, batch_size=20, relax_cell=False, fmax=0.01):
        """Batch relaxation using MACE"""
        self.fmax = fmax
        self.calc = calculator
        self.optimizer = optimizer
        self.filter = FrechetCellFilter if relax_cell else None
        self.batch = None
        self.batch_size = batch_size

    def relax(self, atoms_list, inplace=True):
        """Relax a bunch of atoms"""
        self.trajectories = {}
        if inplace:
            atoms_to_relax = {i: atoms for i, atoms in enumerate(atoms_list)}
        else:
            atoms_to_relax = {i: atoms.copy() for i, atoms in enumerate(atoms_list)}

        for i, atoms in atoms_to_relax.items():
            atoms.info['_batch_relax_index'] = i
        relaxed_atoms = {}

        # Initialize the batch relax object
        relax_batch = RelaxBatch([], 
                                 self.calc, optimizer=self.optimizer, fmax=self.fmax, 
                                 filter=self.filter)

        last_report = 0
        while len(relaxed_atoms) != len(atoms_list):

            # Fill batch with unrelaxed atoms 
            while len(relax_batch.opt_list) < self.batch_size and len(atoms_to_relax) > 0:
                relax_batch.insert(atoms_to_relax.pop(list(atoms_to_relax.keys())[0]))
            relax_batch.step_batch()
            for atoms in relax_batch.pop_relaxed():
                key = atoms.info['_batch_relax_index'] 
                # Save this atoms to the collection of relaxed atoms
                relaxed_atoms[key] = atoms
            nrelaxed = len(relaxed_atoms)
            if nrelaxed % 100 and last_report != nrelaxed: 
                print(f'Relaxed {nrelaxed}/{len(atoms_list)} atoms')
                last_report=nrelaxed

        # Reconstruct a list of relaxed atoms in the original order
        relaxed_atoms_list = [relaxed_atoms[i] for i in range(len(relaxed_atoms))] 
        for atoms in relaxed_atoms_list:
            del atoms.info['_batch_relax_index']
        return relaxed_atoms_list


def benchmark_batch_size(atoms_list, calculator, optimizer=FIRE, batch_sizes=[4, 8, 16, 32], **kwargs):
    """
    Relax the same bucket of structure with different batch size and record the timings.
    """
    from time import time
    results = {}
    for size in batch_sizes:
        relax_atoms = [atoms.copy() for atoms in atoms_list]
        br = BatchRelaxer(calculator, optimizer=optimizer, batch_size=size, **kwargs)
        start = time()
        print(f'Testing batch size: {size}')
        br.relax(relax_atoms)
        results[size] = (time() - start)
    best_size = sorted(results.items(), key=lambda x: x[1])[0][0]
    return best_size, results
