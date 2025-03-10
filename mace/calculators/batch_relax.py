"""
Module for doing batch relaxation
"""

from logging import getLogger
from typing import Any, Dict, List

from ase import Atoms
from ase.calculators.singlepoint import SinglePointCalculator
from ase.filters import FrechetCellFilter
from ase.optimize import FIRE
from ase.optimize.optimize import Optimizer
from ase.stress import full_3x3_to_voigt_6_stress

from mace import data
from mace.tools import torch_geometric

logger = getLogger(__name__)


class RelaxBatch:
    """
    A collection of atoms to be relaxed with a certain optimizers

    The energies and forces are computed as a single batch to increase the efficiency
    """

    def __init__(
        self,
        calculator,
        optimizer=FIRE,
        fmax=0.01,
        atoms_filter=None,
        max_n_steps=500,
    ):
        """
        Instantiate a RelaxBatch object
        """
        self.optimizer = optimizer
        self.opt_list = []
        for opt in self.opt_list:
            opt.fmax = fmax
        self.fmax = fmax
        self.all_atoms = []
        self.opt_flags = []
        self.atoms_filter = atoms_filter
        assert len(calculator.models) == 1, "Committee models are not supported"
        self.model = calculator.models[0]
        self.calc = calculator
        self.max_n_steps = max_n_steps

    def __repr__(self):
        return f"RelaxBatch with {len(self.all_atoms)} atoms (active {len(self.active_opt_index)})"

    def insert(self, atoms) -> None:
        """Insert an atoms object to the batch"""
        atoms.calc = SinglePointCalculator(atoms)
        if not self.atoms_filter:
            opt_instance = self.optimizer(atoms)
        else:
            opt_instance = self.optimizer(self.atoms_filter(atoms))
        opt_instance.fmax = self.fmax
        self.opt_list.append(opt_instance)
        self.opt_flags.append(True)
        self.all_atoms.append(atoms)

    def pop_relaxed(self) -> List[Atoms]:
        """Pop the relaxed atoms"""
        idx_kept = self.active_opt_index
        relaxed = [
            self.all_atoms[i] for i in range(len(self.all_atoms)) if i not in idx_kept
        ]
        if self.atoms_filter is not None:
            relaxed = [atoms.atoms for atoms in relaxed]
        # Remove the relaxed atoms
        self.opt_list = [self.opt_list[i] for i in idx_kept]
        self.all_atoms = [self.all_atoms[i] for i in idx_kept]
        self.opt_flags = [True] * len(idx_kept)
        return relaxed

    @property
    def active_atoms(self) -> List[Atoms]:
        idx = self.active_opt_index
        if self.atoms_filter is not None:
            return [self.opt_list[i].atoms.atoms for i in idx]
        return [self.opt_list[i].atoms for i in idx]

    @property
    def active_opts(self) -> List[Optimizer]:
        return [self.opt_list[i] for i in self.active_opt_index]

    def compute(self, skip_inactive=True) -> Dict[str, Any]:
        """Compute the energy, forces and stress for the active atoms"""
        if skip_inactive:
            active_index = self.active_opt_index
            to_calc = self.active_atoms
        else:
            active_index = list(range(len(self.all_atoms)))
            to_calc = self.all_atoms

        configs = [
            data.config_from_atoms(atoms, charges_key=self.calc.charges_key)
            for atoms in to_calc
        ]
        data_loader = torch_geometric.dataloader.DataLoader(
            dataset=[
                data.AtomicData.from_config(
                    config,
                    z_table=self.calc.z_table,
                    cutoff=self.calc.r_max,
                    heads=self.calc.heads,
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
            training=self.calc.use_compile,
        )
        results = {}
        results["energies"] = out["energy"].detach().cpu().numpy()
        results["stresses"] = out["stress"].detach().cpu().numpy()  # [n_graph, 3, 3]
        node_forces = out["forces"].detach().cpu().numpy()  # [n_nodes, 3]
        # Break the forces per node into a list of force arrays for each atoms
        results["forces"] = []
        pointer = 0
        for atoms in to_calc:
            results["forces"].append(node_forces[pointer : pointer + len(atoms), :])
            pointer += len(atoms)
        return results

    def _clone_batch(self, batch):
        batch_clone = batch.clone()
        if self.calc.use_compile:
            batch_clone["node_attrs"].requires_grad_(True)
            batch_clone["positions"].requires_grad_(True)
        return batch_clone

    @property
    def active_opt_index(self) -> List[int]:
        """Return the index of active optimizers"""
        return [i for i, flag in enumerate(self.opt_flags) if flag is True]

    def step_batch(self):
        """
        Compute the energy and forces and take step for the optimizers
        """
        results = self.compute()
        for i, (opt, atoms) in enumerate(zip(self.active_opts, self.active_atoms)):
            sp_calc = SinglePointCalculator(
                atoms,
                energy=results["energies"][i] * self.calc.energy_units_to_eV,
                forces=results["forces"][i]
                * self.calc.energy_units_to_eV
                / self.calc.length_units_to_A,
                stress=full_3x3_to_voigt_6_stress(
                    results["stresses"][i]
                    * self.calc.energy_units_to_eV
                    / self.calc.length_units_to_A**3
                ),
            )
            if self.atoms_filter is None:
                opt.atoms.calc = sp_calc
            else:
                opt.atoms.atoms.calc = sp_calc
            if opt.converged() or (get_nstep(opt) >= self.max_n_steps):
                self.opt_flags[i] = False
            else:
                # Step the optimizer
                opt.step()


def get_nstep(opt):
    """
    Get the current step number of the optimizer object
    """
    try:
        return getattr(opt, "Nsteps")
    except AttributeError:
        return getattr(opt, "nsteps")


class BatchRelaxer:
    """
    Relax a collection of atoms in batch with increased efficiency
    """

    def __init__(
        self,
        calculator,
        optimizer,
        batch_size=20,
        relax_cell=False,
        report_every=10,
    ):
        """Batch relaxation using MACE"""
        self.calc = calculator
        self.optimizer = optimizer
        self.filter = FrechetCellFilter if relax_cell else None
        self.batch_size = batch_size
        self.report_every = report_every

    def __repr__(self):
        return f"BatchRelaxer with batch size: {self.batch_size}"

    def relax(self, atoms_list, inplace=True, max_n_steps=200, fmax=0.02):
        """Relax a bunch of atoms"""
        if inplace:
            atoms_to_relax = dict(enumerate(atoms_list))
        else:
            atoms_to_relax = {i: atoms.copy() for i, atoms in enumerate(atoms_list)}

        for i, atoms in atoms_to_relax.items():
            atoms.info["_batch_relax_index"] = i
        relaxed_atoms = {}

        # Initialize the batch relax object
        relax_batch = RelaxBatch(
            self.calc,
            optimizer=self.optimizer,
            fmax=fmax,
            atoms_filter=self.filter,
            max_n_steps=max_n_steps,
        )

        last_report = 0
        while len(relaxed_atoms) != len(atoms_list):
            # Fill batch with unrelaxed atoms
            while (
                len(relax_batch.opt_list) < self.batch_size and len(atoms_to_relax) > 0
            ):
                relax_batch.insert(atoms_to_relax.pop(list(atoms_to_relax.keys())[0]))
            relax_batch.step_batch()
            for atoms in relax_batch.pop_relaxed():
                key = atoms.info["_batch_relax_index"]
                # Record a flag for the atoms that has been relaxed
                atoms.info["_batch_relaxed"] = True
                # Save this atoms to the collection of relaxed atoms
                relaxed_atoms[key] = atoms
            nrelaxed = len(relaxed_atoms)

            # Report the progress
            if nrelaxed % self.report_every == 0 and last_report != nrelaxed:
                print(f"Relaxed {nrelaxed}/{len(atoms_list)} atoms")
                last_report = nrelaxed

        # Reconstruct a list of relaxed atoms in the original order
        relaxed_atoms_list = [relaxed_atoms[i] for i in range(len(relaxed_atoms))]
        for atoms in relaxed_atoms_list:
            del atoms.info["_batch_relax_index"]
        return relaxed_atoms_list


def benchmark_batch_size(
    atoms_list: List[Atoms],
    calculator,
    optimizer=FIRE,
    batch_sizes=(4, 8, 16, 32),
    fmax=0.02,
    max_n_steps=100,
    **kwargs,
) -> tuple:
    """
    Relax the same bucket of structure with different batch size and record the timings.
    """
    from time import time

    results = {}
    for size in batch_sizes:
        relax_atoms = [atoms.copy() for atoms in atoms_list]
        br = BatchRelaxer(calculator, optimizer=optimizer, batch_size=size, **kwargs)
        start = time()
        print(f"Testing batch size: {size}")
        br.relax(relax_atoms, fmax=fmax, max_n_steps=max_n_steps)
        results[size] = time() - start
    best_size = sorted(results.items(), key=lambda x: x[1])[0][0]
    return best_size, results
