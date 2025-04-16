from typing import Dict, List, Optional, Tuple
import os
import copy

import torch
from e3nn.util.jit import compile_mode
from e3nn.util import jit
from ase.data import chemical_symbols
from time import perf_counter

try:
    from lammps.mliap.mliap_unified_abc import MLIAPUnified
except:
    pass

class LAMMPS_MLIAP_MACE(MLIAPUnified):
    def __init__(
        self,
        model,
        **kwargs
    ):
        super().__init__()
        self.ndescriptors = 1
        self.model = LAMMPS_MACE_MLIAP(model, **kwargs)
        # self.model = jit.compile(copy.deepcopy(self.model))
        self.element_types = [chemical_symbols[s] for s in model.atomic_numbers]
        self.num_species = len(self.element_types)
        self.rcutfac = 0.5*model.r_max # Half of radial cutoff
        # Inferring type from the model
        self.dtype = model.r_max.dtype
        self.device = 'cpu'
        self.nparams = 1 # TODO: estimate the actual number
        self.first_run = False
        self.step = 0
        self.time = perf_counter()
        self.debug_time = False
        self.debug_profile = False

    def compute_gradients(self, data):
        pass

    def compute_descriptors(self, data):
        pass

    def compute_forces(self, data):
        
        natoms = data.nlocal
        ntotal = data.ntotal
        nghosts = ntotal - natoms
        npairs = data.npairs
        species = torch.as_tensor(data.elems, dtype=torch.int64)

        # Do things only once
        if not self.first_run:
            using_kokkos = "kokkos" in data.__class__.__module__.lower()
            if using_kokkos:
                # Getting device in torch format
                self.device = species.device
            else:
                self.device = "cpu"
            if not using_kokkos:
                raise ValueError("Only kokkos backend supported for now")
            if self.device=="cpu":
                raise ValueError("Only GPU HW supported for now")
            
            self.model = self.model.to(self.device)
            self.debug_time = os.environ.get('MACE_TIME', 'False').lower() in ('true', '1', 't')
            self.debug_profile = os.environ.get('MACE_PROFILE', 'False').lower() in ('true', '1', 't')
            self.first_run = True

        # Code for timing
        if self.debug_time:
            newtime = perf_counter()
            if self.step>0:
                print(f"Step: {self.step-1}, time: {1000*(newtime-self.time)} ms")
            self.time = newtime

        # Code for profiling
        if self.debug_profile:
            # Arbitrarily profile from step 5 to step 10
            if self.step==5:
                # torch.cuda.profiler.profile()
                torch.cuda.profiler.start()
            if self.step==10:
                torch.cuda.profiler.stop()
                exit() # Just to make sure we stop

        self.step += 1

        if natoms>0 and npairs>1:            
            # Making batch
            batch = {
                "vectors": torch.as_tensor(data.rij).to(self.dtype), # n_pairs
                "node_attrs": torch.nn.functional.one_hot(species, 
                    num_classes=self.num_species).to(self.dtype), # n_total, nspecies
                "edge_index": torch.stack([
                    torch.as_tensor(data.pair_j, dtype=torch.int64),
                    torch.as_tensor(data.pair_i, dtype=torch.int64)], dim=0), #n_pairs
                # "batch": torch.cat([
                #     torch.zeros(natoms, dtype=torch.int64, device=self.device),
                #     torch.ones(nghosts, dtype=torch.int64, device=self.device)], dim=0), # n_total
                "batch": torch.zeros(natoms, dtype=torch.int64, device=self.device), # n_atoms
                "lammps_class": data,
                "natoms": (natoms, nghosts),
            }
            Ee, Ei, fij = self.model(batch)
            # Recomputing Etotal for improved accuracy
            E = torch.sum(Ei[:natoms])
            # print(Ee.detach().cpu().numpy(), E.detach().cpu().numpy())
            # Making sure computations are finished
            torch.cuda.synchronize()
            
            # Copying values at the existing energies address in fp64
            if self.dtype == torch.float32:
                # Ei = Ei.double()
                fij = fij.double()
            eatoms = torch.as_tensor(data.eatoms)
            eatoms.copy_(Ei[:natoms])
            data.energy = E
            data.update_pair_forces_gpu(fij)


# TODO: Should we bake this into the other class? Or keep for sake of integration?
@compile_mode("script")
class LAMMPS_MACE_MLIAP(torch.nn.Module):
    def __init__(self, model, **kwargs):
        super().__init__()
        self.model = model
        self.register_buffer("atomic_numbers", model.atomic_numbers)
        self.register_buffer("r_max", model.r_max)
        self.register_buffer("num_interactions", model.num_interactions)
        if not hasattr(model, "heads"):
            model.heads = [None]
        self.register_buffer(
            "head",
            torch.tensor(
                self.model.heads.index(kwargs.get("head", self.model.heads[-1])),
                dtype=torch.long,
            ).unsqueeze(0),
        )

        for param in self.model.parameters():
            param.requires_grad = False

    def forward(
        self,
        data: Dict[str, torch.Tensor],
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[torch.Tensor]]:
        # Disregarding heads for now...
        # data["head"] = self.head
        out = self.model(
            data,
            training=False,
            compute_force=False,
            compute_virials=False,
            compute_stress=False,
            compute_displacement=False,
        )
        total_energy = out['energy'][0]
        # ...for pedantic compile
        # assert total_energy is not None
        # total_energy = total_energy[0]
        node_energy = out["node_energy"]
        # Compute forces w.r.t each edge
        edge_forces = torch.autograd.grad(
            outputs=[total_energy],
            inputs=[data['vectors']],
            grad_outputs=None,
            retain_graph=False,
            create_graph=False,
            allow_unused=True,
        )[0]
        return total_energy, node_energy, edge_forces

