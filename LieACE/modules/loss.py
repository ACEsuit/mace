import torch
from torch_geometric.data import Batch

from lieace.tools import TensorDict


def mean_squared_error_energy(ref: Batch, pred: TensorDict) -> torch.Tensor:
    # energy: [n_graphs, ]
    return torch.mean(torch.square(ref['energy'] - pred['energy']))  # []


def weighted_mean_squared_error_energy(ref: Batch, pred: TensorDict) -> torch.Tensor:
    # energy: [n_graphs, ]
    num_atoms = ref.ptr[1:] - ref.ptr[:-1]  # [n_graphs,]
    return torch.mean(torch.square(ref['energy'] - pred['energy']) / num_atoms)  # []


def mean_squared_error_forces(ref: Batch, pred: TensorDict) -> torch.Tensor:
    # forces: [n_atoms, 3]
    return torch.mean(torch.square(ref['forces'] - pred['forces']))  # []


class EnergyForcesLoss(torch.nn.Module):
    def __init__(self, energy_weight=1.0, forces_weight=1.0) -> None:
        super().__init__()
        self.register_buffer('energy_weight', torch.tensor(energy_weight, dtype=torch.get_default_dtype()))
        self.register_buffer('forces_weight', torch.tensor(forces_weight, dtype=torch.get_default_dtype()))

    def forward(self, ref: Batch, pred: TensorDict) -> torch.Tensor:
        return (self.energy_weight * mean_squared_error_energy(ref, pred) +
                self.forces_weight * mean_squared_error_forces(ref, pred))

    def __repr__(self):
        return (f'{self.__class__.__name__}(energy_weight={self.energy_weight:.3f}, '
                f'forces_weight={self.forces_weight:.3f})')


class WeightedEnergyForcesLoss(torch.nn.Module):
    def __init__(self, energy_weight=1.0, forces_weight=1.0) -> None:
        super().__init__()
        self.register_buffer('energy_weight', torch.tensor(energy_weight, dtype=torch.get_default_dtype()))
        self.register_buffer('forces_weight', torch.tensor(forces_weight, dtype=torch.get_default_dtype()))

    def forward(self, ref: Batch, pred: TensorDict) -> torch.Tensor:
        return (self.energy_weight * weighted_mean_squared_error_energy(ref, pred) +
                self.forces_weight * mean_squared_error_forces(ref, pred))

    def __repr__(self):
        return (f'{self.__class__.__name__}(energy_weight={self.energy_weight:.3f}, '
                f'forces_weight={self.forces_weight:.3f})')


class ACELoss(torch.nn.Module):
    def __init__(self, energy_weight=1.0, forces_weight=1.0) -> None:
        super().__init__()
        self.register_buffer('w_e_sq', torch.tensor(energy_weight**2, dtype=torch.get_default_dtype()))
        self.register_buffer('w_f_sq', torch.tensor(forces_weight**2, dtype=torch.get_default_dtype()))

    def forward(self, ref: Batch, pred: TensorDict) -> torch.Tensor:
        num_atoms = ref.ptr[1:] - ref.ptr[:-1]  # [n_graphs,]
        c_e = torch.sum(torch.square(ref['energy'] - pred['energy']) / num_atoms)
        c_f = torch.sum(torch.square(ref['forces'] - pred['forces']))
        return (self.w_e_sq * c_e + self.w_f_sq * c_f) / len(ref.batch)

    def __repr__(self):
        return f'{self.__class__.__name__}(energy_weight_squared={self.w_e_sq:.3f}, ' \
               f'forces_weight_squared={self.w_f_sq:.3f})'