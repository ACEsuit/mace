###########################################################################################
# Implementation of different loss functions
# Authors: Ilyes Batatia, Gregor Simm
# This program is distributed under the MIT License (see MIT.md)
###########################################################################################

import torch

from mace.tools import TensorDict
from mace.tools.torch_geometric import Batch


def mean_squared_error_energy(ref: Batch, pred: TensorDict) -> torch.Tensor:
    # energy: [n_graphs, ]
    return torch.mean(torch.square(ref["energy"] - pred["energy"]))  # []


def weighted_mean_squared_error_energy(ref: Batch, pred: TensorDict) -> torch.Tensor:
    # energy: [n_graphs, ]
    configs_weight = ref.weight  # [n_graphs, ]
    configs_energy_weight = ref.energy_weight  # [n_graphs, ]
    num_atoms = ref.ptr[1:] - ref.ptr[:-1]  # [n_graphs,]
    return torch.mean(
        configs_weight
        * configs_energy_weight
        * torch.square((ref["energy"] - pred["energy"]) / num_atoms)
    )  # []


def weighted_mean_squared_stress(ref: Batch, pred: TensorDict) -> torch.Tensor:
    # energy: [n_graphs, ]
    configs_weight = ref.weight.view(-1, 1, 1)  # [n_graphs, ]
    configs_stress_weight = ref.stress_weight.view(-1, 1, 1)  # [n_graphs, ]
    num_atoms = (ref.ptr[1:] - ref.ptr[:-1]).view(-1, 1, 1)  # [n_graphs,]
    return torch.mean(
        configs_weight
        * configs_stress_weight
        * torch.square((ref["stress"] - pred["stress"]) / num_atoms)
    )  # []


def weighted_mean_squared_virials(ref: Batch, pred: TensorDict) -> torch.Tensor:
    # energy: [n_graphs, ]
    configs_weight = ref.weight.view(-1, 1, 1)  # [n_graphs, ]
    configs_virials_weight = ref.virials_weight.view(-1, 1, 1)  # [n_graphs, ]
    num_atoms = (ref.ptr[1:] - ref.ptr[:-1]).view(-1, 1, 1)  # [n_graphs,]
    return torch.mean(
        configs_weight
        * configs_virials_weight
        * torch.square((ref["virials"] - pred["virials"]) / num_atoms)
    )  # []


def mean_squared_error_forces(ref: Batch, pred: TensorDict) -> torch.Tensor:
    # forces: [n_atoms, 3]
    configs_weight = torch.repeat_interleave(
        ref.weight, ref.ptr[1:] - ref.ptr[:-1]
    ).unsqueeze(
        -1
    )  # [n_atoms, 1]
    configs_forces_weight = torch.repeat_interleave(
        ref.forces_weight, ref.ptr[1:] - ref.ptr[:-1]
    ).unsqueeze(
        -1
    )  # [n_atoms, 1]
    return torch.mean(
        configs_weight
        * configs_forces_weight
        * torch.square(ref["forces"] - pred["forces"])
    )  # []


def weighted_mean_squared_error_dipole(ref: Batch, pred: TensorDict) -> torch.Tensor:
    # dipole: [n_graphs, ]
    num_atoms = (ref.ptr[1:] - ref.ptr[:-1]).unsqueeze(-1)  # [n_graphs,1]
    return torch.mean(torch.square((ref["dipole"] - pred["dipole"]) / num_atoms))  # []
    # return torch.mean(torch.square((torch.reshape(ref['dipole'], pred["dipole"].shape) - pred['dipole']) / num_atoms))  # []


class WeightedEnergyForcesLoss(torch.nn.Module):
    def __init__(self, energy_weight=1.0, forces_weight=1.0) -> None:
        super().__init__()
        self.register_buffer(
            "energy_weight",
            torch.tensor(energy_weight, dtype=torch.get_default_dtype()),
        )
        self.register_buffer(
            "forces_weight",
            torch.tensor(forces_weight, dtype=torch.get_default_dtype()),
        )

    def forward(self, ref: Batch, pred: TensorDict) -> torch.Tensor:
        return self.energy_weight * weighted_mean_squared_error_energy(
            ref, pred
        ) + self.forces_weight * mean_squared_error_forces(ref, pred)

    def __repr__(self):
        return (
            f"{self.__class__.__name__}(energy_weight={self.energy_weight:.3f}, "
            f"forces_weight={self.forces_weight:.3f})"
        )


class WeightedForcesLoss(torch.nn.Module):
    def __init__(self, forces_weight=1.0) -> None:
        super().__init__()
        self.register_buffer(
            "forces_weight",
            torch.tensor(forces_weight, dtype=torch.get_default_dtype()),
        )

    def forward(self, ref: Batch, pred: TensorDict) -> torch.Tensor:
        return self.forces_weight * mean_squared_error_forces(ref, pred)

    def __repr__(self):
        return f"{self.__class__.__name__}(" f"forces_weight={self.forces_weight:.3f})"


class WeightedEnergyForcesStressLoss(torch.nn.Module):
    def __init__(self, energy_weight=1.0, forces_weight=1.0, stress_weight=1.0) -> None:
        super().__init__()
        self.register_buffer(
            "energy_weight",
            torch.tensor(energy_weight, dtype=torch.get_default_dtype()),
        )
        self.register_buffer(
            "forces_weight",
            torch.tensor(forces_weight, dtype=torch.get_default_dtype()),
        )
        self.register_buffer(
            "stress_weight",
            torch.tensor(stress_weight, dtype=torch.get_default_dtype()),
        )

    def forward(self, ref: Batch, pred: TensorDict) -> torch.Tensor:
        return (
            self.energy_weight * weighted_mean_squared_error_energy(ref, pred)
            + self.forces_weight * mean_squared_error_forces(ref, pred)
            + self.stress_weight * weighted_mean_squared_stress(ref, pred)
        )

    def __repr__(self):
        return (
            f"{self.__class__.__name__}(energy_weight={self.energy_weight:.3f}, "
            f"forces_weight={self.forces_weight:.3f}, stress_weight={self.stress_weight:.3f})"
        )


class WeightedHuberEnergyForcesStressLoss(torch.nn.Module):
    def __init__(
        self, energy_weight=1.0, forces_weight=1.0, stress_weight=1.0, huber_delta=0.01
    ) -> None:
        super().__init__()
        self.huber_loss = torch.nn.HuberLoss(reduction="mean", delta=huber_delta)
        self.register_buffer(
            "energy_weight",
            torch.tensor(energy_weight, dtype=torch.get_default_dtype()),
        )
        self.register_buffer(
            "forces_weight",
            torch.tensor(forces_weight, dtype=torch.get_default_dtype()),
        )
        self.register_buffer(
            "stress_weight",
            torch.tensor(stress_weight, dtype=torch.get_default_dtype()),
        )

    def forward(self, ref: Batch, pred: TensorDict) -> torch.Tensor:
        num_atoms = ref.ptr[1:] - ref.ptr[:-1]
        return (
            self.energy_weight
            * self.huber_loss(ref["energy"] / num_atoms, pred["energy"] / num_atoms)
            + self.forces_weight * self.huber_loss(ref["forces"], pred["forces"])
            + self.stress_weight * self.huber_loss(ref["stress"], pred["stress"])
        )

    def __repr__(self):
        return (
            f"{self.__class__.__name__}(energy_weight={self.energy_weight:.3f}, "
            f"forces_weight={self.forces_weight:.3f}, stress_weight={self.stress_weight:.3f})"
        )


class WeightedEnergyForcesVirialsLoss(torch.nn.Module):
    def __init__(
        self, energy_weight=1.0, forces_weight=1.0, virials_weight=1.0
    ) -> None:
        super().__init__()
        self.register_buffer(
            "energy_weight",
            torch.tensor(energy_weight, dtype=torch.get_default_dtype()),
        )
        self.register_buffer(
            "forces_weight",
            torch.tensor(forces_weight, dtype=torch.get_default_dtype()),
        )
        self.register_buffer(
            "virials_weight",
            torch.tensor(virials_weight, dtype=torch.get_default_dtype()),
        )

    def forward(self, ref: Batch, pred: TensorDict) -> torch.Tensor:
        return (
            self.energy_weight * weighted_mean_squared_error_energy(ref, pred)
            + self.forces_weight * mean_squared_error_forces(ref, pred)
            + self.virials_weight * weighted_mean_squared_virials(ref, pred)
        )

    def __repr__(self):
        return (
            f"{self.__class__.__name__}(energy_weight={self.energy_weight:.3f}, "
            f"forces_weight={self.forces_weight:.3f}, virials_weight={self.virials_weight:.3f})"
        )


class DipoleSingleLoss(torch.nn.Module):
    def __init__(self, dipole_weight=1.0) -> None:
        super().__init__()
        self.register_buffer(
            "dipole_weight",
            torch.tensor(dipole_weight, dtype=torch.get_default_dtype()),
        )

    def forward(self, ref: Batch, pred: TensorDict) -> torch.Tensor:
        return (
            self.dipole_weight * weighted_mean_squared_error_dipole(ref, pred) * 100.0
        )  # multiply by 100 to have the right scale for the loss

    def __repr__(self):
        return f"{self.__class__.__name__}(" f"dipole_weight={self.dipole_weight:.3f})"


class WeightedEnergyForcesDipoleLoss(torch.nn.Module):
    def __init__(self, energy_weight=1.0, forces_weight=1.0, dipole_weight=1.0) -> None:
        super().__init__()
        self.register_buffer(
            "energy_weight",
            torch.tensor(energy_weight, dtype=torch.get_default_dtype()),
        )
        self.register_buffer(
            "forces_weight",
            torch.tensor(forces_weight, dtype=torch.get_default_dtype()),
        )
        self.register_buffer(
            "dipole_weight",
            torch.tensor(dipole_weight, dtype=torch.get_default_dtype()),
        )

    def forward(self, ref: Batch, pred: TensorDict) -> torch.Tensor:
        return (
            self.energy_weight * weighted_mean_squared_error_energy(ref, pred)
            + self.forces_weight * mean_squared_error_forces(ref, pred)
            + self.dipole_weight * weighted_mean_squared_error_dipole(ref, pred) * 100
        )

    def __repr__(self):
        return (
            f"{self.__class__.__name__}(energy_weight={self.energy_weight:.3f}, "
            f"forces_weight={self.forces_weight:.3f}, dipole_weight={self.dipole_weight:.3f})"
        )
