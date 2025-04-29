###########################################################################################
# Implementation of different loss functions
# Authors: Ilyes Batatia, Gregor Simm
# This program is distributed under the MIT License (see MIT.md)
###########################################################################################

import torch

from mace.tools import TensorDict
from mace.tools.torch_geometric import Batch

import torch.nn as nn

from torch_scatter import scatter

import torch.distributed as dist


def mean_squared_error_energy(ref: Batch, pred: TensorDict) -> torch.Tensor:
    # energy: [n_graphs, ]
    return torch.mean(torch.square(ref["energy"] - pred["energy"]))  # []


def weighted_mean_abs_error_energy(ref: Batch, pred: TensorDict) -> torch.tensor:
    # energy: [n_graphs, ]
    configs_weight = ref.weight  # [n_graphs, ]
    configs_energy_weight = ref.energy_weight  # [n_graphs, ]
    num_atoms = ref.ptr[1:] - ref.ptr[:-1]  # [n_graphs,]
    return torch.mean(
        configs_weight
        * configs_energy_weight
        * torch.abs((ref["energy"] - pred["energy"]) / num_atoms)
    )  # []

def weighted_mean_squared_error_energy(ref: Batch, pred: TensorDict) -> torch.tensor:
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
    return torch.mean(
        configs_weight
        * configs_stress_weight
        * torch.square(ref["stress"] - pred["stress"])
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


def conditional_mse_forces(ref: Batch, pred: TensorDict) -> torch.Tensor:
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

    # Define the multiplication factors for each condition
    factors = torch.tensor([1.0, 0.7, 0.4, 0.1])

    # Apply multiplication factors based on conditions
    c1 = torch.norm(ref["forces"], dim=-1) < 100
    c2 = (torch.norm(ref["forces"], dim=-1) >= 100) & (
        torch.norm(ref["forces"], dim=-1) < 200
    )
    c3 = (torch.norm(ref["forces"], dim=-1) >= 200) & (
        torch.norm(ref["forces"], dim=-1) < 300
    )

    err = ref["forces"] - pred["forces"]

    se = torch.zeros_like(err)

    se[c1] = torch.square(err[c1]) * factors[0]
    se[c2] = torch.square(err[c2]) * factors[1]
    se[c3] = torch.square(err[c3]) * factors[2]
    se[~(c1 | c2 | c3)] = torch.square(err[~(c1 | c2 | c3)]) * factors[3]

    return torch.mean(configs_weight * configs_forces_weight * se)


def conditional_huber_forces(
    ref: Batch, pred: TensorDict, huber_delta: float
) -> torch.Tensor:
    # Define the multiplication factors for each condition
    factors = huber_delta * torch.tensor([1.0, 0.7, 0.4, 0.1])

    # Apply multiplication factors based on conditions
    c1 = torch.norm(ref["forces"], dim=-1) < 100
    c2 = (torch.norm(ref["forces"], dim=-1) >= 100) & (
        torch.norm(ref["forces"], dim=-1) < 200
    )
    c3 = (torch.norm(ref["forces"], dim=-1) >= 200) & (
        torch.norm(ref["forces"], dim=-1) < 300
    )
    c4 = ~(c1 | c2 | c3)

    se = torch.zeros_like(pred["forces"])

    se[c1] = torch.nn.functional.huber_loss(
        ref["forces"][c1], pred["forces"][c1], reduction="none", delta=factors[0]
    )
    se[c2] = torch.nn.functional.huber_loss(
        ref["forces"][c2], pred["forces"][c2], reduction="none", delta=factors[1]
    )
    se[c3] = torch.nn.functional.huber_loss(
        ref["forces"][c3], pred["forces"][c3], reduction="none", delta=factors[2]
    )
    se[c4] = torch.nn.functional.huber_loss(
        ref["forces"][c4], pred["forces"][c4], reduction="none", delta=factors[3]
    )

    return torch.mean(se)


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


class OMat24Loss(torch.nn.Module):
    def __init__(
        self, energy_weight=1.0, forces_weight=1.0, stress_weight=1.0, huber_delta=0.01,
        head_stress_mask=None
    ) -> None:
        super().__init__()
        self.huber_loss = torch.nn.HuberLoss(reduction="mean", delta=huber_delta)
        self.register_buffer(
            "energy_weight",
            torch.tensor(energy_weight, dtype=torch.get_default_dtype()),
k        )
        self.register_buffer(
            "forces_weight",
            torch.tensor(forces_weight, dtype=torch.get_default_dtype()),
        )
        self.register_buffer(
            "stress_weight",
            torch.tensor(stress_weight, dtype=torch.get_default_dtype()),
        )
        self.head_stress_mask=head_stress_mask

    def forward(self, ref: Batch, pred: TensorDict) -> torch.Tensor:
        num_atoms = ref.ptr[1:] - ref.ptr[:-1]
        if self.head_stress_mask is None:
            return (
                self.energy_weight
                * weighted_mean_abs_error_energy(ref, pred)
                + self.forces_weight * mean_squared_error_forces(ref, pred)
                + self.stress_weight * self.huber_loss(ref["stress"], pred["stress"])
            )
        else:
            stress_musk = self.head_stress_mask[ref.head].view(-1, 1, 1)
            return (
                self.energy_weight
                * weighted_mean_abs_error_energy(ref, pred)
                + self.forces_weight * mean_squared_error_forces(ref, pred)
                + self.stress_weight * self.huber_loss(ref["stress"] * stress_musk, pred["stress"] * stress_musk)
            )

    def __repr__(self):
        return (
            f"{self.__class__.__name__}(energy_weight={self.energy_weight:.3f}, "
            f"forces_weight={self.forces_weight:.3f}, stress_weight={self.stress_weight:.3f})"
        )

class UniversalLoss(torch.nn.Module):
    def __init__(
        self, energy_weight=1.0, forces_weight=1.0, stress_weight=1.0, huber_delta=0.01,
        head_stress_mask=None
    ) -> None:
        super().__init__()
        self.huber_delta = huber_delta
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
        self.head_stress_mask=head_stress_mask

    def forward(self, ref: Batch, pred: TensorDict) -> torch.Tensor:
        num_atoms = ref.ptr[1:] - ref.ptr[:-1]
        if self.head_stress_mask is None:
            return (
                self.energy_weight
                * self.huber_loss(ref["energy"] / num_atoms, pred["energy"] / num_atoms)
                + self.forces_weight
                * conditional_huber_forces(ref, pred, huber_delta=self.huber_delta)
                + self.stress_weight * self.huber_loss(ref["stress"], pred["stress"])
            )
        else:
            stress_musk = self.head_stress_mask[ref.head].view(-1, 1, 1)
            return (
                self.energy_weight
                * self.huber_loss(ref["energy"] / num_atoms, pred["energy"] / num_atoms)
                + self.forces_weight
                * conditional_huber_forces(ref, pred, huber_delta=self.huber_delta)
                + self.stress_weight * self.huber_loss(ref["stress"] * stress_musk, pred["stress"] * stress_musk)
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

class DDPLoss(nn.Module):
    """
    This class is a wrapper around a loss function that does a few things
    like handle nans and importantly ensures the reduction is done
    correctly for DDP. The main issue is that DDP averages gradients
    over replicas — this only works out of the box if the dimension
    you are averaging over is completely consistent across all replicas.
    In our case, that is not true for the number of atoms per batch and
    there are edge cases when the batch size differs between replicas
    e.g. if the dataset size is not divisible by the batch_size.

    Scalars are relatively straightforward to handle, but vectors and higher tensors
    are a bit trickier. Below are two examples of forces.

    Forces input: [Nx3] target: [Nx3]
    Forces are a vector of length 3 (x,y,z) for each atom.
    Number of atoms per batch (N) is different for each DDP replica.

    MSE example:
    #### Local loss computation ####
    local_loss = MSELoss(input, target) -> [Nx3]
    num_samples = local_loss.numel() -> [Nx3]
    local_loss = sum(local_loss [Nx3]) -> [1] sum reduces the loss to a scalar
    global_samples = all_reduce(num_samples) -> [N0x3 + N1x3 + N2x3 + ...] = [1] where N0 is the number of atoms on replica 0
    local_loss = local_loss * world_size / global_samples -> [1]
    #### Global loss computation ####
    global_loss = sum(local_loss / world_size) -> [1]
    == sum(local_loss / global_samples) # this is the desired corrected mean

    Norm example:
    #### Local loss computation ####
    local_loss = L2MAELoss(input, target) -> [N]
    num_samples = local_loss.numel() -> [N]
    local_loss = sum(local_loss [N]) -> [1] sum reduces the loss to a scalar
    global_samples = all_reduce(num_samples) -> [N0 + N1 + N2 + ...] = [1] where N0 is the number of atoms on replica 0
    local_loss = local_loss * world_size / global_samples -> [1]
    #### Global loss computation ####
    global_loss = sum(local_loss / world_size) -> [1]
    == sum(local_loss / global_samples) # this is the desired corrected mean
    """

    def __init__(
        self,
        loss_fn,
        reduction: str = "mean",
        zero_nans: bool = False
    ) -> None:
        super().__init__()
        self.loss_fn = loss_fn
        # default reduction is mean
        self.reduction = reduction if reduction is not None else "mean"
        self.reduction_map = {
            "mean": self.mean,
            "sum": self.sum,
        }
        self.zero_nans = zero_nans
        assert self.reduction in list(
            self.reduction_map.keys()
        ), "Reduction must be one of: 'mean', 'sum'"

    def sum(self, input, loss, natoms):
        # this sum will reduce the loss down to a single scalar
        return torch.sum(loss)

    def _ddp_mean(self, num_samples, loss):
        global_samples = all_reduce(num_samples, device=loss.device)
        # Multiply by world size since gradients are averaged across DDP replicas
        # warning this is probably incorrect for any model parallel approach
        return loss * get_world_size() / global_samples

    def mean(self, input, loss, natoms):
        # total elements to take the mean over
        # could be batch_size, num_atoms, num_atomsx3, etc
        num_samples = loss.numel()
        # this sum will reduce the loss down from num_sample -> 1
        loss = self.sum(input, loss, natoms)
        return self._ddp_mean(num_samples, loss)

    def _reduction(self, input, loss, natoms):
        if self.reduction in self.reduction_map:
            return self.reduction_map[self.reduction](input, loss, natoms)
        else:
            raise ValueError("Reduction must be one of: 'mean', 'sum'")

    def forward(
        self,
        input: torch.Tensor,
        target: torch.Tensor,
        natoms: torch.Tensor,
    ):
        # ensure torch doesn't do any unwanted broadcasting
        assert (
            input.shape == target.shape
        ), f"Mismatched shapes: {input.shape} and {target.shape}"

        if self.zero_nans:
            # zero out nans, if any
            found_nans_or_infs = not torch.all(input.isfinite())
            if found_nans_or_infs is True:
                input = torch.nan_to_num(input, nan=0.0)

        loss = self.loss_fn(input, target, natoms)
        return self._reduction(input, loss, natoms)

class L2NormLoss(nn.Module):
    """
    Currently this loss is intened to used with vectors.
    """

    def __init__(self) -> None:
        super().__init__()

    def forward(
        self, pred: torch.Tensor, target: torch.Tensor, natoms: torch.Tensor
    ) -> torch.Tensor:
        assert target.dim() == 2
        assert target.shape[1] != 1
        return torch.linalg.vector_norm(pred - target, ord=2, dim=-1)

class MAELoss(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.loss = nn.L1Loss()
        # reduction should be none as it is handled in DDPLoss
        self.loss.reduction = "none"

    def forward(
        self, pred: torch.Tensor, target: torch.Tensor, natoms: torch.Tensor
    ) -> torch.Tensor:
        return self.loss(pred, target)

class PerAtomMAELoss(nn.Module):
    """
    Simply divide a loss by the number of atoms/nodes in the graph.
    Current this loss is intened to used with scalar values, not vectors or higher tensors.
    """

    def __init__(self) -> None:
        super().__init__()
        self.loss = nn.L1Loss()
        # reduction should be none as it is handled in DDPLoss
        self.loss.reduction = "none"

    def forward(
        self, pred: torch.Tensor, target: torch.Tensor, natoms: torch.Tensor
    ) -> torch.Tensor:
        _natoms = torch.reshape(natoms, target.shape)
        # check if target is a scalar
        assert target.dim() == 1 or (target.dim() == 2 and target.shape[1] == 1)
        # check per_atom shape
        assert (target / _natoms).shape == target.shape
        return self.loss(pred / _natoms, target / _natoms)

class PerAtomMAEDDPGraphSizeDeBias(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.loss = nn.L1Loss()
        # reduction should be none as it is handled in DDPLoss
        self.loss.reduction = "none"

    def forward(
        self, pred: torch.Tensor, target: torch.Tensor, natoms: torch.Tensor
    ) -> torch.Tensor:
        _natoms = torch.reshape(natoms, target.shape)
        # check if target is a scalar
        assert target.dim() == 1 or (target.dim() == 2 and target.shape[1] == 1)
        # check per_atom shape
        assert (target / _natoms).shape == target.shape
        avg = self.loss(pred / _natoms, target / _natoms)
        result = torch.repeat_interleave(avg, natoms.view(-1), dim=0)
        return result

class MSELoss(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.loss = nn.MSELoss()
        # reduction should be none as it is handled in DDPLoss
        self.loss.reduction = "none"

    def forward(
        self, pred: torch.Tensor, target: torch.Tensor, natoms: torch.Tensor
    ) -> torch.Tensor:
        return self.loss(pred, target)

class DDP_weighted_MAE_peratomE(nn.Module):
    def __init__(
        self, energy_weight=1.0, forces_weight=1.0, stress_weight=1.0,
        head_stress_mask=None
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
            "stress_weight",
            torch.tensor(stress_weight, dtype=torch.get_default_dtype()),
        )
        self.head_stress_mask=head_stress_mask
        self.loss_dict = {
            "ddp_energy_mae" : MAELoss(),
            "ddp_energy_peratom_mae": PerAtomMAELoss(),
            "ddp_forces_mae"  : L2NormLoss(),
            "ddp_stress_mae" : L2NormLoss(), 
            "ddp_energy_peratom_mae_debias": PerAtomMAEDDPGraphSizeDeBias(),
        }
        self.loss_dict = nn.ModuleDict(self.loss_dict)
        
        for key in self.loss_dict.keys():
            if "ddp" in key:
                self.loss_dict[key] = DDPLoss(self.loss_dict[key], 'mean', zero_nans=False)

        self.force_loss = "ddp_forces_mae"
        self.energy_loss = "ddp_energy_peratom_mae"
        self.stress_loss = "ddp_stress_mae"


    def forward(self, ref: Batch, pred: TensorDict) -> torch.Tensor:
        e_pred, e_label = pred["energy"], ref["energy"]
        f_pred, f_label = pred["forces"], ref["forces"]
        s_pred, s_label = pred["stress"], ref["stress"]
        n_atoms = ref.ptr[1:] - ref.ptr[:-1]

        e_loss = self.energy_weight * self.loss_dict[self.energy_loss](e_pred, e_label, n_atoms)
        f_loss = self.forces_weight * self.loss_dict[self.force_loss](f_pred, f_label, n_atoms)
        s_loss = self.stress_weight * self.loss_dict[self.stress_loss](s_pred.view(-1, 9), s_label.view(-1, 9), n_atoms)
        return e_loss + f_loss + s_loss 

class DDP_weighted_MAE_all(nn.Module):
    def __init__(
        self, energy_weight=1.0, forces_weight=1.0, stress_weight=1.0,
        head_stress_mask=None
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
            "stress_weight",
            torch.tensor(stress_weight, dtype=torch.get_default_dtype()),
        )
        self.head_stress_mask=head_stress_mask
        self.loss_dict = {
            "ddp_energy_mae" : MAELoss(),
            "ddp_energy_peratom_mae": PerAtomMAELoss(),
            "ddp_forces_mae"  : L2NormLoss(),
            "ddp_stress_mae" : L2NormLoss(), 
            "ddp_energy_peratom_mae_debias": PerAtomMAEDDPGraphSizeDeBias(),
        }
        self.loss_dict = nn.ModuleDict(self.loss_dict)
        
        for key in self.loss_dict.keys():
            if "ddp" in key:
                self.loss_dict[key] = DDPLoss(self.loss_dict[key], 'mean', zero_nans=False)

        self.force_loss = "ddp_forces_mae"
        self.energy_loss = "ddp_energy_peratom_mae_debias"
        self.stress_loss = "ddp_stress_mae"


    def forward(self, ref: Batch, pred: TensorDict) -> torch.Tensor:
        e_pred, e_label = pred["energy"], ref["energy"]
        f_pred, f_label = pred["forces"], ref["forces"]
        s_pred, s_label = pred["stress"], ref["stress"]
        n_atoms = ref.ptr[1:] - ref.ptr[:-1]

        e_loss = self.energy_weight * self.loss_dict[self.energy_loss](e_pred, e_label, n_atoms)
        f_loss = self.forces_weight * self.loss_dict[self.force_loss](f_pred, f_label, n_atoms)
        s_loss = self.stress_weight * self.loss_dict[self.stress_loss](s_pred.view(-1, 9), s_label.view(-1, 9), n_atoms)
        return e_loss + f_loss + s_loss 

def all_reduce(
    data, group=dist.group.WORLD, average: bool = False, device=None
) -> torch.Tensor:
    if get_world_size() == 1:
        return data
    tensor = data
    if not isinstance(data, torch.Tensor):
        tensor = torch.tensor(data)
    if device is not None:
        tensor = tensor.to(device)
    dist.all_reduce(tensor, group=group)
    if average:
        tensor /= get_world_size()
    if not isinstance(data, torch.Tensor):
        result = tensor.cpu().numpy() if tensor.numel() > 1 else tensor.item()
    else:
        result = tensor
    return result

def get_world_size() -> int:
    return dist.get_world_size() if initialized() else 1

def initialized() -> bool:
    return dist.is_available() and dist.is_initialized()
