from pathlib import Path
from typing import Any, Callable, Dict, Optional, Union

import torch

from mace.tools import atomic_numbers_to_indices, utils

try:
    import torch_sim as ts
    from torch_sim.models.interface import ModelInterface
    from torch_sim.neighbors import torchsim_nl

    _TORCHSIM_IMPORT_ERROR: Optional[ImportError] = None
except ImportError as exc:
    ts = None  # type: ignore[assignment]
    torchsim_nl = None  # type: ignore[assignment]
    _TORCHSIM_IMPORT_ERROR = exc

    class ModelInterface(torch.nn.Module):  # type: ignore[no-redef]  # pylint: disable=abstract-method
        """Fallback base class when torch-sim is not installed."""


def to_one_hot(
    indices: torch.Tensor, num_classes: int, dtype: torch.dtype
) -> torch.Tensor:
    """Generate one-hot vectors from class indices."""
    shape = indices.shape[:-1] + (num_classes,)
    out = torch.zeros(shape, device=indices.device, dtype=dtype).view(shape)
    out.scatter_(dim=-1, index=indices, value=1)
    return out.view(*shape)


class MaceTorchSimModel(ModelInterface):
    """TorchSim wrapper around a MACE model."""

    def __init__(
        self,
        model: Union[str, Path, torch.nn.Module],
        device: Optional[torch.device] = None,
        dtype: torch.dtype = torch.float64,
        neighbor_list_fn: Optional[Callable] = None,
        compute_forces: bool = True,
        compute_stress: bool = True,
        enable_cueq: bool = False,
        atomic_numbers: Optional[torch.Tensor] = None,
        system_idx: Optional[torch.Tensor] = None,
    ) -> None:
        if _TORCHSIM_IMPORT_ERROR is not None:
            raise ImportError(
                "MaceTorchSimModel requires torch-sim-atomistic. "
                "Install with `pip install torch-sim-atomistic` "
                "or `pip install -e '.[torchsim]'`."
            ) from _TORCHSIM_IMPORT_ERROR

        super().__init__()
        self._device = device or torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )
        self._dtype = dtype
        self._compute_forces = compute_forces
        self._compute_stress = compute_stress
        self._memory_scales_with = "n_atoms_x_density"
        self.neighbor_list_fn = neighbor_list_fn or torchsim_nl

        if isinstance(model, (str, Path)):
            self.model = torch.load(str(model), map_location=self.device)
        elif isinstance(model, torch.nn.Module):
            self.model = model.to(self.device)
        else:
            raise TypeError("model must be a path or torch.nn.Module")

        self.model = self.model.eval().to(device=self._device)
        if self.dtype is not None:
            self.model = self.model.to(dtype=self.dtype)

        if enable_cueq:
            try:
                from mace.cli.convert_e3nn_cueq import run as run_e3nn_to_cueq
            except ImportError as exc:
                raise ImportError(
                    "cuequivariance is not installed so CuEq acceleration cannot be used"
                ) from exc
            self.model = run_e3nn_to_cueq(self.model, device=self.device.type)

        self.r_max = self.model.r_max
        self.z_table = utils.AtomicNumberTable(
            [int(z) for z in self.model.atomic_numbers]
        )
        self.model.atomic_numbers = (
            self.model.atomic_numbers.detach().clone().to(device=self.device)
        )

        self.atomic_numbers_in_init = atomic_numbers is not None
        if atomic_numbers is not None:
            if system_idx is None:
                system_idx = torch.zeros(
                    len(atomic_numbers), dtype=torch.long, device=self.device
                )
            self.setup_from_system_idx(atomic_numbers, system_idx)

    def setup_from_system_idx(
        self, atomic_numbers: torch.Tensor, system_idx: torch.Tensor
    ) -> None:
        """Prepare cached batch tensors from atom-wise system assignment."""
        if atomic_numbers.shape[0] != system_idx.shape[0]:
            raise ValueError("atomic_numbers and system_idx must have same shape[0]")

        self.atomic_numbers = atomic_numbers.to(device=self.device, dtype=torch.long)
        self.system_idx = system_idx.to(device=self.device, dtype=torch.long)

        if self.system_idx.numel() == 0:
            raise ValueError("at least one atom is required")

        self.n_systems = int(self.system_idx.max().item()) + 1
        self.n_atoms_per_system = []
        ptr = [0]
        for idx in range(self.n_systems):
            n_atoms = int((self.system_idx == idx).sum().item())
            self.n_atoms_per_system.append(n_atoms)
            ptr.append(ptr[-1] + n_atoms)

        self.ptr = torch.tensor(ptr, dtype=torch.long, device=self.device)
        self.total_atoms = int(self.atomic_numbers.shape[0])

        atomic_indices = torch.tensor(
            atomic_numbers_to_indices(
                self.atomic_numbers.detach().cpu().numpy(), z_table=self.z_table
            ),
            dtype=torch.long,
            device=self.device,
        ).unsqueeze(-1)

        self.node_attrs = to_one_hot(
            atomic_indices,
            num_classes=len(self.z_table),
            dtype=self.dtype,
        )

    def forward(self, state: Any) -> Dict[str, torch.Tensor]:
        """Compute energies, forces and stresses for one or more systems."""
        if ts is None:
            raise RuntimeError(
                "torch-sim is required to call MaceTorchSimModel.forward"
            )

        if isinstance(state, ts.SimState):
            sim_state = state.clone()
        else:
            state_dict = dict(state)
            if "masses" not in state_dict:
                state_dict["masses"] = torch.ones_like(state_dict["positions"])
            sim_state = ts.SimState(**state_dict)

        if sim_state.device != self.device or sim_state.dtype != self.dtype:
            sim_state = sim_state.to(self.device, self.dtype)

        state_atomic_numbers = getattr(sim_state, "atomic_numbers", None)
        if state_atomic_numbers is None and not self.atomic_numbers_in_init:
            raise ValueError(
                "atomic_numbers must be provided in the constructor or in forward."
            )

        if state_atomic_numbers is not None and self.atomic_numbers_in_init:
            if not torch.equal(state_atomic_numbers, self.atomic_numbers):
                raise ValueError(
                    "atomic_numbers in state do not match constructor values."
                )

        if sim_state.system_idx is None:
            if not hasattr(self, "system_idx"):
                raise ValueError(
                    "system_idx must be provided if not set during initialization"
                )
            sim_state.system_idx = self.system_idx

        if not self.atomic_numbers_in_init:
            cached_atomic_numbers = getattr(self, "atomic_numbers", None)
            cached_system_idx = getattr(self, "system_idx", None)
            needs_setup = state_atomic_numbers is not None and (
                cached_atomic_numbers is None
                or cached_system_idx is None
                or not torch.equal(state_atomic_numbers, cached_atomic_numbers)
                or not torch.equal(sim_state.system_idx, cached_system_idx)
            )
            if needs_setup:
                self.setup_from_system_idx(state_atomic_numbers, sim_state.system_idx)

        wrapped_positions = (
            ts.transforms.pbc_wrap_batched(
                sim_state.positions,
                sim_state.cell,
                sim_state.system_idx,
                sim_state.pbc,
            )
            if sim_state.pbc.any()
            else sim_state.positions
        )

        edge_index, mapping_system, unit_shifts = self.neighbor_list_fn(
            wrapped_positions,
            sim_state.row_vector_cell,
            sim_state.pbc,
            self.r_max,
            sim_state.system_idx,
        )
        shifts = ts.transforms.compute_cell_shifts(
            sim_state.row_vector_cell, unit_shifts, mapping_system
        )

        data_dict = {
            "ptr": self.ptr,
            "node_attrs": self.node_attrs,
            "batch": sim_state.system_idx,
            "pbc": sim_state.pbc,
            "cell": sim_state.row_vector_cell,
            "positions": wrapped_positions,
            "edge_index": edge_index,
            "unit_shifts": unit_shifts,
            "shifts": shifts,
            "total_charge": sim_state.charge,
            "total_spin": sim_state.spin,
        }

        out = self.model(
            data_dict,
            compute_force=self.compute_forces,
            compute_stress=self.compute_stress,
        )

        n_systems = sim_state.n_systems
        results: Dict[str, torch.Tensor] = {}

        energy = out.get("energy")
        if energy is None:
            results["energy"] = torch.zeros(
                n_systems, device=self.device, dtype=self.dtype
            )
        else:
            results["energy"] = energy.detach()

        if self.compute_forces:
            forces = out.get("forces")
            if forces is None:
                forces = torch.zeros_like(sim_state.positions)
            results["forces"] = forces.detach()

        if self.compute_stress:
            stress = out.get("stress")
            if stress is None:
                stress = torch.zeros(
                    n_systems, 3, 3, device=self.device, dtype=self.dtype
                )
            results["stress"] = stress.detach()

        return results
