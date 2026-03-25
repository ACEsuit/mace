"""MACE TorchSim model interface."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Callable, Dict, Optional, Union

import torch

from mace.tools import atomic_numbers_to_indices, utils

log = logging.getLogger(__name__)

_PAD_MULTIPLE = 64
_PAD_HEADROOM = 1.25


def _round_up(value: int, multiple: int) -> int:
    return ((value + multiple - 1) // multiple) * multiple


try:
    import torch_sim as ts
    from torch_sim.models.interface import ModelInterface
    from torch_sim.neighbors import torchsim_nl

    _TORCHSIM_IMPORT_ERROR: Optional[ImportError] = None
except ImportError as exc:
    ts = None  # type: ignore[assignment]
    torchsim_nl = None  # type: ignore[assignment]
    _TORCHSIM_IMPORT_ERROR = exc

    class ModelInterface(torch.nn.Module):  # type: ignore[no-redef]
        """Fallback base class when torch-sim is not installed."""

        def forward(self, state):
            raise NotImplementedError


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
        enable_oeq: bool = False,
        compile_mode: Optional[str] = None,
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
        self._enable_cueq = enable_cueq
        self._enable_oeq = enable_oeq
        self._uses_accelerated = enable_cueq or enable_oeq
        self._use_compile = compile_mode is not None
        self._memory_scales_with = "n_atoms_x_density"
        self.neighbor_list_fn = neighbor_list_fn or torchsim_nl

        self._atom_budget = 0
        self._edge_budget = 0
        self._system_budget = 0
        self._budgets_ready = False
        self._buf_positions = None
        self._buf_node_attrs = None
        self._buf_batch = None
        self._buf_edge_index = None
        self._buf_shifts = None
        self._buf_unit_shifts = None
        self._buf_ptr = None
        self._buf_cell = None
        self._buf_head = None

        if isinstance(model, (str, Path)):
            self.model = torch.load(
                str(model), map_location=self._device, weights_only=False
            )
        elif isinstance(model, torch.nn.Module):
            self.model = model.to(self._device)
        else:
            raise TypeError("model must be a path or torch.nn.Module")

        self.model = self.model.eval().to(device=self._device)
        if self._dtype is not None:
            self.model = self.model.to(dtype=self._dtype)

        if enable_cueq and enable_oeq:
            from mace.cli.convert_e3nn_hybrid import run as run_hybrid

            self.model = run_hybrid(self.model, device=self._device.type)
        elif enable_cueq:
            from mace.cli.convert_e3nn_cueq import run as run_cueq

            try:
                self.model = run_cueq(self.model, device=self._device.type)
            except (RuntimeError, ValueError):
                from mace.cli.convert_e3nn_hybrid import run as run_hybrid

                log.warning(
                    "cueq conv_fusion failed (non-uniform irreps), "
                    "falling back to hybrid (cueq + oeq)"
                )
                self.model = run_hybrid(self.model, device=self._device.type)
                self._enable_oeq = True
                self._uses_accelerated = True
        elif enable_oeq:
            from mace.cli.convert_e3nn_oeq import run as run_oeq

            self.model = run_oeq(self.model, device=self._device.type)

        self.model = self.model.to(device=self._device).eval()

        if compile_mode is not None:
            self._setup_compile(compile_mode)

        for p in self.model.parameters():
            p.requires_grad = False

        self.r_max = float(self.model.r_max)
        self.z_table = utils.AtomicNumberTable(
            [int(z) for z in self.model.atomic_numbers]
        )
        self.model.atomic_numbers = (
            self.model.atomic_numbers.detach().clone().to(device=self._device)
        )
        self._n_elements = len(self.z_table)

        self.atomic_numbers_in_init = atomic_numbers is not None
        if atomic_numbers is not None:
            if system_idx is None:
                system_idx = torch.zeros(
                    len(atomic_numbers), dtype=torch.long, device=self._device
                )
            self.setup_from_system_idx(atomic_numbers, system_idx)

    def _setup_compile(self, compile_mode: str) -> None:
        import torch._dynamo as dynamo

        from mace.tools.compile import configure_autograd_for_compile, simplify

        if self._enable_oeq:
            # oeq ops are opaque to AOTAutograd; autograd.grad must run in eager
            try:
                configure_autograd_for_compile(allow_autograd=True)
                configure_autograd_for_compile(allow_autograd=False)
            except (TypeError, AttributeError):
                pass
        else:
            configure_autograd_for_compile(allow_autograd=True)

        self.model = simplify(self.model)
        self.model = torch.compile(
            self.model,
            mode=compile_mode,
            fullgraph=False,
        )

    def setup_from_system_idx(
        self, atomic_numbers: torch.Tensor, system_idx: torch.Tensor
    ) -> None:
        if atomic_numbers.shape[0] != system_idx.shape[0]:
            raise ValueError("atomic_numbers and system_idx must have same shape[0]")

        self.atomic_numbers = atomic_numbers.to(device=self._device, dtype=torch.long)
        self.system_idx = system_idx.to(device=self._device, dtype=torch.long)

        if self.system_idx.numel() == 0:
            raise ValueError("at least one atom is required")

        self.n_systems = int(self.system_idx.max().item()) + 1
        self.n_atoms_per_system = []
        ptr = [0]
        for idx in range(self.n_systems):
            n_atoms = int((self.system_idx == idx).sum().item())
            self.n_atoms_per_system.append(n_atoms)
            ptr.append(ptr[-1] + n_atoms)

        self.ptr = torch.tensor(ptr, dtype=torch.long, device=self._device)
        self.total_atoms = int(self.atomic_numbers.shape[0])

        atomic_indices = torch.tensor(
            atomic_numbers_to_indices(
                self.atomic_numbers.detach().cpu().numpy(), z_table=self.z_table
            ),
            dtype=torch.long,
            device=self._device,
        ).unsqueeze(-1)

        self.node_attrs = to_one_hot(
            atomic_indices,
            num_classes=self._n_elements,
            dtype=self._dtype,
        )

    def _ensure_budgets(self, n_atoms: int, n_edges: int, n_systems: int) -> None:
        changed = False
        if n_atoms > self._atom_budget:
            old = self._atom_budget
            self._atom_budget = _round_up(int(n_atoms * _PAD_HEADROOM), _PAD_MULTIPLE)
            if old:
                log.warning("Atom budget %d -> %d (recompile)", old, self._atom_budget)
            changed = True
        if n_edges > self._edge_budget:
            old = self._edge_budget
            self._edge_budget = _round_up(int(n_edges * _PAD_HEADROOM), _PAD_MULTIPLE)
            if old:
                log.warning("Edge budget %d -> %d (recompile)", old, self._edge_budget)
            changed = True
        if n_systems > self._system_budget:
            old = self._system_budget
            self._system_budget = _round_up(
                int(n_systems * _PAD_HEADROOM), _PAD_MULTIPLE
            )
            if old:
                log.warning(
                    "System budget %d -> %d (recompile)", old, self._system_budget
                )
            changed = True

        if changed or not self._budgets_ready:
            self._allocate_buffers()
            self._budgets_ready = True
            log.info(
                "Padding budgets: %d atoms, %d edges, %d systems (real: %d, %d, %d)",
                self._atom_budget,
                self._edge_budget,
                self._system_budget,
                n_atoms,
                n_edges,
                n_systems,
            )

    def _allocate_buffers(self) -> None:
        A = self._atom_budget
        E = self._edge_budget
        S = self._system_budget
        dev = self._device
        dt = self._dtype
        cell_scale = self.r_max * 2.0

        self._buf_positions = torch.zeros(A, 3, device=dev, dtype=dt)
        self._buf_node_attrs = torch.zeros(A, self._n_elements, device=dev, dtype=dt)
        self._buf_node_attrs[:, 0] = 1.0
        self._buf_batch = torch.zeros(A, dtype=torch.long, device=dev)

        self._buf_edge_index = torch.full((2, E), A - 1, dtype=torch.long, device=dev)
        self._buf_shifts = torch.zeros(E, 3, device=dev, dtype=dt)
        self._buf_shifts[:, 0] = cell_scale
        self._buf_unit_shifts = torch.zeros(E, 3, device=dev, dtype=dt)
        self._buf_unit_shifts[:, 0] = 1.0

        self._buf_ptr = torch.full((S + 1,), A, dtype=torch.long, device=dev)
        self._buf_ptr[0] = 0
        self._buf_cell = (
            torch.eye(3, device=dev, dtype=dt).unsqueeze(0).expand(S, -1, -1).clone()
            * cell_scale
        )
        self._buf_head = torch.zeros(S, dtype=torch.long, device=dev)

    def _fill_padded_data(
        self,
        data_dict: Dict[str, torch.Tensor],
        n_real_atoms: int,
        n_real_edges: int,
        n_real_systems: int,
    ) -> Dict[str, torch.Tensor]:
        """Copy real data into fixed-size buffers; padding tail is pre-filled."""
        A = self._atom_budget
        S = self._system_budget

        self._buf_node_attrs[:n_real_atoms] = data_dict["node_attrs"]
        self._buf_batch[:n_real_atoms] = data_dict["batch"]
        self._buf_batch[n_real_atoms:] = S - 1

        self._buf_edge_index[:, :n_real_edges] = data_dict["edge_index"]
        self._buf_edge_index[:, n_real_edges:] = A - 1
        self._buf_shifts[:n_real_edges] = data_dict["shifts"]
        self._buf_unit_shifts[:n_real_edges] = data_dict["unit_shifts"]

        real_ptr = data_dict["ptr"]
        self._buf_ptr[: n_real_systems + 1] = real_ptr
        self._buf_ptr[n_real_systems + 1 : S] = n_real_atoms
        self._buf_ptr[S] = A

        self._buf_cell[:n_real_systems] = data_dict["cell"]
        self._buf_head[:n_real_systems] = (
            data_dict["head"]
            if "head" in data_dict
            else torch.zeros(n_real_systems, dtype=torch.long, device=self._device)
        )

        pad_count = A - n_real_atoms
        pad_pos = torch.zeros(pad_count, 3, device=self._device, dtype=self._dtype)
        padded_positions = torch.cat([data_dict["positions"], pad_pos])

        padded: Dict[str, torch.Tensor] = {
            "positions": padded_positions,
            "node_attrs": self._buf_node_attrs,
            "batch": self._buf_batch,
            "edge_index": self._buf_edge_index,
            "shifts": self._buf_shifts,
            "unit_shifts": self._buf_unit_shifts,
            "ptr": self._buf_ptr,
            "cell": self._buf_cell,
            "head": self._buf_head,
            "pbc": data_dict["pbc"],
        }

        if "displacement" in data_dict:
            pad_disp = torch.zeros(
                S - n_real_systems, 3, 3, device=self._device, dtype=self._dtype
            )
            padded["displacement"] = torch.cat([data_dict["displacement"], pad_disp])

        return padded

    @property
    def device(self) -> torch.device:
        return self._device

    @property
    def dtype(self) -> torch.dtype:
        return self._dtype

    @property
    def compute_forces(self) -> bool:
        return self._compute_forces

    @property
    def compute_stress(self) -> bool:
        return self._compute_stress

    def forward(self, state: Any) -> Dict[str, torch.Tensor]:
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

        cutoff = self.r_max
        if self.neighbor_list_fn is torchsim_nl:
            cutoff = torch.as_tensor(
                self.r_max,
                device=wrapped_positions.device,
                dtype=wrapped_positions.dtype,
            )

        edge_index, mapping_system, unit_shifts = self.neighbor_list_fn(
            wrapped_positions,
            sim_state.row_vector_cell,
            sim_state.pbc,
            cutoff,
            sim_state.system_idx,
        )
        shifts = ts.transforms.compute_cell_shifts(
            sim_state.row_vector_cell, unit_shifts, mapping_system
        )

        n_real_atoms = wrapped_positions.shape[0]
        n_real_edges = edge_index.shape[1]
        wrapped_positions = wrapped_positions.requires_grad_(True)

        data_dict: Dict[str, torch.Tensor] = {
            "ptr": self.ptr,
            "node_attrs": self.node_attrs,
            "batch": sim_state.system_idx,
            "head": torch.zeros(self.n_systems, dtype=torch.long, device=self._device),
            "pbc": sim_state.pbc,
            "cell": sim_state.row_vector_cell,
            "positions": wrapped_positions,
            "edge_index": edge_index,
            "unit_shifts": unit_shifts,
            "shifts": shifts,
            "total_charge": sim_state.charge,
            "total_spin": sim_state.spin,
        }

        oeq_compile = self._use_compile and self._enable_oeq
        if oeq_compile and self._compute_stress:
            displacement = torch.zeros(
                (self.n_systems, 3, 3),
                dtype=wrapped_positions.dtype,
                device=self._device,
            )
            displacement = displacement + wrapped_positions.sum() * 0.0
            data_dict["displacement"] = displacement

        if self._use_compile:
            self._ensure_budgets(n_real_atoms, n_real_edges, self.n_systems)
            data_dict = self._fill_padded_data(
                data_dict,
                n_real_atoms,
                n_real_edges,
                self.n_systems,
            )

        training = self._use_compile and not oeq_compile

        out = self.model(
            data_dict,
            compute_force=self._compute_forces,
            compute_stress=self._compute_stress,
            training=training,
        )

        n_systems = self.n_systems
        results: Dict[str, torch.Tensor] = {}

        energy = out.get("energy")
        if energy is None:
            results["energy"] = torch.zeros(
                n_systems, device=self.device, dtype=self.dtype
            )
        else:
            results["energy"] = energy[:n_systems].detach()

        if self._compute_forces:
            forces = out.get("forces")
            if forces is None:
                forces = torch.zeros_like(sim_state.positions)
            results["forces"] = forces[:n_real_atoms].detach()

        if self._compute_stress:
            stress = out.get("stress")
            if stress is None:
                stress = torch.zeros(
                    n_systems, 3, 3, device=self.device, dtype=self.dtype
                )
            results["stress"] = stress[:n_systems].detach()

        return results
