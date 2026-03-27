"""MACE TorchSim model interface."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import TYPE_CHECKING, Callable, cast

import torch

from mace.modules.models import MACE
from mace.tools import atomic_numbers_to_indices, utils

try:
    from torch_sim.models.interface import ModelInterface
    from torch_sim.neighbors import torchsim_nl
    from torch_sim.transforms import compute_cell_shifts, pbc_wrap_batched

    _TORCHSIM_IMPORT_ERROR: ImportError | None = None
except ImportError as exc:
    _TORCHSIM_IMPORT_ERROR = exc
    ModelInterface = None  # type: ignore[assignment]
    torchsim_nl = None  # type: ignore[assignment]
    compute_cell_shifts = None  # type: ignore[assignment]
    pbc_wrap_batched = None  # type: ignore[assignment]

_TSModelInterface = ModelInterface if ModelInterface is not None else object

if TYPE_CHECKING:
    from torch_sim.state import SimState
log = logging.getLogger(__name__)

_PAD_MULTIPLE = 64
_PAD_HEADROOM = 1.25


def _round_up(value: int, multiple: int) -> int:
    """Round up to the nearest multiple."""
    return ((value + multiple - 1) // multiple) * multiple


def to_one_hot(
    indices: torch.Tensor, num_classes: int, dtype: torch.dtype
) -> torch.Tensor:
    """Generate one-hot vectors from class indices."""
    shape = indices.shape[:-1] + (num_classes,)
    out = torch.zeros(shape, device=indices.device, dtype=dtype).view(shape)
    out.scatter_(dim=-1, index=indices, value=1)
    return out.view(*shape)


class MaceTorchSimModel(_TSModelInterface):
    """TorchSim wrapper around a MACE model."""

    def __init__(
        self,
        model: str | Path | MACE,
        device: torch.device | None = None,
        dtype: torch.dtype = torch.float64,
        neighbor_list_fn: Callable | None = None,
        compute_forces: bool = True,
        compute_stress: bool = True,
        enable_cueq: bool = False,
        enable_oeq: bool = False,
        compile_mode: str | None = None,
        head: str | int | None = None,
        atomic_numbers: torch.Tensor | None = None,
        system_idx: torch.Tensor | None = None,
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
        self._use_cudagraphs = compile_mode in ("reduce-overhead", "max-autotune")
        self._memory_scales_with = "n_atoms_x_density"
        self.neighbor_list_fn = neighbor_list_fn or torchsim_nl

        self._atom_budget = 0
        self._edge_budget = 0
        self._system_budget = 0
        self._budgets_ready = False
        self._buf_node_attrs: torch.Tensor | None = None
        self._buf_batch: torch.Tensor | None = None
        self._buf_edge_index: torch.Tensor | None = None
        self._buf_shifts: torch.Tensor | None = None
        self._buf_unit_shifts: torch.Tensor | None = None
        self._buf_ptr: torch.Tensor | None = None
        self._buf_cell: torch.Tensor | None = None
        self._buf_head: torch.Tensor | None = None
        self._buf_total_charge: torch.Tensor | None = None
        self._buf_total_spin: torch.Tensor | None = None

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

        assert isinstance(self.model, MACE), (
            "model must be derived from MACE base class"
        )

        self.r_max = self.model.r_max
        self.z_table = utils.AtomicNumberTable(
            [int(z) for z in self.model.atomic_numbers]
        )
        self.model.atomic_numbers = (
            self.model.atomic_numbers.detach().clone().to(device=self._device)
        )
        self._n_elements = len(self.z_table)

        available_heads = list(getattr(self.model, "heads", ["Default"]))
        self._head_index = self._resolve_head(head, available_heads)

        for p in self.model.parameters():
            p.requires_grad = False

        if compile_mode is not None:
            self._setup_compile(compile_mode)

        self._cached_z: torch.Tensor | None = None
        self._cached_system_idx: torch.Tensor | None = None
        self._z_fixed = atomic_numbers is not None
        self.node_attrs: torch.Tensor = torch.empty(0)
        self.n_systems: int = 0
        self.n_atoms_per_system: list[int] = []
        self.ptr: torch.Tensor = torch.empty(0)

        if atomic_numbers is not None:
            if system_idx is None:
                system_idx = torch.zeros(
                    len(atomic_numbers), dtype=torch.long, device=self._device
                )
            self._update_system_layout(atomic_numbers, system_idx)

    @staticmethod
    def _resolve_head(head: str | int | None, available_heads: list) -> int:
        if isinstance(head, int):
            if head < 0 or head >= len(available_heads):
                raise ValueError(
                    f"Head index {head} out of range for {available_heads}"
                )
            return head
        if isinstance(head, str):
            if head not in available_heads:
                raise ValueError(
                    f"Head {head!r} not in available heads {available_heads}"
                )
            return available_heads.index(head)
        if len(available_heads) == 1:
            return 0
        default = [h for h in available_heads if h.lower() == "default"]
        if default:
            return available_heads.index(default[0])
        log.warning(
            "No head specified for multi-head model, defaulting to '%s'. "
            "Available heads: %s",
            available_heads[0],
            available_heads,
        )
        return 0

    def _setup_compile(self, compile_mode: str) -> None:
        # Side effect: ensure Dynamo is initialized before torch.compile.
        import torch._dynamo as dynamo  # pylint: disable=unused-import  # noqa: F401

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
        self.model = cast(  # make type checker happy that this acts like a MACE model
            MACE,
            torch.compile(self.model, mode=compile_mode, fullgraph=False),
        )

    def _setup_node_attrs(self, atomic_numbers: torch.Tensor) -> None:
        self.node_attrs = to_one_hot(
            torch.tensor(
                atomic_numbers_to_indices(
                    atomic_numbers.detach().cpu().numpy(), z_table=self.z_table
                ),
                dtype=torch.long,
                device=self._device,
            ).unsqueeze(-1),
            num_classes=self._n_elements,
            dtype=self._dtype,
        )

    def _setup_ptr(self, system_idx: torch.Tensor) -> None:
        counts = torch.bincount(system_idx)
        self.n_systems = len(counts)
        self.n_atoms_per_system = counts.tolist()
        self.ptr = torch.cat([counts.new_zeros(1), counts.cumsum(0)]).to(
            device=self._device
        )

    def _update_system_layout(
        self, atomic_numbers: torch.Tensor, system_idx: torch.Tensor
    ) -> None:
        if atomic_numbers.shape[0] != system_idx.shape[0]:
            raise ValueError("atomic_numbers and system_idx must have same shape[0]")
        if atomic_numbers.numel() == 0:
            raise ValueError("at least one atom is required")
        z = atomic_numbers.to(device=self._device, dtype=torch.long)
        idx = system_idx.to(device=self._device, dtype=torch.long)
        self._cached_z = z
        self._cached_system_idx = idx
        self._setup_node_attrs(z)
        self._setup_ptr(idx)

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
        self._buf_head = torch.full(
            (S,), self._head_index, dtype=torch.long, device=dev
        )
        self._buf_total_charge = torch.zeros(S, device=dev, dtype=dt)
        self._buf_total_spin = torch.ones(S, device=dev, dtype=dt)

    def _fill_padded_data(
        self,
        data_dict: dict[str, torch.Tensor],
        optionals: dict[str, torch.Tensor | None],
        n_real_atoms: int,
        n_real_edges: int,
        n_real_systems: int,
    ) -> tuple[dict[str, torch.Tensor], dict[str, torch.Tensor | None]]:
        """Copy real data into fixed-size buffers; padding tail is pre-filled."""
        assert (
            self._buf_node_attrs is not None
            and self._buf_batch is not None
            and self._buf_edge_index is not None
            and self._buf_shifts is not None
            and self._buf_unit_shifts is not None
            and self._buf_ptr is not None
            and self._buf_cell is not None
            and self._buf_head is not None
        ), "_allocate_buffers() must be called before _fill_padded_data()"
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
        self._buf_head.fill_(self._head_index)
        self._buf_head[:n_real_systems] = data_dict["head"]

        pad_count = A - n_real_atoms
        pad_pos = torch.zeros(pad_count, 3, device=self._device, dtype=self._dtype)
        padded_positions = torch.cat([data_dict["positions"], pad_pos])

        padded: dict[str, torch.Tensor] = {
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
            disp = data_dict["displacement"]
            assert disp is not None
            pad_disp = torch.zeros(
                S - n_real_systems, 3, 3, device=self._device, dtype=self._dtype
            )
            padded["displacement"] = torch.cat([disp, pad_disp])

        assert self._buf_total_charge is not None and self._buf_total_spin is not None
        buf_map: dict[str, tuple[torch.Tensor, float]] = {
            "total_charge": (self._buf_total_charge, 0.0),
            "total_spin": (self._buf_total_spin, 1.0),
        }
        padded_opt: dict[str, torch.Tensor | None] = {}
        for key, (buf, pad_fill) in buf_map.items():
            val = optionals.get(key)
            if val is None:
                padded_opt[key] = None
            else:
                buf[:n_real_systems] = val[:n_real_systems]
                buf[n_real_systems:] = pad_fill
                padded_opt[key] = buf

        return padded, padded_opt

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

    def _ensure_layout(self, state: SimState) -> None:
        state_z = state.atomic_numbers
        if self._z_fixed:
            if self._cached_z is not None and not torch.equal(state_z, self._cached_z):
                raise ValueError(
                    "atomic_numbers in state do not match constructor values."
                )
            return
        if self._cached_z is None or not torch.equal(state_z, self._cached_z):
            self._update_system_layout(state_z, state.system_idx)
        elif self._cached_system_idx is None or not torch.equal(
            state.system_idx, self._cached_system_idx
        ):
            self._setup_ptr(state.system_idx)
            self._cached_system_idx = state.system_idx

    def forward(self, state: SimState, **_kwargs) -> dict[str, torch.Tensor]:  # ty:ignore[invalid-method-override]
        self._ensure_layout(state)

        wrapped_positions = (
            pbc_wrap_batched(
                state.positions,
                state.cell,
                state.system_idx,
                state.pbc,
            )
            if state.pbc.any()
            else state.positions
        )

        edge_index, mapping_system, unit_shifts = self.neighbor_list_fn(
            wrapped_positions,
            state.row_vector_cell,
            state.pbc,
            self.r_max,
            state.system_idx,
        )
        shifts = compute_cell_shifts(state.row_vector_cell, unit_shifts, mapping_system)

        n_real_atoms = wrapped_positions.shape[0]
        n_real_edges = edge_index.shape[1]
        wrapped_positions = wrapped_positions.requires_grad_(True)

        data_dict: dict[str, torch.Tensor] = {
            "ptr": self.ptr,
            "node_attrs": self.node_attrs,
            "batch": state.system_idx,
            "head": torch.full(
                (self.n_systems,),
                self._head_index,
                dtype=torch.long,
                device=self._device,
            ),
            "pbc": state.pbc,
            "cell": state.row_vector_cell,
            "positions": wrapped_positions,
            "edge_index": edge_index,
            "unit_shifts": unit_shifts,
            "shifts": shifts,
        }
        optionals: dict[str, torch.Tensor | None] = {
            "total_charge": getattr(state, "charge", None),
            "total_spin": getattr(state, "spin", None),
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
            data_dict, optionals = self._fill_padded_data(
                data_dict,
                optionals,
                n_real_atoms,
                n_real_edges,
                self.n_systems,
            )

        data_dict: dict[str, torch.Tensor | None] = {**data_dict, **optionals}

        training = self._use_compile and not oeq_compile

        if self._use_cudagraphs:
            torch.compiler.cudagraph_mark_step_begin()

        out = self.model(
            data_dict,
            compute_force=self._compute_forces,
            compute_stress=self._compute_stress,
            training=training,
        )

        n_systems = self.n_systems
        results: dict[str, torch.Tensor] = {}

        energy = out.get("energy")
        if energy is None:
            results["energy"] = torch.zeros(
                n_systems, device=self.device, dtype=self.dtype
            )
        else:
            e = energy[:n_systems].detach()
            results["energy"] = e.clone() if self._use_cudagraphs else e

        if self._compute_forces:
            forces = out.get("forces")
            if forces is None:
                forces = torch.zeros_like(state.positions)
            f = forces[:n_real_atoms].detach()
            results["forces"] = f.clone() if self._use_cudagraphs else f

        if self._compute_stress:
            stress = out.get("stress")
            if stress is None:
                stress = torch.zeros(
                    n_systems, 3, 3, device=self.device, dtype=self.dtype
                )
            s = stress[:n_systems].detach()
            results["stress"] = s.clone() if self._use_cudagraphs else s

        return results
