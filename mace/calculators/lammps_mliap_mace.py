import importlib
import json
import logging
import math
import os
import sys
import time
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import torch
from ase.data import chemical_symbols
from e3nn.util.jit import compile_mode

try:
    from lammps.mliap.mliap_unified_abc import MLIAPUnified
except ImportError:

    class MLIAPUnified:  # pragma: no cover
        def __init__(self):
            pass


# -----------------------------
# Helpers / configuration
# -----------------------------


def _env_bool(name: str, default: bool) -> bool:
    return os.environ.get(name, str(default)).lower() in ("true", "1", "t", "yes", "y")


def _env_int(name: str, default: int) -> int:
    try:
        return int(os.environ.get(name, str(default)))
    except (TypeError, ValueError):
        return default


def _env_float(name: str, default: float) -> float:
    try:
        return float(os.environ.get(name, str(default)))
    except (TypeError, ValueError):
        return default


def _parse_vec3(s: str, default=(0.0, 0.0, 0.0)) -> Tuple[float, float, float]:
    if s is None:
        return default
    s = str(s).strip()
    if not s:
        return default
    parts = [p.strip() for p in s.replace(";", ",").split(",")]
    if len(parts) != 3:
        return default
    try:
        return (float(parts[0]), float(parts[1]), float(parts[2]))
    except (TypeError, ValueError):
        return default


@contextmanager
def timer(name: str, enabled: bool = True):
    if not enabled:
        yield
        return
    start = time.perf_counter()
    try:
        yield
    finally:
        elapsed = time.perf_counter() - start
        logging.info(f"Timer - {name}: {elapsed*1000:.3f} ms")


class MACELammpsConfig:
    """Configuration settings for MACE-LAMMPS integration."""

    def __init__(self):
        # Existing debugging knobs
        self.debug_time = _env_bool("MACE_TIME", False)
        self.debug_profile = _env_bool("MACE_PROFILE", False)
        self.profile_start_step = _env_int("MACE_PROFILE_START", 5)
        self.profile_end_step = _env_int("MACE_PROFILE_END", 10)

        self.allow_cpu = _env_bool("MACE_ALLOW_CPU", False)
        self.force_cpu = _env_bool("MACE_FORCE_CPU", False)

        # MACEField output control
        # Set stride <= 0 to disable that quantity.
        self.stride_pol = _env_int("MACEFIELD_STRIDE_POL", 100)
        self.stride_becs = _env_int("MACEFIELD_STRIDE_BECS", 0)
        self.stride_alpha = _env_int("MACEFIELD_STRIDE_POLARIZABILITY", 0)

        # Optional JSONL dump (one line per step where *any* field output is computed)
        self.jsonl_path = os.environ.get("MACEFIELD_JSONL", "").strip()
        self.jsonl_flush = _env_bool("MACEFIELD_JSONL_FLUSH", True)

        # If True, keep last computed P/BEC/alpha available between stride steps
        self.hold_last = _env_bool("MACEFIELD_HOLD_LAST", True)

        # Electric-field driver (for time-dependent E(t))
        #   constant: use MACEFIELD_EFIELD or (0,0,0)
        #   cos_step / sin_step: use step and PERIOD_STEPS
        #   cos_time / sin_time: use time and FREQ (1/ps etc) and DT (ps)
        #   runtime: read vector from macefield_runtime (if present)
        self.efield_mode = (
            os.environ.get("MACEFIELD_EFIELD_MODE", "constant").strip().lower()
        )
        self.efield_const = _parse_vec3(
            os.environ.get("MACEFIELD_EFIELD", ""), (0.0, 0.0, 0.0)
        )
        self.efield_amp = _parse_vec3(
            os.environ.get("MACEFIELD_E0", ""), (0.0, 0.0, 0.0)
        )
        self.efield_offset = _parse_vec3(
            os.environ.get("MACEFIELD_EOFFSET", ""), (0.0, 0.0, 0.0)
        )
        self.efield_phase = _env_float("MACEFIELD_PHASE", 0.0)

        self.period_steps = _env_int("MACEFIELD_PERIOD_STEPS", 0)
        self.freq = _env_float(
            "MACEFIELD_FREQ", 0.0
        )  # in 1/(time units) for *_time modes
        self.dt = _env_float(
            "MACEFIELD_DT", 0.0
        )  # time step, e.g. ps for units metal (if not detectable)

        self.step0 = _env_int("MACEFIELD_STEP0", 0)

    def want_any_field_output(self, step: int) -> bool:
        return self.want_pol(step) or self.want_becs(step) or self.want_alpha(step)

    def want_pol(self, step: int) -> bool:
        return self.stride_pol > 0 and (step % self.stride_pol == 0)

    def want_becs(self, step: int) -> bool:
        return self.stride_becs > 0 and (step % self.stride_becs == 0)

    def want_alpha(self, step: int) -> bool:
        return self.stride_alpha > 0 and (step % self.stride_alpha == 0)


@dataclass
class FieldOutputs:
    electric_field: Optional[torch.Tensor] = None  # [3]
    polarization: Optional[torch.Tensor] = None  # [3]
    polarizability: Optional[torch.Tensor] = None  # [3,3]
    becs: Optional[torch.Tensor] = None  # [natoms,3,3]


def _try_import_runtime():
    try:
        # Dynamic import avoids pylint E0401 when module is optional at runtime.
        return importlib.import_module("macefield_runtime")
    except (ModuleNotFoundError, ImportError):
        return None


_RUNTIME = _try_import_runtime()


def _runtime_set(key: str, value):
    if _RUNTIME is None:
        return
    try:
        _RUNTIME.set(key, value)
    except (AttributeError, TypeError, ValueError, RuntimeError):
        # Don’t hard-fail MD if runtime helper isn’t present/compatible
        pass


def _as_cpu_numpy(x: Optional[torch.Tensor]):
    if x is None:
        return None
    return x.detach().cpu().numpy()


def _as_python(obj):
    # For JSON serialisation
    if obj is None:
        return None
    if isinstance(obj, (float, int, str)):
        return obj
    try:
        import numpy as np

        if isinstance(obj, np.ndarray):
            return obj.tolist()
    except (ImportError, ModuleNotFoundError):
        pass
    try:
        return obj.tolist()
    except (AttributeError, TypeError):
        return str(obj)


# -----------------------------
# Electric field driver
# -----------------------------


class ElectricFieldDriver:
    def __init__(self, cfg: MACELammpsConfig):
        self.cfg = cfg

    def _dt_from_data(self, data) -> float:
        # Best effort: ML-IAP data object may or may not expose dt
        for name in ("dt", "timestep", "time_step"):
            if hasattr(data, name):
                try:
                    v = float(getattr(data, name))
                    if v > 0:
                        return v
                except (TypeError, ValueError):
                    pass
        return self.cfg.dt

    def field(self, step: int, data=None) -> torch.Tensor:
        mode = self.cfg.efield_mode

        if mode == "runtime":
            if _RUNTIME is not None:
                try:
                    v = _RUNTIME.get("electric_field", None)
                    if v is not None:
                        ex, ey, ez = v
                        return torch.tensor(
                            [ex, ey, ez], dtype=torch.get_default_dtype()
                        )
                except (AttributeError, TypeError, ValueError, KeyError):
                    pass
            # fallback
            ex, ey, ez = self.cfg.efield_const
            return torch.tensor([ex, ey, ez], dtype=torch.get_default_dtype())

        if mode == "constant":
            ex, ey, ez = self.cfg.efield_const
            return torch.tensor([ex, ey, ez], dtype=torch.get_default_dtype())

        ax, ay, az = self.cfg.efield_amp
        ox, oy, oz = self.cfg.efield_offset
        phase = self.cfg.efield_phase

        if mode in ("cos_step", "sin_step"):
            if self.cfg.period_steps <= 0:
                # fallback: constant
                return torch.tensor([ox, oy, oz], dtype=torch.get_default_dtype())
            theta = 2.0 * math.pi * (float(step) / float(self.cfg.period_steps)) + phase
            s = math.cos(theta) if mode == "cos_step" else math.sin(theta)
            return torch.tensor(
                [ox + ax * s, oy + ay * s, oz + az * s], dtype=torch.get_default_dtype()
            )

        if mode in ("cos_time", "sin_time"):
            dt = self._dt_from_data(data)
            if dt <= 0.0 or self.cfg.freq == 0.0:
                return torch.tensor([ox, oy, oz], dtype=torch.get_default_dtype())
            t = float(step) * dt
            theta = 2.0 * math.pi * self.cfg.freq * t + phase
            s = math.cos(theta) if mode == "cos_time" else math.sin(theta)
            return torch.tensor(
                [ox + ax * s, oy + ay * s, oz + az * s], dtype=torch.get_default_dtype()
            )

        # Unknown mode -> constant
        ex, ey, ez = self.cfg.efield_const
        return torch.tensor([ex, ey, ez], dtype=torch.get_default_dtype())


# -----------------------------
# Model wrappers
# -----------------------------


@compile_mode("script")
class MACEEdgeForcesWrapper(torch.nn.Module):
    """Wrapper that adds per-pair force computation to a MACE model."""

    def __init__(self, model: torch.nn.Module, **kwargs):
        super().__init__()
        self.model = model
        self.register_buffer("atomic_numbers", model.atomic_numbers)
        self.register_buffer("r_max", model.r_max)
        self.register_buffer("num_interactions", model.num_interactions)
        self.register_buffer(
            "total_charge",
            kwargs.get(
                "total_charge", torch.tensor([0.0], dtype=torch.get_default_dtype())
            ),
        )
        self.register_buffer(
            "total_spin",
            kwargs.get(
                "total_spin", torch.tensor([1.0], dtype=torch.get_default_dtype())
            ),
        )

        if not hasattr(model, "heads"):
            model.heads = ["Default"]

        head_name = kwargs.get("head", model.heads[-1])
        head_idx = model.heads.index(head_name)
        self.register_buffer("head", torch.tensor([head_idx], dtype=torch.long))

        for p in self.model.parameters():
            p.requires_grad = False

    def forward(
        self, data: Dict[str, torch.Tensor]
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        data["head"] = self.head
        data["total_charge"] = self.total_charge
        data["total_spin"] = self.total_spin

        out = self.model(
            data,
            training=False,
            compute_force=False,
            compute_virials=False,
            compute_stress=False,
            compute_displacement=False,
            compute_edge_forces=True,
            lammps_mliap=True,
        )

        node_energy = out["node_energy"]
        pair_forces = out["edge_forces"]
        total_energy = out["energy"][0]

        if pair_forces is None:
            pair_forces = torch.zeros_like(data["vectors"])

        return total_energy, node_energy, pair_forces


@compile_mode("script")
class MACEFieldEdgeForcesWrapper(torch.nn.Module):
    """Wrapper for MACEField: edge forces + (optionally) P/BEC/alpha."""

    def __init__(self, model: torch.nn.Module, **kwargs):
        super().__init__()
        self.model = model
        self.register_buffer("atomic_numbers", model.atomic_numbers)
        self.register_buffer("r_max", model.r_max)
        self.register_buffer("num_interactions", model.num_interactions)
        self.register_buffer(
            "total_charge",
            kwargs.get(
                "total_charge", torch.tensor([0.0], dtype=torch.get_default_dtype())
            ),
        )
        self.register_buffer(
            "total_spin",
            kwargs.get(
                "total_spin", torch.tensor([1.0], dtype=torch.get_default_dtype())
            ),
        )

        if not hasattr(model, "heads"):
            model.heads = ["Default"]
        head_name = kwargs.get("head", model.heads[-1])
        head_idx = model.heads.index(head_name)
        self.register_buffer("head", torch.tensor([head_idx], dtype=torch.long))

        for p in self.model.parameters():
            p.requires_grad = False

    def forward(
        self,
        data: Dict[str, torch.Tensor],
        electric_field: torch.Tensor,
        compute_polarization: bool,
        compute_becs: bool,
        compute_polarizability: bool,
        # IMPORTANT: BECs require grads wrt positions; in practice this is most reliable
        # if we also request force computation (so positions/vectors are marked for grad
        # in the internal graph preparation). We discard the forces and still use edge_forces.
        compute_force_for_grads: bool,
    ) -> Tuple[
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        Optional[torch.Tensor],
        Optional[torch.Tensor],
        Optional[torch.Tensor],
    ]:
        data["head"] = self.head
        data["total_charge"] = self.total_charge
        data["total_spin"] = self.total_spin

        # Ensure field is present for the model even if caller doesn’t include it in batch
        data["electric_field"] = electric_field.view(1, 3)

        out = self.model(
            data,
            training=False,
            compute_force=compute_force_for_grads,
            compute_virials=False,
            compute_stress=False,
            compute_displacement=False,
            compute_edge_forces=True,
            compute_polarization=compute_polarization,
            compute_becs=compute_becs,
            compute_polarizability=compute_polarizability,
            electric_field=electric_field.view(1, 3),
            lammps_mliap=True,
        )

        node_energy = out["node_energy"]
        pair_forces = out["edge_forces"]
        total_energy = out["energy"][0]

        if pair_forces is None:
            pair_forces = torch.zeros_like(data["vectors"])

        pol = out.get("polarization", None)
        becs = out.get("becs", None)
        alpha = out.get("polarizability", None)

        # LAMMPS runs one “graph”; squeeze graph dimension if present
        if pol is not None and pol.dim() == 2 and pol.shape[0] == 1:
            pol = pol[0]
        if alpha is not None and alpha.dim() == 3 and alpha.shape[0] == 1:
            alpha = alpha[0]

        return total_energy, node_energy, pair_forces, pol, becs, alpha


# -----------------------------
# LAMMPS MLIAP class
# -----------------------------


class LAMMPS_MLIAP_MACE(MLIAPUnified):
    """MACE/MACEField integration for LAMMPS using the MLIAP unified interface."""

    def __init__(self, model, **kwargs):
        super().__init__()
        self.config = MACELammpsConfig()
        self.efield_driver = ElectricFieldDriver(self.config)

        # Detect “field-capable” model
        # hasattr() should not raise (pylint-friendly)
        self.is_macefield = bool(
            hasattr(model, "field_feats") or hasattr(model, "field_linear")
        )

        # Build the appropriate wrapper
        if self.is_macefield:
            self.model = MACEFieldEdgeForcesWrapper(model, **kwargs)
        else:
            self.model = MACEEdgeForcesWrapper(model, **kwargs)

        self.element_types = [chemical_symbols[s] for s in model.atomic_numbers]
        self.num_species = len(self.element_types)
        self.rcutfac = 0.5 * float(model.r_max)
        self.ndescriptors = 1
        self.nparams = 1
        self.dtype = model.r_max.dtype

        self.device = torch.device("cpu")
        self.initialized = False

        # Step counter (for E(t) and strides)
        self.step = int(self.config.step0)

        # Last computed field outputs (optionally held between stride steps)
        self.last_field = FieldOutputs()

        # Initialise runtime store (optional)
        _runtime_set("is_macefield", bool(self.is_macefield))
        _runtime_set("element_types", list(self.element_types))

    def _initialize_device(self, data):
        using_kokkos = "kokkos" in data.__class__.__module__.lower()

        if using_kokkos and not self.config.force_cpu:
            device = torch.as_tensor(data.elems).device
            if device.type == "cpu" and not self.config.allow_cpu:
                raise ValueError(
                    "GPU requested but tensors are on CPU. Set MACE_ALLOW_CPU=true to allow CPU computation."
                )
        else:
            device = torch.device("cpu")

        self.device = device
        self.model = self.model.to(device)
        logging.info(f"MACE model initialized on device: {device}")
        self.initialized = True

    def compute_forces(self, data):
        natoms = data.nlocal
        ntotal = data.ntotal
        nghosts = ntotal - natoms
        npairs = data.npairs
        species = torch.as_tensor(data.elems, dtype=torch.int64)

        if not self.initialized:
            self._initialize_device(data)

        self.step += 1
        self._manage_profiling()

        if natoms == 0 or npairs <= 1:
            return

        with timer("total_step", enabled=self.config.debug_time):
            with timer("prepare_batch", enabled=self.config.debug_time):
                batch = self._prepare_batch(data, natoms, nghosts, species)

            with timer("model_forward", enabled=self.config.debug_time):
                if not self.is_macefield:
                    _, atom_energies, pair_forces = self.model(batch)
                    pol = becs = alpha = None
                    efield = None
                else:
                    # Always apply E(t) each step (it affects energy/forces).
                    efield = (
                        self.efield_driver.field(self.step, data=data)
                        .to(self.device)
                        .to(self.dtype)
                    )

                    # Decide which expensive derivatives to compute this step
                    do_pol = self.config.want_pol(self.step)
                    do_becs = self.config.want_becs(self.step)
                    do_alpha = self.config.want_alpha(self.step)

                    # If we compute BECs, we also request force computation to make sure
                    # positions are differentiable in the internal graph prep.
                    compute_force_for_grads = bool(do_becs)

                    _, atom_energies, pair_forces, pol, becs, alpha = self.model(
                        batch,
                        electric_field=efield,
                        compute_polarization=bool(do_pol or do_becs or do_alpha),
                        compute_becs=bool(do_becs),
                        compute_polarizability=bool(do_alpha),
                        compute_force_for_grads=compute_force_for_grads,
                    )

                if self.device.type != "cpu":
                    torch.cuda.synchronize()

            with timer("update_lammps", enabled=self.config.debug_time):
                self._update_lammps_data(data, atom_energies, pair_forces, natoms)

            # Post: store / dump field outputs (if enabled and computed on this step)
            if self.is_macefield:
                self._handle_field_outputs(efield, pol, becs, alpha, natoms)

    def _prepare_batch(self, data, natoms, nghosts, species):
        # Note: we always pass lammps_class/natoms so the core model can build positions/cell etc.
        return {
            "vectors": torch.as_tensor(data.rij).to(self.dtype).to(self.device),
            "node_attrs": torch.nn.functional.one_hot(
                species.to(self.device), num_classes=self.num_species
            ).to(self.dtype),
            "edge_index": torch.stack(
                [
                    torch.as_tensor(data.pair_j, dtype=torch.int64).to(self.device),
                    torch.as_tensor(data.pair_i, dtype=torch.int64).to(self.device),
                ],
                dim=0,
            ),
            "batch": torch.zeros(natoms, dtype=torch.int64, device=self.device),
            "lammps_class": data,
            "natoms": (natoms, nghosts),
        }

    def _update_lammps_data(self, data, atom_energies, pair_forces, natoms):
        if self.dtype == torch.float32:
            pair_forces = pair_forces.double()
        eatoms = torch.as_tensor(data.eatoms)
        eatoms.copy_(atom_energies[:natoms])
        data.energy = torch.sum(atom_energies[:natoms])
        data.update_pair_forces_gpu(pair_forces)

    def _handle_field_outputs(self, efield, pol, becs, alpha, natoms: int):
        step = self.step
        did_any = self.config.want_any_field_output(step)

        # If none requested this step, either hold previous or clear
        if not did_any:
            if not self.config.hold_last:
                self.last_field = FieldOutputs()
                _runtime_set("polarization", None)
                _runtime_set("becs", None)
                _runtime_set("polarizability", None)
            # Always still expose current E(t)
            if efield is not None:
                _runtime_set("electric_field", _as_python(_as_cpu_numpy(efield)))
            _runtime_set("step", int(step))
            return

        # Update last-field cache (hold-last behaviour is automatic by only updating
        # what was computed; unset outputs remain whatever they were, unless hold_last=False)
        if efield is not None:
            self.last_field.electric_field = efield
        if pol is not None:
            self.last_field.polarization = pol
        if alpha is not None:
            self.last_field.polarizability = alpha
        if becs is not None:
            # In LAMMPS mode, model may include ghosts; keep local atoms only when writing
            try:
                if becs.shape[0] >= natoms:
                    becs = becs[:natoms]
            except (AttributeError, IndexError, TypeError, RuntimeError):
                pass
            self.last_field.becs = becs

        # Push to runtime module (so LAMMPS python variables can read it)
        _runtime_set("step", int(step))
        _runtime_set(
            "electric_field", _as_python(_as_cpu_numpy(self.last_field.electric_field))
        )
        _runtime_set(
            "polarization", _as_python(_as_cpu_numpy(self.last_field.polarization))
        )
        _runtime_set(
            "polarizability", _as_python(_as_cpu_numpy(self.last_field.polarizability))
        )
        _runtime_set("becs", _as_python(_as_cpu_numpy(self.last_field.becs)))

        # Optional JSONL dump (only on steps where something was computed)
        if self.config.jsonl_path:
            rec = {
                "step": int(step),
                "electric_field": _as_python(
                    _as_cpu_numpy(self.last_field.electric_field)
                ),
                "polarization": (
                    _as_python(_as_cpu_numpy(self.last_field.polarization))
                    if self.config.want_pol(step)
                    else None
                ),
                "polarizability": (
                    _as_python(_as_cpu_numpy(self.last_field.polarizability))
                    if self.config.want_alpha(step)
                    else None
                ),
                "becs": (
                    _as_python(_as_cpu_numpy(self.last_field.becs))
                    if self.config.want_becs(step)
                    else None
                ),
            }
            try:
                with open(self.config.jsonl_path, "a", encoding="utf-8") as f:
                    f.write(json.dumps(rec) + "\n")
                    if self.config.jsonl_flush:
                        f.flush()
            except OSError:
                pass

    def _manage_profiling(self):
        if not self.config.debug_profile:
            return
        if self.step == self.config.profile_start_step:
            logging.info(f"Starting CUDA profiler at step {self.step}")
            torch.cuda.profiler.start()
        if self.step == self.config.profile_end_step:
            logging.info(f"Stopping CUDA profiler at step {self.step}")
            torch.cuda.profiler.stop()
            logging.info("Profiling complete. Exiting.")
            sys.exit()

    def compute_descriptors(self, data):
        pass

    def compute_gradients(self, data):
        pass
