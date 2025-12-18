import logging
import os
import sys
import time
from contextlib import contextmanager
from typing import Dict, Tuple

import torch
from ase.data import chemical_symbols
from e3nn.util.jit import compile_mode

try:
    from lammps.mliap.mliap_unified_abc import MLIAPUnified
except ImportError:

    class MLIAPUnified:
        def __init__(self):
            pass


class MACELammpsConfig:
    """Configuration settings for MACE-LAMMPS integration."""

    def __init__(self):
        self.debug_time = self._get_env_bool("MACE_TIME", False)
        self.debug_profile = self._get_env_bool("MACE_PROFILE", False)
        self.profile_start_step = int(os.environ.get("MACE_PROFILE_START", "5"))
        self.profile_end_step = int(os.environ.get("MACE_PROFILE_END", "10"))
        self.allow_cpu = self._get_env_bool("MACE_ALLOW_CPU", False)
        self.force_cpu = self._get_env_bool("MACE_FORCE_CPU", False)

        # Constant fallback E-field (used if not pulling from LAMMPS variables)
        self.electric_field = self._get_env_vec3("MACE_EFIELD", (0.0, 0.0, 0.0))

        # Pull E-field from LAMMPS equal-style variables each step
        # Modes: "env" (default) or "lammps_var"
        self.efield_mode = os.environ.get("MACE_EFIELD_MODE", "env").strip().lower()

        # Names of equal-style variables (without v_) e.g. "Ex Ey Ez"
        self.efield_vars = self._get_env_list3("MACE_EFIELD_VARS", ("Ex", "Ey", "Ez"))

        # Optional debug logging
        self.efield_debug = self._get_env_bool("MACE_EFIELD_DEBUG", False)
        self.efield_debug_attrs = self._get_env_bool("MACE_EFIELD_DEBUG_ATTRS", False)

    @staticmethod
    def _get_env_bool(var_name: str, default: bool) -> bool:
        return os.environ.get(var_name, str(default)).lower() in (
            "true",
            "1",
            "t",
            "yes",
        )

    @staticmethod
    def _get_env_vec3(
        var_name: str, default: Tuple[float, float, float]
    ) -> Tuple[float, float, float]:
        raw = os.environ.get(var_name, None)
        if raw is None or str(raw).strip() == "":
            return default
        s = str(raw).strip().replace(",", " ")
        parts = [p for p in s.split() if p]
        if len(parts) != 3:
            raise ValueError(
                f"{var_name} must have 3 components, got {len(parts)} from: {raw!r}. "
                f"Use e.g. '{var_name}=0 0 0.3' or '{var_name}=0,0,0.3'."
            )
        try:
            return (float(parts[0]), float(parts[1]), float(parts[2]))
        except (TypeError, ValueError) as exc:
            raise ValueError(f"Failed to parse {var_name}={raw!r} as 3 floats") from exc

    @staticmethod
    def _get_env_list3(
        var_name: str, default: Tuple[str, str, str]
    ) -> Tuple[str, str, str]:
        raw = os.environ.get(var_name, None)
        if raw is None or str(raw).strip() == "":
            return default
        s = str(raw).strip().replace(",", " ")
        parts = [p for p in s.split() if p]
        if len(parts) != 3:
            raise ValueError(
                f"{var_name} must have 3 names, got {len(parts)} from: {raw!r}. "
                f"Use e.g. '{var_name}=Ex Ey Ez' or '{var_name}=Ex,Ey,Ez'."
            )
        return (parts[0], parts[1], parts[2])


@contextmanager
def timer(name: str, enabled: bool = True):
    """Context manager for timing code blocks."""
    if not enabled:
        yield
        return

    start = time.perf_counter()
    try:
        yield
    finally:
        elapsed = time.perf_counter() - start
        logging.info(f"Timer - {name}: {elapsed*1000:.3f} ms")


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
        """Compute energies and per-pair forces."""
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


class LAMMPS_MLIAP_MACE(MLIAPUnified):
    """MACE integration for LAMMPS using the MLIAP interface.

    Supports passing a time-dependent electric field defined in LAMMPS as equal-style variables
    Ex/Ey/Ez (or user-specified via MACE_EFIELD_VARS) by setting:

      MACE_EFIELD_MODE=lammps_var
      MACE_EFIELD_VARS="Ex Ey Ez"

    Otherwise falls back to constant MACE_EFIELD="0 0 0.3".
    """

    def __init__(self, model, **kwargs):
        super().__init__()
        self.config = MACELammpsConfig()
        self.model = MACEEdgeForcesWrapper(model, **kwargs)
        self.element_types = [chemical_symbols[s] for s in model.atomic_numbers]
        self.num_species = len(self.element_types)
        self.rcutfac = 0.5 * float(model.r_max)
        self.ndescriptors = 1
        self.nparams = 1
        self.dtype = model.r_max.dtype
        self.device = "cpu"
        self.initialized = False
        self.step = 0

        # E-field tensor; updated in-place each step (so batch always sees updated value)
        self.electric_field = torch.tensor(
            self.config.electric_field, dtype=self.dtype
        ).view(1, 3)

        # Best-effort detection of MACEField-like models
        self.is_macefield = bool(
            hasattr(model, "field_feats") or hasattr(model, "field_linear")
        )

        # Cached python handle to running LAMMPS instance (if obtainable)
        self._lmp_handle = None
        self._warned_no_lmp = False
        self._warned_not_macefield = False
        self._dumped_attrs = False

    def _initialize_device(self, data):
        using_kokkos = "kokkos" in data.__class__.__module__.lower()

        if using_kokkos and not self.config.force_cpu:
            device = torch.as_tensor(data.elems).device
            if device.type == "cpu" and not self.config.allow_cpu:
                raise ValueError(
                    "GPU requested but tensor is on CPU. Set MACE_ALLOW_CPU=true to allow CPU computation."
                )
        else:
            device = torch.device("cpu")

        self.device = device
        self.model = self.model.to(device)
        self.electric_field = self.electric_field.to(device=device, dtype=self.dtype)

        logging.info(f"MACE model initialized on device: {device}")
        self.initialized = True

    def _get_lammps_handle(self, data):
        """Best-effort: get a python 'lammps' handle to the *running* LAMMPS instance."""
        if self._lmp_handle is not None:
            return self._lmp_handle

        if self.config.efield_debug_attrs and not self._dumped_attrs:
            try:
                logging.info(f"[MACE_EFIELD_DEBUG_ATTRS] data class: {data.__class__}")
                logging.info(
                    f"[MACE_EFIELD_DEBUG_ATTRS] data module: {data.__class__.__module__}"
                )
                logging.info(
                    f"[MACE_EFIELD_DEBUG_ATTRS] dir(data) sample: {sorted(dir(data))[:200]}"
                )
            except (AttributeError, TypeError, ValueError, RuntimeError):
                # purely best-effort debug output
                pass
            self._dumped_attrs = True

        # 1) If data (or self) already has an object exposing extract_variable, use it directly
        for obj in (data, self):
            for name in ("lmp", "_lmp", "lammps", "_lammps"):
                try:
                    h = getattr(obj, name)
                except AttributeError:
                    h = None
                if h is not None and hasattr(h, "extract_variable"):
                    self._lmp_handle = h
                    return h

            if hasattr(obj, "extract_variable"):
                self._lmp_handle = obj
                return obj

        # 2) Try to wrap an existing pointer/capsule if present
        ptr_candidates = []
        for obj in (data, self):
            for name in (
                "lmpptr",
                "_lmpptr",
                "lmp_ptr",
                "_lmp_ptr",
                "ptr",
                "_ptr",
                "lammps_ptr",
                "_lammps_ptr",
            ):
                try:
                    c = getattr(obj, name)
                except AttributeError:
                    c = None
                if c is not None:
                    ptr_candidates.append(c)

        # 3) Also check __main__ for embedded-python globals (sometimes exposed)
        try:
            import __main__  # type: ignore

            for name in ("lmp", "lmpptr", "lmp_ptr"):
                c = getattr(__main__, name, None)
                if c is not None:
                    if hasattr(c, "extract_variable"):
                        self._lmp_handle = c
                        return c
                    ptr_candidates.append(c)
        except (ImportError, AttributeError, RuntimeError, TypeError):
            pass

        # Try constructing a lammps python object from ptr
        try:
            from lammps import lammps as lammps_py  # type: ignore
        except (ImportError, ModuleNotFoundError):
            lammps_py = None

        if lammps_py is None:
            return None

        for c in ptr_candidates:
            try:
                if hasattr(c, "extract_variable"):
                    self._lmp_handle = c
                    return c

                if isinstance(c, int):
                    h = lammps_py(ptr=c)
                    if hasattr(h, "extract_variable"):
                        self._lmp_handle = h
                        return h

                if hasattr(c, "value") and isinstance(getattr(c, "value"), int):
                    h = lammps_py(ptr=int(c.value))
                    if hasattr(h, "extract_variable"):
                        self._lmp_handle = h
                        return h
            except (TypeError, ValueError, RuntimeError, AttributeError):
                continue

        return None

    def _update_electric_field_from_lammps(self, data):
        """If enabled, pull Ex/Ey/Ez from LAMMPS equal-style variables each step."""
        if self.config.efield_mode != "lammps_var":
            return

        if not self.is_macefield and not self._warned_not_macefield:
            logging.warning(
                "MACE_EFIELD_MODE=lammps_var set but underlying model does not look like MACEField; "
                "will still attach electric_field to batch, but model may ignore it."
            )
            self._warned_not_macefield = True

        lmp = self._get_lammps_handle(data)
        if lmp is None:
            if not self._warned_no_lmp:
                logging.warning(
                    "MACE_EFIELD_MODE=lammps_var but could not access running LAMMPS handle; "
                    "falling back to static MACE_EFIELD."
                )
                self._warned_no_lmp = True
            return

        ex_name, ey_name, ez_name = self.config.efield_vars
        try:
            ex = lmp.extract_variable(ex_name, None, 0)  # 0 = equal-style
            ey = lmp.extract_variable(ey_name, None, 0)
            ez = lmp.extract_variable(ez_name, None, 0)
        except (AttributeError, TypeError, RuntimeError) as exc:
            raise RuntimeError(
                f"Failed to extract LAMMPS variables {self.config.efield_vars} via "
                "extract_variable(..., type=0)."
            ) from exc

        if ex is None or ey is None or ez is None:
            raise ValueError(
                f"One or more LAMMPS variables not found: {self.config.efield_vars}. "
                "Define them as equal-style variables in your input, e.g. 'variable Ez equal ...'."
            )

        exf, eyf, ezf = float(ex), float(ey), float(ez)

        # In-place update so the batch always sees the updated value (same tensor object)
        self.electric_field[0, 0] = exf
        self.electric_field[0, 1] = eyf
        self.electric_field[0, 2] = ezf

        if self.config.efield_debug and (self.step < 5 or (self.step % 1000 == 0)):
            logging.info(
                f"[MACE_EFIELD] step={self.step} "
                f"E=({exf:.6f},{eyf:.6f},{ezf:.6f}) vars={self.config.efield_vars}"
            )

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

        # Update E-field from LAMMPS variables each step (if enabled)
        self._update_electric_field_from_lammps(data)

        if natoms == 0 or npairs <= 1:
            return

        with timer("total_step", enabled=self.config.debug_time):
            with timer("prepare_batch", enabled=self.config.debug_time):
                batch = self._prepare_batch(data, natoms, nghosts, species)

            with timer("model_forward", enabled=self.config.debug_time):
                _, atom_energies, pair_forces = self.model(batch)

                if self.device.type != "cpu":
                    torch.cuda.synchronize()

            with timer("update_lammps", enabled=self.config.debug_time):
                self._update_lammps_data(data, atom_energies, pair_forces, natoms)

    def _prepare_batch(self, data, natoms, nghosts, species):
        """Prepare the input batch for the MACE model."""
        batch = {
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

        # Always attach electric_field (MACEField will use it; plain MACE should ignore unknown key)
        batch["electric_field"] = self.electric_field
        return batch

    def _update_lammps_data(self, data, atom_energies, pair_forces, natoms):
        """Update LAMMPS data structures with computed energies and forces."""
        if self.dtype == torch.float32:
            pair_forces = pair_forces.double()
        eatoms = torch.as_tensor(data.eatoms)
        eatoms.copy_(atom_energies[:natoms])
        data.energy = torch.sum(atom_energies[:natoms])
        data.update_pair_forces_gpu(pair_forces)

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
