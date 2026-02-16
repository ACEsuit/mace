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
        self.pad_num_atoms = self._get_env_int("MACE_MLIAP_PAD_NUM_ATOMS", 0)
        self.pad_num_pairs = self._get_env_int(
            "MACE_MLIAP_PAD_NUM_PAIRS",
            self._get_env_int("MACE_MLIAP_PAD_NUM_EDGES", 0),
        )

    @staticmethod
    def _get_env_bool(var_name: str, default: bool) -> bool:
        return os.environ.get(var_name, str(default)).lower() in (
            "true",
            "1",
            "t",
            "yes",
        )

    @staticmethod
    def _get_env_int(var_name: str, default: int) -> int:
        try:
            return int(os.environ.get(var_name, str(default)))
        except ValueError:
            logging.warning(
                "Invalid integer value for %s; using default %s.", var_name, default
            )
            return default


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


class _LammpsExchangeProxy:
    """Exchange wrapper that keeps padded fake atoms untouched."""

    def __init__(self, base, n_real: int, n_fake: int, n_ghost: int):
        self._base = base
        self._n_real = int(n_real)
        self._n_fake = int(n_fake)
        self._n_ghost = int(n_ghost)

    def _exchange(self, feats: torch.Tensor, out: torch.Tensor, vec_len: int, reverse: bool):
        if self._n_fake <= 0:
            if reverse:
                self._base.reverse_exchange(feats, out, vec_len)
            else:
                self._base.forward_exchange(feats, out, vec_len)
            return

        expected_with_fake = self._n_real + self._n_fake + self._n_ghost
        expected_base = self._n_real + self._n_ghost
        if feats.shape[0] != expected_with_fake:
            if reverse:
                self._base.reverse_exchange(feats, out, vec_len)
            else:
                self._base.forward_exchange(feats, out, vec_len)
            return

        base_input = torch.empty(
            (expected_base, vec_len), dtype=feats.dtype, device=feats.device
        )
        base_input[: self._n_real] = feats[: self._n_real]
        base_input[self._n_real :] = feats[self._n_real + self._n_fake :]

        base_output = torch.empty_like(base_input)
        if reverse:
            self._base.reverse_exchange(base_input, base_output, vec_len)
        else:
            self._base.forward_exchange(base_input, base_output, vec_len)

        out[: self._n_real] = base_output[: self._n_real]
        out[self._n_real : self._n_real + self._n_fake] = feats[
            self._n_real : self._n_real + self._n_fake
        ]
        out[self._n_real + self._n_fake :] = base_output[self._n_real :]

    def forward_exchange(self, feats: torch.Tensor, out: torch.Tensor, vec_len: int):
        self._exchange(feats, out, vec_len, reverse=False)

    def reverse_exchange(self, feats: torch.Tensor, out: torch.Tensor, vec_len: int):
        self._exchange(feats, out, vec_len, reverse=True)

    def __getattr__(self, item):
        return getattr(self._base, item)


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
        # Keep no-padding behavior identical while preserving static-shape tracing.
        if isinstance(data["lammps_class"], _LammpsExchangeProxy):
            data["head"] = self.head.repeat(2)
            data["total_charge"] = self.total_charge.repeat(2)
            data["total_spin"] = self.total_spin.repeat(2)
        else:
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
    """MACE integration for LAMMPS using the MLIAP interface."""

    def __init__(self, model, **kwargs):
        super().__init__()
        self.config = MACELammpsConfig()
        pad_num_atoms = int(kwargs.pop("pad_num_atoms", self.config.pad_num_atoms))
        pad_num_pairs = int(
            kwargs.pop(
                "pad_num_pairs",
                kwargs.pop("pad_num_edges", self.config.pad_num_pairs),
            )
        )
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
        self.pad_num_atoms = max(pad_num_atoms, 0)
        self.pad_num_pairs = max(pad_num_pairs, 0)
        self.fake_pair_distance = max(float(model.r_max) * 2.0, 1.0)
        self._warned_pair_padding_without_atoms = False

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
                batch = self._prepare_batch(data, natoms, nghosts, npairs, species)

            with timer("model_forward", enabled=self.config.debug_time):
                _, atom_energies, pair_forces = self.model(batch)

                if self.device.type != "cpu":
                    torch.cuda.synchronize()

            with timer("update_lammps", enabled=self.config.debug_time):
                self._update_lammps_data(
                    data, atom_energies, pair_forces, natoms, npairs
                )

    def _prepare_batch(self, data, natoms, nghosts, npairs, species):
        """Prepare the input batch for the MACE model."""
        # Strict no-padding parity path: return the same structure as the original
        # MLIAP implementation.
        if self.pad_num_atoms <= natoms and self.pad_num_pairs <= npairs:
            return {
                "vectors": torch.as_tensor(data.rij)
                .to(self.dtype)
                .to(self.device)
                .requires_grad_(True),
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

        target_atoms = max(natoms, self.pad_num_atoms)
        target_pairs = max(npairs, self.pad_num_pairs)
        pad_atoms = target_atoms - natoms
        pad_pairs = target_pairs - npairs

        if pad_pairs > 0 and pad_atoms <= 0:
            if not self._warned_pair_padding_without_atoms:
                logging.warning(
                    "Skipping MLIAP pair padding because no fake atoms are available. "
                    "Set pad_num_atoms above the runtime atom count to enable fixed-pair padding."
                )
                self._warned_pair_padding_without_atoms = True
            pad_pairs = 0
            target_pairs = npairs

        node_attrs = torch.nn.functional.one_hot(
            species[:natoms].to(self.device), num_classes=self.num_species
        ).to(self.dtype)
        if pad_atoms > 0:
            fake_node_attrs = torch.zeros(
                (pad_atoms, self.num_species), dtype=self.dtype, device=self.device
            )
            fake_node_attrs[:, 0] = 1.0
            node_attrs = torch.cat([node_attrs, fake_node_attrs], dim=0)

        vectors = torch.as_tensor(data.rij).to(self.dtype).to(self.device)
        if pad_pairs > 0:
            fake_vectors = torch.zeros((pad_pairs, 3), dtype=self.dtype, device=self.device)
            fake_vectors[:, 0] = self.fake_pair_distance
            vectors = torch.cat([vectors, fake_vectors], dim=0)

        real_edge_index = torch.stack(
            [
                torch.as_tensor(data.pair_j, dtype=torch.int64).to(self.device),
                torch.as_tensor(data.pair_i, dtype=torch.int64).to(self.device),
            ],
            dim=0,
        )
        if pad_pairs > 0:
            fake_edge_ids = torch.arange(pad_pairs, dtype=torch.int64, device=self.device)
            fake_senders = natoms + torch.remainder(fake_edge_ids, pad_atoms)
            fake_receivers = natoms + torch.remainder(fake_edge_ids + 1, pad_atoms)
            fake_edge_index = torch.stack([fake_senders, fake_receivers], dim=0)
            edge_index = torch.cat([real_edge_index, fake_edge_index], dim=1)
        else:
            edge_index = real_edge_index

        batch = torch.zeros(target_atoms, dtype=torch.int64, device=self.device)
        if pad_atoms > 0:
            batch[natoms:] = 1

        lammps_class = data
        if pad_atoms > 0:
            lammps_class = _LammpsExchangeProxy(
                data,
                n_real=natoms,
                n_fake=pad_atoms,
                n_ghost=nghosts,
            )

        return {
            "vectors": vectors.requires_grad_(True),
            "node_attrs": node_attrs,
            "edge_index": edge_index,
            "batch": batch,
            "lammps_class": lammps_class,
            "natoms": (target_atoms, nghosts),
        }

    def _update_lammps_data(self, data, atom_energies, pair_forces, natoms, npairs):
        """Update LAMMPS data structures with computed energies and forces."""
        pair_forces_real = pair_forces[:npairs].detach()
        if self.dtype == torch.float32:
            pair_forces_real = pair_forces_real.double()
        eatoms = torch.as_tensor(data.eatoms)
        atom_energies_real = atom_energies[:natoms].detach()
        eatoms.copy_(atom_energies_real)
        data.energy = atom_energies_real.sum().item()
        data.update_pair_forces_gpu(pair_forces_real)

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
