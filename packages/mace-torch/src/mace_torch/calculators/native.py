"""ASE dynamics using the complete native periodic CUDA evaluation graph."""

from __future__ import annotations

import importlib
from pathlib import Path

import numpy as np
import torch
from ase import Atoms
from ase.calculators.calculator import Calculator, all_changes
from ase.stress import full_3x3_to_voigt_6_stress

from mace_torch.neighbors.native_cuda import _load


class NativeMACECalculator(Calculator):
    """Execute a periodic CUDA ``native.pt2`` with GPU neighbours.

    Energy, forces and virial are all AOT compiled. ASE owns the host positions
    and integrator. Units are eV, Angstrom and eV/Angstrom; stress is in eV/Å³.
    The package fixes atom count, cell, charge, spin and external field. Site
    energies are the local readout and exclude the electrostatic contribution.
    """

    implemented_properties = ("energy", "free_energy", "energies", "forces", "stress")

    def __init__(self, package: str | Path, *, neighbor_library: str | Path):
        super().__init__()
        path = str(Path(package).resolve(strict=True))
        # The PyBind method is present in torch but missing from its type stub.
        metadata = torch._C._aoti.AOTIModelPackageLoader.load_metadata_from_package(  # ty: ignore[unresolved-attribute]
            path, "model"
        )
        if (
            metadata["mace_native_version"] != "1.0"
            or metadata["mace_evaluation"] != "periodic_system"
            or metadata["mace_device"] != "cuda"
        ):
            raise ValueError(
                "NativeMACECalculator requires a periodic CUDA 1.0 package"
            )
        if metadata["mace_backend"] == "cueq":
            importlib.import_module("cuequivariance_ops_torch")
        self.device = torch.device("cuda", 0)
        capability = "".join(map(str, torch.cuda.get_device_capability(self.device)))
        if metadata["AOTI_COMPUTE_CAPABILITY"] != capability:
            raise ValueError(
                "native package targets another GPU; re-export on this device"
            )
        self.dtype = {"float32": torch.float32, "float64": torch.float64}[
            metadata["mace_dtype"]
        ]
        self._input_dtype = np.dtype(metadata["mace_dtype"])
        self._count = int(metadata["mace_nodes"])
        self._capacity = int(metadata["mace_edges"])
        self._cutoff = float(metadata["mace_cutoff"])
        self._cell = np.fromstring(metadata["mace_cell"], sep=" ").reshape(3, 3)
        self._device_cell = torch.tensor(
            self._cell, dtype=self.dtype, device=self.device
        )
        self._padding_shift = (
            int(np.ceil(2 * self._cutoff / np.linalg.norm(self._cell[0]))) + 1
        )
        self._species = {
            int(z): i for i, z in enumerate(metadata["mace_atomic_numbers"].split())
        }
        self._fixed_inputs = {
            "charge": np.asarray(
                float(metadata["mace_total_charge"]), dtype=self._input_dtype
            ),
            "spin": np.asarray(
                float(metadata["mace_total_spin"]), dtype=self._input_dtype
            ),
            "external_field": np.fromstring(
                metadata["mace_external_field"], sep=" "
            ).astype(self._input_dtype),
        }
        _load(Path(neighbor_library))
        self._model = torch._inductor.aoti_load_package(path, device_index=0)

    def _validate(self, atoms: Atoms) -> None:
        if len(atoms) != self._count or not atoms.pbc.all():
            raise ValueError(
                "native package requires its exported atom count and full periodicity"
            )
        if not np.allclose(atoms.cell, self._cell, atol=1e-8, rtol=0):
            raise ValueError("native package requires its exported fixed cell")
        if not np.isfinite(atoms.positions).all():
            raise ValueError("native positions must be finite")
        for name, expected in self._fixed_inputs.items():
            keys = (name, f"total_{name}") if name != "external_field" else (name,)
            for key in keys:
                if key in atoms.info and not np.array_equal(
                    np.asarray(atoms.info[key], dtype=self._input_dtype), expected
                ):
                    raise ValueError(
                        f"native package fixes {key}; re-export to change it"
                    )

    def check_state(self, atoms, tol=1e-15):
        # ASE does not normally track info changes, including charge and field.
        self._validate(atoms)
        return super().check_state(atoms, tol)

    def calculate(self, atoms=None, properties=("energy",), system_changes=all_changes):
        super().calculate(atoms, properties, system_changes)
        assert self.atoms is not None
        self._validate(self.atoms)
        try:
            elements = [self._species[int(z)] for z in self.atoms.numbers]
        except KeyError as error:
            raise ValueError(f"unknown native atomic number {error.args[0]}") from error
        positions = torch.tensor(
            self.atoms.positions, dtype=torch.float64, device=self.device
        )
        index, shifts, _ = torch.ops.mace_neighbors.periodic(
            positions,
            self._cell.ravel().tolist(),
            self._cutoff,
            self._capacity,
            self._padding_shift,
        )
        with torch.no_grad():
            energy, sites, forces, virial = self._model(
                shifts.to(self.dtype),
                index,
                torch.tensor(elements, dtype=torch.int64, device=self.device),
                positions.to(self.dtype),
                self._device_cell,
            )
        total = float(energy.item())
        self.results = {
            "energy": total,
            "free_energy": total,
            "energies": sites.cpu().numpy(),
            "forces": forces.cpu().numpy(),
            "stress": full_3x3_to_voigt_6_stress(
                -virial.cpu().numpy() / self.atoms.get_volume()
            ),
        }
