###########################################################################################
# The ASE Calculator for MACE
# Authors: Ilyes Batatia, David Kovacs
# This program is distributed under the MIT License (see MIT.md)
###########################################################################################

import logging
import time

# pylint: disable=wrong-import-position
import os
from glob import glob
from pathlib import Path
from typing import Dict, List, Optional, Union

os.environ["TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD"] = "1"

import numpy as np
import torch
from ase.calculators.calculator import Calculator, all_changes
from ase.stress import full_3x3_to_voigt_6_stress
from e3nn import o3

from mace import data as mace_data
from mace.modules.utils import extract_invariant
from mace.tools import torch_geometric, torch_tools, utils
from mace.tools.compile import prepare, disable_e3nn_codegen, simplify
from mace.tools.scripts_utils import extract_model

try:
    import torch._dynamo as dynamo
except ImportError:
    dynamo = None

try:
    from mace.cli.convert_e3nn_cueq import run as run_e3nn_to_cueq

    CUEQQ_AVAILABLE = True
except (ImportError, ModuleNotFoundError):
    CUEQQ_AVAILABLE = False
    run_e3nn_to_cueq = None

try:
    from mace.cli.convert_e3nn_oeq import run as run_e3nn_to_oeq

    OEQ_AVAILABLE = True
except (ImportError, ModuleNotFoundError):
    OEQ_AVAILABLE = False
    run_e3nn_to_oeq = None

try:
    from mace.cli.convert_e3nn_hybrid import run as run_e3nn_to_hybrid

    HYBRID_AVAILABLE = True
except (ImportError, ModuleNotFoundError):
    HYBRID_AVAILABLE = False
    run_e3nn_to_hybrid = None

try:
    import intel_extension_for_pytorch as ipex

    has_ipex = True
except ImportError:
    has_ipex = False

_EDGE_PAD_MULTIPLE = 64
_EDGE_PAD_HEADROOM = 1.25


def _round_up(value: int, multiple: int) -> int:
    return ((value + multiple - 1) // multiple) * multiple


def get_model_dtype(model: torch.nn.Module) -> torch.dtype:
    """Get the dtype of the model"""
    mode_dtype = next(model.parameters()).dtype
    if mode_dtype == torch.float64:
        return "float64"
    if mode_dtype == torch.float32:
        return "float32"
    raise ValueError(f"Unknown dtype {mode_dtype}")


class MACECalculator(Calculator):
    """MACE ASE Calculator

    Supports accelerated backends (cueq, openequivariance, hybrid) and
    torch.compile with automatic graph padding to avoid recompilation
    during geometry relaxation / MD.

    Args:
        model_paths: path(s) to model file(s); supports wildcards for committees
        models: pre-loaded model(s) as an alternative to model_paths
        device: 'cuda', 'cpu', or 'xpu'
        compile_mode: torch.compile mode ('default', 'reduce-overhead', etc.)
                      or None to disable compilation
        enable_cueq: use cuequivariance for symmetric contractions / linear
        enable_oeq: use openequivariance for channelwise tensor product
        pad_num_atoms: fixed atom count for padding (0 = auto-estimate on first call)
        pad_num_edges: fixed edge count for padding (0 = auto-estimate on first call)
        warmup: if True, run one dummy forward pass after init to trigger compilation
    """

    def __init__(
        self,
        model_paths: Union[list, str, None] = None,
        models: Union[List[torch.nn.Module], torch.nn.Module, None] = None,
        device: str = "cpu",
        energy_units_to_eV: float = 1.0,
        length_units_to_A: float = 1.0,
        default_dtype="",
        charges_key="Qs",
        info_keys=None,
        arrays_keys=None,
        model_type="MACE",
        compile_mode=None,
        fullgraph=True,
        enable_cueq=False,
        enable_oeq=False,
        pad_num_atoms: int = 0,
        pad_num_edges: int = 0,
        warmup: bool = False,
        **kwargs,
    ):
        Calculator.__init__(self, **kwargs)

        self._enable_cueq = enable_cueq
        self._enable_oeq = enable_oeq
        self._uses_accelerated_backend = enable_cueq or enable_oeq

        if self._uses_accelerated_backend:
            assert model_type in [
                "MACE",
                "PolarMACE",
            ], "CuEq/OEq only supports MACE and PolarMACE models"
        if enable_cueq and enable_oeq:
            if not HYBRID_AVAILABLE:
                raise ImportError(
                    "Hybrid cueq+oeq mode requires both cuequivariance and "
                    "openequivariance to be installed"
                )
        elif enable_cueq and not CUEQQ_AVAILABLE:
            raise ImportError(
                "cuequivariance is not installed so CuEq acceleration cannot be used"
            )
        elif enable_oeq and not OEQ_AVAILABLE:
            raise ImportError(
                "openequivariance is not installed so OEq acceleration cannot be used"
            )
        if "model_path" in kwargs:
            deprecation_message = (
                "'model_path' argument is deprecated, please use 'model_paths'"
            )
            if model_paths is None:
                logging.warning(f"{deprecation_message} in the future.")
                model_paths = kwargs["model_path"]
            else:
                raise ValueError(
                    f"both 'model_path' and 'model_paths' given, {deprecation_message} only."
                )

        if (model_paths is None) == (models is None):
            raise ValueError(
                "Exactly one of 'model_paths' or 'models' must be provided"
            )

        self.results = {}
        if info_keys is None:
            info_keys = {
                "total_spin": "spin",
                "total_charge": "charge",
                "external_field": "external_field",
            }
        if arrays_keys is None:
            arrays_keys = {}
        self.info_keys = info_keys
        self.arrays_keys = arrays_keys

        self.model_type = model_type
        self.compute_atomic_stresses = False

        if model_type not in [
            "MACE",
            "DipoleMACE",
            "EnergyDipoleMACE",
            "DipolePolarizabilityMACE",
            "PolarMACE",
        ]:
            raise ValueError(
                f"Give a valid model_type: [MACE, PolarMACE, DipoleMACE, DipolePolarizabilityMACE, EnergyDipoleMACE], {model_type} not supported"
            )

        if model_type in ["MACE", "EnergyDipoleMACE", "PolarMACE"]:
            self.implemented_properties.extend(
                [
                    "energy",
                    "energies",
                    "free_energy",
                    "node_energy",
                    "forces",
                    "stress",
                ]
            )
            if kwargs.get("compute_atomic_stresses", False):
                self.implemented_properties.extend(["stresses", "virials"])
                self.compute_atomic_stresses = True
        if model_type in ["EnergyDipoleMACE", "DipoleMACE", "DipolePolarizabilityMACE"]:
            self.implemented_properties.extend(["dipole"])
        if model_type == "DipolePolarizabilityMACE":
            self.implemented_properties.extend(
                [
                    "charges",
                    "polarizability",
                    "polarizability_sh",
                ]
            )

        # ── Load models ──────────────────────────────────────────────────
        if model_paths is not None:
            if isinstance(model_paths, str):
                model_paths_glob = glob(model_paths)
                if len(model_paths_glob) == 0:
                    raise ValueError(f"Couldn't find MACE model files: {model_paths}")
                model_paths = model_paths_glob
            elif isinstance(model_paths, Path):
                model_paths = [model_paths]

            if len(model_paths) == 0:
                raise ValueError("No mace file names supplied")
            self.num_models = len(model_paths)
            self.models = [
                torch.load(f=model_path, map_location=device)
                for model_path in model_paths
            ]
        elif models is not None:
            if not isinstance(models, list):
                models = [models]
            if len(models) == 0:
                raise ValueError("No models supplied")
            self.models = models
            self.num_models = len(models)

        if self.num_models > 1:
            logging.info(f"Running committee mace with {self.num_models} models")

            if model_type in ["MACE", "EnergyDipoleMACE", "PolarMACE"]:
                self.implemented_properties.extend(
                    ["energy_comm", "energy_var", "forces_comm", "stress_var"]
                )
            if model_type in [
                "DipoleMACE",
                "EnergyDipoleMACE",
                "DipolePolarizabilityMACE",
            ]:
                self.implemented_properties.extend(["dipole_var"])

        # ── Ensure dtype consistency ─────────────────────────────────────
        for model in self.models:
            model.to(device)

        if has_ipex and device == "xpu":
            for model in self.models:
                model = ipex.optimize(model)

        r_maxs = [model.r_max.cpu() for model in self.models]
        r_maxs = np.array(r_maxs)
        if not np.all(r_maxs == r_maxs[0]):
            raise ValueError(f"committee r_max are not all the same {' '.join(r_maxs)}")
        self.r_max = float(r_maxs[0])

        self.device = torch_tools.init_device(device)
        self.energy_units_to_eV = energy_units_to_eV
        self.length_units_to_A = length_units_to_A
        self.z_table = utils.AtomicNumberTable(
            [int(z) for z in self.models[0].atomic_numbers]
        )
        self.charges_key = charges_key
        if self.model_type == "PolarMACE":
            self.density_dim = (
                getattr(self.models[0], "atomic_multipoles_max_l", 0) + 1
            ) ** 2

        try:
            self.available_heads: List[str] = self.models[0].heads  # type: ignore
        except AttributeError:
            self.available_heads = ["Default"]
        kwarg_head = kwargs.get("head", None)
        if kwarg_head is not None:
            self.head = kwarg_head
            if isinstance(self.head, str):
                if self.head not in self.available_heads:
                    last_head = self.available_heads[-1]
                    logging.warning(
                        f"Head {self.head} not found in available heads {self.available_heads}, defaulting to the last head: {last_head}"
                    )
                    self.head = last_head
        elif len(self.available_heads) == 1:
            self.head = self.available_heads[0]
        else:
            self.head = [
                head for head in self.available_heads if head.lower() == "default"
            ]
            if len(self.head) == 0:
                raise ValueError(
                    "Head keyword was not provided, and no head in the model is 'default'. "
                    "Please provide a head keyword to specify the head you want to use. "
                    f"Available heads are: {self.available_heads}"
                )
            self.head = self.head[0]

        logging.info(f"Using head {self.head} out of  {self.available_heads}")

        model_dtype = get_model_dtype(self.models[0])
        if default_dtype == "":
            logging.warning(
                f"No dtype selected, switching to {model_dtype} to match model dtype."
            )
            default_dtype = model_dtype
        if model_dtype != default_dtype:
            logging.warning(
                f"Default dtype {default_dtype} does not match model dtype {model_dtype}, converting models to {default_dtype}."
            )
            if default_dtype == "float64":
                self.models = [model.double() for model in self.models]
            elif default_dtype == "float32":
                self.models = [model.float() for model in self.models]
        torch_tools.set_default_dtype(default_dtype)

        # ── Backend conversion (cueq / oeq / hybrid) ────────────────────
        if enable_cueq and enable_oeq:
            logging.info(
                "Converting models to hybrid cueq+oeq: "
                "cueq for symmetric contractions/linear, oeq for conv TP"
            )
            self.models = [
                run_e3nn_to_hybrid(model, device=device).to(device)
                for model in self.models
            ]
        elif enable_cueq:
            logging.info("Converting models to CuEq for acceleration")
            self.models = [
                run_e3nn_to_cueq(model, device=device).to(device)
                for model in self.models
            ]
        elif enable_oeq:
            logging.info("Converting models to OEq for acceleration")
            self.models = [
                run_e3nn_to_oeq(model, device=device).to(device)
                for model in self.models
            ]

        # ── torch.compile ────────────────────────────────────────────────
        self.use_compile = False
        if compile_mode is not None:
            logging.info(f"Torch compile is enabled with mode: {compile_mode}")
            if self._enable_oeq:
                # oeq custom ops break when autograd.grad is inlined in the
                # compiled graph.  Ensure it is a graph break instead.
                if dynamo is not None:
                    try:
                        dynamo.disallow_in_graph(torch.autograd.grad)
                    except Exception:
                        pass
            else:
                if dynamo is not None:
                    dynamo.allow_in_graph(torch.autograd.grad)
            if self._uses_accelerated_backend:
                with disable_e3nn_codegen():
                    self.models = [simplify(m) for m in self.models]
                self.models = [
                    torch.compile(m, mode=compile_mode, fullgraph=False)
                    for m in self.models
                ]
            else:
                self.models = [
                    torch.compile(
                        prepare(extract_model)(model=model, map_location=device),
                        mode=compile_mode,
                        fullgraph=fullgraph,
                    )
                    for model in self.models
                ]
            self.use_compile = True

        for model in self.models:
            for param in model.parameters():
                param.requires_grad = False

        # ── Padding ──────────────────────────────────────────────────────
        if pad_num_atoms <= 0:
            pad_num_atoms = int(os.environ.get("MACE_ASE_PAD_NUM_ATOMS", "0"))
        if pad_num_edges <= 0:
            pad_num_edges = int(os.environ.get("MACE_ASE_PAD_NUM_EDGES", "0"))
        self.pad_num_atoms = max(int(pad_num_atoms), 0)
        self.pad_num_edges = max(int(pad_num_edges), 0)
        self._padding_initialized = self.pad_num_atoms > 0 and self.pad_num_edges > 0

        # ── Warmup ───────────────────────────────────────────────────────
        if warmup and self.use_compile:
            logging.info(
                "Warmup requested -- will trigger on first calculate() call"
            )

    def check_state(self, atoms, tol: float = 1e-15) -> list:
        """
        Check for any system changes since the last calculation.

        Args:
            atoms (ase.Atoms): The atomic structure to check.
            tol (float): Tolerance for detecting changes.

        Returns:
            list: A list of changes detected in the system.
        """

        def _infos_equal(a: dict, b: dict) -> bool:
            if a.keys() != b.keys():
                return False
            for k in a:
                va, vb = a[k], b[k]
                if isinstance(va, np.ndarray) or isinstance(vb, np.ndarray):
                    continue
                if va != vb:
                    return False
            return True

        state = super().check_state(atoms, tol=tol)
        if (not state) and (not _infos_equal(self.atoms.info, atoms.info)):
            state.append("info")
        return state

    @staticmethod
    def _slice_real_outputs(
        out: Dict[str, Union[torch.Tensor, None]], num_real_atoms: int
    ) -> Dict[str, Union[torch.Tensor, None]]:
        """Strip padding from model outputs, keeping only real-atom results."""
        graph_level_keys = {
            "energy", "stress", "virials", "dipole",
            "polarizability", "polarizability_sh",
            "displacement", "contributions",
        }
        atom_level_keys = {
            "node_energy", "forces", "charges",
            "atomic_stresses", "atomic_virials",
            "atomic_dipoles", "node_feats",
        }
        sliced: Dict[str, Union[torch.Tensor, None]] = {}
        for key, value in out.items():
            if value is None or not torch.is_tensor(value):
                sliced[key] = value
            elif key in graph_level_keys and value.ndim > 0:
                sliced[key] = value[0]
            elif key in atom_level_keys:
                sliced[key] = value[:num_real_atoms]
            else:
                sliced[key] = value
        return sliced

    def _create_result_tensors(
        self, num_models: int, num_atoms: int, batch, out: dict
    ) -> dict:
        tensor_shapes = {
            "energy": [],
            "node_energy": [num_atoms],
            "forces": [num_atoms, 3],
            "stress": [3, 3],
            "atomic_stresses": [num_atoms, 3, 3],
            "atomic_virials": [num_atoms, 3, 3],
            "dipole": [3],
            "charges": [num_atoms],
            "polarizability": [3, 3],
            "polarizability_sh": [6],
        }
        if self.model_type == "PolarMACE":
            tensor_shapes.update(
                {
                    "interaction_energy": [],
                    "electrostatic_energy": [],
                    "electron_energy": [],
                    "spins": [num_atoms],
                    "density_coefficients": [num_atoms, self.density_dim],
                    "spin_charge_density": [num_atoms, 2, self.density_dim],
                }
            )
        dict_of_tensors = {}
        for key in out:
            if key not in tensor_shapes or out.get(key) is None:
                continue
            shape = [num_models] + tensor_shapes[key]
            dict_of_tensors[key] = torch.zeros(*shape, device=self.device)

        node_e0 = None
        if "node_energy" in out:
            node_heads = batch["head"][batch["batch"]][:num_atoms]
            num_atoms_arange = torch.arange(num_atoms)
            node_e0 = (
                self.models[0]
                .atomic_energies_fn(batch["node_attrs"][:num_atoms])[
                    num_atoms_arange, node_heads
                ]
                .detach()
                .cpu()
                .numpy()
            )

        return dict_of_tensors, node_e0

    def _auto_estimate_padding(self, real_num_atoms: int, real_num_edges: int):
        """Set padding targets on first call based on actual graph size."""
        if self._padding_initialized:
            return
        self.pad_num_atoms = real_num_atoms
        self.pad_num_edges = _round_up(
            int(real_num_edges * _EDGE_PAD_HEADROOM), _EDGE_PAD_MULTIPLE
        )
        self._padding_initialized = True
        logging.info(
            "Auto-estimated padding: %d atoms, %d edges (real: %d atoms, %d edges)",
            self.pad_num_atoms, self.pad_num_edges, real_num_atoms, real_num_edges,
        )

    def _atoms_to_batch(self, atoms):
        self.arrays_keys.update({self.charges_key: "charges"})
        keyspec = mace_data.KeySpecification(
            info_keys=self.info_keys, arrays_keys=self.arrays_keys
        )
        config = mace_data.config_from_atoms(
            atoms, key_specification=keyspec, head_name=self.head
        )
        real_graph = mace_data.AtomicData.from_config(
            config,
            z_table=self.z_table,
            cutoff=self.r_max,
            heads=self.available_heads,
        )

        real_num_atoms = int(real_graph["node_attrs"].shape[0])
        real_num_edges = int(real_graph["edge_index"].shape[1])

        if self.use_compile and not self._padding_initialized:
            self._auto_estimate_padding(real_num_atoms, real_num_edges)

        if real_num_edges > self.pad_num_edges and self._padding_initialized:
            old = self.pad_num_edges
            self.pad_num_edges = _round_up(
                int(real_num_edges * _EDGE_PAD_HEADROOM), _EDGE_PAD_MULTIPLE
            )
            logging.warning(
                "Edge count %d exceeded pad budget %d -- bumping to %d "
                "(will trigger one recompile)",
                real_num_edges, old, self.pad_num_edges,
            )

        target_num_atoms = max(real_num_atoms, self.pad_num_atoms)
        target_num_edges = max(real_num_edges, self.pad_num_edges)
        pad_atoms = target_num_atoms - real_num_atoms
        pad_edges = target_num_edges - real_num_edges

        data_list = [real_graph]
        if pad_atoms > 0 or pad_edges > 0:
            from mace.data.padding_tools import build_fake_padding_graph

            if pad_edges > 0 and pad_atoms <= 0:
                pad_atoms = 1
            data_list.append(
                build_fake_padding_graph(
                    real_graph,
                    num_atoms=pad_atoms,
                    num_edges=pad_edges,
                    r_max=self.r_max,
                )
            )

        batch = torch_geometric.Batch.from_data_list(data_list).to(self.device)
        return batch

    def _clone_batch(self, batch):
        batch_clone = batch.clone()
        if self.use_compile:
            batch_clone["node_attrs"].requires_grad_(True)
            batch_clone["positions"].requires_grad_(True)
        return batch_clone

    # pylint: disable=dangerous-default-value
    def calculate(self, atoms=None, properties=None, system_changes=all_changes):
        """
        Calculate properties.
        :param atoms: ase.Atoms object
        :param properties: [str], properties to be computed, used by ASE internally
        :param system_changes: [str], system changes since last calculation, used by ASE internally
        :return:
        """
        Calculator.calculate(self, atoms)

        batch_base = self._atoms_to_batch(atoms)
        num_real_atoms = len(atoms)
        is_padded = self.pad_num_atoms > 0 or self.pad_num_edges > 0

        compute_stress = self.model_type in ["MACE", "EnergyDipoleMACE", "PolarMACE"]
        # For oeq/hybrid + compile: create displacement outside the compiled
        # graph so autograd.grad (which runs as a graph break) can
        # differentiate energy w.r.t. displacement for stress.
        oeq_compile = self.use_compile and self._enable_oeq

        ret_tensors = None
        node_e0 = None
        for i, model in enumerate(self.models):
            batch = self._clone_batch(batch_base)
            model_dtype = next(model.parameters()).dtype
            for key in batch.keys:
                value = batch[key]
                if torch.is_tensor(value) and torch.is_floating_point(value):
                    batch[key] = value.to(dtype=model_dtype)
            batch_dict = batch.to_dict()

            if oeq_compile and compute_stress:
                positions = batch_dict["positions"]
                num_graphs = int(batch_dict["ptr"].numel() - 1)
                displacement = torch.zeros(
                    (num_graphs, 3, 3),
                    dtype=positions.dtype,
                    device=positions.device,
                )
                displacement = displacement + positions.sum() * 0.0
                batch_dict["displacement"] = displacement

            out = model(
                batch_dict,
                compute_stress=compute_stress,
                training=self.use_compile and not oeq_compile,
                compute_edge_forces=self.compute_atomic_stresses,
                compute_atomic_stresses=self.compute_atomic_stresses,
            )
            if is_padded:
                out = self._slice_real_outputs(out, num_real_atoms)
            if i == 0:
                ret_tensors, node_e0 = self._create_result_tensors(
                    self.num_models, num_real_atoms, batch, out
                )
            for key, val in ret_tensors.items():
                if out.get(key) is not None:
                    val[i] = out[key].detach()

        # covert from ret_tensors to calculator results dict
        self.results = {}
        scalar_tensors = set(["energy"])
        results_store_ensemble = set(["energy", "forces", "stress", "dipole"])
        results_map = [
            ("energy", "energy", self.energy_units_to_eV),
            ("node_energy", "node_energy", self.energy_units_to_eV),
            ("forces", "forces", self.energy_units_to_eV / self.length_units_to_A),
            ("stress", "stress", self.energy_units_to_eV / self.length_units_to_A**3),
            (
                "stresses",
                "atomic_stresses",
                self.energy_units_to_eV / self.length_units_to_A**3,
            ),
            (
                "virials",
                "atomic_virials",
                self.energy_units_to_eV / self.length_units_to_A**3,
            ),
            ("dipole", "dipole", 1.0),
            ("charges", "charges", 1.0),
            ("polarizability", "polarizability", 1.0),
            ("polarizability_sh", "polarizability_sh", 1.0),
        ]
        if self.model_type == "PolarMACE":
            results_map.extend(
                [
                    (
                        "interaction_energy",
                        "interaction_energy",
                        self.energy_units_to_eV,
                    ),
                    (
                        "electrostatic_energy",
                        "electrostatic_energy",
                        self.energy_units_to_eV,
                    ),
                    ("electron_energy", "electron_energy", self.energy_units_to_eV),
                    ("spins", "spins", 1.0),
                    ("density_coefficients", "density_coefficients", 1.0),
                    ("spin_charge_density", "spin_charge_density", 1.0),
                ]
            )
        for results_key, ret_key, unit_conv in results_map:
            if ret_tensors.get(ret_key) is not None:
                data = torch.mean(ret_tensors[ret_key], dim=0).cpu()
                if ret_key in scalar_tensors:
                    data = data.item()
                else:
                    data = data.numpy()
                self.results[results_key] = data * unit_conv

                if self.num_models > 1 and results_key in results_store_ensemble:
                    data = ret_tensors[results_key].cpu().numpy()
                    data *= unit_conv
                    self.results[results_key + "_comm"] = data

                    data = torch.var(
                        ret_tensors[results_key], dim=0, unbiased=False
                    ).cpu()
                    if ret_key in scalar_tensors:
                        data = data.item()
                    else:
                        data = data.numpy()
                    data *= unit_conv
                    self.results[results_key + "_var"] = data

        # special cases
        if self.results.get("energy") is not None:
            self.results["free_energy"] = self.results["energy"]
        if self.results.get("node_energy") is not None:
            self.results["energies"] = self.results["node_energy"].copy()
            self.results["node_energy"] -= node_e0
        if self.results.get("stress") is not None:
            self.results["stress"] = full_3x3_to_voigt_6_stress(self.results["stress"])
        if self.results.get("stresses") is not None:
            self.results["stresses"] = np.asarray(
                [
                    full_3x3_to_voigt_6_stress(stress)
                    for stress in self.results["stresses"]
                ]
            )

    def get_dielectric_derivatives(self, atoms=None):
        if atoms is None and self.atoms is None:
            raise ValueError("atoms not set")
        if atoms is None:
            atoms = self.atoms
        if self.model_type not in ["DipoleMACE", "DipolePolarizabilityMACE"]:
            raise NotImplementedError(
                "Only implemented for DipoleMACE or DipolePolarizabilityMACE models"
            )
        batch = self._atoms_to_batch(atoms)
        outputs = [
            model(
                self._clone_batch(batch).to_dict(),
                compute_dielectric_derivatives=True,
                training=self.use_compile,
            )
            for model in self.models
        ]
        dipole_derivatives = [
            output["dmu_dr"].clone().detach().cpu().numpy() for output in outputs
        ]
        if self.models[0].use_polarizability:
            polarizability_derivatives = [
                output["dalpha_dr"].clone().detach().cpu().numpy() for output in outputs
            ]
            if self.num_models == 1:
                dipole_derivatives = dipole_derivatives[0]
                polarizability_derivatives = polarizability_derivatives[0]
            del outputs, batch, atoms
            return dipole_derivatives, polarizability_derivatives
        if self.num_models == 1:
            return dipole_derivatives[0]
        del outputs, batch, atoms
        return dipole_derivatives

    def get_hessian(self, atoms=None):
        if atoms is None and self.atoms is None:
            raise ValueError("atoms not set")
        if atoms is None:
            atoms = self.atoms
        if self.model_type != "MACE":
            raise NotImplementedError("Only implemented for MACE models")
        batch = self._atoms_to_batch(atoms)
        hessians = [
            model(
                self._clone_batch(batch).to_dict(),
                compute_hessian=True,
                compute_stress=False,
                training=self.use_compile,
            )["hessian"]
            for model in self.models
        ]
        hessians = [hessian.detach().cpu().numpy() for hessian in hessians]
        if self.num_models == 1:
            return hessians[0]
        return hessians

    def get_descriptors(self, atoms=None, invariants_only=True, num_layers=-1):
        """Extracts the descriptors from MACE model.
        :param atoms: ase.Atoms object
        :param invariants_only: bool, if True only the invariant descriptors are returned
        :param num_layers: int, number of layers to extract descriptors from, if -1 all layers are used
        :return: np.ndarray (num_atoms, num_interactions, invariant_features) of invariant descriptors if num_models is 1 or list[np.ndarray] otherwise
        """
        if atoms is None and self.atoms is None:
            raise ValueError("atoms not set")
        if atoms is None:
            atoms = self.atoms
        if self.model_type != "MACE":
            raise NotImplementedError("Only implemented for MACE models")
        num_interactions = int(self.models[0].num_interactions)
        if num_layers == -1:
            num_layers = num_interactions
        batch = self._atoms_to_batch(atoms)
        descriptors = [model(batch.to_dict())["node_feats"] for model in self.models]

        irreps_out = o3.Irreps(str(self.models[0].products[0].linear.irreps_out))
        l_max = irreps_out.lmax
        num_invariant_features = irreps_out.dim // (l_max + 1) ** 2
        per_layer_features = [irreps_out.dim for _ in range(num_interactions)]
        per_layer_features[-1] = (
            num_invariant_features  # Equivariant features not created for the last layer
        )

        if invariants_only:
            descriptors = [
                extract_invariant(
                    descriptor,
                    num_layers=num_layers,
                    num_features=num_invariant_features,
                    l_max=l_max,
                )
                for descriptor in descriptors
            ]
        to_keep = np.sum(per_layer_features[:num_layers])
        descriptors = [
            descriptor[:, :to_keep].detach().cpu().numpy() for descriptor in descriptors
        ]

        if self.num_models == 1:
            return descriptors[0]
        return descriptors
