"""ML-IAP's in-process Python interface to a neutral deployment bundle.

The callback ABI, environment controls and forward/reverse exchange follow
``mace/calculators/lammps_mliap_mace.py`` and ``mace/tools/utils.py::LAMMPS_MP``.
Model construction and first-order edge derivatives use the V1 loader/engine.
The separate ``mace/native`` pair style runs the exported artifact without Python.
"""

from __future__ import annotations

import logging
import os
import time
from importlib import import_module
from pathlib import Path
from typing import cast

import numpy as np
import torch
from ase.data import chemical_symbols
from mace_core.edges import plan_edges
from torch import nn
from torch.autograd.function import once_differentiable

from mace_torch.backends import resolve_backend
from mace_torch.deploy.loader import DeployError, load_deployed
from mace_torch.deploy.manifest import DeployManifest
from mace_torch.graph.edges import apply_edge_plan
from mace_torch.graph.padding import pad_batch
from mace_torch.models.base import MACEModel
from mace_torch.physics.outputs import DerivativeEngine

try:
    MLIAPUnified = import_module("lammps.mliap.mliap_unified_abc").MLIAPUnified
except ImportError:
    # Keep the contract tier and deployment tools usable without liblammps.
    MLIAPUnified = object


logger = logging.getLogger(__name__)


def _env_bool(name):
    return os.environ.get(name, "false").lower() in {"true", "1", "t", "yes"}


def _synchronize(device):
    if device.type == "cuda":
        torch.cuda.synchronize(device)


class _Halo(torch.autograd.Function):
    """LAMMPS owns the communication and its adjoint, including MPI reduction."""

    @staticmethod
    def forward(ctx, features, data, transport_device):
        ctx.data, ctx.transport_device = data, transport_device
        ctx.shape = features.shape
        ctx.device = features.device
        local = features[: data.nlocal].flatten(1)
        send = local.new_zeros((data.ntotal, local.shape[1]))
        send[: data.nlocal] = local
        send = send.to(transport_device).contiguous()
        received = torch.empty_like(send)
        _synchronize(transport_device)
        data.forward_exchange(send, received, send.shape[1])
        _synchronize(transport_device)
        result = torch.zeros_like(features)
        result[: data.ntotal] = received.to(features.device).view_as(
            features[: data.ntotal]
        )
        return result

    @staticmethod
    @once_differentiable
    def backward(ctx, gradient):
        data = ctx.data
        send = gradient[: data.ntotal].flatten(1).to(ctx.transport_device).contiguous()
        received = torch.empty_like(send)
        _synchronize(ctx.transport_device)
        data.reverse_exchange(send, received, send.shape[1])
        _synchronize(ctx.transport_device)
        result = gradient.new_zeros(ctx.shape)
        result[: data.nlocal] = (
            received[: data.nlocal].to(ctx.device).view_as(result[: data.nlocal])
        )
        return result, None, None


class _Region:
    """Compile a pure layer forward, leaving its communication pre-hook eager."""

    def __init__(self, forward):
        self.eager = forward
        self.compiled = torch.compile(forward, dynamic=True, fullgraph=True)

    def __call__(self, *args, **kwargs):
        from torch._dynamo.exc import BackendCompilerFailed, Unsupported

        if self.compiled is not None:
            try:
                return self.compiled(*args, **kwargs)
            except (BackendCompilerFailed, Unsupported) as error:
                logger.warning(
                    "ML-IAP layer compilation failed (%s), using eager evaluation",
                    error,
                )
                self.compiled = None
        return self.eager(*args, **kwargs)


class _OwnedEnergy(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model
        self.n_local = 0

    def forward(self, graph):
        output = self.model(graph)
        output.total_energy = output.node_energies[: self.n_local].sum().reshape(1)
        return output


class MACEMLIAPUnified(MLIAPUnified):
    """Activate ML-IAP, call ``load_unified(instance)``, then use EXISTS.

    ``halo_exchange=True`` needs KOKKOS forward/reverse exchange for multiple
    layers. With it disabled, use ``pair_style mliap unified EXISTS 1`` and a
    ghost shell covering ``manifest.receptive_field``. Energies are eV and
    ML-IAP pair forces are +dE/d(x_j-x_i), in eV/Å, in the caller's edge order.
    """

    def __init__(self, bundle_path: str | Path):
        super().__init__()
        self.path = Path(bundle_path).resolve()
        self.manifest = DeployManifest.read(self.path / "deploy.json")
        if self.manifest.target != "lammps-mliap":
            raise DeployError("MACEMLIAPUnified requires a lammps-mliap bundle")
        self.element_types = [chemical_symbols[z] for z in self.manifest.species]
        self.num_species = len(self.element_types)
        self.ndescriptors = self.nparams = 1
        cutoff = (
            self.manifest.r_max
            if self.manifest.halo_exchange
            else self.manifest.receptive_field
        )
        self.rcutfac = cutoff / 2
        self.debug_time = _env_bool("MACE_TIME")
        self.debug_profile = _env_bool("MACE_PROFILE")
        self.profile_start = int(os.environ.get("MACE_PROFILE_START", "5"))
        self.profile_end = int(os.environ.get("MACE_PROFILE_END", "10"))
        self.allow_cpu = _env_bool("MACE_ALLOW_CPU")
        self.force_cpu = _env_bool("MACE_FORCE_CPU")
        self.step = 0
        self.deployed = None
        self.data = None

    def _setup(self, data, transport_device):
        kokkos = "kokkos" in type(data).__module__.lower()
        if (
            kokkos
            and transport_device.type == "cpu"
            and not (self.allow_cpu or self.force_cpu)
        ):
            raise ValueError(
                "GPU requested but tensor is on CPU. Set MACE_ALLOW_CPU=true "
                "to allow CPU computation."
            )
        self.device = torch.device("cpu") if self.force_cpu else transport_device
        deployed = load_deployed(self.path, device=str(self.device))
        if deployed.config.model.model not in {"plain", "scale_shift"}:
            raise DeployError("ML-IAP requires a plain or scale_shift energy model")
        if deployed.config.model.graph_features:
            raise DeployError("ML-IAP cannot supply extra graph features")
        self.dtype = getattr(torch, self.manifest.dtype)
        self.head = deployed.heads.index(deployed.select_head())
        self.numbers = torch.tensor(deployed.z_table.zs, device=self.device)
        self.attributes = torch.eye(
            self.num_species, device=self.device, dtype=self.dtype
        )
        backend = resolve_backend(deployed.config.model.backend)
        self.edge_plan = plan_edges(backend.capabilities().edge_features, ())
        model = cast(MACEModel, deployed.model)
        self.owned = _OwnedEnergy(model)
        self.engine = DerivativeEngine(self.owned, deployed.derivative_engine.energy)
        self.engine.eval().requires_grad_(False)
        interactions = model.backbone.interactions
        if self.manifest.halo_exchange:
            for layer in interactions[1:]:
                layer.register_forward_pre_hook(self._exchange)
        if self.manifest.compile_regions:
            for layer in (*interactions, *model.backbone.products):
                layer.forward = _Region(layer.forward)
        self.transport_device = transport_device
        self.deployed = deployed

    def _exchange(self, module, inputs):
        return (
            _Halo.apply(inputs[0], self.data, self.transport_device),
            *inputs[1:],
        )

    def compute_forces(self, data):
        start = time.perf_counter()
        if data.nlistatoms != data.nlocal:
            raise RuntimeError("ML-IAP requires all local atoms in the neighbor list")
        if self.manifest.num_interactions > 1:
            if self.manifest.halo_exchange:
                if not all(
                    callable(getattr(data, name, None))
                    for name in ("forward_exchange", "reverse_exchange")
                ):
                    raise RuntimeError(
                        f"{type(data).__name__} lacks forward/reverse ghost exchange. "
                        "Build LAMMPS with -D PKG_KOKKOS=yes and use -k on -sf kk, "
                        "or deploy a single-layer model."
                    )
            elif data.ntotal > data.nlocal and data.natomneigh <= data.nlocal:
                raise RuntimeError(
                    "halo_exchange=False requires ghost neighbors. Use pair_style "
                    "mliap unified EXISTS 1 and the full receptive_field ghost shell"
                )
        elements = torch.as_tensor(data.elems)
        if self.deployed is None:
            self._setup(data, elements.device)
        elif elements.device != self.transport_device:
            raise RuntimeError("ML-IAP transport device changed after initialization")
        self.step += 1
        if (
            self.debug_profile
            and self.step == self.profile_start
            and self.device.type == "cuda"
        ):
            torch.cuda.profiler.start()
        if self.debug_profile and self.step == self.profile_end:
            if self.device.type == "cuda":
                torch.cuda.profiler.stop()
            raise SystemExit(0)
        self.owned.n_local = data.nlocal
        elements = elements.to(device=self.device, dtype=torch.long)
        # ML-IAP leaves these pointers null when the first step has no pairs.
        edge_index = elements.new_empty((2, 0))
        shifts = self.attributes.new_empty((0, 3))
        if data.npairs:
            edge_index = torch.stack(
                [
                    torch.as_tensor(data.pair_j, device=self.device, dtype=torch.long),
                    torch.as_tensor(data.pair_i, device=self.device, dtype=torch.long),
                ]
            )
            # For edge j -> i, V1 uses x_i - x_j and ML-IAP supplies x_j - x_i.
            shifts = -torch.as_tensor(data.rij, device=self.device, dtype=self.dtype)
        graph = {
            "positions": self.attributes.new_zeros((data.ntotal, 3)),
            "shifts": shifts,
            "edge_index": edge_index,
            "node_attrs": self.attributes[elements],
            "node_atomic_numbers": self.numbers[elements],
            "batch": torch.zeros_like(elements),
            "ptr": elements.new_tensor([0, data.ntotal]),
            "head": elements.new_tensor([self.head]),
        }
        if self.manifest.pad_spec is not None:
            graph, _ = pad_batch(
                graph,
                self.manifest.pad_spec,
                self.manifest.r_max,
                atomic_number_table=self.deployed.z_table,
            )
        order = None
        if any(a.name == "sort_by_sender" for a in self.edge_plan.adapters):
            order = torch.argsort(graph["edge_index"][0], stable=True)
        graph = apply_edge_plan(graph, self.edge_plan)
        self.data = data
        try:
            with torch.enable_grad():
                output = self.engine(graph, compute=("edge_forces",), training=False)
        finally:
            self.data = None
        # Reverse the coordinate change above: dE/d(rij) = -dE/d(V1 vector).
        pair_forces = -output.extras["edge_forces"].detach()
        if order is not None:
            pair_forces = torch.empty_like(pair_forces).index_copy_(
                0, order, pair_forces
            )
        pair_forces = (
            pair_forces[: data.npairs]
            .to(device=self.transport_device, dtype=torch.float64)
            .contiguous()
        )
        site = output.node_energies[: data.nlocal].detach().double()
        # eatoms uses ilist order, which can differ from atom-storage order.
        if data.nlocal:
            iatoms = torch.as_tensor(data.iatoms, device=self.device, dtype=torch.long)
            site = site[iatoms[: data.nlocal]]
        data.energy = site.sum().item()
        if callable(getattr(data, "update_pair_forces_gpu", None)):
            if data.nlocal:
                torch.as_tensor(data.eatoms).copy_(site)
            _synchronize(self.transport_device)
            if data.npairs:
                data.update_pair_forces_gpu(pair_forces)
        else:
            if data.nlocal:
                data.eatoms = np.ascontiguousarray(site.cpu().numpy())
            if data.npairs:
                data.update_pair_forces(np.ascontiguousarray(pair_forces.cpu().numpy()))
        _synchronize(self.transport_device)
        if self.debug_time:
            logger.info(
                "Timer - ML-IAP step: %.3f ms", 1000 * (time.perf_counter() - start)
            )

    def compute_descriptors(self, data):
        """Unused by the unified inference interface."""

    def compute_gradients(self, data):
        """Unused by the unified inference interface."""
