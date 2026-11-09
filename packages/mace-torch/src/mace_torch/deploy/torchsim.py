"""TorchSim's ModelInterface over the shared V1 deployment and graph builders.

The state/cell mapping follows TorchSim's pinned ModelInterface and SimState:
https://github.com/TorchSim/torch-sim/tree/d6d8e9fc30129e01a05b551c2d71912821c9b68b/torch_sim
Cells there hold column vectors; the MACE graph uses row vectors. Units agree.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from pathlib import Path
from typing import Any

import numpy as np
import torch
from mace_core.config.precision import PrecisionConfig
from mace_core.data.configuration import Configuration
from mace_core.graph import GRAPH_SCHEMA, FixedPadding, NoPadding, PaddingInfo
from mace_core.neighbors import NeighborListConfig
from mace_core.observables import DEFAULT_CATALOGUE
from torch_sim.models.interface import ModelInterface  # ty: ignore[unresolved-import]

from mace_torch.deploy.loader import DeployError, load_deployed
from mace_torch.graph import build_batch, pad_batch, unpad_outputs
from mace_torch.neighbors import resolve_neighbor_provider


class MaceTorchSimModel(ModelInterface):
    """Load a neutral checkpoint/bundle once and evaluate TorchSim states.

    ``compile_mode`` enables full-graph, static-shape compilation of the model.
    ``padding`` can set explicit budgets; otherwise compiled evaluation uses
    power-of-two buckets including a dummy atom and structure. The shared graph
    library owns padding and the derivative engine plans edges afterwards.

    Extra inputs use their declared V1 names. ``input_names`` maps a declared
    name to a state attribute when bridging an older caller, for example
    ``{"external_field": "external_E_field"}``. Geometry is read afresh each
    step, including the cell, neighbours, species and externally supplied strain.
    """

    def __init__(
        self,
        bundle: str | Path,
        *,
        device: str = "cpu",
        head: str | None = None,
        backend: str | None = None,
        precision: PrecisionConfig | None = None,
        compute_forces: bool = True,
        compute_stress: bool = True,
        compile_mode: str | None = None,
        compile_backend: str | Callable = "inductor",
        padding: FixedPadding | NoPadding | None = None,
        neighbor_list: NeighborListConfig | None = None,
        input_names: Mapping[str, str] | None = None,
    ) -> None:
        super().__init__()
        self.deployed = load_deployed(
            bundle, device=device, backend=backend, precision=precision
        )
        if "energy" not in {spec.name for spec in self.deployed.outputs.observables}:
            raise DeployError("TorchSim requires a model with an energy observable")
        self._device = next(self.deployed.engine.parameters()).device
        self._dtype = self.deployed.dtype
        self._compute_forces = compute_forces
        self._compute_stress = compute_stress
        self.head = self.deployed.select_head(head)
        self.neighbors = resolve_neighbor_provider(neighbor_list, device=self.device)
        self.padding = padding
        self.compiled = compile_mode is not None
        catalogue = self.deployed.config.catalogue or DEFAULT_CATALOGUE
        specs = {spec.name: spec for spec in catalogue.inputs}
        specs.update({spec.name: spec for spec in self.deployed.model.input_specs})
        self.input_specs = tuple(
            spec for spec in specs.values() if spec.name not in {"pos", "strain"}
        )
        self.input_names = dict(input_names or {})
        unknown = self.input_names.keys() - {spec.name for spec in self.input_specs}
        if unknown:
            raise ValueError(
                f"input_names contains undeclared inputs: {sorted(unknown)}"
            )
        self.output_catalogue = catalogue
        self.extra_observables = self.deployed.model.extra_observables
        if self.compiled:
            self.deployed.derivative_engine.compile_model(
                mode=compile_mode, backend=compile_backend
            )

    def _graph(self, state) -> tuple[dict[str, Any], PaddingInfo]:
        if state.device != self.device:
            raise ValueError(
                f"state device {state.device} differs from model device {self.device}"
            )

        def host(value):
            return value.detach().cpu().numpy()

        positions = host(state.positions.to(torch.float64))
        numbers = host(state.atomic_numbers)
        counts = host(torch.bincount(state.system_idx)).tolist()
        cells = host(state.cell.mT.to(torch.float64))
        pbc = np.broadcast_to(host(state.pbc), (len(counts), 3))
        properties = {}
        for spec in self.input_specs:
            value = getattr(state, self.input_names.get(spec.name, spec.name), None)
            if value is not None:
                expected = len(numbers) if spec.per_atom else len(counts)
                if (
                    not torch.is_tensor(value)
                    or value.ndim == 0
                    or len(value) != expected
                ):
                    raise ValueError(
                        f"input {spec.name!r} needs {expected} leading rows"
                    )
                properties[spec.name] = host(value)
        configurations, start = [], 0
        for index, count in enumerate(counts):
            end = start + count
            configurations.append(
                Configuration(
                    atomic_numbers=numbers[start:end],
                    positions=positions[start:end],
                    cell=cells[index],
                    pbc=tuple(pbc[index]),
                    head=self.head,
                    properties={
                        spec.name: properties[spec.name][start:end]
                        if spec.per_atom
                        else properties[spec.name][index]
                        for spec in self.input_specs
                        if spec.name in properties
                    },
                )
            )
            start = end
        features = dict(self.deployed.config.model.feature_storages)
        # Custom differentiable inputs need the same declared row as embedded
        # features, even when no embedding consumes them.
        features.update(
            {
                spec.name: "atom" if spec.per_atom else "graph"
                for spec in self.input_specs
                if spec.name not in GRAPH_SCHEMA and spec.name in properties
            }
        )
        graph = build_batch(
            configurations,
            atomic_number_table=self.deployed.z_table,
            cutoff=self.deployed.r_max,
            neighbor_backend=self.neighbors,
            heads=self.deployed.heads,
            features=features,
            dtype=self.dtype,
            device=self.device,
        )
        displacement = getattr(state, "displacement", None)
        if displacement is not None:
            if displacement.shape != (len(counts), 3, 3):
                raise ValueError("displacement must have shape [n_systems, 3, 3]")
            graph["displacement"] = displacement.to(
                device=self.device, dtype=self.dtype
            )
        policy = self.padding
        if policy is None:
            if self.compiled:

                def bucket(count):
                    return 1 << max(count - 1, 0).bit_length()

                policy = FixedPadding(
                    nodes_budget=bucket(len(numbers) + 1),
                    edges_budget=bucket(graph["edge_index"].shape[1]),
                    graphs_budget=bucket(len(counts) + 1),
                )
            else:
                policy = NoPadding()
        return pad_batch(
            graph,
            policy,
            self.deployed.r_max,
            atomic_number_table=self.deployed.z_table,
            features=features,
        )

    @torch.inference_mode(False)
    def forward(self, state, **kwargs) -> dict[str, torch.Tensor]:
        if kwargs:
            raise TypeError(
                f"unrecognized TorchSim evaluation arguments: {sorted(kwargs)}"
            )
        graph, info = self._graph(state)
        compute = tuple(
            name
            for name, enabled in (
                ("forces", self.compute_forces),
                ("stress", self.compute_stress),
            )
            if enabled
        )
        with torch.enable_grad():
            output = self.deployed.compute(graph, compute=compute)
        output = unpad_outputs(
            output,
            info,
            catalogue=self.output_catalogue,
            extra_observables=self.extra_observables,
        )
        return {name: output.get(name).detach() for name in output.names()}
