"""Compiled inference with a reusable topology and differentiable geometry."""

from __future__ import annotations

import logging

import jax
import jax.numpy as jnp
from mace_core.elements import AtomicNumberTable
from mace_core.graph import (
    AutoPadding,
    FixedPadding,
    NoPadding,
    PaddingBudget,
    padding_budget,
)
from mace_core.outputs import MACEOutput

from mace_jax.graph import pad_batch, unpad_outputs
from mace_jax.physics.outputs import strain_outputs

logger = logging.getLogger(__name__)
_TOPOLOGY = (
    "batch",
    "ptr",
    "pbc",
    "node_attrs",
    "node_atomic_numbers",
    "edge_index",
    "unit_shifts",
)


class JITPipeline:
    """A single model and padding policy. Counters increment when JAX traces Python."""

    def __init__(self, apply, params, policy=None):
        if isinstance(policy, NoPadding):
            raise ValueError("JIT inference requires fixed or auto padding")
        self.policy = AutoPadding() if policy is None else policy
        self.capacity = None
        self.r_max = apply.config.model.r_max
        self.elements = AtomicNumberTable(apply.elements)
        self.params = params
        self.energy_traces = 0
        self.output_traces = 0

        @jax.jit
        def energies(state, positions, cell, topology):
            self.energy_traces += 1
            edge_graph = topology["batch"][topology["edge_index"][0]]
            shifts = jnp.einsum(
                "ei,eij->ej",
                topology["unit_shifts"],
                cell[edge_graph],
                precision=jax.lax.Precision.HIGHEST,
            )
            graph = dict(topology, positions=positions, cell=cell, shifts=shifts)
            output = apply(state, graph, compute=())
            return output.total_energy, output.node_energies

        @jax.jit
        def outputs(state, positions, cell, topology):
            self.output_traces += 1

            def strained(moved, deformed, _strain):
                return energies(state, moved, deformed, topology)

            out = strain_outputs(
                strained, positions, cell, topology, ("forces", "stress", "virials")
            )
            return (
                out.total_energy,
                out.node_energies,
                out.forces,
                out.stress,
                out.virials,
            )

        self._energies = energies
        self._outputs = outputs

    def prepare(self, graph):
        """Pad a rebuilt neighbour list to the retained capacity."""
        policy = self.policy
        previous = self.capacity
        if previous is not None:
            fixed = FixedPadding(
                nodes_budget=previous.nodes,
                edges_budget=previous.edges,
                graphs_budget=previous.graphs,
                on_overflow=policy.on_overflow,
            )
            sizes = dict(
                n_nodes=graph["positions"].shape[0],
                n_edges=graph["edge_index"].shape[1],
                n_graphs=graph["ptr"].shape[0] - 1,
            )
            needed = padding_budget(fixed, **sizes)
            if needed != previous and isinstance(policy, AutoPadding):
                target = padding_budget(policy, **sizes)
                assert target is not None and needed is not None
                fixed = FixedPadding(
                    nodes_budget=max(needed.nodes, target.nodes),
                    edges_budget=max(needed.edges, target.edges),
                    graphs_budget=max(needed.graphs, target.graphs),
                )
            policy = fixed
        padded, info = pad_batch(
            graph, policy, self.r_max, atomic_number_table=self.elements
        )
        self.capacity = PaddingBudget(
            padded["positions"].shape[0],
            padded["edge_index"].shape[1],
            padded["ptr"].shape[0] - 1,
        )
        if previous is not None and self.capacity != previous:
            logger.info(
                "JAX inference padding grew from %s to %s", previous, self.capacity
            )
        return padded, info

    def energy(self, positions, cell, graph):
        """Per-graph energies [eV] for padded positions and lattice vectors [Å]."""
        topology = {key: graph[key] for key in _TOPOLOGY}
        return self._energies(self.params, positions, cell, topology)[0]

    def __call__(self, graph):
        """Evaluate a natural batch; return real graph and atom rows only."""
        # Keep labels and host metadata outside the inference boundary.
        inputs = {key: graph[key] for key in ("positions", "cell", *_TOPOLOGY)}
        padded, info = self.prepare(inputs)
        topology = {key: padded[key] for key in _TOPOLOGY}
        values = self._outputs(
            self.params, padded["positions"], padded["cell"], topology
        )
        return unpad_outputs(MACEOutput(*values), info)
