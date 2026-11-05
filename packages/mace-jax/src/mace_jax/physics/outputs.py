"""Forces -dE/dR [eV/Å], stress +dE/dstrain/V [eV/Å³], virials -dE/dstrain [eV]."""

from __future__ import annotations

import jax
import jax.numpy as jnp
from mace_core.outputs import MACEOutput


def evaluate(
    model, graph, compute=("forces", "stress", "virials")
) -> MACEOutput[jax.Array]:
    unknown = set(compute) - {"energy", "node_energies", "forces", "stress", "virials"}
    if unknown:
        raise ValueError(f"JAX does not support requested outputs {sorted(unknown)}")
    if not set(compute) & {"forces", "stress", "virials"}:
        return model(graph)
    cells = graph["cell"].reshape((-1, 3, 3))
    strain = jnp.zeros_like(cells)
    edge_graph = graph["batch"][graph["edge_index"][0]]

    def energy(positions, displacement):
        symmetric = (displacement + jnp.swapaxes(displacement, -1, -2)) / 2
        shifted = dict(graph)
        shifted["positions"] = positions + jnp.einsum(
            "ni,nij->nj", positions, symmetric[graph["batch"]]
        )
        shifted["shifts"] = graph["shifts"] + jnp.einsum(
            "ei,eij->ej", graph["shifts"], symmetric[edge_graph]
        )
        shifted["cell"] = cells + jnp.einsum("nij,njk->nik", cells, symmetric)
        output = model(shifted)
        return jnp.sum(output.total_energy), (output.total_energy, output.node_energies)

    (_, (total, nodes)), (position_gradient, strain_gradient) = jax.value_and_grad(
        energy, argnums=(0, 1), has_aux=True
    )(graph["positions"], strain)
    output = MACEOutput(total_energy=total, node_energies=nodes)
    if "forces" in compute:
        output.forces = -position_gradient
    if "virials" in compute:
        output.virials = -strain_gradient
    if "stress" in compute:
        volumes = jnp.abs(jnp.linalg.det(cells))
        periodic = jnp.any(graph["pbc"].reshape((-1, 3)), axis=-1)
        valid = periodic & (volumes > 0)
        stress = strain_gradient / jnp.where(valid, volumes, 1)[:, None, None]
        output.stress = jnp.where(
            valid[:, None, None] & jnp.isfinite(stress) & (jnp.abs(stress) < 1e10),
            stress,
            0,
        )
    return output
