"""The DATA-3 construction, including its inactive edges and its host-only counts."""

from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from ase import Atoms
from mace_core.elements import AtomicNumberTable
from mace_core.graph import FixedPadding, NoPadding, PaddingInfo
from mace_jax.graph import graph_from_atoms, pad_batch, unpad_outputs
from mace_jax.nn.radial import polynomial_envelope


@pytest.mark.parametrize("dtype", ["float32", "float64"])
def test_padding_is_append_only_and_uses_inert_last_atom_loops(dtype):
    jax.config.update("jax_enable_x64", True)
    table = AtomicNumberTable((6, 8))
    graph: dict[str, Any] = graph_from_atoms(
        Atoms("CO", positions=[[0, 0, 0], [1, 0, 0]]),
        elements=(6, 8),
        cutoff=2.5,
        dtype=dtype,
    )
    graph.update(
        sender_order=jnp.arange(2),
        sender_counts=jnp.ones(2, dtype=int),
        structure_sizes=(2,),
        structure_periodic=(False,),
        structure_padding=(False,),
        tags=("CO",),
        custom=jnp.ones((2, 2)),
    )

    def pad(data, policy):
        return pad_batch(
            data, policy, 2.5, atomic_number_table=table, features={"custom": "atom"}
        )

    natural, info = pad(graph, NoPadding())
    assert natural.keys() == graph.keys() and info == PaddingInfo(2, 2, 1)
    budget = FixedPadding(
        nodes_budget=5, edges_budget=7, graphs_budget=3, on_overflow="error"
    )
    padded, info = pad(graph, budget)
    for key, value in graph.items():
        if key == "edge_index":
            np.testing.assert_array_equal(padded[key][:, :2], value)
        else:
            np.testing.assert_array_equal(padded[key][: len(value)], value)
    assert info == PaddingInfo(2, 2, 1)
    np.testing.assert_array_equal(padded["node_attrs"][2:], [[1, 0]] * 3)
    np.testing.assert_array_equal(padded["node_atomic_numbers"][2:], 6)
    np.testing.assert_array_equal(padded["edge_index"][:, 2:], 4)
    np.testing.assert_array_equal(padded["unit_shifts"][2:], [[1, 0, 0]] * 5)
    np.testing.assert_array_equal(padded["shifts"][2:], [[5, 0, 0]] * 5)
    np.testing.assert_array_equal(padded["cell"][1:], [5 * np.eye(3)] * 2)
    np.testing.assert_allclose(padded["rcell"][1:], [np.eye(3) * (2 * np.pi / 5)] * 2)
    np.testing.assert_array_equal(padded["volume"][1:], 125)
    for key in ("pbc", "total_charge", "positions", "custom"):
        np.testing.assert_array_equal(padded[key][len(graph[key]) :], 0)
    np.testing.assert_array_equal(padded["total_spin"][1:], 1)
    np.testing.assert_array_equal(padded["batch"], [0, 0, 2, 2, 2])
    np.testing.assert_array_equal(padded["ptr"], [0, 2, 2, 5])
    np.testing.assert_array_equal(padded["sender_order"], np.arange(7))
    np.testing.assert_array_equal(padded["sender_counts"], [1, 1, 0, 0, 5])
    assert padded["structure_sizes"] == (2, 0, 3)
    assert padded["structure_periodic"] == (False, False, False)
    assert padded["structure_padding"] == (False, True, True)
    assert padded["tags"] == ("CO", None, None)
    src, dst = padded["edge_index"]
    vectors = padded["positions"][dst] - padded["positions"][src] + padded["shifts"]
    cutoff = jax.jit(
        lambda v: polynomial_envelope(jnp.linalg.norm(v, axis=-1), 2.5, 5)
    )(vectors)
    np.testing.assert_array_equal(cutoff[2:], 0)
    assert np.all(np.asarray(cutoff[:2]) > 0)
    out = unpad_outputs(
        {"forces": padded["positions"], "energy": padded["volume"]}, info
    )
    np.testing.assert_array_equal(out["forces"], graph["positions"])
    np.testing.assert_array_equal(out["energy"], graph["volume"])
    with pytest.raises(ValueError, match="unknown_label"):
        pad(dict(graph, unknown_label=graph["forces"]), budget)
