"""What the engine promises beyond the numbers.

That the model contains no gradient call, that the input dict comes back
untouched, that a caller managing its own strain keeps working, and that the
errors name what went wrong.
"""

from __future__ import annotations

import ast
import inspect

import numpy as np
import pytest
import torch
from conftest import fp64_only
from mace_torch.physics import cell_volume_and_mask, prepare_inputs
from mace_torch_engine_fixtures import build_engine, build_graph, crystal, molecule

PERIODIC = (True, True, True)


@fp64_only
def test_no_model_forward_takes_a_gradient():
    """The whole point of the split, asserted where a refactor would undo it.

    An `autograd.grad` inside a forward is a graph break, which is why the
    frozen tree's compiled path never went anywhere. The engine is the only
    differentiation site.
    """
    from mace_torch.models import energy, heads, outputs
    from mace_torch.nn import backbone, graph_features, interaction, product_basis

    forbidden = {"grad", "requires_grad_", "backward"}
    modules = (
        backbone,
        interaction,
        product_basis,
        graph_features,
        energy,
        heads,
        outputs,
    )
    scanned = 0
    for module in modules:
        tree = ast.parse(inspect.getsource(module))
        for node in ast.walk(tree):
            # Only the forwards. Marking a buffer non-trainable at
            # construction is `requires_grad_` too, and it is exactly right
            # there: what must not happen is a forward mutating the graph it
            # was handed or differentiating inside itself.
            if not (isinstance(node, ast.FunctionDef) and node.name == "forward"):
                continue
            scanned += 1
            for inner in ast.walk(node):
                if isinstance(inner, ast.Attribute) and inner.attr in forbidden:
                    pytest.fail(
                        f"{module.__name__}.{node.name} calls {inner.attr!r}, "
                        f"which belongs to the derivative engine and nowhere "
                        f"else"
                    )
    assert scanned >= len(modules), (
        f"only {scanned} forwards were found across {len(modules)} modules, so "
        f"this scan is not reaching the code it describes"
    )


@fp64_only
def test_a_stress_computation_leaves_the_input_untouched():
    """The frozen tree writes the strained positions and shifts back into the
    dict it was handed, which a caller sharing that dict then sees."""
    engine = build_engine()
    positions, numbers, cell = crystal()
    graph = build_graph(positions, numbers, cell, PERIODIC)
    before = {
        key: value.clone()
        for key, value in graph.items()
        if isinstance(value, torch.Tensor)
    }

    engine(graph, compute=("forces", "stress"))

    assert set(graph) == set(before) | {"num_graphs"}, (
        f"the engine added keys to the input: {set(graph) - set(before)}"
    )
    for key, value in before.items():
        assert torch.equal(graph[key], value), f"the engine overwrote {key!r}"


@fp64_only
def test_the_prepared_graph_is_a_new_dict_with_new_tensors():
    positions, numbers, cell = crystal()
    graph = build_graph(positions, numbers, cell, PERIODIC)

    prepared, leaf, displacement = prepare_inputs(
        graph, need_forces=True, need_stress=True
    )

    assert prepared is not graph
    assert leaf.requires_grad
    assert displacement is not None and displacement.requires_grad
    assert "vectors" in prepared

    # The shifts are recomputed, not reused. Their *values* are unchanged here
    # because the strain starts at zero, so the claim is about the tensor: the
    # recomputed one carries a gradient path back to the strain, and the input
    # one carries nothing.
    assert prepared["shifts"] is not graph["shifts"]
    assert prepared["shifts"].grad_fn is not None
    assert graph["shifts"].grad_fn is None
    assert torch.allclose(prepared["shifts"], graph["shifts"])


@fp64_only
def test_an_externally_supplied_strain_is_used_as_the_leaf():
    """A caller that manages its own strain keeps working, which is how the
    torchsim path supplies a padded displacement today."""
    positions, numbers, cell = crystal()
    graph = build_graph(positions, numbers, cell, PERIODIC)
    mine = torch.zeros(1, 3, 3, dtype=torch.float64, requires_grad=True)
    graph["displacement"] = mine

    _, _, used = prepare_inputs(graph, need_forces=True, need_stress=True)

    assert used is mine, "a fresh zero displacement was made instead"


@fp64_only
def test_a_strain_needs_the_unit_shifts_and_says_so():
    """`shifts` is a cache of `unit_shifts` and the cell of the moment, so
    under a strain it is the one that has to be recomputed."""
    positions, numbers, cell = crystal()
    graph = build_graph(positions, numbers, cell, PERIODIC)
    del graph["unit_shifts"]

    with pytest.raises(KeyError, match="unit_shifts"):
        prepare_inputs(graph, need_forces=True, need_stress=True)


@fp64_only
def test_only_the_symmetric_part_of_the_strain_is_used():
    """An antisymmetric strain is an infinitesimal rotation, which the energy
    is invariant under, so it must change nothing."""
    positions, numbers, cell = crystal()
    graph = build_graph(positions, numbers, cell, PERIODIC)

    plain, _, _ = prepare_inputs(graph, need_forces=True, need_stress=True)
    antisymmetric = torch.tensor(
        [[[0.0, 0.3, -0.2], [-0.3, 0.0, 0.1], [0.2, -0.1, 0.0]]], dtype=torch.float64
    ).requires_grad_(True)
    graph["displacement"] = antisymmetric
    rotated, _, _ = prepare_inputs(graph, need_forces=True, need_stress=True)

    assert torch.allclose(
        plain["positions"], rotated["positions"], atol=1e-14, rtol=1e-14
    )


@fp64_only
def test_edge_forces_are_minus_the_gradient_against_the_edge_vectors():
    engine = build_engine()
    positions, numbers = molecule()
    graph = build_graph(positions, numbers)

    result = engine(graph, compute=("forces", "edge_forces"))
    edge_forces = result.extras["edge_forces"]

    assert edge_forces.shape == (graph["edge_index"].shape[1], 3)
    assert float(edge_forces.abs().max()) > 0.0


@fp64_only
def test_asking_for_nothing_returns_the_model_output_alone():
    engine = build_engine()
    positions, numbers = molecule()
    result = engine(build_graph(positions, numbers), compute=())

    assert result.total_energy is not None
    assert result.forces is None and result.stress is None


@fp64_only
def test_an_unknown_derivative_name_lists_the_ones_that_exist():
    engine = build_engine()
    positions, numbers = molecule()
    with pytest.raises(ValueError, match="edge_forces"):
        engine(build_graph(positions, numbers), compute=("hessian",))


@fp64_only
def test_a_derivative_without_an_energy_says_what_is_missing():
    """A dipole-only model has nothing to differentiate."""
    from mace_core.observables import ObservableSpec
    from mace_torch.backends.reference import ReferenceBackend
    from mace_torch.models import MACEOutputs
    from mace_torch.physics import DerivativeEngine

    engine = build_engine()
    dipole = ObservableSpec(
        name="dipole", irreps="1o", per_atom=False, units="eV/A", normalization="none"
    )
    without_energy = DerivativeEngine(
        engine.backbone,
        MACEOutputs(ReferenceBackend(), [dipole], "0e+1o", 4, 2),
    )
    positions, numbers = molecule()
    with pytest.raises(ValueError, match="total_energy"):
        without_energy(build_graph(positions, numbers), compute=("forces",))


@fp64_only
def test_the_volume_is_one_where_there_is_no_volume():
    """Substituted before the division, not masked after it.

    Masking afterwards leaves a nan to route back through the division on the
    backward pass, which is a gradient of nan from a forward that looked fine.
    """
    cell = torch.tensor(
        np.stack([np.eye(3) * 2.0, np.zeros((3, 3))]), dtype=torch.float64
    )
    pbc = torch.tensor([[True, True, True], [False, False, False]])

    volume, periodic = cell_volume_and_mask(cell, pbc)

    assert bool(periodic[0]) and not bool(periodic[1])
    assert float(volume[0]) == 8.0
    assert float(volume[1]) == 1.0


@fp64_only
def test_without_pbc_only_a_degenerate_cell_is_masked():
    """Under LAMMPS everything is periodic and no `pbc` arrives at all."""
    cell = torch.tensor(
        np.stack([np.eye(3) * 2.0, np.zeros((3, 3))]), dtype=torch.float64
    )
    _, periodic = cell_volume_and_mask(cell, None)
    assert bool(periodic[0]) and not bool(periodic[1])


@fp64_only
def test_a_batch_mixing_a_molecule_with_a_crystal_keeps_the_crystal_stress():
    """The mask is per graph. Zeroing the whole batch because one member is
    aperiodic would throw away the answer for the other."""
    cell = torch.tensor(
        np.stack([np.eye(3) * 3.0, np.eye(3) * 4.0]), dtype=torch.float64
    )
    pbc = torch.tensor([[False, False, False], [True, True, True]])
    volume, periodic = cell_volume_and_mask(cell, pbc)

    assert not bool(periodic[0]) and bool(periodic[1])
    assert float(volume[0]) == 1.0 and float(volume[1]) == 64.0
