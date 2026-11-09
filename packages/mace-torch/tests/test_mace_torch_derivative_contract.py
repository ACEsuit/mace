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
from mace_torch_engine_fixtures import (
    ENERGY,
    build_engine,
    build_graph,
    crystal,
    molecule,
)

PERIODIC = (True, True, True)


@pytest.mark.parametrize("trainable", [False, True])
@pytest.mark.parametrize("training", [False, True])
def test_position_independent_energy_has_zero_derivatives(dtype, trainable, training):
    from mace_core.outputs import MACEOutput
    from mace_torch.physics import DerivativeEngine

    class ConstantEnergy(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.energy = torch.nn.Parameter(
                torch.tensor([2.0], dtype=dtype), requires_grad=trainable
            )

        def forward(self, graph):
            return MACEOutput(total_energy=self.energy)

    engine = DerivativeEngine(ConstantEnergy(), ENERGY)
    graph = build_graph(np.zeros((1, 3)), [1])
    result = engine(
        graph,
        compute=("forces", "stress", "virials", "hessian", "edge_forces"),
        training=training,
    )
    assert torch.equal(result.total_energy, torch.tensor([2.0], dtype=dtype))
    for value in (
        result.forces,
        result.stress,
        result.virials,
        *result.extras.values(),
    ):
        assert torch.equal(value, torch.zeros_like(value))
    assert result.extras["hessian"].shape == (3, 1, 3)
    assert result.extras["edge_forces"].shape == (0, 3)
    if training:
        result.forces.square().sum().backward()
        assert engine.model.energy.grad is None


@fp64_only
def test_disabled_autograd_does_not_report_zero_forces():
    engine = build_engine()
    positions, numbers = molecule()
    with torch.no_grad(), pytest.raises(RuntimeError, match="gradient tracking"):
        engine(build_graph(positions, numbers), compute=("forces",))


@fp64_only
def test_no_model_forward_takes_a_gradient():
    """The whole point of the split, asserted where a refactor would undo it.

    An `autograd.grad` inside a forward is a graph break, which is why the
    frozen tree's compiled path never went anywhere. The engine is the only
    differentiation site.
    """
    from mace_torch.models import energy, outputs, readouts
    from mace_torch.nn import backbone, graph_features, interaction, product_basis

    forbidden = {"grad", "requires_grad_", "backward"}
    modules = (
        backbone,
        interaction,
        product_basis,
        graph_features,
        energy,
        readouts,
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
    keys_before = set(graph)
    host_before = {
        key: value
        for key, value in graph.items()
        if not isinstance(value, torch.Tensor)
    }
    before = {
        key: value.clone()
        for key, value in graph.items()
        if isinstance(value, torch.Tensor)
    }

    engine(graph, compute=("forces", "stress"))

    assert set(graph) == keys_before, (
        f"the engine added keys to the input: {set(graph) - set(before)}"
    )
    for key, value in host_before.items():
        assert graph[key] == value
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
    assert "vectors" not in prepared

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
def test_edge_forces_have_one_nonzero_vector_per_edge():
    engine = build_engine()
    positions, numbers = molecule()
    graph = build_graph(positions, numbers)

    result = engine(graph, compute=("forces", "edge_forces"))
    edge_forces = result.extras["edge_forces"]

    assert edge_forces.shape == (graph["edge_index"].shape[1], 3)
    assert float(edge_forces.abs().max()) > 0.0


@fp64_only
def test_asking_for_edge_forces_as_well_leaves_the_forces_as_they_were():
    """The edge vectors are differentiated where they are, as a function of
    the positions. Made a leaf of their own, they cut the energy off from the
    positions, and the forces asked for beside them came back as zeros."""
    engine = build_engine()
    positions, numbers = molecule()
    alone = engine(build_graph(positions, numbers), compute=("forces",))
    both = engine(build_graph(positions, numbers), compute=("forces", "edge_forces"))
    assert float(alone.forces.abs().max()) > 0.0
    torch.testing.assert_close(both.forces, alone.forces, rtol=0.0, atol=0.0)


@fp64_only
def test_the_edge_forces_add_up_to_the_forces():
    """Each edge vector is its receiver's position minus its sender's, so the
    export edge gradient contributes positively to the sender's force and
    negatively to the receiver's force."""
    engine = build_engine()
    positions, numbers = molecule()
    graph = build_graph(positions, numbers)
    result = engine(graph, compute=("forces", "edge_forces"))
    edge_forces = result.extras["edge_forces"]
    sender, receiver = graph["edge_index"]
    assembled = torch.zeros_like(result.forces)
    assembled.index_add_(0, sender, edge_forces)
    assembled.index_add_(0, receiver, -edge_forces)
    torch.testing.assert_close(assembled, result.forces, rtol=1e-12, atol=1e-14)


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
        engine(build_graph(positions, numbers), compute=("curvature",))


@fp64_only
def test_a_derivative_without_an_energy_says_what_is_missing():
    """A dipole-only model has nothing to differentiate."""
    from mace_core.observables import ObservableSpec
    from mace_torch.backends.reference import ReferenceBackend
    from mace_torch.models import MACEOutputs
    from mace_torch.physics import DerivativeEngine

    engine = build_engine()
    dipole = ObservableSpec(name="dipole", irreps="1o", per_atom=False, units="eV/A")
    from mace_torch_engine_fixtures import FeatureReadoutModel

    without_energy = DerivativeEngine(
        FeatureReadoutModel(
            engine.model.backbone,
            MACEOutputs(ReferenceBackend(), [dipole], ["0e+1o", "0e"], 4),
        ),
        ENERGY,
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


@fp64_only
def test_the_hessian_is_the_derivative_of_minus_the_forces():
    """Against central differences of the forces, which is what it is."""
    engine = build_engine()
    positions, numbers = molecule()
    result = engine(build_graph(positions, numbers), compute=("forces", "hessian"))
    hessian = result.extras["hessian"]
    count = positions.shape[0]
    assert hessian.shape == (3 * count, count, 3)
    step = 1e-5
    for atom, axis in ((0, 0), (1, 2)):
        shifted = []
        for sign in (1.0, -1.0):
            moved = np.array(positions, dtype=float)
            moved[atom, axis] += sign * step
            forces = engine(build_graph(moved, numbers), compute=("forces",)).forces
            shifted.append(forces)
        column = -(shifted[0] - shifted[1]) / (2 * step)
        # Column (atom, axis) of the Hessian is row (atom, axis) by symmetry.
        torch.testing.assert_close(
            hessian[3 * atom + axis], column, rtol=1e-6, atol=1e-8
        )


@fp64_only
def test_the_atoms_shares_add_up_to_the_structure_s_virials_and_stress():
    engine = build_engine()
    positions, numbers, cell = crystal()
    graph = build_graph(positions, numbers, cell=cell, pbc=(True, True, True))
    result = engine(
        graph,
        compute=("forces", "stress", "virials", "atomic_virials", "atomic_stresses"),
    )
    torch.testing.assert_close(
        result.extras["atomic_virials"].sum(dim=0),
        result.virials[0],
        rtol=1e-10,
        atol=1e-12,
    )
    torch.testing.assert_close(
        result.extras["atomic_stresses"].sum(dim=0),
        result.stress[0],
        rtol=1e-10,
        atol=1e-12,
    )
