"""The coupling-tree label that names each path in the canonical layout.

Path order and per-path normalization do not pin the layout on their own. The
enumerated paths are linearly dependent, so which of them survives the
reduction is a free choice, and two implementations resolving it differently
span the same space with vectors no reordering relates. The label is what makes
that choice recoverable from the file: a path is named by the tree that built
it, not by where it landed.
"""

import numpy as np
import pytest
from mace_core.clebsch_gordan.irreps import Irrep, Irreps
from mace_core.clebsch_gordan.real_basis import wigner_3j_real
from mace_core.clebsch_gordan.reduced_basis import (
    CouplingTree,
    full_path_labels,
    full_symmetric_tensor_product_basis,
    path_labels,
    reduced_symmetric_tensor_product_basis,
    symmetrize,
)

ATOL = 1e-12

GRID = [
    ("0e+1o", 3, "0e+1o"),
    ("0e+1o+2e", 3, "0e+1o"),
    ("0e+1o+2e+3o", 3, "0e+1o+2e"),
    ("0e+1o+2e", 4, "0e+1o"),
    ("2x0e+1o", 2, "0e+1o"),
]


@pytest.mark.parametrize(("irreps_in", "correlation", "keep_ir"), GRID)
def test_every_path_carries_exactly_one_label(irreps_in, correlation, keep_ir):
    basis = reduced_symmetric_tensor_product_basis(irreps_in, correlation, keep_ir)
    labels = path_labels(irreps_in, correlation, keep_ir)
    assert labels.keys() == basis.keys()
    for target, array in basis.items():
        assert len(labels[target]) == array.shape[0]


@pytest.mark.parametrize(("irreps_in", "correlation", "keep_ir"), GRID)
def test_the_labels_are_unique_within_an_output_irrep(irreps_in, correlation, keep_ir):
    for target, trees in path_labels(irreps_in, correlation, keep_ir).items():
        assert len(set(trees)) == len(trees), target


@pytest.mark.parametrize(("irreps_in", "correlation", "keep_ir"), GRID)
def test_a_label_reports_its_own_output_irrep_and_body_order(
    irreps_in, correlation, keep_ir
):
    for target, trees in path_labels(irreps_in, correlation, keep_ir).items():
        for tree in trees:
            assert str(tree.target) == target
            assert tree.correlation == correlation
            assert len(tree.factors) == correlation


@pytest.mark.parametrize(("irreps_in", "correlation", "keep_ir"), GRID)
def test_each_step_couples_a_real_slice_into_a_reachable_irrep(
    irreps_in, correlation, keep_ir
):
    """The chain of intermediates has to obey the selection rules end to end."""
    parsed = Irreps.parse(irreps_in)
    slice_irreps = [ir for _, ir in parsed.slices()]
    for trees in path_labels(irreps_in, correlation, keep_ir).values():
        for tree in trees:
            first_index, running = tree.steps[0]
            assert slice_irreps[first_index] == running
            for index, intermediate in tree.steps[1:]:
                assert 0 <= index < len(slice_irreps)
                assert intermediate in set(running.couple(slice_irreps[index]))
                running = intermediate


@pytest.mark.parametrize(("irreps_in", "correlation", "keep_ir"), GRID)
def test_a_label_survives_the_round_trip_through_its_written_form(
    irreps_in, correlation, keep_ir
):
    for trees in path_labels(irreps_in, correlation, keep_ir).values():
        for tree in trees:
            assert CouplingTree.parse(str(tree)) == tree


def test_the_written_form_is_the_documented_one():
    tree = CouplingTree(((0, Irrep(0, 1)), (1, Irrep(1, -1)), (2, Irrep(2, 1))))
    assert str(tree) == "0:0e|1:1o|2:2e"
    assert tree.factors == (0, 1, 2)
    assert tree.target == Irrep(2, 1)


@pytest.mark.parametrize("text", ["0e", "0:0e|1o", "x:0e", ""])
def test_a_malformed_label_names_the_step_it_choked_on(text):
    with pytest.raises(ValueError, match="coupling step"):
        CouplingTree.parse(text)


@pytest.mark.parametrize(("irreps_in", "correlation", "keep_ir"), GRID)
def test_the_reduced_labels_are_a_subsequence_of_the_unreduced_ones(
    irreps_in, correlation, keep_ir
):
    """The reduction drops paths; it never renames or reorders the survivors."""
    reduced = path_labels(irreps_in, correlation, keep_ir)
    full = full_path_labels(irreps_in, correlation, keep_ir)
    for target, trees in reduced.items():
        remaining = iter(full[target])
        assert all(tree in remaining for tree in trees), target


@pytest.mark.parametrize(("irreps_in", "correlation", "keep_ir"), GRID)
def test_the_unreduced_basis_carries_one_label_per_path_too(
    irreps_in, correlation, keep_ir
):
    basis = full_symmetric_tensor_product_basis(irreps_in, correlation, keep_ir)
    labels = full_path_labels(irreps_in, correlation, keep_ir)
    for target, array in basis.items():
        assert len(labels[target]) == array.shape[0]


def test_asking_for_one_irrep_gives_the_same_labels_as_asking_for_several():
    """A label is a property of the path, not of what else was requested."""
    alone = path_labels("0e+1o+2e+3o", 3, "2e")
    together = path_labels("0e+1o+2e+3o", 3, "0e+1o+2e")
    assert alone["2e"] == together["2e"]


def _rebuild(tree: CouplingTree, irreps_in: Irreps) -> np.ndarray:
    """Build a path's tensor from its label alone, with no index into the basis.

    Reimplemented here rather than imported: the point of the test is that the
    label carries the whole construction, including the normalization, so a
    reader of the file can regenerate the tensor without the generator.
    """
    pieces = list(irreps_in.slices())
    width = irreps_in.dimension
    first_index, first_irrep = tree.steps[0]
    path = np.zeros((first_irrep.dimension, width))
    path[:, pieces[first_index][0]] = np.eye(first_irrep.dimension)
    running = first_irrep
    for index, intermediate in tree.steps[1:]:
        piece, slice_irrep = pieces[index]
        coupling = wigner_3j_real(
            intermediate.degree, running.degree, slice_irrep.degree
        )
        grown = np.zeros((intermediate.dimension, *path.shape[1:], width))
        grown[..., piece] = np.einsum("oml,m...->o...l", coupling, path)
        path, running = grown, intermediate
    path = symmetrize(path, tree.correlation)
    path = path / np.linalg.norm(path)
    flat = path.reshape(-1)
    leading = np.flatnonzero(np.abs(flat) > 1e-9)
    if leading.size and flat[leading[0]] < 0:
        path = -path
    return path


@pytest.mark.parametrize(("irreps_in", "correlation", "keep_ir"), GRID)
def test_the_label_determines_the_tensor(irreps_in, correlation, keep_ir):
    """The decisive property: the name is complete, so position carries nothing.

    A backend that reads a checkpoint holds labels and weights. If the label
    regenerates the basis vector exactly, the backend can place the weight
    itself, whatever order it enumerates in.
    """
    parsed = Irreps.parse(irreps_in)
    basis = reduced_symmetric_tensor_product_basis(irreps_in, correlation, keep_ir)
    labels = path_labels(irreps_in, correlation, keep_ir)
    for target, trees in labels.items():
        for position, tree in enumerate(trees):
            np.testing.assert_allclose(
                _rebuild(tree, parsed), basis[target][position], atol=ATOL
            )
