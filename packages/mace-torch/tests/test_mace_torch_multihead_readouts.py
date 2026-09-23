"""One readout per head, sharing the backbone and nothing after it.

A multi-head model is several levels of theory fitted on one set of node
features. What makes that work is that each level of theory has its own
readout: with one readout shared between them, two heads pull the same weights
towards two different targets, and the only freedom left per head is an
isolated-atom energy and an affine map. A fine-tune that keeps a replay head
beside a new one is exactly that case.

The claims, in order: the layout helpers put each head's copy where the
docstring says; one head's readout is disjoint from another's, by gradient; a
head's state moves between heads and between models; and a two-head model
whose first head carries a single-head model's readout reproduces that model.
The last is the one that ties the layout, the per-atom selection and the gated
readout's mask together, since any of the three being wrong breaks it.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch
from conftest import assert_close, fp64_only
from mace_core.elements import AtomicNumberTable, ResolvedE0s
from mace_core.kernels.canonical import linear_bias_table, linear_weight_table
from mace_core.kernels.descriptors import LinearDescriptor
from mace_core.kernels.precision import PrecisionConfig
from mace_core.neighbors import get_neighborhood
from mace_core.observables import ObservableSpec
from mace_torch.backends.reference import ReferenceBackend
from mace_torch.models import EnergyOutputHead, MACEOutputs, ScaleShiftSpec
from mace_torch.models.heads import copy_heads, head_columns, per_head_irreps
from mace_torch.nn import MACEBackbone

ATOMIC_NUMBERS = [1, 8]
CUTOFF = 5.0
HIDDEN = "0e+1o"
HEADS = ("replay", "target")

ENERGY = ObservableSpec(name="energy", irreps="0e", per_atom=False, units="eV")
DIPOLE = ObservableSpec(name="dipole", irreps="1o", per_atom=False, units="e*A")


def precision_name():
    return str(torch.get_default_dtype()).removeprefix("torch.")


def energy_head(heads=HEADS):
    """Identical energies and scales on every head, so only the readout differs."""
    values = {name: {1: -13.6, 8: -2040.0} for name in heads}
    return EnergyOutputHead(
        ResolvedE0s(values),
        list(heads),
        AtomicNumberTable(ATOMIC_NUMBERS),
        ScaleShiftSpec("std", (1.3,) * len(heads), (0.2,) * len(heads)),
        PrecisionConfig(model=precision_name(), accumulate="float64"),
    )


def randomized(module, seed):
    generator = torch.Generator().manual_seed(seed)
    with torch.no_grad():
        for parameter in module.parameters():
            parameter.copy_(
                torch.randn(parameter.shape, generator=generator, dtype=parameter.dtype)
            )
    return module


def backbone(seed=0):
    return randomized(
        MACEBackbone(
            ReferenceBackend(),
            atomic_numbers=ATOMIC_NUMBERS,
            num_layers=2,
            num_features=4,
            lmax=1,
            hidden_irreps=HIDDEN,
            correlation=2,
            cutoff=CUTOFF,
            avg_num_neighbors=4.0,
            precision=precision_name(),
        ),
        seed,
    )


def outputs(heads=HEADS, observables=(ENERGY,), seed=1):
    return randomized(
        MACEOutputs(
            ReferenceBackend(),
            list(observables),
            hidden_irreps=HIDDEN,
            num_features=4,
            num_layers=2,
            energy_head=energy_head(heads) if ENERGY in observables else None,
            precision=precision_name(),
            num_heads=len(heads),
        ),
        seed,
    )


WATER = np.array([[0.0, 0.0, 0.0], [0.95, 0.0, 0.0], [-0.24, 0.93, 0.0]])


def graph(heads_of_structures=(0,), positions=WATER):
    """One water per entry, each assigned the given head."""
    count = len(heads_of_structures)
    blocks = [np.asarray(positions) + 20.0 * index for index in range(count)]
    everything = np.concatenate(blocks)
    neighborhood = get_neighborhood(everything, CUTOFF, (False, False, False), None)
    numbers = [8, 1, 1] * count
    index = {z: i for i, z in enumerate(ATOMIC_NUMBERS)}
    dtype = torch.get_default_dtype()
    return {
        "positions": torch.tensor(everything, dtype=dtype),
        "atomic_numbers": torch.tensor(numbers),
        "element_index": torch.tensor([index[z] for z in numbers]),
        "edge_index": torch.tensor(neighborhood.edge_index),
        "shifts": torch.tensor(neighborhood.shifts, dtype=dtype),
        "batch": torch.arange(count).repeat_interleave(3),
        "num_graphs": count,
        "head": torch.tensor(list(heads_of_structures)),
    }


# ---------------------------------------------------------------------------
# The layout
# ---------------------------------------------------------------------------


def test_one_head_leaves_the_declaration_as_written():
    """So a single-head model builds exactly the ops it built before."""
    assert per_head_irreps("0e", 1) == "0e"
    assert per_head_irreps("0e+2e", 1) == "0e+2e"


def test_several_heads_multiply_each_term():
    """The frozen tree's `len(heads)x0e`, and its generalization."""
    assert per_head_irreps("0e", 2) == "2x0e"
    assert per_head_irreps("0e+1o", 3) == "3x0e+3x1o"
    assert per_head_irreps("2x1o", 2) == "4x1o"


def test_each_heads_copy_is_where_the_docstring_says():
    """`2x0e+2x1o`: the scalars at 0 and 1, the vectors at 2:5 and 5:8."""
    columns = head_columns("0e+1o", 2)
    assert columns.tolist() == [[0, 2, 3, 4], [1, 5, 6, 7]]


def test_a_term_with_several_copies_keeps_them_together_per_head():
    """Head-major within a term: head 0's two vectors, then head 1's."""
    assert copy_heads("2x1o", 2) == [0, 0, 1, 1]
    assert head_columns("2x1o", 2).tolist() == [
        [0, 1, 2, 3, 4, 5],
        [6, 7, 8, 9, 10, 11],
    ]


def test_the_weight_table_is_the_reference_backends_plan():
    """The table in `mace_core` states the layout; the reference holds it.
    They are two descriptions of one ordering and have to agree."""
    from mace_torch.backends.reference.backend import _linear_plan

    descriptor = LinearDescriptor(irreps_in="3x0e+2x1o", irreps_out="2x0e+1x1o")
    *_, count, _ = _linear_plan(descriptor)
    table = linear_weight_table(descriptor.irreps_in, descriptor.irreps_out)
    assert sorted(table.values()) == list(range(count))
    # Output copies outermost: both of the first scalar output's weights come
    # before any weight of the second.
    assert table[(0, 0)] == 0 and table[(0, 2)] == 2 and table[(1, 0)] == 3


def test_only_an_even_scalar_output_carries_a_bias():
    """A constant added to anything else would pick out a direction."""
    assert linear_bias_table("2x0e+1o+0e+0o") == {0: 0, 1: 1, 3: 2}


# ---------------------------------------------------------------------------
# Shapes
# ---------------------------------------------------------------------------


def readout_count(module):
    return sum(
        parameter.numel()
        for name, parameter in module.named_parameters()
        if ".readouts." in f".{name}"
    )


def test_the_linear_readouts_grow_with_the_heads_and_the_gated_one_squares():
    """Per layer, the linear readout is one copy per head. The gated readout's
    first map is too, and its second is dense across heads, as in the frozen
    tree: its cross-head blocks multiply a masked zero."""
    single = outputs(heads=HEADS[:1])
    double = outputs(heads=HEADS)
    head = single.heads["energy"]
    linear = head.readouts[0].weight.numel()
    first = head.readouts[1].first.weight.numel()
    second = head.readouts[1].second.weight.numel()
    assert readout_count(double) == 2 * linear + 2 * first + 4 * second
    assert readout_count(single) == linear + first + second


# ---------------------------------------------------------------------------
# Each head has its own readout
# ---------------------------------------------------------------------------


def gradient_support(head_index: int) -> set[str]:
    """Which readout weights a structure on this head sends any gradient to."""
    model_backbone = backbone()
    model_outputs = outputs()
    g = graph((head_index,))
    energy = model_outputs(g, model_backbone(g)).total_energy.sum()
    energy.backward()
    touched = set()
    for name, parameter in model_outputs.named_parameters():
        if parameter.grad is None:
            continue
        for position in torch.nonzero(parameter.grad).flatten().tolist():
            touched.add(f"{name}[{position}]")
    return touched


@fp64_only
def test_the_two_heads_share_no_readout_weight():
    """The defining property. A weight either head can move is a weight a
    replay head and a new head would fight over."""
    first, second = gradient_support(0), gradient_support(1)
    assert first and second
    assert not first & second


@fp64_only
def test_each_head_reaches_as_many_weights_as_a_single_head_model_has():
    """No head is reduced, and none takes more than its share."""
    single = readout_count(outputs(heads=HEADS[:1]))
    assert len(gradient_support(0)) == single
    assert len(gradient_support(1)) == single


# ---------------------------------------------------------------------------
# A head's state
# ---------------------------------------------------------------------------


def test_a_state_round_trips():
    model = outputs()
    head = model.heads["energy"]
    before = {name: p.detach().clone() for name, p in model.named_parameters()}
    head.load_head_state(1, head.head_state(1))
    for name, parameter in model.named_parameters():
        assert torch.equal(parameter, before[name]), name


def test_writing_one_head_leaves_the_other_alone():
    model = outputs()
    head = model.heads["energy"]
    untouched = head.head_state(0)
    donor = outputs(seed=7).heads["energy"].head_state(1)
    head.load_head_state(1, donor)
    for layer_before, layer_after in zip(untouched, head.head_state(0), strict=True):
        for op in layer_before:
            assert torch.equal(layer_before[op]["weight"], layer_after[op]["weight"])
    for layer_donor, layer_after in zip(donor, head.head_state(1), strict=True):
        for op in layer_donor:
            assert torch.equal(layer_donor[op]["weight"], layer_after[op]["weight"])


@fp64_only
def test_copying_a_head_makes_the_two_heads_agree():
    """With identical energies and scales on both heads, a copied readout is
    the only thing that could make their energies equal."""
    model_backbone = backbone()
    model_outputs = outputs()
    head = model_outputs.heads["energy"]
    g = graph((0, 1))
    before = model_outputs(g, model_backbone(g)).total_energy
    assert abs(float((before[0] - before[1]).detach())) > 1e-3
    head.load_head_state(1, head.head_state(0))
    after = model_outputs(g, model_backbone(g)).total_energy
    assert_close(after[1], after[0].detach().numpy(), "a copied readout")


@fp64_only
def test_a_single_head_readout_loaded_into_head_one_reproduces_that_model():
    """The end-to-end check of the layout, the per-atom selection and the
    mask: a single-head model's readout, loaded into one head of a two-head
    model on the same backbone, gives the single-head model's energies."""
    model_backbone = backbone()
    single = outputs(heads=HEADS[:1], seed=3)
    double = outputs(heads=HEADS, seed=4)
    double.heads["energy"].load_head_state(1, single.heads["energy"].head_state(0))
    g_single = graph((0, 0))
    g_double = graph((1, 1))
    expected = single(g_single, model_backbone(g_single)).total_energy
    got = double(g_double, model_backbone(g_double)).total_energy
    assert_close(got, expected.detach().numpy(), "one head of a two-head model")


def test_a_state_from_a_different_model_is_refused():
    model = outputs().heads["energy"]
    shorter = model.head_state(0)[:1]
    with pytest.raises(ValueError, match="different model"):
        model.load_head_state(0, shorter)


def test_a_head_that_does_not_exist_is_refused():
    with pytest.raises(IndexError, match="does not exist"):
        outputs().heads["energy"].head_state(2)


# ---------------------------------------------------------------------------
# Non-scalar observables, and refusals
# ---------------------------------------------------------------------------


@fp64_only
def test_a_multi_head_dipole_takes_its_own_heads_copy():
    """A linear readout of a vector, for two heads: the copies differ, and a
    structure's dipole is the copy of its own head."""
    model_backbone = backbone()
    model_outputs = outputs(observables=(DIPOLE,))
    g = graph((0, 1))
    dipoles = model_outputs(g, model_backbone(g)).dipole
    assert dipoles.shape == (2, 3)
    assert float((dipoles[0] - dipoles[1]).abs().max()) > 1e-3


def test_an_energy_head_for_a_different_number_of_heads_is_refused():
    with pytest.raises(ValueError, match="rows do not line up"):
        MACEOutputs(
            ReferenceBackend(),
            [ENERGY],
            hidden_irreps=HIDDEN,
            num_features=4,
            num_layers=2,
            energy_head=energy_head(HEADS),
            num_heads=3,
        )


def test_several_heads_without_a_per_atom_head_index_is_refused():
    head = outputs().heads["energy"]
    layers = [torch.zeros(3, 4, 4), torch.zeros(3, 4, 4)]
    with pytest.raises(ValueError, match="which head's readout"):
        head.per_layer(layers)
