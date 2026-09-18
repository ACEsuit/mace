"""The output layer: typed results, the energy head, and the symmetries.

Three groups of claim. That a declared observable becomes a working head with
no file in the package edited, which is the property the whole design exists
for. That the energy head's three documented traps stay shut: the fp64 table,
the two separate reductions, and the per-quantity dtype split. And that every
field of the result transforms the way its declaration says it does.
"""

from __future__ import annotations

import ast
import inspect

import numpy as np
import pytest
import torch
from conftest import assert_close, fp64_only
from mace_core.clebsch_gordan.real_basis import wigner_d_real
from mace_core.elements import AtomicNumberTable, ResolvedE0s
from mace_core.kernels.precision import PrecisionConfig
from mace_core.neighbors import get_neighborhood
from mace_core.observables import ObservableSpec
from mace_torch.backends.reference import ReferenceBackend
from mace_torch.kernels import segment_sum
from mace_torch.models import EnergyOutputHead, MACEOutputs, ScaleShiftSpec
from mace_torch.nn import MACEBackbone

ATOMIC_NUMBERS = [1, 8]
CUTOFF = 5.0
HIDDEN = "0e+1o+2e"
E0S = ResolvedE0s({"default": {1: -13.6, 8: -2040.0}})


def spec(name, irreps, per_atom=False, units="eV"):
    return ObservableSpec(
        name=name,
        irreps=irreps,
        per_atom=per_atom,
        units=units,
        normalization="none",
    )


ENERGY = spec("energy", "0e")
DIPOLE = spec("dipole", "1o", units="eV/A")
POLARIZABILITY = spec("polarizability", "0e+2e", units="A^3")


def precision_name():
    return str(torch.get_default_dtype()).removeprefix("torch.")


def energy_head(scale=1.0, shift=0.0, zbl_in_scale_shift=True, **kwargs):
    return EnergyOutputHead(
        E0S,
        ["default"],
        AtomicNumberTable(ATOMIC_NUMBERS),
        ScaleShiftSpec("std", (scale,), (shift,)),
        PrecisionConfig(model=precision_name(), accumulate="float64"),
        zbl_in_scale_shift=zbl_in_scale_shift,
        **kwargs,
    )


def randomize(module, seed=0):
    """Give every weight a value before anything is asserted about it.

    The reference backend starts its contraction weights at zero, and a
    symmetry test on an output that is zero passes whatever the code does. The
    scale is unit normal: the chain is multiplicative over two layers and a
    readout, so a timid scale lands the outputs at 1e-7, where an absolute
    tolerance of 1e-12 is a very weak claim. :func:`assert_nontrivial` is what
    keeps this honest if the scale is ever changed again.
    """
    generator = torch.Generator().manual_seed(seed)
    with torch.no_grad():
        for parameter in module.parameters():
            parameter.copy_(
                torch.randn(parameter.shape, generator=generator, dtype=parameter.dtype)
            )
    return module


def assert_nontrivial(value, what: str, floor: float = 1e-3) -> None:
    """A symmetry claim about a value near zero is not a claim.

    Every equivariance test below states this first, so a change that quietly
    drives an output to zero turns the suite red instead of green.
    """
    largest = float(torch.as_tensor(value).detach().abs().max())
    assert largest > floor, (
        f"{what} is {largest:.3e}, which is too close to zero for a symmetry "
        f"test to mean anything. The weights or the geometry need changing."
    )


def make_stack(observables=(ENERGY, DIPOLE, POLARIZABILITY), seed=0, **head_kwargs):
    torch.manual_seed(seed)
    backend = ReferenceBackend()
    backbone = MACEBackbone(
        backend,
        atomic_numbers=ATOMIC_NUMBERS,
        num_layers=2,
        num_features=4,
        lmax=2,
        hidden_irreps=HIDDEN,
        correlation=2,
        cutoff=CUTOFF,
        avg_num_neighbors=6.0,
        precision=precision_name(),
    )
    head = (
        energy_head(**head_kwargs)
        if any(s.name == "energy" for s in observables)
        else None
    )
    outputs = MACEOutputs(
        backend,
        list(observables),
        hidden_irreps=HIDDEN,
        num_features=4,
        num_layers=2,
        energy_head=head,
        precision=precision_name(),
    )
    return randomize(backbone, seed), randomize(outputs, seed + 1)


def water():
    return (
        np.array([[0.0, 0.0, 0.0], [0.95, 0.0, 0.0], [-0.24, 0.93, 0.0]]),
        [8, 1, 1],
    )


def make_graph(positions, numbers):
    positions = np.asarray(positions, dtype=float)
    neighborhood = get_neighborhood(positions, CUTOFF, (False, False, False), None)
    dtype = torch.get_default_dtype()
    index = {z: i for i, z in enumerate(ATOMIC_NUMBERS)}
    return {
        "positions": torch.tensor(positions, dtype=dtype),
        "atomic_numbers": torch.tensor(list(numbers)),
        "element_index": torch.tensor([index[z] for z in numbers]),
        "edge_index": torch.tensor(neighborhood.edge_index),
        "shifts": torch.tensor(neighborhood.shifts, dtype=dtype),
        "batch": torch.zeros(len(numbers), dtype=torch.long),
        "num_graphs": 1,
        "head": torch.zeros(1, dtype=torch.long),
    }


def run(positions=None, numbers=None, **kwargs):
    if positions is None:
        positions, numbers = water()
    backbone, outputs = make_stack(**kwargs)
    graph = make_graph(positions, numbers)
    return outputs(graph, backbone(graph))


# ---------------------------------------------------------------------------
# The typed result, and heads from declarations alone
# ---------------------------------------------------------------------------


@fp64_only
def test_the_result_is_the_typed_dataclass_with_fields_by_name():
    result = run()
    assert result.total_energy.shape == (1,)
    assert result.node_energies.shape == (3,)
    assert result.dipole.shape == (1, 3)
    assert result.get("energy") is result.total_energy
    assert "polarizability" in result.extras


@fp64_only
def test_a_new_observable_needs_only_a_declaration():
    """The property the design exists for, asserted rather than asserted about.

    A rank-2 per-atom observable nobody wrote code for gets a head, the right
    width, and one row per atom.
    """
    quadrupole = spec("quadrupole", "2e", per_atom=True, units="e*A^2")
    result = run(observables=(ENERGY, quadrupole))
    assert result.extras["quadrupole"].shape == (3, 5)


@fp64_only
def test_a_per_atom_observable_is_not_reduced_and_a_graph_one_is():
    per_atom = run(observables=(spec("charge", "0e", per_atom=True),))
    per_graph = run(observables=(spec("charge", "0e", per_atom=False),))
    assert per_atom.extras["charge"].shape == (3, 1)
    assert per_graph.extras["charge"].shape == (1, 1)
    assert_close(
        per_graph.extras["charge"],
        per_atom.extras["charge"].sum(0, keepdim=True).detach().numpy(),
        "the graph-level reduction",
    )


@fp64_only
def test_an_observable_with_no_data_behind_it_is_an_error_naming_the_key():
    _, outputs = make_stack(observables=(ENERGY, DIPOLE))
    with pytest.raises(KeyError, match="dipole"):
        outputs.check_data(["energy", "forces"])
    outputs.check_data(["energy", "dipole", "forces"])


@fp64_only
def test_an_energy_declaration_without_a_head_is_refused():
    backend = ReferenceBackend()
    with pytest.raises(ValueError, match="isolated-atom"):
        MACEOutputs(backend, [ENERGY], HIDDEN, 4, 2, energy_head=None)
    with pytest.raises(ValueError, match="not declared"):
        MACEOutputs(backend, [DIPOLE], HIDDEN, 4, 2, energy_head=energy_head())


@fp64_only
def test_an_observable_the_features_cannot_carry_is_refused():
    """Silently zero is the failure mode this replaces.

    A rank-2 observable read out of scalar-and-vector features gives a column
    of zeros and a loss term that can never go down, with nothing raised
    anywhere.
    """
    with pytest.raises(ValueError, match="2e"):
        MACEOutputs(ReferenceBackend(), [POLARIZABILITY], "0e+1o", 4, 2)


@fp64_only
def test_a_repeated_declaration_is_refused():
    with pytest.raises(ValueError, match="dipole"):
        MACEOutputs(ReferenceBackend(), [DIPOLE, DIPOLE], HIDDEN, 4, 2)


# ---------------------------------------------------------------------------
# The energy head
# ---------------------------------------------------------------------------


def test_the_e0_table_is_float64_whatever_the_model_computes_in():
    """Registering it at the build-time default rounds it at construction.

    An isolated-atom energy is tens or thousands of eV, so fp32 loses the last
    few decimals of every atom's contribution before training starts, and no
    later cast gets them back.
    """
    head = energy_head()
    assert head.e0_table.dtype == torch.float64
    assert not head.e0_table.requires_grad
    assert "e0_table" not in dict(head.named_parameters())
    assert_close(
        head.e0_table.numpy(), np.array([[-13.6, -2040.0]]), "the resolved table"
    )


def test_the_e0_gather_is_the_one_hot_matmul():
    """Same numbers, without building the one-hot. Exact, not close."""
    head = energy_head()
    element = torch.tensor([1, 0, 0])
    one_hot = torch.zeros(3, 2, dtype=torch.float64)
    one_hot[torch.arange(3), element] = 1.0
    gathered = head.e0_table[torch.zeros(3, dtype=torch.long), element]
    assert torch.equal(gathered, one_hot @ head.e0_table[0])


@fp64_only
def test_the_two_reductions_stay_separate():
    """Fusing them loses the interaction energy inside the E0 sum.

    The fused form sums one array whose entries differ by five orders of
    magnitude; the separate form sums two arrays that are each well scaled and
    adds the results once. Asserted through the real reduction primitive, not
    through `.sum()`, because `.sum()` pairs terms and hides exactly the effect
    being guarded against.
    """
    count = 4000
    interaction = torch.full((count,), 0.001, dtype=torch.float32)
    e0 = torch.full((count,), -2040.0, dtype=torch.float32)
    batch = torch.zeros(count, dtype=torch.long)

    separate = segment_sum(interaction, batch, 1) + segment_sum(e0, batch, 1)
    fused = segment_sum(interaction + e0, batch, 1)
    exact = count * (0.001 - 2040.0)

    assert abs(float(separate) - exact) < abs(float(fused) - exact), (
        f"the fused reduction was not worse here, so this guard is not "
        f"measuring anything: separate {float(separate)}, fused {float(fused)}, "
        f"exact {exact}"
    )
    assert abs(float(fused) - exact) > 0.1, (
        f"the fused reduction lost only {abs(float(fused) - exact)} eV, so the "
        f"guard needs a harder case to stay meaningful"
    )


def test_the_head_uses_two_reductions_and_not_one():
    """The structure itself, read off the source.

    The measurement above shows the fused form is worse. This shows the head
    does not use it, which is the part a refactor could quietly undo.
    """
    from mace_torch.models import energy

    source = inspect.getsource(energy.EnergyOutputHead.forward)
    assert source.count("segment_sum(") == 2, (
        "the energy head should call the reduction exactly twice, once for the "
        "site energies and once for the isolated-atom energies"
    )


@fp64_only
def test_the_dtype_split_is_per_quantity():
    """Total in the model's dtype, per-atom in the accumulation dtype."""
    head = EnergyOutputHead(
        E0S,
        ["default"],
        AtomicNumberTable(ATOMIC_NUMBERS),
        ScaleShiftSpec("std", (1.0,), (0.0,)),
        PrecisionConfig(model="float32", accumulate="float64"),
    )
    terms = head(
        [torch.tensor([0.1, 0.2, 0.3], dtype=torch.float32)],
        None,
        torch.tensor([1, 0, 0]),
        torch.tensor([0]),
        torch.zeros(3, dtype=torch.long),
        1,
    )
    assert terms.total_energy.dtype == torch.float32
    assert terms.node_energy.dtype == torch.float64
    assert terms.interaction_energy.dtype == torch.float32


@fp64_only
def test_the_accumulation_type_is_a_floor_the_device_may_not_meet():
    """A device without float64 degrades, and says so when the model is built.

    This is what the frozen tree already does on MPS, where its fp64 helper is
    the identity. The difference is that there it is discovered as a dtype
    error later.
    """
    plain = energy_head()
    assert plain.accumulate == "float64"
    assert plain.build_report() is None

    degraded = EnergyOutputHead(
        E0S,
        ["default"],
        AtomicNumberTable(ATOMIC_NUMBERS),
        ScaleShiftSpec("std", (1.0,), (0.0,)),
        PrecisionConfig(model="float32", accumulate="float64"),
        supports_float64=False,
    )
    assert degraded.accumulate == "float32"
    assert "no float64" in (degraded.build_report() or "")


@fp64_only
def test_the_pair_repulsion_placement_changes_the_energy():
    """Two model classes in the frozen tree, one field here.

    Inside the scaled sum the repulsion picks up the scale and the shift;
    outside it does not. The two are not a preference, they are different
    trained models.
    """
    repulsion = torch.tensor([1.0, 2.0, 3.0])
    arguments = (
        [torch.tensor([0.1, 0.2, 0.3])],
        repulsion,
        torch.tensor([1, 0, 0]),
        torch.tensor([0]),
        torch.zeros(3, dtype=torch.long),
        1,
    )
    inside = energy_head(scale=3.0, shift=0.5, zbl_in_scale_shift=True)(*arguments)
    outside = energy_head(scale=3.0, shift=0.5, zbl_in_scale_shift=False)(*arguments)

    e0 = -13.6 - 13.6 - 2040.0
    site = torch.tensor([0.1, 0.2, 0.3])
    assert_close(
        inside.total_energy,
        np.array([float((3.0 * (site + repulsion) + 0.5).sum()) + e0]),
        "the scaled placement",
    )
    assert_close(
        outside.total_energy,
        np.array([float((3.0 * site + 0.5 + repulsion).sum()) + e0]),
        "the unscaled placement",
    )


@fp64_only
def test_the_two_energies_have_the_same_derivative_against_positions():
    """The E0 branch has no path back to the positions, so the engine above
    can differentiate the total uniformly and pay nothing for it."""
    positions, numbers = water()
    backbone, outputs = make_stack(observables=(ENERGY,))
    graph = make_graph(positions, numbers)
    graph["positions"] = graph["positions"].requires_grad_(True)
    result = outputs(graph, backbone(graph))

    from_total = torch.autograd.grad(
        result.total_energy.sum(), graph["positions"], retain_graph=True
    )[0]
    from_interaction = torch.autograd.grad(
        result.extras["interaction_energy"].sum(), graph["positions"]
    )[0]
    assert torch.equal(from_total, from_interaction)


@fp64_only
def test_the_output_layer_never_hardcodes_a_cast_to_double():
    """The dtype rule is declarative. A literal `.double()` would override it
    on a device that cannot honour it."""
    from mace_torch.models import energy, heads, outputs

    for module in (energy, heads, outputs):
        tree = ast.parse(inspect.getsource(module))
        for node in ast.walk(tree):
            if isinstance(node, ast.Attribute) and node.attr == "double":
                pytest.fail(f"{module.__name__} casts to float64 unconditionally")


# ---------------------------------------------------------------------------
# Symmetries of every field
# ---------------------------------------------------------------------------


def random_rotation(seed):
    generator = np.random.default_rng(seed)
    matrix, _ = np.linalg.qr(generator.normal(size=(3, 3)))
    if np.linalg.det(matrix) < 0:
        matrix[:, 0] *= -1
    return matrix


def test_every_field_transforms_as_its_declaration_says():
    """A scalar does not move, a vector rotates, a rank-2 mixes within itself.

    One test rather than three, because what is being checked is that the
    declaration alone decides it.
    """
    positions, numbers = water()
    rotation = random_rotation(5)
    plain = run(positions, numbers)
    turned = run(positions @ rotation.T, numbers)

    assert_nontrivial(plain.extras["interaction_energy"], "the interaction energy")
    assert_nontrivial(plain.dipole, "the dipole")
    assert_nontrivial(plain.extras["polarizability"][:, 1:], "the polarizability's 2e")

    assert_close(turned.total_energy, plain.total_energy.detach().numpy(), "the energy")
    assert_close(
        turned.node_energies, plain.node_energies.detach().numpy(), "the site energies"
    )
    wigner = torch.tensor(wigner_d_real(1, rotation), dtype=plain.dipole.dtype)
    assert_close(turned.dipole, (plain.dipole @ wigner).detach().numpy(), "the dipole")

    before = plain.extras["polarizability"]
    after = turned.extras["polarizability"]
    rank_two = torch.tensor(wigner_d_real(2, rotation), dtype=before.dtype)
    expected = torch.cat([before[:, :1], before[:, 1:] @ rank_two], dim=-1)
    assert_close(after, expected.detach().numpy(), "the polarizability")


def test_the_energy_is_invariant_under_inversion_and_the_dipole_flips():
    positions, numbers = water()
    plain = run(positions, numbers)
    inverted = run(-positions, numbers)

    assert_nontrivial(plain.dipole, "the dipole")
    assert_close(
        inverted.total_energy, plain.total_energy.detach().numpy(), "the energy"
    )
    assert_close(
        inverted.dipole, -plain.dipole.detach().numpy(), "the dipole under inversion"
    )


def test_every_field_is_translation_invariant():
    positions, numbers = water()
    plain = run(positions, numbers)
    shifted = run(positions + np.array([12.0, -3.0, 7.5]), numbers)

    assert_nontrivial(plain.dipole, "the dipole")
    for name in plain.names():
        assert_close(
            shifted.get(name), plain.get(name).detach().numpy(), f"{name} under a shift"
        )


# ---------------------------------------------------------------------------
# Graph-level input features
# ---------------------------------------------------------------------------


@fp64_only
def test_graph_features_reach_the_scalars_and_leave_the_rest_alone():
    """Adding to a higher irrep would break equivariance: the added value does
    not rotate, and the thing it is added to does."""
    from mace_torch.nn import FeatureSpec, GraphFeatureEmbedding

    embedding = randomize(
        GraphFeatureEmbedding(
            [FeatureSpec("total_charge", "continuous", embedding_dim=8)],
            num_features=4,
            num_scalars=1,
        )
    )
    graph = make_graph(*water())
    graph["total_charge"] = torch.zeros(1, dtype=torch.float64)
    features = torch.randn(3, 4, 9, dtype=torch.float64)

    updated = embedding(graph, features)
    assert torch.equal(updated[..., 1:], features[..., 1:])
    assert not torch.equal(updated[..., :1], features[..., :1])


@fp64_only
def test_a_per_structure_feature_reaches_every_atom_of_that_structure():
    from mace_torch.nn import FeatureSpec, GraphFeatureEmbedding

    embedding = randomize(
        GraphFeatureEmbedding(
            [FeatureSpec("total_spin", "categorical", embedding_dim=4, num_classes=3)],
            num_features=2,
            num_scalars=1,
        )
    )
    graph = make_graph(*water())
    graph["batch"] = torch.tensor([0, 0, 1])
    graph["total_spin"] = torch.tensor([1, 2])
    features = torch.zeros(3, 2, 1, dtype=torch.float64)

    updated = embedding(graph, features)
    assert torch.equal(updated[0], updated[1]), (
        "two atoms of the same structure got different values from a "
        "per-structure feature"
    )
    assert not torch.equal(updated[0], updated[2])


@fp64_only
def test_a_declared_feature_with_no_key_is_an_error_naming_it():
    from mace_torch.nn import FeatureSpec, GraphFeatureEmbedding

    embedding = GraphFeatureEmbedding(
        [FeatureSpec("elec_temp", "continuous", embedding_dim=4)],
        num_features=2,
        num_scalars=1,
    )
    with pytest.raises(KeyError, match="elec_temp"):
        embedding(make_graph(*water()), torch.zeros(3, 2, 1, dtype=torch.float64))


@fp64_only
def test_a_categorical_feature_without_classes_is_refused():
    from mace_torch.nn import FeatureSpec

    with pytest.raises(ValueError, match="num_classes"):
        FeatureSpec("total_spin", "categorical", embedding_dim=4)


@fp64_only
def test_an_embedding_with_nothing_declared_is_refused():
    from mace_torch.nn import GraphFeatureEmbedding

    with pytest.raises(ValueError, match="no features"):
        GraphFeatureEmbedding([], num_features=2, num_scalars=1)
