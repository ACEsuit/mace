"""The energy head: the E0 table, the two sums, the dtypes, the pair
repulsion's placement, and what it saves. Its numbers against the frozen tree
are in tests/parity/test_energy_head_parity.py."""

import inspect
import warnings

import pytest
import torch
from conftest import PRECISIONS, assert_close, fp32_only, fp64_only, precision
from mace_core.elements import ResolvedE0s
from mace_torch.kernels import canonical_state, load_canonical_state, segment_sum
from mace_torch.models import EnergyOutputHead, energy

HEADS = ("pbe", "mp2")
ATOMIC_NUMBERS = (1, 8)
#: Neither is a float32 number, so a table held at float32 would show it.
E0S = {"pbe": {1: -13.6, 8: -2041.3}, "mp2": {1: -13.2, 8: -2040.1}}
RESOLVED = ResolvedE0s(values=E0S, provenance={"pbe": "average", "mp2": "table"})
SCALE, SHIFT = (1.7, 0.6), (0.3, -0.4)


def _head(
    dtype,
    *,
    zbl_in_scale_shift=True,
    precision=None,
    node_energy_dtype="float64",
    device="cpu",
    scale=SCALE,
    shift=SHIFT,
):
    return EnergyOutputHead(
        RESOLVED,
        HEADS,
        ATOMIC_NUMBERS,
        scale,
        shift,
        zbl_in_scale_shift=zbl_in_scale_shift,
        precision=precision or PRECISIONS[dtype],
        node_energy_dtype=node_energy_dtype,
        device=device,
    )


def _batch(dtype, num_nodes=7, num_layers=2, seed=0):
    generator = torch.Generator().manual_seed(seed)
    layers = [
        torch.randn(num_nodes, generator=generator, dtype=dtype)
        for _ in range(num_layers)
    ]
    element_index = torch.randint(
        0, len(ATOMIC_NUMBERS), (num_nodes,), generator=generator
    )
    batch = torch.tensor([0, 0, 0, 1, 1, 2, 2])[:num_nodes]
    node_head = torch.tensor([0, 1, 1])[batch]
    return layers, element_index, node_head, batch


def test_the_table_is_float64_and_not_trained_whatever_the_model_computes_in(dtype):
    head = _head(dtype)
    assert head.e0_table.dtype == torch.float64
    assert not head.e0_table.requires_grad
    assert not list(head.parameters())
    assert head.e0_table.tolist() == RESOLVED.to_array(HEADS, ATOMIC_NUMBERS)
    assert head.scale.dtype == head.shift.dtype == dtype


def test_each_structure_reads_its_heads_row(dtype):
    """Each atom's energy is its head's E0 for its element plus its head's
    scale times its site energies plus its head's shift."""
    head = _head(dtype)
    layers, element_index, node_head, batch = _batch(dtype)
    terms = head(layers, None, element_index, node_head, batch, 3)
    e0 = torch.tensor(RESOLVED.to_array(HEADS, ATOMIC_NUMBERS))[
        node_head, element_index
    ]
    scale = torch.tensor(SCALE)[node_head]
    shift = torch.tensor(SHIFT)[node_head]
    expected = e0 + scale * (layers[0] + layers[1]).double() + shift
    assert_close(terms.node_energy, expected, "node energies")


def test_the_total_is_two_sums_and_one_add(dtype):
    """Summed together, the isolated-atom energies swallow the digits of the
    interaction energies. The guard is on the reduction the head really uses:
    the total is the sum of the E0s plus the sum of the interaction energies,
    bit for bit, and at ten thousand atoms in float32 the fused sum differs
    from it by more than an eV."""
    num_nodes = 10_000
    head = _head(dtype)
    generator = torch.Generator().manual_seed(1)
    layers = [1e-3 * torch.randn(num_nodes, generator=generator, dtype=dtype)]
    element_index = torch.ones(num_nodes, dtype=torch.int64)
    batch = torch.zeros(num_nodes, dtype=torch.int64)
    node_head = torch.zeros(num_nodes, dtype=torch.int64)
    terms = head(layers, None, element_index, node_head, batch, 1)

    node_e0 = head.e0_table[0, element_index].to(dtype)
    node_interaction = head.scale[0] * layers[0] + head.shift[0]
    e0_sum = segment_sum(node_e0, batch, 1)
    interaction_sum = segment_sum(node_interaction, batch, 1)
    assert torch.equal(terms.interaction_energy, interaction_sum)
    assert torch.equal(terms.total_energy, e0_sum + interaction_sum)
    if dtype == torch.float32:
        fused = segment_sum(node_e0 + node_interaction, batch, 1)
        assert (fused - terms.total_energy).abs().item() > 1.0


@pytest.mark.parametrize("accumulate", ["model", "float64"])
def test_the_total_is_in_the_model_dtype_and_the_atoms_in_the_accumulation_dtype(
    dtype, accumulate
):
    """`ScaleShiftMACE` accumulates the per-atom energies in float64, `MACE`
    in its own dtype; both sum the totals in the model's dtype."""
    accumulate = precision() if accumulate == "model" else accumulate
    head = _head(dtype, node_energy_dtype=accumulate)
    layers, element_index, node_head, batch = _batch(dtype)
    terms = head(layers, None, element_index, node_head, batch, 3)
    assert terms.total_energy.dtype == terms.interaction_energy.dtype == dtype
    assert (
        terms.node_energy.dtype
        == {"float32": torch.float32, "float64": torch.float64}[accumulate]
    )


@fp64_only
def test_an_accumulation_narrower_than_the_model_is_refused(dtype):
    with pytest.raises(ValueError, match="narrower than the model's precision"):
        _head(dtype, node_energy_dtype="float32")


@pytest.mark.parametrize(
    ("options", "message"),
    [
        ({"precision": "bfloat16"}, "precision='bfloat16' is not one of"),
        ({"node_energy_dtype": "float16"}, "node_energy_dtype='float16' is not one of"),
    ],
)
def test_a_dtype_a_model_does_not_compute_in_is_refused(dtype, options, message):
    with pytest.raises(ValueError, match=message):
        _head(dtype, **options)


@fp32_only
def test_a_device_without_float64_warns_when_built_and_holds_the_model_dtype(dtype):
    with pytest.warns(UserWarning, match="mps has no float64"):
        head = _head(dtype, device="mps")
    assert head.e0_table.dtype == dtype
    assert head.node_energy_dtype == dtype
    if torch.backends.mps.is_available():
        head = head.to("mps")
        layers, element_index, node_head, batch = _batch(dtype)
        terms = head(
            [layer.to("mps") for layer in layers],
            None,
            element_index.to("mps"),
            node_head.to("mps"),
            batch.to("mps"),
            3,
        )
        assert terms.node_energy.dtype == dtype


def test_a_float64_model_on_a_device_without_float64_is_refused():
    with pytest.raises(ValueError, match="precision='float64' on mps"):
        _head(torch.float64, device="mps")


def test_a_repeated_head_is_refused(dtype):
    with pytest.raises(ValueError, match="repeat a name"):
        EnergyOutputHead(
            ResolvedE0s(
                values={"a": dict.fromkeys(ATOMIC_NUMBERS, -1.0)},
                provenance={"a": "table"},
            ),
            ["a", "a"],
            ATOMIC_NUMBERS,
            [1.0, 1.0],
            [0.0, 0.0],
            zbl_in_scale_shift=True,
            precision=precision(),
        )


def test_a_device_with_float64_builds_without_a_warning(dtype):
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        _head(dtype, device="cpu")


@pytest.mark.parametrize("zbl_in_scale_shift", [True, False])
def test_the_pair_repulsion_is_scaled_only_inside_the_scale_shift(
    dtype, zbl_in_scale_shift
):
    """Inside, the pair term is scaled with the site energies; outside, it is
    added as it is. The two differ by ``(scale - 1) P`` per atom."""
    head = _head(dtype, zbl_in_scale_shift=zbl_in_scale_shift)
    layers, element_index, node_head, batch = _batch(dtype)
    pair = torch.linspace(0.5, 3.0, 7, dtype=dtype)
    without = head(layers, None, element_index, node_head, batch, 3)
    with_pair = head(layers, pair, element_index, node_head, batch, 3)
    added = head.scale[node_head] * pair if zbl_in_scale_shift else pair
    assert_close(with_pair.node_energy - without.node_energy, added, "pair term")


def test_the_total_and_the_interaction_energy_have_one_gradient(dtype):
    """The isolated-atom energies do not depend on the positions, so a
    derivative engine may differentiate the total."""
    head = _head(dtype)
    _, element_index, node_head, batch = _batch(dtype)
    positions = torch.randn(7, 3, dtype=dtype, requires_grad=True)
    layers = [positions.pow(2).sum(-1), positions.sin().sum(-1)]
    terms = head(layers, positions[:, 0].exp(), element_index, node_head, batch, 3)
    (from_total,) = torch.autograd.grad(
        terms.total_energy.sum(), positions, retain_graph=True
    )
    (from_interaction,) = torch.autograd.grad(terms.interaction_energy.sum(), positions)
    assert torch.equal(from_total, from_interaction)


def test_the_saved_state_is_the_table_in_float64_and_the_scale_and_shift(dtype):
    head = _head(dtype)
    state = canonical_state(head)[""]
    assert set(state) == {"e0_table", "scale", "shift"}
    assert state["e0_table"].dtype == torch.float64

    other = EnergyOutputHead(
        ResolvedE0s(
            values={name: dict.fromkeys(ATOMIC_NUMBERS, 0.0) for name in HEADS},
            provenance=dict.fromkeys(HEADS, "table"),
        ),
        HEADS,
        ATOMIC_NUMBERS,
        (1.0, 1.0),
        (0.0, 0.0),
        zbl_in_scale_shift=True,
        precision=precision(),
    )
    load_canonical_state(other, canonical_state(head))
    for name in ("e0_table", "scale", "shift"):
        assert torch.equal(getattr(other, name), getattr(head, name)), name


def test_a_saved_table_narrower_than_the_heads_is_refused(dtype):
    head = _head(dtype)
    state = canonical_state(head)[""]
    state["e0_table"] = state["e0_table"].float()
    with pytest.raises(ValueError, match="already lost the digits"):
        head.load_canonical(state)


@fp32_only
def test_a_head_without_float64_saves_its_table_as_it_holds_it(dtype):
    """A table rounded on a device without float64 is saved rounded, in its
    own dtype, so a head that holds float64 refuses it instead of taking the
    rounding for digits. A head of its own kind loads it."""
    with pytest.warns(UserWarning):
        rounded = _head(dtype, device="mps")
    state = canonical_state(rounded)
    assert state[""]["e0_table"].dtype == torch.float32
    with pytest.raises(ValueError, match="already lost the digits"):
        load_canonical_state(_head(dtype), state)
    with pytest.warns(UserWarning):
        load_canonical_state(_head(dtype, device="mps"), state)


@pytest.mark.parametrize(
    ("scale", "shift", "message"),
    [
        ((1.0, float("nan")), SHIFT, "scale .* is not finite"),
        (SCALE, (float("inf"), 0.0), "shift .* is not finite"),
        ((1.0, 0.0), SHIFT, "is not positive"),
        ((1.0, -2.0), SHIFT, "is not positive"),
    ],
)
def test_a_scale_or_shift_that_is_no_number_is_refused(dtype, scale, shift, message):
    with pytest.raises(ValueError, match=message):
        _head(dtype, scale=scale, shift=shift)


@pytest.mark.parametrize("name", ["scale", "shift"])
def test_one_scale_and_one_shift_per_head(dtype, name):
    with pytest.raises(ValueError, match=f"{name} has 1 value"):
        _head(dtype, **{name: (1.0,)})


def test_a_head_without_energies_is_an_error_at_build(dtype):
    with pytest.raises(KeyError, match="head 'ccsd'"):
        EnergyOutputHead(
            RESOLVED,
            ("pbe", "ccsd"),
            ATOMIC_NUMBERS,
            SCALE,
            SHIFT,
            zbl_in_scale_shift=True,
        )


def test_the_head_writes_no_double_cast():
    """The per-atom energies take ``node_energy_dtype`` and the sums the
    model's. A literal ``.double()`` would fix one of them whatever the
    arguments say, and on a device without float64 it would fail."""
    assert ".double()" not in inspect.getsource(energy)


@fp32_only
def test_an_embedding_energy_keeps_its_digits_beside_a_large_e0(dtype):
    """Each per-atom term is widened before it is added: summed in float32
    first, a 0.0123 eV term beside a 2041 eV one loses about 1e-5 eV."""
    head = _head(dtype)
    layers, element_index, node_head, batch = _batch(dtype)
    embedding = torch.full((7,), 0.0123, dtype=dtype)
    terms = head(layers, None, element_index, node_head, batch, 3, embedding)
    e0 = head.e0_table[node_head, element_index].to(dtype).double()
    interaction = (
        head.scale[node_head] * torch.stack(layers).sum(0) + head.shift[node_head]
    ).double()
    expected = e0 + embedding.double() + interaction
    torch.testing.assert_close(terms.node_energy, expected, atol=1e-12, rtol=0.0)
    totals = torch.stack([expected[batch == g].sum() for g in range(3)])
    torch.testing.assert_close(terms.total_energy.double(), totals, atol=2e-3, rtol=0.0)
