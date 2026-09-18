"""One forward, two energies, two dtypes: the legacy split, pinned.

`ScaleShiftMACE.forward` returns `total_energy` in the model's dtype and
`node_energy` in float64 (`mace/modules/models.py:581-582`), because the
per-atom quantity is assembled through `safe_double`. Plain `MACE` does not
(`:399`). So a `--default_dtype float32` run emits a float32 total and a
float64 per-atom decomposition **from the same call**, and no single dtype
policy reproduces both. This file turns that into committed characterization
rather than leaving it to be discovered by whoever writes the rewrite's
precision configuration -- it is the per-quantity specification BKD-2's
reduction dtype is written against.

**The MPS carve-out.** `safe_double` returns the tensor unchanged on Apple's
MPS backend, which has no float64 (`mace/modules/utils.py:22-31`). So the
`node_energy` dtype assertions below are CPU/CUDA statements: on MPS the same
model returns float32 for both quantities, and that is the documented
behaviour rather than a failure. Nothing here runs on MPS.

**Which classes widen is not a rule, it is a list.** `ScaleShiftMACE`,
`MACELES` and `PolarMACE` widen; `MagneticScaleShiftMACE` -- also a
scale-shift subclass -- does not, and neither does `EnergyDipolesMACE`. The
table below is checked against the package rather than remembered, so a new
model class that returns a `node_energy` has to be classified before this
suite passes.

**Reductions here use `scatter_sum`, never `.sum()`**, per the rule this file
also pins: `index_add_` accumulates sequentially, and at float32 with
realistic E0s the difference between summing before and after adding the E0s
is whole eV -- which torch's blocked-pairwise `.sum()` hides.
"""

import ast
import inspect
import io
import textwrap
from pathlib import Path

import numpy as np
import pytest
import torch

from mace.modules import extensions, models
from mace.modules.blocks import AtomicEnergiesBlock, ScaleShiftBlock
from mace.modules.utils import safe_double
from mace.tools import torch_tools
from mace.tools.scatter import scatter_sum
from tests.golden import harness
from tests.golden.anchors import anchor_graph, anchor_path, load_anchor

TOL = harness.FP64_CPU_REFERENCE

#: model class -> does its forward build `node_energy` through safe_double.
#: Measured off the source below, and confirmed at runtime for the two
#: classes with a committed anchor.
WIDENS_NODE_ENERGY = {
    "MACE": False,
    "ScaleShiftMACE": True,
    "EnergyDipolesMACE": False,
    "MACELES": True,
    "PolarMACE": True,
    "MagneticScaleShiftMACE": False,
}


@pytest.fixture(name="fixtures", scope="module")
def fixture_fixtures():
    return harness.load_fixtures()


# ---------------------------------------------------------------------------
# The dtype matrix: 2 quantities x 2 dtypes x 2 model classes
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("anchor", ["tiny_mace", "tiny_scaleshift"])
@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_the_two_energies_come_back_in_the_dtypes_the_legacy_stack_gives_them(
    anchor, dtype, fixtures
):
    """The whole matrix in one assertion table.

    `total_energy` is always the model's dtype. `node_energy` is float64
    whenever the class widens, *including* in a float32 run -- that is the
    inconsistency, and it is reproduced rather than fixed.
    """
    model = load_anchor(anchor, dtype)
    widens = WIDENS_NODE_ENERGY[type(model).__name__]
    with torch_tools.default_dtype(
        "float32" if dtype is torch.float32 else "float64"
    ):
        out = model(anchor_graph(model, fixtures["water_cluster"], dtype))
    assert out["energy"].dtype is dtype
    assert out["node_energy"].dtype is (torch.float64 if widens else dtype)
    assert out["forces"].dtype is dtype


@pytest.mark.parametrize("anchor", ["tiny_mace", "tiny_scaleshift"])
def test_the_two_energies_agree_at_fp64_and_disagree_at_fp32(anchor, fixtures):
    """The consequence of the split, as a number rather than as a warning.

    At float64 the total and the reduced per-atom energies are the same
    quantity computed twice and agree to the reference row. At float32 they
    are not: for the widening class they are not even the same dtype, and the
    difference is real -- 1.1e-8 eV on this fixture for the scale-shift
    anchor and 3.8e-6 eV for the plain one, both far above the float32
    round-off of the individual node energies. A rewrite that reports one of
    them as "the" energy has changed the other.
    """
    atoms = fixtures["water_cluster"]
    gaps = {}
    for dtype, name in ((torch.float64, "float64"), (torch.float32, "float32")):
        model = load_anchor(anchor, dtype)
        with torch_tools.default_dtype(name):
            graph = anchor_graph(model, atoms, dtype)
            out = model(graph, compute_force=False)
        reduced = scatter_sum(
            out["node_energy"], graph["batch"], dim=0, dim_size=1
        )
        gaps[name] = abs(float(reduced) - float(out["energy"]))
    assert gaps["float64"] < TOL.atol
    assert gaps["float32"] > 0.0


def test_the_widening_survives_the_reduction_and_the_narrow_total_does_not(
    fixtures,
):
    """Why the split is not merely cosmetic.

    Reducing the float64 `node_energy` of a float32 ScaleShiftMACE run gives
    a *different, and here closer*, answer than the float32 `total_energy`:
    measured against the float64 reference on the triclinic fixture, 5.8e-9
    eV against 7.9e-8 eV. So the two quantities are not two spellings of one
    number even in the mean.
    """
    atoms = fixtures["triclinic_bulk"]
    model64 = load_anchor("tiny_scaleshift", torch.float64)
    with torch_tools.default_dtype("float64"):
        exact = float(
            model64(
                anchor_graph(model64, atoms, torch.float64), compute_force=False
            )["energy"].detach()
        )
    model32 = load_anchor("tiny_scaleshift", torch.float32)
    with torch_tools.default_dtype("float32"):
        graph = anchor_graph(model32, atoms, torch.float32)
        out = model32(graph, compute_force=False)
    total_error = abs(float(out["energy"].detach()) - exact)
    reduced = float(
        scatter_sum(out["node_energy"].detach(), graph["batch"], dim=0, dim_size=1)
    )
    assert out["node_energy"].dtype is torch.float64
    assert abs(reduced - exact) < total_error


# ---------------------------------------------------------------------------
# Which classes widen: read out of the package, not remembered
# ---------------------------------------------------------------------------


def _node_energy_expressions(cls):
    """Every expression in `cls.forward` that produces `node_energy`.

    Two forms exist in the tree: an assignment to a local named
    `node_energy`, and a `"node_energy": ...` entry built directly in the
    returned dict (PolarMACE does the latter). A class that does both -- the
    common case, where the dict entry is just the local -- is read from its
    assignments, and the entry is required to be that bare local so nothing
    can be hidden in the second form.
    """
    source = textwrap.dedent(inspect.getsource(cls.forward))
    tree = ast.parse(source)
    assignments, entries = [], []
    for node in ast.walk(tree):
        if isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name) and target.id == "node_energy":
                    assignments.append(node.value)
        elif isinstance(node, ast.Dict):
            for key, value in zip(node.keys, node.values):
                if isinstance(key, ast.Constant) and key.value == "node_energy":
                    entries.append(value)
    if not assignments:
        return entries
    for entry in entries:
        assert (
            isinstance(entry, ast.Name) and entry.id == "node_energy"
        ), f"{cls.__name__} builds a second node_energy in its return dict"
    return assignments


def _calls_safe_double(expression):
    return any(
        isinstance(node, ast.Name) and node.id == "safe_double"
        for node in ast.walk(expression)
    )


def _model_classes_returning_a_node_energy():
    classes = {}
    for module in (models, extensions):
        for name, obj in vars(module).items():
            if not (isinstance(obj, type) and issubclass(obj, torch.nn.Module)):
                continue
            if obj.__module__ != module.__name__ or not hasattr(obj, "forward"):
                continue
            if _node_energy_expressions(obj):
                classes[name] = obj
    return classes


def test_every_class_that_reports_a_node_energy_is_classified():
    """A new model class has to declare which side of the split it is on.

    The table is not a comment: it is asserted against the set of classes
    that actually build a `node_energy`, so adding a seventh without deciding
    fails here rather than surfacing as an unexplained dtype downstream.
    """
    discovered = set(_model_classes_returning_a_node_energy())
    assert discovered == set(WIDENS_NODE_ENERGY), (
        "model classes returning node_energy have changed; classify the new "
        f"ones: {sorted(discovered ^ set(WIDENS_NODE_ENERGY))}"
    )


@pytest.mark.parametrize("name,widens", sorted(WIDENS_NODE_ENERGY.items()))
def test_the_widening_table_matches_the_source(name, widens):
    """Including the two counterexamples the audit turned up.

    `MagneticScaleShiftMACE` is a scale-shift subclass and does *not* widen,
    while `MACELES` and `PolarMACE` do -- so "scale-shift implies float64
    node energies" is a false rule that a port would otherwise be tempted to
    apply uniformly.
    """
    cls = _model_classes_returning_a_node_energy()[name]
    expressions = _node_energy_expressions(cls)
    assert expressions, name
    assert all(_calls_safe_double(e) for e in expressions) is widens


def test_safe_double_is_a_no_op_on_mps_by_construction():
    """The carve-out itself, on the only device this laptop has.

    `safe_double` branches on `t.device.type`, so the CPU behaviour is a
    widening and the MPS behaviour is documented in the source. Asserting the
    CPU half here means the branch is not silently deleted; the MPS half is
    what the module docstring above warns a future MPS run about.
    """
    tensor = torch.ones(3, dtype=torch.float32)
    assert safe_double(tensor).dtype is torch.float64
    assert torch.ones(3, dtype=torch.float64).dtype is safe_double(tensor).dtype


# ---------------------------------------------------------------------------
# The reduction primitive
# ---------------------------------------------------------------------------


def test_scatter_sum_exposes_a_float32_split_that_torch_sum_hides():
    """Why test arithmetic here uses `scatter_sum` and never `.sum()`.

    `scatter_sum` is `index_add_`: a sequential accumulation, which is what
    the model does to turn per-atom energies into a total. Torch's `.sum()`
    is blocked pairwise and is roughly a thousand times more accurate on the
    same data, so a test that reduces with `.sum()` measures a reduction the
    model never performs.

    Measured with 1,000 atoms at E0 = 1234.5678 eV against interaction
    energies of order 0.1 eV, in float32: adding the E0s and the interaction
    energies as two separate scatters, versus one scatter over their sum,
    differ by 5.0 eV. The same comparison through `.sum()` differs by 0.125.
    """
    n_atoms = 1000
    e0 = torch.full((n_atoms,), 1234.5678, dtype=torch.float32)
    generator = torch.Generator().manual_seed(20260810)
    interaction = (torch.rand(n_atoms, generator=generator, dtype=torch.float64) - 0.5) * 0.2
    interaction32 = interaction.to(torch.float32)
    index = torch.zeros(n_atoms, dtype=torch.long)

    split = scatter_sum(e0, index, dim=0, dim_size=1) + scatter_sum(
        interaction32, index, dim=0, dim_size=1
    )
    joint = scatter_sum(e0 + interaction32, index, dim=0, dim_size=1)
    scatter_gap = abs(float(split) - float(joint))

    pairwise_gap = abs(
        float(e0.sum() + interaction32.sum()) - float((e0 + interaction32).sum())
    )
    exact = float(e0.to(torch.float64).sum() + interaction.sum())

    assert scatter_gap > 1.0, f"measured 5.0 eV, got {scatter_gap}"
    assert pairwise_gap < scatter_gap / 10.0
    # and the order that adds first is the accurate one, which is what the
    # model does -- E0 enters per node, before the reduction.
    assert abs(float(joint) - exact) < abs(float(split) - exact)


# ---------------------------------------------------------------------------
# AtomicEnergiesBlock: where the E0s enter, and in which dtype
# ---------------------------------------------------------------------------

E0_VALUES = np.array([-13.6, -1023.123456789012345, 0.1])


def _block(values, dtype_name):
    with torch_tools.default_dtype(dtype_name):
        return AtomicEnergiesBlock(values)


def test_the_e0_buffer_is_rounded_once_at_construction():
    """Under whatever the global default dtype was at that moment.

    Nothing re-rounds it afterwards, so a float32-built model carries
    float32-rounded E0s forever, even when it is later run in float64.
    """
    float32_block = _block(E0_VALUES, "float32")
    float64_block = _block(E0_VALUES, "float64")
    assert float32_block.atomic_energies.dtype is torch.float32
    assert float64_block.atomic_energies.dtype is torch.float64
    assert float64_block.atomic_energies.tolist() == E0_VALUES.tolist()
    assert float32_block.atomic_energies.tolist() != E0_VALUES.tolist()


def test_the_forward_casts_the_buffer_to_the_inputs_dtype_in_both_directions():
    """`.to(dtype=x.dtype)` on every call (mace/modules/blocks.py:379-381).

    The *input* dtype wins, not the buffer's. Two consequences, both pinned:
    a float32 buffer under a float64 forward is **widened, not re-rounded**
    -- the returned values are the float32 approximations carried in a
    float64 tensor -- and, in the other direction, a float64 buffer under a
    float32 forward is silently narrowed. The second is the one that costs
    precision, and it is the direction a reader of the buffer's dtype would
    not expect.
    """
    float32_block = _block(E0_VALUES, "float32")
    widened = float32_block(torch.eye(3, dtype=torch.float64))
    assert widened.dtype is torch.float64
    assert widened.flatten().tolist() == [
        float(np.float32(value)) for value in E0_VALUES
    ]
    assert widened.flatten().tolist() != E0_VALUES.tolist()

    float64_block = _block(E0_VALUES, "float64")
    narrowed = float64_block(torch.eye(3, dtype=torch.float32))
    assert narrowed.dtype is torch.float32
    assert float64_block.atomic_energies.dtype is torch.float64, "buffer untouched"


@pytest.mark.parametrize("dtype_name", ["float32", "float64"])
def test_the_one_hot_matmul_is_bit_identical_to_a_gather(dtype_name):
    """The E0 lookup is a matmul against a one-hot, and could be an index.

    On CPU the two are bitwise equal at both dtypes, which is what lets a
    port replace one with the other. The caveat is TF32 on CUDA, which
    perturbs a matmul bitwise and would break the equality -- so the flag is
    asserted rather than assumed. It is off by default on the torch this
    project builds against, and this test is the tripwire if that changes.
    """
    assert torch.backends.cuda.matmul.allow_tf32 is False
    dtype = getattr(torch, dtype_name)
    block = _block(E0_VALUES, dtype_name)
    index = torch.tensor([0, 1, 2, 1, 0])
    one_hot = torch.zeros(5, 3, dtype=dtype)
    one_hot[torch.arange(5), index] = 1.0
    by_matmul = block(one_hot)
    by_gather = torch.atleast_2d(block.atomic_energies).T.to(dtype)[index]
    assert torch.equal(by_matmul, by_gather)


def test_where_the_e0s_enter_the_model(fixtures):
    """The readout sum, per node -- and for ScaleShiftMACE, outside the scale.

    `node_energy = node_e0 + scale_shift(everything else)`, so shifting the
    shift buffer by d moves every node energy by exactly d and the total by
    n_atoms * d, while the E0 contribution is untouched by either buffer.
    This is the application order that differs between the two classes, and
    the reason `--E0s` and `--mean`/`--std` are not interchangeable knobs.
    """
    atoms = fixtures["water_cluster"]
    model = load_anchor("tiny_scaleshift")
    with torch_tools.default_dtype("float64"):
        graph = anchor_graph(model, atoms)
        before = model(graph, compute_force=False)
        delta = 0.25
        model.scale_shift.shift = model.scale_shift.shift + delta
        after = model(anchor_graph(model, atoms), compute_force=False)
    per_node = (after["node_energy"] - before["node_energy"]).detach()
    assert torch.allclose(
        per_node,
        torch.full_like(per_node, delta),
        atol=TOL.atol,
        rtol=TOL.rtol,
    )
    assert float(after["energy"] - before["energy"]) == pytest.approx(
        len(atoms) * delta, abs=TOL.atol
    )


# ---------------------------------------------------------------------------
# The load paths, and what they do to the buffers
# ---------------------------------------------------------------------------


BUFFER_NAMES = ("atomic_energies_fn.atomic_energies", "scale_shift.scale", "scale_shift.shift")


def _buffers(model):
    named = dict(model.named_buffers())
    return {name: named[name] for name in BUFFER_NAMES if name in named}


def test_a_full_pickle_round_trip_preserves_every_buffer_bit_for_bit():
    model = load_anchor("tiny_scaleshift", torch.float32)
    stream = io.BytesIO()
    torch.save(model, stream)
    stream.seek(0)
    # deliberately loaded under the *other* global default dtype: nothing in
    # the load path consults it, so the buffers must come back float32.
    with torch_tools.default_dtype("float64"):
        restored = torch.load(stream, weights_only=False)
    original, reloaded = _buffers(model), _buffers(restored)
    assert set(original) == set(reloaded) and original
    for name, tensor in original.items():
        assert reloaded[name].dtype is tensor.dtype, name
        assert torch.equal(reloaded[name], tensor), name


def test_a_same_dtype_state_dict_resume_preserves_every_buffer():
    model = load_anchor("tiny_scaleshift", torch.float64)
    rebuilt = load_anchor("tiny_scaleshift", torch.float64)
    rebuilt.scale_shift.scale = rebuilt.scale_shift.scale * 3.0
    rebuilt.load_state_dict(model.state_dict())
    for name, tensor in _buffers(model).items():
        assert torch.equal(dict(rebuilt.named_buffers())[name], tensor), name


def test_the_rebuild_dtype_wins_on_a_cross_dtype_state_dict_load():
    """The only re-cast in the tree, and it is explicit.

    Loading a float32 state dict into a float64 model gives float64 tensors
    holding the float32 values -- widened, never recovered. Pinned because it
    is the one place a dtype changes on a load, and a port that instead keeps
    the checkpoint's dtype produces a model whose buffers disagree with its
    parameters.
    """
    narrow = load_anchor("tiny_scaleshift", torch.float32)
    wide = load_anchor("tiny_scaleshift", torch.float64)
    wide.load_state_dict(narrow.state_dict())
    for name, tensor in _buffers(narrow).items():
        loaded = dict(wide.named_buffers())[name]
        assert loaded.dtype is torch.float64, name
        assert torch.equal(loaded, tensor.to(torch.float64)), name


def test_the_convert_device_cli_preserves_the_buffers(tmp_path, monkeypatch):
    """`mace_convert_device` is torch.load -> .to(device) -> torch.save.

    Exercised here for cpu -> cpu, which is the only hop this machine can
    make; what it pins is that the round trip through the CLI is value- and
    dtype-preserving, so a cross-device conversion that changes a buffer is
    changing it on the `.to()`, not on the serialisation.
    """
    from mace.cli.convert_device import main  # noqa: PLC0415

    source = tmp_path / "anchor.model"
    source.write_bytes(Path(anchor_path("tiny_scaleshift")).read_bytes())
    output = tmp_path / "converted.model"
    monkeypatch.setattr(
        "sys.argv",
        ["mace_convert_device", "-t", "cpu", "-o", str(output), str(source)],
    )
    main()

    original = torch.load(source, weights_only=False, map_location="cpu")
    converted = torch.load(output, weights_only=False, map_location="cpu")
    for name, tensor in _buffers(original).items():
        assert torch.equal(dict(converted.named_buffers())[name], tensor), name
        assert dict(converted.named_buffers())[name].dtype is tensor.dtype, name


@pytest.mark.parametrize("dtype_name", ["float32", "float64"])
def test_the_scale_shift_buffers_are_also_frozen_at_construction(dtype_name):
    with torch_tools.default_dtype(dtype_name):
        block = ScaleShiftBlock(scale=0.1234567890123456789, shift=-1.5)
    expected = getattr(torch, dtype_name)
    assert block.scale.dtype is expected
    assert block.shift.dtype is expected
