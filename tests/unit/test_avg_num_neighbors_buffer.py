"""`avg_num_neighbors` is a buffer, including on models pickled before it was.

It became a registered buffer with a declared `torch.Tensor` type so that
TorchScript could see it. `_load_from_state_dict` covers checkpoints loaded as
state dicts; these tests cover the other way MACE ships a model, which is a
`torch.load` of the whole pickled module -- the ASE calculator, every
`mace/cli` tool, and the pretrained artifacts inside the wheel. Unpickling
restores `__dict__` directly, so neither `__init__` nor `_load_from_state_dict`
runs, and without `__setstate__` the instance keeps its plain Python float.

That is invisible in eager mode, because a float promotes against whatever it
divides. It is fatal under `torch.jit.script`, which is why
`mace_create_lammps_model` is the test that matters here: exporting any
pre-buffer checkpoint died on

    Could not cast attribute 'avg_num_neighbors' to type Tensor
"""

import io

import numpy as np
import pytest
import torch
import torch.nn.functional as F
from e3nn import o3

from mace import modules, tools

#: not exactly representable in float32, so a float32/float64 mix-up shows up as
#: a changed value rather than passing by luck (4.59375, the value the committed
#: anchor carries, is exact in both and would hide it).
AVG_NUM_NEIGHBORS = 8.317_483_291_745_2

TABLE = tools.AtomicNumberTable([1, 6])


def _build_model() -> modules.ScaleShiftMACE:
    return modules.ScaleShiftMACE(
        r_max=3.0,
        num_bessel=4,
        num_polynomial_cutoff=5,
        max_ell=2,
        interaction_cls=modules.interaction_classes[
            "RealAgnosticResidualInteractionBlock"
        ],
        interaction_cls_first=modules.interaction_classes[
            "RealAgnosticInteractionBlock"
        ],
        num_interactions=2,
        num_elements=2,
        hidden_irreps=o3.Irreps("8x0e + 8x1o"),
        MLP_irreps=o3.Irreps("4x0e"),
        gate=F.silu,
        atomic_energies=np.array([-1.0, -5.0]),
        avg_num_neighbors=AVG_NUM_NEIGHBORS,
        atomic_numbers=TABLE.zs,
        correlation=2,
        atomic_inter_scale=1.0,
        atomic_inter_shift=0.0,
    )


def _make_legacy(
    model: torch.nn.Module, value: float = AVG_NUM_NEIGHBORS
) -> torch.nn.Module:
    """The pickled shape of a pre-buffer checkpoint: a plain float attribute.

    Written by hand rather than by committing an old checkpoint, so the test
    says what the old shape *was* instead of depending on a binary.

    `value` is set from the double it started as, not read back out of the
    buffer: before the buffer existed nothing rounded it, so a real pre-buffer
    checkpoint carries the full-precision float. Reading it back through the
    current buffer -- which `__init__` builds at the *process* default dtype --
    would hand the test an already-rounded number and hide exactly the
    precision question the dtype case below asks.
    """
    for block in model.interactions:
        del block._buffers["avg_num_neighbors"]
        block.__dict__["avg_num_neighbors"] = value
    return model


def _round_trip(model: torch.nn.Module) -> torch.nn.Module:
    """Save and load the whole module, which is how MACE ships a checkpoint."""
    buffer = io.BytesIO()
    torch.save(model, buffer)
    buffer.seek(0)
    return torch.load(buffer, weights_only=False, map_location="cpu")


def test_a_legacy_pickle_comes_back_with_the_buffer():
    legacy = _make_legacy(_build_model())
    assert "avg_num_neighbors" not in legacy.interactions[0]._buffers

    loaded = _round_trip(legacy)

    for block in loaded.interactions:
        assert "avg_num_neighbors" in block._buffers
        assert "avg_num_neighbors" not in block.__dict__
        assert isinstance(block.avg_num_neighbors, torch.Tensor)


def test_a_legacy_pickle_can_be_torchscripted():
    """The failure this fixes, at its narrowest: the declared type is checked
    only when the module is scripted, which is what the LAMMPS export does."""
    loaded = _round_trip(_make_legacy(_build_model()))
    assert torch.jit.script(loaded.interactions[0]) is not None


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_the_promoted_buffer_keeps_the_value_at_the_models_precision(dtype):
    loaded = _round_trip(_make_legacy(_build_model().to(dtype)))

    buffer = loaded.interactions[0].avg_num_neighbors
    assert buffer.dtype == dtype
    assert buffer.item() == torch.tensor(AVG_NUM_NEIGHBORS, dtype=dtype).item()


def test_the_dtype_follows_the_model_not_the_process_default():
    """A float participated at the precision of whatever it divided, so taking
    the process default would silently round the normalization of a float64
    model loaded under a float32 default -- every message is scaled by it."""
    previous = torch.get_default_dtype()
    torch.set_default_dtype(torch.float32)  # set, not asserted: xdist shares a process
    try:
        loaded = _round_trip(_make_legacy(_build_model().double()))
    finally:
        torch.set_default_dtype(previous)

    buffer = loaded.interactions[0].avg_num_neighbors
    assert buffer.dtype == torch.float64
    assert buffer.item() == AVG_NUM_NEIGHBORS


def test_the_promoted_buffer_lands_beside_the_weights():
    """A float has no device; a buffer does, and it has to be the right one.

    Asserted on every host, but it can only *fail* where a second device
    exists, so the gpu case below is the one that measures it. Together they
    say the same thing: after promotion the module's tensors are colocated.
    """
    loaded = _round_trip(_make_legacy(_build_model()))

    block = loaded.interactions[0]
    assert block.avg_num_neighbors.device == next(block.parameters()).device


@pytest.mark.gpu
def test_the_promoted_buffer_follows_a_map_location_onto_the_device():
    """`torch.load(..., map_location="cuda")` is a load path that never calls
    `.to()`, so nothing downstream repairs a buffer left on the CPU.

    Building the buffer without a device left the module split across two
    devices. `avg_num_neighbors` is zero-dim, so the division itself survives
    under torch's cpu-scalar rule; what breaks is every caller that assumes a
    module's tensors are colocated, DDP among them.

    Scoped to these buffers on purpose. "every buffer sits on the parameter
    device" is the assertion a reader reaches for next, and it fails for a
    reason that has nothing to do with this: e3nn keeps its Wigner 3j tables
    inside `TensorProduct._compiled_main_left_right`, a `torch.fx.GraphModule`
    that unpickles by regenerating itself, so those buffers come back on the
    CPU whatever `map_location` said. That reproduces on a model pickled long
    after this promotion existed, so it is e3nn's to answer for, not this.
    """
    buffer = io.BytesIO()
    torch.save(_make_legacy(_build_model()), buffer)
    buffer.seek(0)
    loaded = torch.load(buffer, weights_only=False, map_location="cuda")

    for block in loaded.interactions:
        assert block.avg_num_neighbors.device.type == "cuda"
        assert block.avg_num_neighbors.device == next(block.parameters()).device


def test_a_model_pickled_after_the_buffer_landed_is_untouched():
    """The promotion must not fire twice, or re-cast a buffer somebody set."""
    model = _build_model().double()
    loaded = _round_trip(model)

    buffer = loaded.interactions[0].avg_num_neighbors
    assert buffer.dtype == torch.float64
    assert buffer.item() == pytest.approx(AVG_NUM_NEIGHBORS)
    assert "avg_num_neighbors" not in loaded.interactions[0].__dict__


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_promotion_reproduces_the_arithmetic_of_the_float_it_replaced(dtype):
    """The reason the dtype follows the model, measured rather than argued.

    The reference is the *legacy* block itself -- still holding the plain float,
    which eager mode is happy to divide by -- so the comparison is against what
    the checkpoint computed before the buffer existed, in the same module with
    the same weights. Bit-exact, not close: the promotion is a change of storage
    and must not be a change of numerics.
    """
    torch.manual_seed(2026)
    legacy = _make_legacy(_build_model().to(dtype))
    block = legacy.interactions[0]

    node_attrs = torch.zeros(3, 2, dtype=dtype)
    node_attrs[:, 0] = 1.0
    node_feats = torch.randn(3, block.node_feats_irreps.dim, dtype=dtype)
    edge_index = torch.tensor([[0, 1, 1, 2], [1, 0, 2, 1]])
    vectors = torch.randn(4, 3, dtype=dtype)
    edge_attrs = o3.spherical_harmonics(
        block.edge_attrs_irreps, vectors, normalize=True, normalization="component"
    )
    edge_feats = torch.randn(4, block.edge_feats_irreps.dim, dtype=dtype)

    def messages(module):
        with torch.no_grad():
            out, _ = module(
                node_attrs=node_attrs,
                node_feats=node_feats,
                edge_attrs=edge_attrs,
                edge_feats=edge_feats,
                edge_index=edge_index,
            )
        return out

    reference = messages(block)
    promoted = messages(_round_trip(legacy).interactions[0])

    torch.testing.assert_close(promoted, reference, rtol=0, atol=0)
