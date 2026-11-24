"""The v1 output layer on the v1 backbone against whole legacy models.

Every weight the two layers rebuild is carried from the legacy model, by
:class:`tests.parity.legacy_models.V1Model`. The pair repulsion is the v1
module on the v1 edge lengths, so the forces see it as the legacy ones do.
Energies, per-atom energies and forces are compared at the
``closed_form_fp64`` row: the two stacks differ only in the order of their
fp64 arithmetic.

There is no committed multi-head anchor. The two-head model is built from the
trained single-head one, with the frozen tree's own readout classes and
arguments and a second row of every per-head constant, all of them different
from the first.
"""

import pytest
import torch
from e3nn import o3
from mace_core.config import ModelConfig, ReadoutConfig
from mace_torch.kernels import canonical_state

from mace import modules
from mace.tools import torch_tools, utils
from mace.tools.finetuning_utils import load_foundations_elements
from tests.golden import anchors, harness
from tests.parity.legacy_models import (
    ANCHOR_CONFIG,
    ELEMENTS,
    V1Model,
    forces,
    live_model,
    perturb,
)

CLOSED_FORM = harness.tolerance("closed_form_fp64")
FIXTURES = ("water_cluster", "triclinic_bulk", "dimer_short", "isolated_atom")


@pytest.fixture(name="fixtures", scope="module")
def fixture_fixtures():
    return harness.load_fixtures(names=FIXTURES, elements=ELEMENTS)


def _v1(legacy, last_only: bool = False) -> V1Model:
    """The v1 model of a legacy one of the anchors' backbone.

    The legacy gated readout's middle holds every head's; one head's is the
    configured width.
    """
    gated = legacy.readouts[-1]
    width = gated.hidden_irreps.dim // len(legacy.heads)
    config = ModelConfig(
        **ANCHOR_CONFIG,
        interaction_first="plain",
        interaction="residual",
        readout=ReadoutConfig(mlp_irreps=f"{width}x0e", last_only=last_only),
    )
    return V1Model(legacy, config)


def _graph(legacy, fixtures, heads=None):
    graph = dict(
        anchors.anchor_batch(
            legacy, [fixtures[name] for name in FIXTURES], torch.float64
        ).to_dict()
    )
    if heads is not None:
        graph["head"] = torch.tensor(heads)
    return graph


def _compare(legacy, v1, graph, what):
    expected = legacy(dict(graph), compute_force=True)
    output, positions = v1(graph)
    for name, ours, theirs in (
        ("energy", output.total_energy, expected["energy"]),
        ("node_energy", output.node_energies, expected["node_energy"]),
        ("forces", forces(output.total_energy, positions), expected["forces"]),
    ):
        torch.testing.assert_close(
            ours,
            theirs,
            atol=CLOSED_FORM.atol,
            rtol=CLOSED_FORM.rtol,
            msg=lambda message, name=name: f"{what}, {name}: {message}",
        )


@pytest.mark.parametrize("anchor", ["tiny_scaleshift", "tiny_mace"])
def test_an_anchor_is_reproduced_energy_and_forces(fp64, fixtures, anchor):
    legacy = anchors.load_anchor(anchor)
    _compare(legacy, _v1(legacy), _graph(legacy, fixtures), anchor)


def test_a_model_reading_out_only_its_last_layer_is_reproduced(fp64, fixtures):
    legacy = perturb(
        live_model(
            modules.ScaleShiftMACE,
            seed=11,
            hidden_irreps="16x0e + 16x1o",
            correlation=3,
            atomic_inter_scale=1.7,
            atomic_inter_shift=0.3,
            use_last_readout_only=True,
            use_reduced_cg=False,
        ),
        seed=12,
    )
    assert len(legacy.readouts) == 1
    v1 = _v1(legacy, last_only=True)
    assert v1.outputs.readouts["energy"].layers == (1,)
    _compare(legacy, v1, _graph(legacy, fixtures), "last_only")


# ---------------------------------------------------------------------------
# Two heads
# ---------------------------------------------------------------------------


def _two_head_model():
    """The trained anchor with a second head: readouts rebuilt for two heads
    with the frozen tree's classes and arguments, seeded, and a second row of
    isolated-atom energies, scale and shift."""
    legacy = anchors.load_anchor("tiny_scaleshift")
    hidden = legacy.readouts[0].linear.irreps_in
    last = legacy.readouts[1].linear_1.irreps_in
    width = legacy.readouts[1].hidden_irreps
    torch.manual_seed(21)
    with torch_tools.default_dtype("float64"):
        legacy.readouts = torch.nn.ModuleList(
            [
                modules.LinearReadoutBlock(hidden, o3.Irreps("2x0e")),
                modules.NonLinearReadoutBlock(
                    last,
                    (2 * width).simplify(),
                    torch.nn.functional.silu,
                    o3.Irreps("2x0e"),
                    2,
                ),
            ]
        ).double()
        energies = torch.atleast_2d(legacy.atomic_energies_fn.atomic_energies)
        legacy.atomic_energies_fn = modules.AtomicEnergiesBlock(
            torch.cat([energies, energies + torch.tensor([[0.7, -1.3, 2.1]])]).numpy()
        )
        scale = float(legacy.scale_shift.scale)
        shift = float(legacy.scale_shift.shift)
        legacy.scale_shift = modules.ScaleShiftBlock(
            scale=[scale, 1.9 * scale], shift=[shift, shift - 0.4]
        )
    legacy.heads = ["first", "second"]
    return legacy.double()


@pytest.fixture(name="two_heads", scope="module")
def fixture_two_heads():
    with torch_tools.default_dtype("float64"):
        legacy = _two_head_model()
        return legacy, _v1(legacy)


@pytest.mark.parametrize("heads", [[0, 0, 0, 0], [1, 1, 1, 1], [0, 1, 1, 0]])
def test_every_head_of_a_two_head_model_is_reproduced(fp64, fixtures, two_heads, heads):
    legacy, v1 = two_heads
    _compare(legacy, v1, _graph(legacy, fixtures, heads), f"heads {heads}")


def test_each_head_holds_what_the_frozen_tree_copies_into_it(fp64):
    """The frozen tree starts a multi-head fine-tune by copying a one-head
    model's readout into every head, scaling the gated second map by the
    ratio of the two fan-ins. Each head's readout state is then the one-head
    readout's own."""
    single = anchors.load_anchor("tiny_scaleshift")
    two = _two_head_model()
    load_foundations_elements(
        two,
        single,
        utils.AtomicNumberTable([int(z) for z in single.atomic_numbers]),
        load_readout=True,
        max_L=1,
    )
    copied = _v1(two).outputs.readouts["energy"]
    original = canonical_state(_v1(single).outputs.readouts["energy"])
    for head in (0, 1):
        for path, tensors in copied.head_state(head).items():
            torch.testing.assert_close(
                tensors["weight"],
                original[path]["weight"],
                atol=CLOSED_FORM.atol,
                rtol=CLOSED_FORM.rtol,
                msg=lambda message, head=head, path=path: (
                    f"head {head}, {path}: {message}"
                ),
            )
