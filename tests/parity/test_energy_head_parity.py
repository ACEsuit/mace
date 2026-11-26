"""The v1 energy head against the frozen tree's energy assembly, in one process.

The legacy models are run as they are, and what reaches their energy assembly
is captured with forward hooks: each layer's readout and the pair repulsion.
The v1 head is given the same per-layer site energies, the legacy model's
isolated-atom energies, scale and shift, and must give its energies back.

`ScaleShiftMACE` sums the isolated-atom energies and the interaction energies
separately, as the v1 head does, so it is reproduced bit for bit, in float64
and in float32. The plain `MACE` sums every contribution separately and adds
the sums, so the v1 head's two sums agree with it to rounding only.
"""

import numpy as np
import pytest
import torch
from mace_torch.models import EnergyOutputHead

from mace import modules
from tests.golden import anchors, harness
from tests.parity.legacy_models import ELEMENTS, energy_head_from, live_model

CLOSED_FORM = harness.tolerance("closed_form_fp64")

FIXTURES = ("water_cluster", "triclinic_bulk", "dimer_short", "isolated_atom")

#: Large and distinct isolated-atom energies, two heads' worth.
E0_ROWS = ((-13.6, -1029.8, -2041.3), (-13.2, -1030.4, -2040.1))


@pytest.fixture(name="fixtures", scope="module")
def fixture_fixtures():
    return harness.load_fixtures(names=FIXTURES, elements=ELEMENTS)


def _run_legacy(legacy, graph):
    """The legacy outputs, and what reached its energy assembly."""
    captured: dict[str, list] = {"layers": [], "pair": []}
    handles = [
        readout.register_forward_hook(
            lambda _module, _inputs, output: captured["layers"].append(output)
        )
        for readout in legacy.readouts
    ]
    if hasattr(legacy, "pair_repulsion_fn"):
        handles.append(
            legacy.pair_repulsion_fn.register_forward_hook(
                lambda _module, _inputs, output: captured["pair"].append(output)
            )
        )
    try:
        output = legacy(dict(graph), compute_force=False)
    finally:
        for handle in handles:
            handle.remove()
    node_head = graph["head"][graph["batch"]]
    rows = torch.arange(node_head.shape[0])
    layers = [layer[rows, node_head] for layer in captured["layers"]]
    pair = captured["pair"][0] if captured["pair"] else None
    return output, layers, pair


def _v1(head: EnergyOutputHead, graph, layers, pair):
    return head(
        layers,
        pair,
        graph["node_attrs"].argmax(dim=-1),
        graph["head"][graph["batch"]],
        graph["batch"],
        int(graph["ptr"].shape[0] - 1),
    )


def _graph(legacy, fixtures, dtype, head=0):
    graph = dict(
        anchors.anchor_batch(
            legacy, [fixtures[name] for name in FIXTURES], dtype
        ).to_dict()
    )
    graph["head"] = torch.full_like(graph["head"], head)
    return graph


@pytest.fixture(name="dtype", params=[torch.float64, torch.float32], ids=str)
def fixture_dtype(request):
    """The dtype under test, set as the process default as well: the legacy
    model reads the default while it runs, and a float32 model under a
    float64 default returns float64 energies."""
    previous = torch.get_default_dtype()
    torch.set_default_dtype(request.param)
    try:
        yield request.param
    finally:
        torch.set_default_dtype(previous)


def test_the_scaleshift_anchor_is_reproduced_bit_for_bit(fixtures, dtype):
    legacy = anchors.load_anchor("tiny_scaleshift", dtype)
    graph = _graph(legacy, fixtures, dtype)
    expected, layers, pair = _run_legacy(legacy, graph)
    precision = {torch.float64: "float64", torch.float32: "float32"}[dtype]
    ours = _v1(energy_head_from(legacy, precision), graph, layers, pair)

    assert ours.total_energy.dtype == dtype
    assert ours.node_energy.dtype == torch.float64
    assert torch.equal(ours.total_energy, expected["energy"])
    assert torch.equal(ours.interaction_energy, expected["interaction_energy"])
    assert torch.equal(ours.node_energy, expected["node_energy"])


def test_the_plain_anchor_is_reproduced_to_rounding(fp64, fixtures):
    legacy = anchors.load_anchor("tiny_mace")
    graph = _graph(legacy, fixtures, torch.float64)
    expected, layers, pair = _run_legacy(legacy, graph)
    ours = _v1(energy_head_from(legacy), graph, layers, pair)

    for name, value in (
        ("energy", ours.total_energy),
        ("node_energy", ours.node_energy),
    ):
        assert value.dtype == expected[name].dtype, name
        torch.testing.assert_close(
            value, expected[name], atol=CLOSED_FORM.atol, rtol=CLOSED_FORM.rtol
        )


def _pair_repulsion_model(model_class, seed, heads=("Default",), **extra):
    return live_model(
        model_class,
        seed=seed,
        atomic_energies=np.squeeze(E0_ROWS[: len(heads)]),
        pair_repulsion=True,
        heads=list(heads),
        **extra,
    )


@pytest.mark.parametrize(
    "model_class,extra",
    [
        (modules.MACE, {}),
        (
            modules.ScaleShiftMACE,
            {"atomic_inter_scale": 1.7, "atomic_inter_shift": 0.3},
        ),
    ],
    ids=["MACE", "ScaleShiftMACE"],
)
def test_each_class_places_the_pair_repulsion_where_its_flag_says(
    fp64, fixtures, model_class, extra
):
    """`ScaleShiftMACE` scales and shifts the pair repulsion with the
    interaction energy, and with a scale that is not one the other placement
    is more than an eV away. `MACE` adds it unscaled; with no scale or shift of
    its own the two placements give it the same energy. The short dimer makes
    the pair term large."""
    legacy = _pair_repulsion_model(model_class, seed=3, **extra)
    graph = _graph(legacy, fixtures, torch.float64)
    expected, layers, pair = _run_legacy(legacy, graph)
    assert pair is not None and pair.abs().max() > 1.0

    ours = energy_head_from(legacy)
    terms = _v1(ours, graph, layers, pair)
    torch.testing.assert_close(
        terms.total_energy,
        expected["energy"],
        atol=CLOSED_FORM.atol,
        rtol=CLOSED_FORM.rtol,
    )

    ours.zbl_in_scale_shift = not ours.zbl_in_scale_shift
    swapped = _v1(ours, graph, layers, pair)
    if model_class is modules.ScaleShiftMACE:
        assert (swapped.total_energy - expected["energy"]).abs().max() > 1.0
    else:
        torch.testing.assert_close(
            swapped.total_energy,
            terms.total_energy,
            atol=CLOSED_FORM.atol,
            rtol=CLOSED_FORM.rtol,
        )


def test_each_head_reads_its_own_row(fp64, fixtures):
    """A two-head `ScaleShiftMACE`, each head with its own isolated-atom
    energies, scale and shift, reproduced for a structure on either head."""
    legacy = _pair_repulsion_model(
        modules.ScaleShiftMACE,
        seed=5,
        heads=("first", "second"),
        atomic_inter_scale=[1.7, 0.6],
        atomic_inter_shift=[0.3, -0.4],
    )
    ours = energy_head_from(legacy)
    for head in (0, 1):
        graph = _graph(legacy, fixtures, torch.float64, head=head)
        expected, layers, pair = _run_legacy(legacy, graph)
        terms = _v1(ours, graph, layers, pair)
        assert torch.equal(terms.total_energy, expected["energy"]), head
        assert torch.equal(terms.node_energy, expected["node_energy"]), head
