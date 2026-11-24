"""The tiny AtomicDielectricMACE anchor reproduces its committed reference.

This is the class behind MACE-MDP. It reads out a charge, a dipole and a
polarizability from every layer, normalizes the charges to the structure's
total charge, and returns the position Jacobians of the dipole and the
polarizability. The rewrite expresses it as observables rather than as a model
class, and this reference is what that has to reproduce. Like the dipole
anchor it runs per pull request, on CPU, with no network and no optional
dependency, and carries no capability marker.
"""

import json

import numpy as np
import pytest
import torch

from mace.tools import torch_tools
from tests.golden import harness, routes
from tests.golden.anchors import load_checkpoint
from tests.golden.build_dielectric_anchor import (
    ANCHOR_CONFIG,
    MODEL_PATH,
    SIDECAR_PATH,
    build_model,
)
from tests.golden.targets import dielectric

#: For comparisons whose sides differ only in the order of the fp64
#: operations: the recipe rebuilt against the committed checkpoint, and an
#: output against its closed form in the same process.
CLOSED_FORM = harness.tolerance("closed_form_fp64")

REFERENCE_PATH = harness.REFERENCES_DIR / dielectric.REFERENCE_NAME


def _forward(model, graph):
    with torch_tools.default_dtype("float64"):
        return model(graph, **dielectric.FORWARD_KWARGS)


@pytest.fixture(name="fixtures", scope="module")
def fixture_fixtures():
    return harness.load_fixtures(names=dielectric.FIXTURE_NAMES)


def test_anchor_reproduces_its_reference(fixtures):
    snapshot = harness.snapshot_outputs(
        routes.ForwardRoute(
            load_checkpoint(MODEL_PATH),
            dielectric.projection,
            forward_kwargs=dielectric.FORWARD_KWARGS,
        ),
        fixtures,
        dtype="float64",
        device="cpu",
        backend="e3nn",
    )
    reference = harness.load_reference(REFERENCE_PATH)
    harness.compare_to_reference(
        snapshot, reference, row=harness.FP64_CPU_REFERENCE.name
    )


def test_the_named_fixtures_are_molecular_and_hco(fixtures):
    """A dipole under periodic boundaries depends on the choice of cell, and
    the anchor knows H, C and O only, so every named fixture is aperiodic and
    made of those elements."""
    for name, atoms in fixtures.items():
        assert not atoms.pbc.any(), f"{name} is periodic"
        assert set(atoms.numbers) <= {1, 6, 8}, f"{name} has {set(atoms.numbers)}"


def test_anchor_is_the_class_it_claims_to_be():
    model = load_checkpoint(MODEL_PATH)
    assert type(model).__name__ == "AtomicDielectricMACE"
    assert not hasattr(model, "atomic_energies_fn")
    assert not hasattr(model, "scale_shift")
    assert model.use_polarizability


def test_the_reference_pins_a_response_and_not_zeros():
    """A channel that is zero on every fixture passes against a rewrite that
    drops it. A polarizability head over features without ``2e`` is the case in
    point: its polarizability is exactly isotropic. So on the water cluster
    every channel, and the anisotropic part of the polarizability, has to stand
    clear of the tolerance it is compared at."""
    floor = harness.FP64_CPU_REFERENCE.atol
    reference = harness.load_reference(REFERENCE_PATH)
    water = reference["fixtures"]["water_cluster"]["outputs"]

    def values(name):
        return np.asarray(water[name]["value"], dtype=float)

    for name in dielectric.CHANNELS:
        assert np.abs(values(name)).max() > floor, f"{name} is zero"
    alpha = values("polarizability")
    anisotropic = alpha - np.trace(alpha) / 3.0 * np.eye(3)
    assert np.abs(anisotropic).max() > floor
    # The last five spherical components are the 2e block.
    assert np.abs(values("polarizability_sh")[1:]).max() > floor


def test_the_recipe_still_builds_the_committed_anchor():
    """The builder is the recipe the sidecar points to, so it has to produce
    the committed checkpoint: every parameter and buffer, not only the ones a
    seed does not touch.

    The shapes are compared exactly. The Clebsch-Gordan basis is where that
    matters: ``AtomicDielectricMACE`` accepts ``use_reduced_cg`` and ignores
    it, so the product basis is the plain e3nn one, and the reduced basis has a
    different extent. The values are compared at the ``closed_form_fp64`` row,
    since e3nn builds the basis through torch linalg and the order of the fp64
    operations moves between torch builds.
    """
    committed = load_checkpoint(MODEL_PATH)
    with torch_tools.default_dtype("float64"):
        rebuilt = build_model().to(torch.float64)

    committed_state = committed.state_dict()
    rebuilt_state = rebuilt.state_dict()
    assert set(rebuilt_state) == set(committed_state)
    assert any(
        "U_matrix" in name for name in committed_state
    ), "the anchor has no symmetric-contraction basis"
    for name, tensor in committed_state.items():
        other = rebuilt_state[name]
        assert other.shape == tensor.shape, (
            f"{name}: the recipe now builds {tuple(other.shape)} and the "
            f"committed anchor carries {tuple(tensor.shape)}"
        )
        if tensor.is_floating_point():
            harness.assert_within(CLOSED_FORM, other.numpy(), tensor.numpy(), name)
        else:
            assert torch.equal(other, tensor), name

    for product in committed.products:
        contraction = product.symmetric_contractions
        assert type(contraction).__module__.startswith("mace."), (
            "the committed anchor holds a cuequivariance contraction object, "
            "which cannot be unpickled on a machine without the package"
        )


def test_the_charges_follow_the_total_charge(fixtures):
    """The normalization the committed reference cannot pin.

    None of the fixtures carries a total charge, so every reference structure
    is neutral and the charges sum to zero. The class shifts every atom of a
    structure by one constant so the charges sum to ``total_charge``
    (mace/modules/models.py:1132-1135), and the graph dipole adds
    ``sum_i q_i r_i`` in e*Angstrom. Setting a total charge ``Q`` on ``N``
    atoms must therefore move each charge by ``Q / N`` and the dipole by
    ``(Q / N) sum_i r_i``, with no unit conversion. The reference pins the
    neutral response; this pins the response to a total charge.
    """
    model = load_checkpoint(MODEL_PATH)
    atoms = fixtures["water_cluster"]
    neutral_graph = routes.graph_batch(model, atoms)
    assert torch.count_nonzero(neutral_graph["total_charge"]) == 0, (
        "the fixture now carries a total charge, so the committed reference "
        "was taken on a charged structure and this test's premise is gone"
    )
    positions = neutral_graph["positions"].detach().clone()
    neutral = _forward(model, neutral_graph)

    total = 1.0
    charged_graph = routes.graph_batch(model, atoms)
    charged_graph["total_charge"] = torch.full_like(
        charged_graph["total_charge"], total
    )
    charged = _forward(model, charged_graph)

    shift = total / len(atoms)
    harness.assert_within(
        CLOSED_FORM,
        (charged["charges"] - neutral["charges"]).detach().numpy(),
        np.full(len(atoms), shift),
        "the charges did not move by Q / N",
    )
    harness.assert_within(
        CLOSED_FORM,
        float(charged["charges"].sum().detach()),
        total,
        "the charges do not sum to the total charge",
    )
    harness.assert_within(
        CLOSED_FORM,
        (charged["dipole"][0] - neutral["dipole"][0]).detach().numpy(),
        (shift * positions.sum(dim=0)).numpy(),
        "the dipole did not move by (Q / N) sum_i r_i in e*Angstrom",
    )


@pytest.mark.parametrize("total", [0.0, 1.0])
def test_the_jacobians_respect_translation(fixtures, total):
    """The Jacobians against the one thing known in closed form.

    Translating a structure by ``t`` leaves every charge, atomic dipole and
    polarizability unchanged and moves the dipole by ``Q t``. Summed over
    atoms, ``dmu_dr`` is therefore ``Q`` times the identity and ``dalpha_dr``
    is zero. That holds only if the dipole's Jacobian includes the charge
    baseline, which is the part a rewrite computing it from the atomic dipoles
    alone would miss.
    """
    model = load_checkpoint(MODEL_PATH)
    atoms = fixtures["water_cluster"]
    graph = routes.graph_batch(model, atoms)
    graph["total_charge"] = torch.full_like(graph["total_charge"], total)
    out = _forward(model, graph)

    dmu_dr = out["dmu_dr"].detach().numpy()
    dalpha_dr = out["dalpha_dr"].detach().numpy()
    assert dmu_dr.shape == (3, len(atoms), 3)
    assert dalpha_dr.shape == (9, len(atoms), 3)
    harness.assert_within(
        CLOSED_FORM,
        dmu_dr.sum(axis=1),
        total * np.eye(3),
        "sum_i dmu/dr_i is not Q times the identity",
    )
    harness.assert_within(
        CLOSED_FORM,
        dalpha_dr.sum(axis=1),
        np.zeros((9, 3)),
        "sum_i dalpha/dr_i is not zero",
    )


def test_reference_carries_dtype_units_and_provenance():
    reference = harness.load_reference(REFERENCE_PATH)
    assert reference["dtype"] == "float64"
    assert reference["device"] == "cpu"
    assert reference["backend"] == "e3nn"
    provenance = reference["provenance"]
    assert provenance["source"].endswith(MODEL_PATH.name)
    assert provenance["recipe"] == "tests/golden/build_dielectric_anchor.py"
    assert provenance["tolerance_row"] == harness.FP64_CPU_REFERENCE.name
    assert set(reference["fixtures"]) == set(dielectric.FIXTURE_NAMES)
    for entry in reference["fixtures"].values():
        assert set(entry["outputs"]) == set(dielectric.CHANNELS)
        for name, channel in entry["outputs"].items():
            declared = harness.CHANNELS[name]
            assert channel["kind"] == declared.kind, name
            assert channel["unit"] == declared.unit, name


def test_sidecar_records_how_the_anchor_was_built():
    sidecar = json.loads(SIDECAR_PATH.read_text(encoding="utf-8"))
    assert sidecar["model"] == MODEL_PATH.name
    assert sidecar["class"] == "AtomicDielectricMACE"
    assert sidecar["dtype"] == "float64"
    assert sidecar["seed"]
    assert sidecar["command"]
    assert "--target dielectric" in sidecar["regenerate_with"]
    assert sidecar["config"] == ANCHOR_CONFIG
    assert sidecar["atomic_energies"] is None


def test_anchor_checkpoint_stays_small():
    size_mb = MODEL_PATH.stat().st_size / 1e6
    assert size_mb < 1.5, f"{MODEL_PATH.name} is {size_mb:.2f} MB"
