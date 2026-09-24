"""The dipole and dielectric models on their own terms, with no legacy code.

Their numbers against the frozen tree are pinned in ``tests/parity``. What is
pinned here is what the models promise regardless of any oracle: the dipole
turns with the structure and the polarizability turns as a matrix, predicted
charges add up to the structure's total, the declared derivatives are the
derivatives of what the model reports, and only the two models the frozen
tree has can be built.
"""

from __future__ import annotations

from typing import cast

import numpy as np
import pytest
import torch
from conftest import fp64_only
from mace_core.observables import DEFAULT_CATALOGUE, ObservableSpec
from mace_torch.backends.reference import ReferenceBackend
from mace_torch.kernels.initialization import initialize_model_weights
from mace_torch.models.dipoles import (
    E_ANGSTROM_PER_DEBYE,
    DipoleModel,
    DipoleSettings,
)
from mace_torch.models.heads import ObservableHead
from mace_torch.physics import DerivativeEngine
from mace_torch_engine_fixtures import ENERGY, build_graph, molecule
from scipy.spatial.transform import Rotation

FIXED = DipoleSettings(charges="fixed")
PREDICTED = DipoleSettings(charges="predicted", polarizability=True)
RESPONSES = [DEFAULT_CATALOGUE.observable(n) for n in ("dipole", "polarizability")]


def build_model(settings, readout_hidden="4x0e+4x1o+4x2e") -> DipoleModel:
    model = DipoleModel(
        ReferenceBackend(),
        atomic_numbers=[1, 8],
        settings=settings,
        num_layers=2,
        num_features=3,
        lmax=2,
        hidden_irreps="0e+1o+2e",
        num_radial=4,
        cutoff=5.0,
        correlation=2,
        readout_hidden=readout_hidden,
    )
    initialize_model_weights(model, seed=4)
    return model


def graph_of(positions, numbers, charges=None, total_charge=0.0):
    graph = build_graph(positions, numbers)
    count = len(numbers)
    graph["charges"] = torch.tensor(
        np.linspace(-0.4, 0.4, count) if charges is None else charges
    )
    graph["total_charge"] = torch.tensor([total_charge])
    return graph


def evaluate(model, graph, compute=()):
    return DerivativeEngine(model, None, responses=RESPONSES)(graph, compute=compute)


# ---------------------------------------------------------------------------
# Which models exist
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("charges", "polarizability"), [("fixed", True), ("predicted", False)]
)
def test_only_the_two_models_the_frozen_tree_has_can_be_built(charges, polarizability):
    with pytest.raises(ValueError, match="not a model that exists"):
        DipoleSettings(charges=charges, polarizability=polarizability)


def test_a_charge_source_that_is_neither_is_refused():
    with pytest.raises(ValueError, match="'fixed'"):
        DipoleSettings(charges="learned")  # ty: ignore[invalid-argument-type]


def test_the_fixed_charge_model_keeps_only_vectors_in_its_last_layer():
    assert build_model(FIXED).backbone.layer_irreps[-1] == "1o"
    assert build_model(PREDICTED).backbone.layer_irreps[-1] == "0e+1o+2e"


# ---------------------------------------------------------------------------
# Symmetry
# ---------------------------------------------------------------------------


@fp64_only
@pytest.mark.parametrize("settings", [FIXED, PREDICTED], ids=["fixed", "predicted"])
def test_the_dipole_turns_with_the_structure(dtype, settings):
    positions, numbers = molecule()
    rotation = Rotation.from_euler("xyz", [0.4, -0.9, 1.3]).as_matrix()
    model = build_model(settings)
    plain = evaluate(model, graph_of(positions, numbers))
    turned = evaluate(model, graph_of(positions @ rotation.T, numbers))
    expected = plain.dipole.detach().numpy() @ rotation.T
    np.testing.assert_allclose(turned.dipole.detach().numpy(), expected, atol=1e-12)
    assert np.abs(expected).max() > 1e-6


@fp64_only
def test_the_polarizability_turns_as_a_symmetric_matrix(dtype):
    positions, numbers = molecule()
    rotation = Rotation.from_euler("xyz", [1.1, 0.2, -0.6]).as_matrix()
    model = build_model(PREDICTED)
    plain = evaluate(model, graph_of(positions, numbers)).extras["polarizability"]
    turned = evaluate(model, graph_of(positions @ rotation.T, numbers)).extras[
        "polarizability"
    ]
    alpha = plain.detach().numpy()[0]
    np.testing.assert_allclose(alpha, alpha.T, atol=1e-14)
    np.testing.assert_allclose(
        turned.detach().numpy()[0], rotation @ alpha @ rotation.T, atol=1e-12
    )
    assert np.abs(alpha - np.trace(alpha) / 3 * np.eye(3)).max() > 1e-6


# ---------------------------------------------------------------------------
# Charges
# ---------------------------------------------------------------------------


@fp64_only
@pytest.mark.parametrize("total", [0.0, -1.0, 2.0])
def test_predicted_charges_add_up_to_the_total_charge(dtype, total):
    positions, numbers = molecule()
    result = evaluate(build_model(PREDICTED), graph_of(positions, numbers, None, total))
    charges = result.extras["charges"]
    assert charges.shape == (len(numbers),)
    assert abs(float(charges.sum()) - total) < 1e-12


@fp64_only
def test_fixed_charges_add_their_dipole_in_debye(dtype):
    positions, numbers = molecule()
    model = build_model(FIXED)
    charges = np.array([0.3, -0.1, -0.1, -0.1])
    charged = evaluate(model, graph_of(positions, numbers, charges))
    neutral = evaluate(model, graph_of(positions, numbers, np.zeros(4)))
    difference = (charged.dipole - neutral.dipole).detach().numpy()[0]
    np.testing.assert_allclose(
        difference, charges @ positions / E_ANGSTROM_PER_DEBYE, atol=1e-12
    )


def test_the_debye_factor_is_the_frozen_tree_s_and_not_ase_s():
    """Measured: the two differ in the eighth digit, and the trained models of
    this kind were fitted against the frozen tree's."""
    from mace_core.units import DEBYE

    assert E_ANGSTROM_PER_DEBYE == 0.20819433270935597
    assert abs(E_ANGSTROM_PER_DEBYE / DEBYE - 1) > 1e-9


# ---------------------------------------------------------------------------
# The declared derivatives
# ---------------------------------------------------------------------------


@fp64_only
@pytest.mark.parametrize(
    ("name", "of"), [("dmu_dr", "dipole"), ("dalpha_dr", "polarizability")]
)
def test_a_response_derivative_is_the_derivative_of_what_is_reported(dtype, name, of):
    positions, numbers = molecule()
    model = build_model(PREDICTED)
    result = evaluate(model, graph_of(positions, numbers), compute=(name,))
    derivative = result.extras[name].detach().numpy()
    components = 3 if of == "dipole" else 9
    assert derivative.shape == (components, len(numbers), 3)

    step = 1e-5
    for atom, axis in [(0, 0), (1, 2), (3, 1)]:
        shifted = []
        for sign in (1, -1):
            moved = positions.copy()
            moved[atom, axis] += sign * step
            value = evaluate(model, graph_of(moved, numbers)).get(of)
            shifted.append(value.detach().numpy().reshape(-1))
        numeric = (shifted[0] - shifted[1]) / (2 * step)
        np.testing.assert_allclose(derivative[:, atom, axis], numeric, atol=1e-8)


def test_a_model_with_no_energy_is_refused_an_energy_derivative():
    positions, numbers = molecule()
    with pytest.raises(ValueError, match="forces"):
        evaluate(build_model(FIXED), graph_of(positions, numbers), compute=("forces",))


def test_a_response_the_model_does_not_produce_is_named():
    positions, numbers = molecule()
    with pytest.raises(ValueError, match="'polarizability'"):
        evaluate(
            build_model(FIXED), graph_of(positions, numbers), compute=("dalpha_dr",)
        )


def test_an_energy_model_can_still_ask_for_forces_beside_a_response():
    """The engine takes both kinds in one call when a model has both."""
    engine = DerivativeEngine(build_model(FIXED), ENERGY, responses=RESPONSES)
    assert "dmu_dr" in engine.responses
    assert engine.derivative_names() == {}


# ---------------------------------------------------------------------------
# The readout middle
# ---------------------------------------------------------------------------


@fp64_only
def test_a_scalar_middle_gives_the_last_layer_no_vector_to_read(dtype):
    """The frozen tree's rule: an output irrep the middle does not carry has no
    path through the second map, so the last layer adds exactly nothing to it.
    """
    positions, numbers = molecule()
    model = build_model(FIXED, readout_hidden="4x0e")
    head = cast(ObservableHead, model.outputs.heads["atomic_dipoles"])
    graph = graph_of(positions, numbers)
    layers = model.backbone(graph)
    last = head.per_layer(layers)[-1]
    assert torch.count_nonzero(last) == 0
    assert torch.count_nonzero(head.per_layer(layers)[0]) > 0


def test_a_middle_without_scalars_is_refused():
    with pytest.raises(ValueError, match="no scalars"):
        build_model(FIXED, readout_hidden="4x1o")


def test_the_heads_are_implementation_and_not_catalogue_names():
    """`dipole` and `polarizability` are what a configuration declares; the
    per-atom readouts behind them are the model's own."""
    heads = {spec.name for spec in PREDICTED.heads}
    assert heads == {"charges", "atomic_dipoles", "polarizability_sh"}
    assert all(isinstance(spec, ObservableSpec) for spec in PREDICTED.heads)
    assert set(DipoleModel.PRODUCED) == {"dipole", "polarizability"}
