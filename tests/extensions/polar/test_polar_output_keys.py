"""`PolarMACE.forward` returns the electrostatic keys its consumers read.

Seven of the model output keys in the inventory are produced in one dict in
`PolarMACE.forward` and nowhere else: `spins`, `total_charge`, `fermi_level`,
`external_field`, `spin_density`, `spin_charge_density` and `charges_history`. The
existing polar tests check equivariance, the cueq parity and the density cube CLI,
none of which asserts that a given key is present or what shape it has, so any of
the seven could be renamed or dropped and only a downstream KeyError would say so.

Shapes are the point rather than values: `fermi_level` is per graph and `spins` is
per atom, and a key that quietly changed which of those it is would still be
"present". The values are checked only for being finite, since the weights are
random here. The numbers belong to the golden references, which is a different
claim from the contract this file pins.
"""

import pytest
import torch

from tests.extensions.polar.test_polar_models import (
    _build_minimal_batch,
    _build_minimal_model,
)

N_ATOMS, N_GRAPHS = 2, 1


@pytest.fixture(name="outputs", scope="module")
def fixture_outputs():
    previous = torch.get_default_dtype()
    torch.set_default_dtype(torch.float64)
    try:
        model = _build_minimal_model("cpu", torch.float64)
        batch = _build_minimal_batch("cpu", torch.float64)
        return model(batch, training=False)
    finally:
        torch.set_default_dtype(previous)


@pytest.mark.parametrize(
    "key,kind",
    [
        ("spins", "per_atom_scalar"),
        ("total_charge", "per_graph_scalar"),
        ("fermi_level", "per_graph_scalar"),
        ("external_field", "per_graph_vector"),
    ],
)
def test_the_electrostatic_keys_are_present_with_the_right_extent(outputs, key, kind):
    value = outputs.get(key)

    assert value is not None, f"{key} is missing from the forward's output"
    assert isinstance(value, torch.Tensor)
    if kind == "per_atom_scalar":
        assert value.shape == (N_ATOMS,)
    elif kind == "per_graph_scalar":
        assert value.shape == (N_GRAPHS,)
    else:
        assert value.shape == (N_GRAPHS, 3)
    assert torch.isfinite(value).all(), f"{key} carries non-finite entries"


@pytest.mark.parametrize(
    "key,ndim",
    [("spin_density", 2), ("spin_charge_density", 3), ("charges_history", 4)],
)
def test_the_density_keys_keep_their_rank(outputs, key, ndim):
    """These three are multipole-shaped, so the rank is the contract a consumer
    indexes against. `charges_history` carries the fixed-point iterations in its
    last axis, which is why it has one more."""
    value = outputs[key]

    assert value.ndim == ndim
    assert value.shape[0] == N_ATOMS
    assert torch.isfinite(value).all()


def test_charges_and_spins_are_the_leading_multipole(outputs):
    """`charges` and `spins` are documented as the l=0 component of the density
    coefficients, not independent quantities. If they stop agreeing, one of the
    two readouts changed and the other did not."""
    assert torch.equal(outputs["charges"], outputs["density_coefficients"][:, 0])
    assert torch.equal(outputs["spins"], outputs["spin_density"][:, 0])


def test_electrostatic_potentials_are_absent_unless_asked_for(outputs):
    """`return_electrostatic_potentials` is False on this model, so the key is
    present and None rather than missing: consumers test the value, not the key.
    """
    assert "electrostatic_potentials" in outputs
    assert outputs["electrostatic_potentials"] is None
