"""The dipole families on the v1 backbone and output layer, against the frozen
classes, in one process.

The three classes are configurations here and nothing else:

* `AtomicDipolesMACE` declares a structure's dipole, keeps the vectors in its
  last layer, and adds the dipole of the structure's own charges in Debye.
* `AtomicDielectricMACE` reads out charges, dipole and polarizability through
  one readout, keeps every degree in its last layer, shifts the predicted
  charges to the structure's total charge and adds their dipole in e*A, and
  reports the polarizability as a matrix too.
* `EnergyDipolesMACE` reads out the energy and the dipole through one readout.

Every legacy weight the two layers rebuild is carried into them. The position
Jacobians of the dipole and the polarizability are taken around the model, as
a derivative engine takes them. Values are compared at the ``closed_form_fp64``
row against the live model, and the dielectric ones at the
``fp64_cpu_reference`` row against its committed reference as well.
"""

import numpy as np
import pytest
import torch
from mace_core.clebsch_gordan import symmetric_matrix_basis
from mace_core.clebsch_gordan.irreps import Irreps
from mace_core.config import (
    BesselBasisConfig,
    ModelConfig,
    PolynomialCutoffConfig,
    ReadoutConfig,
)
from mace_core.observables import InputSpec, ObservableCatalogue, ObservableSpec

from mace import modules
from mace.tools.torch_tools import get_change_of_basis
from tests.golden import anchors, build_dielectric_anchor, build_dipole_anchor, harness
from tests.golden.targets import dielectric
from tests.parity.legacy_models import ANCHOR_CONFIG, V1Model, live_model, perturb

CLOSED_FORM = harness.tolerance("closed_form_fp64")
FP64 = harness.tolerance("fp64_cpu_reference")
FIXTURES = ("water_cluster", "dimer_short", "isolated_atom")

DIPOLE = ObservableSpec(name="dipole", irreps="1o", per_atom=False, units="Debye")
CHARGES = ObservableSpec(name="charges", irreps="0e", per_atom=True, units="e")
POLARIZABILITY = ObservableSpec(
    name="polarizability", irreps="0e+2e", per_atom=False, units="Debye*A/V"
)
ENERGY = ObservableSpec(name="energy", irreps="0e", per_atom=False, units="eV")
CHARGES_INPUT = InputSpec(name="charges", irreps="0e", per_atom=True, units="e")
TOTAL_CHARGE = InputSpec(name="total_charge", irreps="0e", per_atom=False, units="e")


@pytest.fixture(name="fixtures", scope="module")
def fixture_fixtures():
    return harness.load_fixtures(names=FIXTURES)


def _model_config(anchor_config, **model) -> ModelConfig:
    """The v1 configuration of a committed anchor's recipe."""
    hidden = Irreps.parse(anchor_config["hidden_irreps"].replace(" ", ""))
    (num_channels,) = {mul for mul, _ in hidden.terms}
    return ModelConfig(
        r_max=anchor_config["r_max"],
        num_interactions=anchor_config["num_interactions"],
        num_channels=num_channels,
        max_L=max(ir.degree for _, ir in hidden.terms),
        max_ell=anchor_config["max_ell"],
        correlation=anchor_config["correlation"],
        radial_basis=BesselBasisConfig(num_basis=anchor_config["num_bessel"]),
        cutoff=PolynomialCutoffConfig(
            polynomial_order=anchor_config["num_polynomial_cutoff"]
        ),
        radial_mlp=tuple(anchor_config.get("radial_MLP", (64, 64, 64))),
        interaction_first="plain",
        interaction="residual",
        **model,
    )


def _jacobian(values: torch.Tensor, positions: torch.Tensor) -> torch.Tensor:
    """``[k, n_atoms, 3]``: every component of ``values`` against every
    coordinate."""
    rows = [
        torch.autograd.grad(component, positions, retain_graph=True)[0]
        for component in values.reshape(-1)
    ]
    return torch.stack(rows)


def _close(ours, theirs, row, what):
    torch.testing.assert_close(
        ours.detach(),
        theirs.detach().to(ours.dtype),
        atol=row.atol,
        rtol=row.rtol,
        msg=lambda message: f"{what}: {message}",
    )


# ---------------------------------------------------------------------------
# The dipole model
# ---------------------------------------------------------------------------


def _dipole_v1():
    legacy = anchors.load_checkpoint(build_dipole_anchor.MODEL_PATH)
    config = _model_config(
        build_dipole_anchor.ANCHOR_CONFIG,
        last_layer_irreps="1o",
        observables=ObservableCatalogue(inputs=(CHARGES_INPUT,), observables=(DIPOLE,)),
        readout=ReadoutConfig(
            mlp_irreps="8x0e",
            dipole_charges="input",
            atomic_contributions={"dipole": "atomic_dipoles"},
        ),
    )
    return legacy, V1Model(legacy, config)


def test_a_dipole_model_is_reproduced_with_and_without_charges(fp64, fixtures):
    """The fixtures carry no charges, so the charge part is also checked on
    the water cluster with charges set: the frozen class reads them from the
    structure and converts their dipole to Debye."""
    legacy, v1 = _dipole_v1()
    assert v1.outputs.readouts["dipole"].layers == (0,)
    for name in FIXTURES:
        graph = anchors.anchor_graph(legacy, fixtures[name], torch.float64)
        if name == "water_cluster":
            graph["charges"] = torch.tensor(
                [0.4, -0.2, -0.2, 0.3, -0.15, -0.15, 0.5, -0.25, -0.25],
                dtype=torch.float64,
            )
        expected = legacy(dict(graph), compute_force=False)
        output, _ = v1(graph)
        _close(output.dipole, expected["dipole"], CLOSED_FORM, f"{name} dipole")
        _close(
            output.extras["atomic_dipoles"],
            expected["atomic_dipoles"],
            CLOSED_FORM,
            f"{name} atomic dipoles",
        )


def test_a_dipole_model_matches_its_committed_reference(fp64, fixtures):
    legacy, v1 = _dipole_v1()
    reference = harness.load_reference(
        harness.REFERENCES_DIR / "tiny_dipoles_e3nn_cpu_fp64.json"
    )
    assert set(reference["fixtures"]) == set(FIXTURES)
    for name in FIXTURES:
        graph = anchors.anchor_graph(legacy, fixtures[name], torch.float64)
        output, _ = v1(graph)
        committed = reference["fixtures"][name]["outputs"]
        for channel, value in (
            ("dipole", output.dipole[0]),
            ("atomic_dipoles", output.extras["atomic_dipoles"]),
        ):
            expected = torch.tensor(
                np.asarray(committed[channel]["value"], dtype=float)
            )
            _close(value, expected, FP64, f"{name} {channel}")


# ---------------------------------------------------------------------------
# The dielectric model
# ---------------------------------------------------------------------------


def _dielectric_v1():
    legacy = anchors.load_checkpoint(build_dielectric_anchor.MODEL_PATH)
    config = _model_config(
        build_dielectric_anchor.ANCHOR_CONFIG,
        last_layer_irreps="0e+1o+2e",
        observables=ObservableCatalogue(
            inputs=(TOTAL_CHARGE,),
            observables=(
                CHARGES,
                DIPOLE.model_copy(update={"units": "e*A"}),
                POLARIZABILITY,
            ),
        ),
        readout=ReadoutConfig(
            mlp_irreps="4x0e+4x1o+4x2e",
            groups=(("charges", "polarizability", "dipole"),),
            dipole_charges="predicted",
            symmetric_matrices=("polarizability",),
            atomic_contributions={"dipole": "atomic_dipoles"},
        ),
    )
    return legacy, V1Model(legacy, config)


def _dielectric_outputs(v1, graph):
    output, positions = v1(graph)
    extras = output.extras
    return {
        "charges": extras["charges"],
        "dipole": output.dipole[0],
        "atomic_dipoles": extras["atomic_dipoles"],
        "polarizability": extras["polarizability"][0],
        "polarizability_sh": extras["polarizability_sh"][0],
        "dmu_dr": _jacobian(output.dipole[0], positions),
        "dalpha_dr": _jacobian(extras["polarizability"][0], positions),
    }


@pytest.mark.parametrize("total_charge", [0.0, 1.0])
def test_a_dielectric_model_is_reproduced_and_so_are_its_jacobians(
    fp64, fixtures, total_charge
):
    legacy, v1 = _dielectric_v1()
    for name in FIXTURES:
        graph = anchors.anchor_graph(legacy, fixtures[name], torch.float64)
        graph["total_charge"] = torch.full_like(graph["total_charge"], total_charge)
        expected = legacy(dict(graph), compute_dielectric_derivatives=True)
        ours = _dielectric_outputs(v1, graph)
        for channel, value in ours.items():
            theirs = expected[channel]
            if channel in ("dipole", "polarizability", "polarizability_sh"):
                theirs = theirs[0]
            _close(value, theirs, CLOSED_FORM, f"{name} {channel}")


def test_a_dielectric_model_matches_its_committed_reference(fp64, fixtures):
    legacy, v1 = _dielectric_v1()
    reference = harness.load_reference(
        harness.REFERENCES_DIR / dielectric.REFERENCE_NAME
    )
    for name in FIXTURES:
        graph = anchors.anchor_graph(legacy, fixtures[name], torch.float64)
        ours = _dielectric_outputs(v1, graph)
        committed = reference["fixtures"][name]["outputs"]
        for channel, value in ours.items():
            expected = torch.tensor(
                np.asarray(committed[channel]["value"], dtype=float)
            )
            _close(value, expected, FP64, f"{name} {channel}")


def test_the_matrix_basis_is_the_frozen_trees(fp64):
    torch.testing.assert_close(
        torch.tensor(symmetric_matrix_basis()),
        get_change_of_basis().double(),
        atol=CLOSED_FORM.atol,
        rtol=CLOSED_FORM.rtol,
    )


# ---------------------------------------------------------------------------
# The energy and dipole model
# ---------------------------------------------------------------------------


def test_an_energy_and_dipole_model_is_reproduced(fp64, fixtures):
    legacy = perturb(
        live_model(modules.EnergyDipolesMACE, seed=31, mlp_irreps="8x0e + 8x1o"),
        seed=32,
    )
    config = ModelConfig(
        **ANCHOR_CONFIG | {"num_channels": 8, "correlation": 2},
        interaction_first="plain",
        interaction="residual",
        last_layer_irreps="0e+1o",
        observables=ObservableCatalogue(
            inputs=(CHARGES_INPUT,), observables=(ENERGY, DIPOLE)
        ),
        readout=ReadoutConfig(
            mlp_irreps="8x0e+8x1o",
            groups=(("energy", "dipole"),),
            dipole_charges="input",
            atomic_contributions={"dipole": "atomic_dipoles"},
        ),
    )
    v1 = V1Model(legacy, config)
    for name in FIXTURES:
        graph = anchors.anchor_graph(legacy, fixtures[name], torch.float64)
        graph["charges"] = torch.linspace(
            -0.3, 0.3, graph["positions"].shape[0], dtype=torch.float64
        )
        expected = legacy(dict(graph), compute_force=False)
        output, _ = v1(graph)
        _close(output.total_energy, expected["energy"], CLOSED_FORM, f"{name} energy")
        _close(output.dipole, expected["dipole"], CLOSED_FORM, f"{name} dipole")
        _close(
            output.extras["atomic_dipoles"],
            expected["atomic_dipoles"],
            CLOSED_FORM,
            f"{name} atomic dipoles",
        )
