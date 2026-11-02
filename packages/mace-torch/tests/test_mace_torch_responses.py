"""Declared responses against geometry, node features and graph features."""

from __future__ import annotations

import itertools

import pytest
import torch
from conftest import fp64_only
from mace_core.observables import DerivativeRequest, InputSpec, ObservableSpec
from mace_core.outputs import MACEOutput
from mace_torch.physics import DerivativeEngine

from tests.golden.harness import CLOSED_FORM_FP64, FP64_CPU_REFERENCE


def graph():
    return {
        "positions": torch.tensor(
            [[0.2, 0.3, -0.1], [0.7, -0.2, 0.4], [-0.3, 0.5, 0.1]]
        ),
        "batch": torch.tensor([0, 1, 1]),
        "ptr": torch.tensor([0, 1, 3]),
        "cell": torch.eye(3).repeat(2, 1, 1) * 2,
        "pbc": torch.tensor([[False] * 3, [True] * 3]),
        "edge_index": torch.empty(2, 0, dtype=torch.long),
        "shifts": torch.empty(0, 3),
        "unit_shifts": torch.empty(0, 3),
        "drive": torch.tensor([[0.1, 0.4], [0.2, -0.3]]),
        "site_feature": torch.tensor(
            [[0.3, 0.4, 0.5], [0.6, -0.2, 0.8], [0.9, 0.7, -0.1]]
        ),
    }


INPUTS = (
    InputSpec(
        name="drive", irreps="2x0e", per_atom=False, units="test", differentiable=True
    ),
    InputSpec(
        name="site_feature",
        irreps="1o",
        per_atom=True,
        units="test",
        differentiable=True,
    ),
)


class ResponseModel(torch.nn.Module):
    def __init__(self, spec):
        super().__init__()
        self.spec = spec
        self.weight = torch.nn.Parameter(torch.tensor(0.7))

    def forward(self, data):
        batch = data["batch"]
        site = self.weight * (
            data["positions"].square().sum(-1)
            + data["site_feature"].pow(3).sum(-1)
            + data["drive"].square().sum(-1)[batch]
            + 0.1 * data["cell"].square().sum((-1, -2))[batch]
        )
        energy = site.new_zeros(2).index_add(0, batch, site)
        value = site if self.spec.per_atom else energy
        if not self.spec.is_scalar:
            value = torch.stack((value, 0.3 * value.square()), dim=-1)
        return MACEOutput(total_energy=energy, extras={self.spec.name: value})


def specification(wrt, per_atom=False, scalar=False):
    return ObservableSpec(
        name="signal",
        irreps="0e" if scalar else "2x0e",
        per_atom=per_atom,
        units="test",
        derivatives=({"wrt": wrt, "name": "response", "sign": -1},),
    )


@fp64_only
@pytest.mark.parametrize("scalar", [False, True])
@pytest.mark.parametrize(
    "wrt,per_atom",
    [
        ("pos", False),
        ("strain", False),
        ("drive", False),
        ("site_feature", False),
        ("strain", True),
        ("drive", True),
    ],
)
def test_responses_match_finite_differences_without_mixing_structures(
    wrt, per_atom, scalar
):
    spec = specification(wrt, per_atom, scalar)
    model = ResponseModel(spec)
    engine = DerivativeEngine(model, None, inputs=INPUTS, responses=[spec])
    data = graph()
    before = {key: value.clone() for key, value in data.items()}
    response = engine(data, compute=("response",), training=True).get("response")
    target = (
        data["cell"] if wrt == "strain" else data["positions" if wrt == "pos" else wrt]
    )
    components = 1 if scalar else 2
    rows = 3 if per_atom or wrt in {"pos", "site_feature"} else 2
    expected = response.new_zeros(components, rows, *target.shape[1:])
    step = 1e-5
    for index in itertools.product(*(range(size) for size in target.shape)):
        samples = []
        for offset in (-step, step):
            moved = dict(data)
            if wrt == "strain":
                strain = torch.zeros_like(target)
                strain[index] = offset
                strain = (strain + strain.transpose(-1, -2)) / 2
                moved["positions"] = data["positions"] + torch.einsum(
                    "ni,nij->nj", data["positions"], strain[data["batch"]]
                )
                moved["cell"] = data["cell"] + data["cell"] @ strain
            else:
                key = "positions" if wrt == "pos" else wrt
                moved[key] = data[key].clone()
                moved[key][index] += offset
            samples.append(model(moved).get("signal").reshape(-1, components))
        difference = (samples[1] - samples[0]) / (2 * step)
        if per_atom:
            selected = data["batch"] == index[0]
            expected[(slice(None), selected, *index[1:])] = -difference[selected].T
        else:
            structure = (
                data["batch"][index[0]] if wrt in {"pos", "site_feature"} else index[0]
            )
            expected[(slice(None), *index)] = -difference[structure]
    torch.testing.assert_close(
        response,
        expected[0] if scalar else expected,
        atol=FP64_CPU_REFERENCE.atol,
        rtol=FP64_CPU_REFERENCE.rtol,
    )
    response.square().sum().backward()
    assert model.weight.grad is not None
    assert torch.isfinite(model.weight.grad) and model.weight.grad.abs() > 0
    for key, value in before.items():
        assert torch.equal(data[key], value) and not data[key].requires_grad


@fp64_only
def test_responses_and_energy_derivatives_share_the_forward_and_keep_training_graph():
    spec = specification("drive")
    spec = spec.model_copy(
        update={"derivatives": (*spec.derivatives, DerivativeRequest(wrt="strain"))}
    )
    energy = ObservableSpec(name="energy", irreps="0e", per_atom=False, units="eV")
    model = ResponseModel(spec)
    engine = DerivativeEngine(model, energy, inputs=INPUTS, responses=[spec])
    data = graph()
    names = ("forces", "stress", "d_signal_d_strain", "response", "d_energy_d_drive")
    result = engine(data, compute=names, training=True)
    loss = sum(result.get(name).square().sum() for name in names)
    loss.backward()
    assert model.weight.grad is not None
    assert torch.isfinite(model.weight.grad) and model.weight.grad.abs() > 0
    for name in names:
        separate = engine(data, compute=(name,)).get(name)
        torch.testing.assert_close(
            result.get(name),
            separate,
            atol=CLOSED_FORM_FP64.atol,
            rtol=CLOSED_FORM_FP64.rtol,
        )


@fp64_only
@pytest.mark.parametrize("wrt", ["drive", "strain"])
def test_response_double_backward(wrt):
    spec = specification(wrt)
    engine = DerivativeEngine(
        ResponseModel(spec), None, inputs=INPUTS, responses=[spec]
    )
    data = graph()
    key = "displacement" if wrt == "strain" else wrt
    value = (
        torch.zeros_like(data["cell"]) if wrt == "strain" else data[key]
    ).requires_grad_()

    def response(value):
        return engine({**data, key: value}, compute=("response",), training=True).get(
            "response"
        )

    assert torch.autograd.gradcheck(
        response, (value,), atol=FP64_CPU_REFERENCE.atol, rtol=FP64_CPU_REFERENCE.rtol
    )
    assert torch.autograd.gradgradcheck(
        response, (value,), atol=FP64_CPU_REFERENCE.atol, rtol=FP64_CPU_REFERENCE.rtol
    )


@fp64_only
def test_missing_and_nondifferentiable_response_inputs_are_errors():
    spec = specification("drive")
    model = ResponseModel(spec)
    with pytest.raises(ValueError, match=r"response.*undeclared.*drive"):
        DerivativeEngine(model, None, responses=[spec])
    engine = DerivativeEngine(model, None, inputs=INPUTS, responses=[spec])
    data = graph()
    del data["drive"]
    with pytest.raises(KeyError, match=r"response.*drive"):
        engine(data, compute=("response",))
    fixed = INPUTS[0].model_copy(update={"differentiable": False})
    engine = DerivativeEngine(model, None, inputs=[fixed], responses=[spec])
    with pytest.raises(ValueError, match=r"response.*drive.*differentiable"):
        engine(graph(), compute=("response",))


@fp64_only
def test_atom_to_atom_responses_are_refused_at_construction():
    spec = specification("pos", per_atom=True)
    with pytest.raises(ValueError, match="pair of atoms"):
        DerivativeEngine(ResponseModel(spec), None, responses=[spec])


@fp64_only
@pytest.mark.parametrize("training", [False, True])
def test_only_training_retains_the_response_graph(training):
    spec = specification("drive")
    engine = DerivativeEngine(
        ResponseModel(spec), None, inputs=INPUTS, responses=[spec]
    )
    result = engine(graph(), compute=("response",), training=training)
    if training:
        result.get("signal").sum().backward()
        assert engine.backbone.get_parameter("weight").grad is not None
    else:
        with pytest.raises(RuntimeError, match="second time"):
            result.get("signal").sum().backward()


@fp64_only
def test_a_zero_input_is_present_and_an_unused_target_has_zero_response():
    spec = specification("drive")

    class ConstantModel(torch.nn.Module):
        def forward(self, data):
            return MACEOutput(extras={"signal": data["positions"].new_ones(2, 2)})

    engine = DerivativeEngine(ConstantModel(), None, inputs=INPUTS, responses=[spec])
    data = graph()
    data["drive"].zero_()
    response = engine(data, compute=("response",), training=True).get("response")
    assert response.shape == (2, 2, 2) and torch.count_nonzero(response) == 0
    response.sum().backward()


@fp64_only
@pytest.mark.parametrize("wrt,per_atom", [("drive", False), ("strain", True)])
def test_unpadding_keeps_all_components_of_each_real_response(wrt, per_atom):
    from mace_core.graph import PaddingInfo
    from mace_core.observables import DEFAULT_CATALOGUE, ObservableCatalogue
    from mace_torch.graph import unpad_outputs

    spec = specification(wrt, per_atom)
    catalogue = ObservableCatalogue(
        inputs=(*DEFAULT_CATALOGUE.inputs, *INPUTS), observables=(spec,)
    )
    engine = DerivativeEngine(
        ResponseModel(spec), None, inputs=INPUTS, responses=[spec]
    )
    output = engine(graph(), compute=("response",))
    real = unpad_outputs(output, PaddingInfo(1, 0, 1), catalogue=catalogue)
    torch.testing.assert_close(
        real.get("response"), output.get("response")[:, :1], atol=0, rtol=0
    )
