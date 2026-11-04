"""One forward and one backward of a training loss, against the committed golden.

An error table and a loss-decrease check both stay green while a gradient is
wrong by the size of initialisation noise, which is what the committed digest
exists to catch. This is the converted model taking that same step.

Every trainable parameter is checked. Permuted canonical layouts preserve the
stored values; reduced contractions preserve the function through W_new = M W_old.
Their gradients obey the chain rule, g_old = M.T g_new. Comparing in the old
coordinates checks the whole backward without claiming equal parameterizations.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
import torch
from mace_core.config.precision import PrecisionConfig
from mace_core.elements import AtomicNumberTable, ResolvedE0s
from mace_core.observables import ObservableSpec
from mace_torch.backends.reference import ReferenceBackend
from mace_torch.models import EnergyOutputHead, MACEModel, ScaleShiftSpec
from mace_torch.physics import DerivativeEngine

from tests.golden.harness import CLOSED_FORM_FP64, FP64_CPU_REFERENCE
from tests.parity.anchors import load_anchor
from tests.parity.fm00_convert import (
    build_config,
    energy_constants_to_canonical,
    model_arguments,
    recorded_basis,
    transfer_weights,
)
from tests.parity.fm00_projection import ProjectionError
from tests.parity.legacy_weights import (
    canonical_contraction_bases,
    legacy_contraction_blocks,
    legacy_to_reference_map,
    linear_permutation,
    skip_permutation,
)

GOLDEN = Path(__file__).resolve().parents[1] / "golden"
ENERGY = ObservableSpec(
    name="energy",
    irreps="0e",
    per_atom=False,
    units="eV",
    derivatives=[{"wrt": "pos", "name": "forces", "sign": -1, "units": "eV/A"}],
)
CHANNEL_IN = "0e+1o+2e"


def converted(legacy, *, basis="reduced"):
    numbers = [int(z) for z in legacy.atomic_numbers.tolist()]
    config = build_config(legacy)
    values, scale, shift = energy_constants_to_canonical(legacy)
    head = EnergyOutputHead(
        ResolvedE0s(values),
        ["default"],
        AtomicNumberTable(numbers),
        ScaleShiftSpec("std", scale, shift),
        PrecisionConfig(),
    )
    arguments = model_arguments(config)
    arguments["config"] = arguments["config"].model_copy(
        update={"clebsch_gordan_basis": basis}
    )
    model = MACEModel(
        ReferenceBackend(),
        observables=[ENERGY],
        energy_head=head,
        **arguments,
    )
    transfer_weights(legacy, model, config["correlation"])
    return model, numbers


def assert_all_parameter_gradients(legacy, model):
    """Apply the conversion's transpose Jacobian, with complete parameter coverage."""
    names = {id(parameter): name for name, parameter in legacy.named_parameters()}
    checked_legacy, checked_native = set(), set()

    def compare(source, target, gradient):
        assert source.grad is not None, names[id(source)]
        assert target.grad is not None, names[id(source)]
        assert id(source) not in checked_legacy
        assert id(target) not in checked_native
        torch.testing.assert_close(
            gradient,
            source.grad,
            atol=CLOSED_FORM_FP64.atol,
            rtol=CLOSED_FORM_FP64.rtol,
            msg=names[id(source)],
        )
        checked_legacy.add(id(source))
        checked_native.add(id(target))

    def linear(source, target):
        order = linear_permutation(source)
        assert torch.equal(source.weight, target.weight[order])
        compare(source.weight, target.weight, target.weight.grad[order])

    linear(legacy.node_embedding.linear, model.backbone.node_embedding)
    for source, target in zip(
        legacy.interactions, model.backbone.interactions, strict=True
    ):
        linear(source.linear_up, target.body.linear_up)
        linear(source.linear, target.body.linear)
        order = skip_permutation(source.skip_tp)
        assert torch.equal(source.skip_tp.weight, target.skip.weight.flatten()[order])
        compare(
            source.skip_tp.weight,
            target.skip.weight,
            target.skip.weight.grad.flatten()[order],
        )
        for index, weight in enumerate(target.body.radial_mlp.weights):
            original = getattr(source.conv_tp_weights, f"layer{index}").weight
            assert torch.equal(original, weight)
            compare(original, weight, weight.grad)
    for source, target in zip(legacy.products, model.backbone.products, strict=True):
        linear(source.linear, target.linear)
        descriptor = target.contraction.descriptor
        for position, contraction in enumerate(
            source.symmetric_contractions.contractions
        ):
            bases = canonical_contraction_bases(descriptor, position)
            for coupling, original, order in legacy_contraction_blocks(contraction):
                weight = target.contraction.weights[
                    position * descriptor.correlation + order - 1
                ]
                mapping = torch.from_numpy(
                    legacy_to_reference_map(coupling, bases[order - 1], order)
                )
                torch.testing.assert_close(
                    weight,
                    torch.einsum("pq,zpc->zqc", mapping, original),
                    atol=CLOSED_FORM_FP64.atol,
                    rtol=CLOSED_FORM_FP64.rtol,
                )
                compare(
                    original, weight, torch.einsum("pq,zqc->zpc", mapping, weight.grad)
                )
    for source, target in zip(
        legacy.readouts, model.outputs.heads["energy"].readouts, strict=True
    ):
        if hasattr(source, "linear_1"):
            linear(source.linear_1, target.first)
            linear(source.linear_2, target.second)
        else:
            linear(source.linear, target)
    assert checked_legacy == {id(p) for p in legacy.parameters() if p.numel()}
    # Zero-length bias parameters carry no degrees of freedom.
    assert checked_native == {id(p) for p in model.parameters() if p.numel()}


@pytest.mark.parametrize(
    "anchor,reference",
    [
        ("tiny_scaleshift.model", "tiny_scaleshift_train_step_grad_fp64.json"),
        ("tiny_mace.model", "tiny_mace_train_step_grad_fp64.json"),
    ],
)
def test_the_converted_model_takes_the_same_training_step(
    isolated, fp64, anchor, reference
):
    from mace.modules.loss import WeightedEnergyForcesLoss
    from tests.golden.anchors import anchor_batch, load_training_structures
    from tests.golden.train_step import LOSS_WEIGHTS, N_STRUCTURES, _digest, deviations

    legacy = load_anchor(anchor)
    model, numbers = converted(legacy)
    batch = anchor_batch(
        legacy, load_training_structures(limit=N_STRUCTURES), torch.float64
    )
    loss_function = WeightedEnergyForcesLoss(**LOSS_WEIGHTS).to(torch.float64)
    legacy_output = legacy(batch.to_dict(), training=True, compute_force=True)
    legacy_loss = loss_function(batch, legacy_output)
    legacy_loss.backward()
    golden = json.loads((GOLDEN / "references" / reference).read_text())
    gradients = {name: p.grad for name, p in legacy.named_parameters()}
    assert all(gradient is not None for gradient in gradients.values())
    digest = {
        "loss": float(legacy_loss.detach()),
        "global_gradient_norm": float(
            torch.sqrt(sum(g.square().sum() for g in gradients.values()))
        ),
        "parameters": {name: _digest(g) for name, g in gradients.items()},
    }
    problems = deviations(digest, golden, FP64_CPU_REFERENCE)
    assert not problems, "\n".join(problems)

    graph = {
        "positions": batch.positions,
        "node_atomic_numbers": torch.tensor(
            [numbers[index] for index in batch.node_attrs.argmax(1).tolist()]
        ),
        "node_attrs": batch.node_attrs,
        "ptr": batch.ptr,
        "edge_index": batch.edge_index,
        "shifts": batch.shifts,
        "unit_shifts": batch.unit_shifts,
        "cell": batch.cell.reshape(-1, 3, 3),
        "batch": batch.batch,
        "num_graphs": int(batch.num_graphs),
        "head": torch.zeros(int(batch.num_graphs), dtype=torch.long),
    }
    output = DerivativeEngine(model, ENERGY)(graph, compute=("forces",), training=True)
    loss = loss_function(
        batch, {"energy": output.total_energy, "forces": output.forces}
    )

    assert abs(float(loss) - golden["loss"]) < 1e-12, (
        f"the loss is {float(loss)} against a committed {golden['loss']}"
    )

    loss.backward()
    assert_all_parameter_gradients(legacy, model)

    checked = 0
    for name, entry in golden["parameters"].items():
        if "conv_tp_weights.layer" not in name:
            continue
        layer = int(name.split(".")[1])
        index = int(name.split("layer")[1][0])
        gradient = (
            model.backbone.interactions[layer].body.radial_mlp.weights[index].grad
        )
        assert gradient is not None, f"{name} received no gradient"
        flat = gradient.reshape(-1)
        position = torch.arange(flat.numel(), dtype=torch.float64)
        projection = float((flat * torch.cos(position + 1.0)).sum())
        assert abs(projection - entry["projection"]) < 1e-12, (
            f"{name}: the gradient projection is {projection} against a "
            f"committed {entry['projection']}"
        )
        checked += 1

    assert checked == 8, (
        f"only {checked} radial gradients were compared, and there are eight"
    )


def test_the_recorded_basis_is_read_and_not_guessed(fp64):
    """The frozen tree has three settings that disagree about this.

    A flag that defaults one way, a model class that defaults the other, and a
    wrapper that forces it back when a library is missing. A converter that
    picks a default is picking which of three models comes out.
    """
    legacy = load_anchor("tiny_scaleshift.model")
    correlation = build_config(legacy)["correlation"]
    for product, targets in zip(legacy.products, (["0e", "1o"], ["0e"]), strict=True):
        for position, contraction in enumerate(
            product.symmetric_contractions.contractions
        ):
            assert (
                recorded_basis(contraction, CHANNEL_IN, targets[position], correlation)
                == "full"
            )


@pytest.mark.parametrize("anchor", ["tiny_scaleshift.model", "tiny_mace.model"])
def test_the_converted_stress_path_differentiates_twice(isolated, fp64, anchor):
    from tests.golden.harness import load_fixtures
    from tests.parity.test_backbone_parity import _v1_graph

    legacy = load_anchor(anchor)
    model, _ = converted(legacy)
    atoms = load_fixtures(names=["triclinic_bulk"])["triclinic_bulk"]
    graph = _v1_graph(legacy, atoms)
    engine = DerivativeEngine(model, ENERGY)

    def energy(displacement):
        output = engine(
            {**graph, "displacement": displacement},
            compute=("stress",),
            training=True,
        )
        assert torch.isfinite(output.stress).all()
        return output.total_energy

    strain = torch.zeros((1, 3, 3), requires_grad=True)
    (gradient,) = torch.autograd.grad(energy(strain).sum(), strain)
    assert torch.count_nonzero(gradient), "the supplied strain must affect the energy"
    options = {
        "eps": 1e-6,
        "atol": FP64_CPU_REFERENCE.atol,
        "rtol": FP64_CPU_REFERENCE.rtol,
    }
    assert torch.autograd.gradcheck(energy, (strain,), **options)
    assert torch.autograd.gradgradcheck(energy, (strain,), **options)


def test_a_basis_that_is_neither_is_refused(fp64):
    """Rather than converted into something that runs and is wrong."""

    class Unrecognised:
        def __init__(self):
            self.weights_max = torch.zeros(3, 7, 16)
            self.weights = [torch.zeros(3, 2, 16), torch.zeros(3, 1, 16)]

    with pytest.raises(ProjectionError, match="written against neither"):
        recorded_basis(Unrecognised(), CHANNEL_IN, "0e", 3)
