"""One forward and one backward of a training loss, against the committed golden.

An error table and a loss-decrease check both stay green while a gradient is
wrong by the size of initialisation noise, which is what the committed digest
exists to catch. This is the converted model taking that same step.

**What can be compared, and what cannot.** The loss is a function of the
model's outputs, so it is the same number whatever the parameters are called.
The per-parameter gradients are not: the conversion deliberately changes the
parametrization, projecting the symmetric contraction onto the reduced basis,
so its gradients live in a different space and comparing them would be
comparing weights, which the conversion never claims to preserve. The radial
networks are the part that transfers verbatim, tensor for tensor, so their
gradients are comparable and they are what is checked here.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import pytest
import torch
from fm00_convert import (
    build_config,
    energy_constants_to_canonical,
    recorded_basis,
    transfer_weights,
)
from fm00_projection import ProjectionError

from mace_core.elements import AtomicNumberTable, ResolvedE0s
from mace_core.kernels.precision import PrecisionConfig
from mace_core.observables import ObservableSpec
from mace_torch.backends.reference import ReferenceBackend
from mace_torch.models import EnergyOutputHead, MACEModel, ScaleShiftSpec
from mace_torch.physics import DerivativeEngine

GOLDEN = Path(__file__).resolve().parents[1] / "golden"
ENERGY = ObservableSpec(
    name="energy", irreps="0e", per_atom=False, units="eV", normalization="none"
)
CHANNEL_IN = "0e+1o+2e"


def load_anchor(name: str):
    previous = os.environ.get("TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD")
    os.environ["TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD"] = "1"
    try:
        return torch.load(
            GOLDEN / "models" / name, map_location="cpu", weights_only=False
        )
    finally:
        if previous is None:
            os.environ.pop("TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD", None)
        else:
            os.environ["TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD"] = previous


def converted(legacy):
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
    model = MACEModel(
        ReferenceBackend(), observables=[ENERGY], energy_head=head, **config
    )
    transfer_weights(legacy, model, config["correlation"])
    return model, numbers


@pytest.mark.parametrize(
    "anchor,reference",
    [
        ("tiny_scaleshift.model", "tiny_scaleshift_train_step_grad_fp64.json"),
        ("tiny_mace.model", "tiny_mace_train_step_grad_fp64.json"),
    ],
)
def test_the_converted_model_takes_the_same_training_step(fp64, anchor, reference):
    from mace.modules.loss import WeightedEnergyForcesLoss

    from tests.golden.anchors import anchor_batch, load_training_structures
    from tests.golden.train_step import LOSS_WEIGHTS, N_STRUCTURES

    legacy = load_anchor(anchor)
    model, numbers = converted(legacy)
    batch = anchor_batch(
        legacy, load_training_structures(limit=N_STRUCTURES), torch.float64
    )

    graph = {
        "positions": batch.positions,
        "atomic_numbers": torch.tensor(
            [numbers[index] for index in batch.node_attrs.argmax(1).tolist()]
        ),
        "element_index": batch.node_attrs.argmax(1),
        "edge_index": batch.edge_index,
        "shifts": batch.shifts,
        "unit_shifts": batch.unit_shifts,
        "cell": batch.cell.reshape(-1, 3, 3),
        "batch": batch.batch,
        "num_graphs": int(batch.num_graphs),
        "head": torch.zeros(int(batch.num_graphs), dtype=torch.long),
    }
    output = DerivativeEngine(model)(graph, compute=("forces",), training=True)
    loss = WeightedEnergyForcesLoss(**LOSS_WEIGHTS).to(torch.float64)(
        batch, {"energy": output.total_energy, "forces": output.forces}
    )

    golden = json.loads((GOLDEN / "references" / reference).read_text())
    assert abs(float(loss) - golden["loss"]) < 1e-12, (
        f"the loss is {float(loss)} against a committed {golden['loss']}"
    )

    loss.backward()

    checked = 0
    for name, entry in golden["parameters"].items():
        if "conv_tp_weights.layer" not in name:
            continue
        layer = int(name.split(".")[1])
        index = int(name.split("layer")[1][0])
        gradient = model.backbone.interactions[layer].body.radial.weights[index].grad
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
                recorded_basis(
                    contraction, CHANNEL_IN, targets[position], correlation
                )
                == "full"
            )


def test_a_basis_that_is_neither_is_refused(fp64):
    """Rather than converted into something that runs and is wrong."""

    class Unrecognised:
        weights_max = torch.zeros(3, 7, 16)
        weights = [torch.zeros(3, 2, 16), torch.zeros(3, 1, 16)]

    with pytest.raises(ProjectionError, match="written against neither"):
        recorded_basis(Unrecognised(), CHANNEL_IN, "0e", 3)
