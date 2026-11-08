"""Complete CLI objectives against the frozen classes, including gradients.

The oracle ignores property weights in Huber, dipole/polarizability and the
L1/L2 force term. Those terms use unit property weights in the shared-domain
comparison; native missing-label and property-weight support is checked
separately. Stage cases compare the same objective with explicit new weights,
not the legacy loop's implicit choice of a different objective at stage two.
"""

import argparse

import pytest
import torch
from mace_core.config.resolved import ResolvedConfig
from mace_core.observables import DEFAULT_CATALOGUE, resolve_requested
from mace_core.outputs import FIELD_BY_OBSERVABLE, MACEOutput
from mace_torch.cli import migrate
from mace_torch.cli.run_train import parse
from mace_torch.data import TrainingBatch
from mace_torch.train import build_loss

from tests.golden.harness import CLOSED_FORM_FP64

pytestmark = pytest.mark.usefixtures("isolated", "fp64")

PRESETS = {
    "weighted": ("energy", "forces"),
    "forces_only": ("forces",),
    "stress": ("energy", "forces", "stress"),
    "virials": ("energy", "forces", "virials"),
    "huber": ("energy", "forces", "stress"),
    "universal": ("energy", "forces", "stress", "magforces"),
    "l1l2energyforces": ("energy", "forces"),
    "energy_forces_dipole": ("energy", "forces", "dipole"),
    "dipole": ("dipole",),
    "dipole_polar": ("dipole", "polarizability"),
}
WEIGHTS = {
    "energy": 1.7,
    "forces": 0.6,
    "stress": 2.3,
    "virials": 3.1,
    "dipole": 1.9,
    "polarizability": 0.8,
    "magforces": 4.2,
}


def configuration(preset, stage_two, delta):
    flags = ["--loss", preset]
    if preset in {"dipole", "dipole_polar"}:
        flags += ["--model", "dipole" if preset == "dipole" else "dielectric"]
    if preset == "universal":
        flags += ["--model", "magnetic"]
    active_weights = {
        name: value for name, value in WEIGHTS.items() if name in PRESETS[preset]
    }
    for name, value in active_weights.items():
        flags += [f"--{name}_weight", str(value)]
    if preset in {"huber", "universal"}:
        flags += ["--huber_delta", str(delta)]
    if stage_two:
        flags += ["--stage_two", "--start_stage_two", "1", "--max_num_epochs", "2"]
        for name, value in active_weights.items():
            flags += [f"--swa_{name}_weight", str(value * 2.5)]
    return parse(flags)


def fixture_batch(preset, weight_case):
    generator = torch.Generator().manual_seed(19)
    names = set(PRESETS[preset])
    if "forces" in names:
        names.add("energy")  # The derivative's source is also declared.
    shapes = {
        "energy": (3,),
        "forces": (9, 3),
        "stress": (3, 3, 3),
        "virials": (3, 3, 3),
        "dipole": (3, 3),
        "polarizability": (3, 3, 3),
        "magforces": (9, 3),
    }
    targets = {
        name: torch.randn(shapes[name], generator=generator) for name in sorted(names)
    }
    if "forces" in targets:
        # All four Universal bands, at and around their boundaries.
        targets["forces"] = torch.zeros(9, 3)
        targets["forces"][:, 0] = torch.tensor(
            [0.0, 99.0, 100.0, 199.0, 200.0, 299.0, 300.0, 301.0, 900.0]
        )
    predictions = {
        name: (value + torch.linspace(-0.9, 1.1, value.numel()).reshape(value.shape))
        .detach()
        .requires_grad_()
        for name, value in targets.items()
    }
    weights = {}
    for name in names:
        weighted = (
            preset not in {"huber", "dipole", "dipole_polar"}
            and name != "dipole"
            and not (preset == "l1l2energyforces" and name == "forces")
        )
        weights[name] = (
            torch.tensor(
                [0.0, 0.5, 2.0] if weight_case == "masked" else [0.5, 2.0, 1.25]
            )
            if weighted and weight_case in {"property", "masked"}
            else torch.ones(3)
        )
    batch = TrainingBatch(
        graph={
            "ptr": torch.tensor([0, 2, 5, 9]),
            "batch": torch.tensor([0, 0, 1, 1, 1, 2, 2, 2, 2]),
            "weight": torch.ones(3)
            if weight_case == "unit"
            else torch.tensor([0.0, 3.0, 2.0]),
        },
        targets=targets,
        property_weights=weights,
    )
    return batch, predictions


def output(values):
    core = {
        FIELD_BY_OBSERVABLE.get(name, name): value
        for name, value in values.items()
        if name not in {"polarizability", "magforces"}
    }
    return MACEOutput(
        **core,
        extras={
            name: values[name]
            for name in ("polarizability", "magforces")
            if name in values
        },
    )


def parts(batch, predictions):
    for index, (start, stop) in enumerate(((0, 2), (2, 5), (5, 9))):

        def subset(values, start=start, stop=stop, index=index):
            return {
                name: value[start:stop]
                if name in {"forces", "magforces"}
                else value[index : index + 1]
                for name, value in values.items()
            }

        yield (
            TrainingBatch(
                graph={
                    "ptr": torch.tensor([0, stop - start]),
                    "batch": torch.zeros(stop - start, dtype=torch.long),
                    "weight": batch.graph["weight"][index : index + 1],
                },
                targets=subset(batch.targets),
                property_weights={
                    name: value[index : index + 1]
                    for name, value in batch.property_weights.items()
                },
            ),
            output(subset(predictions)),
        )


@pytest.mark.parametrize("preset", PRESETS)
@pytest.mark.parametrize("stage_two", [False, True])
@pytest.mark.parametrize("weight_case", ["unit", "structure", "property", "masked"])
@pytest.mark.parametrize("chunked", [False, True])
def test_complete_cli_objective_and_prediction_gradients(
    preset, stage_two, weight_case, chunked
):
    from mace.tools.scripts_utils import get_loss_fn
    from mace.tools.torch_geometric.data import Data

    delta = 0.4 if stage_two else 0.03
    config = configuration(preset, stage_two, delta)
    # Resolved JSON is the checkpoint contract, including compatibility policy.
    config = ResolvedConfig.from_dict(config.to_resolved_dict())
    requested = resolve_requested(
        config.model.observables, config.catalogue or DEFAULT_CATALOGUE
    )
    loss = build_loss(requested, config.loss, config.schedule()[-1])
    batch, predictions = fixture_batch(preset, weight_case)
    if chunked:
        totals = loss.element_counts(batch)
        actual = sum(
            loss(pred, part, totals) for part, pred in parts(batch, predictions)
        )
    else:
        actual = loss(output(predictions), batch)
    arguments = argparse.Namespace(
        loss=preset,
        huber_delta=delta,
        **{
            f"{name}_weight": value * (2.5 if stage_two else 1.0)
            for name, value in WEIGHTS.items()
        },
    )
    oracle = get_loss_fn(arguments, dipole_only=preset == "dipole", compute_dipole=True)
    expected = oracle(
        Data(
            **batch.graph,
            **batch.targets,
            **{
                f"{name}_weight": value
                for name, value in batch.property_weights.items()
            },
        ),
        predictions,
    )
    tensors = tuple(predictions.values())
    actual_gradients = torch.autograd.grad(
        actual, tensors, retain_graph=True, allow_unused=True
    )
    expected_gradients = torch.autograd.grad(expected, tensors, allow_unused=True)
    for got, want in zip(
        (actual, *actual_gradients), (expected, *expected_gradients), strict=True
    ):
        if want is None:
            assert got is None or torch.count_nonzero(got) == 0
        else:
            torch.testing.assert_close(
                got, want, atol=CLOSED_FORM_FP64.atol, rtol=CLOSED_FORM_FP64.rtol
            )


@pytest.mark.parametrize("preset", PRESETS)
def test_missing_labels_keep_zero_gradient_without_mutating_the_batch(preset):
    import numpy as np
    from mace_core.data.configuration import Configuration
    from mace_core.elements import AtomicNumberTable
    from mace_torch.data import GraphDataset, collate_training, target_specs

    config = configuration(preset, False, 0.4)
    requested = resolve_requested(
        config.model.observables, config.catalogue or DEFAULT_CATALOGUE
    )
    source, _ = fixture_batch(preset, "unit")
    configurations = []
    for index, (start, stop) in enumerate(((0, 2), (2, 5), (5, 9))):
        properties = (
            {
                name: value[start:stop].numpy()
                if name in {"forces", "magforces"}
                else value[index].numpy()
                for name, value in source.targets.items()
            }
            if index != 1
            else {}
        )
        configurations.append(
            Configuration(
                atomic_numbers=np.ones(stop - start, dtype=int),
                positions=np.arange((stop - start) * 3).reshape(-1, 3).astype(float),
                properties=properties,
                weight=(0.0, 3.0, 2.0)[index],
                property_weights=dict.fromkeys(properties, 0.5 + index),
            )
        )
    table = AtomicNumberTable([1])
    dataset = GraphDataset(
        configurations, cutoff=0.5, z_table=table, targets=target_specs(requested)
    )
    batch = collate_training([dataset[i] for i in range(3)], z_table=table)
    targets_before = {name: value.clone() for name, value in batch.targets.items()}
    weights_before = {
        name: value.clone() for name, value in batch.property_weights.items()
    }
    predictions = {
        name: (value + 0.17).requires_grad_() for name, value in batch.targets.items()
    }
    loss = build_loss(requested, config.loss)
    actual = loss(output(predictions), batch)
    gradients = torch.autograd.grad(actual, tuple(predictions.values()))
    changed = {}
    for (name, value), gradient in zip(predictions.items(), gradients, strict=True):
        rows = slice(2, 5) if name in {"forces", "magforces"} else slice(1, 2)
        assert torch.count_nonzero(gradient[rows]) == 0, name
        assert batch.property_weights[name][1] == 0
        changed[name] = value.detach().clone()
        changed[name][rows] += 1000.0
        torch.testing.assert_close(
            batch.targets[name], targets_before[name], atol=0.0, rtol=0.0
        )
        torch.testing.assert_close(
            batch.property_weights[name], weights_before[name], atol=0.0, rtol=0.0
        )
    torch.testing.assert_close(loss(output(changed), batch), actual, atol=0.0, rtol=0.0)


@pytest.mark.parametrize("preset", ["huber", "universal", "l1l2energyforces"])
def test_native_weighting_defaults_remain_linear_in_structure_and_property_weights(
    preset,
):
    from dataclasses import replace

    from mace_core.config.loss import LossConfig
    from pydantic import TypeAdapter

    kind = "l1l2" if preset == "l1l2energyforces" else preset
    requested = resolve_requested(["energy", "forces", "stress"], DEFAULT_CATALOGUE)
    batch, predictions = fixture_batch("huber", "unit")
    loss = build_loss(
        requested, TypeAdapter(LossConfig).validate_python({"kind": kind})
    )
    reference = loss(output(predictions), batch)
    weighted = replace(
        batch,
        graph={**batch.graph, "weight": torch.full((3,), 2.0)},
        property_weights={name: torch.full((3,), 3.0) for name in batch.targets},
    )
    torch.testing.assert_close(
        loss(output(predictions), weighted),
        6.0 * reference,
        atol=CLOSED_FORM_FP64.atol,
        rtol=CLOSED_FORM_FP64.rtol,
    )


@pytest.mark.parametrize("preset", PRESETS)
def test_flags_legacy_yaml_and_resolved_json_select_the_same_objective(
    tmp_path, preset
):
    import json

    import yaml

    values = {"loss": preset}
    if preset in {"dipole", "dipole_polar", "universal"}:
        values["model"] = {
            "dipole": "dipole",
            "dipole_polar": "dielectric",
            "universal": "magnetic",
        }[preset]
    flags = [token for name, value in values.items() for token in (f"--{name}", value)]
    expected = parse(flags)
    path = tmp_path / "legacy.yaml"
    path.write_text(yaml.safe_dump(values))
    converted = tmp_path / "v1.yaml"
    migrate.run(argparse.Namespace(command=[str(path)], output=converted))
    actual = parse(["--config", str(converted)])
    assert actual == expected
    path = tmp_path / "resolved.json"
    path.write_text(json.dumps(actual.to_resolved_dict()))
    assert parse(["--config", str(path)]) == actual
    if preset in {"dipole", "energy_forces_dipole"}:
        assert actual.loss.weights["dipole"] == 100.0
    if preset == "forces_only":
        assert actual.loss.weights["energy"] == 0.0
    if preset == "virials":
        energy = next(
            spec for spec in actual.catalogue.observables if spec.name == "energy"
        )
        strain = next(item for item in energy.derivatives if item.wrt == "strain")
        assert (strain.name, strain.sign, strain.extensive, strain.units) == (
            "virials",
            -1,
            True,
            "eV",
        )


@pytest.mark.parametrize(
    "preset",
    ["forces_only", "stress", "virials", "huber", "universal", "l1l2energyforces"],
)
def test_objective_and_all_mapped_model_gradients_match(preset):
    from mace_torch.physics import DerivativeEngine

    from mace.tools.scripts_utils import get_loss_fn
    from tests.golden.anchors import anchor_batch, load_training_structures
    from tests.parity.anchors import load_anchor
    from tests.parity.test_fm00_training_step import (
        assert_all_parameter_gradients,
        converted,
    )

    legacy = load_anchor("tiny_scaleshift.model").double()
    model, numbers = converted(legacy)
    frames = load_training_structures(limit=3)
    frames[1] = frames[1][:2]
    for index, atoms in enumerate(frames):
        atoms.cell = torch.eye(3).numpy() * 8.0
        atoms.pbc = True
        atoms.info["config_weight"] = (0.0, 3.0, 2.0)[index]
    batch = anchor_batch(legacy, frames)
    config = configuration(preset, False, 0.2)
    # This anchor has no magnetic moment input; Universal's magnetic term is
    # independently compared above with weighted references in all four bands.
    if preset == "universal":
        document = config.to_resolved_dict()
        document["model"]["observables"].remove("magforces")
        document["loss"]["use_structure_weights"].pop("magforces")
        document["loss"]["weights"].pop("magforces")
        config = ResolvedConfig.from_dict(document)
    catalogue = config.catalogue or DEFAULT_CATALOGUE
    requested = resolve_requested(config.model.observables, catalogue)
    prediction = legacy(
        batch.to_dict(),
        training=True,
        compute_force=True,
        compute_stress=True,
        compute_virials=True,
    )
    names = {term.name for term in build_loss(requested, config.loss).terms}
    targets = {name: batch[name] for name in names}
    # Give the strain terms finite nonzero labels on every graph, so a missing
    # target cannot hide an inverted sign or an incorrect volume normalization.
    for name in names & {"stress", "virials"}:
        targets[name] = torch.arange(27).reshape(3, 3, 3) / 100.0
        batch[name] = targets[name]
        batch[f"{name}_weight"] = torch.ones(3)
    arguments = argparse.Namespace(
        loss=preset,
        huber_delta=0.2,
        **{f"{name}_weight": value for name, value in WEIGHTS.items()},
    )
    expected = get_loss_fn(arguments, False, False)(batch, prediction)
    expected.backward()
    native = TrainingBatch(
        graph={
            **batch.to_dict(),
            "cell": batch.cell.reshape(-1, 3, 3),
            "node_atomic_numbers": torch.tensor(numbers)[batch.node_attrs.argmax(-1)],
        },
        targets=targets,
        property_weights={name: batch[f"{name}_weight"] for name in names},
    )
    engine = DerivativeEngine(model, requested.observables[0])
    actual = build_loss(requested, config.loss)(
        engine(native.graph, compute=tuple(requested.derivatives), training=True),
        native,
    )
    actual.backward()
    torch.testing.assert_close(
        actual, expected, atol=CLOSED_FORM_FP64.atol, rtol=CLOSED_FORM_FP64.rtol
    )
    assert_all_parameter_gradients(legacy, model)
