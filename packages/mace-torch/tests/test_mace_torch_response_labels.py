"""Response labels are row-first; engine responses retain component-first layout."""

import numpy as np
import pytest
import torch
from ase import Atoms
from ase.io import write
from conftest import fp64_only
from mace_core.config.loss import LossConfig
from mace_core.config.resolved import ResolvedConfig
from mace_core.data import Configuration
from mace_core.elements import AtomicNumberTable
from mace_core.observables import (
    DEFAULT_CATALOGUE,
    DerivativeRequest,
    InputSpec,
    ObservableCatalogue,
    ObservableSpec,
    resolve_requested,
)
from mace_core.outputs import MACEOutput
from mace_torch.cli.run_train import run
from mace_torch.data import GraphDataset, collate_training, target_specs
from mace_torch.data.graphs import targets_from_configuration
from mace_torch.deploy.loader import load_deployed
from mace_torch.finetune.stages import build
from mace_torch.train.loop import evaluate
from mace_torch.train.loss import build_loss
from mace_torch.train.metrics import RunningMetrics, metric_specs

from tests.golden.harness import CLOSED_FORM_FP64, EXACT


@pytest.fixture
def deterministic_execution(monkeypatch):
    """A bitwise reload check requires the same CUDA reduction order."""
    monkeypatch.setenv("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    previous = torch.are_deterministic_algorithms_enabled()
    previous_warn_only = torch.is_deterministic_algorithms_warn_only_enabled()
    torch.use_deterministic_algorithms(True)
    try:
        yield
    finally:
        torch.use_deterministic_algorithms(previous, warn_only=previous_warn_only)


def declarations(
    irreps="2e",
    per_atom=False,
    wrt="pos",
    input_irreps="1o",
    input_per_atom=True,
    extensive=False,
):
    return ObservableCatalogue(
        inputs=(
            InputSpec(
                name=wrt,
                irreps=input_irreps,
                per_atom=input_per_atom,
                units="input-unit",
                differentiable=True,
            ),
        ),
        observables=(
            ObservableSpec(
                name="q",
                irreps=irreps,
                per_atom=per_atom,
                units="q-unit",
                derivatives=(DerivativeRequest(wrt=wrt, extensive=extensive),),
            ),
        ),
    )


def requested_for(catalogue):
    observable = catalogue.observable("q")
    name = observable.derivative_name(catalogue.inputs[0].name)
    return resolve_requested(("q", name), catalogue), name


@pytest.mark.parametrize(
    "irreps,per_atom,wrt,input_irreps,input_per_atom,shape,atom,axis",
    [
        ("2e", False, "pos", "1o", True, (5, 3), True, 1),
        ("2e", False, "strain", "0e+2e", False, (5, 3, 3), False, 1),
        ("2e", False, "feature", "2x0e", False, (5, 2), False, 1),
        ("0e", True, "feature", "0e", False, (), True, 0),
        ("0e+1o", True, "feature", "0e", False, (4,), True, 1),
        ("0e", False, "feature", "2x0e", True, (2,), True, 0),
    ],
)
def test_label_shapes_and_rows_come_from_both_sides_of_the_derivative(
    irreps, per_atom, wrt, input_irreps, input_per_atom, shape, atom, axis
):
    requested, name = requested_for(
        declarations(irreps, per_atom, wrt, input_irreps, input_per_atom)
    )
    layout = {spec.name: spec for spec in target_specs(requested)}[name]
    assert (layout.value_shape, layout.per_atom, layout.output_row_axis) == (
        shape,
        atom,
        axis,
    )


@pytest.mark.parametrize("name,shape", [("dmu_dr", (3, 3)), ("dalpha_dr", (9, 3))])
def test_existing_cartesian_dielectric_responses_keep_their_component_counts(
    name, shape
):
    requested = resolve_requested((name,), DEFAULT_CATALOGUE)
    layout = {spec.name: spec for spec in target_specs(requested)}[name]
    assert layout.value_shape == shape
    assert layout.per_atom and layout.output_row_axis == 1


def labelled_batch(catalogue):
    requested, name = requested_for(catalogue)
    layouts = target_specs(requested)
    layout = next(spec for spec in layouts if spec.name == name)
    assert layout.value_shape is not None
    configurations = []
    for index, count in enumerate((2, 3)):
        shape = (count, *layout.value_shape) if layout.per_atom else layout.value_shape
        target = np.arange(np.prod(shape) or 1, dtype=float).reshape(shape)
        if layout.per_atom:
            target = target.reshape(count, -1)  # extxyz atom columns are flat.
        configurations.append(
            Configuration(
                atomic_numbers=np.ones(count, dtype=int),
                positions=np.column_stack((np.arange(count), np.zeros((count, 2)))),
                properties={"q": None, name: target if index == 0 else None},
                property_weights={name: 0.5},
                weight=2.0,
            )
        )
    dataset = GraphDataset(
        configurations, cutoff=1.5, z_table=AtomicNumberTable([1]), targets=layouts
    )
    return (
        requested,
        name,
        layout,
        collate_training([dataset[0], dataset[1]], z_table=AtomicNumberTable([1])),
    )


@fp64_only
@pytest.mark.parametrize(
    "wrt,per_atom,extensive",
    [
        ("pos", False, False),
        ("pos", False, True),
        ("strain", False, False),
        ("strain", False, True),
        ("feature", True, False),
        ("feature", True, True),
    ],
)
def test_loss_and_metrics_mask_rows_without_reordering_components(
    wrt, per_atom, extensive
):
    catalogue = declarations(
        per_atom=per_atom,
        wrt=wrt,
        input_irreps="1o" if wrt == "pos" else "0e+2e" if wrt == "strain" else "0e",
        input_per_atom=wrt == "pos",
        extensive=extensive,
    )
    requested, name, layout, batch = labelled_batch(catalogue)
    reference = batch.targets[name]
    row_count = 2 if layout.per_atom else 1
    values = reference.clone()
    values[row_count:] = 1000.0
    response = values.movedim(0, layout.output_row_axis).contiguous().requires_grad_()
    output = MACEOutput(extras={"q": batch.targets["q"], name: response})
    loss = build_loss(requested, LossConfig(weights={"q": 0.0, name: 1.0}))
    # Five components and five atoms: the old reshape silently scrambled them.
    assert loss(output, batch).item() == 0.0
    with torch.no_grad():
        response.movedim(layout.output_row_axis, 0)[:row_count] += 2.0
    actual = loss(output, batch)
    expected = (
        (0.4 if extensive else 1.6) if layout.per_atom else (0.5 if extensive else 2.0)
    )
    assert actual.item() == pytest.approx(
        expected, abs=CLOSED_FORM_FP64.atol, rel=CLOSED_FORM_FP64.rtol
    )
    actual.backward()
    gradients = response.grad.movedim(layout.output_row_axis, 0)
    assert torch.count_nonzero(gradients[:row_count])
    assert not torch.count_nonzero(gradients[row_count:])
    metrics = RunningMetrics(metric_specs(requested), loss)
    metrics.update(output, batch)
    measured = metrics.compute()
    assert measured[f"mae_{name}"] == 2.0
    assert measured[f"rmse_{name}"] == 2.0
    if extensive and not layout.per_atom:
        assert measured[f"rmse_{name}_per_atom"] == 1.0


def test_non_row_first_labels_are_refused_instead_of_silently_reshaped():
    requested, name = requested_for(declarations())
    configuration = Configuration(
        atomic_numbers=np.array([1, 1]),
        positions=np.zeros((2, 3)),
        properties={name: np.zeros((5, 2, 3))},
    )
    with pytest.raises(ValueError, match=r"target.*rows first"):
        targets_from_configuration(configuration, target_specs(requested), 2)


@fp64_only
def test_a_batch_without_response_labels_has_zero_gradient_and_no_response_metric():
    requested, name, _, batch = labelled_batch(declarations())
    batch.property_weights[name].zero_()
    response = torch.ones_like(batch.targets[name]).movedim(0, 1).requires_grad_()
    output = MACEOutput(extras={"q": batch.targets["q"], name: response})
    loss = build_loss(requested, LossConfig(weights={"q": 0.0, name: 1.0}))
    value = loss(output, batch)
    assert value.item() == 0
    value.backward()
    assert response.grad is not None and not torch.count_nonzero(response.grad)
    metrics = RunningMetrics(metric_specs(requested), loss)
    metrics.update(output, batch)
    assert f"rmse_{name}" not in metrics.compute()


@fp64_only
def test_loss_refuses_a_reference_with_the_wrong_rows_even_when_numel_matches():
    requested, name, _, batch = labelled_batch(
        declarations(wrt="strain", input_per_atom=False)
    )
    response = torch.ones_like(batch.targets[name]).movedim(0, 1)
    batch.targets[name] = batch.targets[name].movedim(0, 1)
    output = MACEOutput(extras={"q": batch.targets["q"], name: response})
    with pytest.raises(ValueError, match=r"target.*row-first shape"):
        build_loss(requested, LossConfig())(output, batch)


@fp64_only
@pytest.mark.parametrize("device", ["cpu", pytest.param("cuda", marks=pytest.mark.gpu)])
@pytest.mark.parametrize(
    "irreps,per_atom,wrt,input_per_atom",
    [
        ("2e", False, "pos", True),
        ("2e", False, "strain", False),
        ("2e", False, "elec_temp", False),
        ("0e", True, "elec_temp", False),
        ("1o", True, "elec_temp", False),
        ("0e", False, "strain", False),
    ],
)
def test_response_labels_train_evaluate_and_reload(
    tmp_path, device, irreps, per_atom, wrt, input_per_atom, deterministic_execution
):
    catalogue = declarations(
        irreps,
        per_atom,
        wrt,
        "1o" if wrt == "pos" else "0e+2e" if wrt == "strain" else "0e",
        input_per_atom,
    )
    observable = catalogue.observable("q")
    requested, response_name = requested_for(catalogue)
    frames = []
    for index, count in enumerate((2, 3, 4, 2, 3, 4)):
        atoms = Atoms(
            "H" * count,
            positions=np.array(
                [
                    [0.0, 0.0, 0.0],
                    [0.8 + 0.02 * index, 0.1, 0.0],
                    [0.1, 0.9, 0.2],
                    [0.2, 0.3, 1.0],
                ]
            )[:count],
        )
        if per_atom:
            atoms.arrays["q"] = np.zeros((count, observable.dimension))
        else:
            atoms.info["q"] = np.zeros(observable.dimension)
        if wrt == "elec_temp":
            if input_per_atom:
                atoms.arrays[wrt] = 0.2 + 0.1 * np.arange(count)
            else:
                atoms.info[wrt] = 0.2 + index * 0.1
        frames.append(atoms)
    train_file, valid_file = tmp_path / "train.xyz", tmp_path / "valid.xyz"
    write(train_file, frames[:4])
    write(valid_file, frames[4:])
    document = {
        "runtime": {"work_dir": str(tmp_path), "device": device, "seed": 7},
        "catalogue": catalogue.model_dump(),
        "data": {
            "heads": {
                "default": {
                    "train_file": str(train_file),
                    "valid_file": str(valid_file),
                }
            },
            "pin_memory": False,
        },
        "model": {
            "observables": ["q"],
            "num_channels": 2,
            "num_interactions": 2,
            "max_ell": 2,
            "hidden_irreps": "0e+1o+2e",
            "correlation": 2,
            "radial_mlp": [4],
            "readout": {"mlp_irreps": "4x0e+4x1o+4x2e"},
            "graph_features": [
                {
                    "name": wrt,
                    "kind": "continuous",
                    "embedding_dim": 2,
                    "per": "atom" if input_per_atom else "graph",
                }
            ]
            if wrt == "elec_temp"
            else [],
        },
        "training": {
            "max_num_epochs": 4,
            "batch_size": 4,
            "valid_batch_size": 4,
            "lr": 0.001,
            "scheduler": {"kind": {"kind": "constant"}},
        },
        "loss": {"weights": {"q": 0.0, response_name: 1.0}},
    }
    # Produce labels from a seeded model; the later learner sees a 20% changed
    # response. Loss arithmetic and axis ownership are pinned independently above.
    teacher = build(ResolvedConfig.model_validate(document))
    teacher.model.to(device)
    frame_index = 0
    for loader in (
        teacher.data.train_eval_loaders["default"],
        teacher.data.valid_loaders["default"],
    ):
        for batch in loader:
            batch = batch.to(device)
            output = teacher.model(batch.graph, compute=(response_name,))
            response = output.get(response_name).detach().cpu()
            rows = response if observable.is_scalar else response.movedim(0, 1)
            assert torch.count_nonzero(rows)
            offset = 0
            for graph_index in range(len(batch.graph["ptr"]) - 1):
                atoms = frames[frame_index]
                if per_atom or input_per_atom:
                    value = rows[offset : offset + len(atoms)].numpy() * 1.2
                    atoms.arrays[response_name] = value.reshape(len(atoms), -1)
                else:
                    atoms.info[response_name] = rows[graph_index].numpy() * 1.2
                offset += len(atoms)
                frame_index += 1
    # Each split includes a missing response; its zero placeholder must have
    # the same component shape and receive no gradient or metric contribution.
    for index in (0, 4):
        (
            frames[index].arrays if per_atom or input_per_atom else frames[index].info
        ).pop(response_name)
    write(train_file, frames[:4])
    write(valid_file, frames[4:])
    document["model"]["observables"].append(response_name)
    config = ResolvedConfig.model_validate(document)
    initial = build(config)
    initial.model.to(device)
    loss = build_loss(requested, config.loss)
    specs = metric_specs(requested)
    loader = initial.data.train_eval_loaders["default"]
    before = evaluate(
        initial.model, loader, loss, specs, device=device, compute=(response_name,)
    )
    assert before[f"rmse_{response_name}"] > 0
    trained = run(config)
    after = evaluate(
        trained.model, loader, loss, specs, device=device, compute=(response_name,)
    )
    assert after["loss"] < before["loss"]
    assert after[f"rmse_{response_name}"] < before[f"rmse_{response_name}"]
    assert trained.checkpoint_path is not None
    loaded = load_deployed(trained.checkpoint_path, device=device)
    batch = next(iter(loader)).to(device)
    expected = trained.model(batch.graph, compute=(response_name,)).get(response_name)
    actual = loaded.engine(batch.graph, compute=(response_name,)).get(response_name)
    torch.testing.assert_close(actual, expected, rtol=EXACT.rtol, atol=EXACT.atol)
    measured = evaluate(
        loaded.engine, loader, loss, specs, device=device, compute=(response_name,)
    )
    assert measured == after
