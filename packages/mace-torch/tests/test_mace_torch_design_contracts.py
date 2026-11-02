"""Architectural seams exercised through their observable behavior."""

from __future__ import annotations

import numpy as np
import pytest
import torch
from ase import Atoms
from ase.io import write
from conftest import fp64_only
from mace_core.config.model import ModelConfig
from mace_core.config.resolved import ResolvedConfig
from mace_core.observables import InputSpec, ObservableCatalogue, ObservableSpec
from mace_torch.cli.run_train import run
from mace_torch.deploy.loader import load_deployed
from mace_torch.finetune.stages import build
from mace_torch.models.normalization import ObservableNormalization
from mace_torch.serialization import canonical_state


@fp64_only
@pytest.mark.parametrize("normalize", [False, True])
@pytest.mark.parametrize("device", ["cpu", pytest.param("cuda", marks=pytest.mark.gpu)])
@pytest.mark.parametrize(
    "irreps,per_atom,width,derivatives",
    [
        ("0e", False, 1, ()),
        ("0e", True, 1, ()),
        ("1o", False, 3, ()),
        ("0e+2e", True, 6, ()),
        ("2e", False, 5, ("pos", "strain")),
        ("2e", False, 5, ("elec_temp",)),
    ],
)
def test_custom_observable_trains_saves_and_loads_without_energy(
    tmp_path, irreps, per_atom, width, derivatives, device, normalize
):
    catalogue = ObservableCatalogue(
        inputs=(
            InputSpec(name="pos", irreps="1o", per_atom=True, units="A"),
            InputSpec(name="strain", irreps="0e+2e", per_atom=False, units="1"),
            InputSpec(
                name="elec_temp",
                irreps="0e",
                per_atom=False,
                units="K",
                differentiable=True,
            ),
        ),
        observables=(
            ObservableSpec(
                name="custom_property",
                irreps=irreps,
                per_atom=per_atom,
                units="test-unit",
                derivatives=derivatives,
            ),
        ),
    )
    frames = []
    for index in range(6):
        atoms = Atoms(
            "H2O", positions=[[0, 0, 0], [0.8 + index * 0.03, 0, 0], [0.1, 0.9, 0]]
        )
        atoms.info["elec_temp"] = 0.2 + index * 0.1
        value = np.arange(width, dtype=float) * 0.2 + index * 0.1 + 0.5
        if per_atom:
            atoms.arrays["custom_property"] = (
                np.full(len(atoms), value[0])
                if width == 1
                else np.tile(value, (len(atoms), 1))
            )
        else:
            atoms.info["custom_property"] = float(value[0]) if width == 1 else value
        if index == 0:
            # A missing label must collate with labelled scalars/vectors/tensors
            # and contribute zero to the loss, without changing its shape.
            (atoms.arrays if per_atom else atoms.info).pop("custom_property")
        frames.append(atoms)
    datafile = tmp_path / "custom.xyz"
    write(datafile, frames)
    config = ResolvedConfig.model_validate(
        {
            "runtime": {
                "work_dir": str(tmp_path),
                "name": "custom",
                "seed": 7,
                "device": device,
            },
            "data": {
                "heads": {"default": {"train_file": str(datafile)}},
                "valid_fraction": 0.33,
                "pin_memory": False,
            },
            "catalogue": catalogue.model_dump(),
            "model": {
                "observables": ["custom_property"],
                "normalization": {"custom_property": {"default": {"kind": "std"}}}
                if normalize
                else {},
                "num_channels": 2,
                "num_interactions": 2,
                "max_ell": 2,
                "hidden_irreps": "0e+1o+2e",
                "correlation": 2,
                "radial_mlp": [4],
                "readout": {"mlp_irreps": "4x0e+4x1o+4x2e"},
                "graph_features": [
                    {"name": "elec_temp", "kind": "continuous", "embedding_dim": 2}
                ]
                if "elec_temp" in derivatives
                else [],
            },
            "loss": {"weights": {"custom_property": 1.0}},
            "training": {
                "max_num_epochs": 2,
                "batch_size": 2,
                "valid_batch_size": 2,
                "lr": 0.01,
                "scheduler": {"kind": {"kind": "constant"}},
            },
        }
    )
    initial = build(config)
    if normalize:
        normalizer = initial.model.get_submodule(
            "backbone.outputs.normalization.custom_property"
        )
        assert isinstance(normalizer, ObservableNormalization)
        assert torch.all(normalizer.scale > 0)
        assert not torch.all(normalizer.scale == 1)
        assert not list(normalizer.parameters())
    before = {
        path: {key: value.to(device).clone() for key, value in values.items()}
        for path, values in canonical_state(initial.model).items()
    }
    trained = run(config)
    after = canonical_state(trained.model)
    assert any(
        not torch.equal(value, after[path][key])
        for path, values in before.items()
        for key, value in values.items()
    )
    if "elec_temp" in derivatives:
        embedded = [
            (path, key, value)
            for path, values in before.items()
            if "graph_embedding" in path
            for key, value in values.items()
        ]
        assert embedded and any(
            not torch.equal(value, after[path][key]) for path, key, value in embedded
        )
        from mace_torch.finetune.freeze import freeze

        freeze(initial.model, 1)
        features = initial.model.get_submodule("backbone.backbone.graph_embedding")
        assert all(not parameter.requires_grad for parameter in features.parameters())
    assert trained.checkpoint_path is not None
    loaded = load_deployed(trained.checkpoint_path, device=device)
    assert loaded.engine.energy is None
    batch = next(iter(initial.data.train_eval_loaders["default"])).to(device)
    expected = trained.model(batch.graph, compute=()).get("custom_property")
    actual = loaded.engine(batch.graph, compute=()).get("custom_property")
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    assert loaded.metadata.config.resolved["catalogue"] == catalogue.model_dump(
        mode="json"
    )
    for wrt in derivatives:
        name = catalogue.observable("custom_property").derivative_name(wrt)
        assert name == f"d_custom_property_d_{wrt}"
        response = loaded.engine(batch.graph, compute=(name,)).get(name)
        assert torch.isfinite(response).all() and torch.count_nonzero(response)
        # Independent perturbations of the physical input of the trained model.
        step = 1e-5
        samples = []
        for offset in (-step, step):
            moved = dict(batch.graph)
            if wrt == "pos":
                moved["positions"] = batch.graph["positions"].clone()
                moved["positions"][0, 0] += offset
                tested = response[:, 0, 0]
                assert response.shape == (width, len(moved["positions"]), 3)
            elif wrt == "strain":
                strain = torch.zeros_like(batch.graph["cell"])
                strain[0, 0, 0] = offset
                moved["positions"] = batch.graph["positions"] + torch.einsum(
                    "ni,nij->nj", batch.graph["positions"], strain[batch.graph["batch"]]
                )
                moved["cell"] = batch.graph["cell"] + batch.graph["cell"] @ strain
                moved["shifts"] = torch.einsum(
                    "ni,nij->nj",
                    batch.graph["unit_shifts"].to(moved["cell"]),
                    moved["cell"][batch.graph["batch"][batch.graph["edge_index"][0]]],
                )
                tested = response[:, 0, 0, 0]
                assert response.shape == (width, len(moved["cell"]), 3, 3)
            else:
                moved[wrt] = batch.graph[wrt].clone()
                moved[wrt][0] += offset
                tested = response[:, 0].reshape(width)
                assert response.shape[1:] == moved[wrt].shape
            samples.append(loaded.engine(moved, compute=()).get("custom_property")[0])
        from tests.golden.harness import FP64_CPU_REFERENCE

        torch.testing.assert_close(
            tested,
            (samples[1] - samples[0]) / (2 * step),
            atol=FP64_CPU_REFERENCE.atol,
            rtol=FP64_CPU_REFERENCE.rtol,
        )
        loaded.engine.zero_grad(set_to_none=True)
        differentiable = loaded.engine(batch.graph, compute=(name,), training=True).get(
            name
        )
        differentiable.square().mean().backward()
        gradients = [
            parameter.grad
            for parameter in loaded.engine.parameters()
            if parameter.grad is not None
        ]
        assert gradients and all(
            torch.isfinite(gradient).all() for gradient in gradients
        )
        assert any(torch.count_nonzero(gradient) for gradient in gradients)


def test_a_second_catalogue_cannot_silently_replace_the_config():
    from mace_core.observables import DEFAULT_CATALOGUE

    config = ResolvedConfig(catalogue=DEFAULT_CATALOGUE)
    with pytest.raises(ValueError, match="catalogue disagrees"):
        build(config, catalogue=ObservableCatalogue())


@fp64_only
def test_model_config_controls_repulsion_and_readout(tmp_path):
    from mace_core.observables import DEFAULT_CATALOGUE
    from mace_torch.backends.reference import ReferenceBackend
    from mace_torch.models import MACEModel
    from mace_torch.serialization import load_canonical_state
    from test_mace_torch_training_pipeline import configuration

    config = configuration(tmp_path)
    architecture = ModelConfig.model_validate(
        {
            **config.model.model_dump(),
            "pair_repulsion": True,
            "readout": {"mlp_irreps": "4x0e", "last_only": True},
        }
    )
    config = config.model_copy(update={"model": architecture})
    built = build(config)
    staged = built.model.backbone
    assert isinstance(staged, MACEModel)
    direct = MACEModel(
        ReferenceBackend(),
        architecture,
        list(built.data.z_table.zs),
        [DEFAULT_CATALOGUE.observable("energy")],
        staged.outputs.energy_head,
        avg_num_neighbors=built.data.statistics.avg_num_neighbors,
    )
    assert direct.repulsion is not None
    assert direct.outputs.heads["energy"].reachable == [
        architecture.num_interactions - 1
    ]
    load_canonical_state(direct, canonical_state(staged))
    graph = next(iter(built.data.train_eval_loaders["default"])).graph
    torch.testing.assert_close(
        direct(graph).total_energy, staged(graph).total_energy, rtol=0, atol=0
    )


@fp64_only
def test_checkpoint_survives_an_implementation_wrapper_and_a_renamed_attribute(
    tmp_path,
):
    from mace_torch.canonical import canonical_modules
    from mace_torch.serialization import load_checkpoint, save_checkpoint
    from mace_torch_engine_fixtures import build_graph
    from test_mace_torch_serialization import SETTINGS, trained
    from test_mace_torch_serialization import build as rebuild

    class Wrapped(torch.nn.Module):
        def __init__(self, operator):
            super().__init__()
            self.renamed_implementation = operator

        def forward(self, features):
            return self.renamed_implementation(features)

        def canonical_components(self):
            return {"": self.renamed_implementation}

    model = trained()
    graph = build_graph([[0, 0, 0], [0.9, 0, 0], [-0.2, 0.9, 0]], [8, 1, 1])
    before = model(graph).layers
    expected_keys = set(canonical_state(model))
    convolution = model.interactions[0].body
    convolution.linear_up = Wrapped(convolution.linear_up)
    assert set(canonical_state(model)) == expected_keys
    assert (
        canonical_modules(model)["interactions.0.body.linear_up"]
        is convolution.linear_up.renamed_implementation
    )
    save_checkpoint(tmp_path / "wrapped", model, dict(SETTINGS))
    restored = load_checkpoint(tmp_path / "wrapped", rebuild)
    for expected, actual in zip(before, restored(graph).layers, strict=True):
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)


def test_a_component_cannot_silently_omit_canonical_state():
    from mace_torch.canonical import canonical_modules
    from test_mace_torch_serialization import trained

    model = trained()
    model.canonical_components = lambda: {}
    with pytest.raises(ValueError, match="omit"):
        canonical_modules(model)


@fp64_only
@pytest.mark.parametrize(
    "family", ["backbone", "energy", "dipole", "dielectric", "polar", "magnetic"]
)
def test_logical_identities_preserve_existing_format_three_tensors(tmp_path, family):
    """The old named-module writer and the logical writer agree before a rename."""
    from mace_torch.serialization import canonical_metadata
    from test_mace_torch_dipole_model import FIXED, PREDICTED
    from test_mace_torch_dipole_model import build_model as dipole
    from test_mace_torch_magnetic_model import engine as magnetic
    from test_mace_torch_polar_model import build_model as polar
    from test_mace_torch_serialization import trained
    from test_mace_torch_training_pipeline import configuration

    factories = {
        "backbone": trained,
        "energy": lambda: build(configuration(tmp_path)).model,
        "dipole": lambda: dipole(FIXED),
        "dielectric": lambda: dipole(PREDICTED),
        "polar": polar,
        "magnetic": magnetic,
    }
    model = factories[family]()
    previous = {
        path: module.to_canonical()
        for path, module in model.named_modules()
        if callable(getattr(module, "to_canonical", None))
    }
    current = canonical_state(model)
    assert current.keys() == previous.keys()
    for path, tensors in previous.items():
        assert current[path].keys() == tensors.keys()
        for name, value in tensors.items():
            torch.testing.assert_close(current[path][name], value, atol=0, rtol=0)
    previous_metadata = {
        path: module.canonical_metadata()
        for path, module in model.named_modules()
        if callable(getattr(module, "canonical_metadata", None))
    }
    assert canonical_metadata(model) == previous_metadata


@fp64_only
def test_custom_scalar_position_response_and_double_backward():
    from mace_core.observables import DEFAULT_CATALOGUE, DerivativeRequest
    from mace_torch.backends.reference import ReferenceBackend
    from mace_torch.kernels.initialization import initialize_model_weights
    from mace_torch.models import MACEModel
    from mace_torch.physics.outputs import DerivativeEngine
    from mace_torch_engine_fixtures import build_graph

    spec = ObservableSpec(
        name="custom_property",
        irreps="0e",
        per_atom=False,
        units="test-unit",
        derivatives=(DerivativeRequest(wrt="pos"),),
    )
    config = ModelConfig(num_channels=2, num_interactions=2, max_ell=1, correlation=2)
    model = MACEModel(ReferenceBackend(), config, [1, 8], [spec])
    initialize_model_weights(model, 7)
    engine = DerivativeEngine(
        model, None, None, inputs=DEFAULT_CATALOGUE.inputs, responses=[spec]
    )
    graph = build_graph([[0, 0, 0], [0.9, 0, 0], [-0.2, 0.9, 0]], [8, 1, 1])
    name = spec.derivative_name("pos")
    response = engine(graph, compute=(name,), training=True).get(name)
    assert response.shape == (3, 3)
    finite_difference = torch.zeros_like(response)
    step = 1e-5
    for atom in range(3):
        for axis in range(3):
            plus, minus = dict(graph), dict(graph)
            plus["positions"], minus["positions"] = (
                graph["positions"].clone(),
                graph["positions"].clone(),
            )
            plus["positions"][atom, axis] += step
            minus["positions"][atom, axis] -= step
            finite_difference[atom, axis] = (
                model(plus).get(spec.name) - model(minus).get(spec.name)
            ).item() / (2 * step)
    torch.testing.assert_close(response, finite_difference, rtol=1e-5, atol=1e-7)
    response.square().sum().backward()
    gradients = [p.grad for p in model.parameters() if p.grad is not None]
    assert gradients and all(torch.isfinite(gradient).all() for gradient in gradients)
    assert any(torch.count_nonzero(gradient) for gradient in gradients)


def test_distinct_components_cannot_claim_the_same_checkpoint_role():
    from mace_torch.canonical import canonical_modules

    class Ambiguous(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.first = torch.nn.ModuleDict({"child": torch.nn.Identity()})
            self.second = torch.nn.Identity()

        def canonical_components(self):
            return {"branch": self.first, "branch.child": self.second}

    with pytest.raises(ValueError, match="duplicate canonical"):
        canonical_modules(Ambiguous())
