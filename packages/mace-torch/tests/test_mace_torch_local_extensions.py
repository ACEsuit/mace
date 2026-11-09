"""LSC oxidation extensions conserve charge and retain differentiable physics."""

from typing import cast

import numpy as np
import pytest
import torch
from conftest import TOLERANCES, fp64_only
from mace_core.config.local_sources import LocalSplitChargeConfig
from mace_torch.physics import DerivativeEngine
from mace_torch.serialization import canonical_state, load_canonical_state
from scipy.spatial.transform import Rotation
from test_mace_torch_electrostatics_families import ENERGY_STRESS, graph_for, model_for

CASES = [
    {"oxidation_mixer": "sum"},
    {"oxidation_mixer": "product"},
    {"transfer": "oxidation_dependent"},
    {"transfer": "oxidation_dependent", "oxidation_mixer": "sum"},
    {"transfer": "oxidation_dependent", "oxidation_mixer": "product"},
    {"polarizability": True},
    {
        "transfer": "oxidation_dependent",
        "oxidation_mixer": "product",
        "oxidation_range": (-3.0, 5.0),
        "polarizability": True,
    },
]


@pytest.mark.parametrize("options", CASES)
def test_symmetry_charge_conservation_and_canonical_reload(options):
    model = model_for(
        "lsc_mace", profile="full_periodic", per_atom=True, options=options
    )
    graph = graph_for("full_periodic")
    original = model(graph)
    rotation = Rotation.from_rotvec([0.3, -0.2, 0.4]).as_matrix()
    rotated = model(graph_for("full_periodic", rotation=rotation))
    matrix = torch.tensor(rotation, dtype=original.dipole.dtype)
    atol, rtol = TOLERANCES[torch.get_default_dtype()]
    torch.testing.assert_close(
        rotated.total_energy, original.total_energy, atol=atol, rtol=rtol
    )
    torch.testing.assert_close(
        rotated.dipole, original.dipole @ matrix.T, atol=atol, rtol=rtol
    )
    torch.testing.assert_close(
        original.extras["total_charge"],
        graph["oxidation_state"].sum()[None].to(original.dipole.dtype),
    )
    if options.get("polarizability"):
        alpha = original.extras["polarizability"]
        torch.testing.assert_close(alpha, alpha.transpose(-1, -2))
        torch.testing.assert_close(
            rotated.extras["polarizability"],
            matrix @ alpha @ matrix.T,
            atol=atol,
            rtol=rtol,
        )
    permutation = np.arange(len(graph["positions"]))[::-1].copy()
    permuted = model(graph_for("full_periodic", permutation=permutation))
    translated = model(graph_for("full_periodic", shift=np.array([0.2, 0.3, -0.4])))
    torch.testing.assert_close(
        permuted.total_energy, original.total_energy, atol=atol, rtol=rtol
    )
    torch.testing.assert_close(
        translated.total_energy, original.total_energy, atol=atol, rtol=rtol
    )
    rebuilt = model_for(
        "lsc_mace", profile="full_periodic", per_atom=True, options=options
    )
    load_canonical_state(rebuilt, canonical_state(model))
    torch.testing.assert_close(
        rebuilt(graph).total_energy, original.total_energy, atol=0, rtol=0
    )
    if options.get("polarizability"):
        torch.testing.assert_close(
            rebuilt(graph).extras["polarizability"],
            original.extras["polarizability"],
            atol=0,
            rtol=0,
        )


@pytest.mark.parametrize("options", CASES)
def test_disconnected_and_empty_neighbours(options):
    model = model_for("lsc_mace", per_atom=True, options=options)
    graph = graph_for()
    keep = (graph["edge_index"][0] < 3) & (graph["edge_index"][1] < 3)
    for key in ("edge_index", "shifts", "unit_shifts"):
        graph[key] = graph[key][:, keep] if key == "edge_index" else graph[key][keep]
    charge = model(graph).extras["charges"]
    torch.testing.assert_close(charge[:3].sum(), graph["oxidation_state"][:3].sum())
    torch.testing.assert_close(charge[3], graph["oxidation_state"][3])
    for key in ("edge_index", "shifts", "unit_shifts"):
        graph[key] = graph[key][:, :0] if key == "edge_index" else graph[key][:0]
    result = model(graph)
    assert torch.isfinite(result.total_energy).all()
    torch.testing.assert_close(result.extras["charges"], graph["oxidation_state"])


@fp64_only
@pytest.mark.parametrize("options", [CASES[0], CASES[4], CASES[-1]])
@pytest.mark.parametrize("profile", ["molecular", "full_periodic"])
def test_force_stress_finite_differences_and_training_gradients(options, profile):
    model = model_for("lsc_mace", options=options, profile=profile)
    graph = graph_for(profile)
    output = DerivativeEngine(model, ENERGY_STRESS)(
        dict(graph), compute=("forces", "stress"), training=True
    )
    step = 1e-5

    def energy(displacement=0.0, strain=0.0):
        change = torch.zeros_like(graph["positions"])
        change[1, 0] = displacement
        deformation = torch.eye(3)
        deformation[0, 0] += strain
        return model(
            dict(
                graph,
                positions=(graph["positions"] + change) @ deformation,
                cell=graph["cell"] @ deformation,
                shifts=graph["shifts"] @ deformation,
            )
        ).total_energy.sum()

    torch.testing.assert_close(
        -(energy(step) - energy(-step)) / (2 * step),
        output.forces[1, 0],
        atol=1e-6,
        rtol=0,
    )
    if profile != "molecular":
        stress = (
            (energy(strain=step) - energy(strain=-step))
            / (2 * step)
            / torch.linalg.det(graph["cell"]).abs().sum()
        )
        torch.testing.assert_close(stress, output.stress[0, 0, 0], atol=1e-6, rtol=0)
    loss = output.forces.square().sum() + output.stress.square().sum()
    if options.get("polarizability"):
        loss = loss + output.extras["polarizability"].square().sum()
    loss.backward()
    assert all(
        p.grad is not None and torch.isfinite(p.grad).all()
        for p in model.source_maps.parameters()
        if p.numel()
    )
    if model.polarizability_readouts is not None:
        assert all(
            p.grad is not None and torch.isfinite(p.grad).all()
            for p in model.polarizability_readouts.parameters()
            if p.numel()
        )


@fp64_only
def test_positions_and_formal_charge_double_backward():
    model = model_for(
        "lsc_mace", per_atom=True, options=CASES[-1], profile="full_periodic"
    )
    graph = graph_for("full_periodic")

    def energy(positions, formal):
        result = model(dict(graph, positions=positions, oxidation_state=formal))
        return result.total_energy + result.extras["polarizability"].square().sum()

    assert torch.autograd.gradgradcheck(
        energy,
        (
            graph["positions"].requires_grad_(),
            graph["oxidation_state"].requires_grad_(),
        ),
        fast_mode=True,
    )


@fp64_only
def test_polarizability_does_not_change_energy_or_field_response():
    model = model_for("lsc_mace", options={"polarizability": True})
    graph = graph_for()
    original = model(graph)
    with torch.no_grad():
        for p in model.polarizability_readouts.parameters():
            p.add_(0.3)
    changed = model(graph)
    torch.testing.assert_close(
        changed.total_energy, original.total_energy, atol=0, rtol=0
    )
    assert not torch.equal(
        changed.extras["polarizability"], original.extras["polarizability"]
    )


@fp64_only
@pytest.mark.parametrize("options", [CASES[0], CASES[-1]])
def test_fullgraph_compilation(options):
    from mace_torch.electrostatics import LONG_RANGE_PLAN, long_range_plan

    torch.compiler.reset()
    try:
        model = model_for("lsc_mace", options=options)
        graph = graph_for()
        graph[LONG_RANGE_PLAN] = long_range_plan(
            graph, model.solver_spec, dense_pairs=True
        )
        expected = model(graph)
        actual = torch.compile(model, backend="aot_eager", fullgraph=True)(graph)
        torch.testing.assert_close(actual.total_energy, expected.total_energy)
        if options.get("polarizability"):
            torch.testing.assert_close(
                actual.extras["polarizability"], expected.extras["polarizability"]
            )
    finally:
        torch.compiler.reset()


@pytest.mark.parametrize(
    "options",
    [
        {"oxidation_range": (1, 1)},
        {"oxidation_range": (2, 1)},
        {"oxidation_range": (float("nan"), 1)},
        {"oxidation_range": (-3, 3)},
        {"oxidation_mixer": "unknown"},
        {"transfer": "unknown"},
    ],
)
def test_invalid_or_unused_configuration(options):
    with pytest.raises(ValueError):
        LocalSplitChargeConfig(formal_charges={1: 1.0}, **options)


@fp64_only
def test_combined_training_and_deployment(tmp_path):
    from ase.io import read, write
    from mace_core.config.resolved import ResolvedConfig
    from mace_core.elements.default_keys import DefaultKeys
    from mace_core.observables import DEFAULT_CATALOGUE
    from mace_torch.deploy.loader import load_deployed
    from mace_torch.models.electrostatics import LocalSplitChargeModel
    from mace_torch.train import run_data_stage, run_model_stage, run_train_stage
    from mace_torch.train.optimizers import build_optimizer
    from test_mace_torch_full_batch import configuration

    document = configuration(tmp_path, max_num_epochs=2, lr=0.001).model_dump(
        mode="json"
    )
    document["model"].update(
        model="lsc_mace",
        local_sources={
            "kind": "lsc_mace",
            "multipole_max_l": 1,
            "formal_charge_input": "oxidation_state",
            **CASES[-1],
        },
        observables=["energy", "forces", "polarizability"],
    )
    document["electrostatics"].update(enabled=True, periodicity_profile="molecular")
    document["loss"]["weights"]["polarizability"] = 1.0
    path = document["data"]["heads"]["default"]["train_file"]
    frames = read(path, index=":")
    for atoms in frames:
        atoms.arrays["oxidation"] = np.where(atoms.numbers == 8, -2.0, 1.0)
        atoms.info[DefaultKeys.POLARIZABILITY.value] = np.eye(3) * len(atoms) * 0.2
    write(path, frames, format="extxyz")
    document["data"]["embedding_specs"] = {
        "oxidation_state": {"per": "atom", "key": "oxidation"}
    }
    config = ResolvedConfig.model_validate(document)
    data = run_data_stage(config, DEFAULT_CATALOGUE)
    built = run_model_stage(config, data, DEFAULT_CATALOGUE)
    optimized = [
        id(p)
        for group in build_optimizer(built.engine, config.training).param_groups
        for p in group["params"]
    ]
    parameters = {id(p) for p in built.engine.parameters() if p.requires_grad}
    assert len(optimized) == len(parameters) and set(optimized) == parameters
    local = cast(LocalSplitChargeModel, built.engine.get_submodule("model"))
    assert local.backbone.graph_embedding is not None
    assert local.polarizability_readouts is not None
    modules = (
        local.source_maps,
        local.backbone.graph_embedding,
        local.polarizability_readouts,
    )
    originals = [
        [p.detach().clone() for p in module.parameters()] for module in modules
    ]
    trained = run_train_stage(config, built, checkpoint_path=tmp_path / "combined")
    assert np.isfinite(trained.history[-1].train_loss)
    for module, original in zip(modules, originals, strict=True):
        assert any(
            not torch.equal(before, after)
            for before, after in zip(original, module.parameters(), strict=True)
        ), type(module).__name__
    deployed = load_deployed(tmp_path / "combined.json")
    restored = load_deployed(tmp_path / "combined.json")
    assert deployed.metadata.electrostatics is not None
    assert deployed.metadata.electrostatics.solver == "reference"
    assert deployed.metadata.electrostatics.bit_parity
    graph = next(data.train_loader.batches(0, drop_last=False)).graph
    expected, actual = deployed.compute(graph), restored.compute(graph)
    for name in ("total_energy", "forces", "dipole"):
        torch.testing.assert_close(
            getattr(actual, name), getattr(expected, name), atol=0, rtol=0
        )
    torch.testing.assert_close(
        actual.extras["polarizability"],
        expected.extras["polarizability"],
        atol=0,
        rtol=0,
    )
    response = deployed.compute(graph, compute=("dalpha_dr",)).extras["dalpha_dr"]
    step = 1e-5
    shifted = []
    for sign in (1, -1):
        positions = graph["positions"].clone()
        positions[0, 0] += sign * step
        value = deployed.compute(dict(graph, positions=positions))
        shifted.append(value.extras["polarizability"][0].flatten())
    torch.testing.assert_close(
        response[:, 0, 0], (shifted[0] - shifted[1]) / (2 * step), atol=1e-7, rtol=0
    )


@fp64_only
@pytest.mark.parametrize("max_l", [0, 1, 2, 3])
def test_polarizability_readout_higher_features(max_l):
    from mace_core.clebsch_gordan.real_basis import wigner_d_real
    from mace_torch.backends.reference import ReferenceBackend
    from mace_torch.kernels import initialize_model_weights
    from mace_torch.nn.local_sources import LocalPolarizability

    irreps = "+".join(
        f"2x{degree}{'o' if degree % 2 else 'e'}" for degree in range(max_l + 1)
    )
    block = LocalPolarizability(ReferenceBackend(), irreps, "float64")
    initialize_model_weights(block, 43)
    rotation = Rotation.from_rotvec([0.3, -0.1, 0.2]).as_matrix()
    # Charge-aware backbones evaluate harmonics on edges ordered (y, z, x).
    feature_rotation = rotation[np.ix_([1, 2, 0], [1, 2, 0])]
    features, rotated, inverted = [], [], []
    for degree in range(max_l + 1):
        x = torch.randn(3, 2, 2 * degree + 1)
        features.append(x.flatten(1))
        rotated.append(
            (x @ torch.tensor(wigner_d_real(degree, feature_rotation))).flatten(1)
        )
        inverted.append((x * (-1) ** degree).flatten(1))
    features = torch.cat(features, dim=-1).requires_grad_()
    expected = block(features)
    matrix = torch.tensor(rotation)
    torch.testing.assert_close(
        block(torch.cat(rotated, dim=-1)), matrix @ expected @ matrix.T
    )
    torch.testing.assert_close(block(torch.cat(inverted, dim=-1)), expected)
    assert torch.autograd.gradgradcheck(block, (features,), fast_mode=True)
