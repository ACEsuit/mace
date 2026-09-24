"""The magnetic model on its own: symmetry, the zero moment, and a training run.

The numbers are pinned against the frozen tree in ``tests/parity``. What is
checked here is what holds of any set of weights: the energy of a structure
turned together with its moments does not change and its two kinds of force
turn with it, the derivative against a zero moment is a real derivative, and a
magnetic model trains from its configuration and comes back from its
checkpoint computing the same numbers.
"""

from __future__ import annotations

from typing import cast

import numpy as np
import pytest
import torch
from ase import Atoms
from ase.io import write
from mace_core.config import FixedPointSpec
from mace_core.config.data import AugmentationSpec
from mace_core.config.model import MagneticConfig
from mace_core.config.resolved import ResolvedConfig
from mace_core.elements import AtomicNumberTable, ResolvedE0s
from mace_core.kernels.precision import PrecisionConfig
from mace_core.neighbors import get_neighborhood
from mace_core.observables import DEFAULT_CATALOGUE
from mace_torch.backends.reference import ReferenceBackend
from mace_torch.calculators import MACECalculator
from mace_torch.cli.run_train import run
from mace_torch.deploy.loader import load_deployed
from mace_torch.finetune.stages import build
from mace_torch.kernels import initialize_model_weights
from mace_torch.models.energy import EnergyOutputHead, ScaleShiftSpec
from mace_torch.models.magnetic import MagneticModel
from mace_torch.nn.magnetic import OneBodyMomentEnergy
from mace_torch.physics import DerivativeEngine
from mace_torch.train import ModelStageError, run_data_stage
from scipy.spatial.transform import Rotation
from test_mace_torch_extend_elements import water_foundation

NUMBERS = [8, 26]
ENERGY = DEFAULT_CATALOGUE.observable("energy")
MAGMOM = DEFAULT_CATALOGUE.input("magmom")
CLUSTER = np.array(
    [[0.0, 0.0, 0.0], [1.9, 0.1, 0.0], [0.2, 1.8, 0.3], [-1.1, -0.9, 1.2]]
)


def engine(one_body: int = 6, seed: int = 3) -> DerivativeEngine:
    head = EnergyOutputHead(
        ResolvedE0s({"default": {8: -4.25, 26: -6.75}}),
        ["default"],
        AtomicNumberTable(NUMBERS),
        ScaleShiftSpec("std", (0.7,), (0.1,)),
        PrecisionConfig(),
    )
    model = MagneticModel(
        ReferenceBackend(),
        atomic_numbers=NUMBERS,
        observables=[ENERGY],
        energy_head=head,
        saturation=[1.2, 4.5],
        num_features=4,
        lmax=2,
        moment_lmax=2,
        num_moment_basis=6,
        one_body_basis=one_body,
        cutoff=4.0,
        cutoff_order=5,
        correlation=2,
        pair_repulsion=True,
        readout_hidden="4x0e",
    )
    initialize_model_weights(model, seed)
    return DerivativeEngine(model, ENERGY, None, inputs=[MAGMOM])


def graph(positions, moments, numbers=(8, 26, 26, 26)):
    neighborhood = get_neighborhood(
        np.asarray(positions), 4.0, (False, False, False), None
    )
    return {
        "positions": torch.tensor(np.asarray(positions, dtype=float)),
        "atomic_numbers": torch.tensor(list(numbers)),
        "element_index": torch.tensor([NUMBERS.index(z) for z in numbers]),
        "edge_index": torch.tensor(neighborhood.edge_index),
        "shifts": torch.tensor(neighborhood.shifts),
        "unit_shifts": torch.tensor(neighborhood.unit_shifts),
        "cell": torch.tensor(np.asarray(neighborhood.cell, dtype=float)).reshape(
            1, 3, 3
        ),
        "batch": torch.zeros(len(numbers), dtype=torch.long),
        "num_graphs": 1,
        "head": torch.zeros(1, dtype=torch.long),
        "magmom": torch.tensor(np.asarray(moments, dtype=float)),
    }


def moments(seed: int = 0) -> np.ndarray:
    return np.random.default_rng(seed).normal(scale=1.5, size=(4, 3))


def evaluate(model, positions, spins):
    return model(graph(positions, spins), compute=("forces", "magforces"))


@pytest.fixture(name="model")
def fixture_model(dtype):
    if dtype != torch.float64:
        pytest.skip("the symmetry checks are made at float64")
    return engine()


def test_turning_the_structure_with_its_moments_turns_both_forces(model):
    rotation = Rotation.from_euler("zyx", [0.4, -1.1, 2.3]).as_matrix()
    spins = moments()
    before = evaluate(model, CLUSTER, spins)
    after = evaluate(model, CLUSTER @ rotation.T, spins @ rotation.T)
    turn = torch.tensor(rotation)
    assert torch.allclose(before.total_energy, after.total_energy, atol=1e-12)
    assert torch.allclose(before.forces @ turn.T, after.forces, atol=1e-12)
    assert torch.allclose(
        before.extras["magforces"] @ turn.T, after.extras["magforces"], atol=1e-12
    )


def test_turning_the_moments_alone_changes_the_energy(model):
    """The moments couple to the geometry: they are not read as lengths only."""
    rotation = Rotation.from_euler("x", 0.9).as_matrix()
    spins = moments()
    before = evaluate(model, CLUSTER, spins)
    after = evaluate(model, CLUSTER, spins @ rotation.T)
    assert float((before.total_energy - after.total_energy).abs()) > 1e-6


def test_inverting_the_structure_with_its_moments_changes_nothing(model):
    """The moment is read as a polar vector, as the trained models read it."""
    spins = moments()
    before = evaluate(model, CLUSTER, spins)
    after = evaluate(model, -CLUSTER, -spins)
    assert torch.allclose(before.total_energy, after.total_energy, atol=1e-12)


def test_the_derivative_at_a_zero_moment_is_the_finite_difference(model):
    spins = moments()
    spins[1] = 0.0
    result = evaluate(model, CLUSTER, spins)
    analytic = result.extras["magforces"][1, 2]
    assert torch.isfinite(result.extras["magforces"]).all()

    def energy(m_z):
        nudged = spins.copy()
        nudged[1, 2] = m_z
        return float(evaluate(model, CLUSTER, nudged).total_energy)

    step = 1e-5
    numeric = -(energy(step) - energy(-step)) / (2 * step)
    assert abs(float(analytic) - numeric) < 1e-7


def test_the_squashed_length_runs_from_one_to_minus_one_at_saturation(model):
    features = model.backbone.backbone.moments
    element = torch.tensor([0, 1, 1])
    spins = torch.tensor([[0.0, 0.0, 0.0], [0.0, 4.5, 0.0], [6.0, 0.0, 8.0]])
    squashed = features.squashed_length(spins, element).squeeze(-1)
    assert torch.equal(squashed, torch.tensor([1.0, -1.0, -1.0], dtype=torch.float64))


def test_the_one_body_term_reads_the_moment_length_alone(model):
    one_body = model.backbone.one_body
    features = model.backbone.backbone.moments
    element = torch.tensor([0, 1, 1, 1])
    spins = torch.tensor(moments())
    rotation = torch.tensor(Rotation.from_euler("y", 1.3).as_matrix())
    term = one_body(features.squashed_length(spins, element), element, None)
    turned = one_body(
        features.squashed_length(spins @ rotation.T, element), element, None
    )
    assert torch.allclose(term, turned, atol=1e-12)
    assert float(term.abs().max()) > 1e-3
    with torch.no_grad():
        one_body.coefficients.zero_()
    zero = one_body(features.squashed_length(spins, element), element, None)
    assert torch.equal(zero, torch.zeros_like(zero))


# ---------------------------------------------------------------------------
# The configuration
# ---------------------------------------------------------------------------


def test_an_unlisted_element_saturates_at_one_and_an_extra_one_is_ignored():
    settings = MagneticConfig(saturation={26: 4.5, 28: 0.6})
    assert settings.saturation_for([8, 26]) == [1.0, 4.5]


def test_a_saturation_at_or_below_zero_is_refused():
    with pytest.raises(ValueError, match="saturation"):
        MagneticConfig(saturation={26: 0.0})


def structures(path, count=10, seed=0):
    generator = np.random.default_rng(seed)
    frames = []
    for _ in range(count):
        atoms = Atoms(
            "OFe3", positions=CLUSTER + generator.normal(scale=0.05, size=(4, 3))
        )
        atoms.arrays["REF_magmom"] = generator.normal(scale=1.5, size=(4, 3))
        atoms.info["REF_energy"] = -20.0 + generator.normal()
        atoms.arrays["REF_forces"] = generator.normal(scale=0.1, size=(4, 3))
        atoms.arrays["REF_magforces"] = generator.normal(scale=0.1, size=(4, 3))
        frames.append(atoms)
    write(path, frames)
    return path


def configuration(directory, **model):
    return ResolvedConfig.model_validate(
        {
            "runtime": {"work_dir": str(directory), "seed": 2},
            "data": {
                "heads": {
                    "default": {
                        "train_file": str(structures(directory / "train.xyz")),
                        "e0s": {"kind": "table", "values": {8: -4.25, 26: -6.75}},
                    }
                },
                "valid_fraction": 0.2,
                "pin_memory": False,
            },
            "model": {
                "model": "magnetic",
                "observables": ["energy", "forces", "magforces"],
                "r_max": 4.0,
                "num_channels": 4,
                "max_ell": 2,
                "correlation": 2,
                "magnetic": {
                    "saturation": {26: 4.5, 8: 1.2},
                    "num_basis": 6,
                    "lmax": 2,
                    "one_body": True,
                    "one_body_basis": 6,
                },
                **model,
            },
            "training": {"max_num_epochs": 2, "batch_size": 4},
        }
    )


@pytest.fixture(name="trained", scope="module")
def fixture_trained(tmp_path_factory):
    previous = torch.get_default_dtype()
    torch.set_default_dtype(torch.float64)
    try:
        directory = tmp_path_factory.mktemp("magnetic")
        return run(configuration(directory))
    finally:
        torch.set_default_dtype(previous)


def test_a_magnetic_model_comes_back_from_its_checkpoint(trained):
    assert trained.checkpoint_path is not None
    deployed = load_deployed(trained.checkpoint_path)
    assert list(deployed.z_table.zs) == NUMBERS
    probe = graph(CLUSTER, moments(4))
    before = trained.model(probe, compute=("forces", "magforces"))
    after = deployed.engine(probe, compute=("forces", "magforces"))
    assert torch.equal(before.total_energy, after.total_energy)
    assert torch.equal(before.extras["magforces"], after.extras["magforces"])
    assert torch.equal(
        trained.model.backbone.backbone.moments.saturation,
        torch.tensor([1.2, 4.5], dtype=torch.float64),
    )


def one_body_of(engine) -> OneBodyMomentEnergy:
    return cast(OneBodyMomentEnergy, engine.get_submodule("backbone.one_body"))


def test_one_body_coefficients_that_are_not_trained_keep_their_values(tmp_path):
    config = configuration(
        tmp_path,
        magnetic={"one_body": True, "one_body_basis": 6, "train_one_body": False},
    )
    built = build(config)
    start = one_body_of(built.model).coefficients.detach().clone()
    trained = run(config)
    assert torch.equal(one_body_of(trained.model).coefficients, start)


def test_magnetic_forces_are_refused_for_a_model_that_reads_no_moments(tmp_path):
    with pytest.raises(ModelStageError, match="magforces"):
        build(configuration(tmp_path, model="scale_shift"))


# ---------------------------------------------------------------------------
# The calculator
# ---------------------------------------------------------------------------


def iron_cluster(seed: int = 5) -> Atoms:
    atoms = Atoms("OFe3", positions=CLUSTER)
    atoms.arrays["REF_magmom"] = moments(seed)
    return atoms


def test_the_calculator_reports_the_magnetic_forces_of_the_model(trained):
    assert trained.checkpoint_path is not None
    calculator = MACECalculator(model_paths=trained.checkpoint_path)
    assert "magforces" in calculator.implemented_properties
    atoms = iron_cluster()
    calculator.calculate(atoms)
    direct = trained.model(
        graph(CLUSTER, atoms.arrays["REF_magmom"]), compute=("forces", "magforces")
    )
    np.testing.assert_allclose(
        calculator.results["magforces"],
        direct.extras["magforces"].detach().numpy(),
        atol=1e-12,
    )


def test_a_structure_without_moments_is_refused(trained):
    calculator = MACECalculator(model_paths=trained.checkpoint_path)
    atoms = iron_cluster()
    del atoms.arrays["REF_magmom"]
    atoms.set_initial_magnetic_moments([0.0, 2.0, 2.0, 2.0])
    with pytest.raises(ValueError, match="initial magnetic moments are not read"):
        calculator.calculate(atoms)


def test_changing_the_moments_invalidates_the_cached_energy(trained):
    atoms = iron_cluster()
    atoms.calc = MACECalculator(model_paths=trained.checkpoint_path)
    first = atoms.get_potential_energy()
    atoms.arrays["REF_magmom"][1] *= 0.5
    assert atoms.get_potential_energy() != first


def test_the_moments_can_be_read_from_another_key(trained):
    atoms = iron_cluster()
    atoms.arrays["spins"] = atoms.arrays.pop("REF_magmom")
    calculator = MACECalculator(model_paths=trained.checkpoint_path, magmom_key="spins")
    calculator.calculate(atoms)
    assert calculator.results["magforces"].shape == (4, 3)


def test_a_relaxation_that_runs_off_to_infinity_is_refused(trained):
    """A model whose energy has no lower bound in the moments, as an untrained
    one generally has not: the solid harmonics grow with the moment. The
    energy that would come back is not a number, and the driver says so rather
    than returning it."""
    spec = FixedPointSpec(
        variable="magmom", max_iter=100, tolerance=1e-8, require_convergence=False
    )
    atoms = iron_cluster()
    atoms.calc = MACECalculator(model_paths=trained.checkpoint_path, fixed_point=spec)
    with pytest.raises(RuntimeError, match="stopped being finite"):
        atoms.get_potential_energy()


def test_a_fixed_point_is_refused_for_a_committee_and_a_model_without_moments(
    trained, tmp_path
):
    spec = FixedPointSpec(variable="magmom")
    path = trained.checkpoint_path
    with pytest.raises(ValueError, match="committee"):
        MACECalculator(model_paths=[path, path], fixed_point=spec)
    energy_model = water_foundation(tmp_path)
    with pytest.raises(ValueError, match="already relaxed"):
        MACECalculator(model_paths=energy_model, fixed_point=spec)


def test_a_hessian_through_the_fixed_point_is_refused(trained):
    spec = FixedPointSpec(variable="magmom", require_convergence=False)
    calculator = MACECalculator(model_paths=trained.checkpoint_path, fixed_point=spec)
    with pytest.raises(NotImplementedError, match="second derivative"):
        calculator.get_hessian(iron_cluster())


def test_a_configured_augmentation_reaches_the_training_batches_alone(tmp_path):
    base = configuration(tmp_path)
    config = base.model_copy(
        update={
            "data": base.data.model_copy(
                update={
                    "augmentations": (
                        AugmentationSpec(name="magnetic_moments", settings={}),
                    ),
                    "valid_fraction": 0.5,
                }
            )
        }
    )
    data = run_data_stage(config, DEFAULT_CATALOGUE)
    torch.manual_seed(0)

    def moments_of(batches):
        return torch.cat([batch.graph["magmom"] for batch in batches])

    def epoch(number):
        return moments_of(data.train_loader.batches(number, drop_last=False))

    # The same epoch twice, so the order is the same and only the draws differ.
    first, second = epoch(0), epoch(0)
    lengths = torch.linalg.vector_norm
    assert not torch.allclose(first, second)
    assert torch.allclose(
        lengths(first, dim=-1).sort().values, lengths(second, dim=-1).sort().values
    )
    for loaders in (data.valid_loaders, data.train_eval_loaders):
        loader = loaders["default"]
        assert torch.equal(moments_of(loader), moments_of(loader))


# ---------------------------------------------------------------------------
# A fine-tune from a magnetic foundation
# ---------------------------------------------------------------------------


def irons(path, count=6, seed=7):
    generator = np.random.default_rng(seed)
    frames = []
    for _ in range(count):
        atoms = Atoms(
            "Fe3", positions=CLUSTER[1:] + generator.normal(scale=0.05, size=(3, 3))
        )
        atoms.arrays["REF_magmom"] = generator.normal(scale=1.5, size=(3, 3))
        atoms.info["REF_energy"] = -18.0 + generator.normal()
        atoms.arrays["REF_forces"] = generator.normal(scale=0.1, size=(3, 3))
        atoms.arrays["REF_magforces"] = generator.normal(scale=0.1, size=(3, 3))
        frames.append(atoms)
    write(path, frames)
    return path


def fine_tune(directory, foundation, train_file, element_table="foundation"):
    return ResolvedConfig.model_validate(
        {
            "runtime": {"work_dir": str(directory), "seed": 4},
            "finetune": {
                "foundation_model": str(foundation),
                "element_table": element_table,
            },
            "data": {
                "heads": {
                    "new": {
                        "train_file": str(train_file),
                        "e0s": {"kind": "foundation"},
                    }
                },
                "valid_fraction": 0.2,
                "pin_memory": False,
            },
            "model": {"observables": ["energy", "forces", "magforces"]},
            "training": {"max_num_epochs": 1, "batch_size": 4},
        }
    )


def iron_graph(numbers_table):
    positions = CLUSTER[1:]
    neighborhood = get_neighborhood(positions, 4.0, (False, False, False), None)
    return {
        "positions": torch.tensor(positions),
        "atomic_numbers": torch.tensor([26, 26, 26]),
        "element_index": torch.tensor([numbers_table.index(26)] * 3),
        "edge_index": torch.tensor(neighborhood.edge_index),
        "shifts": torch.tensor(neighborhood.shifts),
        "unit_shifts": torch.tensor(neighborhood.unit_shifts),
        "cell": torch.tensor(np.asarray(neighborhood.cell, dtype=float)).reshape(
            1, 3, 3
        ),
        "batch": torch.zeros(3, dtype=torch.long),
        "num_graphs": 1,
        "head": torch.zeros(1, dtype=torch.long),
        "magmom": torch.tensor(moments(9)[:3]),
    }


def test_a_fine_tune_inherits_the_moment_architecture(trained, tmp_path):
    assert trained.checkpoint_path is not None
    config = fine_tune(tmp_path, trained.checkpoint_path, irons(tmp_path / "fe.xyz"))
    assert config.model.magnetic == MagneticConfig()
    built = build(config)
    inherited = built.metadata.config.resolved["model"]
    assert inherited["model"] == "magnetic"
    assert inherited["magnetic"]["lmax"] == 2
    assert inherited["magnetic"]["num_basis"] == 6
    assert inherited["magnetic"]["one_body"] is True


@pytest.mark.parametrize("element_table", ["foundation", "data"])
def test_a_fine_tune_starts_from_the_foundation_s_moment_terms(
    trained, tmp_path, element_table
):
    """Forces and magnetic forces before any training are the foundation's, and
    so is the one-body term, which the frozen tree leaves at a random draw."""
    assert trained.checkpoint_path is not None
    config = fine_tune(
        tmp_path, trained.checkpoint_path, irons(tmp_path / "fe.xyz"), element_table
    )
    built = build(config)
    table = list(built.data.z_table.zs)
    assert table == ([8, 26] if element_table == "foundation" else [26])
    before = built.model(iron_graph(table), compute=("forces", "magforces"))
    parent = trained.model(iron_graph(NUMBERS), compute=("forces", "magforces"))
    assert torch.allclose(before.forces, parent.forces, atol=1e-12)
    assert torch.allclose(
        before.extras["magforces"], parent.extras["magforces"], atol=1e-12
    )
    kept = [NUMBERS.index(z) for z in table]
    assert torch.equal(
        one_body_of(built.model).coefficients,
        one_body_of(trained.model).coefficients[kept],
    )
    saturation = cast(
        torch.Tensor, built.model.get_submodule("backbone.backbone.moments").saturation
    )
    assert torch.equal(saturation, torch.tensor([1.2, 4.5], dtype=torch.float64)[kept])


def test_saturations_given_in_the_table_s_order_survive_a_smaller_table(tmp_path):
    previous = torch.get_default_dtype()
    torch.set_default_dtype(torch.float64)
    try:
        (tmp_path / "foundation").mkdir()
        base = configuration(tmp_path / "foundation")
        listed = base.model_copy(
            update={
                "model": base.model.model_copy(
                    update={
                        "magnetic": base.model.magnetic.model_copy(
                            update={"saturation": (1.2, 4.5)}
                        )
                    }
                )
            }
        )
        foundation = run(listed).checkpoint_path
        assert foundation is not None
        config = fine_tune(tmp_path, foundation, irons(tmp_path / "fe.xyz"), "data")
        built = build(config)
        saturation = cast(
            torch.Tensor,
            built.model.get_submodule("backbone.backbone.moments").saturation,
        )
        assert torch.equal(saturation, torch.tensor([4.5], dtype=torch.float64))
    finally:
        torch.set_default_dtype(previous)
