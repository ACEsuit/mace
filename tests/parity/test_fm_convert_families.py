"""FM-5: the shipped converter on response and magnetic checkpoints.

Every comparison goes through extraction and neutral import, rather than the
test-only transfer helpers. Existing anchors and tolerances are read unchanged.
"""

import numpy as np
import pytest
import torch
from mace_core.observables import DEFAULT_CATALOGUE
from mace_core.weights.neutral_format import read_neutral, write_neutral
from mace_torch.data.inputs import graph_inputs_of
from mace_torch.deploy.loader import load_deployed
from mace_torch.deploy.neutral_io import import_neutral
from mace_torch.deploy.reference import verify_against_reference
from mace_torch.serialization import canonical_state
from mace_torch.train import write_model

from tests.golden.harness import FP64_CPU_REFERENCE as ROW
from tests.parity.test_dipole_parity import (
    CASES,
    legacy_anchor,
    legacy_batch,
    legacy_dielectric,
    v1_graph,
)
from tests.parity.test_energy_dipole_parity import inputs, legacy_model
from tests.parity.test_fm_convert_legacy import (
    EXTRACTOR,
    FIXTURES,
    GOLDEN,
    extract_here,
)


def converted(model, tmp_path):
    source = tmp_path / "legacy.model"
    torch.save(model, source)
    sidecar = extract_here(source, tmp_path / "neutral")
    return import_neutral(sidecar, DEFAULT_CATALOGUE), sidecar


def close(actual, expected):
    torch.testing.assert_close(
        actual.detach(), expected.detach(), atol=ROW.atol, rtol=ROW.rtol
    )


@pytest.mark.parametrize("family", ["dipole", "dielectric", "energy_dipole"])
@pytest.mark.parametrize("charged", [False, True])
def test_fm_convert_dipole_values_and_position_gradients(
    fp64, isolated, tmp_path, family, charged
):
    if family == "energy_dipole":
        legacy = legacy_model()
        _, old, graph = inputs(legacy, charged)
    else:
        legacy = (
            legacy_anchor()
            if family == "dipole"
            else legacy_dielectric("4x0e+4x1o+4x2e")
        )
        atoms, charges, charge = CASES["charged pair" if charged else "water"]
        old = legacy_batch(legacy, atoms, charges, charge)
        graph = v1_graph(
            atoms, legacy.atomic_numbers.tolist(), float(legacy.r_max), charges, charge
        )
    imported, _ = converted(legacy, tmp_path)
    graph["positions"].requires_grad_(True)
    expected = legacy(old, training=True)
    got = imported.engine(
        graph, training=True, compute=("forces",) if family == "energy_dipole" else ()
    )
    for quantity in (
        ["dipole", "polarizability", "charges", "atomic_dipoles"]
        if family == "dielectric"
        else ["dipole"]
    ):
        close(got.get(quantity), expected[quantity])
    old_gradient = torch.autograd.grad(
        expected["dipole"].sum(), old["positions"], retain_graph=True
    )[0]
    new_gradient = torch.autograd.grad(
        got.get("dipole").sum(), graph["positions"], retain_graph=True
    )[0]
    close(new_gradient, old_gradient)
    if family == "energy_dipole":
        close(got.total_energy, expected["energy"])
        close(got.forces, expected["forces"])


def test_fm_convert_dipole_committed_reference(fp64, tmp_path):
    imported, _ = converted(legacy_anchor(), tmp_path)
    report = verify_against_reference(
        imported.engine,
        GOLDEN / "references/tiny_dipoles_e3nn_cpu_fp64.json",
        FIXTURES,
        z_table=imported.z_table,
        cutoff=imported.config.model.r_max,
        graph_inputs=graph_inputs_of(imported.config.model),
    )
    assert report.passed, report.describe()
    assert {d.quantity for d in report.deviations} == {"dipole", "atomic_dipoles"}


@pytest.mark.magnetic
def test_fm_convert_magnetic_base_scope(fp64, isolated, tmp_path):
    """The base owns shared state but provides no executable model to convert."""
    import inspect

    from mace.modules.extensions import MagneticMACE, MagneticScaleShiftMACE
    from mace.tools.scripts_utils import extract_config_mace_model
    from tests.golden.magnetic_surfaces import load_anchor

    concrete = load_anchor()
    assert type(concrete) is MagneticScaleShiftMACE
    config = extract_config_mace_model(concrete)
    parameters = inspect.signature(MagneticMACE).parameters
    base = MagneticMACE(
        **{key: value for key, value in config.items() if key in parameters}
    )
    base_state, concrete_state = base.state_dict(), concrete.state_dict()
    assert base_state.keys() < concrete_state.keys()
    for name, value in base_state.items():
        assert value.shape == concrete_state[name].shape, name
    base.load_state_dict({name: concrete_state[name] for name in base_state})
    with pytest.raises(NotImplementedError):
        base({})
    source = tmp_path / "base.model"
    torch.save(base, source)
    output = tmp_path / "refused"
    with pytest.raises(
        EXTRACTOR["ExtractionError"], match="forward raises NotImplementedError"
    ):
        extract_here(source, output)
    assert not output.with_suffix(".json").exists()
    assert not output.with_suffix(".safetensors").exists()


@pytest.mark.magnetic
def test_fm_convert_magnetic_goldens_and_live_legacy(fp64, tmp_path):
    from tests.golden import magnetic_surfaces as surfaces
    from tests.parity.test_magnetic_parity import evaluate

    legacy = surfaces.load_anchor()
    imported, _ = converted(legacy, tmp_path)
    for atoms in surfaces.magnetic_fixtures().values():
        expected = legacy(
            surfaces.build_batch(legacy, atoms),
            compute_force=True,
            compute_magforces=True,
        )
        got = evaluate(
            imported.engine, legacy, atoms, atoms.arrays[surfaces.MAGMOM_KEY]
        )
        for actual, source in [
            (got.total_energy, "energy"),
            (got.forces, "forces"),
            (got.node_energies, "node_energy"),
            (got.extras["magforces"], "magforces"),
        ]:
            close(actual, expected[source])
    report = verify_against_reference(
        imported.engine,
        GOLDEN / "references/tiny_magnetic_e3nn_cpu_fp64.json",
        FIXTURES,
        z_table=imported.z_table,
        cutoff=imported.config.model.r_max,
        graph_inputs=("magmom",),
    )
    assert report.passed, report.describe()
    assert "magforces" in {d.quantity for d in report.deviations}


@pytest.mark.parametrize(
    "family",
    [
        "dipole",
        "dielectric",
        "energy_dipole",
        pytest.param("magnetic", marks=pytest.mark.magnetic),
    ],
)
def test_fm_convert_family_checkpoint_and_neutral_roundtrips(
    fp64, isolated, tmp_path, family
):
    if family == "magnetic":
        from tests.golden.magnetic_surfaces import load_anchor

        legacy = load_anchor()
    else:
        legacy = {
            "dipole": legacy_anchor,
            "energy_dipole": legacy_model,
            "dielectric": lambda: legacy_dielectric("4x0e+4x1o+4x2e"),
        }[family]()
    imported, sidecar = converted(legacy, tmp_path)
    checkpoint = write_model(tmp_path / "native", imported.engine, imported.metadata)
    reloaded = load_deployed(checkpoint)
    before, after = canonical_state(imported.engine), canonical_state(reloaded.engine)
    assert before.keys() == after.keys()
    for op, tensors in before.items():
        for name, value in tensors.items():
            assert torch.equal(value, after[op][name])
    # The neutral artifact itself remains lossless. Projection of full-basis
    # weights into v1 is a separate, functional equivalence (not invertible).
    artifact = read_neutral(sidecar)
    copy = write_neutral(tmp_path / "copy", artifact.sidecar, artifact.tensors)
    assert artifact.sidecar == read_neutral(copy).sidecar
    assert (
        sidecar.with_suffix(".safetensors").read_bytes()
        == copy.with_suffix(".safetensors").read_bytes()
    )


@pytest.mark.parametrize(
    "family",
    [
        "dipole",
        "dielectric",
        "energy_dipole",
        pytest.param("magnetic", marks=pytest.mark.magnetic),
    ],
)
def test_fm_convert_family_unknown_weight_is_refused(fp64, isolated, tmp_path, family):
    if family == "magnetic":
        from tests.golden.magnetic_surfaces import load_anchor

        legacy = load_anchor()
    else:
        legacy = {
            "dipole": legacy_anchor,
            "energy_dipole": legacy_model,
            "dielectric": lambda: legacy_dielectric("4x0e+4x1o+4x2e"),
        }[family]()
    legacy.register_parameter("unmapped_weight", torch.nn.Parameter(torch.ones(1)))
    with pytest.raises(EXTRACTOR["ExtractionError"], match="unmapped_weight"):
        converted(legacy, tmp_path)


@pytest.mark.magnetic
def test_fm_convert_magnetic_scf_preserves_configuration_and_goldens(fp64, tmp_path):
    from mace_torch.calculators import MACECalculator
    from mace_torch.physics.fixed_point import FixedPointDriver

    from tests.golden import magnetic_surfaces as surfaces
    from tests.parity.test_magnetic_parity import v1_graph as magnetic_graph

    legacy = surfaces.load_scf_anchor()
    imported, sidecar = converted(legacy, tmp_path)
    assert read_neutral(sidecar).sidecar.provenance.source_class == "MagneticSCFMACE"
    assert isinstance(imported.engine, FixedPointDriver)
    spec = imported.config.fixed_point
    assert (spec.max_iter, spec.tolerance, spec.step_size, spec.collinear) == (
        legacy.n_scf_step,
        legacy.scf_tol,
        legacy.scf_step_size,
        legacy.use_collinear,
    )
    assert not spec.warm_start
    assert not spec.require_convergence
    checkpoint = write_model(tmp_path / "native", imported.engine, imported.metadata)
    deployed = load_deployed(checkpoint)
    assert deployed.config.fixed_point == spec
    assert isinstance(deployed.engine, FixedPointDriver)
    report = verify_against_reference(
        deployed.engine,
        GOLDEN / "references/tiny_magnetic_scf_e3nn_cpu_fp64.json",
        FIXTURES,
        z_table=deployed.z_table,
        cutoff=deployed.r_max,
        graph_inputs=("magmom",),
    )
    assert report.passed, report.describe()
    assert "equilibrated_magmom" in {d.quantity for d in report.deviations}
    calculator = MACECalculator(model_paths=checkpoint)
    for atoms in surfaces.scf_fixtures().values():
        expected = surfaces.MagneticSCFForward().golden_outputs(atoms)
        graph = magnetic_graph(
            atoms,
            list(deployed.z_table.zs),
            deployed.r_max,
            atoms.arrays[surfaces.MAGMOM_KEY],
        )
        got = deployed.compute(graph, compute=("forces",))
        for actual, name in [
            (got.total_energy[0], "energy"),
            (got.forces, "forces"),
            (got.extras["converged_magmom"], "equilibrated_magmom"),
        ]:
            np.testing.assert_allclose(
                actual.detach().numpy(), expected[name], atol=ROW.atol, rtol=ROW.rtol
            )
        calculator.calculate(atoms, properties=["energy", "forces"])
        np.testing.assert_allclose(
            calculator.results["energy"],
            expected["energy"],
            atol=ROW.atol,
            rtol=ROW.rtol,
        )
        np.testing.assert_allclose(
            calculator.results["MACE_magmoms"],
            expected["equilibrated_magmom"],
            atol=ROW.atol,
            rtol=ROW.rtol,
        )
    with pytest.raises(NotImplementedError, match="second derivative"):
        deployed.compute(graph, compute=("hessian",))
    assert (
        canonical_state(imported.engine.engine).keys()
        == canonical_state(deployed.derivative_engine).keys()
    )
    restored = canonical_state(deployed.derivative_engine)
    for path, values in canonical_state(imported.engine.engine).items():
        for name, value in values.items():
            assert torch.equal(value, restored[path][name])


@pytest.mark.magnetic
def test_fm_convert_magnetic_scf_precision_and_foundation(fp64, tmp_path):
    from mace_core.config.precision import PrecisionConfig
    from mace_torch.finetune.foundation import read_foundation
    from mace_torch.physics import DerivativeEngine
    from mace_torch.physics.fixed_point import FixedPointDriver

    from tests.golden import magnetic_surfaces as surfaces
    from tests.golden.harness import FP32
    from tests.parity.test_magnetic_parity import v1_graph as magnetic_graph

    imported, _ = converted(surfaces.load_scf_anchor(n_scf_step=3), tmp_path)
    checkpoint = write_model(tmp_path / "native", imported.engine, imported.metadata)
    deployed = load_deployed(checkpoint)
    rebuilt = deployed.with_precision(PrecisionConfig(model="float32"), device="cpu")
    assert isinstance(rebuilt.engine, FixedPointDriver)
    assert rebuilt.engine.spec == imported.config.fixed_point
    atoms = surfaces.scf_fixtures()["mag_fe_atom"]
    graph = magnetic_graph(
        atoms,
        list(deployed.z_table.zs),
        deployed.r_max,
        atoms.arrays[surfaces.MAGMOM_KEY],
    )
    expected = deployed.compute(graph, compute=()).total_energy
    graph32 = {
        key: value.float() if value.is_floating_point() else value
        for key, value in graph.items()
    }
    actual = rebuilt.compute(graph32, compute=()).total_energy
    torch.testing.assert_close(
        actual.double(), expected, atol=FP32.atol, rtol=FP32.rtol
    )
    foundation = read_foundation(checkpoint, DEFAULT_CATALOGUE)
    assert isinstance(foundation.engine, DerivativeEngine)
    assert foundation.model is foundation.engine.backbone


@pytest.mark.magnetic
def test_fm_convert_magnetic_scf_cached_state_requires_reset(fp64, tmp_path):
    from tests.golden.magnetic_surfaces import load_scf_anchor

    legacy = load_scf_anchor()
    legacy.cache_magmom = torch.ones(1, 3)
    with pytest.raises(EXTRACTOR["ExtractionError"], match="cache_magmom"):
        converted(legacy, tmp_path)


@pytest.mark.magnetic
@pytest.mark.parametrize("enabled,collinear", [(False, False), (True, True)])
def test_fm_convert_magnetic_scf_nondefault_settings(
    fp64, tmp_path, enabled, collinear
):
    from tests.golden import magnetic_surfaces as surfaces
    from tests.parity.test_magnetic_parity import v1_graph as magnetic_graph

    legacy = surfaces.load_scf_anchor(
        n_scf_step=3,
        scf_tol=1e-7,
        scf_step_size=0.7,
        use_scf=enabled,
        use_collinear=collinear,
    )
    imported, _ = converted(legacy, tmp_path)
    atoms = surfaces.scf_fixtures()["mag_fe_dimer_fm"]
    expected = legacy(
        surfaces.build_batch(legacy.magmom_mace, atoms), compute_force=True
    )
    graph = magnetic_graph(
        atoms,
        list(imported.z_table.zs),
        imported.config.model.r_max,
        atoms.arrays[surfaces.MAGMOM_KEY],
    )
    got = imported.engine(graph, compute=("forces",))
    close(got.total_energy, expected["energy"])
    close(got.forces, expected["forces"])
    if enabled:
        close(got.extras["converged_magmom"], expected["equilibrated_magmom"])
    else:
        assert imported.config.fixed_point is None


@pytest.mark.magnetic
def test_fm_convert_magnetic_config_matches_legacy_keys(fp64):
    from mace.tools.scripts_utils import extract_config_mace_model
    from tests.golden.magnetic_surfaces import load_anchor

    model = load_anchor()
    actual = EXTRACTOR["legacy_config"](model)
    expected = {
        key: EXTRACTOR["jsonable"](value)
        for key, value in extract_config_mace_model(model).items()
    }
    assert actual == expected


@pytest.mark.magnetic
def test_fm_convert_magnetic_wrapper_unknown_tensor_is_refused(fp64, tmp_path):
    from tests.golden.magnetic_surfaces import load_scf_anchor

    model = load_scf_anchor()
    model.register_buffer("unmapped_wrapper_state", torch.ones(1))
    with pytest.raises(EXTRACTOR["ExtractionError"], match="unmapped_wrapper_state"):
        converted(model, tmp_path)


@pytest.mark.magnetic
@pytest.mark.parametrize("layers", [1, 2])
def test_fm_convert_magnetic_without_one_body_or_repulsion(fp64, tmp_path, layers):
    from mace.modules.extensions import MagneticScaleShiftMACE
    from mace.tools.scripts_utils import extract_config_mace_model
    from tests.golden import magnetic_surfaces as surfaces
    from tests.parity.test_magnetic_parity import evaluate

    settings = extract_config_mace_model(surfaces.load_anchor())
    settings.update(
        use_magmom_one_body=False, pair_repulsion=False, num_interactions=layers
    )
    legacy = MagneticScaleShiftMACE(**settings)
    imported, _ = converted(legacy, tmp_path)
    assert imported.engine.backbone.one_body is None
    atoms = surfaces.magnetic_fixtures()["mag_fe3_canted"]
    expected = legacy(
        surfaces.build_batch(legacy, atoms), compute_force=True, compute_magforces=True
    )
    actual = evaluate(imported.engine, legacy, atoms, atoms.arrays[surfaces.MAGMOM_KEY])
    close(actual.total_energy, expected["energy"])
    close(actual.forces, expected["forces"])
    close(actual.extras["magforces"], expected["magforces"])


@pytest.mark.parametrize("family", ["dipole", "dielectric"])
@pytest.mark.parametrize("gate", ["silu", "tanh", "None"])
def test_fm_convert_dipole_gated_vectors(fp64, isolated, tmp_path, family, gate):
    import inspect

    from e3nn import o3

    from mace.modules.models import AtomicDipolesMACE
    from mace.tools.scripts_utils import extract_config_mace_model

    dielectric = legacy_dielectric("4x0e+4x1o+4x2e", gate=gate)
    if family == "dielectric":
        legacy = dielectric
    else:
        settings = extract_config_mace_model(dielectric)
        settings.pop("use_polarizability")
        settings.pop("only_dipole")
        settings["gate"] = {
            "silu": torch.nn.functional.silu,
            "tanh": torch.tanh,
            "None": None,
        }[gate]
        # A dipole-only readout has no scalar outputs; its middle is vectors.
        settings["MLP_irreps"] = o3.Irreps("4x1o")
        accepted = inspect.signature(AtomicDipolesMACE).parameters
        legacy = AtomicDipolesMACE(
            **{name: value for name, value in settings.items() if name in accepted}
        )
    imported, _ = converted(legacy, tmp_path)
    atoms, charges, charge = CASES["charged pair"]
    expected = legacy(legacy_batch(legacy, atoms, charges, charge), training=True)
    graph = v1_graph(
        atoms, legacy.atomic_numbers.tolist(), float(legacy.r_max), charges, charge
    )
    actual = imported.engine(graph, training=True, compute=())
    close(actual.dipole, expected["dipole"])
    close(actual.extras["atomic_dipoles"], expected["atomic_dipoles"])


@pytest.mark.parametrize(
    "family",
    ["dipole", "energy_dipole", pytest.param("magnetic", marks=pytest.mark.magnetic)],
)
def test_fm_convert_family_through_subprocess_and_native_loader(
    fp64, isolated, tmp_path, family
):
    import sys

    from mace_torch.deploy.legacy import convert_legacy

    if family == "energy_dipole":
        source = tmp_path / "joint.model"
        torch.save(legacy_model(), source)
    else:
        source = (
            GOLDEN
            / "models"
            / ("tiny_dipoles.model" if family == "dipole" else "tiny_magnetic.model")
        )
    result = convert_legacy(
        source, tmp_path / "converted", DEFAULT_CATALOGUE, python=sys.executable
    )
    deployed = load_deployed(result.checkpoint)
    assert deployed.metadata.conversion.projection_verified
    assert read_neutral(result.neutral).sidecar.family == family
    assert deployed.config == result.model.config
