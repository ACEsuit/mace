"""ELEC-4: the exported weights, the two frozen references and live LES."""

from __future__ import annotations

import json

import numpy as np
import pytest
import torch
from ase.io import read, write
from ase.stress import full_3x3_to_voigt_6_stress
from mace_core.config.model import ModelConfig
from mace_core.config.resolved import ResolvedConfig
from mace_core.observables import DEFAULT_CATALOGUE
from mace_core.weights.neutral_format import read_neutral, write_neutral
from mace_torch.calculators import MACECalculator
from mace_torch.cli.eval_configs import main as evaluate
from mace_torch.cli.training_config import legacy_document
from mace_torch.deploy.loader import load_deployed
from mace_torch.deploy.neutral_io import import_neutral
from mace_torch.deploy.reference import _evaluate, verify_against_reference
from mace_torch.serialization import canonical_state
from mace_torch.train import write_model
from mace_torch.train.optimizers import parameter_groups

from tests.golden import harness, maceles_surfaces
from tests.golden.build_maceles_anchor import build_model, load_les_arguments
from tests.parity.test_fm_convert_legacy import GOLDEN, extract_here

pytestmark = pytest.mark.les
ROW = harness.FP64_CPU_REFERENCE
ANCHOR = GOLDEN / "models/tiny_maceles.model"


@pytest.fixture(scope="module")
def converted_les(tmp_path_factory):
    path = tmp_path_factory.mktemp("les") / "anchor"
    return extract_here(ANCHOR, path)


def test_native_model_matches_legacy_and_keeps_every_parameter(
    converted_les, fp64, isolated, tmp_path
):
    imported = import_neutral(converted_les, DEFAULT_CATALOGUE)
    legacy = maceles_surfaces.ModelSurface(maceles_surfaces.load_anchor(ANCHOR))
    reference = GOLDEN / "references/tiny_maceles_e3nn_cpu_fp64.json"
    report = verify_against_reference(
        imported.engine,
        reference,
        GOLDEN / "fixtures",
        z_table=imported.z_table,
        cutoff=imported.config.model.r_max,
        atol=ROW.atol,
        rtol=ROW.rtol,
    )
    assert report.passed, report.describe()
    expected_names = {
        "energy",
        "forces",
        "stress",
        "les_energy",
        "latent_charges",
        "latent_dipoles",
        "latent_alphas",
        "latent_kappas",
        "latent_quads",
        "BEC",
    }
    assert {row.quantity for row in report.deviations} == expected_names
    for name, atoms in harness.load_fixtures(
        names=("water_cluster", "triclinic_bulk")
    ).items():
        ref = legacy.golden_outputs(atoms)
        out = _evaluate(
            imported.engine,
            atoms,
            z_table=imported.z_table,
            head=0,
            cutoff=3.5,
            quantities=tuple(ref),
        )
        for key, value in ref.items():
            actual = out[key][0] if key == "les_energy" else out[key]
            assert np.shape(actual) == np.shape(value), (name, key)
            np.testing.assert_allclose(
                actual,
                value,
                atol=ROW.atol,
                rtol=ROW.rtol,
                err_msg=f"{name}/{key}",
            )
    groups = parameter_groups(imported.engine, imported.config.training)
    params = [id(p) for group in groups for p in group["params"]]
    assert len(params) == len(set(params))
    assert set(params) == {
        id(p) for p in imported.engine.parameters() if p.requires_grad
    }

    document = read_neutral(converted_les)
    roundtrip = tmp_path / "roundtrip"
    write_neutral(roundtrip, document.sidecar, document.tensors)
    for suffix in (".safetensors", ".json"):
        assert (
            roundtrip.with_suffix(suffix).read_bytes()
            == converted_les.with_suffix(suffix).read_bytes()
        )
    saved = write_model(tmp_path / "converted", imported.engine, imported.metadata)
    before, after = (
        canonical_state(imported.engine),
        canonical_state(load_deployed(saved).engine),
    )
    assert before.keys() == after.keys()
    for path, op in before.items():
        assert op.keys() == after[path].keys()
        for key, tensor in op.items():
            assert torch.equal(tensor, after[path][key])


@pytest.mark.parametrize("padded", [False, True])
def test_ase_bec_field_and_unpadding(converted_les, tmp_path, padded, fp64, isolated):
    imported = import_neutral(converted_les, DEFAULT_CATALOGUE)
    path = write_model(tmp_path / "converted", imported.engine, imported.metadata)
    calc = MACECalculator(
        path,
        compute_bec=True,
        **maceles_surfaces.FIELD_SETTINGS,
        **({"pad_num_atoms": 12, "pad_num_edges": 180} if padded else {}),
    )
    ref = harness.load_reference(GOLDEN / "references/tiny_maceles_field_cpu_fp64.json")
    for name, atoms in harness.load_fixtures(names=tuple(ref["fixtures"])).items():
        calc.calculate(atoms)
        expected = ref["fixtures"][name]["outputs"]
        aliases = {
            "bec": "BEC",
            "LES_alphas": "latent_alphas",
            "LES_kappas": "latent_kappas",
        }
        for key in (
            "bec",
            "LES_alphas",
            "LES_kappas",
            "energy",
            "energies",
            "node_energy",
            "forces",
            "stress",
        ):
            value = np.asarray(expected[aliases.get(key, key)]["value"])
            if key == "stress":
                value = full_3x3_to_voigt_6_stress(value)
            np.testing.assert_allclose(
                calc.results[key],
                value,
                atol=ROW.atol,
                rtol=ROW.rtol,
                err_msg=f"{name}/{key}",
            )
        independent = maceles_surfaces.bec_force_correction(
            calc.results["bec"],
            calc.results["LES_alphas"],
            atoms.get_volume(),
            maceles_surfaces.FIELD_SETTINGS,
        )
        np.testing.assert_allclose(
            calc.results["field_forces"], independent, atol=ROW.atol, rtol=0
        )


@pytest.mark.parametrize("request_bec", [False, True])
def test_eval_writes_real_latents_and_bec_from_a_batch(
    converted_les, fp64, isolated, tmp_path, request_bec
):
    imported = import_neutral(converted_les, DEFAULT_CATALOGUE)
    path = write_model(tmp_path / "native", imported.engine, imported.metadata)
    structures = harness.load_fixtures(names=("triclinic_bulk", "water_cluster"))
    source, dest = tmp_path / "input.xyz", tmp_path / "out.xyz"
    write(source, list(structures.values()), format="extxyz")
    evaluate(
        [
            "--model",
            str(path),
            "--configs",
            str(source),
            "--output",
            str(dest),
            "--info_prefix",
            "les_",
            "--batch_size",
            "2",
            *(["--compute_bec"] if request_bec else []),
        ]
    )
    refs = harness.load_reference(
        GOLDEN / "references/tiny_maceles_e3nn_cpu_fp64.json"
    )["fixtures"]
    for name, frame in zip(structures, read(dest, ":"), strict=True):
        assert ("les_BEC" in frame.arrays) == request_bec
        for key in (
            *(["BEC"] if request_bec else []),
            "latent_charges",
            "latent_dipoles",
            "latent_quads",
            "latent_kappas",
            "latent_alphas",
        ):
            values = np.asarray(refs[name]["outputs"][key]["value"])
            np.testing.assert_allclose(
                frame.arrays["les_" + key].reshape(len(frame), -1),
                values.reshape(len(frame), -1),
                atol=ROW.atol,
                rtol=0,
            )


def test_inactive_atomwise_flag_survives_as_source_provenance(fp64, isolated, tmp_path):
    from pydantic import ValidationError

    with pytest.raises(ValidationError, match="use_atomwise"):
        ModelConfig(model="les", les={"use_atomwise": True})
    original = build_model(dict(load_les_arguments(), use_atomwise=True))
    source = tmp_path / "old.model"
    torch.save(original, source)
    neutral = extract_here(source, tmp_path / "old")
    assert json.loads(neutral.read_text())["config"]["les_arguments"]["use_atomwise"]
    imported = import_neutral(neutral, DEFAULT_CATALOGUE)
    assert imported.config.model.les.use_atomwise is False
    assert "use_atomwise=True" in imported.metadata.notes
    path = write_model(tmp_path / "imported", imported.engine, imported.metadata)
    restored = load_deployed(path)
    assert "use_atomwise=True" in restored.metadata.notes
    assert restored.config.model.les.use_atomwise is False
    assert not any(
        name.startswith("les.atomwise.") for name, _ in original.named_parameters()
    )
    disabled = build_model(dict(load_les_arguments(), use_atomwise=False))
    for atoms in harness.load_fixtures(
        names=("water_cluster", "triclinic_bulk")
    ).values():
        enabled_outputs = maceles_surfaces.ModelSurface(original).golden_outputs(atoms)
        disabled_outputs = maceles_surfaces.ModelSurface(disabled).golden_outputs(atoms)
        native = _evaluate(
            restored.engine,
            atoms,
            z_table=restored.z_table,
            head=0,
            cutoff=3.5,
            quantities=tuple(enabled_outputs),
        )
        assert enabled_outputs.keys() == disabled_outputs.keys()
        for key, value in enabled_outputs.items():
            np.testing.assert_array_equal(disabled_outputs[key], value, err_msg=key)
            actual = native[key][0] if key == "les_energy" else native[key]
            np.testing.assert_allclose(
                actual, value, atol=ROW.atol, rtol=ROW.rtol, err_msg=key
            )


@pytest.mark.parametrize(
    "settings,degree",
    [
        (
            {
                "use_quad": False,
                "use_dipole": False,
                "use_induced_dipole": False,
                "compute_bec": False,
            },
            1,
        ),
        ({"use_anisotropic_polarizability": True}, 2),
        (
            {
                "use_anisotropic_polarizability": True,
                "alpha_1o_nonlinear_readout": True,
            },
            1,
        ),
    ],
)
def test_other_readout_paths_and_requested_bec(
    settings, degree, tmp_path, fp64, isolated, monkeypatch
):
    from tests.golden.build_maceles_anchor import ANCHOR_CONFIG
    from tests.golden.maceles_surfaces import graph_for

    if degree == 2:
        monkeypatch.setitem(ANCHOR_CONFIG, "hidden_irreps", "4x0e+4x1o+4x2e")
    args = dict(load_les_arguments(), **settings)
    old = build_model(args)
    legacy_path = tmp_path / "original.model"
    torch.save(old, legacy_path)
    imported = import_neutral(
        extract_here(legacy_path, tmp_path / "neutral"), DEFAULT_CATALOGUE
    )
    atoms = harness.load_fixtures(names=["water_cluster"])["water_cluster"]
    legacy = old(graph_for(old, atoms), compute_bec=True)
    got = _evaluate(
        imported.engine,
        atoms,
        z_table=imported.z_table,
        head=0,
        cutoff=3.5,
        quantities=("forces", "BEC"),
    )
    for key in (
        "forces",
        "BEC",
        "latent_quads",
        "latent_alphas",
        "latent_dipoles",
        "latent_kappas",
    ):
        if legacy[key] is None:
            assert key not in got
        else:
            np.testing.assert_allclose(
                got[key],
                legacy[key].detach().numpy(),
                atol=ROW.atol,
                rtol=0,
                err_msg=key,
            )
    if not args["use_dipole"] and not args["use_induced_dipole"]:
        assert got["BEC"].shape == (len(atoms), 3, 3)
        crystal = harness.load_fixtures(names=["triclinic_bulk"])["triclinic_bulk"]
        source_calc = maceles_surfaces.field_calculator(old)
        saved = write_model(
            tmp_path / "charge_only", imported.engine, imported.metadata
        )
        calc = MACECalculator(
            saved, compute_bec=True, **maceles_surfaces.FIELD_SETTINGS
        )
        source_calc.calculate(crystal)
        calc.calculate(crystal)
        raw = (
            old(
                graph_for(
                    old, crystal, maceles_surfaces.FIELD_SETTINGS["external_field"]
                ),
                compute_bec=True,
            )["BEC"]
            .detach()
            .numpy()
        )
        assert calc.results["bec"].shape == (len(crystal), 3, 3)
        np.testing.assert_allclose(
            calc.results["bec"], raw, atol=ROW.atol, rtol=ROW.rtol
        )
        for key in ("bec", "forces", "LES_kappas"):
            np.testing.assert_allclose(
                calc.results[key],
                source_calc.results[key],
                atol=ROW.atol,
                rtol=ROW.rtol,
                err_msg=key,
            )


def test_energy_foundation_transfer_matches_legacy(tmp_path, fp64, isolated):
    """Align fresh LES heads, then independently transfer the same energy weights."""
    from types import SimpleNamespace

    from e3nn import o3
    from mace_torch.finetune.transfer import transfer_foundation

    from mace import modules
    from mace.tools.finetuning_utils import load_foundations_elements
    from mace.tools.model_script_utils import _build_model
    from mace.tools.scripts_utils import extract_config_mace_model
    from mace.tools.utils import AtomicNumberTable

    recipe = extract_config_mace_model(
        maceles_surfaces.load_anchor(GOLDEN / "models/tiny_scaleshift.model")
    )
    recipe["hidden_irreps"] = o3.Irreps("8x0e")
    donor = modules.ScaleShiftMACE(**recipe)
    settings = {"compute_bec": True, "use_induced_charge": True}
    target = _build_model(
        SimpleNamespace(model="FoundationMACELES", les_arguments=settings),
        {},
        recipe,
        donor.heads,
    )

    def convert(model, name):
        source = tmp_path / f"{name}.model"
        torch.save(model, source)
        return import_neutral(extract_here(source, tmp_path / name), DEFAULT_CATALOGUE)

    foundation = convert(donor, "energy")
    native = convert(target, "before_transfer")
    numbers = foundation.z_table.zs
    load_foundations_elements(
        target,
        donor,
        table=AtomicNumberTable(list(numbers)),
        load_readout=True,
        max_L=0,
    )
    transfer_foundation(
        foundation.engine.backbone,
        native.engine.backbone,
        foundation_elements=numbers,
        elements=numbers,
        foundation_heads=foundation.heads,
        heads=native.heads,
        readout_from={head: head for head in native.heads},
    )
    for atoms in harness.load_fixtures(
        names=("water_cluster", "triclinic_bulk")
    ).values():
        old = target(maceles_surfaces.graph_for(target, atoms), compute_bec=True)
        out = _evaluate(
            native.engine,
            atoms,
            z_table=native.z_table,
            head=0,
            cutoff=float(donor.r_max),
        )
        for key in (
            "energy",
            "forces",
            "les_energy",
            "latent_charges",
            "latent_kappas",
            "BEC",
        ):
            expected = old[key].detach().numpy()
            if key == "energy":
                expected = expected[0]
            np.testing.assert_allclose(
                out[key], expected, atol=ROW.atol, rtol=ROW.rtol, err_msg=key
            )


def test_legacy_les_arguments_use_the_typed_section(tmp_path):
    path = tmp_path / "options.yaml"
    path.write_text("compute_bec: true\nsigma: 1.25\n")
    doc = legacy_document({"model": "MACELES", "les_arguments": str(path)})
    config = ResolvedConfig.model_validate(doc)
    assert config.model.model == "les" and config.model.les is not None
    assert config.model.les.sigma == 1.25
    assert config.model.les.compute_bec
    path.write_text("use_atomwise: true\n")
    for document in (
        {"model": {"model": "les", "les": {"use_atomwise": True}}},
        legacy_document({"model": "MACELES", "les_arguments": str(path)}),
    ):
        with pytest.raises(ValueError, match="The model readouts supply the charges"):
            ResolvedConfig.model_validate(document)
    with pytest.raises(ValueError, match="extra"):
        ResolvedConfig.model_validate(
            {"model": {"model": "les", "les": {"misspelled": True}}}
        )
    with pytest.raises(ValueError, match="keep_last_layer_irreps"):
        ResolvedConfig.model_validate(
            {"model": {"model": "les", "keep_last_layer_irreps": False}}
        )


@pytest.mark.gpu
@pytest.mark.cueq
def test_cueq_force_training_and_checkpoint_on_cuda(tmp_path, fp64, isolated):
    from mace_torch.cli.run_train import run

    config = ResolvedConfig.model_validate(
        {
            "runtime": {"device": "cuda", "work_dir": str(tmp_path)},
            "model": {
                "model": "les",
                "backend": "cueq",
                "num_channels": 4,
                "hidden_irreps": "0e+1o",
                "correlation": 2,
                "max_ell": 1,
                "readout": {"mlp_irreps": "4x0e"},
                "les": {"use_dipole": True, "use_induced_charge": True},
            },
            "data": {
                "heads": {
                    "default": {
                        "train_file": str(GOLDEN / "fixtures/tiny_train.xyz"),
                        "e0s": {"kind": "isolated_atoms"},
                    }
                },
                "valid_fraction": 0.2,
            },
            "training": {"max_num_epochs": 1, "batch_size": 3, "valid_batch_size": 3},
            "loss": {"weights": {"energy": 1.0, "forces": 10.0}},
        }
    )
    result = run(config)
    assert all(np.isfinite(epoch.train_loss) for epoch in result.history)
    calc = MACECalculator(result.checkpoint_path, device="cuda", compute_bec=True)
    assert any(
        type(op).__module__ == "mace_torch.backends.cueq"
        for op in calc.models[0].engine.modules()
    )
    atoms = harness.load_fixtures(names=["water_cluster"])["water_cluster"]
    calc.calculate(atoms)
    for name in ("energy", "forces", "bec", "LES_kappas"):
        assert np.isfinite(calc.results[name]).all(), name


@pytest.mark.gpu
@pytest.mark.parametrize(
    "backend", ["reference", pytest.param("cueq", marks=pytest.mark.cueq)]
)
@pytest.mark.parametrize(
    "dtype,row",
    [("float64", harness.FP64_ACCELERATED_BACKEND), ("float32", harness.FP32)],
)
def test_les_cuda_reference_rows(converted_les, backend, dtype, row, fp64, isolated):
    from mace_core.config.precision import PrecisionConfig
    from mace_core.data.backend import DatasetStatistics
    from mace_core.elements import ResolvedE0s
    from mace_torch.serialization import load_canonical_state
    from mace_torch.train.model_stage import build_model

    precision = getattr(PrecisionConfig, dtype)()
    imported = import_neutral(converted_les, DEFAULT_CATALOGUE, precision=precision)
    engine = imported.engine
    if backend == "cueq":
        config = imported.config.model_copy(
            update={
                "model": imported.config.model.model_copy(update={"backend": backend})
            }
        )
        engine, _ = build_model(
            config,
            DEFAULT_CATALOGUE,
            z_table=imported.z_table,
            heads=imported.heads,
            e0s=ResolvedE0s(
                {h: {z: 0.0 for z in imported.z_table.zs} for h in imported.heads}
            ),
            statistics=DatasetStatistics(),
            precision=precision,
            initialize=False,
        )
        load_canonical_state(engine, canonical_state(imported.engine))
        assert any(
            type(op).__module__ == "mace_torch.backends.cueq" for op in engine.modules()
        )
    report = verify_against_reference(
        engine.to("cuda"),
        GOLDEN / "references/tiny_maceles_e3nn_cpu_fp64.json",
        GOLDEN / "fixtures",
        z_table=imported.z_table,
        cutoff=imported.config.model.r_max,
        atol=row.atol,
        rtol=row.rtol,
        device="cuda",
    )
    assert report.passed, report.describe()
    assert len({entry.fixture for entry in report.deviations}) == 6
