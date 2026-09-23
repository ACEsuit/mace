"""The production converter, against the frozen tree it converts from.

The extraction script is the one that ships in the package. Here it runs in
this process against the in-tree legacy package, which is the development
mode, and as a subprocess, which is how it runs everywhere else; the two write
the same bytes. What it writes is imported with no legacy code at all, and the
result is held to the committed goldens and to the live legacy model.
"""

from __future__ import annotations

import json
import runpy
import sys
from pathlib import Path

import numpy as np
import pytest
import torch
from mace_core.clebsch_gordan.conversion import full_to_reduced
from mace_core.clebsch_gordan.reduced_basis import (
    reduced_symmetric_tensor_product_basis,
)
from mace_core.observables import load_default_catalogue
from mace_core.weights.neutral_format import read_neutral, write_neutral
from mace_torch.deploy.legacy import (
    ExtractionFailed,
    convert_legacy,
    extract,
    extractor_path,
)
from mace_torch.deploy.neutral_io import (
    NeutralImportError,
    import_neutral,
    isolated_atom_energies,
)
from mace_torch.deploy.reference import VerificationError, verify_against_reference
from mace_torch.finetune.foundation import read_foundation
from mace_torch.train import write_model

from tests.golden.harness import tolerance
from tests.parity.test_anchor_as_foundation import energies
from tests.parity.test_fm00_training_step import load_anchor

GOLDEN = Path(__file__).resolve().parents[1] / "golden"
FIXTURES = GOLDEN / "fixtures"
CATALOGUE = load_default_catalogue()
ANCHORS = ("tiny_scaleshift", "tiny_mace")
REFERENCE = tolerance("fp64_cpu_reference")

#: The extraction script's own functions, as the development mode runs them.
EXTRACTOR = runpy.run_path(str(extractor_path()))


def extract_here(source: Path, output: Path) -> Path:
    return EXTRACTOR["extract"](Path(source), Path(output))


@pytest.fixture(name="converted", params=ANCHORS)
def fixture_converted(request, fp64, tmp_path):
    name = request.param
    sidecar = extract_here(GOLDEN / "models" / f"{name}.model", tmp_path / name)
    return name, sidecar


def reference(name: str) -> Path:
    return GOLDEN / "references" / f"{name}_e3nn_cpu_fp64.json"


# ---------------------------------------------------------------------------
# The anchors, through the production tool
# ---------------------------------------------------------------------------


def test_a_converted_anchor_reproduces_its_goldens(converted):
    name, sidecar = converted
    imported = import_neutral(sidecar, CATALOGUE)
    report = verify_against_reference(
        imported.engine,
        reference(name),
        FIXTURES,
        z_table=imported.z_table,
        cutoff=imported.config.model.r_max,
        atol=REFERENCE.atol,
        rtol=REFERENCE.rtol,
    )
    assert report.passed, report.describe()
    assert {d.quantity for d in report.deviations} == {"energy", "forces", "stress"}


def test_a_converted_anchor_matches_the_live_legacy_model(converted):
    from tests.golden.anchors import anchor_batch, load_training_structures

    name, sidecar = converted
    legacy = load_anchor(f"{name}.model")
    structures = load_training_structures()
    expected = legacy(
        anchor_batch(legacy, structures, torch.float64).to_dict(),
        training=False,
        compute_force=False,
    )["energy"].detach()
    got = energies(import_neutral(sidecar, CATALOGUE).engine, legacy, structures)
    assert torch.allclose(got, expected, atol=1e-12, rtol=0), (
        f"largest difference {float((got - expected).abs().max()):.3g} eV"
    )


def test_a_subprocess_extraction_writes_the_same_tensors(fp64, tmp_path):
    """The mode every user runs and the one the tests run are one script."""
    source = GOLDEN / "models" / "tiny_scaleshift.model"
    here = read_neutral(extract_here(source, tmp_path / "here"))
    there = read_neutral(extract(source, tmp_path / "there", python=sys.executable))
    assert here.sidecar == there.sidecar
    assert here.tensors.keys() == there.tensors.keys()
    for key, value in here.tensors.items():
        assert np.array_equal(value, there.tensors[key]), key
        assert value.dtype == there.tensors[key].dtype, key


# ---------------------------------------------------------------------------
# The configuration, key for key
# ---------------------------------------------------------------------------


def test_the_configuration_has_every_key_the_legacy_reader_gives(fp64, tmp_path):
    from mace.tools.scripts_utils import extract_config_mace_model

    legacy = load_anchor("tiny_scaleshift.model")
    recorded = read_neutral(
        extract_here(GOLDEN / "models" / "tiny_scaleshift.model", tmp_path / "a")
    ).sidecar.config
    assert set(recorded) == set(extract_config_mace_model(legacy))
    assert len(recorded) == 33
    assert recorded["avg_num_neighbors"] == pytest.approx(
        float(legacy.interactions[0].avg_num_neighbors), abs=0
    )


def test_the_plain_class_has_the_same_keys_but_the_scale_and_shift(fp64, tmp_path):
    """The legacy reader refuses the plain class by its name, and every field
    it reads exists on it; what the plain class lacks is the scale-shift
    block, so that is all that is missing."""
    from mace.tools.scripts_utils import extract_config_mace_model

    scale_shift = set(extract_config_mace_model(load_anchor("tiny_scaleshift.model")))
    recorded = read_neutral(
        extract_here(GOLDEN / "models" / "tiny_mace.model", tmp_path / "a")
    ).sidecar.config
    assert set(recorded) == scale_shift - {"atomic_inter_scale", "atomic_inter_shift"}
    assert len(recorded) == 31


def rewritten(sidecar_path: Path, tmp_path: Path, name: str, **changes):
    artifact = read_neutral(sidecar_path)
    document = artifact.sidecar.model_dump(mode="json")
    for key, value in changes.items():
        document[key] = value(document[key])
    from mace_core.weights.neutral_format import NeutralSidecar

    return write_neutral(
        tmp_path / name, NeutralSidecar.model_validate(document), artifact.tensors
    )


def test_a_missing_field_fails_the_import(converted, tmp_path):
    _, sidecar = converted
    broken = rewritten(
        sidecar,
        tmp_path,
        "missing",
        config=lambda config: {k: v for k, v in config.items() if k != "radial_MLP"},
    )
    with pytest.raises(NeutralImportError, match="radial_MLP"):
        import_neutral(broken, CATALOGUE)


def test_a_field_the_import_does_not_map_fails_it(converted, tmp_path):
    _, sidecar = converted
    broken = rewritten(
        sidecar, tmp_path, "extra", config=lambda config: {**config, "m_max": [1.0]}
    )
    with pytest.raises(NeutralImportError, match="m_max"):
        import_neutral(broken, CATALOGUE)


# ---------------------------------------------------------------------------
# Heads
# ---------------------------------------------------------------------------


def two_headed(path: Path) -> Path:
    """A scale-shift model with two heads, built by the frozen tree from the
    anchor's own configuration and left at its random initialization."""
    from mace import modules
    from mace.tools.scripts_utils import extract_config_mace_model

    config = extract_config_mace_model(load_anchor("tiny_scaleshift.model"))
    energies_one = np.ravel(config.pop("atomic_energies"))
    scale = float(np.ravel(config.pop("atomic_inter_scale"))[0])
    config.pop("atomic_inter_shift")
    config["heads"] = ["pbe", "r2scan"]
    config["atomic_energies"] = np.stack([energies_one, energies_one + 0.1])
    config["atomic_inter_scale"] = np.array([scale, 1.1 * scale])
    config["atomic_inter_shift"] = np.array([0.0, 0.05])
    torch.manual_seed(5)
    model = modules.ScaleShiftMACE(**config)
    torch.save(model, path)
    return path


def test_a_multi_head_checkpoint_keeps_every_head(fp64, isolated, tmp_path):
    from tests.golden.anchors import anchor_batch, load_training_structures

    source = two_headed(tmp_path / "two.model")
    legacy = torch.load(source, map_location="cpu", weights_only=False)
    imported = import_neutral(extract_here(source, tmp_path / "two"), CATALOGUE)
    assert imported.heads == ("pbe", "r2scan")

    structures = load_training_structures(limit=4)
    batch = anchor_batch(legacy, structures, torch.float64).to_dict()
    for head in (0, 1):
        batch["head"] = torch.full((len(structures),), head, dtype=torch.long)
        expected = legacy(batch, training=False, compute_force=False)["energy"]
        got = energies_for_head(imported.engine, legacy, structures, head)
        assert torch.allclose(got, expected.detach(), atol=1e-12, rtol=0), head


def energies_for_head(engine, legacy, structures, head):
    from tests.golden.anchors import anchor_batch

    batch = anchor_batch(legacy, structures, torch.float64)
    numbers = [int(z) for z in legacy.atomic_numbers.tolist()]
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
        "head": torch.full((int(batch.num_graphs),), head, dtype=torch.long),
    }
    return engine(graph, compute=()).total_energy.detach()


def test_a_headless_checkpoint_becomes_one_default_head(fp64, isolated, tmp_path):
    """The published models before heads existed carry no heads at all. They
    convert to one head named default, and say that they were headless."""
    model = load_anchor("tiny_scaleshift.model")
    del model.heads
    torch.save(model, tmp_path / "headless.model")
    sidecar = extract_here(tmp_path / "headless.model", tmp_path / "headless")
    artifact = read_neutral(sidecar)
    assert artifact.sidecar.heads == ("default",)
    assert artifact.sidecar.provenance.headless
    imported = import_neutral(sidecar, CATALOGUE)
    assert imported.heads == ("default",)
    assert "recorded no heads" in imported.metadata.notes


def test_a_source_that_recorded_its_heads_says_so(converted):
    _, sidecar = converted
    assert not read_neutral(sidecar).sidecar.provenance.headless


# ---------------------------------------------------------------------------
# Refusals
# ---------------------------------------------------------------------------

UNSUPPORTED = (
    ("mace.modules.models", "AtomicDipolesMACE"),
    ("mace.modules.models", "EnergyDipolesMACE"),
    ("mace.modules.extensions", "PolarMACE"),
    ("mace.modules.extensions", "MACELES"),
    ("mace.modules.extensions", "MagneticMACE"),
    ("mace.modules.extensions", "MagneticScaleShiftMACE"),
    ("mace.modules.extensions", "MagneticSCFMACE"),
)


@pytest.mark.parametrize(("module", "name"), UNSUPPORTED)
def test_each_unsupported_class_is_refused_by_name(module, name):
    """By identity, before a single weight is read: a magnetic model's energy
    blocks look convertible up to where its moment basis would be dropped."""
    import importlib

    cls = getattr(importlib.import_module(module), name)
    with pytest.raises(EXTRACTOR["ExtractionError"], match=name):
        EXTRACTOR["classify"](cls.__new__(cls))


def test_a_subclass_of_a_supported_class_is_refused():
    from mace.modules import ScaleShiftMACE

    class Calibrated(ScaleShiftMACE):
        pass

    with pytest.raises(EXTRACTOR["ExtractionError"], match="subclass"):
        EXTRACTOR["classify"](Calibrated.__new__(Calibrated))


@pytest.mark.parametrize("anchor", ["tiny_dipoles", "tiny_maceles", "tiny_magnetic"])
def test_an_unsupported_anchor_is_refused_through_the_subprocess(anchor, tmp_path):
    with pytest.raises(ExtractionFailed, match="This converter carries"):
        extract(
            GOLDEN / "models" / f"{anchor}.model",
            tmp_path / anchor,
            python=sys.executable,
        )


def test_a_tensor_nothing_accounts_for_aborts_naming_it(fp64, isolated, tmp_path):
    model = load_anchor("tiny_scaleshift.model")
    model.register_buffer("calibration", torch.ones(2))
    torch.save(model, tmp_path / "extra.model")
    with pytest.raises(EXTRACTOR["ExtractionError"], match="calibration"):
        extract_here(tmp_path / "extra.model", tmp_path / "extra")


# ---------------------------------------------------------------------------
# The basis
# ---------------------------------------------------------------------------


def reduced_tagged(sidecar_path: Path, tmp_path: Path) -> Path:
    """The same model written already in this package's reduced basis."""
    artifact = read_neutral(sidecar_path)
    document = artifact.sidecar.model_dump(mode="json")
    tensors = dict(artifact.tensors)
    for name, spec in artifact.sidecar.ops.items():
        if spec.op_kind != "symmetric_contraction":
            continue
        irreps_in = spec.descriptor["irreps_in"]
        target = spec.descriptor["target"]
        for order in range(1, spec.descriptor["correlation"] + 1):
            weights = artifact.tensor(name, f"weights.{order}")
            source = artifact.tensor(name, f"basis.{order}")
            tensors[f"{name}::weights.{order}"] = full_to_reduced(
                weights, irreps_in, order, target, source=source
            )
            tensors[f"{name}::basis.{order}"] = reduced_symmetric_tensor_product_basis(
                irreps_in, order, target
            )[target]
        document["ops"][name]["clebsch_gordan_basis"] = "reduced"
    from mace_core.weights.neutral_format import NeutralSidecar

    return write_neutral(
        tmp_path / "reduced", NeutralSidecar.model_validate(document), tensors
    )


def test_a_reduced_artifact_takes_its_own_path_to_the_same_model(converted, tmp_path):
    from tests.golden.anchors import load_training_structures

    name, sidecar = converted
    legacy = load_anchor(f"{name}.model")
    structures = load_training_structures(limit=6)
    from_full = energies(import_neutral(sidecar, CATALOGUE).engine, legacy, structures)
    from_reduced = energies(
        import_neutral(reduced_tagged(sidecar, tmp_path), CATALOGUE).engine,
        legacy,
        structures,
    )
    assert torch.allclose(from_full, from_reduced, atol=1e-12, rtol=0)


def mistagged(sidecar_path: Path, tmp_path: Path, tag: str) -> Path:
    return rewritten(
        sidecar_path,
        tmp_path,
        f"mistagged-{tag}",
        ops=lambda ops: {
            name: {**op, "clebsch_gordan_basis": tag}
            if op["op_kind"] == "symmetric_contraction"
            else op
            for name, op in ops.items()
        },
    )


def test_a_full_artifact_tagged_reduced_is_refused(converted, tmp_path):
    _, sidecar = converted
    with pytest.raises(NeutralImportError, match="reduced basis, which has"):
        import_neutral(mistagged(sidecar, tmp_path, "reduced"), CATALOGUE)


def test_a_reduced_artifact_tagged_full_is_refused(converted, tmp_path):
    _, sidecar = converted
    reduced = reduced_tagged(sidecar, tmp_path)
    with pytest.raises(NeutralImportError, match="full basis, which has"):
        import_neutral(mistagged(reduced, tmp_path, "full"), CATALOGUE)


def test_reduced_to_full_is_refused(converted):
    _, sidecar = converted
    with pytest.raises(NeutralImportError, match="under-determined"):
        import_neutral(sidecar, CATALOGUE, basis="full")


def test_a_body_order_held_at_zero_is_refused(converted, tmp_path):
    _, sidecar = converted

    def zero_the_third(ops):
        name = next(
            n for n, op in ops.items() if op["op_kind"] == "symmetric_contraction"
        )
        ops[name]["descriptor"]["zeroed"]["3"] = True
        return ops

    with pytest.raises(NeutralImportError, match="body order 3 at zero"):
        import_neutral(
            rewritten(sidecar, tmp_path, "zeroed", ops=zero_the_third), CATALOGUE
        )


# ---------------------------------------------------------------------------
# Constants v1 computes rather than stores
# ---------------------------------------------------------------------------


def with_tensor(sidecar_path, tmp_path, key, change):
    artifact = read_neutral(sidecar_path)
    tensors = dict(artifact.tensors)
    tensors[key] = change(tensors[key])
    return write_neutral(tmp_path / "changed", artifact.sidecar, tensors)


def test_trained_radial_frequencies_are_refused(converted, tmp_path):
    _, sidecar = converted
    changed = with_tensor(
        sidecar, tmp_path, "radial_basis::weights", lambda w: w * 1.01
    )
    with pytest.raises(NeutralImportError, match="frequencies"):
        import_neutral(changed, CATALOGUE)


def test_other_repulsion_constants_are_refused(converted, tmp_path):
    _, sidecar = converted
    changed = with_tensor(sidecar, tmp_path, "pair_repulsion::c", lambda c: c * 1.01)
    with pytest.raises(NeutralImportError, match="repulsion's c"):
        import_neutral(changed, CATALOGUE)


# ---------------------------------------------------------------------------
# The verification hook and the command
# ---------------------------------------------------------------------------


def breached(name: str, tmp_path: Path) -> Path:
    document = json.loads(reference(name).read_text())
    document["fixtures"]["water_cluster"]["outputs"]["energy"]["value"] += 1e-5
    path = tmp_path / "breached.json"
    path.write_text(json.dumps(document))
    return path


def test_an_injected_breach_fails_the_conversion(fp64, tmp_path):
    source = GOLDEN / "models" / "tiny_scaleshift.model"
    with pytest.raises(VerificationError, match="water_cluster energy") as caught:
        convert_legacy(
            source,
            tmp_path / "converted",
            CATALOGUE,
            python=sys.executable,
            reference=breached("tiny_scaleshift", tmp_path),
            fixtures=FIXTURES,
        )
    assert not caught.value.report.passed
    assert not (tmp_path / "converted.json").exists()


def test_the_command_converts_verifies_and_writes(fp64, tmp_path):
    from mace_torch.cli.convert_legacy import main

    code = main(
        [
            str(GOLDEN / "models" / "tiny_scaleshift.model"),
            str(tmp_path / "converted"),
            "--python",
            sys.executable,
            "--reference",
            str(reference("tiny_scaleshift")),
            "--fixtures",
            str(FIXTURES),
        ]
    )
    assert code == 0
    foundation = read_foundation(tmp_path / "converted", CATALOGUE)
    assert foundation.heads == ("Default",)


def test_the_command_refuses_an_unsupported_class(tmp_path, capsys):
    from mace_torch.cli.convert_legacy import main

    code = main(
        [
            str(GOLDEN / "models" / "tiny_magnetic.model"),
            str(tmp_path / "converted"),
            "--python",
            sys.executable,
        ]
    )
    assert code == 2
    assert "MagneticScaleShiftMACE" in capsys.readouterr().err


def test_the_converted_checkpoint_carries_its_record(converted, tmp_path):
    _, sidecar = converted
    imported = import_neutral(sidecar, CATALOGUE)
    path = write_model(tmp_path / "converted", imported.engine, imported.metadata)
    foundation = read_foundation(path, CATALOGUE)
    provenance = read_neutral(sidecar).sidecar.provenance
    (parent,) = foundation.metadata.parents
    assert provenance.source_sha256 in parent.name
    assert provenance.source_class in foundation.metadata.notes
    assert foundation.e0s == isolated_atom_energies(read_neutral(sidecar))


# ---------------------------------------------------------------------------
# The dielectric family
# ---------------------------------------------------------------------------


def dielectric(path: Path) -> Path:
    """A small dielectric model, built by the frozen tree on the anchor's
    architecture and left at its random initialization."""
    import inspect

    from e3nn import o3

    from mace import modules
    from mace.tools.scripts_utils import extract_config_mace_model

    accepted = inspect.signature(modules.AtomicDielectricMACE.__init__).parameters
    config = {
        key: value
        for key, value in extract_config_mace_model(
            load_anchor("tiny_scaleshift.model")
        ).items()
        if key in accepted and key != "readout_cls"
    }
    config.update(
        MLP_irreps=o3.Irreps("8x0e+8x1o"),
        use_polarizability=True,
        gate=torch.nn.functional.silu,
        atomic_energies=None,
        interaction_cls_first=modules.interaction_classes[
            "RealAgnosticInteractionBlock"
        ],
    )
    torch.manual_seed(3)
    torch.save(modules.AtomicDielectricMACE(**config), path)
    return path


def test_a_dielectric_configuration_is_read_in_full(fp64, isolated, tmp_path):
    from mace.tools.scripts_utils import extract_config_mace_model

    model = torch.load(
        dielectric(tmp_path / "d.model"), map_location="cpu", weights_only=False
    )
    recorded = EXTRACTOR["legacy_config"](model)
    assert set(recorded) == set(extract_config_mace_model(model))
    assert len(recorded) == 32


def test_a_dielectric_model_s_weights_are_refused_by_readout(fp64, isolated, tmp_path):
    """Its dipole and polarizability readouts are not mapped, and saying so by
    name is the alternative to carrying them wrong."""
    source = dielectric(tmp_path / "d.model")
    with pytest.raises(EXTRACTOR["ExtractionError"], match="DipolePolarReadoutBlock"):
        extract_here(source, tmp_path / "d")
