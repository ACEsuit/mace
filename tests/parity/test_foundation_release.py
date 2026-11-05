"""FM-2: every retained artifact, its full surface, all heads and native reload.

The output directory contains candidate safetensors/JSON pairs and a measured
report per artifact. These are local validation products, never uploads.
"""

import hashlib
import json

import numpy as np
import pytest
import torch
from mace_core.foundation_roster import ROSTER as PUBLISHED_ROSTER
from mace_core.metadata import SCHEMA_VERSION
from mace_torch.calculators.ase_calculator import MACECalculator
from mace_torch.deploy.loader import load_deployed
from mace_torch.deploy.neutral_io import import_neutral
from mace_torch.deploy.reference import _evaluate
from mace_torch.serialization import canonical_state
from mace_torch.train import write_model
from mace_torch.train.data_stage import graph_inputs_of

from tests.golden import foundation_artifacts as fa
from tests.golden import harness, routes
from tests.golden.targets.dielectric import QUANTITIES as DIELECTRIC_QUANTITIES
from tests.golden.targets.foundation_references import _dielectric_projection
from tests.golden.targets.foundation_roster import MODELS as NEW_REFERENCES
from tests.golden.targets.foundation_roster import reference_path
from tests.parity.test_fm_convert_legacy import extract_here
from tests.parity.test_foundation_roster import (
    CATALOGUE,
    GOLDENS,
    ROSTER,
    _rebuilt_bases,
    artifact_path,
    legacy_model,
    mdp_path,
)

REFERENCES = {
    **{
        name: harness.REFERENCES_DIR / fa.ARTIFACTS[key].reference
        for key, name in GOLDENS.items()
        if not key.endswith("_fp32")
    },
    **{name: reference_path(name) for name in NEW_REFERENCES},
    "polar-1-s": harness.REFERENCES_DIR / "polar_foundation_cpu_fp64.json",
    "mdp": harness.REFERENCES_DIR / "mdp_foundation_cpu_fp64.json",
}
PARAMS = [
    pytest.param(
        name,
        marks=[
            pytest.mark.network,
            *([pytest.mark.polar] if name.startswith("polar-") else []),
        ],
    )
    for name in REFERENCES
]
REMOVED = {"mp-small": 341_760, "mp-medium-mpa-0": 820_224, "off-medium": 92_160}


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_foundation_release_covers_the_entire_retained_roster():
    assert set(REFERENCES) == set(ROSTER) | {"mdp"}
    assert (
        len(REFERENCES)
        == sum(entry.family is not None for entry in PUBLISHED_ROSTER)
        == 24
    )
    assert all(path.is_file() for path in REFERENCES.values())


def compare(actual, reference, row, *, polar=False):
    # Native Polar retains the agreed long-range cell derivative correction.
    channels = sorted(
        {
            key
            for entry in reference["fixtures"].values()
            for key in entry["outputs"]
            if not (polar and key == "stress")
        }
    )
    harness.compare_to_reference(actual, reference, row=row.name, channels=channels)
    return {
        f"{fixture}/{key}": float(
            np.max(
                np.abs(
                    np.asarray(actual["fixtures"][fixture]["outputs"][key]["value"])
                    - np.asarray(output["value"])
                )
            )
        )
        for fixture, entry in reference["fixtures"].items()
        for key, output in entry["outputs"].items()
        if key in channels
    }


class DielectricOutput:
    """The seven model-surface outputs, including full position Jacobians."""

    golden_surface = "model"

    def __init__(self, deployed, device):
        self.deployed = deployed
        self.device = device

    def golden_outputs(self, atoms):
        model = self.deployed
        output = _evaluate(
            model.engine,
            atoms,
            z_table=model.z_table,
            cutoff=model.config.model.r_max,
            head=0,
            graph_inputs=graph_inputs_of(model.config.model),
            device=self.device,
            quantities=DIELECTRIC_QUANTITIES,
        )
        return {
            name: output[name][0]
            if name in {"dipole", "polarizability", "polarizability_sh"}
            else output[name]
            for name in DIELECTRIC_QUANTITIES
        }


@pytest.mark.parametrize("name", PARAMS)
@pytest.mark.parametrize("device", ["cpu", pytest.param("cuda", marks=pytest.mark.gpu)])
def test_foundation_release_roundtrip_and_full_parity(
    fp64, isolated, tmp_path, record_property, name, device
):
    from mace.calculators import MACECalculator as LegacyCalculator

    source = mdp_path() if name == "mdp" else artifact_path(name)
    source_hash = digest(source)
    reference = harness.load_reference(REFERENCES[name])
    pinned_hash = reference["provenance"].get("sha256")
    if pinned_hash is not None:
        assert source_hash == pinned_hash
    neutral = extract_here(source, tmp_path / "neutral")
    imported = import_neutral(neutral, CATALOGUE)
    published = legacy_model(source)
    rounded = _rebuilt_bases(neutral) or any(
        value.dtype == torch.float32
        for value in published.state_dict().values()
        if value.is_floating_point()
    )
    row = harness.FP32 if rounded else harness.FP64_CPU_REFERENCE
    declared_heads = getattr(published, "heads", None)
    source_heads = tuple(declared_heads) if declared_heads is not None else ("default",)
    assert imported.heads == source_heads
    assert imported.config.model.clebsch_gordan_basis == "reduced"
    assert imported.metadata.schema_version == SCHEMA_VERSION
    if name.startswith("polar-"):
        assert imported.config.electrostatics.solver == "reference"
    old_count = sum(p.numel() for p in published.parameters())
    new_count = sum(p.numel() for p in imported.engine.parameters())
    if name in REMOVED:
        assert old_count - new_count == REMOVED[name]
    checkpoint = write_model(tmp_path / name, imported.engine, imported.metadata)
    deployed = load_deployed(checkpoint, device=device)
    before, after = canonical_state(imported.engine), canonical_state(deployed.engine)
    assert before.keys() == after.keys()
    for operator, tensors in before.items():
        assert tensors.keys() == after[operator].keys()
        for key, value in tensors.items():
            assert torch.equal(value.cpu(), after[operator][key].cpu()), (operator, key)
    fixtures = harness.load_fixtures(names=tuple(reference["fixtures"]))
    reports = {}
    for head in source_heads:
        if name == "mdp":
            native = DielectricOutput(deployed, device)
            legacy = LegacyDielectricOutput(published.to(device).double(), device)
        else:
            native = MACECalculator(models=deployed, device=device, head=head)
            legacy = LegacyCalculator(
                models=[published],
                device=device,
                default_dtype="float64",
                head=head if declared_heads is not None else None,
                model_type="PolarMACE" if name.startswith("polar-") else "MACE",
            )
        actual = harness.snapshot_outputs(
            native, fixtures, dtype="float64", device=device, backend="reference"
        )
        expected = harness.snapshot_outputs(
            legacy, fixtures, dtype="float64", device=device, backend="e3nn"
        )
        reports[head] = {
            "live": compare(actual, expected, row, polar=name.startswith("polar-"))
        }
        golden_head = {
            "mp-mh-0": fa.ARTIFACTS["mh_0"].loader_kwargs["head"],
            "mp-mh-1": fa.ARTIFACTS["mh_1"].loader_kwargs["head"],
            "omol": "omol",
        }.get(name, source_heads[0])
        if head == golden_head:
            reports[head]["golden"] = compare(
                actual, reference, row, polar=name.startswith("polar-")
            )
    assert any("golden" in report for report in reports.values())
    assert digest(source) == source_hash
    conversion = imported.metadata.conversion.model_copy(
        update={"reference": str(REFERENCES[name]), "golden_gate_passed": True}
    )
    write_model(
        tmp_path / name,
        imported.engine,
        imported.metadata.model_copy(update={"conversion": conversion}),
    )
    report = {
        "artifact": name,
        "device": device,
        "tolerance_row": row.name,
        "source_sha256": source_hash,
        "heads": list(source_heads),
        "legacy_parameters": old_count,
        "native_parameters": new_count,
        "schema_version": imported.metadata.schema_version,
        "reference": REFERENCES[name].name,
        "comparisons": reports,
        "files": {
            path.name: digest(path)
            for path in (
                neutral,
                neutral.with_suffix(".safetensors"),
                checkpoint,
                checkpoint.with_suffix(".safetensors"),
            )
        },
        "publication": "deferred; no upload authorized",
    }
    (tmp_path / "verification.json").write_text(json.dumps(report, indent=2) + "\n")
    record_property("artifact", name)
    record_property("evidence", str(tmp_path / "verification.json"))


class LegacyDielectricOutput:
    """Frozen forward with its graph on the same accelerator as its weights."""

    golden_surface = "model"

    def __init__(self, model, device):
        self.model, self.device = model, device

    def golden_outputs(self, atoms):
        graph = {
            name: value.to(self.device) if isinstance(value, torch.Tensor) else value
            for name, value in routes.graph_batch(self.model, atoms).items()
        }
        return _dielectric_projection(
            self.model(graph, compute_dielectric_derivatives=True)
        )
