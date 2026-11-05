"""Diagnose Polar precision against the frozen references and live legacy.

This runner reports failures; it never relaxes a tolerance or promotes a dtype.
Run as ``python -m tests.parity.polar_precision --output report.json``. Cached
legacy files and a canonical-ID -> native checkpoint JSON map are required.
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import platform
import subprocess
from pathlib import Path

import torch
from mace_core.model_registry import get_model
from mace_torch.calculators.foundations import mace_polar

from tests.golden import harness
from tests.parity.test_foundation_roster import artifact_path, legacy_model


def comparison(actual, expected, channels=None, *, row=harness.FP32.name):
    result = {"tolerance_row": row}
    try:
        harness.compare_to_reference(
            actual, expected, row=row, channels=channels, strict_channels=True
        )
    except AssertionError as error:
        result.update(passed=False, error=str(error))
    else:
        result["passed"] = True
    return result


def diagnose(model_id, artifact, *, device, dtype):
    from mace.calculators import MACECalculator as LegacyCalculator

    entry = get_model(model_id)
    if entry.family != "mace_polar":
        raise ValueError(f"{model_id}: this diagnostic measures Polar models")
    reference_path = harness.REFERENCES_DIR / entry.golden_refs[0]
    reference = harness.load_reference(reference_path)
    fixtures = harness.load_fixtures(names=tuple(reference["fixtures"]))
    channels = sorted(
        {
            name
            for fixture in reference["fixtures"].values()
            for name in fixture["outputs"]
            if name != "stress"
        }
    )
    # All three published Polar artifacts contain fp32 constants/weights.
    # FM-2's comparison row remains fp32 even when evaluating in float64.
    previous_dtype = torch.get_default_dtype()
    previous_tf32 = torch.backends.cuda.matmul.allow_tf32
    previous_matmul = torch.get_float32_matmul_precision()
    try:
        torch.set_default_dtype(getattr(torch, dtype))
        torch.set_float32_matmul_precision("highest")
        torch.backends.cuda.matmul.allow_tf32 = False
        source = artifact_path(entry.aliases[0])
        source_hash = hashlib.sha256(source.read_bytes()).hexdigest()
        reference_source_hash = reference["provenance"].get("sha256")
        if reference_source_hash is not None and source_hash != reference_source_hash:
            raise ValueError(f"{model_id}: legacy artifact differs from golden source")
        legacy = LegacyCalculator(
            models=[legacy_model(source)],
            device=device,
            default_dtype=dtype,
            model_type="PolarMACE",
            head="Default",
        )
        native = mace_polar(Path(artifact), device=device, default_dtype=dtype)
        old = harness.snapshot_outputs(
            legacy, fixtures, dtype=dtype, device=device, backend="e3nn"
        )
        new = harness.snapshot_outputs(
            native, fixtures, dtype=dtype, device=device, backend="reference"
        )
        return {
            "model_id": model_id,
            "dtype": dtype,
            "device": device,
            "reference": reference_path.name,
            "reference_sha256": hashlib.sha256(reference_path.read_bytes()).hexdigest(),
            "legacy_artifact_sha256": source_hash,
            "reference_source_sha256": reference_source_hash,
            "reference_dtype": reference["dtype"],
            "fixture_names": list(reference["fixtures"]),
            "exception": "stress retains the agreed ELEC-2 cell derivative correction",
            "native_vs_golden": comparison(new, reference, channels),
            # Legacy can reproduce its own CPU capture without the conversion
            # allowance for rounded coupling bases, including historical stress.
            "legacy_vs_golden": comparison(
                old,
                reference,
                row=harness.FP64_CPU_REFERENCE.name
                if dtype == "float64" and device == "cpu"
                else harness.FP32.name,
            ),
            "native_vs_legacy": comparison(new, old, channels),
        }
    finally:
        torch.set_default_dtype(previous_dtype)
        torch.set_float32_matmul_precision(previous_matmul)
        torch.backends.cuda.matmul.allow_tf32 = previous_tf32


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifacts", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument(
        "--model", nargs="+", default=["mace-polar-1-medium", "mace-polar-1-large"]
    )
    parser.add_argument(
        "--dtype",
        nargs="+",
        choices=("float64", "float32"),
        default=["float64", "float32"],
    )
    args = parser.parse_args()
    artifacts = json.loads(args.artifacts.read_text())
    revision = subprocess.run(
        ["git", "rev-parse", "HEAD"], text=True, capture_output=True, check=False
    )
    report = {
        "code_commit": revision.stdout.strip() if revision.returncode == 0 else None,
        "torch": torch.__version__,
        "platform": platform.platform(),
        "hardware": torch.cuda.get_device_name()
        if args.device == "cuda"
        else platform.machine(),
        "cases": [],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    for model_id in args.model:
        for dtype in args.dtype:
            case = diagnose(
                model_id, artifacts[model_id], device=args.device, dtype=dtype
            )
            report["cases"].append(case)
            args.output.write_text(json.dumps(report, indent=2) + "\n")
            print(
                model_id,
                dtype,
                {
                    key: value["passed"]
                    for key, value in case.items()
                    if key.endswith(("_golden", "_legacy"))
                },
                flush=True,
            )
            gc.collect()
            if args.device == "cuda":
                torch.cuda.empty_cache()
    return int(
        any(
            not case[comparison]["passed"]
            for case in report["cases"]
            for comparison in (
                "native_vs_golden",
                "legacy_vs_golden",
                "native_vs_legacy",
            )
        )
    )


if __name__ == "__main__":
    raise SystemExit(main())
