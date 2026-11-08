"""FM-3 support rows, evaluated against committed goldens in either framework.

Local candidates produce evidence but never a supported badge. Each framework
runs separately so importing the JAX evaluator does not import torch or legacy.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import os
import platform
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
from mace_core.model_registry import get_model, list_models

from tests.golden import harness

# These source artifacts contain fp32 constants/weights. FM-2 measured them
# against the same frozen row; loading them as fp64 does not restore precision.
_ROUNDED = {
    "mace-mp-0b-medium",
    "mace-mh-0",
    "mace-mh-1",
    "mace-omol-0-extra-large",
    "mace-polar-1-small",
    "mace-polar-1-medium",
    "mace-polar-1-large",
}
ROOT = Path(__file__).resolve().parents[2]


def _references(entry):
    if not entry.golden_refs:
        raise ValueError(f"{entry.model_id}: no committed golden declared")
    records = []
    for name in entry.golden_refs:
        path = harness.REFERENCES_DIR / name
        reference = harness.load_reference(path)
        if not reference["fixtures"] or any(
            not fixture["outputs"] for fixture in reference["fixtures"].values()
        ):
            raise ValueError(f"{name}: empty golden cannot establish support")
        records.append(
            {
                "name": name,
                "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                "expected": reference,
            }
        )
    return records


class _DielectricSurface:
    golden_surface = "model"

    def __init__(self, model, device):
        self.model, self.device = model, device

    def golden_outputs(self, atoms):
        from mace_torch.deploy.reference import _evaluate
        from mace_torch.data.inputs import graph_inputs_of

        from tests.golden.targets.dielectric import QUANTITIES

        model = self.model
        values = _evaluate(
            model.engine,
            atoms,
            z_table=model.z_table,
            cutoff=model.config.model.r_max,
            head=0,
            graph_inputs=graph_inputs_of(model.config.model),
            device=self.device,
            quantities=QUANTITIES,
        )
        return {
            name: value[0]
            if name in {"dipole", "polarizability", "polarizability_sh"}
            else value
            for name, value in values.items()
            if name in QUANTITIES
        }


def _torch_checks(entry, reference, records, device, results, hardware):
    import torch
    from mace_torch.backends.precision import interface_dtype
    from mace_torch.calculators import foundations

    from tests.golden import foundation_artifacts

    hardware.update(
        framework_version=torch.__version__,
        accelerator=torch.cuda.get_device_name(device)
        if torch.device(device).type == "cuda"
        else None,
    )
    # Supplying the expected family default as an override would hide a
    # loader whose actual default was accidentally changed.
    for dtype in dict.fromkeys(("float64", entry.default_dtype)):
        head = next(iter(entry.metadata.heads))
        if entry.model_id in {"mace-mh-0", "mace-mh-1"}:
            key = "mh_0" if entry.model_id == "mace-mh-0" else "mh_1"
            head = foundation_artifacts.ARTIFACTS[key].loader_kwargs["head"]
        loader = getattr(foundations, entry.family)
        default = dtype == entry.default_dtype
        options = {} if default else {"default_dtype": dtype}
        row = (
            harness.FP32
            if dtype == "float32" or entry.model_id in _ROUNDED
            else harness.FP64_CPU_REFERENCE
        )
        comparisons = {}
        check = dict(
            dtype=dtype,
            loader_default=default,
            head=head,
            tolerance_row=row.name,
            status="failed",
            comparisons=comparisons,
        )
        results.append(check)
        calculator = loader(reference, device=device, head=head, **options)
        loaded = calculator.models[0]
        actual_dtype = interface_dtype(loaded.engine)
        check["interface_dtype"] = str(actual_dtype).removeprefix("torch.")
        if actual_dtype != getattr(torch, dtype):
            raise ValueError(
                f"{entry.model_id}: expected {dtype} interface, got {actual_dtype}"
            )
        if loaded.config.model != entry.hyperparameters():
            raise ValueError(f"{entry.model_id}: registry hyperparameters differ")
        if loaded.heads != tuple(entry.metadata.heads):
            raise ValueError(f"{entry.model_id}: registry heads differ")
        surface = (
            _DielectricSurface(loaded, device)
            if entry.family == "mace_mdp"
            else calculator
        )
        for record in records:
            expected = record["expected"]
            fixtures = harness.load_fixtures(names=tuple(expected["fixtures"]))
            actual = harness.snapshot_outputs(
                surface, fixtures, dtype=dtype, device=device, backend="reference"
            )
            channels = sorted(
                {
                    key
                    for fixture in expected["fixtures"].values()
                    for key in fixture["outputs"]
                    if not (entry.family == "mace_polar" and key == "stress")
                }
            )
            if not channels:
                raise ValueError("no numerical channels selected")
            harness.compare_to_reference(
                actual, expected, row=row.name, channels=channels
            )
            comparisons[record["name"]] = {
                "channels": channels,
                "max_abs_difference": {
                    f"{fixture}/{key}": float(
                        np.max(
                            np.abs(
                                np.asarray(
                                    actual["fixtures"][fixture]["outputs"][key]["value"]
                                )
                                - np.asarray(output["value"])
                            )
                        )
                    )
                    for fixture, item in expected["fixtures"].items()
                    for key, output in item["outputs"].items()
                    if key in channels
                },
            }
        check["status"] = "passed"
        del surface, calculator, loaded


def _jax_checks(entry, reference, records, device, results, hardware):
    import jax
    from mace_jax.models.registry import SUPPORTED_MODELS

    if entry.model_id not in SUPPORTED_MODELS:
        raise ValueError(f"{entry.model_id}: outside the JAX architecture closure")
    hardware.update(
        framework_version=jax.__version__,
        accelerator=[
            str(item) for item in jax.devices("gpu" if device == "cuda" else device)
        ],
    )
    sys.path.insert(0, str(ROOT / "packages/mace-jax/tests"))
    try:
        from jax_conformance import run_jax_model_conformance

        checks = run_jax_model_conformance(
            entry.model_id, artifact_path=reference, devices=(device,)
        )
    finally:
        sys.path.pop(0)
    results.extend(checks)


def evaluate_entry(entry, *, framework, device="cpu", local_artifact=None):
    """Always emit a row; failures are red, missing publications are pending."""
    row = {
        "model_id": entry.model_id,
        "framework": framework,
        "timestamp": datetime.now(UTC).isoformat(),
        "code_version": importlib.metadata.version("mace-core"),
        "code_commit": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
        ).strip(),
        "worktree_dirty": bool(
            subprocess.check_output(
                ["git", "status", "--porcelain"], cwd=ROOT, text=True
            ).strip()
        ),
        "hardware": {
            "platform": platform.platform(),
            "machine": platform.machine(),
            "device": device,
            "python": platform.python_version(),
        },
        "artifact": str(local_artifact)
        if local_artifact
        else (entry.artifact_ref.url if entry.artifact_ref else None),
        "supported": False,
        "status": "pending_publication",
        "checks": [],
        "scope": (
            "E/F/stress and per-atom energy; O(3), translation and permutation"
            if framework == "jax"
            else "all committed channels on the golden head"
        ),
        "exceptions": (
            [
                "Polar stress retains the agreed long-range cell derivative "
                "correction; "
                "legacy stress is excluded (ELEC-2 evidence in gate3)."
            ]
            if entry.family == "mace_polar"
            else []
        ),
    }
    try:
        row["goldens"] = _references(entry)
        if local_artifact is None and entry.artifact_ref is None:
            return row
        evaluator = {"torch": _torch_checks, "jax": _jax_checks}[framework]
        # Published validation goes through the canonical-name loader, including
        # metadata versions, registry identity and the normal download cache.
        evaluator(
            entry,
            local_artifact or entry.model_id,
            row["goldens"],
            device,
            row["checks"],
            row["hardware"],
        )
        if not row["checks"]:
            raise ValueError("empty conformance result cannot establish support")
        row["supported"] = local_artifact is None
        row["status"] = "supported" if row["supported"] else "validated_local"
    except Exception as error:
        row["status"] = "failed"
        row["error"] = f"{type(error).__name__}: {error}"
    return row


def write_row(directory, row):
    """One badge file per model, with independently measured framework rows."""
    path = Path(directory) / get_model(row["model_id"]).badge_ref
    path.parent.mkdir(parents=True, exist_ok=True)
    data = (
        json.loads(path.read_text()) if path.exists() else {"model_id": row["model_id"]}
    )
    data.setdefault("frameworks", {})[row["framework"]] = row
    path.write_text(json.dumps(data, indent=2) + "\n")
    return path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--framework", choices=("torch", "jax"), required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--artifacts", type=Path, help="local candidate map, never support"
    )
    parser.add_argument(
        "--model", action="append", help="canonical ID; default: entire closure"
    )
    parser.add_argument("--device", default="cpu", choices=("cpu", "cuda"))
    args = parser.parse_args()
    candidates = json.loads(args.artifacts.read_text()) if args.artifacts else {}
    entries = list_models()
    if args.framework == "jax":
        from mace_jax.models.registry import SUPPORTED_MODELS

        entries = [entry for entry in entries if entry.model_id in SUPPORTED_MODELS]
    if args.model:
        entries = [entry for entry in entries if entry.model_id in args.model]
        if {entry.model_id for entry in entries} != set(args.model):
            parser.error("unknown model or outside framework closure")
    failed = False
    for entry in entries:
        if args.artifacts and entry.model_id not in candidates:
            parser.error(f"candidate map missing {entry.model_id}")
        if (
            not args.artifacts
            and entry.artifact_ref
            and os.getenv("MACE_CI_ALLOW_NETWORK") != "1"
        ):
            parser.error("published validation requires MACE_CI_ALLOW_NETWORK=1")
        row = evaluate_entry(
            entry,
            framework=args.framework,
            device=args.device,
            local_artifact=candidates.get(entry.model_id),
        )
        write_row(args.output, row)
        print(
            entry.model_id,
            args.framework,
            row["status"],
            row.get("error", ""),
            flush=True,
        )
        failed |= row["status"] == "failed"
    return int(failed)


if __name__ == "__main__":
    raise SystemExit(main())
