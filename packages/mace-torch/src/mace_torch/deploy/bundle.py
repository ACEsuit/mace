"""Export the checkpoint pair and a validated runtime manifest as one directory."""

from __future__ import annotations

import shutil
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Literal

from mace_core.graph.padding import PaddingPolicy

from mace_torch.deploy.loader import DeployError, load_deployed
from mace_torch.deploy.manifest import DeployManifest


def export_lammps(
    checkpoint: str | Path,
    output: str | Path | None = None,
    *,
    halo_exchange: bool,
    head: str | None = None,
    dtype: Literal["float32", "float64"] | None = None,
    backend: str | None = None,
    compile_regions: bool = True,
    pad_spec: PaddingPolicy | None = None,
) -> Path:
    """Publish a complete local bundle. Preserve weights and head row ordering."""
    source = Path(checkpoint)
    if source.suffix not in {".safetensors", ".json"}:
        raise DeployError("export requires a V1 .safetensors or .json checkpoint")
    target = Path(output) if output is not None else source.with_suffix(".mace-deploy")
    if target.exists():
        raise FileExistsError(f"deployment output already exists: {target}")
    target.parent.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(dir=target.parent) as staging:
        bundle = Path(staging) / "bundle"
        bundle.mkdir()
        for suffix in (".json", ".safetensors"):
            shutil.copyfile(source.with_suffix(suffix), bundle / ("model" + suffix))
        # Export is portable even when the training backend is unavailable here.
        model = load_deployed(bundle / "model.json", backend="reference")
        family = model.config.model
        if family.model not in {"plain", "scale_shift"}:
            raise DeployError(
                f"expected plain or scale_shift for LAMMPS, got {family.model!r}"
            )
        if family.graph_features:
            raise DeployError(
                "this checkpoint requires extra inputs unavailable from ML-IAP"
            )
        precision = dtype or model.config.precision.model
        if precision not in ("float32", "float64"):
            raise DeployError(
                f"choose float32 or float64 for deployment of {precision!r}"
            )
        manifest = DeployManifest(
            schema_version=DeployManifest.VERSION,
            target="lammps-mliap",
            head=model.select_head(head),
            dtype=precision,
            halo_exchange=halo_exchange,
            backend=backend,
            compile_regions=compile_regions,
            pad_spec=pad_spec,
            species=tuple(model.z_table.zs),
            r_max=model.r_max,
            num_interactions=family.num_interactions,
            receptive_field=model.r_max * family.num_interactions,
        )
        manifest.write(bundle / "deploy.json")
        bundle.rename(target)
    return target
