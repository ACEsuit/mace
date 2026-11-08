"""Export the checkpoint pair and a validated runtime manifest as one directory."""

from __future__ import annotations

import shutil
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Literal

from mace_core.graph.padding import FixedPadding, PaddingPolicy

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
    native_capacity: tuple[int, int] | None = None,
    device: str = "cpu",
) -> Path:
    """Publish a complete local bundle. Preserve weights and head row ordering."""
    if native_capacity is None:
        if device != "cpu":
            raise DeployError("--device only applies when compiling a native package")
    else:
        if not compile_regions:
            raise DeployError(
                "native export compiles the model; remove --no-compile-regions"
            )
        budget = FixedPadding(
            nodes_budget=native_capacity[0],
            edges_budget=native_capacity[1],
            graphs_budget=1,
            on_overflow="error",
        )
        if pad_spec is not None and pad_spec != budget:
            raise DeployError("native capacity and pad_spec disagree")
        pad_spec = budget
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
            target="lammps-mliap" if native_capacity is None else "lammps-native",
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
        if native_capacity is not None:
            from mace_torch.deploy.native import compile_native

            compile_native(
                bundle,
                nodes=native_capacity[0],
                edges=native_capacity[1],
                device=device,
            )
        bundle.rename(target)
    return target
