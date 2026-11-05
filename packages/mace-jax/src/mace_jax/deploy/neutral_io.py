"""Read canonical safetensors/JSON without importing either torch stack."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit, urlunsplit

import jax
import numpy as np
from flax import nnx
from mace_core.artifacts import resolve_artifact
from mace_core.config.precision import PrecisionConfig
from mace_core.config.resolved import ResolvedConfig
from mace_core.kernels.registry import get_backend
from mace_core.metadata import ModelMetadata
from mace_core.outputs import MACEOutput
from safetensors.numpy import load_file

from mace_jax.models.energy import MACEModel
from mace_jax.models.registry import SUPPORTED_MODELS
from mace_jax.physics.outputs import evaluate

SUPPORTED_INTERACTIONS = frozenset({"plain", "residual", "density", "density_residual"})
SUPPORTED_OUTPUTS = frozenset(
    {"energy", "forces", "stress", "virials", "node_energies"}
)


class ModelLoadError(ValueError):
    """An artifact outside the JAX inference contract."""


@dataclass(frozen=True)
class ApplyFn:
    """Pure (state, flat graph) evaluation; params includes model buffers."""

    graphdef: Any
    metadata: ModelMetadata
    config: ResolvedConfig
    elements: tuple[int, ...]
    dtype: str

    def __call__(
        self,
        params: nnx.State,
        graph: dict[str, jax.Array],
        *,
        compute: tuple[str, ...] = ("forces", "stress", "virials"),
    ) -> MACEOutput[jax.Array]:
        # fp32 means IEEE fp32 on GPU too; TF32 is outside this contract.
        with jax.default_matmul_precision("highest"):
            return evaluate(nnx.merge(self.graphdef, params), graph, compute=compute)


def _architecture(config: ResolvedConfig) -> None:
    model = config.model
    missing = []
    if len(config.data.heads) != 1:
        raise ModelLoadError(
            "JAX requires a single head; use mace model select-head first "
            "(headless originals count as single-head)."
        )
    if model.clebsch_gordan_basis != "reduced":
        raise ModelLoadError(
            "clebsch_gordan_basis must be 'reduced'; "
            "run the explicit checkpoint converter first"
        )
    if model.model not in {"plain", "scale_shift"}:
        missing.append(f"model={model.model}")
    for name in ("interaction_first", "interaction"):
        value = getattr(model, name)
        if value not in SUPPORTED_INTERACTIONS:
            missing.append(f"{name}={value}")
    if model.radial_basis.kind != "bessel":
        missing.append(f"radial_basis.kind={model.radial_basis.kind}")
    if model.distance_transform.kind not in {"none", "agnesi"}:
        missing.append(f"distance_transform.kind={model.distance_transform.kind}")
    if model.readout.gate != "silu":
        missing.append(f"readout.gate={model.readout.gate}")
    for name in ("bias", "last_only", "from_embedding"):
        if getattr(model.readout, name):
            missing.append(f"readout.{name}")
    if model.graph_features:
        missing.append("graph_features")
    missing.extend(
        f"observable={name}"
        for name in model.observables
        if name not in SUPPORTED_OUTPUTS
    )
    if config.catalogue is not None:
        from mace_core.observables import DEFAULT_CATALOGUE

        if config.catalogue != DEFAULT_CATALOGUE:
            missing.append("custom observable catalogue")
    if config.fixed_point is not None:
        missing.append("fixed_point")
    if model.normalization:
        missing.append("observable normalization")
    if missing:
        raise ModelLoadError(
            f"JAX architecture has missing blocks or outputs: {sorted(missing)}; "
            "use the torch engine for this artifact"
        )


def _paths(reference: str | Path) -> tuple[Path, Path]:
    text = str(reference)
    if text in SUPPORTED_MODELS:
        artifact = SUPPORTED_MODELS[text].artifact_ref
        if artifact is None:
            raise ModelLoadError(
                f"{text}: FM-2 publication and the FM-3 registry URL are pending; "
                "pass its converted local .json or .safetensors path"
            )
        text = artifact
    if text.startswith("https:"):
        parts = urlsplit(text)
        suffix = Path(parts.path).suffix
        if suffix not in {".json", ".safetensors"}:
            raise ModelLoadError(
                "JAX reads canonical .json/.safetensors pairs; "
                "use mace model convert for a legacy pickle"
            )
        urls = [
            urlunsplit(parts._replace(path=str(Path(parts.path).with_suffix(ext))))
            for ext in (".json", ".safetensors")
        ]
        return resolve_artifact(urls[0]), resolve_artifact(urls[1])
    path = Path(text)
    if path.suffix not in {".json", ".safetensors"}:
        raise ModelLoadError(
            "JAX reads canonical .json/.safetensors pairs; "
            "use mace model convert for a legacy pickle"
        )
    return path.with_suffix(".json"), path.with_suffix(".safetensors")


def load_model(
    path_or_id: str | Path, *, backend: str = "reference", dtype: str = "float64"
) -> tuple[ApplyFn, nnx.State]:
    """Load a reduced, single-head energy model. Enables x64 before making arrays.

    Both metadata and tensor manifests are checked before any model weights are
    installed. Contraction paths are matched by coupling-tree name, never index.
    """
    if dtype not in {"float32", "float64"}:
        raise ModelLoadError(
            f"dtype={dtype!r}; JAX inference supports float32 or float64"
        )
    jax.config.update("jax_enable_x64", True)
    sidecar, tensors = _paths(path_or_id)
    document = json.loads(sidecar.read_text())
    if document.get("format") != "mace-v1-checkpoint" or document.get("version") != 3:
        raise ModelLoadError(
            f"{sidecar}: expected mace-v1-checkpoint version 3; "
            "convert the artifact to the native reduced safetensors format"
        )
    metadata = ModelMetadata.from_json(json.dumps(document["config"]))
    config = ResolvedConfig.model_validate(metadata.config.resolved)
    _architecture(config)
    elements = tuple(metadata.elements)
    if not elements:
        raise ModelLoadError(
            "metadata.elements is empty; reconvert with the current converter"
        )
    precision = (
        PrecisionConfig.float64() if dtype == "float64" else PrecisionConfig.float32()
    )
    model = MACEModel(
        get_backend(backend, framework="jax"), config.model, elements, precision
    )
    expected = model.canonical_components()
    flat = load_file(str(tensors))
    state = {}
    declared = set()
    for entry in document["tensors"]:
        name = entry["module"] + "::" + entry["name"]
        if name in declared:
            raise ModelLoadError(f"duplicate canonical tensor {name}")
        declared.add(name)
        if name not in flat:
            raise ModelLoadError(f"missing canonical tensor {name}")
        value = flat[name]
        if list(value.shape) != entry["shape"] or str(value.dtype) != entry["dtype"]:
            raise ModelLoadError(f"{name}: dtype or shape disagrees with the sidecar")
        state.setdefault(entry["module"], {})[entry["name"]] = value
    if declared != set(flat):
        raise ModelLoadError(
            f"undeclared canonical tensors: {sorted(set(flat) - declared)}"
        )
    if set(state) != set(expected):
        raise ModelLoadError(
            "canonical operators differ; "
            f"missing {sorted(set(expected) - set(state))}, "
            f"unsupported {sorted(set(state) - set(expected))}"
        )
    layouts = document.get("layout", {})
    expected_layouts = model.backbone.canonical_layout()
    if set(layouts) != set(expected_layouts):
        raise ModelLoadError("canonical layout modules disagree with the architecture")
    for name, op in expected.items():
        values = state[name]
        expected_layout = expected_layouts.get(name, {})
        layout = layouts.get(name, {})
        if "paths" in expected_layout:
            labels = layout.get("paths", [])
            expected_labels = expected_layout["paths"]
            if len(labels) != len(set(labels)) or set(labels) != set(expected_labels):
                raise ModelLoadError(
                    f"{name}: missing or incompatible named reduced contraction paths"
                )
            order = [labels.index(label) for label in expected_labels]
            values = {**values, "weight": values["weight"][:, order, :]}
            layout = {**layout, "paths": expected_labels}
        if layout != expected_layout:
            raise ModelLoadError(
                f"{name}: canonical layout disagrees with the architecture"
            )
        if (
            name == "backbone.outputs.energy_head"
            and values["e0_table"].dtype != np.float64
        ):
            raise ModelLoadError("e0_table must retain float64 storage")
        op.load_canonical(values)
    graphdef, params = nnx.split(model)
    return ApplyFn(graphdef, metadata, config, elements, dtype), params
