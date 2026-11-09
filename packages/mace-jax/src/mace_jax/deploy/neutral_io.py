"""Read canonical safetensors/JSON without importing either torch stack."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import jax
import numpy as np
from flax import nnx
from mace_core.artifacts import resolve_checkpoint
from mace_core.config.resolved import ResolvedConfig
from mace_core.kernels.registry import get_backend
from mace_core.metadata import ModelMetadata
from mace_core.model_registry import get_model
from mace_core.outputs import MACEOutput
from mace_core.weights.checkpoint import (
    CheckpointError,
    read_checkpoint_manifest,
    validate_layout,
    validate_tensors,
)
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

    @property
    def dtype(self) -> str:
        return self.config.precision.model

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
                f"{text}: native artifact publication is pending; "
                "pass its converted local .json or .safetensors path"
            )
        text = artifact
    try:
        files = resolve_checkpoint(text)
    except ValueError as error:
        raise ModelLoadError(str(error)) from error
    return files.metadata, files.tensors


def load_model(
    path_or_id: str | Path, *, backend: str = "reference", dtype: str | None = None
) -> tuple[ApplyFn, nnx.State]:
    """Load a reduced, single-head energy model. Enables x64 before making arrays.

    Both metadata and tensor manifests are checked before any model weights are
    installed. Contraction paths are matched by coupling-tree name, never index.
    """
    if dtype is not None and dtype not in {"float32", "float64"}:
        raise ModelLoadError(
            f"dtype={dtype!r}; JAX inference supports float32 or float64"
        )
    jax.config.update("jax_enable_x64", True)
    sidecar, tensors = _paths(path_or_id)
    try:
        document = read_checkpoint_manifest(sidecar)
    except CheckpointError as error:
        raise ModelLoadError(str(error)) from error
    metadata = ModelMetadata.from_dict(document.config)
    config = ResolvedConfig.model_validate(metadata.config.resolved)
    _architecture(config)
    if str(path_or_id) in SUPPORTED_MODELS:
        entry = get_model(str(path_or_id))
        if (
            config.model != entry.hyperparameters()
            or tuple(metadata.heads) != tuple(entry.metadata.heads)
            or tuple(metadata.elements) != tuple(entry.metadata.elements)
        ):
            raise ModelLoadError(
                f"{entry.model_id}: published architecture, heads or elements "
                "disagree with registry"
            )
    elements = tuple(metadata.elements)
    if not elements:
        raise ModelLoadError(
            "metadata.elements is empty; reconvert with the current converter"
        )
    precision = config.precision
    if dtype is not None:
        precision = precision.model_copy(update={"model": dtype})
    if precision.ops:
        raise ModelLoadError(
            "JAX inference does not support per-operation precision.ops; "
            "use the torch engine for this artifact"
        )
    if precision.model not in {"float32", "float64"} or precision.accumulate not in {
        "float32",
        "float64",
    }:
        raise ModelLoadError(
            "JAX precision.model and accumulate must be float32 or float64"
        )
    config = config.model_copy(
        update={
            "precision": precision,
            "model": config.model.model_copy(update={"backend": backend}),
        }
    )
    model = MACEModel(
        get_backend(backend, framework="jax"), config.model, elements, precision
    )
    expected = model.canonical_components()
    flat = load_file(str(tensors))
    try:
        validate_tensors(
            document,
            {name: (value.shape, str(value.dtype)) for name, value in flat.items()},
        )
    except CheckpointError as error:
        raise ModelLoadError(str(error)) from error
    state = {}
    for entry in document.tensors:
        state.setdefault(entry.module, {})[entry.name] = flat[entry.key]
    if set(state) != set(expected):
        raise ModelLoadError(
            "canonical operators differ; "
            f"missing {sorted(set(expected) - set(state))}, "
            f"unsupported {sorted(set(state) - set(expected))}"
        )
    layouts = dict(document.layout)
    expected_layouts = model.backbone.canonical_layout()
    for name in expected:
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
        if name in layouts:
            layouts[name] = layout
        state[name] = values
        if (
            name == "backbone.outputs.energy_head"
            and values["e0_table"].dtype != np.float64
        ):
            raise ModelLoadError("e0_table must retain float64 storage")
    try:
        validate_layout(expected_layouts, layouts)
    except CheckpointError as error:
        raise ModelLoadError(str(error)) from error
    for name, op in expected.items():
        op.load_canonical(state[name])
    graphdef, params = nnx.split(model)
    return ApplyFn(graphdef, metadata, config, elements), params
