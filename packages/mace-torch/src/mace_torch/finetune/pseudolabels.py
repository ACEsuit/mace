"""Generating a replay head's labels with the foundation model, and reusing them.

Four things, and each replaces something the frozen tree does quietly:

**A failure is a failure.** Any batch that fails to evaluate stops the run.
The frozen tree used to keep the file's own labels for a batch that failed,
which mixes two levels of theory in one replay set, and still catches a failure
of the whole step and trains on with a warning.

**Nothing is switched off to generate them.** The frozen tree turns every
parameter's gradient off and back on around the loop. Here the model is
evaluated with detached copies of its parameters, so autograd keeps nothing
for a weight gradient and the model's own flags are never touched. The memory
this saves is small, because a force still needs every activation on the way
back to the positions: on an A100, 864 atoms at 128 channels peak at 12082 MiB
against 12314 with trainable parameters, in the same time, and the forces of
the two differ no more than two runs of either. They cannot be taken under
``torch.no_grad()``, since a force is a gradient, so the stage turns gradients
on for itself whatever context it is called in.

**They are an artifact of the run.** Written once, by the first process, into
``<work_dir>/pseudolabels/<head>/``, with a record of what they were made from,
and every process reads that one copy. The frozen tree generates them on every
process, and on a GPU two processes can disagree about a label.

**Reusing them is asked for and checked.** A run names a previous run's labels,
and they are refused if the foundation model's weights or the structures differ
from what the record says they were made from.

The artifact is tensors in safetensors, with a manifest and the record beside
them as JSON: exact float64, and nothing executable.
"""

from __future__ import annotations

import dataclasses
import hashlib
import json
import logging
import warnings
from collections.abc import Callable, Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import torch
from mace_core.config import PseudolabelConfig, PseudolabelProvenance
from mace_core.config.resolved import ResolvedConfig
from mace_core.data.configuration import Configuration, structures_fingerprint
from mace_core.elements import AtomicNumberTable
from safetensors.numpy import load_file, save_file
from torch import nn
from torch.func import functional_call

from mace_torch.data import collate_training
from mace_torch.data.batch import GraphDataset

__all__ = [
    "ARTIFACT_FORMAT",
    "PseudolabelError",
    "foundation_fingerprint",
    "generate_pseudolabels",
    "load_pseudolabel_artifact",
    "pseudolabel_directory",
    "relabelling",
    "validate_pseudolabel_reuse",
    "write_pseudolabel_artifact",
]

logger = logging.getLogger(__name__)

#: What the manifest says it is, so a directory of something else is refused.
ARTIFACT_FORMAT = "mace-v1-pseudolabels"
_LABELS = "labels.safetensors"
_MANIFEST = "manifest.json"
_PROVENANCE = "provenance.json"

#: The per-atom labels among the ones a foundation model gives.
_PER_ATOM = frozenset({"forces", "charges"})


class PseudolabelError(RuntimeError):
    """Labels that could not be generated, read or reused."""


def pseudolabel_directory(work_dir: str | Path, head: str) -> Path:
    """Where a run writes a head's labels."""
    return Path(work_dir) / "pseudolabels" / head


def foundation_fingerprint(path: str | Path) -> str:
    """The sha256 of a v1 checkpoint's weights, the file and not the record."""
    weights = Path(path).with_suffix(".safetensors")
    digest = hashlib.sha256()
    with weights.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


# ---------------------------------------------------------------------------
# Generating them
# ---------------------------------------------------------------------------


def _floor(configuration: Configuration, name: str) -> float:
    """The weight a generated label carries: the file's if it gave one."""
    existing = float(configuration.property_weights.get(name, 0.0))
    return existing if existing > 0.0 else 1.0


def generate_pseudolabels(
    engine: nn.Module,
    configurations: Sequence[Configuration],
    spec: PseudolabelConfig,
    *,
    z_table: AtomicNumberTable,
    cutoff: float,
    head_index: int,
    graph_inputs: Sequence[str] = (),
    batch_size: int = 10,
    device: str = "cpu",
) -> list[Configuration]:
    """The structures, each labelled by the model.

    Pure: the structures given are not changed, and the model is not either.

    Args:
        engine: The foundation model in its derivative engine.
        configurations: The structures to label.
        spec: What to label them with.
        z_table: The model's element table.
        cutoff: Its cutoff, in Angstrom.
        head_index: Which of its heads labels them.
        graph_inputs: The inputs it reads beside the geometry.
        batch_size: How many are evaluated at once.
        device: Where.

    Raises:
        PseudolabelError: If a batch fails to evaluate, naming which, or if
            the model does not produce a property the spec asks for.
    """
    wanted = set(spec.properties)
    compute = [name for name in ("forces", "stress", "virials") if name in wanted]
    dtype = next(engine.parameters()).dtype
    # The weights as constants: a parameter that requires a gradient makes
    # autograd keep every activation that multiplies it, for a weight gradient
    # nothing asks for.
    constants = {name: value.detach() for name, value in engine.named_parameters()}
    dataset = GraphDataset(
        list(configurations),
        cutoff=cutoff,
        z_table=z_table,
        targets=(),
        graph_inputs=graph_inputs,
    )
    labelled: list[Configuration] = []
    batches = (len(dataset) + batch_size - 1) // batch_size
    for number, start in enumerate(range(0, len(dataset), batch_size)):
        chunk = list(configurations[start : start + batch_size])
        try:
            batch = collate_training(
                [dataset[index] for index in range(start, start + len(chunk))],
                z_table=z_table,
                float_dtype="float64" if dtype == torch.float64 else "float32",
            ).to(device)
            graph = dict(batch.graph)
            heads = graph["head"]
            assert isinstance(heads, torch.Tensor)
            graph["head"] = torch.full_like(heads, head_index)
            # A force is a gradient, so gradients are on here whatever the
            # caller's context says; at training=False the engine writes none
            # to a parameter.
            with torch.enable_grad():
                output = functional_call(
                    engine,
                    constants,
                    args=(graph,),
                    kwargs={"compute": tuple(compute), "training": False},
                    strict=False,
                )
        except Exception as error:
            raise PseudolabelError(
                f"labelling failed on batch {number + 1} of {batches} "
                f"(structures {start} to {start + len(chunk) - 1}). A set with "
                f"some of its labels generated and the rest the file's mixes "
                f"two levels of theory, so nothing is relabelled."
            ) from error
        counts = [len(item.atomic_numbers) for item in chunk]
        offsets = np.concatenate([[0], np.cumsum(counts)])
        for position, source in enumerate(chunk):
            labelled.append(
                _labelled(
                    source, output, spec, position, offsets[position : position + 2]
                )
            )
    return labelled


def _labelled(source, output, spec, position, rows) -> Configuration:
    """One structure with its labels replaced by the model's."""
    atoms = slice(int(rows[0]), int(rows[1]))
    properties = dict(source.properties)
    weights = dict(source.property_weights)
    for name in spec.properties:
        if name == "stress":
            had_stress = (
                source.properties.get("stress") is not None
                and float(source.property_weights.get("stress", 0.0)) > 0.0
            )
            if not (had_stress or spec.stress_if_missing):
                continue
        value = {
            "energy": output.total_energy,
            "forces": output.forces,
            "stress": output.stress,
            "virials": output.virials,
            "dipole": output.dipole,
            "charges": output.extras.get("charges"),
        }[name]
        if value is None:
            raise PseudolabelError(
                f"finetune.pseudolabels.properties asks for {name!r} and the "
                f"foundation model does not produce it. Drop it from the "
                f"properties."
            )
        value = value.detach().to(torch.float64).cpu().numpy()
        properties[name] = value[atoms] if name in _PER_ATOM else value[position].copy()
        if name == "energy":
            properties[name] = float(properties[name])
        weights[name] = _floor(source, name)
    return dataclasses.replace(source, properties=properties, property_weights=weights)


# ---------------------------------------------------------------------------
# The artifact
# ---------------------------------------------------------------------------


def write_pseudolabel_artifact(
    directory: str | Path,
    configurations: Sequence[Configuration],
    provenance: PseudolabelProvenance,
) -> Path:
    """Write labelled structures and their record, exactly.

    Returns:
        The directory.
    """
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    tensors, manifest = _encode(configurations)
    save_file(tensors, str(directory / _LABELS))
    (directory / _MANIFEST).write_text(json.dumps(manifest, indent=1, sort_keys=True))
    (directory / _PROVENANCE).write_text(provenance.model_dump_json(indent=1))
    return directory


def load_pseudolabel_artifact(
    directory: str | Path,
) -> tuple[list[Configuration], PseudolabelProvenance]:
    """Read back what :func:`write_pseudolabel_artifact` wrote.

    Raises:
        PseudolabelError: If the directory holds no such artifact.
    """
    directory = Path(directory)
    missing = [
        name
        for name in (_LABELS, _MANIFEST, _PROVENANCE)
        if not (directory / name).is_file()
    ]
    if missing:
        raise PseudolabelError(
            f"{directory} is not a pseudolabel artifact: it lacks {missing}. A "
            f"run writes one per head under <work_dir>/pseudolabels/<head>/."
        )
    manifest = json.loads((directory / _MANIFEST).read_text())
    if manifest.get("format") != ARTIFACT_FORMAT:
        raise PseudolabelError(
            f"{directory / _MANIFEST} is a {manifest.get('format')!r}, not a "
            f"{ARTIFACT_FORMAT!r}."
        )
    provenance = PseudolabelProvenance.model_validate_json(
        (directory / _PROVENANCE).read_text()
    )
    return _decode(load_file(str(directory / _LABELS)), manifest), provenance


def _encode(
    configurations: Sequence[Configuration],
) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    """Structures as concatenated arrays and a manifest of what is where.

    Every numeric property is one flat array over the structures that carry
    it, with a mask of which do and each one's shape in the manifest. A
    property that is not numeric, the head's name say, is kept in the manifest
    as it is.
    """
    counts = np.asarray([len(item.atomic_numbers) for item in configurations])
    tensors: dict[str, np.ndarray] = {
        "n_atoms": counts.astype(np.int64),
        "atomic_numbers": np.concatenate(
            [np.asarray(item.atomic_numbers, dtype=np.int64) for item in configurations]
        ),
        "positions": np.concatenate(
            [np.asarray(item.positions, dtype=np.float64) for item in configurations]
        ).reshape(-1, 3),
        "weight": np.asarray(
            [item.weight for item in configurations], dtype=np.float64
        ),
        "has_cell": np.asarray(
            [item.cell is not None for item in configurations], dtype=np.uint8
        ),
        "cell": np.stack(
            [
                np.zeros((3, 3))
                if item.cell is None
                else np.asarray(item.cell, dtype=np.float64)
                for item in configurations
            ]
        ),
        "has_pbc": np.asarray(
            [item.pbc is not None for item in configurations], dtype=np.uint8
        ),
        "pbc": np.asarray(
            [item.pbc or (False, False, False) for item in configurations],
            dtype=np.uint8,
        ),
    }
    names = sorted(
        {name for item in configurations for name in item.properties}
        | {name for item in configurations for name in item.property_weights}
    )
    properties: dict[str, Any] = {}
    for name in names:
        values = [item.properties.get(name) for item in configurations]
        # Whether the key is there at all, which is not whether it has a value:
        # a declared property the file lacked is present as None.
        keyed = [name in item.properties for item in configurations]
        if any(isinstance(value, str) for value in values):
            properties[name] = {"kind": "text", "values": values, "keyed": keyed}
            continue
        present = [value is not None for value in values]
        shapes = [
            list(np.shape(value)) if value is not None else None for value in values
        ]
        flat = [
            np.asarray(value, dtype=np.float64).reshape(-1)
            for value in values
            if value is not None
        ]
        tensors[f"property/{name}"] = (
            np.concatenate(flat) if flat else np.zeros(0, dtype=np.float64)
        )
        tensors[f"present/{name}"] = np.asarray(present, dtype=np.uint8)
        tensors[f"weight/{name}"] = np.asarray(
            [item.property_weights.get(name, np.nan) for item in configurations],
            dtype=np.float64,
        )
        properties[name] = {
            "kind": "numeric",
            "keyed": keyed,
            "shapes": shapes,
            "scalar": [
                value is not None
                and np.ndim(value) == 0
                and not isinstance(value, np.ndarray)
                for value in values
            ],
        }
    manifest = {
        "format": ARTIFACT_FORMAT,
        "n_configs": len(configurations),
        "config_types": [item.config_type for item in configurations],
        "heads": [item.head for item in configurations],
        "properties": properties,
    }
    return tensors, manifest


def _decode(
    tensors: dict[str, np.ndarray], manifest: dict[str, Any]
) -> list[Configuration]:
    counts = tensors["n_atoms"]
    atom_offsets = np.concatenate([[0], np.cumsum(counts)])
    readers: dict[str, Any] = {}
    for name, entry in manifest["properties"].items():
        if entry["kind"] == "text":
            readers[name] = ("text", entry)
            continue
        present = tensors[f"present/{name}"].astype(bool)
        sizes = [
            int(np.prod(shape)) if shape is not None else 0 for shape in entry["shapes"]
        ]
        value_offsets = np.concatenate([[0], np.cumsum(sizes)])
        readers[name] = ("numeric", entry, present, value_offsets)
    structures = []
    for index in range(int(manifest["n_configs"])):
        atoms = slice(int(atom_offsets[index]), int(atom_offsets[index + 1]))
        properties: dict[str, Any] = {}
        weights: dict[str, float] = {}
        for name, reader in readers.items():
            if reader[0] == "text":
                if reader[1]["keyed"][index]:
                    properties[name] = reader[1]["values"][index]
                continue
            _, entry, present, value_offsets = reader
            weight = float(tensors[f"weight/{name}"][index])
            if not np.isnan(weight):
                weights[name] = weight
            if not entry["keyed"][index]:
                continue
            if not present[index]:
                properties[name] = None
                continue
            flat = tensors[f"property/{name}"][
                int(value_offsets[index]) : int(value_offsets[index + 1])
            ]
            value = flat.reshape(entry["shapes"][index])
            properties[name] = float(value) if entry["scalar"][index] else value.copy()
        structures.append(
            Configuration(
                atomic_numbers=tensors["atomic_numbers"][atoms].copy(),
                positions=tensors["positions"][atoms].copy(),
                properties=properties,
                property_weights=weights,
                cell=tensors["cell"][index].copy()
                if tensors["has_cell"][index]
                else None,
                pbc=_pbc(tensors["pbc"][index]) if tensors["has_pbc"][index] else None,
                weight=float(tensors["weight"][index]),
                config_type=manifest["config_types"][index],
                head=manifest["heads"][index],
            )
        )
    return structures


def _pbc(flags: np.ndarray) -> tuple[bool, bool, bool]:
    return (bool(flags[0]), bool(flags[1]), bool(flags[2]))


# ---------------------------------------------------------------------------
# Reusing them
# ---------------------------------------------------------------------------


def validate_pseudolabel_reuse(
    provenance: PseudolabelProvenance,
    *,
    foundation_fingerprint: str,
    dataset_fingerprint: str,
    torch_version: str | None = None,
    kernel_backend: str | None = None,
) -> None:
    """Refuse labels made from another model or other structures.

    There is no override. Labels from another foundation model teach the
    replay head another model, and labels of other structures are labels of
    something else; generating them again costs minutes.

    Raises:
        PseudolabelError: Naming which of the two differs.
    """
    if provenance.foundation_fingerprint != foundation_fingerprint:
        raise PseudolabelError(
            f"the labels were made by the foundation model with weights "
            f"{provenance.foundation_fingerprint[:12]}, "
            f"{provenance.foundation_model}, and this run's has "
            f"{foundation_fingerprint[:12]}. Generate them again with "
            f"finetune.pseudolabels.enabled."
        )
    if provenance.dataset_fingerprint != dataset_fingerprint:
        raise PseudolabelError(
            f"the labels were made for other structures than this run's head "
            f"{provenance.head!r} reads ({provenance.dataset_fingerprint[:12]} "
            f"against {dataset_fingerprint[:12]}): a different file, "
            f"subselection or order. Generate them again with "
            f"finetune.pseudolabels.enabled."
        )
    drift = [
        f"{name} {then} then, {now} now"
        for name, then, now in (
            ("torch", provenance.torch_version, torch_version),
            ("kernel backend", provenance.kernel_backend, kernel_backend),
        )
        if now is not None and then != now
    ]
    if drift and not provenance.deterministic:
        warnings.warn(
            f"the reused labels were made on {provenance.device}, which does not "
            f"reproduce them bit for bit, and with {'; '.join(drift)}. They are "
            f"valid, but they could not be made again exactly.",
            stacklevel=2,
        )


# ---------------------------------------------------------------------------
# The seam the data stage calls
# ---------------------------------------------------------------------------


def relabelling(
    config: ResolvedConfig,
    foundation,
    *,
    context=None,
    device: str = "cpu",
    root: str | Path | None = None,
) -> Callable[[str, list[Configuration]], list[Configuration]] | None:
    """What the data stage relabels a head's structures with, if anything.

    Args:
        config: The run's configuration.
        foundation: The foundation model, a
            :class:`~mace_torch.finetune.foundation.Foundation`.
        context: The run's distributed context. The first process generates
            and writes; every process reads what it wrote.
        device: Where to generate.
        root: Where each head's labels are written, one directory per head.
            ``<work_dir>/pseudolabels`` unless given.

    Returns:
        ``None`` when the run relabels nothing.
    """
    heads = config.pseudolabelled_heads
    if not heads:
        return None
    spec = config.finetune.pseudolabels
    from mace_torch.train.data_stage import graph_inputs_of

    fingerprint = foundation_fingerprint(foundation.name)
    inputs = graph_inputs_of(foundation.config.model.model)
    backend = foundation.config.model.backend

    def relabel(name: str, structures: list[Configuration]) -> list[Configuration]:
        if name not in heads:
            return structures
        digest = structures_fingerprint(structures, inputs)
        if spec.labels_from is not None:
            labelled, provenance = load_pseudolabel_artifact(
                Path(spec.labels_from) / name
            )
            validate_pseudolabel_reuse(
                provenance,
                foundation_fingerprint=fingerprint,
                dataset_fingerprint=digest,
                torch_version=torch.__version__,
                kernel_backend=backend,
            )
            return labelled
        directory = (
            pseudolabel_directory(config.runtime.work_dir, name)
            if root is None
            else Path(root) / name
        )
        labels_head = _labelling_head(config, name, foundation.heads)
        if context is None or context.is_main:
            labelled = generate_pseudolabels(
                foundation.engine,
                structures,
                spec,
                z_table=foundation.z_table,
                cutoff=foundation.config.model.r_max,
                head_index=foundation.heads.index(labels_head),
                graph_inputs=inputs,
                batch_size=spec.batch_size or config.training.batch_size,
                device=device,
            )
            dtype = next(foundation.engine.parameters()).dtype
            write_pseudolabel_artifact(
                directory,
                labelled,
                PseudolabelProvenance(
                    foundation_model=foundation.name,
                    foundation_fingerprint=fingerprint,
                    dataset_fingerprint=digest,
                    head=name,
                    foundation_head=labels_head,
                    spec=spec,
                    device=device,
                    dtype=str(dtype).removeprefix("torch."),
                    torch_version=torch.__version__,
                    kernel_backend=backend,
                    generated_at=datetime.now(UTC),
                    n_configs=len(labelled),
                    deterministic=device.split(":", 1)[0] == "cpu",
                ),
            )
            logger.info("Relabelled %d structures of head %r", len(labelled), name)
        _share(directory, context)
        labelled, _ = load_pseudolabel_artifact(directory)
        return labelled

    return relabel


def _labelling_head(config: ResolvedConfig, head: str, foundation_heads) -> str:
    """The foundation head that labels a head: the one its readout starts from."""
    wanted = config.data.heads[head].readout_from
    if wanted is None:
        if len(foundation_heads) != 1:
            raise PseudolabelError(
                f"head {head!r} is relabelled and does not say by which of the "
                f"foundation model's heads {list(foundation_heads)}. Set its "
                f"`readout_from`; the frozen tree takes the first and says so "
                f"only in its log."
            )
        return foundation_heads[0]
    if wanted not in foundation_heads:
        raise PseudolabelError(
            f"head {head!r} is relabelled by the foundation head {wanted!r}, "
            f"which the foundation model does not have; its heads are "
            f"{list(foundation_heads)}."
        )
    return wanted


def _share(directory: Path, context) -> None:
    """Give every process the first process's files, byte for byte.

    Broadcast rather than trusted to a shared filesystem: a run directory on a
    node's local disk is invisible to the others, and every process has to
    read the same labels.
    """
    if context is None or not context.distributed:
        return
    import torch.distributed as dist

    names = (_LABELS, _MANIFEST, _PROVENANCE)
    payload: list[Any] = [
        {name: (directory / name).read_bytes() for name in names}
        if context.is_main
        else None
    ]
    dist.broadcast_object_list(payload, src=0)
    if not context.is_main:
        directory.mkdir(parents=True, exist_ok=True)
        for name, content in payload[0].items():
            target = directory / name
            if not target.is_file() or target.read_bytes() != content:
                target.write_bytes(content)
    context.barrier()
