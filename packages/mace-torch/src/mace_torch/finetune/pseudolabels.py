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

The artifact uses the core HDF5 dataset format, with its provenance beside it
as JSON. Property values and dtypes are preserved by the shared data serializer.
"""

from __future__ import annotations

import dataclasses
import hashlib
import logging
import warnings
from collections.abc import Callable, Mapping, Sequence
from datetime import UTC, datetime
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any

import numpy as np
import torch
from mace_core.config import PseudolabelConfig, PseudolabelProvenance
from mace_core.config.resolved import ResolvedConfig
from mace_core.data.backend import DataBackendError
from mace_core.data.backends.hdf5 import HDF5Backend
from mace_core.data.configuration import Configuration, structures_fingerprint
from mace_core.data.keys import KeySpecification
from mace_core.data_spec.shard_format import MANIFEST_NAME, read_manifest, write_shards
from mace_core.elements import AtomicNumberTable
from mace_core.elements.default_keys import Storage
from torch import nn
from torch.func import functional_call

from mace_torch.data import collate_training
from mace_torch.data.batch import GraphDataset

__all__ = [
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
    features: Mapping[str, Storage] | None = None,
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
        features: Custom input storage, retained through graph collation.
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
        features=features,
    )
    labelled: list[Configuration] = []
    batches = (len(dataset) + batch_size - 1) // batch_size
    for number, start in enumerate(range(0, len(dataset), batch_size)):
        chunk = list(configurations[start : start + batch_size])
        try:
            batch = collate_training(
                [dataset[index] for index in range(start, start + len(chunk))],
                float_dtype="float64" if dtype == torch.float64 else "float32",
                features=features,
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
    """Write one head's labelled structures as a core dataset and provenance.

    The provenance supplies the head when the dataset is reopened. Every
    structure must therefore belong to that head, and the recorded count must
    agree. Repeated generation replaces the run's previously written files.
    """
    if len(configurations) != provenance.n_configs or any(
        item.head != provenance.head for item in configurations
    ):
        raise PseudolabelError(
            "pseudolabel provenance head or n_configs disagrees with the structures"
        )
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    # The core writer requires an empty dataset directory. Stage a complete
    # replacement so a failed encode leaves the previous labels intact.
    with TemporaryDirectory(dir=directory) as staging:
        manifest = write_shards(
            configurations,
            staging,
            shard_size=max(len(configurations), 1),
            key_spec=KeySpecification(),
            prefix="labels",
        )
        (Path(staging) / _PROVENANCE).write_text(provenance.model_dump_json(indent=1))
        for name in (*manifest.shards, _PROVENANCE, MANIFEST_NAME):
            (Path(staging) / name).replace(directory / name)
    return directory


def load_pseudolabel_artifact(
    directory: str | Path,
) -> tuple[list[Configuration], PseudolabelProvenance]:
    """Read a core dataset with the head recorded in its pseudolabel provenance."""
    directory = Path(directory)
    missing = [
        name
        for name in (MANIFEST_NAME, _PROVENANCE)
        if not (directory / name).is_file()
    ]
    if missing:
        raise PseudolabelError(
            f"{directory} is not a pseudolabel artifact: it lacks {missing}. A "
            f"run writes one per head under <work_dir>/pseudolabels/<head>/."
        )
    try:
        provenance = PseudolabelProvenance.model_validate_json(
            (directory / _PROVENANCE).read_text()
        )
        dataset = HDF5Backend.open(
            directory, key_spec=KeySpecification(), head=provenance.head
        )
        try:
            if len(dataset) != provenance.n_configs:
                raise PseudolabelError(
                    f"{directory}: dataset count disagrees with provenance.n_configs"
                )
            return list(dataset.iter_range()), provenance
        finally:
            dataset.close()
    except (DataBackendError, OSError, ValueError) as error:
        raise PseudolabelError(
            f"cannot read pseudolabel artifact {directory}: {error}"
        ) from error


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
    device: str | None = None,
    root: str | Path | None = None,
) -> Callable[[str, list[Configuration]], list[Configuration]] | None:
    """What the data stage relabels a head's structures with, if anything.

    Args:
        config: The run's configuration.
        foundation: The foundation model, a
            :class:`~mace_torch.finetune.foundation.Foundation`.
        context: The run's distributed context. The first process generates
            and writes; every process reads what it wrote.
        device: Where to generate, defaulting to the supplied model's device.
        root: Where each head's labels are written, one directory per head.
            ``<work_dir>/pseudolabels`` unless given.

    Returns:
        ``None`` when the run relabels nothing.
    """
    heads = config.pseudolabelled_heads
    if not heads:
        return None
    spec = config.finetune.pseudolabels
    if device is None:
        device = str(next(foundation.engine.parameters()).device)
    from mace_torch.data.inputs import graph_inputs_of

    fingerprint = foundation_fingerprint(foundation.name)
    inputs = graph_inputs_of(foundation.config.model)
    backend = foundation.config.model.backend

    def relabel(name: str, structures: list[Configuration]) -> list[Configuration]:
        if name not in heads:
            return structures
        digest = structures_fingerprint(structures, inputs)
        labels_head = _labelling_head(config, name, foundation.heads)
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
            if provenance.foundation_head != labels_head:
                raise PseudolabelError(
                    f"foundation_head is {provenance.foundation_head!r} in the "
                    f"labels, but this head requests {labels_head!r}; regenerate them"
                )
            for field in ("properties", "stress_if_missing"):
                if getattr(provenance.spec, field) != getattr(spec, field):
                    raise PseudolabelError(
                        f"finetune.pseudolabels.{field} differs from the saved labels; "
                        "regenerate them with the requested settings"
                    )
            return labelled
        directory = (
            pseudolabel_directory(config.runtime.work_dir, name)
            if root is None
            else Path(root) / name
        )
        if context is None or context.is_main:
            labelled = generate_pseudolabels(
                foundation.engine,
                structures,
                spec,
                z_table=foundation.z_table,
                cutoff=foundation.config.model.r_max,
                head_index=foundation.heads.index(labels_head),
                features=foundation.config.model.feature_storages,
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

    names = (
        (MANIFEST_NAME, _PROVENANCE, *read_manifest(directory).shards)
        if context.is_main
        else ()
    )
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
