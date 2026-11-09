"""A trained v1 model, rebuilt from its checkpoint for evaluation.

One loader for every consumer that evaluates a model outside training: the ASE
calculator, the foundation model a fine-tune starts from, and whatever reads a
model next. A checkpoint is the canonical weights and a JSON record; the record
carries the resolved configuration and the isolated-atom energies per head, and
those two say everything a rebuild needs. The tensors then put back every
weight and every constant, so the model is built with placeholders and loaded.

The model is self-contained: its isolated-atom energies are inside it, so a
consumer never adds them back.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from copy import deepcopy
from dataclasses import dataclass, replace
from pathlib import Path
from typing import TYPE_CHECKING, Any, cast

import torch
from mace_core.config.precision import PrecisionConfig
from mace_core.config.resolved import ResolvedConfig
from mace_core.elements import AtomicNumberTable
from mace_core.metadata import ModelMetadata
from mace_core.observables import (
    DEFAULT_CATALOGUE,
    ObservableCatalogue,
    RequestedOutputs,
)
from mace_core.outputs import MACEOutput
from torch import Tensor, nn

from mace_torch.deploy.manifest import DeployManifest

if TYPE_CHECKING:
    from mace_torch.models.base import MACEModel
    from mace_torch.models.magnetic import MagneticModel

__all__ = ["DeployError", "DeployedModel", "load_deployed"]


class DeployError(ValueError):
    """A checkpoint that cannot be rebuilt into a model."""


@dataclass(frozen=True)
class DeployedModel:
    """A model ready to evaluate, with what its checkpoint says about it.

    Attributes:
        engine: The model inside its derivative engine, which is what computes
            forces and stress around it.
        config: The effective configuration, including deployment overrides.
        metadata: The checkpoint's record.
        z_table: Its element table.
        heads: Its heads, in the order the model indexes them.
        e0s: Its isolated-atom energies per head, ``head -> Z -> eV``, as the
            record holds them. For reading, never for adding back.
        outputs: What it was built to produce, observables and derivatives.
        path: Where it was read from.
    """

    engine: nn.Module
    config: ResolvedConfig
    metadata: ModelMetadata
    z_table: AtomicNumberTable
    heads: tuple[str, ...]
    e0s: Mapping[str, Mapping[int, float]]
    outputs: RequestedOutputs
    path: Path
    manifest: DeployManifest | None = None

    def select_head(self, head: str | None = None) -> str:
        """The head this deployment evaluates, or the checkpoint default."""
        if self.manifest is not None:
            if head is not None and head != self.manifest.head:
                raise DeployError(
                    f"bundle selects {self.manifest.head!r}, requested head {head!r}"
                )
            head = self.manifest.head
        return choose_head(self.heads, head)

    @property
    def dtype(self) -> torch.dtype:
        """The configured dtype of graph inputs and operator interfaces."""
        return getattr(torch, self.config.precision.model)

    @property
    def r_max(self) -> float:
        """The cutoff, in Angstrom."""
        return float(self.config.model.r_max)

    @property
    def model(self) -> MACEModel | MagneticModel:
        """The model inside the engine, where the canonical state lives."""
        return cast(
            "MACEModel | MagneticModel",
            self.derivative_engine.get_submodule("model"),
        )

    @property
    def derivative_engine(self):
        """The underlying model at fixed inputs, without inference relaxation."""
        from mace_torch.physics.fixed_point import FixedPointDriver

        return (
            self.engine.engine
            if isinstance(self.engine, FixedPointDriver)
            else self.engine
        )

    def compute(
        self, graph: Mapping[str, Any], compute: Iterable[str] = ("forces",)
    ) -> MACEOutput[Tensor]:
        """Evaluate one batch, with whichever derivatives are asked for."""
        inputs = dict(graph)
        if self.manifest is not None:
            index = self.heads.index(self.manifest.head)
            inputs["head"] = inputs["head"].new_full(inputs["head"].shape, index)
        return self.engine(inputs, compute=tuple(compute), training=False)

    def with_precision(
        self, precision: PrecisionConfig, *, device: str
    ) -> DeployedModel:
        """Rebuild from current weights without mutating this model or reading disk.

        Operator policies are fixed at construction; casting tensors alone
        would leave their compute and accumulation policies unchanged.
        """
        from mace_core.data.backend import DatasetStatistics
        from mace_core.elements import ResolvedE0s

        from mace_torch.models.build import build_model
        from mace_torch.serialization import canonical_state, load_canonical_state

        config = self.config.model_copy(update={"precision": precision})
        engine, outputs = build_model(
            config,
            self.config.catalogue or DEFAULT_CATALOGUE,
            z_table=self.z_table,
            heads=self.heads,
            e0s=ResolvedE0s(self.e0s),
            statistics=DatasetStatistics(),
            initialize=False,
        )
        load_canonical_state(engine, canonical_state(self.derivative_engine))
        return replace(
            self,
            engine=_with_fixed_point(engine.to(device).eval(), config),
            config=config,
            outputs=outputs,
        )

    def with_time_reversal(self, *, batched: bool = True) -> DeployedModel:
        """Project a magnetic model at opposite moments, without changing its weights.

        Returns an independent model whose resolved configuration records the
        projection, so writing its engine and metadata preserves it on reload.
        Compose the calculator's fixed-point driver around this projected model.
        """
        from mace_torch.models.magnetic import MagneticModel
        from mace_torch.physics import DerivativeEngine

        if not isinstance(batched, bool):
            raise DeployError("batched must be a boolean")
        if not isinstance(self.engine, DerivativeEngine):
            raise DeployError("apply time reversal before the fixed-point driver")
        if self.config.model.model != "magnetic":
            raise DeployError("time reversal requires model.model='magnetic'")
        if self.config.model.magnetic.time_reversal != "none":
            raise DeployError("the model already has time-reversal symmetrization")
        magnetic = self.config.model.magnetic.model_copy(
            update={"time_reversal": "batched" if batched else "sequential"}
        )
        config = self.config.model_copy(
            update={
                "model": self.config.model.model_copy(update={"magnetic": magnetic})
            }
        )
        metadata = self.metadata.model_copy(
            update={
                "config": self.metadata.config.model_copy(
                    update={"resolved": config.to_resolved_dict()}
                )
            }
        )
        # Some backend FX modules reconstruct CPU coefficient buffers when
        # deep-copied. Restore the runtime device without casting op dtypes.
        engine = deepcopy(self.engine).to(next(self.engine.parameters()).device)
        model = engine.model
        if not isinstance(model, MagneticModel):
            raise DeployError("time reversal requires a magnetic energy model")
        model.config = config.model
        # A compiled callable may still close over the unprojected model.
        # Copy the actual operator policies/device, but compile the new forward anew.
        engine._compiled_model = None
        return replace(self, engine=engine, config=config, metadata=metadata)


def load_deployed(
    path: str | Path,
    *,
    device: str = "cpu",
    catalogue: ObservableCatalogue | None = None,
    precision: PrecisionConfig | None = None,
    solver: str | None = None,
    backend: str | None = None,
    tensors_path: str | Path | None = None,
) -> DeployedModel:
    """Rebuild a model from a v1 checkpoint.

    Args:
        path: Either file of the checkpoint pair, or a deployment bundle directory.
        tensors_path: Explicit weights file when the cache gives the two
            members different stems. The manifest is still validated.
        device: Where the model is put.
        catalogue: The observables it may declare. The default catalogue
            unless a model declares others.
        precision: What it computes in. Uses the checkpoint's precision unless given.
        backend: Runtime kernel backend. An explicit bundle pin must agree.
        solver: The electrostatics solver to rebuild a long-range model with,
            when it is not the one it recorded. Allowed only between two
            solvers that both reproduce the reference bit for bit.

    Raises:
        DeployError: If the record carries no isolated-atom energies for its
            heads, or heads that disagree about the elements.
        CheckpointError: If the files are not a v1 checkpoint, or the weights
            and the record disagree, or the rebuilt model's long-range solve is
            not the one recorded.
        SolverSubstitutionError: If ``solver`` would change a long-range
            model's numbers.
    """
    from ase.data import atomic_numbers as numbers_of
    from mace_core.data.backend import DatasetStatistics
    from mace_core.elements import ResolvedE0s
    from mace_core.weights.checkpoint import read_checkpoint_manifest

    from mace_torch.models.build import build_model
    from mace_torch.serialization import load_checkpoint

    source = Path(path)
    manifest = None
    if source.is_dir():
        if tensors_path is not None:
            raise DeployError("a deployment bundle supplies its own model.safetensors")
        manifest = DeployManifest.read(source / "deploy.json")
        if backend is not None and manifest.backend not in (None, backend):
            raise DeployError(f"the bundle pins backend {manifest.backend!r}")
        backend = manifest.backend or backend or "reference"
        source = source / "model.json"
    document = read_checkpoint_manifest(source)
    metadata = ModelMetadata.from_dict(document.config)
    config = ResolvedConfig.model_validate(metadata.config.resolved)
    if backend is not None:
        config = config.model_copy(
            update={"model": config.model.model_copy(update={"backend": backend})}
        )
    if manifest is not None and precision is None:
        precision = config.precision.model_copy(update={"model": manifest.dtype})
    if precision is not None:
        config = config.model_copy(update={"precision": precision})
    catalogue = catalogue or config.catalogue or DEFAULT_CATALOGUE
    config = _with_solver(config, metadata, solver, path)
    heads = tuple(config.data.heads)
    missing = [head for head in heads if head not in metadata.heads]
    if not heads or missing:
        raise DeployError(
            f"{path} records no isolated-atom energies for heads "
            f"{missing or list(heads)}, so the model cannot be rebuilt: its "
            f"element table and its energies both come from that record."
        )
    e0s = {}
    for head in heads:
        details = metadata.heads[head].e0
        e0s[head] = {
            int(numbers_of[symbol]): float(energy)
            for symbol, energy in (
                details.values if details is not None else {}
            ).items()
        }
    if not metadata.elements:
        raise DeployError(f"{path} records no elements; the element table is required.")
    z_table = AtomicNumberTable(metadata.elements)

    if manifest is not None:
        choose_head(heads, manifest.head)
        for name, expected in (
            ("species", tuple(z_table.zs)),
            ("r_max", config.model.r_max),
            ("num_interactions", config.model.num_interactions),
        ):
            if getattr(manifest, name) != expected:
                raise DeployError(
                    f"deploy {name} disagrees with {source}: expected {expected}"
                )
    engine, outputs = build_model(
        config,
        catalogue,
        z_table=z_table,
        heads=heads,
        e0s=ResolvedE0s(e0s),
        statistics=DatasetStatistics(),
        initialize=False,
    )
    load_checkpoint(source, engine, manifest=document, tensors_path=tensors_path)
    engine.to(device)
    engine.eval()
    metadata = _check_recorded_solve(engine.get_submodule("model"), metadata, path)
    return DeployedModel(
        engine=_with_fixed_point(engine, config),
        config=config,
        metadata=metadata,
        z_table=z_table,
        heads=heads,
        e0s=e0s,
        outputs=outputs,
        path=Path(path),
        manifest=manifest,
    )


def choose_head(heads: Sequence[str], head: str | None) -> str:
    """The head a structure is evaluated with.

    Raises:
        ValueError: For a head the model does not have, naming the ones it
            does. The frozen tree warns and takes the last one, which
            evaluates a level of theory nobody asked for.
    """
    if head is not None:
        if head not in heads:
            raise ValueError(
                f"the model has no head {head!r}; its heads are {list(heads)}."
            )
        return head
    if len(heads) == 1:
        return heads[0]
    defaults = [name for name in heads if name.lower() == "default"]
    if not defaults:
        raise ValueError(
            f"the model has the heads {list(heads)} and none is called "
            f"'default', so which to evaluate has to be said: pass head=."
        )
    return defaults[0]


def _with_fixed_point(engine, config: ResolvedConfig):
    from mace_torch.physics.fixed_point import FixedPointDriver

    return (
        FixedPointDriver(engine, config.fixed_point) if config.fixed_point else engine
    )


def _with_solver(
    config: ResolvedConfig,
    metadata: ModelMetadata,
    solver: str | None,
    path: str | Path,
) -> ResolvedConfig:
    """The configuration with the solver the model is to be rebuilt with."""
    from mace_core.electrostatics import solver_to_load

    record = metadata.electrostatics
    if record is None:
        if solver is not None:
            raise DeployError(
                f"{path} is a model with no long-range term, so there is no "
                f"electrostatics solver to choose; {solver!r} has nothing to do."
            )
        return config
    chosen = solver_to_load(record.solver, record.bit_parity, solver)
    section = config.electrostatics.model_copy(update={"solver": chosen})
    return config.model_copy(update={"electrostatics": section})


def _check_recorded_solve(
    model: nn.Module, metadata: ModelMetadata, path: str | Path
) -> ModelMetadata:
    """The rebuilt model asks its solver for the solve it was trained with.

    Its solve is derived from the configuration by code, so a default that
    moved between versions would rebuild the same weights around another
    solve. The record is what says which one they were fitted in.
    """
    from mace_core.electrostatics import solver_spec_record

    from mace_torch.serialization import CheckpointError

    record = metadata.electrostatics
    solver_spec = getattr(model, "solver_spec", None)
    if record is None or solver_spec is None:
        return metadata
    rebuilt = solver_spec_record(solver_spec)
    if rebuilt != record.solver_spec:
        differing = sorted(
            key
            for key in set(rebuilt) | set(record.solver_spec)
            if rebuilt.get(key) != record.solver_spec.get(key)
        )
        raise CheckpointError(
            f"{path} was trained with the long-range solve "
            f"{record.solver_spec} and rebuilds as {rebuilt}; they differ in "
            f"{differing}. The weights were fitted in the recorded one."
        )
    return metadata
