"""The reference electrostatics solver, and resolving a solver at build time.

A solver builds the long-range energy op from a descriptor. The op reads the
graph it is handed and nothing else: the positions, the batch, the per-graph
periodicity, and the cell. **The cell is the graph's**, whichever the graph
builder put there: the physical cell when any axis is periodic, with an
all-zero row replaced so the volume is finite, and the box built around the
atoms when none is. The op derives the reciprocal cell and the volume from it,
so a strain the derivative engine applies to that cell reaches both, and the
stress it contributes is the derivative of the energy it computed.

**Nothing is decided in forward.** The solver is resolved by name, checked
against the descriptor, and asked for its op once, when the model is built. Its
periodicity handling is fixed then too: each profile maps to one evaluation of
the reference, never to one chosen from the batch it happens to see.

**The geometry is computed once per forward and shared.** A charge-aware model
reads the potential several times in one evaluation and the energy once more,
all over the same positions and cell. What depends only on those, the
reciprocal cell, the volume and the k-vectors, is :class:`LongRangeGeometry`,
and each op takes it rather than deriving its own. An accelerated solver that
prepares a cache of its own does so from this, once, in its ``prepare``.
"""

from __future__ import annotations

# Discrete long-range planning adapts reforge/perf-integration (8dee319d).
# V1 retains its reference Fourier algebra and widened electrostatics precision.
from collections.abc import Mapping
from dataclasses import dataclass, field, replace
from types import NotImplementedType
from typing import Any, Protocol

import torch
from mace_core.electrostatics import (
    ElectrostaticsSolverDescriptor,
    SolverCapabilities,
    get_solver,
)
from torch import Tensor, nn

__all__ = [
    "LONG_RANGE_PLAN",
    "PBC_HANDLING",
    "ElectrostaticsSolver",
    "LongRangeEnergy",
    "LongRangeFeatures",
    "LongRangeGeometry",
    "LongRangePlan",
    "ReferenceSolver",
    "build_long_range",
    "build_long_range_features",
    "build_scf_solve",
    "long_range_geometry",
    "long_range_plan",
    "reciprocal_cell_and_volume",
]

#: The reference's evaluation for each profile. A bulk crystal is a full Ewald
#: sum, a slab adds the dipole correction along its normal, an open molecule is
#: summed exactly in real space, and a batch mixing them treats each structure
#: by its own periodicity, a molecule among them in the box around it with the
#: monopole and dipole corrections that divide by that box's volume, or, per
#: structure, summed exactly in real space on its own.
PBC_HANDLING: dict[str, str] = {
    "full_periodic": "pbc",
    "z_slab": "slab",
    "molecular": "realspace",
    "partial": "mixed_periodic",
    "per_structure": "per_structure",
}

#: The options the reference understands.
SELF_INTERACTION = "include_self_interaction"


class ElectrostaticsSolver(Protocol):
    """What a solver registered under ``mace.electrostatics_backends.torch`` is."""

    name: str
    capabilities: SolverCapabilities

    def long_range_energy(
        self, descriptor: ElectrostaticsSolverDescriptor
    ) -> nn.Module:
        """The op: ``(graph, source_feats, geometry) -> energy per graph``."""
        ...

    def long_range_features(
        self, descriptor: ElectrostaticsSolverDescriptor
    ) -> nn.Module:
        """The projection op: ``prepare(geometry)`` once, then
        ``(cache, source_feats) -> features per atom`` as often as needed."""
        ...

    def make_scf_solve(
        self, descriptor: ElectrostaticsSolverDescriptor
    ) -> nn.Module | NotImplementedType:
        """A fused self-consistent solve, or ``NotImplemented`` for none."""
        ...


def reciprocal_cell_and_volume(cell: Tensor) -> tuple[Tensor, Tensor]:
    """``2 pi inv(cell)^T`` and ``|det cell|``, per graph.

    A cell with no volume has no reciprocal cell, and gets zeros rather than an
    inverse of a singular matrix, as the frozen tree's graph does.
    """
    cell = cell.view(-1, 3, 3)
    volume = torch.linalg.det(cell).abs()
    inverse, _ = torch.linalg.inv_ex(cell.transpose(-1, -2))
    finite = (volume > 0).view(-1, 1, 1)
    return torch.where(finite, 2 * torch.pi * inverse, torch.zeros_like(cell)), volume


@dataclass
class LongRangeGeometry:
    """What every long-range op of one forward reads, computed once.

    Attributes:
        positions: ``[n_atoms, 3]``, as the graph carries them, strained or not.
        batch: ``[n_atoms]``, which structure each atom belongs to.
        pbc: ``[n_graphs, 3]``.
        cell: ``[n_graphs, 3, 3]``, the graph's.
        rcell: ``[n_graphs, 3, 3]``, ``2 pi inv(cell)^T``, derived from ``cell``.
        volume: ``[n_graphs]``, ``|det cell|``.
        k_vectors: ``[n_k, 3]``, every structure's reciprocal vectors inside the
            cutoff, concatenated. Empty for a solve summed in real space.
        k_norm2: ``[n_k]``.
        k_vector_batch: ``[n_k]``, which structure each belongs to.
        k0_mask: ``[n_k]``.
        sizes: Each structure's atom count, as host integers, when the batch
            carries them as ``structure_sizes``.
        periodic: Whether each structure is periodic along any axis, as host
            booleans, when the batch carries them as ``structure_periodic``.
            With both, no count and no branch of the long-range solve is read
            off the device, so a compiled model traces it at static shapes.
        dense_pairs: Whether a real-space sum runs over every pair of atoms of
            the batch, masked to the pairs within one structure, so its shapes
            depend on the atom count alone.
        prepared: What a solver derives from the rest once and every op of
            the same forward reads, by the solver's own key.
        plans: What a solver derived from the cells and the atom counts alone,
            outside the model, by the solver's own key: the ``solvers`` of the
            batch's :class:`LongRangePlan`, empty without one.
    """

    positions: Tensor
    batch: Tensor
    pbc: Tensor
    cell: Tensor
    rcell: Tensor
    volume: Tensor
    k_vectors: Tensor
    k_norm2: Tensor
    k_vector_batch: Tensor
    k0_mask: Tensor
    sizes: tuple[int, ...] | None = None
    periodic: tuple[bool, ...] | None = None
    dense_pairs: bool = False
    prepared: dict[str, Any] = field(default_factory=dict)
    plans: Mapping[str, Any] = field(default_factory=dict)


#: The graph key a batch's :class:`LongRangePlan` is carried under.
LONG_RANGE_PLAN = "long_range_plan"


@dataclass(frozen=True)
class LongRangePlan:
    """What a batch's long-range solve reads that its cells, its periodicity
    and its atom counts fix.

    Built by :func:`long_range_plan` outside the model, where the host may
    wait on the device, and carried in the graph under
    :data:`LONG_RANGE_PLAN`. With it a forward reads nothing off the device
    for the solve, so a compiled model traces it at static shapes; and since
    nothing in it depends on the positions, a caller whose cells do not change
    builds it once.

    Attributes:
        kspace_cutoff: The cutoff the k-vectors were chosen with.
        sizes: Each structure's atom count.
        periodic: Whether each structure is solved as periodic: when it is
            periodic along any axis, and also when it pads a batch that has
            a periodic structure. A padding structure's values are thrown
            away, so it is solved with the periodic ones rather than summed
            exactly in real space, which for the padding atoms, all stacked
            at one point, is the largest sum of the batch.
        coefficients: ``[n_kvec, 3]``, the candidate k-vectors' integer
            coefficients, or ``None`` when the solve sums no k-vectors.
        kept: ``[n_k]``, the candidates inside the cutoff, as
            :func:`~mace_torch.electrostatics.reference.kspace.k_vector_selection`
            returns them, or ``None`` with ``coefficients``.
        dense_pairs: Whether a real-space sum runs over every pair of the
            batch's atoms, masked to each structure, rather than over the
            pairs of each structure: what a compiled model fed batches of
            one size needs, since the pairs of each structure change with the
            atom counts.
        solvers: What each solver derives from these alone, by its own key.
    """

    kspace_cutoff: float
    sizes: tuple[int, ...]
    periodic: tuple[bool, ...]
    coefficients: Tensor | None = None
    kept: Tensor | None = None
    dense_pairs: bool = False
    solvers: Mapping[str, Any] = field(default_factory=dict)


def _sums_k_vectors(
    descriptor: ElectrostaticsSolverDescriptor, periodic: tuple[bool, ...] | None
) -> bool:
    """Whether a solve reads k-vectors: not when it sums in real space, nor
    for a batch the host knows has no periodic structure under a profile that
    sums such a batch in real space."""
    handling = PBC_HANDLING[descriptor.periodicity_profile]
    open_batch = periodic is not None and not any(periodic)
    return not (
        handling == "realspace"
        or (open_batch and handling in ("per_structure", "auto"))
    )


@torch.no_grad()
def long_range_plan(
    graph: Mapping[str, Any],
    descriptor: ElectrostaticsSolverDescriptor,
    ops: tuple[Any, ...] = (),
    dense_pairs: bool = False,
) -> LongRangePlan:
    """The plan of one batch's solve, with each op's own part of it.

    Reads the device. An op contributes through a ``plan(geometry)`` method
    returning its entries by its solver's key; an op without one plans
    nothing.

    Args:
        graph: The current batch. Selection uses detached cell values; the
            forward evaluates reciprocal vectors from the differentiable cell.
        descriptor: The solve.
        ops: The model's long-range ops.
        dense_pairs: Plan the real-space sums over every pair of the batch.
    """
    from mace_torch.electrostatics.reference.kspace import k_vector_selection

    batch = graph["batch"]
    cell = (
        graph["cell"].detach().to(getattr(torch, descriptor.precision)).view(-1, 3, 3)
    )
    num_graphs = cell.shape[0]
    sizes = graph.get("structure_sizes")
    if sizes is None:
        sizes = torch.bincount(batch, minlength=num_graphs).tolist()
    periodic = graph.get("structure_periodic")
    if periodic is None:
        periodic = graph["pbc"].view(-1, 3).any(dim=1).tolist()
    padding = graph.get("structure_padding")
    if padding is not None and any(
        flag and not pad for flag, pad in zip(periodic, padding, strict=True)
    ):
        periodic = [flag or pad for flag, pad in zip(periodic, padding, strict=True)]
    coefficients = kept = None
    if _sums_k_vectors(descriptor, tuple(periodic)):
        rcell, _ = reciprocal_cell_and_volume(cell)
        coefficients, kept = k_vector_selection(
            float(descriptor.kspace_cutoff), cell, rcell
        )
    plan = LongRangePlan(
        kspace_cutoff=float(descriptor.kspace_cutoff),
        sizes=tuple(int(size) for size in sizes),
        periodic=tuple(bool(flag) for flag in periodic),
        coefficients=coefficients,
        kept=kept,
        dense_pairs=dense_pairs,
    )
    planners = [op.plan for op in ops if hasattr(op, "plan")]
    if not planners:
        return plan
    geometry = long_range_geometry({**graph, LONG_RANGE_PLAN: plan}, descriptor)
    solvers: dict[str, Any] = {}
    for planner in planners:
        solvers.update(planner(geometry))
    return replace(plan, solvers=solvers)


def long_range_geometry(
    graph: Mapping[str, Any], descriptor: ElectrostaticsSolverDescriptor
) -> LongRangeGeometry:
    """The shared geometry of one batch, for one solve.

    A solve summed in real space reads no k-vectors, so none are built for it,
    and neither for a batch the host knows has no periodic structure under a
    profile that sums such a batch in real space. With the batch's
    :class:`LongRangePlan` the k-vectors are its selection, evaluated on the
    graph's cell, and nothing is read off the device.

    Raises:
        ValueError: If the plan was made for another k-space cutoff.
    """
    from mace_torch.electrostatics.reference.kspace import (
        compute_k_vectors_flat,
        k_vectors_from_selection,
    )

    dtype = getattr(torch, descriptor.precision)
    positions = graph["positions"].to(dtype)
    cell = graph["cell"].to(dtype).view(-1, 3, 3)
    rcell, volume = reciprocal_cell_and_volume(cell)
    plan: LongRangePlan | None = graph.get(LONG_RANGE_PLAN)
    sizes = periodic = None
    if plan is not None:
        if plan.kspace_cutoff != float(descriptor.kspace_cutoff):
            raise ValueError(
                f"the batch's long-range plan was made for a k-space cutoff "
                f"of {plan.kspace_cutoff}, and this solve's is "
                f"{float(descriptor.kspace_cutoff)}. Plan the batch for the "
                f"model it is evaluated with."
            )
        sizes, periodic = plan.sizes, plan.periodic
    if plan is not None and plan.coefficients is not None:
        assert plan.kept is not None
        k_vectors, k_norm2, k_vector_batch, k0_mask = k_vectors_from_selection(
            plan.coefficients, plan.kept, rcell
        )
    elif plan is not None or not _sums_k_vectors(descriptor, periodic):
        k_vectors = positions.new_empty((0, 3))
        k_norm2 = positions.new_empty((0,))
        k_vector_batch = graph["batch"].new_empty((0,))
        k0_mask = positions.new_empty((0,))
    else:
        k_vectors, k_norm2, k_vector_batch, k0_mask = compute_k_vectors_flat(
            float(descriptor.kspace_cutoff), cell, rcell
        )
    return LongRangeGeometry(
        positions=positions,
        batch=graph["batch"],
        pbc=graph["pbc"].view(-1, 3),
        cell=cell,
        rcell=rcell,
        volume=volume,
        k_vectors=k_vectors,
        k_norm2=k_norm2,
        k_vector_batch=k_vector_batch,
        k0_mask=k0_mask,
        sizes=None if sizes is None else tuple(sizes),
        periodic=None if periodic is None else tuple(periodic),
        dense_pairs=plan is not None and plan.dense_pairs,
        plans={} if plan is None else plan.solvers,
    )


class LongRangeEnergy(nn.Module):
    """The reference long-range energy of a batch of Gaussian multipoles.

    Args:
        descriptor: The solve it computes.
        solver: The name of the solver that built it, which is part of the
            model's state for a solver that is not bit for bit the reference.
    """

    def __init__(self, descriptor: ElectrostaticsSolverDescriptor, solver: str) -> None:
        super().__init__()
        from mace_torch.electrostatics.reference.energy import GTOElectrostaticEnergy

        self.descriptor = descriptor
        self.solver = solver
        self.kspace_cutoff = float(descriptor.kspace_cutoff)
        self.energy = GTOElectrostaticEnergy(
            density_max_l=descriptor.multipole_max_l,
            density_smearing_width=descriptor.smearing_width,
            kspace_cutoff=self.kspace_cutoff,
            include_self_interaction=SELF_INTERACTION
            in descriptor.external_field_flags,
            pbc_handling=PBC_HANDLING[descriptor.periodicity_profile],  # ty: ignore[invalid-argument-type]
        )

    def forward(
        self,
        graph: Mapping[str, Any],
        source_feats: Tensor,
        geometry: LongRangeGeometry | None = None,
    ) -> Tensor:
        """The energy of each graph, in eV.

        Args:
            graph: The batch. Its ``cell`` is read as it is, strained or not.
            source_feats: ``[n_atoms, (multipole_max_l + 1)^2]``, each atom's
                multipoles in the ``e3nn`` component order.
            geometry: The batch's shared geometry, when the caller has already
                built it for another op. Built here otherwise.
        """
        if geometry is None:
            geometry = long_range_geometry(graph, self.descriptor)
        return self.energy(
            k_vectors=geometry.k_vectors,
            k_norm2=geometry.k_norm2,
            k_vector_batch=geometry.k_vector_batch,
            k0_mask=geometry.k0_mask,
            source_feats=source_feats,
            node_positions=geometry.positions,
            batch=geometry.batch,
            volume=geometry.volume,
            pbc=geometry.pbc,
            sizes=geometry.sizes,
            periodic=geometry.periodic,
            dense_pairs=geometry.dense_pairs,
        )


class LongRangeFeatures(nn.Module):
    """The potential of the source density, projected onto each atom.

    Two steps, because a charge-aware model projects several densities over
    one geometry: :meth:`prepare` does everything that depends on the
    positions and the cell once, and each call projects one density through
    it.

    Args:
        descriptor: The solve. Its ``features`` say what is projected onto.
        solver: The name of the solver that built it.
    """

    def __init__(self, descriptor: ElectrostaticsSolverDescriptor, solver: str) -> None:
        super().__init__()
        from mace_torch.electrostatics.reference.features import (
            GTOElectrostaticFeatures,
        )

        projection = descriptor.features
        if projection is None:
            raise ValueError(
                "the descriptor declares no feature projection, so there is "
                "nothing to project onto. Give it `features`."
            )
        self.descriptor = descriptor
        self.solver = solver
        self.dimension = projection.dimension
        self.features = GTOElectrostaticFeatures(
            density_max_l=descriptor.multipole_max_l,
            density_smearing_width=descriptor.smearing_width,
            feature_max_l=projection.max_l,
            feature_smearing_widths=list(projection.widths),
            include_self_interaction=projection.include_self_interaction,
            kspace_cutoff=float(descriptor.kspace_cutoff),
            quadrupole_feature_corrections=projection.quadrupole_corrections,
            integral_normalization=projection.normalization,
            pbc_handling=PBC_HANDLING[descriptor.periodicity_profile],  # ty: ignore[invalid-argument-type]
        )

    def prepare(self, geometry: LongRangeGeometry) -> dict[str, Any]:
        """Everything the projection needs that the density does not change."""
        return self.features.precompute_geometry(
            k_vectors=geometry.k_vectors,
            k_norm2=geometry.k_norm2,
            k_vector_batch=geometry.k_vector_batch,
            k0_mask=geometry.k0_mask,
            node_positions=geometry.positions,
            batch=geometry.batch,
            volume=geometry.volume,
            pbc=geometry.pbc,
            sizes=geometry.sizes,
            periodic=geometry.periodic,
            dense_pairs=geometry.dense_pairs,
        )

    def forward(self, cache: dict[str, Any], source_feats: Tensor) -> Tensor:
        """``[n_atoms, features.dimension]``, grouped by order, then by width,
        then by component in the ``e3nn`` order: the declaration of the
        spherical harmonics up to ``max_l``, one copy per width, sorted.

        Args:
            cache: What :meth:`prepare` returned for this batch.
            source_feats: ``[n_atoms, (multipole_max_l + 1)^2]``.
        """
        projected = self.features.forward_dynamic(
            cache=cache, source_feats=source_feats
        )
        return projected.reshape(source_feats.shape[0], self.dimension)


class ReferenceSolver:
    """The in-tree solver every other one is measured against.

    Plain torch, on any device torch runs on, differentiable twice through
    autograd: the long-range sum is an explicit function of the positions, the
    cell and the multipoles, so a force and its training gradient come from the
    same graph. It evaluates once; a self-consistent loop around it belongs to
    the model, so it offers no fused solve.
    """

    name = "reference"
    capabilities = SolverCapabilities(
        ops=frozenset({"long_range_energy", "long_range_features"}),
        devices=frozenset({"cpu", "cuda", "xpu", "mps"}),
        dtypes=frozenset({"float64", "float32"}),
        slab_normals=frozenset({2}),
        field_flags=frozenset({SELF_INTERACTION}),
        supports_double_backward=True,
        bit_parity=True,
    )

    def long_range_energy(
        self, descriptor: ElectrostaticsSolverDescriptor
    ) -> nn.Module:
        return LongRangeEnergy(descriptor, self.name)

    def long_range_features(
        self, descriptor: ElectrostaticsSolverDescriptor
    ) -> nn.Module:
        return LongRangeFeatures(descriptor, self.name)

    def make_scf_solve(
        self, descriptor: ElectrostaticsSolverDescriptor
    ) -> nn.Module | NotImplementedType:
        del descriptor
        return NotImplemented


def build_long_range(
    descriptor: ElectrostaticsSolverDescriptor,
    *,
    solver: str = "reference",
    trains_derivatives: bool = False,
) -> nn.Module:
    """The long-range op for a model, resolved once.

    Args:
        descriptor: The solve.
        solver: The registered solver to build it with.
        trains_derivatives: Whether the model trains on forces or stress, which
            needs the solve differentiated twice.

    Raises:
        SolverNotAvailableError: If no solver has that name, or it did not
            import.
        UnsupportedSolveError: If the solver declines the solve, or cannot be
            differentiated twice and the model trains on derivatives. Nothing
            is substituted for it.
    """
    backend: ElectrostaticsSolver = get_solver(solver)
    backend.capabilities.require(descriptor, solver)
    if trains_derivatives:
        backend.capabilities.require_double_backward(
            solver, "training on forces or stress"
        )
    return backend.long_range_energy(descriptor)


def build_long_range_features(
    descriptor: ElectrostaticsSolverDescriptor,
    *,
    solver: str = "reference",
    trains_derivatives: bool = False,
) -> nn.Module:
    """The potential projection for a model, resolved once.

    The same rules as :func:`build_long_range`, and the same solver: a model
    reads its features and its energy from one solver, since two would be two
    conventions for the same density.

    Raises:
        SolverNotAvailableError: If no solver has that name, or it did not
            import.
        UnsupportedSolveError: If the solver declines the solve, has no
            projection op, or cannot be differentiated twice and the model
            trains on derivatives.
    """
    backend: ElectrostaticsSolver = get_solver(solver)
    backend.capabilities.require(descriptor, solver)
    if trains_derivatives:
        backend.capabilities.require_double_backward(
            solver, "training on forces or stress"
        )
    return backend.long_range_features(descriptor)


def build_scf_solve(
    descriptor: ElectrostaticsSolverDescriptor, *, solver: str = "reference"
) -> nn.Module | None:
    """A solver's fused self-consistent solve, or ``None`` when it has none.

    ``None`` is the model's cue to run the loop itself, one long-range
    evaluation per iteration.
    """
    backend: ElectrostaticsSolver = get_solver(solver)
    backend.capabilities.require(descriptor, solver)
    span = backend.make_scf_solve(descriptor)
    return None if span is NotImplemented else span
