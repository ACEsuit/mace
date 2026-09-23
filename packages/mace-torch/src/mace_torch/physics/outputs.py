"""Forces and stress, taken around the model call.

The frozen tree takes them inside ``forward``: it sets ``requires_grad`` on the
positions, injects a strain, runs the model, and calls ``torch.autograd.grad``,
all in one method. Each of those gradient calls is a graph break, which is why
the compiled path never went anywhere.

Here the model contains no gradient call at all and the differentiation happens
in three phases around it.

* **A, before.** Make the positions, and the strain when a stress is wanted,
  graph leaves. Apply the strain to the positions and the cell, recompute the
  shifts from ``unit_shifts`` and the strained cell, and form the edge vectors.
  Hand the model a ready graph.
* **B, the model.** The backbone and the output layer. No ``autograd.grad``, no
  ``requires_grad_``, no branching on which derivatives were asked for. This is
  the region that compiles whole.
* **C, after.** One ``autograd.grad`` on the energy, with every target at once.

Phase A is why this is a bracket and not a stage. The strain has to reach the
positions and the cell **before** the edge vectors are formed, so the engine
reaches upstream of the backbone as well as downstream of it. That ordering is
the one structural thing to get right here, and it is not visible from the
signatures.

**The sign of the stress, stated once.** ``autograd.grad`` gives
``dE/dstrain``. The virial is its negative and the stress is not::

    virials = -dE/dstrain
    stress  = +dE/dstrain / V   =   -virials / V

The frozen tree reaches the same place by computing the stress from the
gradient *before* it negates it into the virial, which reads as though the two
shared a sign. Measured on the tree: ``max|stress * V + virials| = 1.2e-35``.
Both conventions are spelled out here because getting this backwards produces
numbers of the right magnitude, and a stress of the wrong sign is a model that
relaxes a crystal in the wrong direction.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from typing import Any

import torch
from mace_core.observables import InputSpec, ObservableSpec
from mace_core.outputs import MACEOutput
from torch import Tensor, nn

__all__ = [
    "DerivativeEngine",
    "cell_volume_and_mask",
    "prepare_inputs",
    "stress_from_strain_gradient",
]


def cell_volume_and_mask(
    cell: Tensor, pbc: Tensor | None = None
) -> tuple[Tensor, Tensor]:
    """Per-graph cell volume, and which graphs it is a volume of.

    A graph with no periodic direction has nothing to divide a virial by. The
    cell it carries is the padding box the neighbour list builds around the
    atoms, so ``virials / det(cell)`` is a number set by that padding and not a
    stress: on a water molecule the energy and the virial are bit-identical at
    every box size while the reported stress tracks ``1/V`` exactly.

    Args:
        cell: ``[n_graphs, 3, 3]`` or ``[n_graphs * 3, 3]``.
        pbc: ``[n_graphs, 3]`` of bool, or ``None`` when the caller is always
            periodic, as under LAMMPS. With ``None`` only a degenerate cell is
            masked.

    Returns:
        The volume, which is **one** where the mask is false, and the mask. The
        substitution is not cosmetic: masking after the division still routes a
        nan back through it on the backward pass.
    """
    cell = cell.view(-1, 3, 3)
    volume = torch.linalg.det(cell).abs()
    periodic = volume > 0.0
    if pbc is not None:
        periodic = torch.logical_and(periodic, pbc.view(-1, 3).any(dim=-1))
    return torch.where(periodic, volume, torch.ones_like(volume)), periodic


def stress_from_strain_gradient(
    strain_gradient: Tensor, cell: Tensor, pbc: Tensor | None = None
) -> Tensor:
    """``dE/dstrain / V``, per graph, zero where there is no volume.

    Args:
        strain_gradient: ``dE/dstrain``, ``[n_graphs, 3, 3]``. This is the raw
            gradient, **not** the virial: the virial is its negative.
        cell: ``[n_graphs, 3, 3]``.
        pbc: ``[n_graphs, 3]`` of bool, or ``None``.
    """
    volume, periodic = cell_volume_and_mask(cell, pbc)
    stress = strain_gradient / volume.view(-1, 1, 1)
    stress = torch.where(periodic.view(-1, 1, 1), stress, torch.zeros_like(stress))
    # The near-degenerate backstop. A cell that is not quite singular divides
    # to something finite and enormous, which the mask above does not catch.
    return torch.where(torch.abs(stress) < 1e10, stress, torch.zeros_like(stress))


def _symmetric_strain(
    positions: Tensor,
    unit_shifts: Tensor,
    cell: Tensor,
    sender: Tensor,
    batch: Tensor,
    num_graphs: int,
    displacement: Tensor | None,
) -> tuple[Tensor, Tensor, Tensor, Tensor]:
    """Strain the positions and the cell, and recompute the shifts.

    Only the symmetric part of the strain is used. An antisymmetric part is an
    infinitesimal rotation, which the energy is invariant under, so its
    gradient is zero and keeping it would make the virial non-symmetric for no
    physical reason.

    The shifts are recomputed rather than strained. ``unit_shifts`` counts
    whole cells and does not move, so it is the authoritative quantity here and
    ``shifts`` is a cache derived from it and the cell of the moment. Straining
    the cached shifts would leave them inconsistent with the strained cell.

    Returns:
        The strained positions, the recomputed shifts, the strained cell and
        the displacement leaf.
    """
    if displacement is None:
        # The leaf has to be part of the graph before any einsum touches it, or
        # `autograd.grad` reports it unused on a structure whose energy happens
        # not to reach it yet. Adding a zero multiple of the positions is what
        # connects it; the frozen tree does the same thing.
        displacement = torch.zeros(
            (num_graphs, 3, 3), dtype=positions.dtype, device=positions.device
        )
        displacement = displacement + positions.sum() * 0.0

    symmetric = 0.5 * (displacement + displacement.transpose(-1, -2))
    positions = positions + torch.einsum("be,bec->bc", positions, symmetric[batch])
    cell = cell.view(-1, 3, 3)
    cell = cell + torch.matmul(cell, symmetric)
    # `unit_shifts` counts whole cells, so it is naturally an integer array.
    # It multiplies a cell here, so it is cast rather than required to arrive
    # as a float: a caller holding it as counts is holding it correctly.
    shifts = torch.einsum("be,bec->bc", unit_shifts.to(cell.dtype), cell[batch[sender]])
    return positions, shifts, cell, displacement


def prepare_inputs(
    graph: Mapping[str, Any],
    need_forces: bool = True,
    need_stress: bool = False,
    leaves: Iterable[str] = (),
) -> tuple[dict[str, Any], Tensor, Tensor | None]:
    """Phase A: grad leaves, strain, shifts, edge vectors.

    Args:
        graph: The flat dict. **Read and never written to.** The frozen tree
            writes the strained positions and shifts back into the dict it was
            given, which a caller sharing that dict sees.
        need_forces: Whether the positions become a graph leaf.
        need_stress: Whether a strain is injected at all.
        leaves: Names of declared inputs that also become graph leaves. They
            are made leaves here rather than in the model, which is what keeps
            the model free of `requires_grad_`.

    Returns:
        A new dict, the positions leaf, and the displacement leaf or ``None``.

    An externally supplied ``displacement`` key is used as the leaf instead of a
    fresh zero one, so a caller that manages its own strain keeps working.
    """
    positions = graph["positions"]
    if need_forces and not positions.requires_grad:
        positions = positions.clone().requires_grad_(True)

    prepared = dict(graph)
    displacement: Tensor | None = None
    shifts = graph["shifts"]

    if need_stress:
        if "unit_shifts" not in graph:
            raise KeyError(
                "a stress needs `unit_shifts`: the shifts are recomputed from "
                "it and the strained cell, and straining the cached `shifts` "
                "instead would leave them inconsistent with that cell. The "
                "keys present are "
                f"{sorted(graph)}."
            )
        supplied = graph.get("displacement")
        if supplied is not None and not supplied.requires_grad:
            supplied = supplied.clone().requires_grad_(True)
        cell = graph["cell"]
        positions, shifts, cell, displacement = _symmetric_strain(
            positions,
            graph["unit_shifts"],
            cell,
            graph["edge_index"][0],
            graph["batch"],
            int(graph["num_graphs"]),
            supplied,
        )
        prepared["cell"] = cell

    prepared["positions"] = positions
    prepared["shifts"] = shifts
    if displacement is not None:
        prepared["displacement"] = displacement
    for name in leaves:
        value = graph[name]
        prepared[name] = (
            value if value.requires_grad else value.clone().requires_grad_(True)
        )

    sender, receiver = graph["edge_index"][0], graph["edge_index"][1]
    prepared["vectors"] = positions[receiver] - positions[sender] + shifts
    return prepared, positions, displacement


class DerivativeEngine(nn.Module):
    """The only place in the stack that calls ``autograd.grad``.

    Args:
        backbone: The node-feature model.
        output_layer: The observable heads.
        inputs: The declared model inputs beyond the positions and the cell.
            The differentiable ones become graph leaves in phase A and can be
            asked for by name. Nothing here knows what any of them mean: a
            magnetic moment and an external field go through the same code, and
            neither appears as a literal in it.

    The energy that is differentiated is always ``total_energy``, for both of
    the shapes the frozen tree has. Its plain model differentiates the total
    and its scale-shift model differentiates the interaction energy only, and
    with readout-only isolated-atom energies the two agree exactly: the E0
    branch is a table lookup with no path back to the positions, so
    ``autograd.grad`` never traverses it. One call covers both.
    """

    def __init__(
        self,
        backbone: nn.Module,
        energy: ObservableSpec,
        output_layer: nn.Module | None = None,
        inputs: Iterable[InputSpec] = (),
    ) -> None:
        super().__init__()
        # Either a backbone and an output layer, or one model that is already
        # both. The engine brackets a model; that the model came in two pieces
        # was an assumption, and a model with a pair repulsion in it does not.
        self.backbone = backbone
        self.output_layer = output_layer
        self.energy = energy
        self.inputs = list(inputs)
        self.differentiable_inputs = [
            spec for spec in self.inputs if spec.differentiable
        ]

    def derivative_names(self) -> dict[str, str]:
        """The name each declared input's energy derivative is reported under.

        Read off the energy observable's own declaration. A pair with a name of
        its own says so there, beside its sign and its units, so a quantity
        like ``magforces`` needs no code here and no table anywhere.
        """
        return {
            spec.name: self.energy.derivative_name(spec.name)
            for spec in self.differentiable_inputs
        }

    def forward(
        self,
        graph: Mapping[str, Any],
        compute: Iterable[str] = ("forces",),
        training: bool = False,
    ) -> MACEOutput[Tensor]:
        """The observables, plus whichever derivatives were asked for.

        Args:
            graph: The flat dict, read only.
            compute: Any of ``forces``, ``stress``, ``virials``,
                ``edge_forces``.
            training: ``True`` keeps the graph alive so the derivative can
                itself be differentiated, which is what force training needs.
        """
        wanted = set(compute)
        by_input = self.derivative_names()
        known = {"forces", "stress", "virials", "edge_forces"} | set(by_input.values())
        unknown = sorted(wanted - known)
        if unknown:
            undeclared = [
                name
                for name in unknown
                if any(
                    self.energy.derivative_name(spec.name) == name
                    for spec in self.inputs
                )
            ]
            if undeclared:
                raise ValueError(
                    f"{undeclared} would come from an input that is declared "
                    f"but not differentiable. Set `differentiable: true` on it, "
                    f"or stop asking for the derivative."
                )
            raise ValueError(
                f"{unknown} are not derivatives this computes. The choices are "
                f"{sorted(known)}."
            )
        need_strain = bool(wanted & {"stress", "virials"})
        need_forces = bool(wanted & {"forces", "edge_forces"}) or need_strain
        # Only the inputs whose derivative was actually asked for. A leaf that
        # nobody differentiates still holds the whole backward graph alive.
        leaves = [
            spec for spec in self.differentiable_inputs if by_input[spec.name] in wanted
        ]
        for spec in leaves:
            if spec.name not in graph:
                raise KeyError(
                    f"{by_input[spec.name]!r} was requested, so the declared "
                    f"input {spec.name!r} has to be in the graph, and it is "
                    f"not. The keys present are {sorted(graph)}. Absence is "
                    f"the key being absent; a value of all zeros is a present "
                    f"input whose value is zero."
                )

        prepared, positions, displacement = prepare_inputs(
            graph,
            need_forces=need_forces,
            need_stress=need_strain,
            leaves=[spec.name for spec in leaves],
        )
        if self.output_layer is None:
            output = self.backbone(prepared)
        else:
            output = self.output_layer(prepared, self.backbone(prepared))
        if not wanted:
            return output

        targets: list[Tensor] = []
        order: list[str] = []
        if need_forces:
            targets.append(positions)
            order.append("positions")
        if need_strain:
            assert displacement is not None
            targets.append(displacement)
            order.append("displacement")
        if "edge_forces" in wanted:
            targets.append(prepared["vectors"])
            order.append("vectors")
        for spec in leaves:
            targets.append(prepared[spec.name])
            order.append(spec.name)

        energy = output.total_energy
        if energy is None:
            raise ValueError(
                "a derivative was requested but the model produced no "
                "`total_energy`. Declare the `energy` observable, or ask for "
                "no derivatives."
            )
        gradients = torch.autograd.grad(
            outputs=[energy],
            inputs=targets,
            grad_outputs=[torch.ones_like(energy)],
            retain_graph=training or len(targets) > 1,
            create_graph=training,
            allow_unused=True,
        )
        by_name = dict(zip(order, gradients, strict=True))

        if "forces" in wanted:
            gradient = by_name["positions"]
            # A completely dissociated structure has no edges, so the energy
            # does not depend on the positions at all and the gradient comes
            # back as None rather than as zeros.
            output.forces = (
                torch.zeros_like(positions) if gradient is None else -gradient
            )
        if need_strain:
            strain_gradient = by_name["displacement"]
            if strain_gradient is None:
                strain_gradient = torch.zeros_like(displacement)
            if "virials" in wanted:
                output.virials = -strain_gradient
            if "stress" in wanted:
                output.stress = stress_from_strain_gradient(
                    strain_gradient,
                    prepared["cell"],
                    # Membership rather than `.get`: the parity harness hands
                    # this a legacy `Batch`, which implements `__contains__`
                    # and not `.get`, so the shorter spelling raises an
                    # AttributeError naming neither the key nor the caller.
                    graph["pbc"] if "pbc" in graph else None,  # noqa: SIM401
                )
        for spec in leaves:
            name = by_input[spec.name]
            gradient = by_name[spec.name]
            if gradient is None:
                gradient = torch.zeros_like(prepared[spec.name])
            output.extras[name] = self.energy.derivative_sign(spec.name) * gradient

        if "edge_forces" in wanted:
            gradient = by_name["vectors"]
            # The sign is flipped once, here, so the deployment adapters do not
            # each flip it again on their own.
            output.extras["edge_forces"] = (
                torch.zeros_like(prepared["vectors"]) if gradient is None else -gradient
            )
        return output
