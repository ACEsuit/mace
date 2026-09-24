"""A charge-aware model: MACE plus a self-consistent density of multipoles.

The model reads a density of Gaussian multipoles off the node features, one
per spin channel, fixes each channel's total to the structure's charge and
spin, and then refines it for a fixed number of steps. Each step projects the
potential of the current density onto every atom through the long-range
solver, adds an applied field, and predicts an increment from what each atom
reads. The energy is the local model's, plus the electrostatic energy of the
final density, plus optionally a local energy of that density, plus the
applied field's work on its dipole.

**The recursion is unrolled, not converged.** It runs exactly
``num_recursion_steps`` steps and forces are the derivative through every one
of them. There is no fixed point and so no implicit derivative: a force taken
at a converged density, as a variational model's is, would be a different
force from the one this model was trained against.

**The long-range ops are resolved once, when the model is built.** Both the
projection and the energy come from the solver the model names, through the
dispatch layer, and they share one geometry per forward. The positions and the
cell they read are the graph's, strained by the derivative engine when a
stress is wanted, so the reciprocal cell and the volume move with the strain
and the stress is the derivative of the energy computed.

**Charge and spin are graph inputs.** ``total_charge`` is in units of the
elementary charge and ``total_spin`` is the multiplicity ``2S + 1``, so a
neutral singlet is ``0`` and ``1``. ``external_field`` is the applied field,
in V/Angstrom, one vector per structure.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import torch
from mace_core.electrostatics import (
    ElectrostaticsSolverDescriptor,
    FeatureProjection,
    PeriodicityProfile,
    RealspaceMethod,
)
from mace_core.kernels.descriptors import LinearDescriptor, RadialKind
from mace_core.kernels.precision import Precision
from mace_core.observables import ObservableSpec
from mace_core.outputs import MACEOutput
from torch import Tensor, nn

from mace_torch.electrostatics import (
    build_long_range,
    build_long_range_features,
    long_range_geometry,
)
from mace_torch.electrostatics.reference.gto_utils import (
    DisplacedGTOExternalFieldBlock,
    gto_basis_kspace_cutoff,
)
from mace_torch.kernels import segment_sum
from mace_torch.models.base import MACEModel, _constructed_in
from mace_torch.models.energy import EnergyOutputHead
from mace_torch.nn.field import BiasReadout, ChargeUpdate, ElectronEnergyReadout
from mace_torch.nn.layout import expanded_irreps

__all__ = ["POLAR_EXTRA_ROWS", "POLAR_GRAPH_INPUTS", "PolarModel", "PolarSettings"]

#: What each quantity the model adds to ``extras`` has a row for.
POLAR_EXTRA_ROWS: dict[str, str] = {
    "electrostatic_energy": "graph",
    "electron_energy": "graph",
    "total_charge": "graph",
    "charges": "atom",
    "spins": "atom",
    "density_coefficients": "atom",
    "spin_charge_density": "atom",
    "fukui_functions": "atom",
}


#: The per-structure inputs the model reads, which the data stage writes into
#: every graph it builds for it.
POLAR_GRAPH_INPUTS: tuple[str, ...] = ("total_charge", "total_spin", "external_field")


def _spherical(max_l: int, copies: int = 1) -> str:
    """The spherical harmonics up to ``max_l``, ``copies`` of each, sorted."""
    return "+".join(
        f"{copies}x{degree}{'e' if degree % 2 == 0 else 'o'}"
        for degree in range(max_l + 1)
    )


@dataclass(frozen=True)
class PolarSettings:
    """Everything about the charge-aware part of the model.

    Attributes:
        multipole_max_l: The highest multipole order of the density.
        multipole_width: The Gaussian width of each atom's density, in
            Angstrom.
        feature_max_l: The highest order the potential is projected onto.
        feature_widths: The widths it is projected onto, in Angstrom.
        feature_norms: One divisor per order and width, order major, applied
            to the projected potential. ``None`` divides by one.
        num_recursion_steps: How many refinement steps the density takes.
        kspace_cutoff_factor: The reciprocal-space cutoff, as a multiple of
            the one the widths call for.
        feature_self_interaction: Whether an atom's projection includes its
            own density's potential.
        energy_self_interaction: Whether the electrostatic energy includes each
            atom's interaction with its own density.
        add_local_electron_energy: Whether the local energy of the density is
            added to the total. It is computed and reported either way.
        quadrupole_feature_corrections: Whether an open system's projection
            carries the quadrupole correction.
        fukui_hidden: The width of the Fukui readout's middle.
        periodicity_profile: The systems the solve is set up for.
        slab_normal: The slab correction's axis, for ``z_slab``.
        realspace_method: How an open system is summed in real space.
    """

    multipole_max_l: int = 1
    multipole_width: float = 1.0
    feature_max_l: int = 1
    feature_widths: tuple[float, ...] = (1.0,)
    feature_norms: tuple[float, ...] | None = None
    num_recursion_steps: int = 1
    kspace_cutoff_factor: float = 1.5
    feature_self_interaction: bool = False
    energy_self_interaction: bool = False
    add_local_electron_energy: bool = False
    quadrupole_feature_corrections: bool = False
    fukui_hidden: int = 16
    periodicity_profile: PeriodicityProfile = "partial"
    slab_normal: int | None = None
    realspace_method: RealspaceMethod = "finite_difference"

    def __post_init__(self) -> None:
        expected = len(self.feature_widths) * (self.feature_max_l + 1)
        if self.feature_norms is not None and len(self.feature_norms) != expected:
            raise ValueError(
                f"feature_norms has {len(self.feature_norms)} entries and the "
                f"projection has {expected}: one per order up to "
                f"{self.feature_max_l}, for each of the {len(self.feature_widths)} "
                f"widths."
            )
        if self.num_recursion_steps < 0:
            raise ValueError(
                f"num_recursion_steps is {self.num_recursion_steps}; it counts "
                f"refinement steps, so it is at least 0."
            )

    def descriptor(self, precision: Precision) -> ElectrostaticsSolverDescriptor:
        """The solve this model asks its solver for."""
        kspace_cutoff = self.kspace_cutoff_factor * gto_basis_kspace_cutoff(
            [self.multipole_width, *self.feature_widths],
            max(self.multipole_max_l, self.feature_max_l),
        )
        return ElectrostaticsSolverDescriptor(
            periodicity_profile=self.periodicity_profile,
            multipole_max_l=self.multipole_max_l,
            kspace_cutoff=float(kspace_cutoff),
            smearing_width=self.multipole_width,
            slab_normal=self.slab_normal,
            features=FeatureProjection(
                max_l=self.feature_max_l,
                widths=tuple(self.feature_widths),
                include_self_interaction=self.feature_self_interaction,
                quadrupole_corrections=self.quadrupole_feature_corrections,
            ),
            realspace_method=self.realspace_method,
            external_field_flags=(
                frozenset({"include_self_interaction"})
                if self.energy_self_interaction
                else frozenset()
            ),
            precision=precision,
        )


class PolarModel(MACEModel):
    """MACE with a self-consistent density of Gaussian multipoles.

    Args:
        backend: The kernel backend. Consulted at construction only.
        atomic_numbers: The element table, ascending.
        observables: What the model reads out through heads. The dipole is not
            among them: it is the density's, computed here.
        energy_head: The energy head.
        settings: The charge-aware part.
        solver: The registered electrostatics solver both long-range ops come
            from.
        trains_derivatives: Whether the model is trained on forces or stress,
            which needs the solver differentiable twice.
        element_agnostic_product: One set of product weights for every
            element. A choice, not a requirement: every published model of
            this kind sets it, and the frozen tree's command line defaults it
            off.
        The rest are :class:`MACEModel`'s. The last layer keeps every irrep and
        the harmonics read the edge as ``(y, z, x)``, always: the density's
        dipoles are read off the last layer in the solver's component order.
    """

    #: What the model computes itself, from its density, so it has no head.
    PRODUCED: frozenset[str] = frozenset({"dipole"})

    field_norms: Tensor
    #: Whichever solver built it: ``prepare(geometry)`` once per forward, then
    #: called once per density. A module, typed loosely because a solver's own
    #: op need not be the reference's class.
    projection: Any

    def __init__(
        self,
        backend,
        atomic_numbers: Sequence[int],
        observables: Sequence[ObservableSpec],
        energy_head: EnergyOutputHead,
        settings: PolarSettings,
        solver: str = "reference",
        trains_derivatives: bool = False,
        num_layers: int = 2,
        num_features: int = 16,
        lmax: int = 2,
        hidden_irreps: str = "0e+1o",
        num_radial: int = 8,
        cutoff: float = 5.0,
        correlation: int = 3,
        avg_num_neighbors: float = 1.0,
        radial_kind: RadialKind = "bessel",
        precision: Precision = "float64",
        cutoff_order: int = 6,
        readout_hidden: int | str = 16,
        num_heads: int = 1,
        element_agnostic_product: bool = False,
    ) -> None:
        produced = sorted(
            spec.name for spec in observables if spec.name in self.PRODUCED
        )
        if produced:
            raise ValueError(
                f"{produced} is computed from the model's density, so it has no "
                f"head of its own. Leave it out of the declared observables."
            )
        super().__init__(
            backend,
            atomic_numbers=atomic_numbers,
            observables=observables,
            energy_head=energy_head,
            num_layers=num_layers,
            num_features=num_features,
            lmax=lmax,
            hidden_irreps=hidden_irreps,
            num_radial=num_radial,
            cutoff=cutoff,
            correlation=correlation,
            avg_num_neighbors=avg_num_neighbors,
            radial_kind=radial_kind,
            precision=precision,
            cutoff_order=cutoff_order,
            readout_hidden=readout_hidden,
            num_heads=num_heads,
            last_layer_irreps=hidden_irreps,
            element_agnostic_product=element_agnostic_product,
            # The density's dipoles are read off the degree one features, and
            # the solver holds a dipole as (y, z, x).
            edge_axes=(1, 2, 0),
        )
        self.settings = settings
        self.num_recursion_steps = settings.num_recursion_steps
        self.add_local_electron_energy = settings.add_local_electron_energy
        self.descriptor = settings.descriptor(precision)

        node_irreps = expanded_irreps(hidden_irreps, num_features)
        one_spin = _spherical(settings.multipole_max_l)
        self.density_irreps = f"{one_spin}+{one_spin}"
        self.density_width = (settings.multipole_max_l + 1) ** 2
        field = _spherical(settings.feature_max_l, len(settings.feature_widths))
        self.potential_irreps = f"{field}+{field}"

        with _constructed_in(precision):

            def linear(irreps_in: str, irreps_out: str):
                return backend.make_linear(
                    LinearDescriptor(
                        irreps_in=irreps_in, irreps_out=irreps_out, precision=precision
                    )
                )

            self.source_maps = nn.ModuleList(
                linear(node_irreps, self.density_irreps) for _ in range(num_layers)
            )
            self.layer_mixer = nn.ModuleList(
                linear(node_irreps, node_irreps) for _ in range(num_layers)
            )
            self.fukui_readout = BiasReadout(
                backend,
                node_irreps,
                f"{settings.fukui_hidden}x0e",
                "2x0e",
                precision,
            )
            self.updates = nn.ModuleList(
                ChargeUpdate(
                    backend,
                    node_irreps,
                    self.potential_irreps,
                    self.density_irreps,
                    num_elements=len(atomic_numbers),
                    precision=precision,
                )
                for _ in range(settings.num_recursion_steps)
            )
            self.electron_energy = ElectronEnergyReadout(
                backend,
                node_irreps,
                self.potential_irreps,
                self.density_irreps,
                precision,
            )
            self.external_field = DisplacedGTOExternalFieldBlock(
                settings.feature_max_l, list(settings.feature_widths), "receiver"
            )
            norms = settings.feature_norms or (1.0,) * (
                len(settings.feature_widths) * (settings.feature_max_l + 1)
            )
            expanded = [
                norms[degree * len(settings.feature_widths) + width]
                for degree in range(settings.feature_max_l + 1)
                for width in range(len(settings.feature_widths))
                for _ in range(2 * degree + 1)
            ]
            self.register_buffer(
                "field_norms", torch.tensor(expanded, dtype=getattr(torch, precision))
            )
            # Under the model's precision too: both ops build their basis
            # constants in the process default.
            self.projection = build_long_range_features(
                self.descriptor, solver=solver, trains_derivatives=trains_derivatives
            )
            self.coulomb = build_long_range(
                self.descriptor, solver=solver, trains_derivatives=trains_derivatives
            )
        self.solver = solver

    def solver_record(self) -> dict[str, Any]:
        """What a checkpoint records about the long-range solver.

        The solver's name, whether it reproduces the reference bit for bit,
        and the solve as data. See :class:`mace_core.metadata.ElectrostaticsRecord`.
        """
        from mace_core.electrostatics import descriptor_record, get_solver

        return {
            "solver": self.solver,
            "bit_parity": bool(get_solver(self.solver).capabilities.bit_parity),
            "descriptor": descriptor_record(self.descriptor),
        }

    def _fix_totals(
        self,
        density: Tensor,
        fukui: Tensor,
        spin_totals: Tensor,
        batch: Tensor,
        num_graphs: int,
    ) -> tuple[Tensor, Tensor]:
        """Move each spin channel's total onto its target, along the Fukui
        weights, which are normalized per structure first.

        Returns:
            The corrected density, and the normalized weights.
        """
        accumulate = torch.float64 if density.dtype == torch.float64 else density.dtype
        norm = segment_sum(fukui.to(accumulate), batch, num_graphs)[batch]
        norm = torch.where(norm == 0, torch.ones_like(norm), norm).to(density.dtype)
        fukui = fukui / norm
        totals = segment_sum(density[:, :, 0].to(accumulate), batch, num_graphs)[
            batch
        ].to(density.dtype)
        correction = torch.zeros_like(density)
        correction[:, :, 0] = fukui * (spin_totals - totals)
        return density + correction, fukui

    def forward(self, graph: Mapping[str, Any]) -> MACEOutput[Tensor]:
        """The energy, its parts, and the density it came from.

        The graph is read and never written to. Beyond an energy model's keys
        it needs ``total_charge`` and ``total_spin`` ``[n_graphs]`` and
        ``external_field`` ``[n_graphs, 3]``.
        """
        for key in ("total_charge", "total_spin", "external_field"):
            if key not in graph:
                raise KeyError(
                    f"the graph carries no {key!r}, which a charge-aware model "
                    f"reads. The data stage writes it for every structure, "
                    f"from the file or from its default."
                )
        features = self.backbone(graph)
        output = self.outputs(graph, features, None)
        assert output.total_energy is not None

        positions = graph["positions"]
        batch = graph["batch"]
        num_graphs = int(graph["num_graphs"])
        nodes = positions.shape[0]
        dtype = positions.dtype
        element = self.backbone.element_index(graph["atomic_numbers"])
        one_hot = torch.nn.functional.one_hot(
            element, len(self.backbone.atomic_numbers)
        ).to(dtype)

        geometry = long_range_geometry(graph, self.descriptor)
        cache = self.projection.prepare(geometry)

        density = sum(
            source(layer)
            for source, layer in zip(self.source_maps, features, strict=True)
        )
        density = density.view(nodes, 2, self.density_width)
        mixed = sum(
            mixer(layer)
            for mixer, layer in zip(self.layer_mixer, features, strict=True)
        )

        # The multiplicity is 2S + 1, so the two channels hold (Q +- 2S) / 2.
        charge = graph["total_charge"].to(dtype)
        unpaired = graph["total_spin"].to(dtype) - 1
        spin_totals = torch.stack(
            [(charge + unpaired) / 2, (charge - unpaired) / 2], dim=-1
        )[batch]
        density, fukui = self._fix_totals(
            density, self.fukui_readout(features[-1]), spin_totals, batch, num_graphs
        )
        initial_density = density

        field = graph["external_field"].to(dtype).view(num_graphs, 3)
        applied = torch.cat([torch.zeros_like(field[:, :1]), field], dim=-1)
        counts = segment_sum(torch.ones_like(positions[:, 0]), batch, num_graphs)
        barycenter = segment_sum(positions, batch, num_graphs) / counts.unsqueeze(-1)
        half_external = 0.5 * self.external_field(
            batch, positions - barycenter[batch], applied
        )

        # With no refinement step the local energy reads no potential at all.
        potential = positions.new_zeros((nodes, 2 * self.field_norms.shape[0]))
        for update in self.updates:
            alpha = self.projection(cache, density[:, 0])
            beta = self.projection(cache, density[:, 1])
            potential = torch.cat(
                [
                    (alpha + half_external) / self.field_norms,
                    (beta + half_external) / self.field_norms,
                ],
                dim=-1,
            )
            step = update(one_hot, mixed, potential, density.reshape(nodes, -1))
            density = density + step[:, :-2].view(nodes, 2, self.density_width)
            density, fukui = self._fix_totals(
                density, step[:, -2:], spin_totals, batch, num_graphs
            )

        local = self.electron_energy(
            features[-1],
            potential,
            (initial_density + density).reshape(nodes, -1),
        )
        electron_energy = segment_sum(local, batch, num_graphs)
        if not self.add_local_electron_energy:
            electron_energy = torch.zeros_like(electron_energy)

        charge_density = density.sum(dim=1)
        spin_density = density[:, 0] - density[:, 1]
        dipole = segment_sum(positions * charge_density[:, :1], batch, num_graphs)
        if self.density_width > 1:
            # The dipole components are held in the (y, z, x) order.
            dipole = (
                dipole
                + segment_sum(charge_density[:, 1:4], batch, num_graphs)[:, [2, 0, 1]]
            )
        electrostatic_energy = self.coulomb(graph, charge_density, geometry)

        output.total_energy = (
            output.total_energy
            + electron_energy
            + electrostatic_energy
            + (field * dipole).sum(dim=-1)
        )
        output.dipole = dipole
        output.extras.update(
            {
                "electrostatic_energy": electrostatic_energy,
                "electron_energy": electron_energy,
                "total_charge": segment_sum(charge_density[:, 0], batch, num_graphs),
                "charges": charge_density[:, 0],
                "spins": spin_density[:, 0],
                "density_coefficients": charge_density,
                "spin_charge_density": density,
                "fukui_functions": fukui,
            }
        )
        return output
