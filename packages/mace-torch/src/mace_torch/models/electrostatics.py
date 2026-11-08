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
from typing import Any, cast

import torch
from mace_core.clebsch_gordan.irreps import Irreps
from mace_core.config.electrostatics import ElectrostaticsConfig
from mace_core.config.local_sources import LocalSplitChargeConfig
from mace_core.config.model import ModelConfig, PolarConfig
from mace_core.electrostatics import (
    ElectrostaticsSolverDescriptor,
    FeatureProjection,
    ScfSpec,
)
from mace_core.kernels.descriptors import LinearDescriptor
from mace_core.kernels.precision import Precision, widest
from mace_core.observables import InputSpec, ObservableSpec, irreps_dimension
from mace_core.outputs import MACEOutput
from torch import Tensor, nn

from mace_torch.backends.layout import expanded_irreps
from mace_torch.backends.precision import cast_at_boundary, constructed_in
from mace_torch.electrostatics import (
    LONG_RANGE_PLAN,
    build_long_range,
    build_long_range_features,
    build_scf_solve,
    long_range_geometry,
    long_range_plan,
)
from mace_torch.electrostatics.reference.gto_utils import (
    DisplacedGTOExternalFieldBlock,
    gto_basis_kspace_cutoff,
)
from mace_torch.kernels import segment_sum
from mace_torch.models.base import MACEModel
from mace_torch.models.energy import EnergyOutputHead
from mace_torch.models.heads import ObservableHead
from mace_torch.nn.charge_transfer import SplitChargeSource
from mace_torch.nn.edges import EdgeInputs
from mace_torch.nn.field import BiasReadout, ChargeUpdate, ElectronEnergyReadout
from mace_torch.nn.fixed_point_scf import AffineFieldUpdate, FixedPointEnergy
from mace_torch.physics.charge_equilibration import ChargeEquilibration

__all__ = [
    "POLAR_GRAPH_INPUTS",
    "FixedPointSCFModel",
    "LocalChargeModel",
    "LocalSplitChargeModel",
    "PolarModel",
    "QEqModel",
]


def _spherical(max_l: int, copies: int = 1) -> str:
    """The spherical harmonics up to ``max_l``, ``copies`` of each, sorted."""
    return "+".join(
        f"{copies}x{degree}{'e' if degree % 2 == 0 else 'o'}"
        for degree in range(max_l + 1)
    )


def _polar_extra_observables(max_l: int) -> tuple[ObservableSpec, ...]:
    density = _spherical(max_l)
    declarations = (
        ("electrostatic_energy", "0e", False, "eV"),
        ("electron_energy", "0e", False, "eV"),
        ("total_charge", "0e", False, "e"),
        ("charges", "0e", True, "e"),
        ("spins", "0e", True, "1"),
        ("density_coefficients", density, True, "e*Å^l"),
        ("spin_density", density, True, "e*Å^l"),
        (
            "spin_charge_density",
            _spherical(max_l, 2),
            True,
            "e*Å^l",
        ),
        ("fukui_functions", "2x0e", True, "1"),
        ("fermi_level", "0e", False, "eV"),
        ("external_field", "1o", False, "eV/(e*Å)"),
    )
    return tuple(
        ObservableSpec(name=name, irreps=irreps, per_atom=per_atom, units=units)
        for name, irreps, per_atom, units in declarations
    )


POLAR_INPUT_SPECS = (
    InputSpec(name="total_charge", irreps="0e", per_atom=False, units="e"),
    InputSpec(name="total_spin", irreps="0e", per_atom=False, units="1"),
    InputSpec(name="external_field", irreps="1o", per_atom=False, units="eV/(e*Å)"),
)
POLAR_GRAPH_INPUTS = tuple(spec.name for spec in POLAR_INPUT_SPECS)


def _polar_descriptor(
    polar: PolarConfig, electrostatics: ElectrostaticsConfig, precision: Precision
) -> ElectrostaticsSolverDescriptor:
    """Resolve the solver descriptor from validated architecture and solver settings."""
    kspace_cutoff = electrostatics.kspace_cutoff_factor * gto_basis_kspace_cutoff(
        [polar.multipole_width, *polar.feature_widths],
        max(polar.multipole_max_l, polar.feature_max_l),
    )
    return ElectrostaticsSolverDescriptor(
        periodicity_profile=electrostatics.periodicity_profile,
        multipole_max_l=polar.multipole_max_l,
        kspace_cutoff=float(kspace_cutoff),
        smearing_width=polar.multipole_width,
        slab_normal=electrostatics.slab_normal,
        features=FeatureProjection(
            max_l=polar.feature_max_l,
            widths=polar.feature_widths,
            include_self_interaction=polar.feature_self_interaction,
            quadrupole_corrections=polar.quadrupole_feature_corrections,
        ),
        realspace_method="finite_difference",
        external_field_flags=(
            frozenset({"include_self_interaction"})
            if polar.energy_self_interaction
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
        electrostatics: The solver and boundary conditions. The architecture
            is read directly from ``config.polar``.
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
        config: ModelConfig,
        atomic_numbers: Sequence[int],
        observables: Sequence[ObservableSpec],
        energy_head: EnergyOutputHead,
        electrostatics: ElectrostaticsConfig,
        trains_derivatives: bool = False,
        *,
        avg_num_neighbors: float = 1.0,
        radial_precision: Precision | None = None,
        precision: Precision = "float64",
        num_heads: int = 1,
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
            config,
            atomic_numbers=atomic_numbers,
            observables=observables,
            energy_head=energy_head,
            avg_num_neighbors=avg_num_neighbors,
            radial_precision=radial_precision,
            precision=precision,
            num_heads=num_heads,
            last_layer_irreps=config.node_irreps,
            # The solver stores dipoles in (y, z, x) order.
            edge_axes=(1, 2, 0),
        )
        num_layers, num_features = config.num_interactions, config.num_channels
        hidden_irreps = config.node_irreps
        settings = config.polar
        solver = electrostatics.solver
        self.num_recursion_steps = settings.num_recursion_steps
        self.add_local_electron_energy = settings.add_local_electron_energy
        # Displaced charges and projected potentials are subtracted after
        # reduction. Carry the complete solve at the configured accumulation
        # floor; casting only the final sum cannot recover cancelled bits.
        self.descriptor = _polar_descriptor(
            settings, electrostatics, widest(precision, energy_head.accumulate)
        )
        self.total_dtype = getattr(torch, self.descriptor.precision)

        node_irreps = expanded_irreps(hidden_irreps, num_features)
        one_spin = _spherical(settings.multipole_max_l)
        self.density_irreps = f"{one_spin}+{one_spin}"
        self.density_width = (settings.multipole_max_l + 1) ** 2
        field = _spherical(settings.feature_max_l, len(settings.feature_widths))
        self.potential_irreps = f"{field}+{field}"

        with constructed_in(precision):

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
                f"{irreps_dimension(config.readout.mlp_irreps)}x0e",
                "2x0e",
                precision,
                gate=config.readout.gate,
            )
            self.updates = nn.ModuleList(
                ChargeUpdate(
                    backend,
                    node_irreps,
                    self.potential_irreps,
                    self.density_irreps,
                    num_elements=len(atomic_numbers),
                    precision=precision,
                    accumulate=self.descriptor.precision,
                )
                for _ in range(settings.num_recursion_steps)
            )
            self.electron_energy = ElectronEnergyReadout(
                backend,
                node_irreps,
                self.potential_irreps,
                self.density_irreps,
                precision,
                accumulate=self.descriptor.precision,
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
        with constructed_in(self.descriptor.precision):
            self.projection = build_long_range_features(
                self.descriptor, solver=solver, trains_derivatives=trains_derivatives
            )
            self.coulomb = build_long_range(
                self.descriptor, solver=solver, trains_derivatives=trains_derivatives
            )
        if self.descriptor.precision != precision:
            cast_at_boundary(self.projection, self.descriptor.precision, precision)
            cast_at_boundary(self.coulomb, self.descriptor.precision, precision)
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
        # Rounding the weights or the residual before adding the correction
        # can leave two representable, opposite charges with a nonzero sum.
        weights = fukui.to(self.total_dtype)
        norm = segment_sum(weights, batch, num_graphs)[batch]
        weights = weights / torch.where(norm == 0, torch.ones_like(norm), norm)
        charges = density[:, :, 0].to(self.total_dtype)
        residual = spin_totals - segment_sum(charges, batch, num_graphs)[batch]
        charges = (charges + weights * residual).to(density.dtype)
        corrected = torch.cat((charges.unsqueeze(-1), density[:, :, 1:]), dim=-1)
        return corrected, weights.to(fukui.dtype)

    @property
    def input_specs(self) -> tuple[InputSpec, ...]:
        return super().input_specs + POLAR_INPUT_SPECS

    @property
    def extra_observables(self) -> tuple[ObservableSpec, ...]:
        return super().extra_observables + _polar_extra_observables(
            self.config.polar.multipole_max_l
        )

    def prepare_model_inputs(self, graph: dict[str, Any], requests=()) -> None:
        """Resolve discrete long-range choices outside the compiled forward.

        Runtimes may supply a reusable plan while cells, periodicity and atom
        counts stay fixed. Otherwise each call plans the current geometry.
        """
        if LONG_RANGE_PLAN not in graph:
            graph[LONG_RANGE_PLAN] = long_range_plan(
                graph,
                self.descriptor,
                (self.projection, self.coulomb),
                dense_pairs=any(graph.get("structure_padding", ())),
            )

    def forward(self, graph: Mapping[str, Any]) -> MACEOutput[Tensor]:
        """The energy, its parts, and the density it came from.

        The graph is read and never written to. Beyond an energy model's keys
        it needs ``total_charge`` and ``total_spin`` ``[n_graphs]`` and
        ``external_field`` ``[n_graphs, 3]``.

        Extras include ``spin_density`` ``[n_atoms, (multipole_max_l + 1)**2]``,
        the alpha minus beta multipoles in units of e A**l; ``external_field``
        ``[n_graphs, 3]`` in V/Angstrom; and ``fermi_level`` ``[n_graphs]`` in eV.
        The Fermi level is zero: this model fixes charge and spin and does not
        couple to an electron reservoir.
        """
        for key in ("total_charge", "total_spin", "external_field"):
            if key not in graph:
                raise KeyError(
                    f"the graph carries no {key!r}, which a charge-aware model "
                    f"reads. The data stage writes it for every structure, "
                    f"from the file or from its default."
                )
        features = list(self.backbone(graph).layers)
        output = self.outputs(graph, features, None)
        assert output.total_energy is not None

        positions = graph["positions"]
        batch = graph["batch"]
        num_graphs = int(graph["ptr"].shape[0] - 1)
        nodes = positions.shape[0]
        dtype = positions.dtype
        element = graph["node_attrs"].argmax(-1)
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
                "total_charge": segment_sum(
                    charge_density[:, 0].to(self.total_dtype), batch, num_graphs
                ).to(dtype),
                "charges": charge_density[:, 0],
                "spins": spin_density[:, 0],
                "density_coefficients": charge_density,
                "spin_density": spin_density,
                "spin_charge_density": density,
                "fukui_functions": fukui,
                "fermi_level": applied[:, 0],
                "external_field": field,
            }
        )
        return output

    def canonical_components(self) -> dict[str, nn.Module | None]:
        return {
            **super().canonical_components(),
            "source_maps": self.source_maps,
            "layer_mixer": self.layer_mixer,
            "fukui_readout": self.fukui_readout,
            "updates": self.updates,
            "electron_energy": self.electron_energy,
        }


class _ChargeAwareModel(MACEModel):
    """Shared input, solver-metadata and planning contracts for source models."""

    produced_observables: dict[str, ObservableSpec]
    descriptor: ElectrostaticsSolverDescriptor
    solver: str

    @staticmethod
    def declared_inputs(config: ModelConfig) -> tuple[InputSpec, ...]:
        return ()

    def solver_record(self) -> dict[str, Any]:
        """Persist solver identity through the existing artifact contract."""
        from mace_core.electrostatics import descriptor_record, get_solver

        return {
            "solver": self.solver,
            "bit_parity": bool(get_solver(self.solver).capabilities.bit_parity),
            "descriptor": descriptor_record(self.descriptor),
        }

    @property
    def input_specs(self) -> tuple[InputSpec, ...]:
        specs = {spec.name: spec for spec in super().input_specs}
        specs.update({spec.name: spec for spec in self.declared_inputs(self.config)})
        return tuple(specs.values())

    @property
    def extra_observables(self) -> tuple[ObservableSpec, ...]:
        return super().extra_observables + tuple(
            spec for name, spec in self.produced_observables.items() if name != "dipole"
        )

    def prepare_model_inputs(self, graph: dict[str, Any], requests=()) -> None:
        if LONG_RANGE_PLAN not in graph:
            graph[LONG_RANGE_PLAN] = long_range_plan(graph, self.descriptor)


class LocalChargeModel(_ChargeAwareModel):
    """Local multipoles followed by one dispatched Coulomb evaluation.

    Based on the LocalCharges architecture in ACEsuit/mace-scf at
    784a510339aacc4412b61b6926ef1a92bca387c0 (MIT). This is a new model,
    with V1's canonical weights and derivative engine; no SCF or total-charge
    projection is applied. Dipole coefficients use the solver's (y,z,x) axes.
    """

    kind = "lc_mace"

    @staticmethod
    def declared_inputs(config: ModelConfig) -> tuple[InputSpec, ...]:
        return (POLAR_INPUT_SPECS[2],)

    @staticmethod
    def solver_descriptor(
        config: ModelConfig,
        electrostatics: ElectrostaticsConfig,
        precision: Precision,
    ) -> ElectrostaticsSolverDescriptor:
        """Resolve model physics and solver settings once at the build boundary."""
        settings = config.local_sources
        if settings is None:
            raise ValueError("model.local_sources is required")
        return ElectrostaticsSolverDescriptor(
            periodicity_profile=electrostatics.periodicity_profile,
            slab_normal=electrostatics.slab_normal,
            multipole_max_l=settings.multipole_max_l,
            smearing_width=settings.multipole_width,
            kspace_cutoff=float(
                electrostatics.kspace_cutoff_factor
                * gto_basis_kspace_cutoff(
                    [settings.multipole_width],
                    settings.multipole_max_l,
                )
            ),
            external_field_flags=frozenset({"include_self_interaction"})
            if settings.energy_self_interaction
            else frozenset(),
            precision=precision,
        )

    def __init__(
        self,
        backend,
        config: ModelConfig,
        atomic_numbers: Sequence[int],
        observables: Sequence[ObservableSpec],
        energy_head: EnergyOutputHead | None,
        *,
        settings: ElectrostaticsSolverDescriptor,
        solver: str = "reference",
        trains_derivatives: bool = False,
        avg_num_neighbors: float = 1.0,
        precision: Precision = "float64",
        num_heads: int = 1,
        radial_precision: Precision | None = None,
    ) -> None:
        sources = config.local_sources
        if sources is None or sources.kind != self.kind:
            raise ValueError(f"model.local_sources.kind must be {self.kind!r}")
        if energy_head is None:
            raise ValueError(f"{self.kind} requires an energy observable")
        if sources.multipole_max_l > config.max_L:
            raise ValueError("local_sources.multipole_max_l exceeds model.max_L")
        self.produced_observables = {
            spec.name: spec
            for spec in (
                ObservableSpec(name="dipole", irreps="1o", per_atom=False, units="e*Å"),
                ObservableSpec(
                    name="density_coefficients",
                    irreps=_spherical(sources.multipole_max_l),
                    per_atom=True,
                    units="e*Å^l",
                ),
                ObservableSpec(name="charges", irreps="0e", per_atom=True, units="e"),
                ObservableSpec(
                    name="total_charge", irreps="0e", per_atom=False, units="e"
                ),
                ObservableSpec(
                    name="electrostatic_energy", irreps="0e", per_atom=False, units="eV"
                ),
                ObservableSpec(
                    name="field_energy", irreps="0e", per_atom=False, units="eV"
                ),
            )
        }
        if isinstance(sources, LocalSplitChargeConfig) and sources.polarizability:
            self.produced_observables["polarizability"] = ObservableSpec(
                name="polarizability", irreps="0e+2e", per_atom=False, units="Å^3"
            )
        super().__init__(
            backend,
            config,
            atomic_numbers,
            [
                spec
                for spec in observables
                if spec.name not in self.produced_observables
            ],
            energy_head,
            avg_num_neighbors=avg_num_neighbors,
            precision=precision,
            num_heads=num_heads,
            radial_precision=radial_precision,
            last_layer_irreps=config.node_irreps,
            edge_axes=(1, 2, 0),
        )
        self.settings = sources
        self.total_dtype = getattr(torch, energy_head.accumulate)
        density_irreps = _spherical(sources.multipole_max_l)
        self.descriptor = settings
        with constructed_in(precision):
            self.source_maps = nn.ModuleList(
                self._make_source(backend, config, layer, density_irreps, precision)
                for layer in self.backbone.layer_irreps
            )
        with constructed_in(self.descriptor.precision):
            self.coulomb = build_long_range(
                self.descriptor,
                solver=solver,
                trains_derivatives=trains_derivatives,
            )
        if self.descriptor.precision != precision:
            cast_at_boundary(self.coulomb, self.descriptor.precision, precision)
        self.solver = solver

    def _make_source(self, backend, config, layer, density_irreps, precision):
        return backend.make_linear(
            LinearDescriptor(
                irreps_in=layer,
                irreps_out=density_irreps,
                precision=precision,
            )
        )

    def density_and_dipole(self, graph, layers) -> tuple[Tensor, Tensor]:
        density = torch.stack(
            [
                source(layer)
                for source, layer in zip(self.source_maps, layers, strict=True)
            ]
        ).sum(0)
        dipole = segment_sum(
            graph["positions"].to(self.total_dtype)
            * density[:, :1].to(self.total_dtype),
            graph["batch"],
            graph["ptr"].shape[0] - 1,
        )
        return density, dipole

    def transform_outputs(self, graph, layers, output) -> MACEOutput[Tensor]:
        """Add Coulomb and +field·dipole energies (eV) before differentiation.

        Site energies retain their short-range meaning, as in the reference.
        The shared solver and graph builder own periodic/aperiodic boundaries.
        """
        density, dipole = self.density_and_dipole(graph, layers)
        batch, count = graph["batch"], graph["ptr"].shape[0] - 1
        if self.settings.multipole_max_l:
            dipole = (
                dipole
                + segment_sum(
                    density[:, 1:4].to(self.total_dtype),
                    batch,
                    count,
                )[:, [2, 0, 1]]
            )
        field = graph["external_field"].reshape(count, 3).to(self.total_dtype)
        field_energy = (field * dipole).sum(-1)
        electrostatic_energy = self.coulomb(graph, density)
        assert output.total_energy is not None
        output.total_energy = output.total_energy + electrostatic_energy + field_energy
        output.dipole = dipole
        output.extras.update(
            {
                "density_coefficients": density,
                "charges": density[:, 0],
                "total_charge": segment_sum(
                    density[:, 0].to(self.total_dtype), batch, count
                ),
                "electrostatic_energy": electrostatic_energy,
                "field_energy": field_energy,
            }
        )
        return output

    def canonical_components(self) -> dict[str, nn.Module | None]:
        return {**super().canonical_components(), "source_maps": self.source_maps}


class LocalSplitChargeModel(LocalChargeModel):
    """Conservative local transfers and multipoles around formal charges.

    The charge-flow and periodic polarization conventions follow LocalSplitCharges
    in the same pinned mace-scf reference. Each directed transfer is added to
    its receiver and subtracted from its sender, including periodic self-images.
    """

    kind = "lsc_mace"
    formal_charges: Tensor

    def _make_source(self, backend, config, layer, density_irreps, precision):
        return SplitChargeSource(
            backend,
            config,
            config.local_sources.multipole_max_l,
            precision,
        )

    @staticmethod
    def declared_inputs(config: ModelConfig) -> tuple[InputSpec, ...]:
        settings = config.local_sources
        inputs = LocalChargeModel.declared_inputs(config)
        if (
            isinstance(settings, LocalSplitChargeConfig)
            and settings.formal_charge_input
        ):
            inputs += (
                InputSpec(
                    name=settings.formal_charge_input,
                    irreps="0e",
                    per_atom=True,
                    units="e",
                ),
            )
        return inputs

    def __init__(
        self,
        backend,
        config: ModelConfig,
        atomic_numbers: Sequence[int],
        observables: Sequence[ObservableSpec],
        energy_head: EnergyOutputHead | None,
        *,
        settings: ElectrostaticsSolverDescriptor,
        solver: str = "reference",
        trains_derivatives: bool = False,
        avg_num_neighbors: float = 1.0,
        precision: Precision = "float64",
        num_heads: int = 1,
        radial_precision: Precision | None = None,
    ):
        super().__init__(
            backend,
            config,
            atomic_numbers,
            observables,
            energy_head,
            settings=settings,
            solver=solver,
            trains_derivatives=trains_derivatives,
            avg_num_neighbors=avg_num_neighbors,
            precision=precision,
            num_heads=num_heads,
            radial_precision=radial_precision,
        )
        sources = self.settings
        assert isinstance(sources, LocalSplitChargeConfig)
        if sources.formal_charges is not None and set(sources.formal_charges) != set(
            atomic_numbers
        ):
            raise ValueError(
                "local_sources.formal_charges must cover exactly "
                "the model's atomic_numbers"
            )
        # A fixed species table belongs to resolved config; this is a derived buffer.
        self.register_buffer(
            "formal_charges",
            torch.tensor(
                [
                    (sources.formal_charges or {}).get(number, 0.0)
                    for number in atomic_numbers
                ],
                dtype=getattr(torch, precision),
            ),
            persistent=False,
        )
        from mace_torch.nn.local_sources import (
            LocalPolarizability,
            OxidationGraphEmbedding,
        )

        with constructed_in(precision):
            if sources.oxidation_mixer != "none":
                self.backbone.graph_embedding = OxidationGraphEmbedding(
                    backend,
                    config,
                    atomic_numbers,
                    precision,
                    self.backbone.graph_embedding,
                )
            self.polarizability_readouts = (
                nn.ModuleList(
                    LocalPolarizability(backend, layer, precision)
                    for layer in self.backbone.layer_irreps
                )
                if sources.polarizability
                else None
            )

    def density_and_dipole(self, graph, layers) -> tuple[Tensor, Tensor]:
        settings = self.settings
        assert isinstance(settings, LocalSplitChargeConfig)
        formal = (
            graph[settings.formal_charge_input].reshape(-1)
            if settings.formal_charge_input
            else graph["node_attrs"] @ self.formal_charges
        ).to(self.total_dtype)
        positions, batch = graph["positions"], graph["batch"]
        sender, receiver = graph["edge_index"]
        vectors = positions[receiver] - positions[sender] + graph["shifts"]
        edges = EdgeInputs(
            vectors[:, self.backbone.edge_axes],
            graph["node_atomic_numbers"],
            graph["edge_index"],
            self.backbone.spherical_harmonics,
            self.backbone.radial_embedding,
        )
        layers = [
            cast(SplitChargeSource, source).condition(layer, formal)
            for source, layer in zip(self.source_maps, layers, strict=True)
        ]
        flux = (
            torch.stack(
                [
                    cast(SplitChargeSource, source).transfer(
                        layer, edges.attributes, edges.radial, graph["edge_index"]
                    )
                    for source, layer in zip(self.source_maps, layers, strict=True)
                ]
            )
            .to(self.total_dtype)
            .sum(0)
        )
        charges = (
            formal
            + segment_sum(flux, receiver, positions.shape[0])
            - segment_sum(flux, sender, positions.shape[0])
        )
        multipoles = torch.stack(
            [
                source(layer)
                for source, layer in zip(self.source_maps, layers, strict=True)
            ]
        ).sum(0)
        density = torch.cat(
            (charges[:, None].to(multipoles.dtype), multipoles[:, 1:]), dim=-1
        )
        count = graph["ptr"].shape[0] - 1
        dipole = segment_sum(
            positions.to(self.total_dtype) * formal[:, None], batch, count
        )
        # Use image-aware bond vectors, not charge*wrapped_position: the latter
        # loses the polarization branch when a neighbour crosses the boundary.
        dipole = dipole + segment_sum(
            flux[:, None] * vectors.to(self.total_dtype), batch[sender], count
        )
        return density, dipole

    def transform_outputs(self, graph, layers, output):
        output = super().transform_outputs(graph, layers, output)
        if self.polarizability_readouts is not None:
            atomic = torch.stack(
                [
                    readout(layer).to(self.total_dtype)
                    for readout, layer in zip(
                        self.polarizability_readouts, layers, strict=True
                    )
                ]
            ).sum(0)
            output.extras["polarizability"] = segment_sum(
                atomic, graph["batch"], graph["ptr"].shape[0] - 1
            )
        return output

    def canonical_components(self):
        components = super().canonical_components()
        if self.polarizability_readouts is not None:
            components["polarizability_readouts"] = self.polarizability_readouts
        return components


class QEqModel(_ChargeAwareModel):
    """Quadratic charge equilibration with a fixed total charge per structure.

    Architecture and energy convention follow MACEQEq in ACEsuit/mace-scf at
    784a510339aacc4412b61b6926ef1a92bca387c0 (MIT). The constrained linear
    solve supplies its implicit derivatives; no iteration history is detached.
    """

    @staticmethod
    def declared_inputs(config: ModelConfig) -> tuple[InputSpec, ...]:
        settings = config.qeq
        assert settings is not None
        return (
            POLAR_INPUT_SPECS[0],
            POLAR_INPUT_SPECS[2],
            *tuple(
                InputSpec(name=name, irreps="0e", per_atom=True, units=units)
                for name, units in (
                    (settings.electronegativity_input, "eV/e"),
                    (settings.hardness_input, "eV/e^2"),
                )
                if name is not None
            ),
        )

    @staticmethod
    def solver_descriptor(
        config: ModelConfig, electrostatics: ElectrostaticsConfig, precision: Precision
    ) -> ElectrostaticsSolverDescriptor:
        settings = config.qeq
        if settings is None:
            raise ValueError("model.qeq is required")
        return ElectrostaticsSolverDescriptor(
            periodicity_profile=electrostatics.periodicity_profile,
            slab_normal=electrostatics.slab_normal,
            multipole_max_l=0,
            smearing_width=settings.multipole_width,
            kspace_cutoff=float(
                electrostatics.kspace_cutoff_factor
                * gto_basis_kspace_cutoff(
                    [settings.multipole_width, *settings.cutoff_widths], 0
                )
            ),
            external_field_flags=frozenset({"include_self_interaction"})
            if settings.energy_self_interaction
            else frozenset(),
            features=FeatureProjection(
                0,
                (settings.multipole_width,),
                normalization="multipoles",
                include_self_interaction=settings.energy_self_interaction,
                quadrupole_corrections=True,
            ),
            scf_spec=ScfSpec(
                method="qeq"
                if settings.charge_method == "energy"
                else "qeq_nonperiodic"
            ),
            precision=precision,
        )

    def __init__(
        self,
        backend,
        config: ModelConfig,
        atomic_numbers: Sequence[int],
        observables: Sequence[ObservableSpec],
        energy_head: EnergyOutputHead | None,
        *,
        settings: ElectrostaticsSolverDescriptor,
        solver: str = "reference",
        trains_derivatives: bool = False,
        avg_num_neighbors: float = 1.0,
        precision: Precision = "float64",
        num_heads: int = 1,
        radial_precision: Precision | None = None,
    ) -> None:
        if config.qeq is None or config.model != "qeq":
            raise ValueError("QEq requires model.qeq and model.model='qeq'")
        if energy_head is None:
            raise ValueError("QEq requires an energy observable")
        self.produced_observables = {
            name: ObservableSpec(
                name=name, irreps=irreps, per_atom=per_atom, units=units
            )
            for name, irreps, per_atom, units in (
                ("dipole", "1o", False, "e*Å"),
                ("charges", "0e", True, "e"),
                ("density_coefficients", "0e", True, "e"),
                ("total_charge", "0e", False, "e"),
                ("electronegativity", "0e", True, "eV/e"),
                ("hardness", "0e", True, "eV/e^2"),
                ("qeq_energy", "0e", False, "eV"),
                ("electrostatic_energy", "0e", False, "eV"),
            )
        }
        super().__init__(
            backend,
            config,
            atomic_numbers,
            [
                spec
                for spec in observables
                if spec.name not in self.produced_observables
            ],
            energy_head,
            avg_num_neighbors=avg_num_neighbors,
            precision=precision,
            num_heads=num_heads,
            radial_precision=radial_precision,
            last_layer_irreps=config.node_irreps,
            edge_axes=(1, 2, 0),
        )
        self.settings = config.qeq
        self.descriptor, self.solver = settings, solver
        self.solve_dtype = getattr(torch, settings.precision)
        with constructed_in(precision):
            self.source_maps = nn.ModuleDict(
                {
                    name: ObservableHead(
                        backend,
                        self.produced_observables[name],
                        [config.node_irreps] * config.num_interactions,
                        config.num_channels,
                        nonlinear=True,
                        readout_irreps=config.readout.mlp_irreps,
                        precision=precision,
                        num_heads=num_heads,
                        gate=config.readout.gate,
                    )
                    for name in (
                        ("electronegativity", "hardness")
                        if self.settings.train_hardness
                        else ("electronegativity",)
                    )
                }
            )
        with constructed_in(settings.precision):
            self.coulomb = build_long_range(
                settings, solver=solver, trains_derivatives=trains_derivatives
            )
            span = build_scf_solve(settings, solver=solver)
            if span is None:
                span = ChargeEquilibration(
                    build_long_range_features(
                        settings,
                        solver=solver,
                        trains_derivatives=trains_derivatives,
                    )
                )
            self.charge_solve = span

    def transform_outputs(self, graph, layers, output) -> MACEOutput[Tensor]:
        """Add QEq energy at the constrained solution; emit charges in e.

        Electronegativity includes ``+position·field`` (eV/e); this yields
        ``+field·dipole`` in the energy and changes the equilibrium charges.
        Site energies retain the short-range meaning of the shared backbone.
        """
        batch, count = graph["batch"], graph["ptr"].shape[0] - 1
        node_head = graph.get("node_head")
        if node_head is None and "head" in graph:
            node_head = graph["head"][batch]
        electronegativity = (
            self.source_maps["electronegativity"](layers, node_head)
            .reshape(-1)
            .to(self.solve_dtype)
        )
        if self.settings.electronegativity_input is not None:
            electronegativity = electronegativity + graph[
                self.settings.electronegativity_input
            ].reshape(-1).to(self.solve_dtype)
        positions = graph["positions"].to(self.solve_dtype)
        electronegativity = electronegativity + (
            positions * graph["external_field"].to(self.solve_dtype)[batch]
        ).sum(-1)
        hardness = (
            graph[self.settings.hardness_input].reshape(-1).to(self.solve_dtype)
            if self.settings.hardness_input is not None
            else torch.full_like(electronegativity, self.settings.default_hardness)
        )
        if self.settings.train_hardness:
            hardness = hardness + self.source_maps["hardness"](
                layers, node_head
            ).reshape(-1).to(self.solve_dtype)
        geometry = long_range_geometry(graph, self.descriptor)
        charges = self.charge_solve(
            graph,
            electronegativity,
            hardness,
            graph["total_charge"].to(self.solve_dtype),
            geometry,
        )
        electrostatic_energy = self.coulomb(graph, charges[:, None], geometry)
        local = charges * electronegativity + 0.5 * hardness * charges.square()
        qeq_energy = segment_sum(local, batch, count) + electrostatic_energy
        assert output.total_energy is not None
        output.total_energy = output.total_energy + qeq_energy
        output.dipole = segment_sum(positions * charges[:, None], batch, count)
        output.extras.update(
            {
                "charges": charges,
                "density_coefficients": charges[:, None],
                "total_charge": segment_sum(charges, batch, count),
                "electronegativity": electronegativity,
                "hardness": hardness,
                "qeq_energy": qeq_energy,
                "electrostatic_energy": electrostatic_energy,
            }
        )
        return output

    def canonical_components(self) -> dict[str, nn.Module | None]:
        return {**super().canonical_components(), "source_maps": self.source_maps}


class FixedPointSCFModel(_ChargeAwareModel):
    """FixedPointSCF with affine or nonlinear learned field responses.

    Architecture follows ACEsuit/mace-scf FixedPointCore (MIT), revision
    784a510339aacc4412b61b6926ef1a92bca387c0. Both solve the residual equation
    and preserve implicit first and second derivatives at its converged root.
    """

    projection: Any

    @staticmethod
    def declared_inputs(config: ModelConfig) -> tuple[InputSpec, ...]:
        settings = config.fixed_point_scf
        assert settings is not None
        constraint = (
            POLAR_INPUT_SPECS[0]
            if settings.constant_charge
            else InputSpec(name="fermi_level", irreps="0e", per_atom=False, units="eV")
        )
        return (constraint, POLAR_INPUT_SPECS[2])

    @staticmethod
    def solver_descriptor(
        config: ModelConfig, electrostatics: ElectrostaticsConfig, precision: Precision
    ) -> ElectrostaticsSolverDescriptor:
        settings = config.fixed_point_scf
        if settings is None:
            raise ValueError("model.fixed_point_scf is required")
        return ElectrostaticsSolverDescriptor(
            periodicity_profile=electrostatics.periodicity_profile,
            slab_normal=electrostatics.slab_normal,
            multipole_max_l=settings.multipole_max_l,
            smearing_width=settings.multipole_width,
            kspace_cutoff=float(
                electrostatics.kspace_cutoff_factor
                * gto_basis_kspace_cutoff(
                    [settings.multipole_width, *settings.feature_widths],
                    max(settings.multipole_max_l, settings.feature_max_l),
                )
            ),
            external_field_flags=frozenset({"include_self_interaction"})
            if settings.energy_self_interaction
            else frozenset(),
            features=FeatureProjection(
                settings.feature_max_l,
                settings.feature_widths,
                include_self_interaction=settings.feature_self_interaction,
                quadrupole_corrections=settings.quadrupole_corrections,
            ),
            scf_spec=ScfSpec(
                method=(
                    "affine_fixed_charge"
                    if settings.constant_charge
                    else "affine_fixed_fermi"
                )
                if settings.affine_response
                else (
                    "nonlinear_fixed_charge"
                    if settings.constant_charge
                    else "nonlinear_fixed_fermi"
                ),
                max_iters=settings.max_iters,
                tolerance=settings.tolerance,
            ),
            precision=precision,
        )

    def __init__(
        self,
        backend,
        config: ModelConfig,
        atomic_numbers: Sequence[int],
        observables: Sequence[ObservableSpec],
        energy_head: EnergyOutputHead | None,
        *,
        settings: ElectrostaticsSolverDescriptor,
        solver: str = "reference",
        trains_derivatives: bool = False,
        avg_num_neighbors: float = 1.0,
        precision: Precision = "float64",
        num_heads: int = 1,
        radial_precision: Precision | None = None,
    ):
        from mace_torch.backends import layout
        from mace_torch.nn.scf_nonlinear import (
            NonlinearElectronEnergy,
            NonlinearFieldUpdate,
        )
        from mace_torch.nn.scf_response import ResponseOps
        from mace_torch.physics.affine_fixed_point import AffineFixedPoint
        from mace_torch.physics.nonlinear_fixed_point import NonlinearFixedPoint

        options = config.fixed_point_scf
        if options is None or config.model != "fixed_point_scf":
            raise ValueError("FixedPointSCF requires model.fixed_point_scf")
        if num_heads != 1:
            raise ValueError(
                "FixedPointSCF supports one head, as the reference architecture"
            )
        if energy_head is None:
            raise ValueError("FixedPointSCF requires an energy observable")
        if max(options.multipole_max_l, options.feature_max_l) > config.max_L:
            raise ValueError("FixedPointSCF density/field degree exceeds model.max_L")
        if options.needs_edges and config.max_ell < 2:
            raise ValueError(
                "FixedPointSCF many-body updates/readouts require model.max_ell >= 2"
            )
        if options.update in ("scaled", "scaled_dot"):
            terms = tuple(Irreps.parse(config.node_irreps))
            if (
                tuple((ir.degree, ir.parity) for _, ir in terms) != ((0, 1), (1, -1))
                or terms[0][0] != terms[1][0]
            ):
                raise ValueError(
                    "FixedPointSCF scaled updates require equally many "
                    "0e and 1o node features"
                )
        if options.update_scales is not None and len(options.update_scales) != len(
            atomic_numbers
        ):
            raise ValueError(
                "fixed_point_scf.update_scales needs one value per atomic number"
            )
        self.density_irreps = _spherical(options.multipole_max_l)
        self.potential_irreps = _spherical(
            options.feature_max_l, len(options.feature_widths)
        )
        self.produced_observables = {
            name: ObservableSpec(
                name=name, irreps=irreps, per_atom=per_atom, units=units
            )
            for name, irreps, per_atom, units in (
                ("dipole", "1o", False, "e*Å"),
                ("charges", "0e", True, "e"),
                ("total_charge", "0e", False, "e"),
                ("density_coefficients", self.density_irreps, True, "e*Å^l"),
                (
                    "field_independent_charge_density",
                    self.density_irreps,
                    True,
                    "e*Å^l",
                ),
                ("fermi_level", "0e", False, "eV"),
                ("electrostatic_features", self.potential_irreps, True, "eV/e"),
                ("electrostatic_energy", "0e", False, "eV"),
                ("electron_energy", "0e", False, "eV"),
            )
        }
        super().__init__(
            backend,
            config,
            atomic_numbers,
            [
                spec
                for spec in observables
                if spec.name not in self.produced_observables
            ],
            energy_head,
            avg_num_neighbors=avg_num_neighbors,
            precision=precision,
            num_heads=num_heads,
            radial_precision=radial_precision,
            last_layer_irreps=config.node_irreps,
            edge_axes=(1, 2, 0),
        )
        self.settings, self.descriptor, self.solver = options, settings, solver
        self.solve_dtype, self.model_dtype = (
            getattr(torch, settings.precision),
            getattr(torch, precision),
        )
        self.layout, self.density_terms = (
            layout.layout_of(backend),
            layout.terms(self.density_irreps),
        )
        # The backbone exposes its expanded layer declaration in backend layout.
        node_irreps = self.backbone.layer_irreps[0]
        with constructed_in(precision):

            def linear(source, target):
                return backend.make_linear(
                    LinearDescriptor(
                        irreps_in=source, irreps_out=target, precision=precision
                    )
                )

            self.source_maps = nn.ModuleList(
                [
                    linear(node_irreps, self.density_irreps)
                    if options.local_source == "linear"
                    else BiasReadout(
                        backend,
                        node_irreps,
                        config.readout.mlp_irreps,
                        self.density_irreps,
                        precision,
                        has_bias=False,
                        gate=config.readout.gate,
                    )
                    for _ in range(config.num_interactions)
                ]
            )
            self.layer_mixer = nn.ModuleList(
                [
                    linear(node_irreps, node_irreps)
                    for _ in range(config.num_interactions)
                ]
            )
            self.updates = nn.ModuleList(
                [
                    AffineFieldUpdate(
                        backend,
                        node_irreps,
                        self.potential_irreps,
                        self.density_irreps,
                        len(atomic_numbers),
                        precision,
                        biased=options.potential_embedding == "biased_linear",
                    )
                    if options.affine_response
                    else NonlinearFieldUpdate(
                        ResponseOps(),
                        node_irreps,
                        self.potential_irreps,
                        self.density_irreps,
                        len(atomic_numbers),
                        config.radial_basis.num_basis,
                        options,
                        precision,
                    )
                ]
            )
            self.electron_energy = (
                FixedPointEnergy(
                    backend,
                    options.electron_energy,
                    node_irreps,
                    self.potential_irreps,
                    self.density_irreps,
                    precision,
                )
                if options.electron_energy in ("quadratic_field", "linear_charges")
                else (
                    NonlinearElectronEnergy(
                        backend,
                        options.electron_energy,
                        node_irreps,
                        self.potential_irreps,
                        self.density_irreps,
                        len(atomic_numbers),
                        config.radial_basis.num_basis,
                        precision,
                    )
                    if options.electron_energy != "none"
                    else None
                )
            )
        self.node_terms = layout.terms(node_irreps)
        self.extended_readout = options.electron_energy not in (
            "none",
            "quadratic_field",
            "linear_charges",
        )
        scales = options.density_scales or (1.0,) * len(atomic_numbers)
        if len(scales) != len(atomic_numbers):
            raise ValueError(
                "fixed_point_scf.density_scales needs one value per atomic number"
            )
        self.register_buffer(
            "density_scales",
            torch.tensor(scales, dtype=self.model_dtype),
            persistent=False,
        )
        norms = options.feature_norms or (1.0,) * (
            len(options.feature_widths) * (options.feature_max_l + 1)
        )
        expanded = [
            norms[degree * len(options.feature_widths) + width]
            for degree in range(options.feature_max_l + 1)
            for width in range(len(options.feature_widths))
            for _ in range(2 * degree + 1)
        ]
        self.register_buffer(
            "field_norms",
            torch.tensor(expanded, dtype=self.solve_dtype),
            persistent=False,
        )
        with constructed_in(settings.precision):
            self.projection = build_long_range_features(
                settings, solver=solver, trains_derivatives=trains_derivatives
            )
            self.coulomb = build_long_range(
                settings, solver=solver, trains_derivatives=trains_derivatives
            )
            self.applied_potential = DisplacedGTOExternalFieldBlock(
                options.feature_max_l, list(options.feature_widths), "receiver"
            )
        span = build_scf_solve(settings, solver=solver)
        self.fixed_point_solve = (
            span
            if span is not None
            else (
                AffineFixedPoint(options.constant_charge)
                if options.affine_response
                else NonlinearFixedPoint(
                    options.constant_charge, options.max_iters, options.tolerance
                )
            )
        )

    def transform_outputs(self, graph, layers, output):
        """Add the converged electrostatic and local electron energies, in eV."""
        count = graph["ptr"].shape[0] - 1
        batch = graph["batch"]
        positions = graph["positions"].to(self.solve_dtype)
        geometry = long_range_geometry(graph, self.descriptor)
        cache = self.projection.prepare(geometry)
        local = sum(
            self.layout.to_canonical(cast(Any, source)(layer), self.density_terms)
            for source, layer in zip(self.source_maps, layers, strict=True)
        )
        if self.settings.local_source == "linear":
            local = local * (graph["node_attrs"] @ self.density_scales)[:, None]
        mixed = sum(
            mixer(layer) for mixer, layer in zip(self.layer_mixer, layers, strict=True)
        )
        local = cast(Tensor, local).to(self.solve_dtype)
        membership = torch.nn.functional.one_hot(batch, num_classes=count).to(
            self.solve_dtype
        )
        fermi = (
            positions.new_zeros((count,))
            if self.settings.constant_charge
            else graph["fermi_level"].reshape(-1).to(self.solve_dtype)
        )
        applied = torch.cat(
            (
                (fermi - self.settings.fermi_level_offset)[:, None],
                graph["external_field"].to(self.solve_dtype),
            ),
            dim=-1,
        )
        external = self.applied_potential(batch, positions, applied)
        one_hot = graph["node_attrs"]
        edge_attributes, edge_radial = self._response_edges(graph)
        if self.settings.affine_response:
            density, fermi = self._solve_affine(
                graph, cache, local, mixed, membership, fermi, external
            )
        else:
            density, fermi = self._solve_nonlinear(
                graph,
                cache,
                local,
                mixed,
                membership,
                fermi,
                edge_attributes,
                edge_radial,
            )
        final_applied = torch.cat(
            (
                (fermi - self.settings.fermi_level_offset)[:, None],
                graph["external_field"].to(self.solve_dtype),
            ),
            dim=-1,
        )
        fields = (
            self.projection(cache, density)
            + self.applied_potential(batch, positions, final_applied)
        ) / self.field_norms
        electron = fermi.new_zeros(fermi.shape)
        if self.electron_energy is not None:
            arguments = (
                layers[-1],
                fields.to(self.model_dtype),
                local.to(self.model_dtype),
                density.to(self.model_dtype),
            )
            atom_energy = (
                self.electron_energy(
                    *arguments,
                    one_hot,
                    edge_attributes,
                    edge_radial,
                    graph["edge_index"],
                )
                if self.extended_readout
                else self.electron_energy(*arguments)
            )
            electron = segment_sum(atom_energy.to(self.solve_dtype), batch, count)
        coulomb = self.coulomb(graph, density, geometry)
        dipole = segment_sum(positions * density[:, :1], batch, count)
        if self.settings.multipole_max_l == 1:
            dipole = dipole + segment_sum(density[:, 1:4], batch, count)[:, [2, 0, 1]]
        output.total_energy = (
            output.total_energy
            + electron
            + coulomb
            + (graph["external_field"].to(self.solve_dtype) * dipole).sum(-1)
        )
        output.dipole = dipole
        output.extras.update(
            density_coefficients=density,
            charges=density[:, 0],
            total_charge=segment_sum(density[:, 0], batch, count),
            fermi_level=fermi,
            electrostatic_features=fields,
            electrostatic_energy=coulomb,
            electron_energy=electron,
            field_independent_charge_density=local,
        )
        return output

    def _response_edges(self, graph):
        positions = graph["positions"]
        if not self.settings.needs_edges:
            empty = positions.new_empty((graph["edge_index"].shape[1], 0))
            return empty, empty
        sender, receiver = graph["edge_index"]
        edges = EdgeInputs(
            (positions[receiver] - positions[sender] + graph["shifts"])[
                :, self.backbone.edge_axes
            ],
            graph["node_atomic_numbers"],
            graph["edge_index"],
            self.backbone.spherical_harmonics,
            self.backbone.radial_embedding,
        )
        return edges.attributes[:, :9], edges.radial

    def _solve_affine(self, graph, cache, local, mixed, membership, fermi, external):
        positions = graph["positions"].to(self.solve_dtype)
        batch, count = graph["batch"], graph["ptr"].shape[0] - 1
        one_hot = graph["node_attrs"]
        # Assemble the affine map from its linear response; keep the learned
        # offset separate so column construction never subtracts large biases.
        update = cast(AffineFieldUpdate, self.updates[0])
        independent = update(
            mixed, one_hot, (external / self.field_norms).to(self.model_dtype)
        ).to(self.solve_dtype)
        basis = torch.eye(
            local.numel(), dtype=self.solve_dtype, device=positions.device
        )
        response = torch.stack(
            [
                update.response(
                    mixed,
                    one_hot,
                    (
                        self.projection(cache, column.reshape_as(local))
                        / self.field_norms
                    ).to(self.model_dtype),
                )
                .to(self.solve_dtype)
                .reshape(-1)
                for column in basis.unbind()
            ],
            dim=1,
        )
        fermi_columns = []
        for column in torch.eye(
            count, dtype=self.solve_dtype, device=positions.device
        ).unbind():
            potential = torch.cat(
                (column[:, None], positions.new_zeros((count, 3))), dim=-1
            )
            fields = self.applied_potential(batch, positions, potential)
            fermi_columns.append(
                update.response(
                    mixed, one_hot, (fields / self.field_norms).to(self.model_dtype)
                )
                .to(self.solve_dtype)
                .reshape(-1)
            )
        fermi_response = torch.stack(fermi_columns, dim=1)
        total_charge = (
            graph["total_charge"].to(self.solve_dtype)
            if self.settings.constant_charge
            else fermi.new_zeros(fermi.shape)
        )
        density, fermi = self.fixed_point_solve(
            response,
            local + independent,
            fermi_response,
            membership,
            total_charge,
            fermi,
        )
        return density, fermi

    def _solve_nonlinear(
        self,
        graph,
        cache,
        local,
        mixed,
        membership,
        fermi,
        edge_attributes,
        edge_radial,
    ):
        from mace_torch.physics.nonlinear_fixed_point import ResponseInputs

        positions = graph["positions"].to(self.solve_dtype)
        batch, count = graph["batch"], graph["ptr"].shape[0] - 1
        basis = torch.eye(
            local.numel(), dtype=self.solve_dtype, device=positions.device
        )
        projection = torch.stack(
            [
                (
                    self.projection(cache, column.reshape_as(local)) / self.field_norms
                ).reshape(-1)
                for column in basis.unbind()
            ],
            dim=1,
        )
        fermi_columns = []
        for column in torch.eye(
            count, dtype=self.solve_dtype, device=positions.device
        ).unbind():
            potential = torch.cat(
                (column[:, None], positions.new_zeros((count, 3))), dim=-1
            )
            fermi_columns.append(
                (
                    self.applied_potential(batch, positions, potential)
                    / self.field_norms
                ).reshape(-1)
            )
        applied = torch.cat(
            (
                positions.new_full((count, 1), -self.settings.fermi_level_offset),
                graph["external_field"].to(self.solve_dtype),
            ),
            dim=-1,
        )
        external = self.applied_potential(batch, positions, applied) / self.field_norms
        inputs = ResponseInputs(
            local,
            projection,
            external,
            torch.stack(fermi_columns, dim=1),
            membership,
            graph["total_charge"].to(self.solve_dtype)
            if self.settings.constant_charge
            else fermi.new_zeros(fermi.shape),
            fermi,
            self.layout.to_canonical(mixed, self.node_terms).to(self.solve_dtype),
            graph["node_attrs"].to(self.solve_dtype),
            edge_attributes.to(self.solve_dtype),
            edge_radial.to(self.solve_dtype),
            graph["edge_index"],
        )
        update = self.updates[0]
        # The response is part of the SCF op: evaluate its live canonical tensors
        # at the configured solve precision, preserving gradients to model weights.
        parameters = {
            name: value.to(self.solve_dtype) if value.is_floating_point() else value
            for name, value in (*update.named_parameters(), *update.named_buffers())
        }
        return self.fixed_point_solve(update, parameters, inputs)

    def canonical_components(self):
        return {
            **super().canonical_components(),
            "source_maps": self.source_maps,
            "layer_mixer": self.layer_mixer,
            "updates": self.updates,
            "electron_energy": self.electron_energy,
        }
