"""A dipole model, with or without a polarizability: no energy, only response.

The frozen tree has two classes for this, one reading out a dipole and one a
dipole with a polarizability, and they differ in more than which readouts they
build. Here they are one model with two settings, and the readouts are
observable heads like any other; what is left in the model is the physics that
turns what the heads read into the dipole a structure has.

**The dipole is a sum of two parts.** Each atom carries a dipole of its own,
read off its vector features, and a charge sitting at its position adds
``q * r``. The two classes take that charge from different places:

* ``charges="fixed"`` reads it from the structure, as a per-atom input, and
  converts the charge dipole from e Angstrom to Debye before adding it. The
  factor is the frozen tree's ``1e-11 / c / e`` with the SI values of ``c``
  and ``e``, which differs from ase's Debye in the eighth digit; it is kept,
  because this is the number every trained model of this kind was fitted
  against.
* ``charges="predicted"`` reads a charge per atom off the scalars, shifts every
  atom of a structure by the same amount so they sum to its total charge, and
  adds the charge dipole with no conversion.

**The polarizability is read out spherically and reported as a matrix.** Its
head reads ``0e+2e``, six components per structure, and the symmetric matrix
is built from them with :func:`~mace_core.clebsch_gordan.symmetric_matrix_basis`.
Both are reported: ``polarizability`` as ``[n_graphs, 3, 3]`` and
``polarizability_sh`` as ``[n_graphs, 6]``.

The derivatives of the dipole and the polarizability with respect to the
positions, which infrared and Raman intensities are computed from, are not
taken here. They are declared derivatives, ``dmu_dr`` and ``dalpha_dr``, and
the derivative engine takes them like any other.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any, Literal

import torch
from mace_core.clebsch_gordan import symmetric_matrix_basis
from mace_core.config.model import ModelConfig
from mace_core.kernels.precision import Precision
from mace_core.observables import DEFAULT_CATALOGUE, InputSpec, ObservableSpec
from mace_core.outputs import MACEOutput
from torch import Tensor

from mace_torch.kernels import segment_sum
from mace_torch.models.base import MACEModel
from mace_torch.models.inputs import CHARGES, TOTAL_CHARGE
from mace_torch.models.outputs import E_ANGSTROM_PER_DEBYE as E_ANGSTROM_PER_DEBYE
from mace_torch.models.outputs import fixed_charge_dipole

__all__ = [
    "DIPOLE_HEADS",
    "DipoleModel",
    "DipoleSettings",
]

#: The readouts the model builds for itself. They are observables to the output
#: layer, which builds and reduces their heads, and implementation to everything
#: else: the model turns them into the dipole and the polarizability.
DIPOLE_HEADS: dict[str, ObservableSpec] = {
    "atomic_dipoles": ObservableSpec(
        name="atomic_dipoles", irreps="1o", per_atom=True, units="e*Å"
    ),
    "charges": ObservableSpec(name="charges", irreps="0e", per_atom=True, units="e"),
    "polarizability_sh": ObservableSpec(
        name="polarizability_sh", irreps="0e+2e", per_atom=False, units="Å^3"
    ),
}


@dataclass(frozen=True)
class DipoleSettings:
    """Which of the two response models this is.

    The two combinations the frozen tree builds are the two there are: fixed
    charges and a dipole alone, or predicted charges with a dipole and a
    polarizability. Predicted charges without a polarizability is not a model
    it has, and it is refused rather than invented.

    Attributes:
        charges: Where the charge each atom carries comes from: ``"fixed"``
            from the structure, ``"predicted"`` from the model.
        polarizability: Whether a polarizability is read out as well.
    """

    charges: Literal["fixed", "predicted"] = "fixed"
    polarizability: bool = False

    def __post_init__(self) -> None:
        if self.charges not in ("fixed", "predicted"):
            raise ValueError(
                f"charges is {self.charges!r}; it is 'fixed', read from the "
                f"structure, or 'predicted', read off the model."
            )
        if self.polarizability != (self.charges == "predicted"):
            raise ValueError(
                f"charges={self.charges!r} with polarizability="
                f"{self.polarizability} is not a model that exists: fixed "
                f"charges go with a dipole alone, and predicted charges with a "
                f"polarizability."
            )

    @property
    def extra_observables(self) -> tuple[ObservableSpec, ...]:
        """Internal readouts and the physical responses computed from them."""
        names = ("dipole", "polarizability") if self.polarizability else ("dipole",)
        return self.heads + tuple(DEFAULT_CATALOGUE.observable(name) for name in names)

    @property
    def heads(self) -> tuple[ObservableSpec, ...]:
        """The readouts this model builds, in the order it builds them."""
        names = ["atomic_dipoles"]
        if self.charges == "predicted":
            names.insert(0, "charges")
        if self.polarizability:
            names.append("polarizability_sh")
        return tuple(DIPOLE_HEADS[name] for name in names)


class DipoleModel(MACEModel):
    """The backbone, per-atom response readouts, and the dipole they add up to.

    Args:
        backend: The kernel backend. Consulted at construction only.
        atomic_numbers: The element table, ascending.
        settings: Where the charges come from, and whether a polarizability is
            read out.
        last_layer_irreps: What the last layer keeps. The fixed-charge model
            reads only vectors off it, and keeping only ``1o`` there is the
            frozen tree's choice for that model; the other keeps every irrep.
        The rest are :class:`MACEModel`'s. There is no energy, no energy head
        and no repulsion.
    """

    polarizability_basis: Tensor

    @staticmethod
    def declared_inputs(config: ModelConfig) -> tuple[InputSpec, ...]:
        return (TOTAL_CHARGE,) if config.model == "dielectric" else (CHARGES,)

    def __init__(
        self,
        backend,
        config: ModelConfig,
        atomic_numbers: Sequence[int],
        settings: DipoleSettings,
        *,
        avg_num_neighbors: float = 1.0,
        radial_precision: Precision | None = None,
        precision: Precision = "float64",
        last_layer_irreps: str | None = None,
    ) -> None:
        if config.pair_repulsion:
            raise ValueError(
                "pair_repulsion requires an energy observable; a dipole model has none"
            )
        if last_layer_irreps is None:
            last_layer_irreps = (
                "1o" if settings.charges == "fixed" else config.node_irreps
            )
        super().__init__(
            backend,
            config,
            atomic_numbers=atomic_numbers,
            observables=settings.heads,
            avg_num_neighbors=avg_num_neighbors,
            radial_precision=radial_precision,
            precision=precision,
            last_layer_irreps=last_layer_irreps,
        )
        self.settings = settings
        self.register_buffer(
            "polarizability_basis",
            torch.tensor(symmetric_matrix_basis(), dtype=getattr(torch, precision)),
            persistent=False,
        )

    @property
    def extra_observables(self) -> tuple[ObservableSpec, ...]:
        return self.settings.extra_observables

    def forward(self, graph: Mapping[str, Any]) -> MACEOutput[Tensor]:
        """The dipole, and the polarizability when there is one."""
        output = super().forward(graph)
        positions = graph["positions"]
        batch = graph["batch"]
        num_graphs = int(graph["ptr"].shape[0] - 1)

        if self.settings.charges == "predicted":
            # The head reads one scalar per atom; a charge is that scalar.
            charges = output.extras["charges"].squeeze(-1)
            counts = segment_sum(torch.ones_like(charges), batch, num_graphs)
            # Every atom is shifted by the same amount, so the charges of each
            # structure add up to its total: the mean minus the total per atom.
            excess = segment_sum(charges, batch, num_graphs) / counts - (
                graph["total_charge"] / counts
            )
            charges = charges - excess[batch]
            output.extras["charges"] = charges
            baseline = segment_sum(positions * charges.unsqueeze(-1), batch, num_graphs)
        else:
            baseline = fixed_charge_dipole(
                positions, graph.get("charges"), batch, num_graphs
            )

        atomic_dipoles = output.extras["atomic_dipoles"]
        output.dipole = segment_sum(atomic_dipoles, batch, num_graphs) + baseline

        if self.settings.polarizability:
            spherical = output.extras["polarizability_sh"]
            output.extras["polarizability"] = torch.einsum(
                "mij,gm->gij", self.polarizability_basis, spherical
            )
        return output
