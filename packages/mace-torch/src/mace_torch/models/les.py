"""Latent multipoles and the external LES energy correction, in eV and Å."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from contextlib import nullcontext
from importlib import import_module
from typing import cast

import torch
from mace_core.clebsch_gordan import symmetric_matrix_basis
from mace_core.clebsch_gordan.irreps import Irreps
from mace_core.config.model import LESConfig, ModelConfig
from mace_core.kernels.descriptors import LinearDescriptor
from mace_core.kernels.precision import Precision
from mace_core.observables import ObservableSpec
from mace_core.outputs import MACEOutput
from torch import Tensor, nn
from torch.overrides import TorchFunctionMode

from mace_torch.backends.precision import constructed_in
from mace_torch.models.base import MACEModel
from mace_torch.models.energy import EnergyOutputHead
from mace_torch.nn.les import TensorReadout

FIELD_FORCES = ObservableSpec(
    name="field_forces", irreps="1o", per_atom=True, units="eV/Å"
)


class _CellVolume(TorchFunctionMode):
    """Use the signed triple product for the pinned solver's 3-by-3 determinant.

    PyTorch's det backward can promote its LU factors to the process default
    dtype and then fail in lu_solve. The triple product differentiates entirely
    in the cell dtype, including strain gradients used in stress training.
    """

    def __torch_function__(self, func, types, args: tuple = (), kwargs=None):
        if func is torch.det:
            a, b, c = args[0].unbind(-2)
            return (a * torch.linalg.cross(b, c)).sum(-1)
        return func(*args, **(kwargs or {}))


class LatentEwaldModel(MACEModel):
    """Local site energies plus LES graph energy from learned multipoles.

    The multipole conventions follow MACELES in mace/modules/extensions.py.
    ChengUCB/les supplies the Ewald solver and the polarization Jacobian.
    """

    matrix_basis: Tensor
    requestable_outputs = frozenset({"BEC"})

    def __init__(
        self,
        backend,
        config: ModelConfig,
        atomic_numbers: Sequence[int],
        observables: Sequence[ObservableSpec],
        energy_head: EnergyOutputHead | None,
        *,
        avg_num_neighbors: float = 1.0,
        precision: Precision = "float64",
        num_heads: int = 1,
        radial_precision: Precision | None = None,
    ) -> None:
        if energy_head is None:
            raise ValueError("LES requires an energy head")
        try:
            solver_class = import_module("les").Les
        except ImportError as error:
            raise ImportError(
                "model.model='les' requires the mace-torch-v1[les] extra"
            ) from error

        settings = config.les or LESConfig()
        self.settings = settings
        self.requires_position_grad = settings.compute_bec or any(
            spec.name == "BEC" for spec in observables
        )
        alphas = set(str(ir) for _, ir in Irreps.parse(settings.alpha_irreps))
        anisotropic = (
            settings.use_induced_dipole and settings.use_anisotropic_polarizability
        )
        scalar_names = ["latent_charges"]
        if settings.use_induced_charge:
            scalar_names.append("latent_kappas")
        if settings.use_induced_dipole and (not anisotropic or "0e" in alphas):
            scalar_names.append("latent_alphas")
        channels = {str(ir) for _, ir in Irreps.parse(config.node_irreps)}
        has_dipoles = settings.use_dipole or settings.use_induced_dipole
        self.produced_observables = {}
        for name, irreps, per_atom, units, active in (
            ("les_energy", "0e", False, "eV", True),
            ("latent_charges", "0e", True, "e", True),
            ("latent_dipoles", "1o", True, "e*Å", has_dipoles),
            ("latent_kappas", "0e", True, "arb", settings.use_induced_charge),
            (
                "latent_quads",
                "2e",
                True,
                "e*Å^2",
                settings.use_quad and bool(channels & {"1o", "2e"}),
            ),
            (
                "latent_alphas",
                "0e+2e" if anisotropic else "0e",
                True,
                "arb",
                settings.use_induced_dipole,
            ),
            # Each full polarization Jacobian is a general Cartesian tensor.
            # bec_output_index selects one row after computing that tensor.
            (
                "BEC",
                "0e+1e+2e" if not has_dipoles else "2x0e+2x1e+2x2e",
                True,
                "e",
                True,
            ),
        ):
            if active:
                self.produced_observables[name] = ObservableSpec(
                    name=name, irreps=irreps, per_atom=per_atom, units=units
                )
        declarations = [
            ObservableSpec(
                name=name,
                irreps="0e",
                per_atom=True,
                units=self.produced_observables[name].units,
            )
            for name in scalar_names
        ]
        super().__init__(
            backend,
            config,
            atomic_numbers,
            [
                *(
                    spec
                    for spec in observables
                    if spec.name not in self.produced_observables
                ),
                *declarations,
            ],
            energy_head,
            last_layer_irreps=config.node_irreps,
            avg_num_neighbors=avg_num_neighbors,
            precision=precision,
            num_heads=num_heads,
            radial_precision=radial_precision,
        )
        with constructed_in(precision):
            self.solver = solver_class(settings.model_dump(mode="json"))
        layer_irreps = (
            self.backbone.layer_irreps[-1:]
            if config.readout.last_only
            else self.backbone.layer_irreps
        )
        readouts = {}
        for name, condition, output in (
            ("dipole", settings.use_dipole, "1o"),
            ("quad_sh", settings.use_quad and "2e" in channels, "2e"),
            ("alpha_sh", anisotropic and "2e" in channels and "2e" in alphas, "0e+2e"),
            ("quad_tensor", settings.use_quad and "1o" in channels, None),
            ("alpha_tensor", anisotropic and "1o" in channels and "1o" in alphas, None),
        ):
            if condition:
                with constructed_in(precision):
                    readouts[name] = nn.ModuleList(
                        backend.make_linear(
                            LinearDescriptor(
                                irreps_in=layer, irreps_out=output, precision=precision
                            )
                        )
                        if output
                        else TensorReadout(
                            backend,
                            layer,
                            precision,
                            nonlinear=settings.alpha_1o_nonlinear_readout,
                            positive=name == "alpha_tensor"
                            and settings.alpha_1o_linear_w_pos,
                        )
                        for layer in layer_irreps
                    )
        self.multipoles = nn.ModuleDict(readouts)
        self.register_buffer(
            "matrix_basis",
            torch.tensor(symmetric_matrix_basis(), dtype=getattr(torch, precision)),
            persistent=False,
        )
        if anisotropic and not (
            "latent_alphas" in scalar_names
            or {"alpha_sh", "alpha_tensor"} & readouts.keys()
        ):
            raise ValueError(
                "LES anisotropic polarizability needs a reachable alpha_irreps channel"
            )

    @property
    def extra_observables(self) -> tuple[ObservableSpec, ...]:
        return super().extra_observables + tuple(self.produced_observables.values())

    def transform_outputs(self, graph, layers, output: MACEOutput[Tensor]):
        settings = self.settings
        extras = output.extras
        charge = extras["latent_charges"].reshape(-1) * settings.output_scale
        kappa = extras.get("latent_kappas")
        if kappa is not None:
            kappa = kappa.reshape(-1) * settings.kappa_scale
            if settings.make_kappa_positive:
                kappa = kappa.square()
            extras["latent_kappas"] = kappa
        alpha = extras.get("latent_alphas")
        if alpha is not None:
            alpha = alpha.reshape(-1)
            if settings.use_anisotropic_polarizability:
                alpha = alpha[:, None, None] * torch.eye(
                    3, device=charge.device, dtype=charge.dtype
                )
        tensors = {}
        for name, module in self.multipoles.items():
            readouts = cast(nn.ModuleList, module)
            parts = [
                op(features)
                for op, features in zip(readouts, layers[-len(readouts) :], strict=True)
            ]
            tensors[name] = torch.stack(parts).sum(0)
        dipole = tensors.get("dipole")
        if dipole is not None:
            dipole = dipole * settings.output_scale
        quad = tensors.get("quad_tensor")
        if "quad_sh" in tensors:
            matrix = torch.einsum(
                "nm,mij->nij", tensors["quad_sh"], self.matrix_basis[1:]
            )
            quad = matrix if quad is None else quad + matrix
        if quad is not None:
            quad = quad * settings.output_scale
            trace = quad.diagonal(dim1=1, dim2=2).sum(-1) / 3
            quad = quad - torch.diag_embed(trace[:, None].expand(-1, 3))
            extras["latent_quads"] = quad
        if "alpha_sh" in tensors:
            matrix = torch.einsum("nm,mij->nij", tensors["alpha_sh"], self.matrix_basis)
            alpha = matrix if alpha is None else alpha + matrix
        if "alpha_tensor" in tensors:
            alpha = (
                tensors["alpha_tensor"]
                if alpha is None
                else alpha + tensors["alpha_tensor"]
            )
        if alpha is not None:
            alpha = alpha * settings.alpha_scale
            if settings.make_alpha_positive:
                alpha = (
                    alpha.square() if alpha.ndim == 1 else alpha @ alpha.transpose(1, 2)
                )
        values = self._solve(graph, charge, dipole, quad, kappa, alpha)
        extras.update(values)
        assert output.total_energy is not None
        output.total_energy = output.total_energy + extras["les_energy"]
        return output

    def prepare_model_inputs(self, graph, requested):
        """Record real structure boundaries before entering the external solver."""
        counts = graph.get("structure_sizes")
        if counts is None:
            counts = graph["ptr"].diff().tolist()
        graph["les_structures"] = (
            counts,
            graph.get("structure_padding", (False,) * len(counts)),
        )
        graph["les_bec"] = self.requires_position_grad or "BEC" in requested

    def _solve(self, graph: Mapping, charge, dipole, quad, kappa, alpha):
        # The external solver accepts a single field shared by its batch.
        # Solve each real structure with its own field, leaving padding out of Ewald.
        positions = graph["positions"]
        cells = torch.where(
            graph["pbc"].reshape(-1, 3).any(-1)[:, None, None],
            graph["cell"].reshape(-1, 3, 3),
            0,
        )
        field = graph.get("external_field")
        counts, padding = graph["les_structures"]
        energies, charges, dipoles, alphas = [], [], [], []
        start = 0
        for index, count in enumerate(counts):
            atoms = slice(start, start + count)
            start += count
            if padding[index] or not count:
                energies.append(charge.new_zeros(1))
                if count:
                    # Padding is appended after real structures. The solver
                    # sets the response rank according to the enabled sources.
                    charges.append(charges[0].new_zeros((count, *charges[0].shape[1:])))
                    if dipoles:
                        dipoles.append(
                            dipoles[0].new_zeros((count, *dipoles[0].shape[1:]))
                        )
                    if alphas:
                        alphas.append(
                            alphas[0].new_zeros((count, *alphas[0].shape[1:]))
                        )
                continue
            # Only the external solve with differentiable, lower-precision
            # cells needs this workaround. TorchFunctionMode is thread-local.
            mixed_det = (
                cells.requires_grad
                and cells.dtype == torch.float32
                and torch.get_default_dtype() == torch.float64
            )
            with _CellVolume() if mixed_det else nullcontext():
                result = self.solver(
                    positions=positions[atoms],
                    cell=cells[index : index + 1],
                    atomic_numbers=graph["node_atomic_numbers"][atoms],
                    latent_charges=charge[atoms],
                    latent_dipoles=None if dipole is None else dipole[atoms],
                    latent_quads=None if quad is None else quad[atoms],
                    latent_kappas=None if kappa is None else kappa[atoms],
                    latent_alphas=None if alpha is None else alpha[atoms],
                    e_ext=None if field is None else field.reshape(-1, 3)[index],
                )
            energies.append(result["E_lr"])
            charges.append(result["latent_charges"])
            if result["latent_dipoles"] is not None:
                dipoles.append(result["latent_dipoles"])
            if result["latent_alphas"] is not None:
                alphas.append(result["latent_alphas"])
        q = torch.cat(charges)
        u = torch.cat(dipoles) if dipoles else None
        values = {"les_energy": torch.cat(energies), "latent_charges": q}
        if u is not None:
            values["latent_dipoles"] = u
        if alphas:
            values["latent_alphas"] = torch.cat(alphas)
        if graph["les_bec"]:
            # Differentiate against exactly the positions used by the backbone;
            # a per-structure view would lose the derivative through its charges.
            values["BEC"] = self.solver.bec(
                q=q,
                u=u,
                r=positions,
                cell=cells,
                batch=graph["batch"],
                output_index=self.settings.bec_output_index,
            )
        return values

    def canonical_components(self):
        return {**super().canonical_components(), "multipoles": self.multipoles}


def field_forces(
    bec: Tensor,
    alphas: Tensor | None,
    field: Tensor,
    *,
    volume: float,
    eps_infty: float | None,
    keep_neutral: bool,
    electric_field_unit: float,
) -> Tensor:
    """LES Born-charge force [N,3], eV/Å, for one applied field [V/Å].

    Adapted from the electric-field correction in mace/calculators/mace.py.
    Polarization is the first BEC axis and position is the second.
    """
    if bec.ndim == 4:
        bec = bec.sum(1)
    if bec.ndim != 3 or bec.shape[1:] != (3, 3):
        raise ValueError(
            "field forces require the full BEC; leave bec_output_index unset"
        )
    if keep_neutral:
        bec = bec - bec.mean(0)
    epsilon = 1.0 if eps_infty is None else eps_infty
    if eps_infty is not None and alphas is not None:
        if volume <= 0:
            raise ValueError("LES dielectric screening requires a positive cell volume")
        # Isotropic alpha or a flattened/full Cartesian polarizability.
        trace = alphas.reshape(len(bec), -1)
        if trace.shape[1] == 9:
            mean_alpha = trace.reshape(-1, 3, 3).diagonal(dim1=1, dim2=2).sum(-1) / 3
        elif trace.shape[1] == 1:
            mean_alpha = trace[:, 0]
        else:
            raise ValueError(f"unknown LES polarizability shape {tuple(alphas.shape)}")
        epsilon = eps_infty / (1 + mean_alpha.sum() / (volume * 5.52635e-3))
    return (
        torch.einsum("nij,i->nj", bec, field.to(bec))
        * (epsilon**0.5)
        * electric_field_unit
    )
