"""Matrix-free Cartesian vibrations for MACE energy models."""

import numpy as np
import torch
from ase import Atoms
from scipy.sparse.linalg import LinearOperator

from mace.calculators.mace import MACECalculator
from mace.tools import torch_tools


def _model_inputs(batch, model, external_field):
    data = batch.clone().to_dict()
    dtype = next(model.parameters()).dtype
    for key, value in data.items():
        if torch.is_tensor(value) and value.is_floating_point():
            data[key] = value.to(dtype=dtype)
    if external_field is not None:
        data["external_field"] = data["positions"].new_tensor(external_field)
    return data


def hessian_operator(calculator: MACECalculator, atoms: Atoms) -> LinearOperator:
    """Return the unweighted Cartesian energy Hessian at a fixed geometry.

    The operator uses the native units of `calculator.get_hessian`. For a
    committee it represents the mean Hessian. Each product takes one reverse
    pass through the force graph, which stays on the calculator's device for
    the lifetime of the operator. Rebuild it after changing the geometry.

    For example, `scipy.sparse.linalg.eigsh(hessian_operator(calc, atoms),
    k=2, which="SA")` computes the two lowest curvatures without assembling
    the dense Hessian.
    """
    if calculator.model_type != "MACE" or calculator.use_compile:
        raise NotImplementedError(
            "hessian_operator requires an eager MACE energy model"
        )

    n_atoms = len(atoms)
    dtype = np.dtype(calculator.default_dtype)
    with torch.inference_mode(False), torch.enable_grad(), torch_tools.default_dtype(
        calculator.default_dtype
    ):
        # Use the same head, neighbour list and padding as get_hessian.
        batch = calculator._atoms_to_batch(atoms)  # pylint: disable=protected-access
        positions = batch["positions"].requires_grad_(True)
        member_forces = []
        for model in calculator.models:
            data = _model_inputs(
                batch, model, getattr(calculator, "external_field", None)
            )
            with torch_tools.default_dtype(data["positions"].dtype):
                out = model(data, training=True, compute_stress=False)
            member_forces.append(out["forces"][:n_atoms])
        forces = sum(member_forces) / calculator.num_models

    @torch.inference_mode(False)
    def matvec(vector):
        if np.iscomplexobj(vector):
            return matvec(vector.real) + 1j * matvec(vector.imag)
        direction = torch.as_tensor(
            np.ascontiguousarray(vector), dtype=positions.dtype, device=positions.device
        ).reshape(n_atoms, 3)
        gradient = None
        if forces.requires_grad:
            (gradient,) = torch.autograd.grad(
                forces, positions, direction, retain_graph=True, allow_unused=True
            )
        if gradient is None:  # A disconnected atom has no restoring force.
            return np.zeros(3 * n_atoms, dtype=dtype)
        return -gradient[:n_atoms].detach().cpu().numpy().reshape(-1)

    return LinearOperator(
        (3 * n_atoms, 3 * n_atoms), matvec=matvec, rmatvec=matvec, dtype=dtype
    )
