"""One training step on a committed anchor, reduced to a comparable digest.

A loss-decrease smoke test and a final-error table cannot see a gradient bug
the size of initialisation noise: both stay green while `d(loss)/d(theta)` is
wrong in a way that only slows convergence. This module produces the thing
that can see it -- one forward and one backward of an energy+forces loss on
the committed anchors, at float64 on CPU, with the committed weights, so the
result depends on nothing that was randomly drawn.

Energy **and** forces on purpose: the forces term differentiates a quantity
that is itself a derivative, so the backward runs through the
`create_graph=True` second-derivative path that force training uses and that
a first-order-only port would get away with otherwise.

**Why a digest and not the raw gradient vector.** The two anchors have 37,704
gradient elements between them; committing them as JSON costs about 1.5 MB in
every clone, more than both checkpoints together, and a reviewer cannot read
a diff of it. Instead each parameter contributes five numbers, chosen so that
no plausible corruption survives all of them:

* `sum` catches a sign flip anywhere (`abs_sum` and `sq_sum` do not);
* `abs_sum` and `sq_sum` catch a magnitude change that cancels in the sum;
* `projection` = sum_k g_k * cos(k + 1) over the C-order flattening catches a
  **permutation**, which the other four cannot see at all -- and a permuted
  gradient is exactly what an irreps-layout mistake produces, which is the
  single most likely porting bug in this codebase;
* `count` and `shape` catch a reshape.

The projection uses `cos` of an integer rather than a seeded random vector so
that a reimplementation in another framework can reproduce it without sharing
an RNG. It is evaluated in float64; the accumulated libm difference across
platforms is of order 1e-14 on these magnitudes, four orders under the
reference row.

Parameter **names** are part of the golden too: a rename shows up here rather
than as a silently smaller comparison.
"""

from __future__ import annotations

from typing import Dict

import numpy as np
import torch

from mace.modules.loss import WeightedEnergyForcesLoss
from mace.tools import torch_tools

from . import harness
from .anchors import (
    ANCHORS,
    anchor_batch,
    anchor_graph,
    load_anchor,
    load_training_structures,
)

#: anchor -> the committed reference for its gradient digest.
GRADIENT_REFERENCES = {
    "tiny_scaleshift": "tiny_scaleshift_train_step_grad_fp64.json",
    "tiny_mace": "tiny_mace_train_step_grad_fp64.json",
}

#: the step is defined by these three choices and nothing else.
SEED = 20260810
N_STRUCTURES = 4
LOSS_WEIGHTS = {"energy_weight": 1.0, "forces_weight": 10.0}

#: finite-difference step for the gradcheck pass-flags. Not a tolerance: the
#: tolerances come from harness.py, like everywhere else here.
GRADCHECK_EPS = 1e-6


def _digest(gradient: torch.Tensor) -> Dict[str, object]:
    flat = gradient.reshape(-1).to(torch.float64)
    index = torch.arange(flat.numel(), dtype=torch.float64)
    return {
        "shape": list(gradient.shape),
        "count": int(flat.numel()),
        "sum": float(flat.sum()),
        "abs_sum": float(flat.abs().sum()),
        "sq_sum": float((flat * flat).sum()),
        "projection": float((flat * torch.cos(index + 1.0)).sum()),
    }


def train_step(anchor: str, perturbation=None) -> dict:
    """Run the step and return the digest payload for ``anchor``.

    ``perturbation`` is ``(parameter_name, flat_index, delta)``, applied to
    the weights before the forward. It exists so a test can ask the only
    question that says whether this golden is worth committing: does a change
    too small to move any error table move the digest?
    """
    torch.manual_seed(SEED)
    model = load_anchor(anchor, torch.float64)
    if perturbation is not None:
        name, index, delta = perturbation
        with torch.no_grad():
            dict(model.named_parameters())[name].view(-1)[index] += delta
    structures = load_training_structures(limit=N_STRUCTURES)
    with torch_tools.default_dtype("float64"):
        batch = anchor_batch(model, structures, torch.float64)
        # A batch whose labels never arrived carries zeros at full weight, and
        # the gradient taken through it is wrong in a way nothing else here can
        # see: same magnitude as the right one, same linear response to a
        # perturbed weight, every parameter still differentiated. All four
        # structures carry `REF_energy` and `REF_forces`, and the label-less
        # isolated atoms are excluded upstream, so this cannot fire on a
        # legitimately unlabelled configuration.
        assert float(batch.energy.abs().sum()) > 0.0, "energy labels missing"
        assert float(batch.forces.abs().sum()) > 0.0, "force labels missing"
        loss_fn = WeightedEnergyForcesLoss(**LOSS_WEIGHTS).to(torch.float64)
        output = model(batch.to_dict(), training=True, compute_force=True)
        loss = loss_fn(batch, output)
        model.zero_grad(set_to_none=True)
        loss.backward()

    gradients = {
        name: parameter.grad
        for name, parameter in model.named_parameters()
        if parameter.grad is not None
    }
    without_gradient = sorted(
        name for name, p in model.named_parameters() if p.grad is None
    )
    total = float(
        torch.sqrt(sum((g.to(torch.float64) ** 2).sum() for g in gradients.values()))
    )
    return {
        "loss": float(loss),
        "global_gradient_norm": total,
        "n_elements": int(sum(g.numel() for g in gradients.values())),
        "parameters_without_a_gradient": without_gradient,
        "parameters": {name: _digest(g) for name, g in sorted(gradients.items())},
    }


def gradcheck_flags(anchor: str) -> Dict[str, bool]:
    """Coarse model-level gradcheck / gradgradcheck pass-flags.

    Run on the two-atom dimer (positions) and the triclinic cell (strain),
    which are the smallest fixtures that reach each derivative. `gradcheck`
    compares the analytic jacobian against a numerical one; `gradgradcheck`
    does the same one order up, which is the derivative force training
    actually backpropagates through. Both are asserted at the fp64 reference
    row rather than at torch's much looser defaults.
    """
    tol = harness.FP64_CPU_REFERENCE
    fixtures = harness.load_fixtures(names=["dimer_short", "triclinic_bulk"])
    model = load_anchor(anchor, torch.float64)

    with torch_tools.default_dtype("float64"):
        molecule = anchor_graph(model, fixtures["dimer_short"], torch.float64)
        crystal = anchor_graph(model, fixtures["triclinic_bulk"], torch.float64)

        def energy_of_positions(positions):
            graph = dict(molecule)
            graph["positions"] = positions
            return model(graph, training=True, compute_force=False)["energy"]

        def energy_of_strain(displacement):
            graph = dict(crystal)
            graph["displacement"] = displacement
            return model(
                graph, training=True, compute_force=False, compute_stress=True
            )["energy"]

        positions = molecule["positions"].detach().clone().requires_grad_(True)
        strain = torch.zeros((1, 3, 3), dtype=torch.float64, requires_grad=True)
        options = {
            "eps": GRADCHECK_EPS,
            "atol": tol.atol,
            "rtol": tol.rtol,
        }
        flags = {
            "gradcheck_positions": bool(
                torch.autograd.gradcheck(energy_of_positions, (positions,), **options)
            ),
            "gradgradcheck_positions": bool(
                torch.autograd.gradgradcheck(
                    energy_of_positions, (positions,), **options
                )
            ),
            "gradcheck_strain": bool(
                torch.autograd.gradcheck(energy_of_strain, (strain,), **options)
            ),
            "gradgradcheck_strain": bool(
                torch.autograd.gradgradcheck(energy_of_strain, (strain,), **options)
            ),
        }
    return flags


def snapshot(anchor: str) -> dict:
    """The whole committed payload for one anchor."""
    step = train_step(anchor)
    return {
        "schema_version": harness.SCHEMA_VERSION,
        "kind": "train_step_gradient",
        "dtype": "float64",
        "device": "cpu",
        "backend": "e3nn",
        "units": {"length": "Ang", "energy": "eV"},
        "metadata": {
            "model": ANCHORS[anchor]["model"].name,
            "model_class": ANCHORS[anchor]["class"],
            "seed": SEED,
            "loss": "WeightedEnergyForcesLoss",
            "loss_weights": LOSS_WEIGHTS,
            "structures": (
                f"first {N_STRUCTURES} non-isolated configurations of "
                "fixtures/tiny_train.xyz"
            ),
            "projection": "sum_k g_k * cos(k + 1), C-order flattening",
        },
        "gradcheck": gradcheck_flags(anchor),
        **step,
    }


def deviations(got: dict, reference: dict, tol) -> list:
    """Every way ``got`` and ``reference`` disagree, as readable lines."""
    problems = []
    if set(got["parameters"]) != set(reference["parameters"]):
        missing = sorted(set(reference["parameters"]) - set(got["parameters"]))
        extra = sorted(set(got["parameters"]) - set(reference["parameters"]))
        problems.append(f"parameter names changed: missing {missing}, extra {extra}")
        return problems
    for scalar in ("loss", "global_gradient_norm"):
        if not np.isclose(got[scalar], reference[scalar], atol=tol.atol, rtol=tol.rtol):
            problems.append(
                f"{scalar}: {got[scalar]!r} != {reference[scalar]!r} "
                f"(delta {got[scalar] - reference[scalar]:.3e})"
            )
    for name, want in reference["parameters"].items():
        have = got["parameters"][name]
        if list(have["shape"]) != list(want["shape"]):
            problems.append(f"{name}: shape {want['shape']} -> {have['shape']}")
            continue
        for field in ("sum", "abs_sum", "sq_sum", "projection"):
            if not np.isclose(
                have[field], want[field], atol=tol.atol, rtol=tol.rtol
            ):
                problems.append(
                    f"{name}.{field}: {have[field]!r} != {want[field]!r} "
                    f"(delta {have[field] - want[field]:.3e})"
                )
    return problems
