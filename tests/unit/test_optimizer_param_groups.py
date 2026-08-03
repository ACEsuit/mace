"""Check that every trainable model parameter is registered in the optimizer.

The training entry point (mace/cli/run_train.py) builds the optimizer from
explicit named parameter groups in mace.tools.scripts_utils.get_params_options.
A submodule missing from those groups would silently receive no gradient
updates, so get_params_options raises for unclaimed trainable parameters and
these tests assert that every trainable model class passes that check.

Polar and Magnetic MACE are tested in the extensions tests.
"""

from __future__ import annotations

import argparse
import json

import numpy as np
import pytest
import torch

from mace import modules
from mace.modules import interaction_classes
from mace.tools.scripts_utils import get_optimizer, get_params_options
from e3nn import o3  # isort: skip  (mace import must come first for torch.load)


COMMON_MODEL_KWARGS = dict(
    r_max=4.0,
    num_bessel=4,
    num_polynomial_cutoff=3,
    max_ell=1,
    interaction_cls=interaction_classes["RealAgnosticResidualInteractionBlock"],
    interaction_cls_first=interaction_classes["RealAgnosticResidualInteractionBlock"],
    num_interactions=2,
    num_elements=2,
    hidden_irreps=o3.Irreps("4x0e + 4x1o"),
    MLP_irreps=o3.Irreps("8x0e"),
    atomic_energies=np.zeros(2),
    avg_num_neighbors=3.0,
    atomic_numbers=[1, 8],
    correlation=1,
    gate=torch.nn.functional.silu,
    radial_MLP=[16, 16],
)


def build_mace() -> torch.nn.Module:
    return modules.MACE(
        **COMMON_MODEL_KWARGS,
        heads=["Default"],
        pair_repulsion=True,
        use_embedding_readout=True,
    )


def build_scale_shift_mace() -> torch.nn.Module:
    return modules.ScaleShiftMACE(
        **COMMON_MODEL_KWARGS,
        heads=["Default"],
        atomic_inter_scale=1.0,
        atomic_inter_shift=0.0,
    )


def build_atomic_dipoles_mace() -> torch.nn.Module:
    kwargs = {**COMMON_MODEL_KWARGS, "atomic_energies": None}
    return modules.AtomicDipolesMACE(**kwargs)


def build_atomic_dielectric_mace() -> torch.nn.Module:
    kwargs = {**COMMON_MODEL_KWARGS, "atomic_energies": None}
    return modules.AtomicDielectricMACE(**kwargs)


def build_energy_dipoles_mace() -> torch.nn.Module:
    return modules.EnergyDipolesMACE(**COMMON_MODEL_KWARGS)


MODEL_BUILDERS = {
    "MACE": build_mace,
    "ScaleShiftMACE": build_scale_shift_mace,
    "AtomicDipolesMACE": build_atomic_dipoles_mace,
    "AtomicDielectricMACE": build_atomic_dielectric_mace,
    "EnergyDipolesMACE": build_energy_dipoles_mace,
}


def _training_args() -> argparse.Namespace:
    """Optimizer-related arguments with the run_train defaults."""
    return argparse.Namespace(
        lr=0.01,
        weight_decay=5e-7,
        amsgrad=True,
        beta=0.9,
        freeze=None,
        optimizer="adam",
        lr_params_factors=json.dumps({}),
    )


@pytest.mark.parametrize(
    "model_builder", MODEL_BUILDERS.values(), ids=MODEL_BUILDERS.keys()
)
def test_all_trainable_parameters_registered_in_optimizer(model_builder) -> None:
    model = model_builder()
    args = _training_args()
    param_options = get_params_options(args, model)
    optimizer = get_optimizer(args, param_options)

    optimizer_parameter_ids = {
        id(parameter)
        for group in optimizer.param_groups
        for parameter in group["params"]
    }
    missing_parameter_names = [
        name
        for name, parameter in model.named_parameters()
        if parameter.requires_grad and id(parameter) not in optimizer_parameter_ids
    ]
    assert not missing_parameter_names, (
        f"{len(missing_parameter_names)} trainable parameters are not registered "
        f"in any optimizer parameter group and would never be updated during "
        f"training:\n" + "\n".join(missing_parameter_names)
    )


def test_trainable_bessel_weights_get_their_own_group() -> None:
    """Foundation-model checkpoints carry trainable Bessel weights, which must
    be claimed by the radial_embedding group instead of raising."""
    model = build_scale_shift_mace()
    bessel_module = model.radial_embedding.bessel_fn
    bessel_weights = bessel_module.bessel_weights
    del bessel_module.bessel_weights
    bessel_module.bessel_weights = torch.nn.Parameter(bessel_weights)

    param_options = get_params_options(_training_args(), model)

    radial_embedding_groups = [
        group
        for group in param_options["params"]
        if group["name"] == "radial_embedding"
    ]
    assert len(radial_embedding_groups) == 1
    registered_parameter_ids = {
        id(parameter) for parameter in radial_embedding_groups[0]["params"]
    }
    assert id(bessel_module.bessel_weights) in registered_parameter_ids


def test_unregistered_submodule_raises() -> None:
    model = build_scale_shift_mace()
    model.unregistered_test_module = torch.nn.Linear(4, 4)

    with pytest.raises(ValueError, match="unregistered_test_module"):
        get_params_options(_training_args(), model)
