from __future__ import annotations

import math
from typing import Callable, List, Tuple

import numpy as np
import pytest
import torch
from e3nn import o3

from mace import data, modules, tools
from mace.data import Configuration
from mace.tools import torch_geometric
from mace.tools.lora_tools import inject_lora


def _random_config() -> Configuration:
    atomic_numbers = np.array([6, 1, 1], dtype=int)
    positions = np.random.normal(scale=0.5, size=(3, 3))
    properties = {
        "energy": np.random.normal(scale=0.1),
        "forces": np.random.normal(scale=0.1, size=(3, 3)),
    }
    prop_weights = {"energy": 1.0, "forces": 1.0}
    return Configuration(
        atomic_numbers=atomic_numbers,
        positions=positions,
        properties=properties,
        property_weights=prop_weights,
        cell=np.eye(3) * 8.0,
        pbc=(True, True, True),
    )


def _build_model() -> Tuple[modules.MACE, tools.AtomicNumberTable]:
    table = tools.AtomicNumberTable([1, 6])
    model = modules.MACE(
        r_max=4.5,
        num_bessel=4,
        num_polynomial_cutoff=3,
        max_ell=1,
        interaction_cls=modules.interaction_classes[
            "RealAgnosticResidualInteractionBlock"
        ],
        interaction_cls_first=modules.interaction_classes[
            "RealAgnosticResidualInteractionBlock"
        ],
        num_interactions=2,
        num_elements=2,
        hidden_irreps=o3.Irreps("16x0e + 16x1o"),
        MLP_irreps=o3.Irreps("16x0e"),
        gate=torch.nn.functional.silu,
        atomic_energies=np.random.normal(scale=0.1, size=len(table.zs)),
        avg_num_neighbors=6.0,
        atomic_numbers=table.zs,
        correlation=2,
        radial_type="bessel",
    )
    return model, table


def _atomic_data_from_config(
    config: Configuration,
    table: tools.AtomicNumberTable,
    cutoff: float = 4.5,
) -> data.AtomicData:
    return data.AtomicData.from_config(config, z_table=table, cutoff=cutoff)


def _forward_energy_forces(
    model: torch.nn.Module,
    configs: List[Configuration],
    table: tools.AtomicNumberTable,
) -> Tuple[torch.Tensor, torch.Tensor]:
    dataset = [_atomic_data_from_config(cfg, table) for cfg in configs]
    loader = torch_geometric.dataloader.DataLoader(
        dataset=dataset,
        batch_size=len(dataset),
        shuffle=False,
        drop_last=False,
    )
    batch = next(iter(loader))
    outputs = model(batch.to_dict())
    energies = outputs["energy"].detach()
    forces = outputs["forces"].detach()
    return energies, forces


def _randomize_lora_parameters(model: torch.nn.Module) -> None:
    with torch.no_grad():
        for name, param in model.named_parameters():
            if "lora_A" in name or "lora_B" in name:
                torch.nn.init.normal_(param, mean=0.0, std=0.05)


def _rotation_matrix() -> np.ndarray:
    axis = np.random.normal(size=3)
    axis /= np.linalg.norm(axis)
    theta = np.random.uniform(0, 2 * math.pi)
    K = np.array(
        [
            [0.0, -axis[2], axis[1]],
            [axis[2], 0.0, -axis[0]],
            [-axis[1], axis[0], 0.0],
        ]
    )
    R = np.eye(3) + math.sin(theta) * K + (1.0 - math.cos(theta)) * (K @ K)
    return R


def _rotate_config(config: Configuration, R: np.ndarray) -> Configuration:
    return Configuration(
        atomic_numbers=config.atomic_numbers.copy(),
        positions=config.positions @ R.T,
        properties=config.properties.copy(),
        property_weights=config.property_weights.copy(),
        cell=config.cell @ R.T if config.cell is not None else None,
        pbc=config.pbc,
        weight=config.weight,
        config_type=config.config_type,
        head=config.head,
    )


def _translate_config(config: Configuration, shift: np.ndarray) -> Configuration:
    return Configuration(
        atomic_numbers=config.atomic_numbers.copy(),
        positions=config.positions + shift.reshape(1, 3),
        properties=config.properties.copy(),
        property_weights=config.property_weights.copy(),
        cell=config.cell,
        pbc=config.pbc,
        weight=config.weight,
        config_type=config.config_type,
        head=config.head,
    )


def _reflect_config(
    config: Configuration, normal: np.ndarray
) -> Tuple[Configuration, np.ndarray]:
    normal = normal / np.linalg.norm(normal)
    R = np.eye(3) - 2.0 * np.outer(normal, normal)
    reflected = _rotate_config(config, R)
    return reflected, R


@pytest.fixture(name="random_configs")
def _random_configs() -> Tuple[Configuration, Configuration]:
    return _random_config(), _random_config()


@pytest.fixture(name="build_lora_model")
def _build_lora_model_fixture() -> (
    Callable[[int, float, bool], Tuple[modules.MACE, tools.AtomicNumberTable]]
):
    def _builder(
        rank: int = 2,
        alpha: float = 0.5,
        randomize: bool = True,
    ) -> Tuple[modules.MACE, tools.AtomicNumberTable]:
        model, table = _build_model()
        inject_lora(model, rank=rank, alpha=alpha)
        if randomize:
            _randomize_lora_parameters(model)
        return model, table

    return _builder


def test_lora_trainable_parameter_count(build_lora_model) -> None:
    model, _ = build_lora_model(rank=2, alpha=0.5, randomize=True)
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    expected = sum(p.numel() for name, p in model.named_parameters() if "lora_" in name)
    assert trainable == expected

    non_lora_trainable = [
        name
        for name, p in model.named_parameters()
        if "lora_" not in name and p.requires_grad
    ]
    assert (
        not non_lora_trainable
    ), f"Non-LoRA parameters trainable: {non_lora_trainable}"

    # Ensure LoRA parameters were randomized away from zero
    for name, param in model.named_parameters():
        if "lora_B" in name:
            assert torch.any(
                torch.abs(param) > 0
            ), f"LoRA parameter {name} incorrectly zero"


def test_lora_symmetry_equivariance(build_lora_model, random_configs) -> None:
    model, table = build_lora_model(rank=2, alpha=0.5, randomize=True)
    model.eval()
    base_cfg = random_configs[0]

    energy, forces = _forward_energy_forces(model, [base_cfg], table)
    energy_val = energy.item()
    forces_val = forces.squeeze(0).detach().numpy()

    # Rotation invariance / covariance
    R = _rotation_matrix()
    rotated_cfg = _rotate_config(base_cfg, R)
    energy_rot, forces_rot = _forward_energy_forces(model, [rotated_cfg], table)
    assert np.allclose(energy_rot.item(), energy_val, rtol=1e-6, atol=1e-6)
    assert np.allclose(
        forces_val @ R.T, forces_rot.squeeze(0).detach().numpy(), rtol=1e-5, atol=1e-5
    )

    # Translation invariance
    shift = np.array([0.17, -0.05, 0.08])
    translated_cfg = _translate_config(base_cfg, shift)
    energy_trans, forces_trans = _forward_energy_forces(model, [translated_cfg], table)
    assert np.allclose(energy_trans.item(), energy_val, rtol=1e-6, atol=1e-6)
    assert np.allclose(
        forces_trans.squeeze(0).detach().numpy(), forces_val, rtol=1e-6, atol=1e-6
    )

    # Reflection invariance / covariance
    reflected_cfg, R_reflect = _reflect_config(base_cfg, np.array([1.0, -2.0, 3.0]))
    energy_ref, forces_ref = _forward_energy_forces(model, [reflected_cfg], table)
    assert np.allclose(energy_ref.item(), energy_val, rtol=1e-6, atol=1e-6)
    assert np.allclose(
        forces_val @ R_reflect.T,
        forces_ref.squeeze(0).detach().numpy(),
        rtol=1e-5,
        atol=1e-5,
    )
