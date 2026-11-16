from __future__ import annotations

import argparse
from pathlib import Path

import ase
import ase.io
import numpy as np
import torch
from e3nn import o3

from mace import modules
from mace.cli import eval_configs


class _DeviceTensor(torch.Tensor):
    """CPU tensor that mimics a non-CPU tensor by rejecting direct __array__."""

    def __getitem__(self, item):
        result = super().__getitem__(item)
        if isinstance(result, torch.Tensor) and not isinstance(result, _DeviceTensor):
            return result.as_subclass(_DeviceTensor)
        return result

    def cpu(self):
        return torch.Tensor(self.as_subclass(torch.Tensor))

    def __array__(self, dtype=None):
        raise TypeError(
            "can't convert cuda:0 device type tensor to numpy. "
            "Use Tensor.cpu() to copy the tensor to host memory first."
        )


def _build_tiny_model(path: Path) -> None:
    model = modules.MACE(
        r_max=3.0,
        num_bessel=4,
        num_polynomial_cutoff=2,
        max_ell=1,
        interaction_cls=modules.interaction_classes[
            "RealAgnosticResidualInteractionBlock"
        ],
        interaction_cls_first=modules.interaction_classes[
            "RealAgnosticResidualInteractionBlock"
        ],
        num_interactions=1,
        num_elements=1,
        hidden_irreps=o3.Irreps("2x0e"),
        MLP_irreps=o3.Irreps("2x0e"),
        atomic_energies=np.array([0.0]),
        avg_num_neighbors=1.0,
        atomic_numbers=[1],
        correlation=1,
        gate=torch.nn.functional.silu,
    )
    torch.save(model, path)


def test_eval_configs_splits_outputs_with_non_cpu_ptr(tmp_path: Path, monkeypatch):
    model_path = tmp_path / "tiny.model"
    configs_path = tmp_path / "structures.xyz"
    output_path = tmp_path / "evaluated.xyz"

    _build_tiny_model(model_path)
    atoms_list = [
        ase.Atoms(
            "H2",
            positions=[[0.0, 0.0, 0.0], [0.0, 0.0, 0.8]],
            cell=[10.0, 10.0, 10.0],
            pbc=True,
        ),
        ase.Atoms(
            "H3",
            positions=[[0.0, 0.0, 0.0], [0.0, 0.0, 0.8], [0.8, 0.0, 0.0]],
            cell=[10.0, 10.0, 10.0],
            pbc=True,
        ),
    ]
    ase.io.write(configs_path, atoms_list, format="extxyz")

    original_dataloader = eval_configs.torch_geometric.dataloader.DataLoader

    class _DevicePtrDataLoader:
        def __init__(self, *args, **kwargs):
            self._inner = original_dataloader(*args, **kwargs)

        def __iter__(self):
            for batch in self._inner:
                batch.ptr = batch.ptr.as_subclass(_DeviceTensor)
                yield batch

    monkeypatch.setattr(
        eval_configs.torch_geometric.dataloader, "DataLoader", _DevicePtrDataLoader
    )

    args = argparse.Namespace(
        configs=str(configs_path),
        model=str(model_path),
        output=str(output_path),
        device="cpu",
        enable_cueq=False,
        default_dtype="float32",
        batch_size=2,
        compute_stress=False,
        compute_bec=False,
        return_contributions=False,
        return_descriptors=True,
        descriptor_num_layers=-1,
        descriptor_aggregation_method=None,
        descriptor_invariants_only=True,
        return_node_energies=True,
        return_magforces=False,
        info_prefix="MACE_",
        head=None,
        magmom_key="magmom",
    )

    eval_configs.run(args)

    evaluated = ase.io.read(output_path, index=":")
    assert len(evaluated) == 2
    assert evaluated[0].arrays["MACE_forces"].shape == (2, 3)
    assert evaluated[1].arrays["MACE_forces"].shape == (3, 3)
    assert evaluated[0].arrays["MACE_node_energies"].shape == (2,)
    assert evaluated[1].arrays["MACE_node_energies"].shape == (3,)
    assert evaluated[0].arrays["MACE_descriptors"].shape[0] == 2
    assert evaluated[1].arrays["MACE_descriptors"].shape[0] == 3
