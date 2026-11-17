from __future__ import annotations

import argparse
from pathlib import Path

import ase
import ase.io
import numpy as np
import pytest
import torch
from e3nn import o3

from mace import modules
from mace.cli import eval_configs
from mace.tools import torch_tools


class _DeviceTensor(torch.Tensor):
    """CPU tensor that mimics a non-CPU tensor by rejecting host indexing/conversion without .cpu()."""

    def __getitem__(self, item):
        result = super().__getitem__(item)
        if isinstance(result, torch.Tensor) and not isinstance(result, _DeviceTensor):
            return result.as_subclass(_DeviceTensor)
        return result

    def __index__(self):
        raise TypeError(
            "can't convert non-CPU device tensor to Python index; "
            "use Tensor.cpu() first."
        )

    def cpu(self):
        return torch.Tensor(self.as_subclass(torch.Tensor))

    def __array__(self, dtype=None):
        raise TypeError(
            "can't convert cuda:0 device type tensor to numpy. "
            "Use Tensor.cpu() to copy the tensor to host memory first."
        )


class MACELES(modules.MACE):
    """Stub MACELES subclass so eval_configs accepts compute_bec=True."""


def _build_tiny_model() -> MACELES:
    return MACELES(
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


def test_device_tensor_rejects_direct_numpy_split_without_cpu():
    ptr = torch.tensor([0, 2, 5], dtype=torch.long).as_subclass(_DeviceTensor)
    arr = np.zeros((5, 3))

    with pytest.raises(TypeError, match="use Tensor.cpu\\(\\) first"):
        np.split(arr, indices_or_sections=ptr[1:], axis=0)

    chunks = np.split(arr, indices_or_sections=torch_tools.to_numpy(ptr[1:]), axis=0)[
        :-1
    ]
    assert [c.shape for c in chunks] == [(2, 3), (3, 3)]


@pytest.mark.parametrize(
    ("raw_value", "expected"),
    [
        ("False", False),
        ("false", False),
        ("0", False),
        ("no", False),
        ("True", True),
        ("true", True),
        ("1", True),
        ("yes", True),
    ],
)
def test_parse_args_descriptor_invariants_only_str2bool(
    monkeypatch, raw_value: str, expected: bool
):
    monkeypatch.setattr(
        "sys.argv",
        [
            "mace_eval_configs",
            "--configs",
            "in.xyz",
            "--model",
            "m.model",
            "--output",
            "out.xyz",
            "--descriptor_invariants_only",
            raw_value,
        ],
    )
    parsed = eval_configs.parse_args()
    assert parsed.descriptor_invariants_only is expected


def test_eval_configs_splits_all_per_atom_outputs_with_non_cpu_ptr(
    tmp_path: Path, monkeypatch
):
    configs_path = tmp_path / "structures.xyz"
    output_path = tmp_path / "evaluated.xyz"

    model = _build_tiny_model()
    monkeypatch.setattr(eval_configs.torch, "load", lambda *args, **kwargs: model)

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

    original_get_model_output = eval_configs.get_model_output

    def _augmented_model_output(
        model_arg, batch_dict, compute_stress, compute_bec, compute_magforces=False
    ):
        output = original_get_model_output(
            model_arg,
            batch_dict,
            compute_stress=compute_stress,
            compute_bec=False,
            compute_magforces=False,
        )
        n_atoms = batch_dict["positions"].shape[0]
        dtype = batch_dict["positions"].dtype
        output["BEC"] = torch.ones((n_atoms, 3, 3), dtype=dtype)
        output["latent_charges"] = torch.ones((n_atoms,), dtype=dtype)
        output["latent_dipoles"] = torch.ones((n_atoms, 3), dtype=dtype)
        output["latent_kappas"] = torch.ones((n_atoms,), dtype=dtype)
        output["latent_alphas"] = torch.ones((n_atoms, 3, 3), dtype=dtype)
        output["latent_quads"] = torch.ones((n_atoms, 3, 3), dtype=dtype)
        output["magforces"] = torch.ones((n_atoms, 3), dtype=dtype)
        return output

    monkeypatch.setattr(eval_configs, "get_model_output", _augmented_model_output)

    args = argparse.Namespace(
        configs=str(configs_path),
        model="unused.model",
        output=str(output_path),
        device="cpu",
        enable_cueq=False,
        default_dtype="float32",
        batch_size=2,
        compute_stress=False,
        compute_bec=True,
        return_contributions=False,
        return_descriptors=True,
        descriptor_num_layers=-1,
        descriptor_aggregation_method=None,
        descriptor_invariants_only=True,
        return_node_energies=True,
        return_magforces=True,
        info_prefix="MACE_",
        head=None,
        magmom_key="magmom",
    )

    eval_configs.run(args)

    evaluated = ase.io.read(output_path, index=":")
    assert len(evaluated) == 2
    for atoms, expected_atoms in zip(evaluated, (2, 3)):
        assert atoms.arrays["MACE_forces"].shape == (expected_atoms, 3)
        assert atoms.arrays["MACE_magforces"].shape == (expected_atoms, 3)
        assert atoms.arrays["MACE_node_energies"].shape == (expected_atoms,)
        assert atoms.arrays["MACE_descriptors"].shape[0] == expected_atoms
        assert atoms.arrays["MACE_BEC"].shape == (expected_atoms, 9)
        assert atoms.arrays["MACE_latent_charges"].shape == (expected_atoms,)
        assert atoms.arrays["MACE_latent_dipoles"].shape == (expected_atoms, 3)
        assert atoms.arrays["MACE_latent_kappas"].shape == (expected_atoms,)
        assert atoms.arrays["MACE_latent_alphas"].shape == (expected_atoms, 9)
        assert atoms.arrays["MACE_latent_quads"].shape == (expected_atoms, 9)


def test_eval_configs_equivariant_descriptors_and_non_default_heads(
    tmp_path: Path, monkeypatch
):
    configs_path = tmp_path / "structures.xyz"
    out_inv = tmp_path / "inv.xyz"
    out_full = tmp_path / "full.xyz"

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
        num_interactions=2,
        num_elements=1,
        hidden_irreps=o3.Irreps("2x0e + 2x1o"),
        MLP_irreps=o3.Irreps("2x0e"),
        atomic_energies=np.zeros((2, 1)),
        avg_num_neighbors=1.0,
        atomic_numbers=[1],
        correlation=1,
        gate=torch.nn.functional.silu,
        heads=["pt_head", "custom_head"],
    )
    monkeypatch.setattr(eval_configs.torch, "load", lambda *args, **kwargs: model)

    atoms = ase.Atoms(
        "H2",
        positions=[[0.0, 0.0, 0.0], [0.0, 0.0, 0.8]],
        cell=[10.0, 10.0, 10.0],
        pbc=True,
    )
    ase.io.write(configs_path, [atoms], format="extxyz")

    for invariants_only, out_path, expected_cols in (
        (True, out_inv, 4),
        (False, out_full, 10),
    ):
        args = argparse.Namespace(
            configs=str(configs_path),
            model="unused.model",
            output=str(out_path),
            device="cpu",
            enable_cueq=False,
            default_dtype="float32",
            batch_size=1,
            compute_stress=False,
            compute_bec=False,
            return_contributions=False,
            return_descriptors=True,
            descriptor_num_layers=-1,
            descriptor_aggregation_method=None,
            descriptor_invariants_only=invariants_only,
            return_node_energies=False,
            return_magforces=False,
            info_prefix="MACE_",
            head=None,
            magmom_key="magmom",
        )
        eval_configs.run(args)
        evaluated = ase.io.read(out_path, index=":")
        assert evaluated[0].arrays["MACE_descriptors"].shape == (2, expected_cols)
