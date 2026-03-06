import argparse
from pathlib import Path

import ase.io
import numpy as np
import torch
from ase.atoms import Atoms
from e3nn import o3

from mace.cli.eval_configs import run as eval_configs_run
from mace.modules import interaction_classes
from mace.modules.models import ScaleShiftMACE
from mace.tools.torch_tools import default_dtype


def _write_configs(tmp_path: Path) -> Path:
    '''Write two configs with different numbers of atoms to xyz file.'''
    configs = [
        Atoms(numbers=[1, 1], positions=[[0, 0, 0], [0, 0, 0.74]]),
        Atoms(
            numbers=[8, 1, 1],
            positions=[[0, 0, 0], [0.96, 0, 0], [-0.32, 0.94, 0]],
        ),
    ]
    path = tmp_path / "fit.xyz"
    ase.io.write(path, configs)
    return path


def _make_multihead_model(tmp_path: Path) -> Path:
    '''The multihead model has all weights set to zero, so outputs atomic energies.'''
    model_config = dict(
        r_max=4.0,
        num_bessel=4,
        num_polynomial_cutoff=3,
        max_ell=1,
        interaction_cls=interaction_classes["RealAgnosticResidualInteractionBlock"],
        interaction_cls_first=interaction_classes["RealAgnosticResidualInteractionBlock"],
        num_interactions=2,
        num_elements=2,
        hidden_irreps=o3.Irreps("8x0e + 8x1o"),
        MLP_irreps=o3.Irreps("4x0e"),
        gate=torch.nn.functional.silu,
        atomic_energies=np.array(
            [
                [0.0, 0.0],  # Default head: H, O
                [1.0, 2.0],  # Alt head: H=1.0, O=2.0
            ]
        ),
        avg_num_neighbors=4,
        atomic_numbers=[1, 8],
        correlation=2,
        radial_type="bessel",
        atomic_inter_shift=[0.0, 0.0],
        atomic_inter_scale=[1.0, 1.0],
        heads=["Default", "Alt"],
    )

    with default_dtype(torch.float32):
        model = ScaleShiftMACE(**model_config)
        for param in model.parameters():
            param.data.zero_()
        path = tmp_path / "multihead.model"
        torch.save(model, path)
    return path


def _run_eval_configs(
    model_path: Path,
    configs_path: Path,
    output_path: Path,
    head: str | None,
    return_node_energies: bool,
) -> None:
    args = argparse.Namespace(
        model=str(model_path),
        configs=str(configs_path),
        output=str(output_path),
        device="cpu",
        default_dtype="float32",
        batch_size=2,
        compute_stress=False,
        compute_bec=False,
        enable_cueq=False,
        return_contributions=False,
        return_descriptors=False,
        return_node_energies=return_node_energies,
        info_prefix="MACE_",
        head=head,
    )
    eval_configs_run(args)


def test_eval_configs_head_selection(tmp_path: Path) -> None:
    model_path = _make_multihead_model(tmp_path)
    configs_path = _write_configs(tmp_path)

    output_default = tmp_path / "out_default.xyz"
    _run_eval_configs(
        model_path=model_path,
        configs_path=configs_path,
        output_path=output_default,
        head=None,
        return_node_energies=False,
    )
    atoms_default = ase.io.read(str(output_default), index=":")
    print(atoms_default)
    default_energies = [at.info["MACE_energy"] for at in atoms_default]
    print(default_energies)
    assert np.allclose(default_energies, [0.0, 0.0])

    output_alt = tmp_path / "out_alt.xyz"
    _run_eval_configs(
        model_path=model_path,
        configs_path=configs_path,
        output_path=output_alt,
        head="Alt",
        return_node_energies=False,
    )
    atoms_alt = ase.io.read(str(output_alt), index=":")
    alt_energies = [at.info["MACE_energy"] for at in atoms_alt]
    assert np.allclose(alt_energies, [2.0, 4.0])


def test_eval_configs_node_energies_varying_atoms(tmp_path: Path) -> None:
    model_path = _make_multihead_model(tmp_path)
    configs_path = _write_configs(tmp_path)

    output_path = tmp_path / "out_node_energies.xyz"
    _run_eval_configs(
        model_path=model_path,
        configs_path=configs_path,
        output_path=output_path,
        head="Alt",
        return_node_energies=True,
    )

    atoms_out = ase.io.read(str(output_path), index=":")
    node_energies = [at.arrays["MACE_node_energies"] for at in atoms_out]

    assert node_energies[0].shape == (2,)
    assert node_energies[1].shape == (3,)
    assert np.allclose(node_energies[0], [1.0, 1.0]) # H, H
    assert np.allclose(node_energies[1], [2.0, 1.0, 1.0]) # O, H, H
