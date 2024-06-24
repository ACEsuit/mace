import tempfile
from unittest.mock import MagicMock

import numpy as np
import pytest
import torch
import torch.nn.functional as F
from e3nn import o3

from mace import data, modules, tools
from mace.tools import scripts_utils, torch_geometric

try:
    import schedulefree
except ImportError:
    pytest.skip(
        "Skipping schedulefree tests due to ImportError", allow_module_level=True
    )

torch.set_default_dtype(torch.float64)

table = tools.AtomicNumberTable([6])
atomic_energies = np.array([1.0], dtype=float)
cutoff = 5.0


def create_mace(device: str, seed: int = 1702):
    torch_geometric.seed_everything(seed)

    model_config = {
        "r_max": cutoff,
        "num_bessel": 8,
        "num_polynomial_cutoff": 6,
        "max_ell": 3,
        "interaction_cls": modules.interaction_classes[
            "RealAgnosticResidualInteractionBlock"
        ],
        "interaction_cls_first": modules.interaction_classes[
            "RealAgnosticResidualInteractionBlock"
        ],
        "num_interactions": 2,
        "num_elements": 1,
        "hidden_irreps": o3.Irreps("8x0e + 8x1o"),
        "MLP_irreps": o3.Irreps("16x0e"),
        "gate": F.silu,
        "atomic_energies": atomic_energies,
        "avg_num_neighbors": 8,
        "atomic_numbers": table.zs,
        "correlation": 3,
        "radial_type": "bessel",
    }
    model = modules.MACE(**model_config)
    return model.to(device)


def create_batch(device: str):
    from ase import build

    size = 2
    atoms = build.bulk("C", "diamond", a=3.567, cubic=True)
    atoms_list = [atoms.repeat((size, size, size))]
    print("Number of atoms", len(atoms_list[0]))

    configs = [data.config_from_atoms(atoms) for atoms in atoms_list]
    data_loader = torch_geometric.dataloader.DataLoader(
        dataset=[
            data.AtomicData.from_config(config, z_table=table, cutoff=cutoff)
            for config in configs
        ],
        batch_size=1,
        shuffle=False,
        drop_last=False,
    )
    batch = next(iter(data_loader))
    batch = batch.to(device)
    batch = batch.to_dict()
    return batch


def do_optimization_step(
    model,
    optimizer,
    device,
):
    batch = create_batch(device)
    model.train()
    optimizer.train()
    optimizer.zero_grad()
    output = model(batch, training=True, compute_force=False)
    loss = output["energy"].mean()
    loss.backward()
    optimizer.step()
    model.eval()
    optimizer.eval()


@pytest.mark.parametrize("device", ["cpu", "cuda"])
def test_can_load_checkpoint(device):
    model = create_mace(device)
    optimizer = schedulefree.adamw_schedulefree.AdamWScheduleFree(model.parameters())
    args = MagicMock()
    args.optimizer = "schedulefree"
    args.scheduler = "ExponentialLR"
    args.lr_scheduler_gamma = 0.9
    lr_scheduler = scripts_utils.LRScheduler(optimizer, args)
    with tempfile.TemporaryDirectory() as d:
        checkpoint_handler = tools.CheckpointHandler(
            directory=d, keep=False, tag="schedulefree"
        )
        for _ in range(10):
            do_optimization_step(model, optimizer, device)
        batch = create_batch(device)
        output = model(batch)
        energy = output["energy"].detach().cpu().numpy()

        state = tools.CheckpointState(
            model=model, optimizer=optimizer, lr_scheduler=lr_scheduler
        )
        checkpoint_handler.save(state, epochs=0, keep_last=False)
        checkpoint_handler.load_latest(
            state=tools.CheckpointState(model, optimizer, lr_scheduler),
            swa=False,
        )
        batch = create_batch(device)
        output = model(batch)
        new_energy = output["energy"].detach().cpu().numpy()
        assert np.allclose(energy, new_energy, atol=1e-9)
