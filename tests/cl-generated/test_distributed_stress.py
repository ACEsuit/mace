"""Multi-process (DDP) regression test for the ``num_graphs == 1`` fast path in
``get_symmetric_displacement``.

We have a single local GPU, so true multi-GPU/nccl is deferred to the raven jobscript.
Here we use the ``gloo`` backend on CPU with ``torch.multiprocessing.spawn`` as a faithful
stand-in: DDP's forward is per-rank independent and only the gradients are all-reduced, so
this exercises exactly the risk multi-GPU adds on top of single-process correctness:
the new branch participating in autograd and grad-sync under an **uneven batch split**.

* rank 0 forwards a batch of 2 graphs  -> original ``"be,bec->bc"`` (else) path
* rank 1 forwards a batch of 1 graph   -> new ``"be,ec->bc"`` (num_graphs == 1) path

A stress-dependent loss is back-propagated, DDP averages the gradients, and we assert the
synced gradient equals a single-process reference (average of the two per-rank gradients).
Equality proves the fast path back-propagates correctly and grad-sync is consistent.

NOTE: gloo + ``mp.spawn`` under pytest is flaky on native Windows; run on WSL/Linux.
"""

import os
import socket
import sys

import numpy as np
import pytest
import torch
from ase import build
from e3nn import o3

from mace import data, modules, tools
from mace.tools import torch_geometric

pytestmark = pytest.mark.skipif(
    sys.platform == "win32",
    reason="gloo + mp.spawn is unreliable on native Windows; run this on WSL/Linux",
)

R_MAX = 5.0


def _build_model():
    """Deterministic small ScaleShiftMACE (same weights across all processes)."""
    torch.manual_seed(0)
    z_table = tools.AtomicNumberTable([14])
    return modules.ScaleShiftMACE(
        r_max=R_MAX,
        num_bessel=8,
        num_polynomial_cutoff=6,
        max_ell=2,
        interaction_cls=modules.interaction_classes[
            "RealAgnosticResidualInteractionBlock"
        ],
        interaction_cls_first=modules.interaction_classes[
            "RealAgnosticResidualInteractionBlock"
        ],
        num_interactions=2,
        num_elements=1,
        hidden_irreps=o3.Irreps("16x0e + 16x1o"),
        MLP_irreps=o3.Irreps("8x0e"),
        gate=torch.nn.functional.silu,
        atomic_energies=np.array([0.0]),
        avg_num_neighbors=4.0,
        atomic_numbers=z_table.zs,
        correlation=3,
        atomic_inter_scale=1.0,
        atomic_inter_shift=0.0,
    )


def _si(strain):
    atoms = build.bulk("Si", "diamond", a=5.43)
    atoms.set_cell(np.dot(atoms.get_cell(), np.eye(3) * strain), scale_atoms=True)
    z_table = tools.AtomicNumberTable([14])
    return data.AtomicData.from_config(
        data.config_from_atoms(
            atoms, key_specification=data.KeySpecification.from_defaults()
        ),
        z_table=z_table,
        cutoff=R_MAX,
    )


# Two distinct configs so rank 0's batch is a genuine 2-graph batch.
_CONFIG_A = lambda: _si(1.02)  # noqa: E731
_CONFIG_B = lambda: _si(1.03)  # noqa: E731


def _batch(dataset):
    loader = torch_geometric.dataloader.DataLoader(
        dataset=dataset, batch_size=len(dataset), shuffle=False, drop_last=False
    )
    return next(iter(loader))


def _stress_loss(model, batch):
    out = model(
        batch.to_dict(),
        training=True,  # create_graph=True so we can backprop through stress/forces
        compute_force=True,
        compute_virials=True,
        compute_stress=True,
    )
    # Depends on stress so the changed branch is part of the autograd graph.
    return (
        out["stress"].pow(2).sum()
        + out["energy"].pow(2).sum()
        + out["forces"].pow(2).sum()
    )


def _grad_vector(module):
    """Flatten grads (Nones -> zeros) in a fixed parameter order."""
    return torch.cat(
        [
            (p.grad if p.grad is not None else torch.zeros_like(p)).flatten()
            for _, p in sorted(module.named_parameters())
        ]
    )


def _worker(rank, world_size, port, out_dir):
    torch.set_default_dtype(torch.float64)
    os.environ["MASTER_ADDR"] = "127.0.0.1"
    os.environ["MASTER_PORT"] = str(port)
    torch.distributed.init_process_group(
        backend="gloo", rank=rank, world_size=world_size
    )
    try:
        model = _build_model()
        ddp = torch.nn.parallel.DistributedDataParallel(model)

        if rank == 0:
            batch = _batch([_CONFIG_A(), _CONFIG_B()])  # 2 graphs -> else branch
        else:
            batch = _batch([_CONFIG_A()])  # 1 graph -> num_graphs == 1 branch

        loss = _stress_loss(ddp, batch)
        loss.backward()

        grad = _grad_vector(model)
        assert torch.isfinite(grad).all(), f"rank {rank}: non-finite gradient"
        torch.save(grad, os.path.join(out_dir, f"grad_{rank}.pt"))
    finally:
        torch.distributed.destroy_process_group()


def _free_port():
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def test_ddp_stress_uneven_split(tmp_path):
    if not torch.distributed.is_available():
        pytest.skip("torch.distributed not available")

    world_size = 2
    port = _free_port()
    torch.multiprocessing.spawn(
        _worker,
        args=(world_size, port, str(tmp_path)),
        nprocs=world_size,
        join=True,
    )

    g0 = torch.load(tmp_path / "grad_0.pt")
    g1 = torch.load(tmp_path / "grad_1.pt")

    # DDP all-reduces (averages) gradients -> both ranks end identical.
    assert torch.allclose(g0, g1, atol=1e-10), "DDP did not synchronize gradients"

    # Single-process reference: average of the per-rank local gradients.
    torch.set_default_dtype(torch.float64)
    ref_model = _build_model()
    _stress_loss(ref_model, _batch([_CONFIG_A(), _CONFIG_B()])).backward()
    grad_rank0 = _grad_vector(ref_model)

    ref_model.zero_grad(set_to_none=True)
    _stress_loss(ref_model, _batch([_CONFIG_A()])).backward()
    grad_rank1 = _grad_vector(ref_model)

    expected = 0.5 * (grad_rank0 + grad_rank1)
    assert torch.allclose(g0, expected, atol=1e-8), (
        "DDP-synced gradient (with the num_graphs==1 fast path on rank 1) "
        "does not match the single-process reference"
    )
