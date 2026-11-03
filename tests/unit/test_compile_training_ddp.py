import importlib
from pathlib import Path

import numpy as np
import pytest
import torch
import torch.distributed as dist
import torch.nn.functional as F
import torch.multiprocessing as mp
from ase import build
from e3nn import o3
from torch.nn.parallel import DistributedDataParallel
from torch.testing import assert_close

from mace import data, modules, tools
from mace.tools import compile as mace_compile
from mace.tools import compile_training
from mace.tools import torch_geometric

train_tools = importlib.import_module("mace.tools.train")


_TABLE = tools.AtomicNumberTable([6])
_CUTOFF = 5.0
_OUTPUT_ARGS = {
    "energy": True,
    "forces": True,
    "virials": True,
    "stress": True,
}


def _create_model() -> torch.nn.Module:
    torch_geometric.seed_everything(1702)
    with mace_compile.disable_e3nn_codegen():
        model = modules.ScaleShiftMACE(
            r_max=_CUTOFF,
            num_bessel=4,
            num_polynomial_cutoff=4,
            max_ell=1,
            interaction_cls=modules.interaction_classes[
                "RealAgnosticResidualInteractionBlock"
            ],
            interaction_cls_first=modules.interaction_classes[
                "RealAgnosticResidualInteractionBlock"
            ],
            num_interactions=2,
            num_elements=1,
            hidden_irreps=o3.Irreps("8x0e + 8x1o"),
            MLP_irreps=o3.Irreps("8x0e"),
            gate=F.silu,
            atomic_energies=np.array([1.0], dtype=float),
            avg_num_neighbors=8,
            atomic_numbers=_TABLE.zs,
            correlation=2,
            radial_type="bessel",
            atomic_inter_scale=1.0,
            atomic_inter_shift=0.0,
        )
    return model


def _create_batch(ranks: list[int]) -> dict[str, torch.Tensor]:
    configs = []
    for rank in ranks:
        for variant in range(2):
            atoms = build.bulk(
                "C",
                "diamond",
                a=3.50 + 0.04 * rank + 0.01 * variant,
                cubic=True,
            )
            atoms.positions[0, 0] += 0.002 * (rank + variant)
            configs.append(data.config_from_atoms(atoms))
    loader = torch_geometric.dataloader.DataLoader(
        dataset=[
            data.AtomicData.from_config(config, z_table=_TABLE, cutoff=_CUTOFF)
            for config in configs
        ],
        batch_size=len(configs),
        shuffle=False,
        drop_last=False,
    )
    return next(iter(loader)).to_dict()


def _objective(outputs: dict[str, torch.Tensor]) -> torch.Tensor:
    return (
        outputs["energy"].square().mean()
        + 0.3 * outputs["forces"].square().mean()
        + 0.2 * outputs["virials"].square().mean()
        + 0.1 * outputs["stress"].square().mean()
    )


def _ddp_worker(
    rank: int,
    world_size: int,
    rendezvous_path: str,
    output_dir: str,
) -> None:
    torch.set_num_threads(1)
    dist.init_process_group(
        backend="gloo",
        init_method=f"file://{rendezvous_path}",
        rank=rank,
        world_size=world_size,
    )
    try:
        with tools.torch_tools.default_dtype(torch.float64):
            model = _create_model()
            batch = _create_batch([rank])
            validation_batch = _create_batch([rank, rank + world_size])
            optimizer = torch.optim.Adam(model.parameters(), lr=1.0e-3)
            facade = mace_compile.compile_training_model(
                model,
                batch,
                output_args=_OUTPUT_ARGS,
                mode="default",
                dynamic=True,
                validation_batch=validation_batch,
            )
            facade.materialize_compiled_backward(batch)
            facade = train_tools._synchronize_compiled_training_state(
                facade,
                device=torch.device("cpu"),
                distributed=True,
            )
            assert facade.is_compiled
            distributed_model = DistributedDataParallel(facade)

            base_parameter_ids = [id(parameter) for parameter in model.parameters()]
            reducer_parameter_ids = [
                id(parameter) for parameter in distributed_model.parameters()
            ]
            assert reducer_parameter_ids == base_parameter_ids
            assert len(reducer_parameter_ids) == len(set(reducer_parameter_ids))

            optimizer.zero_grad(set_to_none=True)
            outputs = distributed_model(batch)
            _objective(outputs).backward()
            optimizer.step()
            train_tools._synchronize_compiled_training_state(
                distributed_model,
                device=torch.device("cpu"),
                distributed=True,
            )

            torch.save(
                {
                    name: value.detach().cpu()
                    for name, value in model.state_dict().items()
                },
                Path(output_dir) / f"rank-{rank}.pt",
            )
    finally:
        dist.destroy_process_group()


@pytest.mark.skipif(
    compile_training._TORCH_TOO_OLD_FOR_DYNAMIC_SHAPES,
    reason="compiled training requires torch >= 2.6 for dynamic-shape lowering",
)
def test_compiled_training_ddp_matches_global_batch_update(tmp_path):
    rendezvous_path = tmp_path / "gloo-rendezvous"
    mp.spawn(
        _ddp_worker,
        args=(2, str(rendezvous_path), str(tmp_path)),
        nprocs=2,
        join=True,
    )

    rank_states = [
        torch.load(tmp_path / f"rank-{rank}.pt", weights_only=True) for rank in range(2)
    ]
    assert rank_states[0].keys() == rank_states[1].keys()
    for name in rank_states[0]:
        assert_close(rank_states[0][name], rank_states[1][name])

    with tools.torch_tools.default_dtype(torch.float64):
        reference_model = _create_model()
        reference_batch = _create_batch([0, 1])
        reference_optimizer = torch.optim.Adam(reference_model.parameters(), lr=1.0e-3)
        reference_optimizer.zero_grad(set_to_none=True)
        reference_outputs = reference_model(
            reference_batch,
            training=True,
            compute_force=True,
            compute_virials=True,
            compute_stress=True,
        )
        _objective(reference_outputs).backward()
        reference_optimizer.step()

    reference_state = reference_model.state_dict()
    assert rank_states[0].keys() == reference_state.keys()
    for name, reference_value in reference_state.items():
        assert_close(rank_states[0][name], reference_value)
