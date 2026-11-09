"""Dataset controls against frozen readers, splits, losses and gradients."""

import pytest
import torch
from mace_core.config.resolved import ResolvedConfig
from mace_core.observables import DEFAULT_CATALOGUE, resolve_requested
from mace_core.outputs import MACEOutput
from mace_torch.train import build_loss, run_data_stage, run_model_stage
from mace_torch.train.loop import evaluate
from mace_torch.train.metrics import metric_specs

from tests.golden.harness import CLOSED_FORM_FP64, FP32
from tests.parity.test_data_backend_parity import assert_same, frames
from tests.parity.test_training_statistics_parity import training_config

pytestmark = pytest.mark.usefixtures("isolated")
TYPE_WEIGHTS = {"bulk": 2.0, "molecule": 0.0}


def weighted_frames(count=7):
    made = frames(count, 19)
    for index, atoms in enumerate(made):
        atoms.info.update(
            config_type=("bulk", "molecule", "other")[index % 3],
            config_weight=0.5 + index * 0.25,
            config_energy_weight=0.5,
            config_forces_weight=1.5,
            head="not_the_loaded_head",
            custom_head="also_not_the_loaded_head",
        )
    return made


def legacy_data(config, directory, head="default", head_key="head"):
    from mace.data import KeySpecification
    from mace.tools.scripts_utils import get_dataset_from_xyz

    spec = config.data.heads[head]
    directory.mkdir(exist_ok=True)
    keys = KeySpecification.from_defaults().update(info_keys={"head": head_key})
    collections, _ = get_dataset_from_xyz(
        str(directory),
        str(spec.train_file),
        str(spec.valid_file) if spec.valid_file is not None else None,
        config.data.valid_fraction,
        keys,
        config_type_weights=spec.config_type_weights,
        test_path=str(spec.test_file) if spec.test_file is not None else None,
        seed=config.runtime.seed,
        head_name=head,
        prefix=head,
    )
    return collections


@pytest.mark.parametrize("fraction", [0.01, 0.25, 0.6])
@pytest.mark.parametrize("seed", [3, 17])
def test_validation_fraction_preserves_members_order_and_recorded_indices(
    tmp_path, fraction, seed
):
    config = training_config(tmp_path, frames(19, 12))
    document = config.model_dump(mode="json")
    document["data"]["heads"]["default"]["valid_file"] = None
    document["data"]["valid_fraction"] = fraction
    document["runtime"]["seed"] = seed
    document["runtime"]["work_dir"] = str(tmp_path)
    config = ResolvedConfig.model_validate(document)
    actual = run_data_stage(config, DEFAULT_CATALOGUE)
    expected = legacy_data(config, tmp_path / "legacy")
    for loader, want in (
        (actual.train_eval_loaders["default"], expected.train),
        (actual.valid_loaders["default"], expected.valid),
    ):
        got = loader.dataset.configurations
        assert len(got) == len(want)
        for index, (native, frozen) in enumerate(zip(got, want, strict=True)):
            assert_same(native, frozen, index)
    assert len(expected.valid) == max(1, int(19 * fraction))
    filename = f"default_valid_indices_{seed}.txt"
    if len(expected.valid) >= 10:
        assert (tmp_path / filename).read_bytes() == (
            tmp_path / "legacy" / filename
        ).read_bytes()
    else:
        assert not (tmp_path / filename).exists()


@pytest.mark.parametrize("explicit_validation", [False, True])
@pytest.mark.parametrize("head_key", ["head", "custom_head"])
def test_type_weights_and_caller_head_match_legacy_on_every_split(
    tmp_path, explicit_validation, head_key
):
    config = training_config(
        tmp_path,
        weighted_frames(),
        extra=["--config_type_weights", str(TYPE_WEIGHTS)],
    )
    document = config.model_dump(mode="json")
    head = document["data"]["heads"]["default"]
    head["test_file"] = head["train_file"]
    head["valid_file"] = head["train_file"] if explicit_validation else None
    config = ResolvedConfig.model_validate(document)
    actual = run_data_stage(config, DEFAULT_CATALOGUE)
    expected = legacy_data(config, tmp_path / "legacy", head_key=head_key)
    tests = [item for _, group in expected.tests for item in group]
    for loader, want in (
        (actual.train_eval_loaders["default"], expected.train),
        (actual.valid_loaders["default"], expected.valid),
    ):
        for index, (native, frozen) in enumerate(
            zip(loader.dataset.configurations, want, strict=True)
        ):
            assert_same(native, frozen, index)
            assert native.head == "default" and "head" not in native.properties
        got_weights = torch.cat([batch.graph["weight"] for batch in loader])
        assert got_weights.tolist() == [item.weight for item in want]
    by_position = {tuple(item.positions.flatten()): item for item in tests}
    for item in actual.test_loaders["default"].dataset.configurations:
        assert_same(item, by_position[tuple(item.positions.flatten())], "test")


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_type_weights_reach_the_loss_and_prediction_gradients(tmp_path, dtype):
    from mace.data import AtomicData
    from mace.modules.loss import WeightedEnergyForcesLoss
    from mace.tools import AtomicNumberTable
    from mace.tools.torch_geometric.dataloader import DataLoader

    torch.set_default_dtype(dtype)
    config = training_config(
        tmp_path,
        weighted_frames(),
        extra=[
            "--config_type_weights",
            str(TYPE_WEIGHTS),
            "--valid_batch_size",
            "7",
            "--loss",
            "weighted",
            "--energy_weight",
            "2.3",
            "--forces_weight",
            "0.7",
        ],
    )
    native = run_data_stage(config, DEFAULT_CATALOGUE)
    batch = next(iter(native.train_eval_loaders["default"]))
    frozen = legacy_data(config, tmp_path / "legacy")
    oracle_batch = next(
        iter(
            DataLoader(
                [
                    AtomicData.from_config(
                        item, z_table=AtomicNumberTable([1, 8]), cutoff=2.5
                    )
                    for item in frozen.train
                ],
                batch_size=7,
                shuffle=False,
            )
        )
    )
    values = {
        name: (value + torch.linspace(-0.7, 0.9, value.numel()).reshape(value.shape))
        .detach()
        .requires_grad_()
        for name, value in batch.targets.items()
    }
    requested = resolve_requested(config.model.observables, DEFAULT_CATALOGUE)
    loss = build_loss(requested, config.loss)
    actual = loss(
        MACEOutput(total_energy=values["energy"], forces=values["forces"]), batch
    )
    expected = WeightedEnergyForcesLoss(
        energy_weight=config.loss.weights["energy"],
        forces_weight=config.loss.weights["forces"],
    )(oracle_batch, values)
    tensors = tuple(values.values())
    got_grad = torch.autograd.grad(actual, tensors, retain_graph=True)
    want_grad = torch.autograd.grad(expected, tensors)
    tolerance = CLOSED_FORM_FP64 if dtype == torch.float64 else FP32
    for got, want in zip((actual, *got_grad), (expected, *want_grad), strict=True):
        torch.testing.assert_close(got, want, atol=tolerance.atol, rtol=tolerance.rtol)
    zero_graphs = oracle_batch.weight == 0
    assert torch.count_nonzero(got_grad[0][zero_graphs]) == 0
    assert torch.count_nonzero(got_grad[1][zero_graphs[oracle_batch.batch]]) == 0


@pytest.mark.parametrize("batch_size", [1, 3, 20])
def test_validation_batches_and_reported_loss_match_legacy(tmp_path, batch_size):
    from mace.data import AtomicData
    from mace.modules.loss import WeightedEnergyForcesLoss
    from mace.tools import AtomicNumberTable
    from mace.tools.torch_geometric.dataloader import DataLoader
    from mace.tools.train import MACELoss

    torch.set_default_dtype(torch.float64)
    config = training_config(
        tmp_path,
        weighted_frames(),
        extra=[
            "--valid_batch_size",
            str(batch_size),
            "--loss",
            "weighted",
            "--energy_weight",
            "2.3",
            "--forces_weight",
            "0.7",
            "--config_type_weights",
            str(TYPE_WEIGHTS),
        ],
    )
    document = config.model_dump(mode="json")
    document["data"]["heads"]["default"]["valid_file"] = str(tmp_path / "train.xyz")
    config = ResolvedConfig.model_validate(document)
    data = run_data_stage(config, DEFAULT_CATALOGUE)
    built = run_model_stage(config, data, DEFAULT_CATALOGUE)
    loader = data.valid_loaders["default"]
    frozen = legacy_data(config, tmp_path / "legacy")
    oracle_loader = DataLoader(
        [
            AtomicData.from_config(item, z_table=AtomicNumberTable([1, 8]), cutoff=2.5)
            for item in frozen.valid
        ],
        batch_size=batch_size,
        shuffle=False,
        drop_last=False,
    )
    oracle = MACELoss(WeightedEnergyForcesLoss(energy_weight=2.3, forces_weight=0.7))
    seen = 0
    for native_batch, frozen_batch in zip(loader, oracle_loader, strict=True):
        count = len(native_batch.graph["ptr"]) - 1
        assert count == frozen_batch.num_graphs
        seen += count
        predictions = built.engine(native_batch.graph, compute=("forces",))
        oracle.update(
            output={
                "energy": predictions.get("energy"),
                "forces": predictions.get("forces"),
            },
            batch=frozen_batch,
        )
    assert seen == 7
    expected, _ = oracle.compute()
    requested = resolve_requested(config.model.observables, DEFAULT_CATALOGUE)
    actual = evaluate(
        built.engine,
        loader,
        build_loss(requested, config.loss),
        metric_specs(requested),
    )
    torch.testing.assert_close(
        torch.tensor(actual["loss"]),
        torch.tensor(expected),
        atol=CLOSED_FORM_FP64.atol,
        rtol=CLOSED_FORM_FP64.rtol,
    )
