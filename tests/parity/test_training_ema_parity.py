"""The averaging schedule used by frozen legacy training, including resume."""

import copy

import pytest
import torch
from mace_torch.train.ema import ExponentialMovingAverage
from torch_ema import ExponentialMovingAverage as LegacyAverage

from tests.golden.harness import CLOSED_FORM_FP64


@pytest.mark.parametrize('warmup', [False, True])
@pytest.mark.parametrize('decay', [0.2, 0.99])
@pytest.mark.parametrize('device', ['cpu', pytest.param('cuda', marks=pytest.mark.gpu)])
def test_averaged_weights_match_legacy_through_updates_and_resume(warmup, decay, device):
    initial = torch.tensor([1.0, -2.0, 0.5], dtype=torch.float64, device=device)
    source = torch.nn.Parameter(initial.clone())
    target = torch.nn.Parameter(initial.clone())
    legacy = LegacyAverage([source], decay=decay, use_num_updates=warmup)
    native = ExponentialMovingAverage([target], decay=decay, warmup=warmup)
    for step in range(12):
        with torch.no_grad():
            value = initial * (step + 2) + step ** 2
            source.copy_(value)
            target.copy_(value)
        legacy.update()
        native.update()
        assert native.num_updates == step + 1
        with legacy.average_parameters(), native.average_parameters():
            torch.testing.assert_close(target, source, atol=CLOSED_FORM_FP64.atol,
                                       rtol=CLOSED_FORM_FP64.rtol)
        assert torch.equal(target, value)
        assert torch.equal(source, value)
        if step == 3:
            restored = ExponentialMovingAverage([target], decay=0.5)
            restored.load_state_dict(copy.deepcopy(native.state_dict()))
            native = restored


@pytest.mark.parametrize('anchor', ['tiny_scaleshift.model', 'tiny_mace.model'])
@pytest.mark.parametrize('device', ['cpu', pytest.param('cuda', marks=pytest.mark.gpu)])
def test_hard_frozen_training_matches_legacy_on_identical_batches(isolated, fp64, anchor, device):
    from types import SimpleNamespace

    from mace.modules.loss import WeightedEnergyForcesLoss
    from mace.tools.scripts_utils import get_params_options
    from mace_core.config.training import TrainingConfig
    from mace_torch.finetune.freeze import freeze
    from mace_torch.physics import DerivativeEngine
    from mace_torch.train.optimizers import build_optimizer
    from tests.golden.anchors import anchor_batch, load_training_structures
    from tests.golden.harness import FP64_CPU_REFERENCE
    from tests.parity.anchors import load_anchor
    from tests.parity.test_fm00_training_step import ENERGY, converted

    legacy = load_anchor(anchor).double()
    model, numbers = converted(legacy)
    engine = DerivativeEngine(model, ENERGY).to(device)
    legacy = legacy.to(device)
    config = TrainingConfig(lr=0.01)
    factors = freeze(engine, 6)
    config = config.model_copy(update={'scheduler': config.scheduler.model_copy(update={'group_factors': factors})})
    native_optimizer = build_optimizer(engine, config)
    options = SimpleNamespace(freeze=6, lr_params_factors='{}', lr=config.lr,
                              weight_decay=config.weight_decay, beta=0.9, amsgrad=True)
    legacy_optimizer = torch.optim.Adam(**get_params_options(options, legacy))
    native_average = ExponentialMovingAverage(engine.parameters(), decay=0.99)
    legacy_average = LegacyAverage(legacy.parameters(), decay=0.99)
    frozen = {name: value.detach().clone() for name,value in engine.named_parameters()
              if not value.requires_grad}
    readouts = {name: value.detach().clone() for name,value in engine.named_parameters()
                if value.requires_grad and value.numel()}
    structures = load_training_structures(limit=4)
    loss = WeightedEnergyForcesLoss(energy_weight=1.0, forces_weight=10.0).double()
    for step in range(6):
        batch = anchor_batch(legacy, structures[step % 2::2], torch.float64).to(device)
        graph = dict(batch.to_dict(), cell=batch.cell.reshape(-1, 3, 3),
                     node_atomic_numbers=torch.tensor(numbers, device=device)[batch.node_attrs.argmax(-1)])
        legacy_optimizer.zero_grad(set_to_none=True)
        native_optimizer.zero_grad(set_to_none=True)
        reference = legacy(batch.to_dict(), training=True, compute_force=True)
        result = engine(graph, compute=('forces',), training=True)
        native = {'energy': result.total_energy, 'forces': result.forces}
        for name in native:
            torch.testing.assert_close(native[name], reference[name],
                                       atol=FP64_CPU_REFERENCE.atol, rtol=FP64_CPU_REFERENCE.rtol)
        loss(batch, reference).backward()
        loss(batch, native).backward()
        for network, optimizer, average in [(legacy, legacy_optimizer, legacy_average),
                                            (engine, native_optimizer, native_average)]:
            torch.nn.utils.clip_grad_norm_(network.parameters(), config.clip_grad)
            optimizer.step()
            average.update()
        with legacy_average.average_parameters(), native_average.average_parameters():
            reference = legacy(batch.to_dict(), training=False, compute_force=True)
            result = engine(graph, compute=('forces',), training=False)
            for got, expected in [(result.total_energy, reference['energy']), (result.forces, reference['forces'])]:
                torch.testing.assert_close(got, expected, atol=FP64_CPU_REFERENCE.atol,
                                           rtol=FP64_CPU_REFERENCE.rtol)
    parameters = dict(engine.named_parameters())
    assert frozen and all(torch.equal(parameters[name], value) for name,value in frozen.items())
    assert readouts and any(not torch.equal(parameters[name], value) for name,value in readouts.items())
