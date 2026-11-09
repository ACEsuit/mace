"""Optimizer regularization follows the frozen parameter-group contract."""

from types import SimpleNamespace

import pytest
import torch
from mace_core.config.training import TrainingConfig
from mace_torch.physics import DerivativeEngine
from mace_torch.train.optimizers import build_optimizer

from tests.parity.anchors import load_anchor
from tests.parity.test_fm00_training_step import ENERGY, converted


@pytest.mark.parametrize("anchor", ["tiny_mace.model", "tiny_scaleshift.model"])
@pytest.mark.parametrize("kind", ["adam", "adamw"])
@pytest.mark.parametrize("device", ["cpu", pytest.param("cuda", marks=pytest.mark.gpu)])
def test_skip_weights_are_not_regularized(isolated, fp64, anchor, kind, device):
    from mace.tools.scripts_utils import get_params_options

    legacy = load_anchor(anchor).double()
    model, _ = converted(legacy)
    engine = DerivativeEngine(model, ENERGY).to(device)
    legacy = legacy.to(device)
    config = TrainingConfig(optimizer={"kind": kind})
    native_optimizer = build_optimizer(engine, config)
    options = SimpleNamespace(
        freeze=0,
        lr_params_factors="{}",
        lr=config.lr,
        weight_decay=config.weight_decay,
        beta=0.9,
        amsgrad=True,
    )
    legacy_optimizer = {"adam": torch.optim.Adam, "adamw": torch.optim.AdamW}[kind](
        **get_params_options(options, legacy)
    )
    weights = [block.skip_tp.weight for block in legacy.interactions]
    weights += [block.skip.weight for block in model.backbone.interactions]
    before = [weight.detach().clone() for weight in weights]
    assert all(weight.numel() and torch.any(weight != 0) for weight in weights)
    # Zero data gradients isolate regularization from a change in coordinates.
    for weight in weights:
        weight.grad = torch.zeros_like(weight)
    legacy_optimizer.step()
    native_optimizer.step()
    for weight, original in zip(weights, before, strict=True):
        assert torch.equal(weight, original)


@pytest.mark.parametrize("device", ["cpu", pytest.param("cuda", marks=pytest.mark.gpu)])
@pytest.mark.parametrize("correlation,basis", [(1, "reduced"), (3, "full")])
def test_regularized_updates_match_in_the_same_coordinates(
    isolated, fp64, device, correlation, basis
):
    from mace_core.config.model import ModelConfig
    from mace_core.observables import DEFAULT_CATALOGUE

    from mace.modules.loss import WeightedEnergyForcesStressLoss
    from mace.tools.scripts_utils import get_params_options
    from tests.golden.anchors import anchor_batch, load_training_structures
    from tests.golden.harness import FP64_CPU_REFERENCE
    from tests.parity.test_backbone_parity import _legacy_model

    # Full coordinates are a diagnostic control, not the new-training contract.
    architecture = ModelConfig(
        num_channels=8,
        max_L=0,
        max_ell=2,
        correlation=correlation,
        interaction_first="plain",
    )
    legacy = _legacy_model(architecture, avg_num_neighbors=2.0, seed=5).double()
    model, numbers = converted(legacy, basis=basis)
    if basis == "full":
        _align_null_components(legacy, model)
    engine = DerivativeEngine(model, DEFAULT_CATALOGUE.observable("energy")).to(device)
    legacy = legacy.to(device)
    config = TrainingConfig(weight_decay=0.02)
    native_optimizer = build_optimizer(engine, config)
    options = SimpleNamespace(
        freeze=0,
        lr_params_factors="{}",
        lr=config.lr,
        weight_decay=config.weight_decay,
        beta=0.9,
        amsgrad=True,
    )
    legacy_optimizer = torch.optim.Adam(**get_params_options(options, legacy))
    loss = (
        WeightedEnergyForcesStressLoss(
            energy_weight=1.0, forces_weight=10.0, stress_weight=1.0
        )
        .double()
        .to(device)
    )
    structures = load_training_structures(limit=4)
    for step in range(6):
        batch = anchor_batch(legacy, structures[step % 2 :: 2], torch.float64).to(
            device
        )
        graph = dict(
            batch.to_dict(),
            cell=batch.cell.reshape(-1, 3, 3),
            node_atomic_numbers=torch.tensor(numbers, device=device)[
                batch.node_attrs.argmax(-1)
            ],
        )
        legacy_optimizer.zero_grad(set_to_none=True)
        native_optimizer.zero_grad(set_to_none=True)
        reference = legacy(
            batch.to_dict(), training=True, compute_force=True, compute_stress=True
        )
        result = engine(graph, compute=("forces", "stress"), training=True)
        native = {
            "energy": result.total_energy,
            "forces": result.forces,
            "stress": result.stress,
        }
        for key in native:
            torch.testing.assert_close(
                native[key],
                reference[key],
                atol=FP64_CPU_REFERENCE.atol,
                rtol=FP64_CPU_REFERENCE.rtol,
            )
        loss(batch, reference).backward()
        loss(batch, native).backward()
        for network, optimizer in [
            (legacy, legacy_optimizer),
            (engine, native_optimizer),
        ]:
            torch.nn.utils.clip_grad_norm_(network.parameters(), config.clip_grad)
            optimizer.step()
    reference = legacy(batch.to_dict(), compute_force=True, compute_stress=True)
    result = engine(graph, compute=("forces", "stress"))
    for actual, expected in [
        (result.total_energy, reference["energy"]),
        (result.forces, reference["forces"]),
        (result.stress, reference["stress"]),
    ]:
        torch.testing.assert_close(
            actual, expected, atol=FP64_CPU_REFERENCE.atol, rtol=FP64_CPU_REFERENCE.rtol
        )


def _align_null_components(legacy, model):
    """Use the same full-basis representative without changing the function.

    The converter removes null components even for a full-basis destination.
    Adam's coupled decay uses those components, although inference does not.
    """
    from tests.golden.harness import CLOSED_FORM_FP64
    from tests.parity.legacy_weights import (
        canonical_contraction_bases,
        legacy_contraction_blocks,
        legacy_to_reference_map,
    )

    removed = []
    with torch.no_grad():
        for source, target in zip(
            legacy.products, model.backbone.products, strict=True
        ):
            for position, contraction in enumerate(
                source.symmetric_contractions.contractions
            ):
                bases = canonical_contraction_bases(
                    target.contraction.kernel_spec, position
                )
                for coupling, weight, order in legacy_contraction_blocks(contraction):
                    mapping = torch.from_numpy(
                        legacy_to_reference_map(coupling, bases[order - 1], order)
                    )
                    canonical = torch.einsum("pq,zpc->zqc", mapping, weight)
                    projected = torch.einsum(
                        "pq,zqc->zpc", torch.linalg.pinv(mapping.T), canonical
                    )
                    torch.testing.assert_close(
                        torch.einsum("pq,zpc->zqc", mapping, projected),
                        canonical,
                        atol=CLOSED_FORM_FP64.atol,
                        rtol=CLOSED_FORM_FP64.rtol,
                    )
                    removed.append(float((weight - projected).abs().max()))
                    weight.copy_(projected)
    assert max(removed) > CLOSED_FORM_FP64.atol
