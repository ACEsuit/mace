"""TRN-5's chunked objective equals legacy's loss on the concatenated set.

Legacy's chunk weighting is deliberately replaced by TRN-5 for mixed atom
counts. The oracle here is its loss evaluated on the whole set, including
every parameter gradient through the canonical conversion's Jacobian.
"""

import pytest
import torch
from mace.modules.loss import WeightedEnergyForcesLoss
from mace_torch.data import TrainingBatch
from mace_torch.physics import DerivativeEngine
from mace_torch.train.ddp import DistributedContext
from mace_torch.train.full_batch import accumulate, set_totals
from mace_torch.train.loss import GeneratedLoss, LossTerm

from tests.golden.anchors import anchor_batch, load_training_structures
from tests.golden.harness import CLOSED_FORM_FP64
from tests.parity.anchors import load_anchor
from tests.parity.test_fm00_training_step import (
    ENERGY,
    assert_all_parameter_gradients,
    converted,
)


@pytest.mark.parametrize("anchor", ["tiny_mace.model", "tiny_scaleshift.model"])
@pytest.mark.parametrize("batch_size", [1, 2, 4])
def test_chunked_loss_and_all_gradients_match_legacy_whole_set(
    isolated, fp64, anchor, batch_size
):
    legacy = load_anchor(anchor).double()
    model, numbers = converted(legacy)
    structures = load_training_structures(limit=5)
    # Synthetic smaller configurations retain the sliced force labels. Their
    # energies are test targets; no new physical reference is being generated.
    for index in (1, 3):
        structures[index] = structures[index][:2]
        structures[index].info["REF_energy"] *= 0.5
    assert len({len(atoms) for atoms in structures}) > 1
    whole = anchor_batch(legacy, structures)
    reference = WeightedEnergyForcesLoss(energy_weight=1.0, forces_weight=10.0)
    expected = reference(whole, legacy(whole.to_dict(), training=True, compute_force=True))
    expected.backward()

    parts = []
    for start in range(0, len(structures), batch_size):
        batch = anchor_batch(legacy, structures[start:start + batch_size])
        parts.append(TrainingBatch(
            graph=dict(batch.to_dict(), cell=batch.cell.reshape(-1, 3, 3),
                       node_atomic_numbers=torch.tensor(numbers)[batch.node_attrs.argmax(-1)]),
            targets={"energy": batch.energy, "forces": batch.forces},
            property_weights={"energy": batch.energy_weight, "forces": batch.forces_weight},
        ))
    loss = GeneratedLoss([
        LossTerm("energy", 1.0, per_atom=False, extensive=True),
        LossTerm("forces", 10.0, per_atom=True, extensive=False),
    ])
    actual = accumulate(DerivativeEngine(model, ENERGY), parts, loss,
                        set_totals(loss, parts, DistributedContext()))
    torch.testing.assert_close(actual, expected.detach(), atol=CLOSED_FORM_FP64.atol,
                               rtol=CLOSED_FORM_FP64.rtol)
    assert_all_parameter_gradients(legacy, model)
