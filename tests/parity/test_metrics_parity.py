"""The rewrite's errors against the frozen tree's, on one fixture, at fp64.

`MACELoss` and `RunningMetrics` are handed the same predictions and the same
references, and every metric both of them report has to agree. That is the
claim the error tables rest on: a table whose columns are the frozen tree's and
whose numbers are not is worse than no table.

They agree on every structure that carries a label, and they are meant to
disagree on one that does not. Both halves are here, because a comparison that
only used fully labelled data would pass against either implementation and say
nothing about the one thing this layer changed.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch
from mace_core.config.loss import LossConfig
from mace_core.data.configuration import Configuration
from mace_core.elements import AtomicNumberTable
from mace_core.observables import load_default_catalogue, resolve_requested
from mace_core.outputs import MACEOutput
from mace_torch.data import GraphDataset, collate_training, target_specs
from mace_torch.train import RunningMetrics, build_loss, metric_specs

CATALOGUE = load_default_catalogue()
REQUESTED = resolve_requested(["energy", "forces"], CATALOGUE)
SPECS = target_specs(REQUESTED)
Z_TABLE = AtomicNumberTable([1])

#: What the two stacks call the same quantity. The frozen tree abbreviates and
#: the rewrite does not, which is the only difference between these columns.
SHARED = {
    "mae_e": "mae_energy",
    "rmse_e": "rmse_energy",
    "q95_e": "q95_energy",
    "mae_e_per_atom": "mae_energy_per_atom",
    "rmse_e_per_atom": "rmse_energy_per_atom",
    "mae_f": "mae_forces",
    "rmse_f": "rmse_forces",
    "q95_f": "q95_forces",
    "rel_mae_f": "rel_mae_forces",
    "rel_rmse_f": "rel_rmse_forces",
}

#: Two atoms per structure, far enough apart that no edge exists. No model runs
#: in this file: the predictions are handed in, so what is compared is the two
#: accumulators and nothing upstream of them.
POSITIONS = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]])


def structures(count: int, labelled: int) -> list[Configuration]:
    """``count`` structures of which the first ``labelled`` carry values."""
    generator = np.random.default_rng(11)
    built = []
    for index in range(count):
        if index < labelled:
            properties = {
                "energy": float(-10.0 - index),
                "forces": generator.normal(scale=0.5, size=(2, 3)),
            }
        else:
            properties = {"energy": None, "forces": None}
        built.append(
            Configuration(
                atomic_numbers=np.array([1, 1]),
                positions=POSITIONS,
                properties=properties,
            )
        )
    return built


def v1_batch(configurations):
    dataset = GraphDataset(configurations, cutoff=0.5, z_table=Z_TABLE, targets=SPECS)
    return collate_training(
        [dataset[index] for index in range(len(configurations))], z_table=Z_TABLE
    )


def predictions(batch):
    """The same wrong answer for both stacks, drawn once."""
    generator = torch.Generator().manual_seed(3)
    energies = batch.targets["energy"].reshape(-1) + torch.randn(
        batch.targets["energy"].reshape(-1).shape,
        generator=generator,
        dtype=torch.float64,
    )
    forces = batch.targets["forces"] + torch.randn(
        batch.targets["forces"].shape, generator=generator, dtype=torch.float64
    )
    return energies, forces


class LegacyBatch:
    """The attributes `MACELoss.update` reads, and nothing else.

    A stub rather than an `AtomicData`: building one would drag in the frozen
    tree's graph construction, which is a different comparison and one that
    already has its own file.
    """

    def __init__(self, batch):
        counts = batch.graph["ptr"]
        self.ptr = counts
        self.num_graphs = int(batch.graph["num_graphs"])
        self.weight = batch.graph["weight"].to(torch.float64)
        self.energy_weight = batch.property_weights["energy"].to(torch.float64)
        self.forces_weight = batch.property_weights["forces"].to(torch.float64)
        self.energy = batch.targets["energy"].reshape(-1).to(torch.float64)
        self.forces = batch.targets["forces"].to(torch.float64)
        self.stress = None
        self.virials = None
        self.dipole = None
        self.polarizability = None
        self.magforces = None


def legacy_metrics(batch, energies, forces):
    from mace.tools.train import MACELoss

    class Zero(torch.nn.Module):
        def forward(self, pred, ref):
            return torch.tensor(0.0, dtype=torch.float64)

    metric = MACELoss(loss_fn=Zero())
    metric.update(LegacyBatch(batch), {"energy": energies, "forces": forces})
    _, aux = metric.compute()
    return aux


def v1_metrics(batch, energies, forces):
    metrics = RunningMetrics(
        metric_specs(REQUESTED), build_loss(REQUESTED, LossConfig())
    )
    metrics.update(MACEOutput(total_energy=energies, forces=forces), batch)
    return metrics.compute()


def both(count, labelled):
    batch = v1_batch(structures(count, labelled))
    energies, forces = predictions(batch)
    return (
        legacy_metrics(batch, energies, forces),
        v1_metrics(batch, energies, forces),
    )


# ---------------------------------------------------------------------------
# Where the two agree
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(("legacy_name", "v1_name"), sorted(SHARED.items()))
def test_every_shared_metric_agrees_on_a_fully_labelled_set(
    fp64, isolated, legacy_name, v1_name
):
    """Six structures, all labelled. Nothing is masked on either side, so the
    two are computing the same average over the same rows."""
    legacy, v1 = both(6, 6)
    assert v1[v1_name] == pytest.approx(legacy[legacy_name], rel=1e-12, abs=1e-12)


def test_the_comparison_is_not_trivially_satisfied(fp64, isolated):
    """The guard: the residuals are real, so an implementation returning zero
    everywhere would not pass the parametrized test above."""
    _, v1 = both(6, 6)
    assert v1["rmse_forces"] > 0.1
    assert v1["rel_rmse_forces"] > 1.0


# ---------------------------------------------------------------------------
# Where they are meant to differ
# ---------------------------------------------------------------------------


def test_the_totals_still_agree_when_a_structure_carries_no_label(fp64, isolated):
    """The frozen tree does mask these, so they are the control."""
    legacy, v1 = both(6, 3)
    for legacy_name, v1_name in (("mae_e", "mae_energy"), ("rmse_e", "rmse_energy")):
        assert v1[v1_name] == pytest.approx(legacy[legacy_name], rel=1e-12)


@pytest.mark.parametrize(
    ("legacy_name", "v1_name"),
    [
        ("mae_e_per_atom", "mae_energy_per_atom"),
        ("rmse_e_per_atom", "rmse_energy_per_atom"),
    ],
)
def test_the_per_atom_energy_error_differs(fp64, isolated, legacy_name, v1_name):
    """`mace/tools/train.py:659-663` appends both the total and the per-atom
    delta and passes only the total to the filter, so a structure with no
    energy contributes a fabricated per-atom row. `rmse_e_per_atom` is the
    default error table's energy column."""
    legacy, v1 = both(6, 3)
    assert v1[v1_name] != pytest.approx(legacy[legacy_name], rel=1e-6)


@pytest.mark.parametrize(
    ("legacy_name", "v1_name"),
    [("rel_mae_f", "rel_mae_forces"), ("rel_rmse_f", "rel_rmse_forces")],
)
def test_the_relative_force_error_differs(fp64, isolated, legacy_name, v1_name):
    """`:666-667` appends the references before the filter runs and never masks
    them, so the denominator is diluted by every row the model was not asked to
    fit and the reported error is inflated."""
    legacy, v1 = both(6, 3)
    assert legacy[legacy_name] > v1[v1_name]


def test_the_per_atom_error_is_the_labelled_structures_alone(fp64, isolated):
    """Which of the two is right: the rewrite's number over a half-labelled
    set is the frozen tree's over the labelled half on its own."""
    _, half = both(6, 3)
    legacy_on_the_labelled_half, _ = both(3, 3)
    assert half["rmse_energy_per_atom"] == pytest.approx(
        legacy_on_the_labelled_half["rmse_e_per_atom"], rel=1e-12
    )
