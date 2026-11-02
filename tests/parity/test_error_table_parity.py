"""All table cells from independent accumulators, then the live renderers.

Predictions are supplied directly: model physics is covered by its own parity
suites. Only legacy's evaluation call is substituted, with metrics produced by
its real accumulator, so formatting does not rerun a model for every table.
"""

from types import SimpleNamespace

import numpy as np
import pytest
import torch
from mace_core.outputs import MACEOutput
from mace_core.tables import TABLE_TYPES, Row, error_table
from mace_torch.data import TrainingBatch
from mace_torch.train.metrics import MetricSpec, RunningMetrics

from tests.golden.harness import CLOSED_FORM_FP64


class ZeroLoss(torch.nn.Module):
    def forward(self, pred, ref):
        return torch.zeros((), dtype=torch.float64)


@pytest.fixture
def measured(fp64, isolated):
    from mace.tools.train import MACELoss

    specs = [
        MetricSpec("energy", False, True),
        MetricSpec("forces", True, False),
        MetricSpec("stress", False, False),
        MetricSpec("virials", False, True),
        MetricSpec("dipole", False, True),
        MetricSpec("polarizability", False, True),
    ]
    legacy = MACELoss(ZeroLoss())
    native = RunningMetrics(specs, ZeroLoss())
    generator = torch.Generator().manual_seed(27)
    for counts in ([2, 3], [4]):
        sizes = torch.tensor(counts)
        graph = {
            "ptr": torch.cat([torch.zeros(1, dtype=torch.long), sizes.cumsum(0)]),
            "batch": torch.arange(len(counts)).repeat_interleave(sizes),
            "weight": torch.ones(len(counts), dtype=torch.float64),
        }
        shapes = {
            "energy": (len(counts),),
            "forces": (sum(counts), 3),
            "stress": (len(counts), 3, 3),
            "virials": (len(counts), 3, 3),
            "dipole": (len(counts), 3),
            "polarizability": (len(counts), 3, 3),
        }
        targets = {
            name: torch.randn(shape, generator=generator)
            for name, shape in shapes.items()
        }
        predictions = {
            name: value + 0.02 * torch.randn(value.shape, generator=generator)
            for name, value in targets.items()
        }
        weights = {name: torch.ones(len(counts)) for name in targets}
        batch = TrainingBatch(graph, targets, weights)
        native.update(
            MACEOutput(
                total_energy=predictions["energy"],
                forces=predictions["forces"],
                stress=predictions["stress"],
                virials=predictions["virials"],
                dipole=predictions["dipole"],
                extras={"polarizability": predictions["polarizability"]},
            ),
            batch,
        )
        old_weights = {
            name + "_weight": value.reshape(-1, *([1] * (len(shapes[name]) - 1)))
            if name in ("dipole", "polarizability")
            else value
            for name, value in weights.items()
        }
        legacy.update(
            SimpleNamespace(
                **targets,
                **old_weights,
                ptr=graph["ptr"],
                weight=graph["weight"],
                num_graphs=len(counts),
                magforces=None,
            ),
            predictions,
        )
    return legacy.compute()[1], native.compute()


@pytest.mark.parametrize("kind", sorted(TABLE_TYPES))
def test_all_table_cells_match_the_live_oracle(measured, monkeypatch, kind):
    from mace.tools import tables_utils

    legacy, native = measured
    # Compare the quantities before rounding as well as the printed table.
    aliases = {"e": "energy", "f": "forces", "mu": "dipole"}
    for key, value in legacy.items():
        native_key = "_".join(aliases.get(part, part) for part in key.split("_"))
        np.testing.assert_allclose(
            native[native_key],
            value,
            atol=CLOSED_FORM_FP64.atol,
            rtol=CLOSED_FORM_FP64.rtol,
        )
    monkeypatch.setattr(tables_utils, "evaluate", lambda *args, **kwargs: (0.0, legacy))
    expected = tables_utils.create_error_table(
        kind,
        {"valid_water": object()},
        None,
        None,
        {},
        False,
        "cpu",
    )
    actual = error_table(kind, [Row("valid_water", "water", native)])
    cells = [
        [cell.strip() for cell in line.strip("|").split("|")]
        for line in actual.splitlines()
        if line.startswith("|")
    ]
    assert cells[0] == expected.field_names
    assert cells[1] == [str(cell).strip() for cell in expected.rows[0]]
