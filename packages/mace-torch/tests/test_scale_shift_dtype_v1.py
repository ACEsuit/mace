"""The dtype each energy family hands out, against the frozen tree's.

The frozen tree returns the total energy in the model dtype for every family,
and the per-atom energies in float64 for its scale-shift model only among
these three: its plain and magnetic models keep them in the model dtype. So a
float32 scale-shift run gives a float32 total and a float64 decomposition from
the same call, and no single dtype reproduces both. The repository's legacy
characterization pins the frozen side; this file pins that the v1 families,
built as a run builds them, give the same dtypes.

The process default is varied by the suite's own fixture, and the answer must
not follow it: the precision is the configuration's, not the interpreter's.
"""

from __future__ import annotations

import pytest
import torch
from mace_core.config.precision import PrecisionConfig
from mace_core.config.resolved import ResolvedConfig
from mace_core.data.backend import DatasetStatistics
from mace_core.elements import AtomicNumberTable, ResolvedE0s
from mace_core.kernels.precision import Precision
from mace_core.observables import DEFAULT_CATALOGUE
from mace_torch.train.model_stage import build_model
from mace_torch_engine_fixtures import ATOMIC_NUMBERS, build_graph, molecule

#: family -> whether its per-atom energies are widened to the accumulation
#: dtype. The frozen tree's classes, in the same order: ``MACE``,
#: ``ScaleShiftMACE``, ``MagneticScaleShiftMACE``.
WIDENS = {"plain": False, "scale_shift": True, "magnetic": False}

_TORCH = {"float64": torch.float64, "float32": torch.float32}


def evaluate(family: str, model: Precision, supports_float64: bool = True):
    config = ResolvedConfig.model_validate(
        {
            "model": {
                "model": family,
                "observables": ["energy"],
                "r_max": 5.0,
                "num_channels": 4,
                "max_ell": 1,
                "hidden_irreps": "0e+1o",
            }
        }
    )
    engine, _ = build_model(
        config,
        DEFAULT_CATALOGUE,
        z_table=AtomicNumberTable(ATOMIC_NUMBERS),
        heads=("default",),
        e0s=ResolvedE0s({"default": {1: -13.6, 8: -2040.0}}),
        statistics=DatasetStatistics(avg_num_neighbors=3.0),
        precision=PrecisionConfig(model=model, accumulate="float64"),
        supports_float64=supports_float64,
    )
    dtype = _TORCH[model]
    graph = {
        key: value.to(dtype) if value.is_floating_point() else value
        for key, value in build_graph(*molecule()).items()
        if isinstance(value, torch.Tensor)
    } | {"num_graphs": 1}
    if family == "magnetic":
        graph["magmom"] = torch.zeros(len(molecule()[1]), 3, dtype=dtype)
    return engine, engine(graph, compute=("forces",))


@pytest.mark.parametrize("model", ["float64", "float32"])
@pytest.mark.parametrize("family", sorted(WIDENS))
def test_each_family_hands_out_the_frozen_trees_dtypes(family, model: Precision):
    _, out = evaluate(family, model)
    dtype = _TORCH[model]
    assert out.total_energy.dtype is dtype
    assert out.forces.dtype is dtype
    assert out.node_energies.dtype is (torch.float64 if WIDENS[family] else dtype)


@pytest.mark.parametrize("family", sorted(WIDENS))
def test_at_float64_the_per_atom_energies_add_up_to_the_total(family):
    _, out = evaluate(family, "float64")
    torch.testing.assert_close(
        out.node_energies.sum().reshape(1), out.total_energy, rtol=1e-12, atol=1e-12
    )


def test_without_float64_the_widening_degrades_to_the_model_dtype_and_says_so(
    caplog,
):
    """What ``safe_double`` does on a device with no float64: the tensor's
    own dtype, rather than an error at the first cast. The build logs it."""
    with caplog.at_level("WARNING", logger="mace_torch.train.model_stage"):
        engine, out = evaluate("scale_shift", "float32", supports_float64=False)
    assert any("no float64" in record.getMessage() for record in caplog.records)
    assert out.node_energies.dtype is torch.float32
    reports = [
        module.build_report()
        for module in engine.modules()
        if hasattr(module, "build_report")
    ]
    assert any(report and "no float64" in report for report in reports)
