"""Plot selection against the frozen renderer, comparing data before styling."""

import json

import matplotlib
import numpy as np
import pandas as pd
import pytest
from matplotlib.figure import Figure
from mace_torch.train.plots import plot_training

from tests.golden.harness import CLOSED_FORM_FP64


@pytest.mark.parametrize("min_epoch", [-1, 0, 2])
@pytest.mark.parametrize("head", ["water", "salt"])
def test_minimum_epoch_matches_legacy_for_each_head(tmp_path, monkeypatch, isolated, min_epoch, head):
    rows = []
    for seed in (1, 2):
        for epoch in range(4):
            rows.append(dict(name="run", seed=seed, epoch=epoch, mode="opt", loss=seed + epoch))
            for name, factor in (("water", 1), ("salt", 2)):
                rows.append(dict(name="run", seed=seed, epoch=epoch, mode="eval", head=name,
                                 loss=seed + epoch, rmse_e=0.01 * factor * (seed + epoch)))
    path = tmp_path / "metrics.jsonl"
    path.write_text("\n".join(json.dumps(row) for row in rows))
    figures = []
    original = Figure.savefig

    def save(figure, *args, **kwargs):
        figures.append(figure)
        return original(figure, *args, **kwargs)

    monkeypatch.setattr(Figure, "savefig", save)
    with matplotlib.rc_context():
        from mace.cli.plot_train import plot

        plot(pd.DataFrame(rows), min_epoch, str(tmp_path / "legacy"), "png",
             True, None, False, "rmse_e", head)
        plot_training([path], tmp_path / "native.png", keys=("rmse_e",),
                      min_epoch=min_epoch, heads=[head], linear=True)
    reference = figures[0].axes[1].lines[0]
    native = figures[1].axes[0].lines[0]
    np.testing.assert_array_equal(native.get_xdata(), reference.get_xdata())
    # Legacy presents errors in meV; V1 labels them in their recorded units.
    np.testing.assert_allclose(native.get_ydata(), reference.get_ydata() / 1000,
                               atol=CLOSED_FORM_FP64.atol, rtol=CLOSED_FORM_FP64.rtol)
