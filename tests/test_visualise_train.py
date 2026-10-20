import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")

from mace.cli.visualise_train import plot_epoch_dependence


def test_plot_epoch_dependence_skips_missing_metric():
    data = pd.DataFrame(
        [
            {"mode": "opt", "epoch": 1, "head": "head", "loss": 1.0},
            {
                "mode": "eval",
                "epoch": 1,
                "head": "head",
                "loss": 1.0,
                "rmse_e_per_atom": 0.1,
                "rmse_stress": None,
            },
        ]
    )
    fig, axes = matplotlib.pyplot.subplots(1, 2)

    plot_epoch_dependence(
        axes,
        data,
        "head",
        "1",
        [
            ("rmse_e_per_atom", "RMSE E/atom [meV]"),
            ("rmse_stress", "RMSE Stress [meV / A^3]"),
        ],
    )

    assert len(axes[1].lines) == 2  # energy error and loaded-model marker
    assert not np.isnan(axes[1].lines[0].get_ydata()).any()
    matplotlib.pyplot.close(fig)
