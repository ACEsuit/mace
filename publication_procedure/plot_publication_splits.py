from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


manifest = Path(
    "publication_splits/publication_manifest.csv"
)
df = pd.read_csv(manifest)

for x_column, output in (
    ("separation", "split_separation.png"),
    ("energy", "split_energy.png"),
    ("force_rms", "split_force_rms.png"),
    ("torque_rms", "split_torque_rms.png"),
):
    fig, ax = plt.subplots(figsize=(7, 5))

    for split in ("train", "valid", "test"):
        values = df.loc[
            df["split_random"] == split,
            x_column,
        ]

        ax.hist(
            values,
            bins=100,
            density=True,
            histtype="step",
            linewidth=1.5,
            label=f"{split} (n={len(values)})",
        )

    ax.set_xlabel(x_column.replace("_", " "))
    ax.set_ylabel("Density")
    ax.legend()
    fig.tight_layout()
    fig.savefig(
        Path("publication_splits") / output,
        dpi=200,
    )
    plt.close(fig)

print("Created random-split distribution plots.")
