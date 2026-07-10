from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from ase.io import read
from mace.calculators import MACECalculator


MODEL = Path("checkpoints/quick_inertia_run-17.model")
TEST = Path("test.xyz")

if not MODEL.exists():
    raise FileNotFoundError(f"Model not found: {MODEL}")

frames = read(TEST, index=":")

calc = MACECalculator(
    model_paths=str(MODEL),
    device="cpu",
    default_dtype="float64",
)

energy_ref = []
energy_pred = []
force_ref = []
force_pred = []

for i, atoms in enumerate(frames):
    if atoms.calc is None:
        raise RuntimeError(f"Frame {i} has no reference ASE calculator")

    energy_ref.append(float(atoms.get_potential_energy()))
    force_ref.extend(
        np.asarray(atoms.get_forces(), dtype=float).reshape(-1)
    )

    predicted = atoms.copy()
    predicted.calc = calc

    energy_pred.append(float(predicted.get_potential_energy()))
    force_pred.extend(
        np.asarray(predicted.get_forces(), dtype=float).reshape(-1)
    )

energy_ref = np.asarray(energy_ref)
energy_pred = np.asarray(energy_pred)
force_ref = np.asarray(force_ref)
force_pred = np.asarray(force_pred)


def metrics(reference, prediction):
    error = prediction - reference

    mae = np.mean(np.abs(error))
    rmse = np.sqrt(np.mean(error**2))

    denominator = np.sum(
        (reference - np.mean(reference)) ** 2
    )

    r2 = (
        1.0 - np.sum(error**2) / denominator
        if denominator > 0.0
        else float("nan")
    )

    return mae, rmse, r2


def parity_plot(reference, prediction, title, label, filename):
    mae, rmse, r2 = metrics(reference, prediction)

    lower = min(reference.min(), prediction.min())
    upper = max(reference.max(), prediction.max())
    padding = 0.05 * max(upper - lower, 1.0e-12)

    lower -= padding
    upper += padding

    fig, ax = plt.subplots(figsize=(6, 6))

    ax.scatter(reference, prediction, s=22, alpha=0.65)
    ax.plot(
        [lower, upper],
        [lower, upper],
        linestyle="--",
        linewidth=1.5,
    )

    ax.set_xlim(lower, upper)
    ax.set_ylim(lower, upper)
    ax.set_aspect("equal", adjustable="box")

    ax.set_xlabel(f"Reference {label}")
    ax.set_ylabel(f"Predicted {label}")
    ax.set_title(title)

    ax.text(
        0.05,
        0.95,
        (
            f"MAE = {mae:.6g}\n"
            f"RMSE = {rmse:.6g}\n"
            f"R² = {r2:.6g}"
        ),
        transform=ax.transAxes,
        va="top",
    )

    fig.tight_layout()
    fig.savefig(filename, dpi=200)
    plt.close(fig)

    print(title)
    print(f"  MAE:  {mae:.8g}")
    print(f"  RMSE: {rmse:.8g}")
    print(f"  R²:   {r2:.8g}")


parity_plot(
    energy_ref,
    energy_pred,
    "Energy parity — held-out test set",
    "energy [reduced units]",
    "energy_parity.png",
)

parity_plot(
    force_ref,
    force_pred,
    "Force-component parity — held-out test set",
    "force component [reduced units]",
    "force_parity.png",
)

np.savez(
    "parity_data.npz",
    energy_reference=energy_ref,
    energy_prediction=energy_pred,
    force_reference=force_ref,
    force_prediction=force_pred,
)

print("\nCreated:")
print("  energy_parity.png")
print("  force_parity.png")
print("  parity_data.npz")
