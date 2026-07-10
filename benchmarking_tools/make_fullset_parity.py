from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from ase.io import read
from mace.calculators import MACECalculator


CHECKPOINT = Path(
    "checkpoints/dimer_full_L3_I2_v1_run-17_epoch-105.pt"
)
TEST_FILE = Path("test.xyz")
DEVICE = "cpu"
DTYPE = "float64"


def metrics(reference: np.ndarray, prediction: np.ndarray):
    error = prediction - reference
    mae = np.mean(np.abs(error))
    rmse = np.sqrt(np.mean(error**2))

    ss_res = np.sum(error**2)
    ss_tot = np.sum(
        (reference - np.mean(reference)) ** 2
    )
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan")

    return mae, rmse, r2


def parity_plot(
    reference: np.ndarray,
    prediction: np.ndarray,
    title: str,
    xlabel: str,
    ylabel: str,
    output: str,
):
    mae, rmse, r2 = metrics(reference, prediction)

    minimum = min(reference.min(), prediction.min())
    maximum = max(reference.max(), prediction.max())
    margin = 0.04 * max(maximum - minimum, 1.0e-12)

    fig, ax = plt.subplots(figsize=(7, 7))
    ax.scatter(reference, prediction, s=12, alpha=0.65)
    ax.plot(
        [minimum - margin, maximum + margin],
        [minimum - margin, maximum + margin],
        linestyle="--",
        linewidth=1.5,
    )

    ax.set_xlim(minimum - margin, maximum + margin)
    ax.set_ylim(minimum - margin, maximum + margin)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)

    ax.text(
        0.04,
        0.96,
        f"MAE = {mae:.6g}\n"
        f"RMSE = {rmse:.6g}\n"
        f"R² = {r2:.6g}",
        transform=ax.transAxes,
        va="top",
    )

    fig.tight_layout()
    fig.savefig(output, dpi=200)
    plt.close(fig)

    print(f"{title}")
    print(f"  MAE:  {mae:.8g}")
    print(f"  RMSE: {rmse:.8g}")
    print(f"  R²:   {r2:.8g}")
    print(f"  saved: {output}")


if not CHECKPOINT.exists():
    raise FileNotFoundError(
        f"Checkpoint not found: {CHECKPOINT}"
    )

frames = read(TEST_FILE, index=":")

calculator = MACECalculator(
    model_paths=str(CHECKPOINT),
    device=DEVICE,
    default_dtype=DTYPE,
)

energy_reference = []
energy_prediction = []
force_reference = []
force_prediction = []

for index, atoms in enumerate(frames, start=1):
    reference_energy = atoms.get_potential_energy()
    reference_forces = atoms.get_forces()

    atoms.calc = calculator

    predicted_energy = atoms.get_potential_energy()
    predicted_forces = atoms.get_forces()

    energy_reference.append(reference_energy / len(atoms))
    energy_prediction.append(predicted_energy / len(atoms))

    force_reference.append(reference_forces.reshape(-1))
    force_prediction.append(predicted_forces.reshape(-1))

    if index % 50 == 0 or index == len(frames):
        print(f"Evaluated {index}/{len(frames)} configurations")

energy_reference = np.asarray(energy_reference)
energy_prediction = np.asarray(energy_prediction)
force_reference = np.concatenate(force_reference)
force_prediction = np.concatenate(force_prediction)

parity_plot(
    energy_reference,
    energy_prediction,
    title="Test energy parity — dimer full L3 I2",
    xlabel="Reference energy per atom [eV]",
    ylabel="Predicted energy per atom [eV]",
    output="dimer_full_test_energy_parity.png",
)

parity_plot(
    force_reference,
    force_prediction,
    title="Test force parity — dimer full L3 I2",
    xlabel="Reference force component [eV / Å]",
    ylabel="Predicted force component [eV / Å]",
    output="dimer_full_test_force_parity.png",
)

np.savez(
    "dimer_full_test_parity_data.npz",
    energy_reference=energy_reference,
    energy_prediction=energy_prediction,
    force_reference=force_reference,
    force_prediction=force_prediction,
)

print("Created:")
print("  dimer_full_test_energy_parity.png")
print("  dimer_full_test_force_parity.png")
print("  dimer_full_test_parity_data.npz")
