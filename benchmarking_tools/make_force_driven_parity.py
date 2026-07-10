from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from ase.io import read
from mace.calculators import MACECalculator

MODEL = Path("checkpoints/inertia_full_tensor_overfit64_run-17.model")
DATA = Path("overfit64.xyz")

if not MODEL.exists():
    raise FileNotFoundError(f"Model not found: {MODEL}")

if not DATA.exists():
    raise FileNotFoundError(f"Dataset not found: {DATA}")

frames = read(DATA, index=":")

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
    ref_energy = float(atoms.get_potential_energy())
    ref_forces = np.asarray(atoms.get_forces(), dtype=float)

    pred_atoms = atoms.copy()
    pred_atoms.calc = calc

    pred_energy = float(pred_atoms.get_potential_energy())
    pred_forces = np.asarray(pred_atoms.get_forces(), dtype=float)

    energy_ref.append(ref_energy)
    energy_pred.append(pred_energy)

    force_ref.extend(ref_forces.reshape(-1))
    force_pred.extend(pred_forces.reshape(-1))

energy_ref = np.asarray(energy_ref)
energy_pred = np.asarray(energy_pred)
force_ref = np.asarray(force_ref)
force_pred = np.asarray(force_pred)


def metrics(ref, pred):
    err = pred - ref
    mae = np.mean(np.abs(err))
    rmse = np.sqrt(np.mean(err**2))
    denom = np.sum((ref - np.mean(ref)) ** 2)
    r2 = 1.0 - np.sum(err**2) / denom if denom > 0 else float("nan")
    return mae, rmse, r2


def make_parity_plot(ref, pred, title, xlabel, ylabel, outfile):
    mae, rmse, r2 = metrics(ref, pred)

    lo = min(ref.min(), pred.min())
    hi = max(ref.max(), pred.max())
    pad = 0.05 * max(hi - lo, 1.0e-12)
    lo -= pad
    hi += pad

    fig, ax = plt.subplots(figsize=(6, 6))
    ax.scatter(ref, pred, s=20, alpha=0.7)
    ax.plot([lo, hi], [lo, hi], "--", linewidth=1.5)

    ax.set_xlim(lo, hi)
    ax.set_ylim(lo, hi)
    ax.set_aspect("equal", adjustable="box")

    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)

    ax.text(
        0.05,
        0.95,
        f"MAE = {mae:.6g}\nRMSE = {rmse:.6g}\nR² = {r2:.6g}",
        transform=ax.transAxes,
        va="top",
    )

    fig.tight_layout()
    fig.savefig(outfile, dpi=200)
    plt.close(fig)

    print(title)
    print(f"  MAE:  {mae:.8g}")
    print(f"  RMSE: {rmse:.8g}")
    print(f"  R²:   {r2:.8g}")
    print(f"  saved: {outfile}")
    print()


make_parity_plot(
    energy_ref,
    energy_pred,
    title="Energy parity — force-driven full-tensor run",
    xlabel="Reference energy [reduced units]",
    ylabel="Predicted energy [reduced units]",
    outfile="energy_parity_force_driven.png",
)

make_parity_plot(
    force_ref,
    force_pred,
    title="Force-component parity — force-driven full-tensor run",
    xlabel="Reference force component [reduced units]",
    ylabel="Predicted force component [reduced units]",
    outfile="force_parity_force_driven.png",
)

np.savez(
    "force_driven_parity_data.npz",
    energy_reference=energy_ref,
    energy_prediction=energy_pred,
    force_reference=force_ref,
    force_prediction=force_pred,
)

print("Created:")
print("  energy_parity_force_driven.png")
print("  force_parity_force_driven.png")
print("  force_driven_parity_data.npz")
