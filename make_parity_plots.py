from pathlib import Path
import argparse

import numpy as np
import matplotlib.pyplot as plt
from ase.io import read
from mace.calculators import MACECalculator

parser = argparse.ArgumentParser()
parser.add_argument("--model", required=True)
parser.add_argument("--data", required=True)
parser.add_argument("--output_prefix", required=True)
parser.add_argument("--device", default="cpu")
parser.add_argument("--default_dtype", default="float64")
args = parser.parse_args()

model_path = Path(args.model)
data_path = Path(args.data)
prefix = Path(args.output_prefix)
prefix.parent.mkdir(parents=True, exist_ok=True)

frames = read(data_path, index=":")

calc = MACECalculator(
    model_paths=str(model_path),
    device=args.device,
    default_dtype=args.default_dtype,
)

e_ref = []
e_pred = []

f_ref = []
f_pred = []

for i, atoms in enumerate(frames):
    ref_energy = float(atoms.get_potential_energy())
    ref_forces = np.asarray(atoms.get_forces(), dtype=float)

    atoms_pred = atoms.copy()
    calc.reset()
    atoms_pred.calc = calc

    pred_energy = float(atoms_pred.get_potential_energy())
    pred_forces = np.asarray(atoms_pred.get_forces(), dtype=float)

    natoms = len(atoms)

    e_ref.append(ref_energy / natoms)
    e_pred.append(pred_energy / natoms)

    f_ref.append(ref_forces.reshape(-1))
    f_pred.append(pred_forces.reshape(-1))

e_ref = np.asarray(e_ref)
e_pred = np.asarray(e_pred)

f_ref = np.concatenate(f_ref)
f_pred = np.concatenate(f_pred)

e_err = e_pred - e_ref
f_err = f_pred - f_ref

e_mae = np.mean(np.abs(e_err))
e_rmse = np.sqrt(np.mean(e_err**2))
f_mae = np.mean(np.abs(f_err))
f_rmse = np.sqrt(np.mean(f_err**2))

def r2_score(y_true, y_pred):
    ss_res = np.sum((y_pred - y_true) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    return 1.0 - ss_res / ss_tot if ss_tot > 0 else np.nan

e_r2 = r2_score(e_ref, e_pred)
f_r2 = r2_score(f_ref, f_pred)

# Energy parity
emin = min(np.min(e_ref), np.min(e_pred))
emax = max(np.max(e_ref), np.max(e_pred))

plt.figure(figsize=(6, 6))
plt.scatter(e_ref, e_pred, s=18, alpha=0.8)
plt.plot([emin, emax], [emin, emax], linestyle="--")
plt.xlabel("Reference energy (eV/atom)")
plt.ylabel("Predicted energy (eV/atom)")
plt.title(
    f"Energy parity\n"
    f"MAE={e_mae*1000:.3f} meV/atom, "
    f"RMSE={e_rmse*1000:.3f} meV/atom, "
    f"R²={e_r2:.4f}"
)
plt.tight_layout()
energy_png = prefix.with_name(prefix.name + "_energy_parity.png")
plt.savefig(energy_png, dpi=200)
plt.close()

# Force parity
fmin = min(np.min(f_ref), np.min(f_pred))
fmax = max(np.max(f_ref), np.max(f_pred))

plt.figure(figsize=(6, 6))
plt.scatter(f_ref, f_pred, s=8, alpha=0.35)
plt.plot([fmin, fmax], [fmin, fmax], linestyle="--")
plt.xlabel("Reference force component (eV/A)")
plt.ylabel("Predicted force component (eV/A)")
plt.title(
    f"Force parity\n"
    f"MAE={f_mae*1000:.3f} meV/A, "
    f"RMSE={f_rmse*1000:.3f} meV/A, "
    f"R²={f_r2:.4f}"
)
plt.tight_layout()
force_png = prefix.with_name(prefix.name + "_force_parity.png")
plt.savefig(force_png, dpi=200)
plt.close()

# Save raw numbers
np.savez(
    prefix.with_name(prefix.name + "_parity_data.npz"),
    e_ref=e_ref,
    e_pred=e_pred,
    f_ref=f_ref,
    f_pred=f_pred,
)

print("Wrote:")
print(" ", energy_png)
print(" ", force_png)
print(" ", prefix.with_name(prefix.name + "_parity_data.npz"))
print()
print("Energy:")
print(f"  MAE  = {e_mae*1000:.6f} meV/atom")
print(f"  RMSE = {e_rmse*1000:.6f} meV/atom")
print(f"  R2   = {e_r2:.6f}")
print()
print("Forces:")
print(f"  MAE  = {f_mae*1000:.6f} meV/A")
print(f"  RMSE = {f_rmse*1000:.6f} meV/A")
print(f"  R2   = {f_r2:.6f}")
