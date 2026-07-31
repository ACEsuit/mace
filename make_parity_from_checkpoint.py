from pathlib import Path
import inspect

import matplotlib.pyplot as plt
import numpy as np
import torch
from ase.io import read
from e3nn import o3

from mace.data import AtomicData
from mace.data.utils import config_from_atoms
from mace.modules.blocks import interaction_classes
from mace.modules.models import ScaleShiftMACE
from mace.tools import AtomicNumberTable


# -----------------------------
# User settings
# -----------------------------
CHECKPOINT = Path(
    "checkpoints/energyfit_moi_128_ew100_fw1_seed29_run-29_epoch-23.pt"
)
DATA_FILE = Path("publication_splits/random_test.xyz")
OUTPUT_PREFIX = Path("energy_overfit/parity/energyfit_moi_128_ew100_fw1_seed29_test")
DEVICE = "cpu"
DTYPE = torch.float64

# Architecture used for this run
R_MAX = 5.0
NUM_RADIAL_BASIS = 16
NUM_CUTOFF_BASIS = 5
NUM_INTERACTIONS = 2
NUM_CHANNELS = 64
MAX_L = 3
CORRELATION = 3
RIGID_FEATURE_MODE = "moi"

INTERACTION_NAME = "RealAgnosticResidualInteractionBlock"


# -----------------------------
# Helpers
# -----------------------------
def extract_state_dict(checkpoint_obj):
    if isinstance(checkpoint_obj, dict):
        for key in ("model", "model_state_dict", "state_dict"):
            if key in checkpoint_obj:
                value = checkpoint_obj[key]
                if isinstance(value, dict):
                    return value
                if hasattr(value, "state_dict"):
                    return value.state_dict()
    if hasattr(checkpoint_obj, "state_dict"):
        return checkpoint_obj.state_dict()
    raise RuntimeError(
        f"Could not extract a state_dict from checkpoint of type {type(checkpoint_obj)}"
    )


def build_hidden_irreps(num_channels: int, max_l: int) -> o3.Irreps:
    parts = []
    for l in range(max_l + 1):
        parity = "e" if l % 2 == 0 else "o"
        parts.append(f"{num_channels}x{l}{parity}")
    return o3.Irreps(" + ".join(parts))


def instantiate_model(z_table):
    interaction = interaction_classes[INTERACTION_NAME]
    hidden_irreps = build_hidden_irreps(NUM_CHANNELS, MAX_L)
    mlp_irreps = o3.Irreps("16x0e")

    # Candidate kwargs; we filter them against the installed constructor signature.
    candidate_kwargs = {
        "r_max": R_MAX,
        "num_bessel": NUM_RADIAL_BASIS,
        "num_polynomial_cutoff": NUM_CUTOFF_BASIS,
        "max_ell": MAX_L,
        "interaction_cls_first": interaction,
        "interaction_cls": interaction,
        "num_interactions": NUM_INTERACTIONS,
        "num_elements": len(z_table),
        "hidden_irreps": hidden_irreps,
        "MLP_irreps": mlp_irreps,
        "atomic_energies": [[0.0] * len(z_table)],
        "avg_num_neighbors": 1.0,
        "atomic_numbers": z_table.zs,
        "correlation": CORRELATION,
        "gate": torch.nn.functional.silu,
        "atomic_inter_scale": 1.0,
        "atomic_inter_shift": 0.0,
        "pair_repulsion": False,
        "distance_transform": None,
        "radial_type": "bessel",
        "heads": ["Default"],
        "cueq_config": None,
        "rigid_feature_mode": RIGID_FEATURE_MODE,
    }

    sig = inspect.signature(ScaleShiftMACE.__init__)
    kwargs = {}

    for name, param in sig.parameters.items():
        if name == "self":
            continue
        if name in candidate_kwargs:
            kwargs[name] = candidate_kwargs[name]

    print("Instantiating ScaleShiftMACE with kwargs:")
    for key in sorted(kwargs):
        print(f"  {key} = {kwargs[key]!r}")

    model = ScaleShiftMACE(**kwargs)
    return model


def atoms_to_model_input(atoms, z_table):
    config = config_from_atoms(atoms)
    data = AtomicData.from_config(
        config,
        z_table=z_table,
        cutoff=R_MAX,
    )

    # Single-graph batch metadata
    data.batch = torch.zeros(len(atoms), dtype=torch.long)
    data.ptr = torch.tensor([0, len(atoms)], dtype=torch.long)
    data.head = torch.zeros(1, dtype=torch.long)

    data = data.to(DEVICE)
    return data.to_dict()


def r2_score(y_true, y_pred):
    ss_res = np.sum((y_pred - y_true) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    if ss_tot == 0:
        return np.nan
    return 1.0 - ss_res / ss_tot


# -----------------------------
# Load data and model
# -----------------------------
print(f"Reading frames from: {DATA_FILE}")
frames = read(DATA_FILE, index=":")
print(f"Loaded {len(frames)} frames")

all_atomic_numbers = sorted(
    {
        int(z)
        for atoms in frames
        for z in atoms.get_atomic_numbers().tolist()
    }
)
z_table = AtomicNumberTable(all_atomic_numbers)
print("Atomic number table:", z_table.zs)

print(f"Loading checkpoint: {CHECKPOINT}")
checkpoint = torch.load(
    CHECKPOINT,
    map_location=DEVICE,
    weights_only=False,
)
state_dict = extract_state_dict(checkpoint)
print(f"Extracted state_dict with {len(state_dict)} tensors")

model = instantiate_model(z_table)
model = model.to(device=DEVICE)
model = model.to(dtype=DTYPE)

missing, unexpected = model.load_state_dict(state_dict, strict=False)

print("\nload_state_dict result:")
print("  missing keys:", missing)
print("  unexpected keys:", unexpected)

model.eval()

# -----------------------------
# Predict
# -----------------------------
e_ref = []
e_pred = []

f_ref = []
f_pred = []

with torch.no_grad():
    for index, atoms in enumerate(frames):
        ref_energy = float(atoms.get_potential_energy())
        ref_forces = np.asarray(atoms.get_forces(), dtype=np.float64)

        model_input = atoms_to_model_input(atoms, z_table)

        output = model(
            model_input,
            compute_force=True,
            compute_virials=False,
            compute_stress=False,
            compute_displacement=False,
        )

        pred_energy = float(output["energy"].detach().cpu().reshape(-1)[0])
        pred_forces = output["forces"].detach().cpu().numpy()

        natoms = len(atoms)

        e_ref.append(ref_energy / natoms)
        e_pred.append(pred_energy / natoms)

        f_ref.append(ref_forces.reshape(-1))
        f_pred.append(pred_forces.reshape(-1))

        if index == 0:
            print("\nFirst frame sanity check:")
            print("  ref energy / atom :", e_ref[-1])
            print("  pred energy / atom:", e_pred[-1])
            print("  ref force shape   :", ref_forces.shape)
            print("  pred force shape  :", pred_forces.shape)

e_ref = np.asarray(e_ref)
e_pred = np.asarray(e_pred)
f_ref = np.concatenate(f_ref)
f_pred = np.concatenate(f_pred)

# -----------------------------
# Metrics
# -----------------------------
e_err = e_pred - e_ref
f_err = f_pred - f_ref

e_mae = np.mean(np.abs(e_err))
e_rmse = np.sqrt(np.mean(e_err**2))
f_mae = np.mean(np.abs(f_err))
f_rmse = np.sqrt(np.mean(f_err**2))

e_r2 = r2_score(e_ref, e_pred)
f_r2 = r2_score(f_ref, f_pred)

print("\nEnergy metrics:")
print(f"  MAE  = {e_mae * 1000:.6f} meV/atom")
print(f"  RMSE = {e_rmse * 1000:.6f} meV/atom")
print(f"  R^2  = {e_r2:.6f}")

print("\nForce metrics:")
print(f"  MAE  = {f_mae * 1000:.6f} meV/A")
print(f"  RMSE = {f_rmse * 1000:.6f} meV/A")
print(f"  R^2  = {f_r2:.6f}")

# -----------------------------
# Save raw data
# -----------------------------
OUTPUT_PREFIX.parent.mkdir(parents=True, exist_ok=True)

np.savez(
    OUTPUT_PREFIX.with_name(OUTPUT_PREFIX.name + "_parity_data.npz"),
    e_ref=e_ref,
    e_pred=e_pred,
    f_ref=f_ref,
    f_pred=f_pred,
)

# -----------------------------
# Energy parity plot
# -----------------------------
emin = min(np.min(e_ref), np.min(e_pred))
emax = max(np.max(e_ref), np.max(e_pred))

plt.figure(figsize=(6, 6))
plt.scatter(e_ref, e_pred, s=20, alpha=0.8)
plt.plot([emin, emax], [emin, emax], linestyle="--")
plt.xlabel("Reference energy (per atom)")
plt.ylabel("Predicted energy (per atom)")
plt.title(
    "Energy parity\n"
    f"MAE={e_mae*1000:.3f} meV/atom, "
    f"RMSE={e_rmse*1000:.3f} meV/atom, "
    f"R²={e_r2:.4f}"
)
plt.tight_layout()
energy_png = OUTPUT_PREFIX.with_name(OUTPUT_PREFIX.name + "_energy_parity.png")
plt.savefig(energy_png, dpi=200)
plt.close()

# -----------------------------
# Force parity plot
# -----------------------------
fmin = min(np.min(f_ref), np.min(f_pred))
fmax = max(np.max(f_ref), np.max(f_pred))

plt.figure(figsize=(6, 6))
plt.scatter(f_ref, f_pred, s=8, alpha=0.35)
plt.plot([fmin, fmax], [fmin, fmax], linestyle="--")
plt.xlabel("Reference force component")
plt.ylabel("Predicted force component")
plt.title(
    "Force parity\n"
    f"MAE={f_mae*1000:.3f} meV/A, "
    f"RMSE={f_rmse*1000:.3f} meV/A, "
    f"R²={f_r2:.4f}"
)
plt.tight_layout()
force_png = OUTPUT_PREFIX.with_name(OUTPUT_PREFIX.name + "_force_parity.png")
plt.savefig(force_png, dpi=200)
plt.close()

print("\nWrote:")
print(" ", energy_png)
print(" ", force_png)
print(" ", OUTPUT_PREFIX.with_name(OUTPUT_PREFIX.name + "_parity_data.npz"))
