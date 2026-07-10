from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from ase import Atoms
from ase.io import read
from mace.calculators import MACECalculator


def normalize_quaternions(q: np.ndarray) -> np.ndarray:
    norm = np.linalg.norm(q, axis=-1, keepdims=True)

    if np.any(norm < 1.0e-14):
        raise ValueError("Encountered zero-norm quaternion.")

    return q / norm


def quaternion_multiply_wxyz(
    q1: np.ndarray,
    q2: np.ndarray,
) -> np.ndarray:
    w1, x1, y1, z1 = np.moveaxis(q1, -1, 0)
    w2, x2, y2, z2 = np.moveaxis(q2, -1, 0)

    return np.stack(
        (
            w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
            w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
            w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
            w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
        ),
        axis=-1,
    )


def axis_angle_quaternion_wxyz(
    axis: int,
    angle: float,
) -> np.ndarray:
    q = np.zeros(4, dtype=np.float64)
    q[0] = np.cos(0.5 * angle)
    q[axis + 1] = np.sin(0.5 * angle)
    return q


def rotate_one_particle_lab_frame(
    quaternions: np.ndarray,
    particle: int,
    axis: int,
    angle: float,
) -> np.ndarray:
    result = quaternions.copy()
    delta = axis_angle_quaternion_wxyz(axis, angle)

    result[particle] = quaternion_multiply_wxyz(
        delta,
        result[particle],
    )

    return normalize_quaternions(result)


def predict(
    atoms: Atoms,
    calculator: MACECalculator,
) -> tuple[float, np.ndarray]:
    trial = atoms.copy()
    trial.calc = None

    # ASE does not detect changes to custom arrays such as quaternions.
    calculator.reset()
    trial.calc = calculator

    energy = float(trial.get_potential_energy())
    forces = np.asarray(
        trial.get_forces(),
        dtype=np.float64,
    )

    return energy, forces


def finite_difference_torque(
    atoms: Atoms,
    calculator: MACECalculator,
    epsilon: float,
) -> np.ndarray:
    if "quaternions" not in atoms.arrays:
        raise KeyError("Frame has no 'quaternions' array.")

    q0 = normalize_quaternions(
        np.asarray(
            atoms.arrays["quaternions"],
            dtype=np.float64,
        )
    )

    torque = np.zeros((len(atoms), 3), dtype=np.float64)

    for particle in range(len(atoms)):
        for axis in range(3):
            plus = atoms.copy()
            minus = atoms.copy()

            plus.arrays["quaternions"] = (
                rotate_one_particle_lab_frame(
                    q0,
                    particle,
                    axis,
                    +epsilon,
                )
            )
            minus.arrays["quaternions"] = (
                rotate_one_particle_lab_frame(
                    q0,
                    particle,
                    axis,
                    -epsilon,
                )
            )

            energy_plus, _ = predict(plus, calculator)
            energy_minus, _ = predict(minus, calculator)

            torque[particle, axis] = -(
                energy_plus - energy_minus
            ) / (2.0 * epsilon)

    return torque


def metrics(
    reference: np.ndarray,
    prediction: np.ndarray,
) -> tuple[float, float, float]:
    error = prediction - reference
    mae = float(np.mean(np.abs(error)))
    rmse = float(np.sqrt(np.mean(error**2)))

    ss_res = float(np.sum(error**2))
    ss_tot = float(
        np.sum(
            (reference - np.mean(reference)) ** 2
        )
    )

    r2 = (
        1.0 - ss_res / ss_tot
        if ss_tot > 0.0
        else float("nan")
    )

    return mae, rmse, r2


def parity_plot(
    reference: np.ndarray,
    prediction: np.ndarray,
    output: Path,
) -> None:
    mae, rmse, r2 = metrics(reference, prediction)

    lower = min(reference.min(), prediction.min())
    upper = max(reference.max(), prediction.max())
    margin = 0.05 * max(upper - lower, 1.0e-12)

    fig, ax = plt.subplots(figsize=(7, 7))

    ax.scatter(
        reference,
        prediction,
        s=10,
        alpha=0.55,
    )

    ax.plot(
        [lower - margin, upper + margin],
        [lower - margin, upper + margin],
        linestyle="--",
        linewidth=1.5,
    )

    ax.set_xlim(lower - margin, upper + margin)
    ax.set_ylim(lower - margin, upper + margin)
    ax.set_xlabel("Reference torque component")
    ax.set_ylabel("Predicted torque component")
    ax.set_title("Test torque parity")

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


def histogram(
    values: np.ndarray,
    output: Path,
    title: str,
    xlabel: str,
) -> None:
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.hist(values, bins=50)
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel("Count")
    fig.tight_layout()
    fig.savefig(output, dpi=200)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--model",
        default=(
            "checkpoints/"
            "dimer_full_L3_I2_v1_run-17.model"
        ),
    )
    parser.add_argument(
        "--data",
        default="test.xyz",
    )
    parser.add_argument(
        "--epsilon",
        type=float,
        default=1.0e-4,
    )
    parser.add_argument(
        "--device",
        default="cpu",
    )
    parser.add_argument(
        "--default-dtype",
        default="float64",
        choices=("float32", "float64"),
    )
    parser.add_argument(
        "--max-frames",
        type=int,
        default=None,
    )
    parser.add_argument(
        "--output-prefix",
        default="test_torque",
    )

    args = parser.parse_args()

    model_path = Path(args.model)
    data_path = Path(args.data)
    prefix = Path(args.output_prefix)

    frames = read(data_path, index=":")

    if args.max_frames is not None:
        frames = frames[: args.max_frames]

    calculator = MACECalculator(
        model_paths=str(model_path),
        device=args.device,
        default_dtype=args.default_dtype,
    )

    torque_reference = []
    torque_prediction = []
    force_reference = []
    force_prediction = []
    rotational_residuals = []
    reference_rotational_residuals = []

    for frame_index, atoms in enumerate(frames):
        if "torques" not in atoms.arrays:
            raise KeyError(
                f"Frame {frame_index} has no 'torques' array."
            )

        ref_torque = np.asarray(
            atoms.arrays["torques"],
            dtype=np.float64,
        )
        ref_force = np.asarray(
            atoms.get_forces(),
            dtype=np.float64,
        )

        _, pred_force = predict(atoms, calculator)

        pred_torque = finite_difference_torque(
            atoms,
            calculator,
            epsilon=args.epsilon,
        )

        positions = np.asarray(
            atoms.positions,
            dtype=np.float64,
        )

        # Shift to center of geometry so the reported residual does not
        # depend on the arbitrary absolute origin.
        centered_positions = (
            positions - positions.mean(axis=0, keepdims=True)
        )

        pred_residual = (
            pred_torque.sum(axis=0)
            + np.cross(
                centered_positions,
                pred_force,
            ).sum(axis=0)
        )

        ref_residual = (
            ref_torque.sum(axis=0)
            + np.cross(
                centered_positions,
                ref_force,
            ).sum(axis=0)
        )

        torque_reference.append(ref_torque.reshape(-1))
        torque_prediction.append(pred_torque.reshape(-1))
        force_reference.append(ref_force.reshape(-1))
        force_prediction.append(pred_force.reshape(-1))
        rotational_residuals.append(pred_residual)
        reference_rotational_residuals.append(ref_residual)

        if (
            (frame_index + 1) % 25 == 0
            or frame_index + 1 == len(frames)
        ):
            print(
                f"Evaluated {frame_index + 1}/"
                f"{len(frames)} frames"
            )

    torque_reference = np.concatenate(torque_reference)
    torque_prediction = np.concatenate(torque_prediction)
    force_reference = np.concatenate(force_reference)
    force_prediction = np.concatenate(force_prediction)

    rotational_residuals = np.asarray(
        rotational_residuals
    )
    reference_rotational_residuals = np.asarray(
        reference_rotational_residuals
    )

    torque_mae, torque_rmse, torque_r2 = metrics(
        torque_reference,
        torque_prediction,
    )

    force_mae, force_rmse, force_r2 = metrics(
        force_reference,
        force_prediction,
    )

    torque_rms = float(
        np.sqrt(np.mean(torque_reference**2))
    )
    relative_torque_rmse = (
        100.0 * torque_rmse / torque_rms
        if torque_rms > 0.0
        else float("nan")
    )

    pred_residual_norm = np.linalg.norm(
        rotational_residuals,
        axis=1,
    )
    ref_residual_norm = np.linalg.norm(
        reference_rotational_residuals,
        axis=1,
    )

    print()
    print("Torque metrics")
    print(f"  MAE:            {torque_mae:.10g}")
    print(f"  RMSE:           {torque_rmse:.10g}")
    print(f"  R²:             {torque_r2:.10g}")
    print(
        f"  Relative RMSE:  "
        f"{relative_torque_rmse:.4f}%"
    )

    print()
    print("Force metrics")
    print(f"  MAE:            {force_mae:.10g}")
    print(f"  RMSE:           {force_rmse:.10g}")
    print(f"  R²:             {force_r2:.10g}")

    print()
    print("Rotational-invariance residual")
    print(
        "  Predicted RMS norm: "
        f"{np.sqrt(np.mean(pred_residual_norm**2)):.10g}"
    )
    print(
        "  Predicted max norm: "
        f"{pred_residual_norm.max():.10g}"
    )
    print(
        "  Reference RMS norm: "
        f"{np.sqrt(np.mean(ref_residual_norm**2)):.10g}"
    )
    print(
        "  Reference max norm: "
        f"{ref_residual_norm.max():.10g}"
    )

    parity_plot(
        torque_reference,
        torque_prediction,
        Path(f"{prefix}_parity.png"),
    )

    histogram(
        pred_residual_norm,
        Path(f"{prefix}_rotational_residual.png"),
        title="Predicted rotational-invariance residual",
        xlabel="Residual norm",
    )

    np.savez(
        f"{prefix}_data.npz",
        torque_reference=torque_reference,
        torque_prediction=torque_prediction,
        force_reference=force_reference,
        force_prediction=force_prediction,
        rotational_residuals=rotational_residuals,
        reference_rotational_residuals=(
            reference_rotational_residuals
        ),
        epsilon=args.epsilon,
    )

    print()
    print("Created:")
    print(f"  {prefix}_parity.png")
    print(f"  {prefix}_rotational_residual.png")
    print(f"  {prefix}_data.npz")


if __name__ == "__main__":
    main()
