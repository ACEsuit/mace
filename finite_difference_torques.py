from __future__ import annotations

import argparse
import copy
from pathlib import Path
from typing import Callable

import numpy as np
from ase import Atoms
from ase.io import read
from mace.calculators import MACECalculator


COMMON_QUATERNION_LAYOUTS = (
    # Scalar-first, wxyz.
    ("c_qw", "c_qx", "c_qy", "c_qz"),
    ("qw", "qx", "qy", "qz"),
    ("quat_w", "quat_x", "quat_y", "quat_z"),
    ("quaternion_w", "quaternion_x", "quaternion_y", "quaternion_z"),

    # Scalar-last, xyzw. This layout is handled separately below.
    ("c_qx", "c_qy", "c_qz", "c_qw"),
    ("qx", "qy", "qz", "qw"),
)


def normalize_quaternions(q: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(q, axis=-1, keepdims=True)

    if np.any(norms < 1.0e-14):
        raise ValueError("Encountered a zero-norm quaternion.")

    return q / norms


def quaternion_multiply_wxyz(
    q1: np.ndarray,
    q2: np.ndarray,
) -> np.ndarray:
    """Hamilton product, with both inputs in wxyz convention."""
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
    """Quaternion for a rotation about one Cartesian lab axis."""
    q = np.zeros(4, dtype=np.float64)
    q[0] = np.cos(0.5 * angle)
    q[axis + 1] = np.sin(0.5 * angle)
    return q


def detect_quaternion_accessor(
    atoms: Atoms,
) -> tuple[
    Callable[[Atoms], np.ndarray],
    Callable[[Atoms, np.ndarray], None],
    str,
]:
    """
    Find quaternion storage and return getter/setter functions.

    The returned arrays always use wxyz ordering internally.
    """

    # One N x 4 array.
    for key in ("quaternions", "quaternion", "quat", "c_quaternion"):
        if key not in atoms.arrays:
            continue

        values = np.asarray(atoms.arrays[key])

        if values.shape != (len(atoms), 4):
            continue

        # Assume the combined array is wxyz, matching the current rigid-body
        # parser. Change this here if your combined array is xyzw.
        def getter(a: Atoms, key: str = key) -> np.ndarray:
            return normalize_quaternions(
                np.asarray(a.arrays[key], dtype=np.float64).copy()
            )

        def setter(
            a: Atoms,
            q_wxyz: np.ndarray,
            key: str = key,
        ) -> None:
            a.arrays[key] = np.asarray(q_wxyz, dtype=np.float64)

        return getter, setter, f"combined array {key!r}, assumed wxyz"

    # Four scalar arrays.
    for names in COMMON_QUATERNION_LAYOUTS:
        if not all(name in atoms.arrays for name in names):
            continue

        is_wxyz = names[0].endswith("qw") or names[0] in {
            "qw",
            "quat_w",
            "quaternion_w",
        }

        if is_wxyz:
            def getter(
                a: Atoms,
                names: tuple[str, ...] = names,
            ) -> np.ndarray:
                q = np.column_stack(
                    [
                        np.asarray(a.arrays[name], dtype=np.float64)
                        for name in names
                    ]
                )
                return normalize_quaternions(q)

            def setter(
                a: Atoms,
                q_wxyz: np.ndarray,
                names: tuple[str, ...] = names,
            ) -> None:
                for column, name in enumerate(names):
                    a.arrays[name] = q_wxyz[:, column].copy()

            return getter, setter, f"scalar arrays {names}, wxyz"

        # Scalar-last source layout: x, y, z, w.
        def getter(
            a: Atoms,
            names: tuple[str, ...] = names,
        ) -> np.ndarray:
            q_xyzw = np.column_stack(
                [
                    np.asarray(a.arrays[name], dtype=np.float64)
                    for name in names
                ]
            )
            q_wxyz = q_xyzw[:, [3, 0, 1, 2]]
            return normalize_quaternions(q_wxyz)

        def setter(
            a: Atoms,
            q_wxyz: np.ndarray,
            names: tuple[str, ...] = names,
        ) -> None:
            q_xyzw = q_wxyz[:, [1, 2, 3, 0]]

            for column, name in enumerate(names):
                a.arrays[name] = q_xyzw[:, column].copy()

        return getter, setter, f"scalar arrays {names}, xyzw"

    available = sorted(atoms.arrays.keys())

    raise KeyError(
        "Could not detect quaternion storage.\n"
        f"Available ASE arrays: {available}\n"
        "Add your quaternion field names to COMMON_QUATERNION_LAYOUTS."
    )


def remove_stale_results(atoms: Atoms) -> None:
    """
    Remove any cached calculator before attaching MACE.

    The original reference energy and forces should be read before calling
    this function.
    """
    atoms.calc = None


def rotate_one_particle_lab_frame(
    q_wxyz: np.ndarray,
    particle: int,
    axis: int,
    angle: float,
) -> np.ndarray:
    """
    Apply a lab-frame increment.

    Lab-frame rotation uses left multiplication:
        q_new = dq * q_old
    """
    result = q_wxyz.copy()
    delta = axis_angle_quaternion_wxyz(axis, angle)

    result[particle] = quaternion_multiply_wxyz(
        delta,
        result[particle],
    )

    return normalize_quaternions(result)


def predict_energy(
    atoms: Atoms,
    calculator: MACECalculator,
) -> float:
    trial = atoms.copy()
    remove_stale_results(trial)

    # ASE does not treat custom arrays such as quaternions as system
    # changes. Reset the calculator so orientation changes trigger a
    # fresh AtomicData conversion and model evaluation.
    calculator.reset()

    trial.calc = calculator
    return float(trial.get_potential_energy())


def finite_difference_torque(
    atoms: Atoms,
    calculator: MACECalculator,
    epsilon: float,
) -> np.ndarray:
    get_quaternions, set_quaternions, description = (
        detect_quaternion_accessor(atoms)
    )

    print(f"Quaternion storage: {description}")

    q0 = get_quaternions(atoms)
    torque = np.zeros((len(atoms), 3), dtype=np.float64)

    for particle in range(len(atoms)):
        for axis in range(3):
            plus = atoms.copy()
            minus = atoms.copy()

            q_plus = rotate_one_particle_lab_frame(
                q0,
                particle,
                axis,
                +epsilon,
            )
            q_minus = rotate_one_particle_lab_frame(
                q0,
                particle,
                axis,
                -epsilon,
            )

            set_quaternions(plus, q_plus)
            set_quaternions(minus, q_minus)

            energy_plus = predict_energy(plus, calculator)
            energy_minus = predict_energy(minus, calculator)

            torque[particle, axis] = -(
                energy_plus - energy_minus
            ) / (2.0 * epsilon)

    return torque


def detect_reference_torque(atoms: Atoms) -> np.ndarray | None:
    for key in (
        "torques",
        "torque",
        "REF_torques",
        "ref_torques",
    ):
        if key not in atoms.arrays:
            continue

        value = np.asarray(atoms.arrays[key], dtype=np.float64)

        if value.shape == (len(atoms), 3):
            print(f"Reference torque array: {key!r}")
            return value.copy()

    return None


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model",
        default=(
            "checkpoints/"
            "dimer_full_L3_I2_v1_run-17.model"
        ),
    )
    parser.add_argument("--data", default="test.xyz")
    parser.add_argument("--index", type=int, default=0)
    parser.add_argument("--epsilon", type=float, default=1.0e-5)
    parser.add_argument("--device", default="cpu")
    parser.add_argument(
        "--default-dtype",
        choices=("float32", "float64"),
        default="float64",
    )
    parser.add_argument(
        "--output",
        default="finite_difference_torque_frame0.npz",
    )
    args = parser.parse_args()

    model_path = Path(args.model)
    data_path = Path(args.data)

    if not model_path.exists():
        raise FileNotFoundError(model_path)

    if not data_path.exists():
        raise FileNotFoundError(data_path)

    atoms = read(data_path, index=args.index)

    # Read labels before replacing the original SinglePointCalculator.
    reference_energy = float(atoms.get_potential_energy())
    reference_forces = np.asarray(
        atoms.get_forces(),
        dtype=np.float64,
    )
    reference_torque = detect_reference_torque(atoms)

    calculator = MACECalculator(
        model_paths=str(model_path),
        device=args.device,
        default_dtype=args.default_dtype,
    )

    predicted_energy = predict_energy(atoms, calculator)
    predicted_torque = finite_difference_torque(
        atoms,
        calculator,
        epsilon=args.epsilon,
    )

    print()
    print(f"Frame: {args.index}")
    print(f"Reference energy: {reference_energy:.12g}")
    print(f"Predicted energy: {predicted_energy:.12g}")
    print(f"Epsilon: {args.epsilon:.3e} rad")
    print()
    print("Predicted lab-frame torque:")
    print(predicted_torque)
    print()
    print("Net predicted torque:")
    print(predicted_torque.sum(axis=0))

    if reference_torque is not None:
        error = predicted_torque - reference_torque
        rmse = np.sqrt(np.mean(error**2))
        mae = np.mean(np.abs(error))

        print()
        print("Reference torque:")
        print(reference_torque)
        print()
        print(f"Torque MAE:  {mae:.12g}")
        print(f"Torque RMSE: {rmse:.12g}")

    np.savez(
        args.output,
        frame_index=args.index,
        epsilon=args.epsilon,
        reference_energy=reference_energy,
        predicted_energy=predicted_energy,
        reference_forces=reference_forces,
        predicted_torque=predicted_torque,
        reference_torque=(
            reference_torque
            if reference_torque is not None
            else np.empty((0, 3), dtype=np.float64)
        ),
    )

    print()
    print(f"Saved: {args.output}")


if __name__ == "__main__":
    main()
