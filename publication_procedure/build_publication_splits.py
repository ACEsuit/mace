from __future__ import annotations

import argparse
import csv
import hashlib
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
from ase import Atoms
from ase.io import read, write


DEFAULT_SUBSET_SIZES = (64, 128, 256, 512, 1024)


@dataclass(frozen=True)
class Record:
    index: int
    energy: float
    energy_per_particle: float
    separation: float
    force_rms: float
    force_max: float
    torque_rms: float
    torque_max: float
    net_force_norm: float
    force_pair_residual_norm: float
    torque_balance_residual_norm: float
    quaternion_norm_error_max: float
    split_random: str
    split_ood: str


def array_or_none(
    atoms: Atoms,
    names: Iterable[str],
) -> np.ndarray | None:
    for name in names:
        if name in atoms.arrays:
            return np.asarray(
                atoms.arrays[name],
                dtype=np.float64,
            )
    return None


def get_energy(atoms: Atoms) -> float:
    try:
        return float(atoms.get_potential_energy())
    except Exception as exc:
        raise RuntimeError(
            "Could not read energy from an ASE calculator result."
        ) from exc


def get_forces(atoms: Atoms) -> np.ndarray:
    try:
        forces = np.asarray(
            atoms.get_forces(),
            dtype=np.float64,
        )
    except Exception as exc:
        raise RuntimeError(
            "Could not read forces from an ASE calculator result."
        ) from exc

    if forces.shape != (len(atoms), 3):
        raise ValueError(
            f"Unexpected force shape {forces.shape}; "
            f"expected {(len(atoms), 3)}."
        )

    return forces


def get_torques(atoms: Atoms) -> np.ndarray:
    torques = array_or_none(
        atoms,
        (
            "torques",
            "REF_torques",
            "ref_torques",
            "torque",
        ),
    )

    if torques is None:
        return np.zeros(
            (len(atoms), 3),
            dtype=np.float64,
        )

    if torques.shape != (len(atoms), 3):
        raise ValueError(
            f"Unexpected torque shape {torques.shape}; "
            f"expected {(len(atoms), 3)}."
        )

    return torques


def get_quaternions(atoms: Atoms) -> np.ndarray | None:
    quaternions = array_or_none(
        atoms,
        (
            "quaternions",
            "quaternion",
            "quat",
        ),
    )

    if quaternions is None:
        return None

    if quaternions.shape != (len(atoms), 4):
        raise ValueError(
            f"Unexpected quaternion shape {quaternions.shape}; "
            f"expected {(len(atoms), 4)}."
        )

    return quaternions


def separation(atoms: Atoms) -> float:
    if len(atoms) != 2:
        raise ValueError(
            "This publication split script currently assumes dimers."
        )

    return float(
        np.linalg.norm(
            np.asarray(atoms.positions[1])
            - np.asarray(atoms.positions[0])
        )
    )


def rotational_balance_residual(
    atoms: Atoms,
    forces: np.ndarray,
    torques: np.ndarray,
) -> np.ndarray:
    positions = np.asarray(
        atoms.positions,
        dtype=np.float64,
    )

    centered = positions - positions.mean(
        axis=0,
        keepdims=True,
    )

    return (
        torques.sum(axis=0)
        + np.cross(centered, forces).sum(axis=0)
    )


def duplicate_hash(atoms: Atoms) -> str:
    """
    Hash geometry, orientations, diameters, energy, forces, and torques.

    Values are rounded to reduce sensitivity to text serialization noise.
    """
    pieces: list[np.ndarray] = [
        np.asarray(atoms.numbers, dtype=np.int64),
        np.round(
            np.asarray(atoms.positions, dtype=np.float64),
            decimals=12,
        ),
    ]

    for key in (
        "quaternions",
        "c_diameter1",
        "c_diameter2",
        "c_diameter3",
        "torques",
    ):
        if key in atoms.arrays:
            pieces.append(
                np.round(
                    np.asarray(atoms.arrays[key]),
                    decimals=12,
                )
            )

    pieces.append(
        np.asarray([get_energy(atoms)], dtype=np.float64)
    )
    pieces.append(
        np.round(get_forces(atoms), decimals=12)
    )

    digest = hashlib.sha256()

    for piece in pieces:
        contiguous = np.ascontiguousarray(piece)
        digest.update(str(contiguous.dtype).encode())
        digest.update(str(contiguous.shape).encode())
        digest.update(contiguous.tobytes())

    return digest.hexdigest()


def quantile_bins(
    values: np.ndarray,
    n_bins: int,
) -> np.ndarray:
    if n_bins < 1:
        raise ValueError("n_bins must be positive.")

    quantiles = np.linspace(0.0, 1.0, n_bins + 1)
    edges = np.quantile(values, quantiles)
    edges = np.unique(edges)

    if len(edges) <= 2:
        return np.zeros(len(values), dtype=np.int64)

    return np.digitize(
        values,
        bins=edges[1:-1],
        right=False,
    )


def combined_strata(
    separations: np.ndarray,
    energies: np.ndarray,
    n_sep_bins: int,
    n_energy_bins: int,
) -> np.ndarray:
    sep_bins = quantile_bins(
        separations,
        n_sep_bins,
    )
    energy_bins = quantile_bins(
        energies,
        n_energy_bins,
    )

    return (
        sep_bins.astype(np.int64) * n_energy_bins
        + energy_bins.astype(np.int64)
    )


def stratified_split(
    strata: np.ndarray,
    train_fraction: float,
    valid_fraction: float,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    train_indices: list[int] = []
    valid_indices: list[int] = []
    test_indices: list[int] = []

    for stratum in np.unique(strata):
        members = np.flatnonzero(strata == stratum)
        members = rng.permutation(members)

        count = len(members)

        n_train = int(
            np.floor(train_fraction * count)
        )
        n_valid = int(
            np.floor(valid_fraction * count)
        )

        if count >= 3:
            n_train = max(n_train, 1)
            n_valid = max(n_valid, 1)

            if n_train + n_valid >= count:
                n_train = max(count - 2, 1)
                n_valid = 1

        train_indices.extend(
            members[:n_train].tolist()
        )
        valid_indices.extend(
            members[n_train:n_train + n_valid].tolist()
        )
        test_indices.extend(
            members[n_train + n_valid:].tolist()
        )

    return (
        np.sort(np.asarray(train_indices, dtype=np.int64)),
        np.sort(np.asarray(valid_indices, dtype=np.int64)),
        np.sort(np.asarray(test_indices, dtype=np.int64)),
    )


def write_indices(
    path: Path,
    indices: np.ndarray,
) -> None:
    np.savetxt(
        path,
        indices,
        fmt="%d",
    )


def write_frames(
    path: Path,
    frames: list[Atoms],
    indices: np.ndarray,
) -> None:
    selected = [frames[int(i)] for i in indices]

    write(
        path,
        selected,
        format="extxyz",
        write_results=True,
    )


def nested_subsets(
    train_indices: np.ndarray,
    subset_sizes: tuple[int, ...],
    rng: np.random.Generator,
) -> dict[int, np.ndarray]:
    permutation = rng.permutation(train_indices)

    subsets: dict[int, np.ndarray] = {}

    for size in subset_sizes:
        if size > len(train_indices):
            continue

        subsets[size] = np.sort(
            permutation[:size]
        )

    subsets[len(train_indices)] = np.sort(
        train_indices
    )

    return subsets


def split_labels(
    total: int,
    train_indices: np.ndarray,
    valid_indices: np.ndarray,
    test_indices: np.ndarray,
) -> list[str]:
    labels = ["unassigned"] * total

    for index in train_indices:
        labels[int(index)] = "train"

    for index in valid_indices:
        labels[int(index)] = "valid"

    for index in test_indices:
        labels[int(index)] = "test"

    return labels


def describe(values: np.ndarray) -> dict[str, float]:
    return {
        "min": float(np.min(values)),
        "max": float(np.max(values)),
        "mean": float(np.mean(values)),
        "std": float(np.std(values)),
        "median": float(np.median(values)),
        "q05": float(np.quantile(values, 0.05)),
        "q25": float(np.quantile(values, 0.25)),
        "q75": float(np.quantile(values, 0.75)),
        "q95": float(np.quantile(values, 0.95)),
    }


def main() -> None:
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--input",
        default="gb_dimers.xyz",
        help="Master extxyz containing all configurations.",
    )
    parser.add_argument(
        "--output-dir",
        default="publication_splits",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=20260710,
    )
    parser.add_argument(
        "--train-fraction",
        type=float,
        default=0.70,
    )
    parser.add_argument(
        "--valid-fraction",
        type=float,
        default=0.15,
    )
    parser.add_argument(
        "--separation-bins",
        type=int,
        default=8,
    )
    parser.add_argument(
        "--energy-bins",
        type=int,
        default=8,
    )
    parser.add_argument(
        "--ood-quantile",
        type=float,
        default=0.85,
        help=(
            "Configurations at or below this separation quantile "
            "form the OOD test set."
        ),
    )
    parser.add_argument(
        "--subset-sizes",
        nargs="+",
        type=int,
        default=list(DEFAULT_SUBSET_SIZES),
    )

    args = parser.parse_args()

    if not (
        0.0 < args.train_fraction < 1.0
        and 0.0 < args.valid_fraction < 1.0
        and args.train_fraction + args.valid_fraction < 1.0
    ):
        raise ValueError(
            "Fractions must be positive and sum to less than one."
        )

    input_path = Path(args.input)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    frames = read(
        input_path,
        index=":",
    )

    if not frames:
        raise RuntimeError(
            f"No configurations found in {input_path}."
        )

    energies = np.asarray(
        [get_energy(atoms) for atoms in frames],
        dtype=np.float64,
    )
    separations = np.asarray(
        [separation(atoms) for atoms in frames],
        dtype=np.float64,
    )

    duplicate_hashes = [
        duplicate_hash(atoms)
        for atoms in frames
    ]
    unique_hash_count = len(set(duplicate_hashes))
    duplicate_count = len(frames) - unique_hash_count

    strata = combined_strata(
        separations=separations,
        energies=energies,
        n_sep_bins=args.separation_bins,
        n_energy_bins=args.energy_bins,
    )

    random_rng = np.random.default_rng(args.seed)

    (
        train_indices,
        valid_indices,
        test_indices,
    ) = stratified_split(
        strata=strata,
        train_fraction=args.train_fraction,
        valid_fraction=args.valid_fraction,
        rng=random_rng,
    )

    random_labels = split_labels(
        total=len(frames),
        train_indices=train_indices,
        valid_indices=valid_indices,
        test_indices=test_indices,
    )

    # OOD protocol:
    # Hold out the shortest-separation tail as test data.
    ood_threshold = float(
        np.quantile(
            separations,
            1.0 - args.ood_quantile,
        )
    )

    ood_test_indices = np.flatnonzero(
        separations <= ood_threshold
    )
    ood_pool_indices = np.flatnonzero(
        separations > ood_threshold
    )

    ood_pool_strata = strata[ood_pool_indices]
    ood_rng = np.random.default_rng(
        args.seed + 1
    )

    (
        local_ood_train,
        local_ood_valid,
        _,
    ) = stratified_split(
        strata=ood_pool_strata,
        train_fraction=(
            args.train_fraction
            / (
                args.train_fraction
                + args.valid_fraction
            )
        ),
        valid_fraction=(
            args.valid_fraction
            / (
                args.train_fraction
                + args.valid_fraction
            )
        ),
        rng=ood_rng,
    )

    ood_train_indices = np.sort(
        ood_pool_indices[local_ood_train]
    )
    ood_valid_indices = np.sort(
        ood_pool_indices[local_ood_valid]
    )

    ood_labels = ["unused"] * len(frames)

    for index in ood_train_indices:
        ood_labels[int(index)] = "train"

    for index in ood_valid_indices:
        ood_labels[int(index)] = "valid"

    for index in ood_test_indices:
        ood_labels[int(index)] = "test_ood_short_range"

    write_indices(
        output_dir / "random_train_indices.txt",
        train_indices,
    )
    write_indices(
        output_dir / "random_valid_indices.txt",
        valid_indices,
    )
    write_indices(
        output_dir / "random_test_indices.txt",
        test_indices,
    )

    write_frames(
        output_dir / "random_train.xyz",
        frames,
        train_indices,
    )
    write_frames(
        output_dir / "random_valid.xyz",
        frames,
        valid_indices,
    )
    write_frames(
        output_dir / "random_test.xyz",
        frames,
        test_indices,
    )

    write_indices(
        output_dir / "ood_train_indices.txt",
        ood_train_indices,
    )
    write_indices(
        output_dir / "ood_valid_indices.txt",
        ood_valid_indices,
    )
    write_indices(
        output_dir / "ood_test_indices.txt",
        ood_test_indices,
    )

    write_frames(
        output_dir / "ood_train.xyz",
        frames,
        ood_train_indices,
    )
    write_frames(
        output_dir / "ood_valid.xyz",
        frames,
        ood_valid_indices,
    )
    write_frames(
        output_dir / "ood_test_short_range.xyz",
        frames,
        ood_test_indices,
    )

    subset_rng = np.random.default_rng(
        args.seed + 2
    )
    subsets = nested_subsets(
        train_indices=train_indices,
        subset_sizes=tuple(
            sorted(set(args.subset_sizes))
        ),
        rng=subset_rng,
    )

    subset_dir = output_dir / "nested_subsets"
    subset_dir.mkdir(
        exist_ok=True,
    )

    for size, indices in subsets.items():
        write_indices(
            subset_dir / f"train_{size}_indices.txt",
            indices,
        )
        write_frames(
            subset_dir / f"train_{size}.xyz",
            frames,
            indices,
        )

    records: list[Record] = []

    force_rms_values = []
    torque_rms_values = []
    torque_balance_values = []
    net_force_values = []

    for index, atoms in enumerate(frames):
        energy = energies[index]
        forces = get_forces(atoms)
        torques = get_torques(atoms)
        quaternions = get_quaternions(atoms)

        force_rms = float(
            np.sqrt(np.mean(forces**2))
        )
        force_max = float(
            np.max(np.abs(forces))
        )
        torque_rms = float(
            np.sqrt(np.mean(torques**2))
        )
        torque_max = float(
            np.max(np.abs(torques))
        )

        net_force = forces.sum(axis=0)
        force_pair_residual = (
            forces[0] + forces[1]
        )

        torque_balance = rotational_balance_residual(
            atoms=atoms,
            forces=forces,
            torques=torques,
        )

        quaternion_norm_error = 0.0

        if quaternions is not None:
            quaternion_norm_error = float(
                np.max(
                    np.abs(
                        np.linalg.norm(
                            quaternions,
                            axis=1,
                        )
                        - 1.0
                    )
                )
            )

        record = Record(
            index=index,
            energy=energy,
            energy_per_particle=(
                energy / len(atoms)
            ),
            separation=separations[index],
            force_rms=force_rms,
            force_max=force_max,
            torque_rms=torque_rms,
            torque_max=torque_max,
            net_force_norm=float(
                np.linalg.norm(net_force)
            ),
            force_pair_residual_norm=float(
                np.linalg.norm(
                    force_pair_residual
                )
            ),
            torque_balance_residual_norm=float(
                np.linalg.norm(
                    torque_balance
                )
            ),
            quaternion_norm_error_max=(
                quaternion_norm_error
            ),
            split_random=random_labels[index],
            split_ood=ood_labels[index],
        )

        records.append(record)
        force_rms_values.append(force_rms)
        torque_rms_values.append(torque_rms)
        torque_balance_values.append(
            record.torque_balance_residual_norm
        )
        net_force_values.append(
            record.net_force_norm
        )

    manifest_path = (
        output_dir
        / "publication_manifest.csv"
    )

    with manifest_path.open(
        "w",
        newline="",
        encoding="utf-8",
    ) as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=list(
                asdict(records[0]).keys()
            ),
        )
        writer.writeheader()

        for record in records:
            writer.writerow(
                asdict(record)
            )

    summary = {
        "input_file": str(input_path),
        "seed": args.seed,
        "configuration_count": len(frames),
        "unique_configuration_hashes": (
            unique_hash_count
        ),
        "exact_duplicate_count": duplicate_count,
        "random_split": {
            "train_count": len(train_indices),
            "valid_count": len(valid_indices),
            "test_count": len(test_indices),
            "train_fraction_requested": (
                args.train_fraction
            ),
            "valid_fraction_requested": (
                args.valid_fraction
            ),
            "test_fraction_requested": (
                1.0
                - args.train_fraction
                - args.valid_fraction
            ),
            "separation_bins": (
                args.separation_bins
            ),
            "energy_bins": args.energy_bins,
        },
        "ood_split": {
            "definition": (
                "Shortest-separation tail held out"
            ),
            "separation_threshold": (
                ood_threshold
            ),
            "train_count": len(
                ood_train_indices
            ),
            "valid_count": len(
                ood_valid_indices
            ),
            "test_count": len(
                ood_test_indices
            ),
        },
        "nested_subset_sizes": sorted(
            subsets.keys()
        ),
        "distributions": {
            "energy": describe(energies),
            "energy_per_particle": describe(
                energies
                / np.asarray(
                    [len(a) for a in frames]
                )
            ),
            "separation": describe(
                separations
            ),
            "force_rms": describe(
                np.asarray(force_rms_values)
            ),
            "torque_rms": describe(
                np.asarray(torque_rms_values)
            ),
            "net_force_norm": describe(
                np.asarray(net_force_values)
            ),
            "torque_balance_residual_norm": (
                describe(
                    np.asarray(
                        torque_balance_values
                    )
                )
            ),
        },
        "files": {
            "manifest": str(manifest_path),
            "random_train": str(
                output_dir / "random_train.xyz"
            ),
            "random_valid": str(
                output_dir / "random_valid.xyz"
            ),
            "random_test": str(
                output_dir / "random_test.xyz"
            ),
            "ood_train": str(
                output_dir / "ood_train.xyz"
            ),
            "ood_valid": str(
                output_dir / "ood_valid.xyz"
            ),
            "ood_test": str(
                output_dir
                / "ood_test_short_range.xyz"
            ),
        },
    }

    summary_path = (
        output_dir
        / "publication_summary.json"
    )

    summary_path.write_text(
        json.dumps(
            summary,
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )

    print()
    print("Publication split generation complete")
    print(f"  Input frames: {len(frames)}")
    print(
        "  Random split: "
        f"{len(train_indices)} train, "
        f"{len(valid_indices)} valid, "
        f"{len(test_indices)} test"
    )
    print(
        "  OOD split: "
        f"{len(ood_train_indices)} train, "
        f"{len(ood_valid_indices)} valid, "
        f"{len(ood_test_indices)} test"
    )
    print(
        "  OOD short-range threshold: "
        f"{ood_threshold:.10g}"
    )
    print(
        "  Exact duplicate count: "
        f"{duplicate_count}"
    )
    print(
        "  Nested subset sizes: "
        f"{sorted(subsets.keys())}"
    )
    print(f"  Output directory: {output_dir}")
    print(f"  Manifest: {manifest_path}")
    print(f"  Summary: {summary_path}")


if __name__ == "__main__":
    main()
