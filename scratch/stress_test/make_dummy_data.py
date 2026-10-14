"""Generate small dummy training/validation data for the stress fast-path test.

Writes periodic water configs (energy + forces + stress) plus the two isolated-atom
reference configs that MACE needs for E0s. No internet / external data required, so this
can run inside a Raven batch job (which has no internet) before launching training.

Usage:
    python make_dummy_data.py --out_dir /ptmp/.../stress_test_data [--n_train 40]
"""

import argparse
from pathlib import Path

import ase.io
import numpy as np
from ase.atoms import Atoms


def _water_configs(n, rng):
    base = Atoms(
        numbers=[8, 1, 1],
        positions=[[0, -2.0, 0], [1, 0, 0], [0, 1, 0]],
        cell=[4] * 3,
        pbc=[True] * 3,
    )
    configs = []
    for _ in range(n):
        c = base.copy()
        c.positions += rng.normal(0.1, size=c.positions.shape)
        c.info["REF_energy"] = float(rng.normal(0.1))
        c.new_array("REF_forces", rng.normal(0.1, size=c.positions.shape))
        c.info["REF_stress"] = rng.normal(0.1, size=6)
        configs.append(c)
    return configs


def _isolated_atoms():
    o = Atoms(numbers=[8], positions=[[0, 0, 0]], cell=[6] * 3, pbc=[True] * 3)
    h = Atoms(numbers=[1], positions=[[0, 0, 0]], cell=[6] * 3, pbc=[True] * 3)
    o.info["REF_energy"] = 0.0
    o.info["config_type"] = "IsolatedAtom"
    h.info["REF_energy"] = 0.0
    h.info["config_type"] = "IsolatedAtom"
    return [o, h]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out_dir", required=True)
    parser.add_argument("--n_train", type=int, default=40)
    parser.add_argument("--n_valid", type=int, default=10)
    parser.add_argument("--seed", type=int, default=5)
    args = parser.parse_args()

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(args.seed)

    train = _isolated_atoms() + _water_configs(args.n_train, rng)
    valid = _water_configs(args.n_valid, rng)
    ase.io.write(out / "train.xyz", train)
    ase.io.write(out / "valid.xyz", valid)
    print(
        f"wrote {out/'train.xyz'} ({len(train)} configs) and "
        f"{out/'valid.xyz'} ({len(valid)} configs)"
    )


if __name__ == "__main__":
    main()
