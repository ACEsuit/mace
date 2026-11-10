"""Small runnable data-to-model workflows; no implicit scientific-data downloads."""

import argparse
import json
from pathlib import Path

import numpy as np
import yaml
from ase import Atoms, units
from ase.build import molecule
from ase.io import read, write
from ase.md.langevin import Langevin
from ase.md.velocitydistribution import MaxwellBoltzmannDistribution


def smoke_data(name, output):
    """Synthetic harmonic labels test plumbing, not a published potential."""
    atoms = molecule("CH4" if name == "ani1x" else "H2O")
    if name == "md22":
        atoms = molecule("C6H6")
    if name == "water":
        atoms.set_cell([6, 6, 6])
        atoms.pbc = True
        atoms.center()
    origin = atoms.positions.copy()
    generator = np.random.default_rng(42)
    frames = []
    for number in sorted(set(atoms.numbers)):
        isolated = Atoms(numbers=[number], positions=[[0, 0, 0]])
        isolated.info.update(REF_energy=-float(number), config_type="IsolatedAtom")
        isolated.arrays["REF_forces"] = np.zeros((1, 3))
        frames.append(isolated)
    for _ in range(8):
        frame = atoms.copy()
        delta = generator.normal(0, 0.025, origin.shape)
        frame.positions = origin + delta
        frame.info["REF_energy"] = -float(sum(frame.numbers)) + float(
            np.sum(delta**2) / 2
        )
        frame.arrays["REF_forces"] = -delta
        frames.append(frame)
    write(output, frames)


def prepare(name, output, dataset=None):
    output = Path(output).resolve()
    output.mkdir(parents=True, exist_ok=True)
    data = Path(dataset).resolve() if dataset else output / "synthetic.xyz"
    if dataset is None:
        smoke_data(name, data)
    config = {
        "runtime": {"work_dir": str(output), "name": name, "seed": 42},
        "data": {
            "heads": {
                "default": {"train_file": str(data), "e0s": {"kind": "isolated_atoms"}}
            },
            "valid_fraction": 0.25,
            "pin_memory": False,
        },
        "model": {
            "observables": ["energy", "forces"],
            "r_max": 3.5,
            "num_interactions": 2,
            "num_channels": 2,
            "hidden_irreps": "0e+1o",
            "max_ell": 1,
            "correlation": 2,
            "readout": {"mlp_irreps": "3x0e"},
        },
        "training": {
            "max_num_epochs": 1,
            "batch_size": 2,
            "valid_batch_size": 2,
            "lr": 0.01,
            "scheduler": {"kind": "constant"},
        },
        "loss": {"weights": {"energy": 1.0, "forces": 1.0}},
    }
    path = output / "run.yaml"
    path.write_text(yaml.safe_dump(config, sort_keys=False))
    return path


def nvt(model, structures, output, steps=5):
    from mace_torch.calculators import MACECalculator

    atoms = next(frame for frame in read(structures, ":") if len(frame) > 1)
    atoms.calc = MACECalculator(model_paths=model, device="cpu")
    MaxwellBoltzmannDistribution(
        atoms, temperature_K=100, rng=np.random.default_rng(81)
    )
    dynamics = Langevin(
        atoms,
        0.1 * units.fs,
        temperature_K=100,
        friction=0.01 / units.fs,
        rng=np.random.default_rng(82),
    )
    dynamics.run(steps)
    record = {
        "energy_eV": atoms.get_potential_energy(),
        "positions_A": atoms.positions.tolist(),
        "steps": steps,
    }
    Path(output).write_text(json.dumps(record, indent=2) + "\n")
    return record


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    training = commands.add_parser("prepare")
    training.add_argument("kind", choices=["md22", "ani1x", "water"])
    training.add_argument("output", type=Path)
    training.add_argument("--dataset", type=Path)
    md = commands.add_parser("nvt")
    md.add_argument("--model", required=True)
    md.add_argument("--structures", required=True)
    md.add_argument("--output", required=True)
    md.add_argument("--steps", type=int, default=5)
    args = parser.parse_args()
    if args.command == "prepare":
        print(prepare(args.kind, args.output, args.dataset))
    else:
        nvt(args.model, args.structures, args.output, args.steps)
