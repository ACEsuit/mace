from pathlib import Path
import numpy as np
import ase.io
from ase import Atoms

def _make_small_real_head_dataset(tmp_path: Path):
    """Create a tiny synthetic dataset for a non-pt head (e.g., DFT)."""
    # Two isolated atoms to define E0s and a few random molecules
    configs = [
        Atoms(numbers=[8], positions=[[0, 0, 0]], cell=[6] * 3),
        Atoms(numbers=[1], positions=[[0, 0, 0]], cell=[6] * 3),
    ]
    configs[0].info["REF_energy"] = 0.0
    configs[0].info["config_type"] = "IsolatedAtom"
    configs[1].info["REF_energy"] = 0.0
    configs[1].info["config_type"] = "IsolatedAtom"

    rng = np.random.default_rng(5)
    for _ in range(4):
        at = Atoms(
            numbers=[8, 1, 1],
            positions=rng.normal(scale=0.1, size=(3, 3)),
            cell=[5, 5, 5],
            pbc=[True, True, True],
        )
        at.info["REF_energy"] = float(rng.normal(0.0))
        at.arrays["REF_forces"] = rng.normal(0.0, 1.0, size=(3, 3))
        at.info["REF_stress"] = rng.normal(0.0, 1.0, size=6)
        configs.append(at)

    real_path = tmp_path / "fit_real_head.xyz"
    ase.io.write(real_path, configs)
    return real_path
