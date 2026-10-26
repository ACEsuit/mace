"""mace-jax builds graphs with matscipy unless the config names a backend."""

import subprocess
import sys

from mace_core.neighbors import NeighborListConfig
from mace_core.neighbors.brute_force import BruteForceBackend
from mace_core.neighbors.matscipy_backend import MatscipyBackend
from mace_jax.neighbors import DEFAULT_NEIGHBOR_BACKEND, resolve_neighbor_backend


def test_mace_jax_defaults_to_matscipy():
    assert DEFAULT_NEIGHBOR_BACKEND == "matscipy"
    assert isinstance(resolve_neighbor_backend(), MatscipyBackend)


def test_resolving_the_default_imports_neither_vesin_nor_torch():
    """vesin would import torch whenever torch is installed."""
    probe = (
        "import sys; from mace_jax.neighbors import resolve_neighbor_backend; "
        "resolve_neighbor_backend(); "
        "print('vesin' in sys.modules, 'torch' in sys.modules)"
    )
    result = subprocess.run(
        [sys.executable, "-c", probe], capture_output=True, text=True, check=True
    )
    assert result.stdout.split() == ["False", "False"]


def test_a_backend_named_in_the_config_wins():
    config = NeighborListConfig(kind="brute_force")
    assert isinstance(resolve_neighbor_backend(config), BruteForceBackend)
