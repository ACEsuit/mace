import importlib.util
import os
import shutil
import tempfile
from pathlib import Path

import numpy as np
import pytest
from ase.atoms import Atoms

# ---------------------------------------------------------------------------
# Capability model
#
# Each optional requirement of a test (a GPU, an optional dependency, network
# access, an external binary) is a "capability", expressed as a pytest marker
# (registered in pyproject.toml). The contract is:
#
#   * locally, a test whose capability is unavailable is SKIPPED;
#   * in CI, each job exports MACE_REQUIRE_CAPS with the capabilities it
#     guarantees (because it installed them / runs on a GPU host); a test that
#     would skip for one of those capabilities FAILS instead. A job can never
#     again go green while silently skipping the thing it exists to test.
#
# Vendor is deliberately NOT a capability: tests always use device="cuda"
# (valid on ROCm too) and GPU jobs select by marker expression per vendor
# (e.g. "gpu and not cueq" on AMD hosts, since cuEquivariance is NVIDIA-only).
# ---------------------------------------------------------------------------


def _module_available(name: str) -> bool:
    try:
        return importlib.util.find_spec(name) is not None
    except (ImportError, ValueError):
        return False


def _gpu_available() -> bool:
    try:
        import torch  # pylint: disable=import-outside-toplevel

        # True on ROCm builds as well; torch.version.hip distinguishes vendors.
        return torch.cuda.is_available()
    except ImportError:
        return False


def _lammps_available() -> bool:
    # Real import, not find_spec: a broken wheel (e.g. missing libmpi) must
    # read as unavailable.
    try:
        import lammps  # noqa: F401  # pylint: disable=import-outside-toplevel,unused-import

        return True
    except (ImportError, OSError):
        return False


CAPABILITY_PROBES = {
    "gpu": _gpu_available,
    "cueq": lambda: _module_available("cuequivariance"),
    "oeq": lambda: _module_available("openequivariance"),
    "polar": lambda: _module_available("graph_longrange"),
    "les": lambda: _module_available("les"),
    "magnetic": lambda: _module_available("sphericart"),
    "torchsim": lambda: _module_available("torch_sim"),
    "schedulefree": lambda: _module_available("schedulefree"),
    "bin_lammps": _lammps_available,
    # Never autodetected: network access is an explicit opt-in so that PR jobs
    # stay deterministic and downloads live only in jobs that chose them.
    "network": lambda: os.environ.get("MACE_CI_ALLOW_NETWORK", "") == "1",
}

_capability_cache: dict = {}


def capability_available(cap: str) -> bool:
    if cap not in _capability_cache:
        _capability_cache[cap] = CAPABILITY_PROBES[cap]()
    return _capability_cache[cap]


def required_caps() -> set:
    return {
        cap.strip()
        for cap in os.environ.get("MACE_REQUIRE_CAPS", "").split(",")
        if cap.strip()
    }


@pytest.hookimpl(tryfirst=True)
def pytest_runtest_setup(item):
    # tryfirst: must run before the builtin skipping plugin, so that a
    # required-but-unavailable capability fails even if a legacy
    # @pytest.mark.skipif on the same test would have skipped it.
    missing = [
        cap
        for cap in CAPABILITY_PROBES
        if item.get_closest_marker(cap) is not None and not capability_available(cap)
    ]
    if not missing:
        return
    # A guaranteed-but-missing capability outranks any other skip reason: the
    # job promised it, so its absence is a broken job, not a skippable test.
    broken = sorted(set(missing) & required_caps())
    if broken:
        pytest.fail(
            f"MACE_REQUIRE_CAPS guarantees {broken} but unavailable on this "
            f"host — the CI job is broken (bad install or wrong runner), "
            f"not the test.",
            pytrace=False,
        )
    pytest.skip(f"requires capabilities {sorted(missing)}")


# Directory-derived markers: a test file inherits the capability/cost markers
# of the directory it lives in, so a new file can never land in the wrong CI
# job by omission. Hardware (`gpu`) and per-test capabilities stay explicit
# in-file because they are transversal, not directory-shaped.
_TESTS_DIR = Path(__file__).parent
_DIR_MARKERS = {
    ("workflows",): ("slow",),
    ("foundations",): ("network",),
    ("benchmarks",): ("benchmark", "slow"),
    ("extensions", "polar"): ("polar",),
    ("extensions", "les"): ("les",),
    ("extensions", "magnetic"): ("magnetic",),
    ("extensions", "torchsim"): ("torchsim",),
    ("extensions", "schedulefree"): ("schedulefree",),
}


def pytest_collection_modifyitems(config, items):  # pylint: disable=unused-argument
    for item in items:
        try:
            rel = Path(item.path).relative_to(_TESTS_DIR).parts[:-1]
        except ValueError:
            continue
        for prefix, marker_names in _DIR_MARKERS.items():
            if rel[: len(prefix)] == prefix:
                for name in marker_names:
                    item.add_marker(getattr(pytest.mark, name))

    # A guaranteed capability with zero collected tests is the other silent
    # failure mode (module-level skips, empty parametrizations): the job would
    # pass while testing nothing. Runs after auto-marking so directory-derived
    # markers count.
    for cap in required_caps():
        if not any(item.get_closest_marker(cap) for item in items):
            raise pytest.UsageError(
                f"MACE_REQUIRE_CAPS guarantees '{cap}' but no collected test "
                f"carries that marker — nothing would exercise it."
            )


def pytest_configure(config):
    """Validate MACE_REQUIRE_CAPS, isolate cache per xdist worker, and guard
    known plugin reseed issues."""
    unknown = required_caps() - set(CAPABILITY_PROBES)
    if unknown:
        raise pytest.UsageError(
            f"MACE_REQUIRE_CAPS contains unknown capabilities: {sorted(unknown)}; "
            f"known: {sorted(CAPABILITY_PROBES)}"
        )

    os.environ.setdefault("TORCHINDUCTOR_DISABLE_PCH", "1")
    # Ensure inductor does not use stale precompiled headers across test runs.
    try:
        import torch._inductor.config as inductor_config

        inductor_config.cpp_cache_precompile_headers = False
    except (ImportError, AttributeError):
        pass
    if hasattr(config.option, "randomly_reset_seed"):
        # Some environments with pytest-randomly + thinc can produce invalid seeds.
        config.option.randomly_reset_seed = False

    worker = os.environ.get("PYTEST_XDIST_WORKER")
    if worker:
        cache_root = Path(tempfile.gettempdir()) / f"mace_cache_{worker}"
        cache_root.mkdir(parents=True, exist_ok=True)
        os.environ["XDG_CACHE_HOME"] = str(cache_root)
        print(f"[XDIST] Worker {worker} using cache {cache_root}")


# ---------------------------------------------------------------------------
# Shared data fixtures
#
# Canonical copies of the fitting/pretraining configuration fixtures that used
# to be duplicated verbatim across test files. Files whose data differs (extra
# keys, different seeds/sizes) keep their own local fixture, which shadows
# these by name.
# ---------------------------------------------------------------------------


@pytest.fixture(name="fitting_configs")
def fixture_fitting_configs():
    from tests.helpers import make_fitting_configs

    return make_fitting_configs()


@pytest.fixture(scope="session", name="trained_tiny_model_path")
def fixture_trained_tiny_model_path(tmp_path_factory):
    """Train a minimal MACE ONCE per session and return the .model path.

    For tests whose subject is not training itself (export paths, integration
    contracts, smoke tests): they share this artifact instead of each training
    their own model.
    """
    import ase.io

    from tests.helpers import base_mace_params, make_fitting_configs, run_mace_train

    tmp = tmp_path_factory.mktemp("tiny_model")
    ase.io.write(tmp / "fit.xyz", make_fitting_configs())
    params = base_mace_params()
    params.update(
        {
            "name": "tiny",
            "hidden_irreps": "16x0e + 16x1o",
            "checkpoints_dir": str(tmp),
            "model_dir": str(tmp),
            "train_file": str(tmp / "fit.xyz"),
            "max_num_epochs": 6,
            "start_swa": 4,
        }
    )
    run_mace_train(params)
    model_path = tmp / "tiny.model"
    assert model_path.exists()
    return model_path


@pytest.fixture(name="pretraining_configs")
def fixture_pretraining_configs():
    configs = []
    for _ in range(10):
        atoms = Atoms(
            numbers=[8, 1, 1],
            positions=np.random.rand(3, 3) * 3,
            cell=[5, 5, 5],
            pbc=[True] * 3,
        )
        atoms.info["REF_energy"] = np.random.normal(0, 1)
        atoms.arrays["REF_forces"] = np.random.normal(0, 1, size=(3, 3))
        atoms.info["REF_stress"] = np.random.normal(0, 1, size=6)
        configs.append(atoms)
    configs.append(
        Atoms(numbers=[8], positions=[[0, 0, 0]], cell=[6] * 3, pbc=[True] * 3),
    )
    configs.append(
        Atoms(numbers=[1], positions=[[0, 0, 0]], cell=[6] * 3, pbc=[True] * 3)
    )
    configs[-2].info["REF_energy"] = -2.0
    configs[-2].info["config_type"] = "IsolatedAtom"
    configs[-1].info["REF_energy"] = -4.0
    configs[-1].info["config_type"] = "IsolatedAtom"
    return configs


def pytest_runtest_logreport(report):
    """Prints a line about available disc space & test duration after each test."""
    if report.when == "call":
        _total, _used, free = shutil.disk_usage("/")
        print(
            f"\n[METRICS] "
            f"{report.nodeid}: "
            f"disc: {free / (2**30):.3f}GB free, "
            f"time: {report.duration:.2f}s"
        )
