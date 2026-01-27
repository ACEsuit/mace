import os
import shutil
import tempfile
from pathlib import Path


def pytest_configure():
    """Isolate cache per xdist worker to avoid concurrent model downloads."""
    worker = os.environ.get("PYTEST_XDIST_WORKER")
    if worker:
        cache_root = Path(tempfile.gettempdir()) / f"mace_cache_{worker}"
        cache_root.mkdir(parents=True, exist_ok=True)
        os.environ["XDG_CACHE_HOME"] = str(cache_root)


def pytest_runtest_logreport(report):
    """Prints a line about available disc space & test duration after each test."""
    if report.when == "call":
        total, used, free = shutil.disk_usage("/")
        print(
            f"\n[METRICS] "
            f"{report.nodeid}: "
            f"disc: {free / (2**30):.3f}GB free, "
            f"time: {report.duration:.2f}s"
        )
