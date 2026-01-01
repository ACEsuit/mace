import shutil


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
