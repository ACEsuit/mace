"""Standalone JAX tests: optional network/GPU gates without importing torch."""

import os

import pytest


def pytest_addoption(parser):
    parser.addoption(
        "--jax-model-artifacts",
        help="JSON mapping canonical IDs to local converted checkpoints",
    )


@pytest.fixture(autouse=True)
def capabilities(request):
    required = set(os.environ.get("MACE_REQUIRE_CAPS", "").split(","))
    for name in ("network", "gpu"):
        if request.node.get_closest_marker(name) is None:
            continue
        if name == "network":
            available = os.environ.get("MACE_CI_ALLOW_NETWORK") == "1"
        else:
            import jax

            try:
                available = bool(jax.devices("gpu"))
            except RuntimeError:
                available = False
        if not available:
            message = f"JAX test requires {name}"
            if name in required:
                pytest.fail(message)
            pytest.skip(message)
