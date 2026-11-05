"""Native publication checks share the existing network foundations job."""

import os

import pytest

if os.getenv("MACE_REGISTRY_BADGES"):
    from mace_core import model_registry as registry
    import mace_torch  # noqa: F401; the nightly promises this installation
else:
    registry = pytest.importorskip("mace_core.model_registry")
    pytest.importorskip("mace_torch")
from tests.foundations.registry_badges import evaluate_entry, write_row


@pytest.mark.parametrize(
    "model_id", [entry.model_id for entry in registry.list_models()]
)
def test_torch_registry_badge(model_id):
    row = evaluate_entry(registry.get_model(model_id), framework="torch")
    write_row(os.getenv("MACE_REGISTRY_BADGES", ".pytest_cache/registry-badges"), row)
    assert row["status"] != "failed", row.get("error")


def test_jax_registry_badges():
    if not os.getenv("MACE_REGISTRY_BADGES"):
        pytest.importorskip("mace_jax")
    from mace_jax.models.registry import SUPPORTED_MODELS

    failures = []
    for model_id in SUPPORTED_MODELS:
        row = evaluate_entry(registry.get_model(model_id), framework="jax")
        write_row(
            os.getenv("MACE_REGISTRY_BADGES", ".pytest_cache/registry-badges"), row
        )
        if row["status"] == "failed":
            failures.append((model_id, row["error"]))
    assert not failures, failures
