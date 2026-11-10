"""Native artifacts converted from the committed frozen checkpoints."""

from pathlib import Path

from mace_core.data.backend import DatasetStatistics
from mace_core.elements import ResolvedE0s
from mace_core.observables import DEFAULT_CATALOGUE
from mace_torch.deploy.loader import load_deployed
from mace_torch.models.build import build_model
from mace_torch.train import write_model

GOLDEN = Path(__file__).resolve().parents[1] / "golden"
FIXTURES = GOLDEN / "fixtures"
ARTIFACTS = Path(__file__).with_name("artifacts")


def reference(name):
    return GOLDEN / "references" / f"{name}_e3nn_cpu_fp64.json"


def model_checkpoint(directory, scale_shift=True, layers=2):
    """Build small seeded native models for deployment ABI tests."""
    import torch

    torch.manual_seed(123)
    name = "tiny_scaleshift" if scale_shift else "tiny_mace"
    loaded = load_deployed(ARTIFACTS / f"{name}.json")
    config = loaded.config.model_copy(
        update={
            "model": loaded.config.model.model_copy(update={"num_interactions": layers})
        }
    )
    engine, _ = build_model(
        config,
        DEFAULT_CATALOGUE,
        z_table=loaded.z_table,
        heads=loaded.heads,
        e0s=ResolvedE0s(loaded.e0s),
        statistics=DatasetStatistics(),
    )
    # Preserve measured normalizations and all compatible frozen parameters.
    # A bare DatasetStatistics has zero scales and is not an inference fixture.
    from mace_torch.serialization import canonical_state, load_canonical_state

    state = canonical_state(engine)
    frozen = canonical_state(loaded.engine)
    for module, weights in state.items():
        for name, value in weights.items():
            original = frozen.get(module, {}).get(name)
            if original is not None and original.shape == value.shape:
                weights[name] = original
    load_canonical_state(engine, state)
    metadata = loaded.metadata.model_copy(
        update={
            "config": loaded.metadata.config.model_copy(
                update={"resolved": config.to_resolved_dict()}
            )
        }
    )
    return write_model(directory / "model", engine, metadata)
