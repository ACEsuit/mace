"""Inference imports and model construction work with training unavailable."""

import subprocess
import sys
import textwrap


def test_inference_can_build_and_serialize_without_importing_training(tmp_path):
    program = r"""
        import importlib.abc
        import sys

        class NoTraining(importlib.abc.MetaPathFinder):
            def find_spec(self, fullname, path=None, target=None):
                if fullname == "mace_torch.train" or fullname.startswith(
                    "mace_torch.train."
                ):
                    raise AssertionError("inference imports training: " + fullname)

        sys.meta_path.insert(0, NoTraining())

        import torch
        from mace_core.config.resolved import ResolvedConfig
        from mace_core.data.backend import DatasetStatistics
        from mace_core.elements import AtomicNumberTable, ResolvedE0s
        from mace_core.metadata import (
            ConfigRecord, E0Details, HeadSummary, ModelMetadata, Provenance,
        )
        from mace_core.observables import DEFAULT_CATALOGUE
        from mace_torch.models.build import build_model
        from mace_torch.calculators.ase_calculator import MACECalculator
        from mace_torch.deploy.loader import load_deployed
        from mace_torch.deploy.neutral_io import import_neutral
        from mace_torch.deploy.openmm import OpenMMEvaluator
        from mace_torch.serialization import canonical_state, write_model

        config = ResolvedConfig.model_validate({
            "data": {"heads": {"default": {}}},
            "model": {
                "num_channels": 2, "num_interactions": 1,
                "max_ell": 1, "hidden_irreps": "0e+1o", "correlation": 1,
            },
        })
        engine, _ = build_model(
            config, DEFAULT_CATALOGUE, z_table=AtomicNumberTable([1]),
            heads=("default",), e0s=ResolvedE0s({"default": {1: 0.0}}),
            statistics=DatasetStatistics(avg_num_neighbors=2.0),
        )
        metadata = ModelMetadata(
            config=ConfigRecord(resolved=config.model_dump(mode="json")),
            elements=[1],
            provenance=Provenance(versions={}),
            heads={"default": HeadSummary(
                e0=E0Details(source="explicit", values={"H": 0.0})
            )},
        )
        write_model(sys.argv[1], engine, metadata)
        loaded = load_deployed(sys.argv[1])
        before, after = canonical_state(engine), canonical_state(loaded.engine)
        assert before.keys() == after.keys()
        for module, state in before.items():
            for name, tensor in state.items():
                assert torch.equal(tensor, after[module][name]), (module, name)
        assert not any(name.startswith("mace_torch.train") for name in sys.modules)
    """
    result = subprocess.run(
        [sys.executable, "-c", textwrap.dedent(program), str(tmp_path / "model")],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stdout + result.stderr
