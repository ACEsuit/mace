"""Joint readouts retain vectors, fixed-charge physics and checkpoint safety."""

import numpy as np
import pytest
import torch
from conftest import fp64_only
from mace_core.config.resolved import ResolvedConfig
from mace_core.data import Configuration
from mace_core.data.backend import DatasetStatistics
from mace_core.data.normalization import NormalizationValues
from mace_core.elements import AtomicNumberTable, ResolvedE0s
from mace_core.observables import DEFAULT_CATALOGUE
from mace_torch.data.graphs import graph_from_configuration
from mace_torch.graph import collate
from mace_torch.models.build import build_model
from mace_torch.models.normalization import ObservableNormalization
from mace_torch.models.outputs import E_ANGSTROM_PER_DEBYE, fixed_charge_dipole

from tests.golden.harness import CLOSED_FORM_FP64, FP64_CPU_REFERENCE


def close(actual, expected, row=FP64_CPU_REFERENCE):
    torch.testing.assert_close(actual, expected, atol=row.atol, rtol=row.rtol)


def build(heads=("default",)):
    config = ResolvedConfig.from_dict(
        {
            "runtime": {"seed": 12},
            "data": {"heads": {name: {} for name in heads}},
            "model": {
                "observables": ["energy", "dipole"],
                "num_channels": 4,
                "num_interactions": 2,
                "correlation": 2,
                "hidden_irreps": "0e+1o+2e",
                "max_ell": 2,
                "readout": {"mlp_irreps": "4x0e+4x1o"},
            },
        }
    )
    engine, _ = build_model(
        config,
        DEFAULT_CATALOGUE,
        z_table=AtomicNumberTable([1]),
        heads=heads,
        e0s=ResolvedE0s({name: {1: 0.0} for name in heads}),
        statistics=DatasetStatistics(avg_num_neighbors=1.0, std=1.0),
    )
    return engine, config


def batch():
    configuration = Configuration(
        atomic_numbers=np.array([1, 1]),
        positions=np.array([[0.2, 0.3, -0.1], [0.9, -0.1, 0.2]]),
        properties={"charges": np.array([-0.3, 0.7])},
    )
    return collate(
        [
            graph_from_configuration(
                configuration, cutoff=5.0, z_table=AtomicNumberTable([1])
            )
        ],
        dtype=torch.float64,
    )


@fp64_only
def test_joint_normalization_leaves_fixed_charges_in_physical_units():
    engine, _ = build(("first", "second"))
    normalizer = ObservableNormalization(
        DEFAULT_CATALOGUE.observable("dipole"),
        [
            NormalizationValues(2.0, (0.0, 0.0, 0.0)),
            NormalizationValues(3.0, (0.0, 0.0, 0.0)),
        ],
        torch.float64,
    )
    graph = batch()
    baseline = fixed_charge_dipole(
        graph["positions"], graph["charges"], graph["batch"], 1
    )
    expected = []
    for head in (0, 1):
        graph["head"].fill_(head)
        expected.append(engine(graph, compute=()).dipole.detach())
    engine.backbone.outputs.normalization["dipole"] = normalizer
    for head, scale in enumerate((2.0, 3.0)):
        graph["head"].fill_(head)
        close(
            engine(graph, compute=()).dipole,
            scale * (expected[head] - baseline) + baseline,
        )


@fp64_only
def test_fixed_charge_term_supports_second_derivatives_and_missing_charges():
    graph = batch()
    positions = graph["positions"].requires_grad_()
    charges = graph["charges"].requires_grad_()

    def evaluate(positions, charges):
        return fixed_charge_dipole(positions, charges, graph["batch"], 1)

    assert torch.autograd.gradgradcheck(evaluate, (positions, charges))
    close(
        fixed_charge_dipole(positions, None, graph["batch"], 1),
        torch.zeros(1, 3),
        CLOSED_FORM_FP64,
    )
    close(
        evaluate(positions, charges),
        (positions * charges[:, None]).sum(0)[None] / E_ANGSTROM_PER_DEBYE,
        CLOSED_FORM_FP64,
    )


@fp64_only
def test_scalar_final_layer_checkpoint_is_refused(tmp_path):
    from mace_core.config.provenance import e0_details
    from mace_core.metadata import ConfigRecord, HeadSummary, ModelMetadata, Provenance
    from mace_torch.backends.reference import ReferenceBackend
    from mace_torch.deploy.loader import load_deployed
    from mace_torch.models import MACEModel
    from mace_torch.physics import DerivativeEngine
    from mace_torch.serialization import CheckpointError
    from mace_torch.train.checkpoint import write_model

    engine, config = build()
    # Recreate the former architecture, not an incomplete subset of a new file.
    old_model = MACEModel(
        ReferenceBackend(),
        config.model,
        atomic_numbers=[1],
        observables=engine.backbone.outputs.specs,
        energy_head=engine.backbone.outputs.energy_head,
        last_layer_irreps="0e",
    )
    old_engine = DerivativeEngine(
        old_model,
        DEFAULT_CATALOGUE.observable("energy"),
        responses=[DEFAULT_CATALOGUE.observable("dipole")],
    )
    metadata = ModelMetadata(
        elements=[1],
        config=ConfigRecord.from_config(config),
        provenance=Provenance(versions={"mace-torch-v1": "before-joint-fix"}),
        heads={
            "default": HeadSummary(
                e0=e0_details(config.data.heads["default"].e0s, {"H": 0.0})
            )
        },
    )
    path = write_model(tmp_path / "old", old_engine, metadata)
    with pytest.raises(CheckpointError, match="different layout"):
        load_deployed(path)
