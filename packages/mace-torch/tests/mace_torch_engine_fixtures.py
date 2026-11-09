"""A small differentiable stack, shared by the two derivative-engine suites.

Not a conftest: these are builders the tests call with their own arguments,
and a fixture that takes five parameters is a function with extra steps.
"""

from __future__ import annotations

import numpy as np
import torch
from mace_core.config.model import ModelConfig
from mace_core.config.precision import PrecisionConfig
from mace_core.data import Configuration
from mace_core.elements import AtomicNumberTable, ResolvedE0s
from mace_core.neighbors import NeighborListConfig
from mace_core.observables import InputSpec, ObservableSpec
from mace_torch.backends.reference import ReferenceBackend
from mace_torch.graph import build_batch
from mace_torch.models import EnergyReadout, MACEOutputs, ScaleShiftSpec
from mace_torch.neighbors import resolve_neighbor_backend
from mace_torch.nn import MACEBackbone
from mace_torch.physics import DerivativeEngine

CUTOFF = 5.0
ATOMIC_NUMBERS = [1, 8]
#: The energy, declaring the two derivatives that have names of their own. They
#: are declared rather than derived because a name and a sign are a property of
#: the quantity, not of a rule: `forces` and `magforces` are both the negative
#: gradient, and anything else takes `d_energy_d_<input>` with the gradient's
#: own sign.
ENERGY = ObservableSpec(
    name="energy",
    irreps="0e",
    per_atom=False,
    units="eV",
    derivatives=[
        {"wrt": "pos", "name": "forces", "sign": -1, "units": "eV/A"},
        {"wrt": "magmom", "name": "magforces", "sign": -1},
    ],
)


#: A per-node differentiable input, used to exercise the declared-input path.
#: Its name is the frozen tree's, so the alias rule that turns
#: `d_energy_d_magmom` into `magforces` is exercised as well. Nothing in the
#: engine or the backbone knows the name; only this fixture and its tests do.
NODE_INPUT = InputSpec(
    name="magmom", irreps="1o", per_atom=True, units="muB", differentiable=True
)


class FeatureReadoutModel(torch.nn.Module):
    """Compose the synthetic feature extractor and readouts before differentiation."""

    def __init__(self, backbone, outputs):
        super().__init__()
        self.backbone = backbone
        self.outputs = outputs

    def forward(self, graph):
        return self.outputs(graph, list(self.backbone(graph).layers))

    def canonical_components(self):
        return {"backbone": self.backbone, "outputs": self.outputs}


def build_engine(seed: int = 0, inputs=(), backend=None) -> DerivativeEngine:
    """A backbone and output layer with weights, wired to the engine.

    The reference backend starts its contraction weights at zero, so an engine
    built on an untouched one has a flat energy and every derivative claim
    below would hold of the constant function.
    """
    torch.manual_seed(seed)
    backend = backend or ReferenceBackend()
    backbone = MACEBackbone(
        backend,
        atomic_numbers=ATOMIC_NUMBERS,
        config=ModelConfig(
            num_interactions=2,
            num_channels=4,
            max_ell=2,
            hidden_irreps="0e+1o",
            correlation=2,
            r_max=CUTOFF,
        ),
        avg_num_neighbors=6.0,
        node_inputs=[spec for spec in inputs if spec.per_atom],
    )
    head = EnergyReadout(
        ResolvedE0s({"default": {1: -13.6, 8: -2040.0}}),
        ["default"],
        AtomicNumberTable(ATOMIC_NUMBERS),
        ScaleShiftSpec("std", (1.0,), (0.0,)),
        PrecisionConfig(),
    )
    outputs = MACEOutputs(backend, [ENERGY], ["0e+1o", "0e"], 4, energy_readout=head)
    generator = torch.Generator().manual_seed(seed + 1)
    with torch.no_grad():
        for module in (backbone, outputs):
            for parameter in module.parameters():
                parameter.copy_(
                    torch.randn(
                        parameter.shape, generator=generator, dtype=parameter.dtype
                    )
                )
    return DerivativeEngine(
        FeatureReadoutModel(backbone, outputs), ENERGY, inputs=inputs
    )


def build_graph(positions, numbers, cell=None, pbc=(False, False, False)) -> dict:
    configuration = Configuration(
        atomic_numbers=np.asarray(numbers),
        positions=np.asarray(positions, dtype=float),
        cell=np.zeros((3, 3)) if cell is None else np.asarray(cell, dtype=float),
        pbc=tuple(pbc),
    )
    return build_batch(
        [configuration],
        atomic_number_table=AtomicNumberTable(ATOMIC_NUMBERS),
        cutoff=CUTOFF,
        neighbor_backend=resolve_neighbor_backend(NeighborListConfig()),
        dtype=torch.float64,
    )


def node_input_values(count: int, seed: int = 7):
    generator = np.random.default_rng(seed)
    return torch.tensor(generator.normal(size=(count, 3)) * 0.4)


def molecule():
    return (
        np.array(
            [[0.0, 0.0, 0.0], [0.95, 0.0, 0.0], [-0.24, 0.93, 0.0], [2.6, 0.4, 0.2]]
        ),
        [8, 1, 1, 1],
    )


def crystal():
    return (
        np.array([[0.0, 0.0, 0.0], [1.7, 1.7, 0.0], [1.7, 0.0, 1.7], [0.0, 1.7, 1.7]]),
        [8, 1, 1, 1],
        np.eye(3) * 3.4,
    )


def energy_of(engine, positions, numbers, cell=None, pbc=(False, False, False)):
    graph = build_graph(positions, numbers, cell, pbc)
    return float(engine(graph, compute=()).total_energy.detach())
