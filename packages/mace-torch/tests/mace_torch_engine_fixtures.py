"""A small differentiable stack, shared by the two derivative-engine suites.

Not a conftest: these are builders the tests call with their own arguments,
and a fixture that takes five parameters is a function with extra steps.
"""

from __future__ import annotations

import numpy as np
import torch
from mace_core.elements import AtomicNumberTable, ResolvedE0s
from mace_core.kernels.precision import PrecisionConfig
from mace_core.neighbors import get_neighborhood
from mace_core.observables import InputSpec, ObservableSpec
from mace_torch.backends.reference import ReferenceBackend
from mace_torch.models import EnergyOutputHead, MACEOutputs, ScaleShiftSpec
from mace_torch.nn import MACEBackbone
from mace_torch.physics import DerivativeEngine

CUTOFF = 5.0
ATOMIC_NUMBERS = [1, 8]
ENERGY = ObservableSpec(
    name="energy", irreps="0e", per_atom=False, units="eV", normalization="none"
)


#: A per-node differentiable input, used to exercise the declared-input path.
#: Its name is the frozen tree's, so the alias rule that turns
#: `d_energy_d_magmom` into `magforces` is exercised as well. Nothing in the
#: engine or the backbone knows the name; only this fixture and its tests do.
NODE_INPUT = InputSpec(
    name="magmom", irreps="1o", per_atom=True, units="muB", differentiable=True
)


def build_engine(seed: int = 0, inputs=()) -> DerivativeEngine:
    """A backbone and output layer with weights, wired to the engine.

    The reference backend starts its contraction weights at zero, so an engine
    built on an untouched one has a flat energy and every derivative claim
    below would hold of the constant function.
    """
    torch.manual_seed(seed)
    backend = ReferenceBackend()
    backbone = MACEBackbone(
        backend,
        atomic_numbers=ATOMIC_NUMBERS,
        num_layers=2,
        num_features=4,
        lmax=2,
        hidden_irreps="0e+1o",
        correlation=2,
        cutoff=CUTOFF,
        avg_num_neighbors=6.0,
        node_inputs=[spec for spec in inputs if spec.per_atom],
    )
    head = EnergyOutputHead(
        ResolvedE0s({"default": {1: -13.6, 8: -2040.0}}),
        ["default"],
        AtomicNumberTable(ATOMIC_NUMBERS),
        ScaleShiftSpec("std", (1.0,), (0.0,)),
        PrecisionConfig(),
    )
    outputs = MACEOutputs(backend, [ENERGY], "0e+1o", 4, 2, energy_head=head)
    generator = torch.Generator().manual_seed(seed + 1)
    with torch.no_grad():
        for module in (backbone, outputs):
            for parameter in module.parameters():
                parameter.copy_(
                    torch.randn(
                        parameter.shape, generator=generator, dtype=parameter.dtype
                    )
                )
    return DerivativeEngine(backbone, outputs, inputs=inputs)


def build_graph(positions, numbers, cell=None, pbc=(False, False, False)) -> dict:
    positions = np.asarray(positions, dtype=float)
    neighborhood = get_neighborhood(positions, CUTOFF, pbc, cell)
    index = {z: i for i, z in enumerate(ATOMIC_NUMBERS)}
    effective = neighborhood.cell if cell is None else np.asarray(cell, dtype=float)
    return {
        "positions": torch.tensor(positions),
        "atomic_numbers": torch.tensor(list(numbers)),
        "element_index": torch.tensor([index[z] for z in numbers]),
        "edge_index": torch.tensor(neighborhood.edge_index),
        "shifts": torch.tensor(neighborhood.shifts),
        "unit_shifts": torch.tensor(neighborhood.unit_shifts),
        "batch": torch.zeros(len(numbers), dtype=torch.long),
        "num_graphs": 1,
        "head": torch.zeros(1, dtype=torch.long),
        "cell": torch.tensor(np.asarray(effective, dtype=float)).reshape(1, 3, 3),
        "pbc": torch.tensor([list(pbc)]),
    }


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
