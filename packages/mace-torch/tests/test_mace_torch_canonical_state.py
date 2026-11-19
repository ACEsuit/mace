"""What a model saves, and that it is enough to rebuild the model.

A tensor in the state dict that no module returns from ``to_canonical`` would be
lost by a checkpoint written from the canonical state, silently, and come back
as whatever the model was built with. So every configuration that changes what
the backbone holds is saved from one model and loaded into another built with
other weights and another neighbour count, and the two must then agree bit for
bit. A tensor that was not saved is one they disagree on.
"""

import re
from typing import Any

import numpy as np
import pytest
import torch
from mace_core.config import (
    AgnesiTransformConfig,
    BesselBasisConfig,
    ChebyshevBasisConfig,
    GaussianBasisConfig,
    ModelConfig,
    SoftTransformConfig,
)
from mace_core.kernels import Precision
from mace_core.observables import InputSpec
from mace_torch.backends.reference import ReferenceBackend
from mace_torch.kernels import (
    canonical_state,
    initialize_model_weights,
    load_canonical_state,
)
from mace_torch.nn.backbone import MACEBackbone
from mace_torch.nn.radial import ZBLBasis

ELEMENTS = (1, 6, 8)
R_MAX = 3.0

#: Every option that changes which tensors the backbone holds.
CONFIGURATIONS: dict[str, dict[str, Any]] = {
    "residual": {},
    "plain_first": {"interaction_first": "plain"},
    "density": {"interaction_first": "density", "interaction": "density_residual"},
    "nonlinear": {
        "interaction": "nonlinear",
        "edge_irreps": "4x0e+4x1o",
        "use_edge_irreps_first": True,
    },
    "bessel_trainable": {"radial_basis": BesselBasisConfig(trainable=True)},
    "gaussian_trainable": {"radial_basis": GaussianBasisConfig(trainable=True)},
    "chebyshev": {"radial_basis": ChebyshevBasisConfig()},
    "agnesi_trainable": {"distance_transform": AgnesiTransformConfig(trainable=True)},
    "soft_trainable": {"distance_transform": SoftTransformConfig(trainable=True)},
    "agnostic_product": {"use_agnostic_product": True},
    "envelope_outside": {"apply_cutoff": False},
}

NODE_INPUT = InputSpec(name="magmom", irreps="1o", per_atom=True, units="muB")


def _backbone(name: str, avg_num_neighbors: float, seed: int) -> MACEBackbone:
    config = ModelConfig(
        r_max=R_MAX,
        num_interactions=2,
        num_channels=4,
        max_L=1,
        max_ell=2,
        correlation=2,
        radial_mlp=(8,),
        **CONFIGURATIONS[name],
    )
    precisions: dict[torch.dtype, Precision] = {
        torch.float64: "float64",
        torch.float32: "float32",
    }
    precision = precisions[torch.get_default_dtype()]
    model = MACEBackbone(
        ReferenceBackend(),
        config,
        ELEMENTS,
        avg_num_neighbors=avg_num_neighbors,
        precision=precision,
        node_inputs=(NODE_INPUT,),
    )
    initialize_model_weights(model, seed)
    return model


def _graph() -> dict[str, torch.Tensor]:
    """A small aperiodic cluster, every pair within the cutoff an edge."""
    dtype = torch.get_default_dtype()
    generator = np.random.default_rng(0)
    positions = generator.uniform(-1.2, 1.2, size=(5, 3))
    distances = np.linalg.norm(positions[:, None] - positions[None], axis=-1)
    sender, receiver = np.nonzero((distances < R_MAX) & (distances > 0.0))
    edge_index = torch.tensor(np.stack([sender, receiver]), dtype=torch.int64)
    return {
        "positions": torch.tensor(positions, dtype=dtype),
        "node_atomic_numbers": torch.tensor([8, 1, 1, 6, 1]),
        "edge_index": edge_index,
        "shifts": torch.zeros(edge_index.shape[1], 3, dtype=dtype),
        "magmom": torch.tensor(generator.normal(size=(5, 3)), dtype=dtype),
    }


def _trained(name: str) -> MACEBackbone:
    """A backbone whose every saved number differs from a freshly built one's,
    trainable constants and the neighbour count included."""
    model = _backbone(name, avg_num_neighbors=3.7, seed=1)
    generator = torch.Generator().manual_seed(5)
    with torch.no_grad():
        for parameter in model.parameters():
            noise = torch.randn(parameter.shape, generator=generator)
            parameter.add_(0.01 * noise.to(parameter.dtype))
    return model


@pytest.mark.parametrize("name", sorted(CONFIGURATIONS))
def test_a_saved_state_rebuilds_the_model_bit_for_bit(name):
    source = _trained(name)
    target = _backbone(name, avg_num_neighbors=1.0, seed=2)
    load_canonical_state(target, canonical_state(source))
    source_state, target_state = source.state_dict(), target.state_dict()
    assert source_state.keys() == target_state.keys()
    for key, tensor in source_state.items():
        assert torch.equal(tensor, target_state[key]), key
    graph = _graph()
    with torch.no_grad():
        for ours, theirs in zip(
            source(graph).layers, target(graph).layers, strict=True
        ):
            assert torch.equal(ours, theirs)


def test_the_neighbour_count_is_saved_where_it_is_read():
    """A block that normalizes by a learned density has no neighbour count."""
    plain = canonical_state(_backbone("residual", 3.7, seed=0))
    density = canonical_state(_backbone("density", 3.7, seed=0))
    assert float(plain["interactions.1.body"]["neighbours"]) == pytest.approx(3.7)
    assert not [path for path in density if path.endswith(".body")]


def test_untrained_constants_are_not_saved():
    """A basis or transform built without ``trainable`` is rebuilt from the config."""
    state = canonical_state(_backbone("residual", 1.0, seed=0))
    assert not [path for path in state if path.startswith("radial_embedding")]


@pytest.mark.parametrize("trainable", [False, True])
def test_the_repulsion_saves_its_constants_only_when_trained(trainable):
    repulsion = ZBLBasis(trainable=trainable)
    state = repulsion.to_canonical()
    assert set(state) == (
        {"screening_length_exponent", "screening_length_prefactor"}
        if trainable
        else set()
    )
    if trainable:
        moved = {name: value + 0.1 for name, value in state.items()}
        repulsion.load_canonical(moved)
        for name, value in repulsion.to_canonical().items():
            assert torch.equal(value, moved[name])


# ---------------------------------------------------------------------------
# A state that does not fit
# ---------------------------------------------------------------------------


def _state_without(state, path, name):
    changed = {key: dict(value) for key, value in state.items()}
    del changed[path][name]
    return changed


@pytest.mark.parametrize(
    ("change", "message"),
    [
        (
            lambda state: _state_without(state, "interactions.0.body", "neighbours"),
            "interactions.0.body: missing 'neighbours'",
        ),
        (
            lambda state: {**state, "extra": {"weight": torch.zeros(1)}},
            "extra: no module here saves state",
        ),
        (
            lambda state: {**state, "radial_embedding": {}},
            "radial_embedding: no module here saves state",
        ),
        (
            lambda state: {**state, "": {}},
            "<root>: no module here saves state",
        ),
        (
            lambda state: {
                **state,
                "interactions.0.body": {
                    **state["interactions.0.body"],
                    "extra": torch.zeros(1),
                },
            },
            "interactions.0.body: unexpected 'extra'",
        ),
        (
            lambda state: {
                **state,
                "node_embedding": {
                    **state["node_embedding"],
                    "weight": torch.zeros(5),
                },
            },
            "'weight' has shape (5,)",
        ),
    ],
)
def test_a_state_that_does_not_fit_is_refused_and_nothing_is_loaded(change, message):
    target = _backbone("residual", avg_num_neighbors=1.0, seed=2)
    before = {key: tensor.clone() for key, tensor in target.state_dict().items()}
    state = canonical_state(_trained("residual"))
    with pytest.raises(ValueError, match=re.escape(message)):
        load_canonical_state(target, change(state))
    for key, tensor in target.state_dict().items():
        assert torch.equal(tensor, before[key]), key


def test_a_tensor_no_module_would_save_is_refused():
    """A backend's own trainable basis without the canonical pair, for one."""

    class Basis(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.frequencies = torch.nn.Parameter(torch.ones(4))

    with pytest.raises(ValueError, match=re.escape("['radial.frequencies']")):
        canonical_state(torch.nn.ModuleDict({"radial": Basis()}))
