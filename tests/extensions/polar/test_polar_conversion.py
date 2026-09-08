"""Serialization contracts for automatic Polar compatibility repair."""

import copy

import pytest
import torch

from tests.extensions.polar.test_polar_models import _build_minimal_model

pytestmark = pytest.mark.polar


def test_legacy_conversion_preserves_tensors_and_rebinds_after_reload(tmp_path):
    from graph_longrange.gto_utils import GTOSelfInteractionBlock

    model = _build_minimal_model(torch.device("cpu"), torch.float64)
    before = {key: value.clone() for key, value in model.state_dict().items()}
    del model.pbc_handling
    descriptor = model.electric_potential_descriptor
    del descriptor.pbc_handling
    del descriptor._precompute_geometry_impl
    del descriptor._forward_dynamic_impl
    del model.coulomb_energy.pbc_handling
    del model.coulomb_energy._forward_impl
    blocks = [m for m in model.modules() if isinstance(m, GTOSelfInteractionBlock)]
    assert len(blocks) == 4
    for block in blocks:
        del block.features_dim
    descriptor.static_quantities = {"stale": torch.ones(1, requires_grad=True)}
    path = tmp_path / "legacy.model"
    torch.save(model, path)
    restored = torch.load(path, weights_only=False)
    assert restored.pbc_handling == "auto"
    assert restored.electric_potential_descriptor.static_quantities is None
    assert (
        restored.electric_potential_descriptor._forward_dynamic_impl.__self__
        is restored.electric_potential_descriptor
    )
    assert restored.coulomb_energy._forward_impl.__self__ is restored.coulomb_energy
    for block in restored.modules():
        if isinstance(block, GTOSelfInteractionBlock):
            assert block.features_dim == block.features_irreps.dim
    assert before.keys() == restored.state_dict().keys()
    for key, value in restored.state_dict().items():
        assert value.dtype == before[key].dtype
        assert torch.equal(value, before[key]), key
    restored.set_electrostatic_pbcs("pbc")
    assert restored.electric_potential_descriptor.pbc_handling == "pbc"
    assert restored.coulomb_energy.pbc_handling == "pbc"
    torch.save(restored, path)
    restored = copy.deepcopy(torch.load(path, weights_only=False))
    assert restored.pbc_handling == "pbc"
    assert restored.electric_potential_descriptor.pbc_handling == "pbc"
    assert restored.coulomb_energy.pbc_handling == "pbc"
    with pytest.raises(ValueError, match="Unsupported pbc_handling"):
        restored.set_electrostatic_pbcs("invalid")
    assert restored.pbc_handling == "pbc"
    assert restored.electric_potential_descriptor.pbc_handling == "pbc"
    assert restored.coulomb_energy.pbc_handling == "pbc"


def test_conversion_rejects_conflicting_schema(tmp_path):
    model = _build_minimal_model(torch.device("cpu"), torch.float32)
    model.coulomb_energy.realspace_energy.self_interaction.features_dim = -1
    path = tmp_path / "invalid.model"
    torch.save(model, path)
    with pytest.raises(ValueError, match="Conflicting"):
        torch.load(path, weights_only=False)
