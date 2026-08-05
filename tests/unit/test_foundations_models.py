"""Unit tests for mace_mp argument handling that need no model download."""

import sys
import types

import pytest

from mace.calculators import foundations_models as fm


@pytest.fixture
def d3_kwargs(monkeypatch):
    """Run mace_mp(dispersion=True) offline, capturing the D3 kwargs."""
    recorded = {}

    class FakeTorchDFTD3Calculator:
        def __init__(self, **kwargs):
            recorded.update(kwargs)

    module = types.ModuleType("torch_dftd.torch_dftd3_calculator")
    module.TorchDFTD3Calculator = FakeTorchDFTD3Calculator
    parent = types.ModuleType("torch_dftd")
    monkeypatch.setitem(sys.modules, "torch_dftd", parent)
    monkeypatch.setitem(sys.modules, "torch_dftd.torch_dftd3_calculator", module)

    monkeypatch.setattr(
        fm, "download_mace_mp_checkpoint", lambda model: "/fake/model.model"
    )
    monkeypatch.setattr(fm, "MACECalculator", lambda **kwargs: object())
    monkeypatch.setattr(fm, "SumCalculator", lambda calcs: calcs)

    return recorded


def test_dispersion_damping_is_forwarded(d3_kwargs):
    fm.mace_mp(model="small", device="cpu", dispersion=True, dispersion_damping="zerom")
    assert d3_kwargs["damping"] == "zerom"


def test_legacy_damping_name_is_accepted(d3_kwargs):
    """`damping` was the released name for `dispersion_damping`.

    It arrives through **kwargs, which are forwarded to TorchDFTD3Calculator
    as well, so an unhandled alias collides with the explicit damping= there
    and raises a multiple-values TypeError instead of being honoured.
    """
    fm.mace_mp(model="small", device="cpu", dispersion=True, damping="zero")
    assert d3_kwargs["damping"] == "zero"


def test_default_damping(d3_kwargs):
    fm.mace_mp(model="small", device="cpu", dispersion=True)
    assert d3_kwargs["damping"] == "bj"
