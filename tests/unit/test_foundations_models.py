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


# ---------------------------------------------------------------------------
# dispersion_xc, dispersion_cutoff and their neighbours
# ---------------------------------------------------------------------------


def test_dispersion_xc_is_forwarded(d3_kwargs):
    """The functional the D3 parameters come from. Passing the wrong one gives a
    physically different correction and no error, so the only symptom is numbers.
    """
    fm.mace_mp(model="small", device="cpu", dispersion=True, dispersion_xc="revpbe")

    assert d3_kwargs["xc"] == "revpbe"


def test_the_default_functional_is_pbe(d3_kwargs):
    """The default is part of the published behaviour: MACE-MP's D3 correction is
    the PBE-parameterised one, and a change of default silently changes every
    dispersion-corrected run."""
    fm.mace_mp(model="small", device="cpu", dispersion=True)

    assert d3_kwargs["xc"] == "pbe"


def test_dispersion_cutoff_is_forwarded(d3_kwargs):
    fm.mace_mp(model="small", device="cpu", dispersion=True, dispersion_cutoff=25.0)

    assert d3_kwargs["cutoff"] == 25.0


def test_the_default_cutoff_is_forty_bohr_in_angstrom(d3_kwargs):
    """Stated in the unit it arrives in. The signature says `40.0 * units.Bohr`,
    so the value handed to torch-dftd is in Angstrom, and a reader comparing it
    against the 40 in the docstring would otherwise think it disagreed.
    """
    from ase import units  # noqa: PLC0415  # pylint: disable=import-outside-toplevel

    fm.mace_mp(model="small", device="cpu", dispersion=True)

    assert d3_kwargs["cutoff"] == pytest.approx(40.0 * units.Bohr)
    assert d3_kwargs["cutoff"] == pytest.approx(21.167, abs=1e-3), (
        "40 Bohr in Angstrom, which is what torch-dftd is given"
    )


def test_the_two_cutoffs_are_separate_knobs(d3_kwargs):
    """`cutoff` and `cnthr` are different radii: the pair interaction range and
    the coordination-number range. They arrive from different arguments and one
    is not a default of the other."""
    fm.mace_mp(
        model="small",
        device="cpu",
        dispersion=True,
        dispersion_cutoff=30.0,
        dispersion_coord_cutoff=12.0,
    )

    assert d3_kwargs["cutoff"] == 30.0
    assert d3_kwargs["cnthr"] == 12.0


def test_the_dispersion_arguments_do_not_reach_the_mace_calculator(monkeypatch):
    """They are D3's, not MACE's. `MACECalculator` would raise on an unexpected
    keyword, so this also pins that they are consumed rather than forwarded.
    """
    seen = {}

    class FakeTorchDFTD3Calculator:
        def __init__(self, **kwargs):
            seen["d3"] = kwargs

    module = types.ModuleType("torch_dftd.torch_dftd3_calculator")
    module.TorchDFTD3Calculator = FakeTorchDFTD3Calculator
    monkeypatch.setitem(sys.modules, "torch_dftd", types.ModuleType("torch_dftd"))
    monkeypatch.setitem(sys.modules, "torch_dftd.torch_dftd3_calculator", module)
    monkeypatch.setattr(
        fm, "download_mace_mp_checkpoint", lambda model: "/fake/model.model"
    )

    def fake_mace_calculator(**kwargs):
        seen["mace"] = kwargs
        return object()

    monkeypatch.setattr(fm, "MACECalculator", fake_mace_calculator)
    monkeypatch.setattr(fm, "SumCalculator", lambda calcs: calcs)

    fm.mace_mp(
        model="small",
        device="cpu",
        dispersion=True,
        dispersion_xc="revpbe",
        dispersion_cutoff=25.0,
    )

    assert seen["d3"]["xc"] == "revpbe"
    assert not [key for key in seen["mace"] if key.startswith("dispersion")]
