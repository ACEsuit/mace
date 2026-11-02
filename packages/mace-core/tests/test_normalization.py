"""Measured normalization obeys head ownership, E0s and spherical symmetry."""

import numpy as np
import pytest
from mace_core.config.model import (
    DipoleRMSScaling,
    ForceRMSScaling,
    ModelConfig,
    NoScaling,
    StandardDeviationScaling,
)
from mace_core.config.resolved import ResolvedConfig
from mace_core.data import Configuration
from mace_core.data.normalization import compute_normalization
from mace_core.elements import ResolvedE0s
from mace_core.observables import ObservableSpec


def frame(head="a", **properties):
    return Configuration(
        atomic_numbers=np.array([1, 1]),
        positions=np.array([[0.0, 0.0, 0.0], [0.8, 0.0, 0.0]]),
        head=head,
        properties=properties,
    )


@pytest.mark.parametrize(
    "recipe,expected",
    [
        (StandardDeviationScaling(), (1.0, 2.0)),
        (ForceRMSScaling(), (2.0, 4.0)),
        (DipoleRMSScaling(), (3.0, 6.0)),
        (NoScaling(), (1.0, 1.0)),
    ],
)
def test_head_statistics_subtract_the_same_e0s_as_the_model(recipe, expected):
    data = [
        frame("a", energy=22.0, forces=np.full((2, 3), 2.0), dipole=[3.0] * 3),
        frame("a", energy=26.0, forces=np.full((2, 3), 2.0), dipole=[3.0] * 3),
        frame("b", energy=404.0, forces=np.full((2, 3), 4.0), dipole=[6.0] * 3),
        frame("b", energy=412.0, forces=np.full((2, 3), 4.0), dipole=[6.0] * 3),
    ]
    result = compute_normalization(
        data,
        [ObservableSpec(name="energy", irreps="0e", per_atom=False, units="eV")],
        {"energy": {"a": recipe, "b": recipe}},
        ResolvedE0s({"a": {1: 10.0}, "b": {1: 200.0}}),
    )["energy"]
    assert tuple(result[head].scale for head in ("a", "b")) == expected
    assert tuple(result[head].shift for head in ("a", "b")) == (
        ((0.0,), (0.0,)) if recipe.kind == "none" else ((2.0,), (4.0,))
    )


def test_non_scalar_means_are_not_subtracted_and_missing_labels_are_ignored():
    spec = ObservableSpec(name="q", irreps="0e+0o+1o", per_atom=False, units="u")
    values = np.array([[1.0, 2.0, 3.0, 4.0, 5.0], [3.0, 6.0, 7.0, 8.0, 9.0]])
    measured = compute_normalization(
        [
            frame(q=values[0]),
            frame(q=None),
            frame(q=values[1]),
            frame("b", q=100 * values[0]),
        ],
        [spec],
        {"q": {"a": StandardDeviationScaling()}},
        ResolvedE0s({}),
    )["q"]["a"]
    assert measured.shift == (2.0, 0.0, 0.0, 0.0, 0.0)
    assert measured.scale == pytest.approx(
        np.sqrt(np.mean((values - measured.shift) ** 2))
    )


@pytest.mark.parametrize("per_atom", [False, True])
def test_constant_targets_keep_a_trainable_scale(per_atom):
    spec = ObservableSpec(
        name="q", irreps="0e", per_atom=per_atom, extensive=True, units="u"
    )
    measured = compute_normalization(
        [frame(q=np.array([3.0, 3.0]) if per_atom else 6.0)],
        [spec],
        {"q": {"a": StandardDeviationScaling()}},
        ResolvedE0s({}),
    )["q"]["a"]
    assert measured.scale == 1.0
    assert measured.shift == (3.0,)


@pytest.mark.parametrize(
    "value,recipe,match",
    [
        (None, StandardDeviationScaling(), "no training labels"),
        (float("nan"), StandardDeviationScaling(), "nonfinite labels"),
        (1.0, ForceRMSScaling(), "forces"),
    ],
)
def test_invalid_statistics_fail_with_the_observable_and_head(value, recipe, match):
    with pytest.raises(ValueError, match=f"normalization.q.a.*{match}"):
        compute_normalization(
            [frame(q=value)],
            [ObservableSpec(name="q", irreps="0e", per_atom=False, units="u")],
            {"q": {"a": recipe}},
            ResolvedE0s({}),
        )


@pytest.mark.parametrize(
    "normalization",
    [
        {"forces": {"default": {"kind": "std"}}},
        {"energy": {"missing": {"kind": "std"}}},
        {"energy": {"default": {"kind": "std", "scale": 2.0}}},
        {"energy": {}},
    ],
)
def test_config_refuses_derivatives_unknown_heads_and_measured_values(normalization):
    with pytest.raises(ValueError):
        ResolvedConfig.model_validate({"model": {"normalization": normalization}})


def test_remapped_energy_retains_the_legacy_default_and_other_readouts_keep_identity():
    model = ModelConfig.model_validate(
        {
            "normalization": {
                "energy": {"a": {"kind": "std"}},
                "q": {"a": {"kind": "std"}},
            }
        }
    )
    assert model.normalization_for_heads({"new": "b"}) == {"q": {"new": NoScaling()}}
    assert model.normalization_for_heads({"new": "a"}) == {
        "energy": {"new": StandardDeviationScaling()},
        "q": {"new": StandardDeviationScaling()},
    }


def test_rms_rejects_nonfinite_vector_labels_even_when_its_target_is_finite():
    with pytest.raises(ValueError, match=r"normalization\.q\.a: nonfinite labels"):
        compute_normalization(
            [frame(q=[float("nan"), 0.0, 0.0], forces=np.ones((2, 3)))],
            [ObservableSpec(name="q", irreps="1o", per_atom=False, units="u")],
            {"q": {"a": ForceRMSScaling()}},
            ResolvedE0s({}),
        )
