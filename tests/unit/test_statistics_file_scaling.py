"""A statistics file's mean/std must be used, not recomputed from the data.

`run_train` reads them from the file onto each head_config, but `configure_model` gated the
branch that consumes them on `args.std`. A run passing `--statistics_file` and no `--std`
therefore fell through to `scaling_classes[...](train_loader, ...)`, a full pass over the
training set whose result replaced the file's values.

Each test stops `configure_model` at `_build_model`, after the scaling has been decided.
"""

import types

import numpy as np
import pytest

from mace import modules, tools
from mace.tools import model_script_utils


class Built(Exception):
    """Raised in place of building the model."""


def head(mean=None, std=None, name="Default"):
    return types.SimpleNamespace(mean=mean, std=std, head_name=name)


def configure(monkeypatch, head_configs, *argv, heads=("Default",)):
    args = tools.build_default_arg_parser().parse_args(
        ["--name=t", "--train_file=t.xyz", "--hidden_irreps=8x0e", *argv]
    )
    args.compute_energy, args.compute_dipole = True, False

    def stop(*_a, **_k):
        raise Built

    monkeypatch.setattr(model_script_utils, "_build_model", stop)
    with pytest.raises(Built):
        model_script_utils.configure_model(
            args,
            None,
            np.zeros((len(heads), 1)),
            heads=list(heads),
            z_table=tools.AtomicNumberTable([1]),
            head_configs=head_configs,
        )
    return args


@pytest.fixture(name="no_rescan")
def no_rescan_fixture(monkeypatch):
    def explode(*_a, **_k):
        raise AssertionError("recomputed the scaling instead of using the statistics file")

    monkeypatch.setitem(modules.scaling_classes, "rms_forces_scaling", explode)


@pytest.mark.parametrize("mean, std", [(0.0, 1.0), (2, 3)])
def test_statistics_mean_and_std_are_used(monkeypatch, no_rescan, mean, std):
    args = configure(monkeypatch, [head(mean, std)])
    # Integers from a JSON file must arrive as floats: the shift helper drops other types.
    assert isinstance(args.mean, float) and args.mean == mean
    assert args.std == [std]
    assert model_script_utils._determine_atomic_inter_shift(args.mean, ["Default"]) == [mean]


def test_command_line_mean_and_std_are_used_without_statistics(monkeypatch, no_rescan):
    args = configure(monkeypatch, [head()], "--mean=0.5", "--std=2.5")
    assert args.mean == 0.5
    assert args.std == [2.5]


def test_without_statistics_it_still_computes(monkeypatch):
    calls = []

    def scaling(*_a, **_k):
        calls.append(1)
        return 0.3, 2.0

    monkeypatch.setitem(modules.scaling_classes, "rms_forces_scaling", scaling)
    args = configure(monkeypatch, [head()])
    assert calls == [1]
    assert (args.mean, args.std) == (0.3, 2.0)


def test_statistics_follow_the_order_of_heads(monkeypatch, no_rescan):
    """run_train puts pt_head first in heads but not in head_configs."""
    configs = [head(1.0, 2.0, name="ft"), head(3.0, 4.0, name="pt_head")]
    args = configure(monkeypatch, configs, heads=("pt_head", "ft"))
    assert args.mean == [3.0, 1.0]
    assert args.std == [4.0, 2.0]
