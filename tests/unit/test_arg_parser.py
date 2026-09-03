"""Unit tests for mace/tools/arg_parser.py.

We test the *contract* of the parsers, not every one of the ~200 flags:

* ``build_default_arg_parser`` builds and parses a minimal valid command line
  (only ``--name`` is required).
* Critical defaults the rest of the codebase relies on (dtype, device, model,
  loss, cutoff, keys, ...).
* The ``--config`` YAML mode (configargparse): YAML values are applied and an
  explicit CLI flag overrides the YAML value.
* Invalid ``choices`` values exit with ``SystemExit``.
* ``build_preprocess_arg_parser`` smoke test + its required arg.
* The small helper converters (``str2bool``, ``check_float_or_none``,
  ``read_yaml``).
"""

import argparse

import pytest

from mace.tools.arg_parser import (
    build_default_arg_parser,
    build_preprocess_arg_parser,
    check_float_or_none,
    read_yaml,
    str2bool,
)

MINIMAL_ARGV = ["--name", "test_experiment"]


# ---------------------------------------------------------------------------
# build_default_arg_parser: minimal parse + required args
# ---------------------------------------------------------------------------


def test_default_parser_builds_and_parses_minimal_command_line():
    parser = build_default_arg_parser()
    args = parser.parse_args(MINIMAL_ARGV)
    assert args.name == "test_experiment"


def test_default_parser_requires_name(capsys):
    parser = build_default_arg_parser()
    with pytest.raises(SystemExit):
        parser.parse_args([])
    capsys.readouterr()  # swallow the argparse usage message


def test_default_parser_train_file_is_optional_at_parse_time():
    # run_train validates data files later; the parser itself must not require
    # --train_file (e.g. multihead configs provide files via --heads).
    args = build_default_arg_parser().parse_args(MINIMAL_ARGV)
    assert args.train_file is None


# ---------------------------------------------------------------------------
# critical defaults the rest of the code assumes
# ---------------------------------------------------------------------------


def test_default_parser_critical_defaults():
    args = build_default_arg_parser().parse_args(MINIMAL_ARGV)

    # precision / device
    assert args.default_dtype == "float64"
    assert args.device == "cpu"

    # model architecture
    assert args.model == "MACE"
    assert args.r_max == 5.0
    assert args.correlation == 3
    assert args.num_interactions == 2
    assert args.radial_type == "bessel"
    assert args.num_radial_basis == 8

    # loss / training
    assert args.loss == "weighted"
    assert args.energy_weight == 1.0
    assert args.forces_weight == 100.0
    assert args.swa_energy_weight == 1000.0
    assert args.batch_size == 10
    assert args.lr == 0.01
    assert args.max_num_epochs == 2048
    assert args.scaling == "rms_forces_scaling"
    assert args.seed == 123

    # evaluation / data split
    assert args.error_table == "PerAtomRMSE"
    assert args.valid_fraction == 0.1

    # xyz info/array keys
    assert args.energy_key == "REF_energy"
    assert args.forces_key == "REF_forces"
    assert args.stress_key == "REF_stress"
    assert args.virials_key == "REF_virials"


def test_stage_two_alias_maps_to_swa_dest():
    # --stage_two_* is the current spelling of the old --swa_* flags; both
    # must land in the same destination.
    args = build_default_arg_parser().parse_args(
        MINIMAL_ARGV + ["--stage_two_energy_weight", "77.0"]
    )
    assert args.swa_energy_weight == 77.0
    args2 = build_default_arg_parser().parse_args(
        MINIMAL_ARGV + ["--swa_energy_weight", "88.0"]
    )
    assert args2.swa_energy_weight == 88.0


# ---------------------------------------------------------------------------
# YAML --config mode (configargparse)
# ---------------------------------------------------------------------------


def test_yaml_config_values_are_applied(tmp_path):
    pytest.importorskip("configargparse")
    cfg = tmp_path / "config.yaml"
    cfg.write_text(
        "r_max: 4.5\n"
        "loss: huber\n"
        "batch_size: 7\n"
        "default_dtype: float32\n",
        encoding="utf-8",
    )
    args = build_default_arg_parser().parse_args(
        MINIMAL_ARGV + ["--config", str(cfg)]
    )
    assert args.r_max == 4.5
    assert args.loss == "huber"
    assert args.batch_size == 7
    assert args.default_dtype == "float32"


def test_cli_flag_overrides_yaml_config(tmp_path):
    pytest.importorskip("configargparse")
    cfg = tmp_path / "config.yaml"
    cfg.write_text("r_max: 4.5\nloss: huber\n", encoding="utf-8")
    args = build_default_arg_parser().parse_args(
        MINIMAL_ARGV + ["--config", str(cfg), "--r_max", "6.0"]
    )
    # explicit CLI value wins over the YAML one...
    assert args.r_max == 6.0
    # ...while untouched YAML values still apply
    assert args.loss == "huber"


# ---------------------------------------------------------------------------
# choices validation
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "flag,bad_value",
    [
        ("--device", "gpu"),  # valid: cpu/cuda/mps/xpu
        ("--default_dtype", "float16"),  # valid: float32/float64
        ("--loss", "not_a_loss"),
        ("--model", "NotAModel"),
        ("--error_table", "bogus"),
    ],
)
def test_invalid_choice_raises_system_exit(flag, bad_value, capsys):
    parser = build_default_arg_parser()
    with pytest.raises(SystemExit):
        parser.parse_args(MINIMAL_ARGV + [flag, bad_value])
    capsys.readouterr()


# ---------------------------------------------------------------------------
# build_preprocess_arg_parser
# ---------------------------------------------------------------------------


def test_preprocess_parser_smoke_and_defaults():
    parser = build_preprocess_arg_parser()
    args = parser.parse_args(["--train_file", "train.xyz"])
    assert args.train_file == "train.xyz"
    assert args.r_max == 5.0
    assert args.valid_fraction == 0.1
    assert args.seed == 123
    assert args.h5_prefix == ""
    assert args.batch_size == 16
    assert args.scaling == "rms_forces_scaling"
    assert args.energy_key == "REF_energy"
    assert args.forces_key == "REF_forces"
    assert args.shuffle is True


def test_preprocess_parser_requires_train_file(capsys):
    parser = build_preprocess_arg_parser()
    with pytest.raises(SystemExit):
        parser.parse_args([])
    capsys.readouterr()


# ---------------------------------------------------------------------------
# helper converters
# ---------------------------------------------------------------------------


def test_str2bool():
    assert str2bool(True) is True
    assert str2bool(False) is False
    for truthy in ("yes", "TRUE", "t", "y", "1"):
        assert str2bool(truthy) is True
    for falsy in ("no", "False", "f", "N", "0"):
        assert str2bool(falsy) is False
    with pytest.raises(argparse.ArgumentTypeError):
        str2bool("maybe")


def test_check_float_or_none():
    assert check_float_or_none("1.5") == 1.5
    assert check_float_or_none("None") is None
    with pytest.raises(argparse.ArgumentTypeError):
        check_float_or_none("abc")


def test_read_yaml(tmp_path):
    yml = tmp_path / "heads.yaml"
    yml.write_text("head1:\n  train_file: a.xyz\n", encoding="utf-8")
    assert read_yaml(str(yml)) == {"head1": {"train_file": "a.xyz"}}
    with pytest.raises(argparse.ArgumentTypeError):
        read_yaml(str(tmp_path / "missing.yaml"))
