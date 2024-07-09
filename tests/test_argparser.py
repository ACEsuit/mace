import pytest

from mace.tools import build_default_arg_parser, check_args


def test_finetuning_with_e0s_average_raises_error():
    parser = build_default_arg_parser()
    args = parser.parse_args(
        [
            "--name",
            "_",
            "--train_file",
            "_",
            "--foundation_model",
            "_",
            "--E0s",
            "average",
        ]
    )
    with pytest.raises(ValueError):
        check_args(args)


def test_force_flag_skips_check():
    parser = build_default_arg_parser()
    args = parser.parse_args(
        [
            "--name",
            "_",
            "--train_file",
            "_",
            "--foundation_model",
            "_",
            "--E0s",
            "average",
            "--force",
        ]
    )
    check_args(args)


def test_finetuning_with_non_average_e0s_does_not_raise_error():
    parser = build_default_arg_parser()
    args = parser.parse_args(
        [
            "--name",
            "_",
            "--train_file",
            "_",
            "--foundation_model",
            "_",
            "--E0s",
            "precomputes_e0s.json",
        ]
    )
    check_args(args)
