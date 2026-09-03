"""`--foundation_model_readout` decides whether the foundation readout is copied.

It stopped being read in November 2023: `6c8de70c` replaced
`load_readout=args.foundation_model_readout` with `load_readout=True`, and the
flag has done nothing since. The behaviour it used to control was reachable only
through `--foundation_filter_elements`, which was introduced later with a name
that describes something it never did: it filtered no elements, it was the value
passed as `load_readout`.

Both spellings now write to the same dest, so the flag named for the readout
controls the readout again and every existing script keeps its behaviour exactly.
The default is unchanged, since both flags defaulted to True.

`load_readout` itself decides whether `readouts.*` weights are transferred from
the foundation model (`finetuning_utils.py`), which is the part these tests check
rather than only the parsing: a flag that parses correctly and reaches the wrong
argument is the failure this file exists for.
"""

import ast
import inspect
from pathlib import Path

import numpy as np
import pytest
import torch
from e3nn import o3

from mace import modules, tools
from mace.tools import arg_parser, model_script_utils
from mace.tools.finetuning_utils import load_foundations_elements

TABLE = tools.AtomicNumberTable([1, 8])


def parse(*argv):
    return arg_parser.build_default_arg_parser().parse_args(
        ["--name=x", "--train_file=y", *argv]
    )


def model(seed):
    torch.manual_seed(seed)
    return modules.ScaleShiftMACE(
        r_max=5,
        num_bessel=8,
        num_polynomial_cutoff=5,
        max_ell=2,
        interaction_cls=modules.interaction_classes[
            "RealAgnosticResidualInteractionBlock"
        ],
        interaction_cls_first=modules.interaction_classes[
            "RealAgnosticResidualInteractionBlock"
        ],
        num_interactions=2,
        num_elements=2,
        hidden_irreps=o3.Irreps("8x0e + 8x1o"),
        MLP_irreps=o3.Irreps("8x0e"),
        gate=torch.nn.functional.silu,
        atomic_energies=np.array([1.0, 3.0]),
        avg_num_neighbors=3,
        atomic_numbers=TABLE.zs,
        correlation=3,
        radial_type="bessel",
        atomic_inter_scale=1.0,
        atomic_inter_shift=0.0,
    )


# ---------------------------------------------------------------------------
# Parsing: one dest, two spellings
# ---------------------------------------------------------------------------


def test_the_readout_is_transferred_by_default():
    assert parse().foundation_model_readout is True


def test_the_flag_turns_the_transfer_off():
    """What it could not do between 2023 and now."""
    assert parse("--foundation_model_readout").foundation_model_readout is False


@pytest.mark.parametrize("value,expected", [("False", False), ("True", True)])
def test_the_deprecated_spelling_still_works(value, expected):
    """`--foundation_filter_elements` has been the only way to reach this since
    April 2024, so scripts using it have to keep working and keep meaning the
    same thing."""
    assert parse("--foundation_filter_elements", value).foundation_model_readout is expected


def test_the_old_spelling_no_longer_has_a_dest_of_its_own():
    """One knob, not two that could disagree."""
    args = parse("--foundation_filter_elements", "False")

    assert not hasattr(args, "foundation_filter_elements")


def test_the_deprecated_help_says_so_and_says_what_it_does():
    """A user who reads `--help` after being surprised should find the answer
    there rather than in this file."""
    tree = ast.parse(Path(inspect.getfile(arg_parser)).read_text(encoding="utf-8"))
    helps = [
        keyword.value.value
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and any(
            isinstance(a, ast.Constant) and a.value == "--foundation_filter_elements"
            for a in node.args
        )
        for keyword in node.keywords
        if keyword.arg == "help" and isinstance(keyword.value, ast.Constant)
    ]

    assert helps, "the alias has no help text"
    assert "deprecated" in helps[0].lower()
    assert "never filtered elements" in helps[0].lower()


# ---------------------------------------------------------------------------
# Effect: what load_readout does with it
# ---------------------------------------------------------------------------


def test_the_configured_flag_is_what_the_loader_receives():
    """The regression that started this: the call site read a different flag."""
    source = Path(inspect.getfile(model_script_utils)).read_text(encoding="utf-8")

    assert "load_readout=args.foundation_model_readout," in source
    assert "args.foundation_filter_elements" not in source


def test_loading_with_the_readout_copies_the_readout_weights():
    target, foundation = model(0), model(1)
    before = target.readouts[0].linear.weight.detach().clone()

    load_foundations_elements(target, foundation, TABLE, load_readout=True, max_L=1)

    assert not torch.allclose(before, target.readouts[0].linear.weight), (
        "the readout was not transferred"
    )


def test_loading_without_it_leaves_them_alone():
    """The other half, and the reason the flag exists: a new head's readout must
    be able to start from its own initialisation."""
    target, foundation = model(0), model(1)
    before = target.readouts[0].linear.weight.detach().clone()

    load_foundations_elements(target, foundation, TABLE, load_readout=False, max_L=1)

    assert torch.allclose(before, target.readouts[0].linear.weight)


def test_the_interactions_are_transferred_either_way():
    """So the flag is about the readout only. If it started gating more than
    that, turning it off would quietly stop being fine-tuning."""
    for load_readout in (True, False):
        target, foundation = model(0), model(1)
        before = target.interactions[0].linear.weight.detach().clone()

        load_foundations_elements(
            target, foundation, TABLE, load_readout=load_readout, max_L=1
        )

        assert not torch.allclose(
            before, target.interactions[0].linear.weight
        ), f"interactions not transferred with load_readout={load_readout}"


# ---------------------------------------------------------------------------
# The same meaning from a config file as from the command line
# ---------------------------------------------------------------------------


def parse_yaml(tmp_path, body):
    config = tmp_path / "config.yaml"
    config.write_text(f"name: x\ntrain_file: y\n{body}\n", encoding="utf-8")
    return arg_parser.build_default_arg_parser().parse_args(["--config", str(config)])


def test_the_bare_flag_still_turns_the_transfer_off():
    """It has been a bare switch since 2023, so it stays one."""
    assert parse("--foundation_model_readout").foundation_model_readout is False


@pytest.mark.parametrize("value,expected", [("True", True), ("False", False)])
def test_an_explicit_value_is_taken_as_written(value, expected):
    args = parse("--foundation_model_readout", value)

    assert args.foundation_model_readout is expected


@pytest.mark.parametrize("key", ["foundation_model_readout", "foundation_filter_elements"])
@pytest.mark.parametrize("value,expected", [("true", True), ("false", False)])
def test_a_config_file_means_what_it_says(tmp_path, key, value, expected):
    """The trap this guards. configargparse turns a config entry into the flag
    plus its value, and a `store_false` switch ignores the value: as a switch,
    `foundation_model_readout: true` applied the flag and turned the transfer
    OFF, while `false` left it on. Both spellings now read the same way from a
    config as from the command line.
    """
    args = parse_yaml(tmp_path, f"{key}: {value}")

    assert args.foundation_model_readout is expected


def test_the_two_spellings_agree_from_a_config_file(tmp_path):
    """One dest, two names, and no way for a config to set them against each
    other by accident."""
    by_new = parse_yaml(tmp_path, "foundation_model_readout: false")
    by_old = parse_yaml(tmp_path, "foundation_filter_elements: false")

    assert by_new.foundation_model_readout == by_old.foundation_model_readout is False
