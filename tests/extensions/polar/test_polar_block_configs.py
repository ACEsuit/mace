"""`--fixedpoint_update_config` and `--field_readout_config`.

Two CLI flags that choose which block PolarMACE builds and how, written as
dict literals in a string. They travel a path nothing tested: the string is
parsed by `_parse_literal_or_none`, handed to the constructor, and there a `type`
key is looked up in a registry while the remaining keys become that block's own
keyword arguments. The existing polar tests pass fully-spelled dicts to the
constructor directly, so both the parse and the registry lookup were unobserved,
and a flag that reached the model as a string rather than a dict would fail
somewhere far from the flag.

Assertions are on the model the CLI path builds, through `_build_model` -- the
same call `configure_model` makes -- rather than on a training, which for a
PolarMACE is minutes and adds nothing here.
"""

import numpy as np
import pytest
import torch
from e3nn import o3

from mace import modules
from mace.modules.field_blocks import (
    AgnosticChargeBiasedLinearPotentialEmbedding,
    AgnosticEmbeddedOneBodyVariableUpdate,
    MLPNonLinearity,
    OneBodyMLPFieldReadout,
    field_readout_blocks,
    field_update_blocks,
)
from mace.tools import build_default_arg_parser
from mace.tools.model_script_utils import _build_model, _parse_literal_or_none
from mace.tools.torch_tools import default_dtype

ATOMIC_NUMBERS = [1, 8]


def build(fixedpoint=None, readout=None):
    """A small PolarMACE, built the way `configure_model` builds one."""
    argv = [
        "--name", "polarblocks",
        "--train_file", "train.xyz",
        "--model", "PolarMACE",
        "--num_recursion_steps", "1",
        "--atomic_multipoles_max_l", "1",
        "--field_feature_max_l", "1",
        "--MLP_irreps", "8x0e",
        "--radial_MLP", "[16]",
    ]
    if fixedpoint is not None:
        argv += ["--fixedpoint_update_config", fixedpoint]
    if readout is not None:
        argv += ["--field_readout_config", readout]
    args = build_default_arg_parser().parse_args(argv)
    args.std, args.mean = 1.0, 0.0

    model_config = dict(
        r_max=4.0,
        num_bessel=4,
        num_polynomial_cutoff=3,
        max_ell=2,
        interaction_cls=modules.interaction_classes[args.interaction],
        num_interactions=2,
        num_elements=len(ATOMIC_NUMBERS),
        hidden_irreps=o3.Irreps("4x0e + 4x1o"),
        edge_irreps=None,
        atomic_energies=np.zeros(len(ATOMIC_NUMBERS)),
        apply_cutoff=args.apply_cutoff,
        avg_num_neighbors=3.0,
        atomic_numbers=ATOMIC_NUMBERS,
        use_reduced_cg=args.use_reduced_cg,
        use_so3=args.use_so3,
        use_edge_irreps_first=args.use_edge_irreps_first,
        cueq_config=None,
    )
    with default_dtype(torch.float64):
        return _build_model(args, model_config, None, ["Default"])


# ---------------------------------------------------------------------------
# the string, before it is a config
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("value", [None, "None", "none", "", "  "])
def test_an_absent_config_is_none_however_it_is_spelled(value):
    """The flags default to `None`, and a YAML config file naturally writes the
    absent case as the string `None` -- which `ast.literal_eval` would turn into
    a name lookup rather than nothing."""
    assert _parse_literal_or_none(value) is None


def test_a_dict_literal_becomes_a_dict():
    parsed = _parse_literal_or_none("{'type': 'OneBodyMLPFieldReadout'}")

    assert parsed == {"type": "OneBodyMLPFieldReadout"}


def test_a_value_that_is_not_a_literal_is_refused():
    """Same reasoning as `--foundation_model_kwargs`: the flag carries code-shaped
    text and must not be able to run any."""
    with pytest.raises(ValueError):
        _parse_literal_or_none("{'type': __import__('os').name}")


# ---------------------------------------------------------------------------
# the defaults
# ---------------------------------------------------------------------------


def test_neither_flag_given_builds_the_default_blocks():
    model = build()

    assert model._fixedpoint_update_config == {}
    assert model._field_readout_config == {}
    assert isinstance(
        model.field_dependent_charges_maps[0], AgnosticEmbeddedOneBodyVariableUpdate
    )
    assert isinstance(model.local_electron_energy, OneBodyMLPFieldReadout)


def test_the_registries_name_the_defaults():
    """`type` is resolved through these, so a block that is not in them cannot be
    asked for by name however the flag is spelled."""
    assert field_update_blocks["AgnosticEmbeddedOneBodyVariableUpdate"] is (
        AgnosticEmbeddedOneBodyVariableUpdate
    )
    assert field_readout_blocks["OneBodyMLPFieldReadout"] is OneBodyMLPFieldReadout


# ---------------------------------------------------------------------------
# what the flags change
# ---------------------------------------------------------------------------


def test_the_flags_reach_the_model_as_dicts():
    """The end of the path: string on the command line, dict on the model. The
    model keeps its own copy including `type`, which the constructor pops off the
    working copy -- so what was asked for stays legible after the block is built.
    """
    model = build(
        fixedpoint="{'type': 'AgnosticEmbeddedOneBodyVariableUpdate'}",
        readout="{'type': 'OneBodyMLPFieldReadout'}",
    )

    assert model._fixedpoint_update_config == {
        "type": "AgnosticEmbeddedOneBodyVariableUpdate"
    }
    assert model._field_readout_config == {"type": "OneBodyMLPFieldReadout"}


def test_a_class_valued_key_chooses_the_block_it_names():
    """`potential_embedding_cls` is a class rather than a scalar, and the flag can
    only carry its name, so it is mapped to an implementation on the way in. The
    block is built with the class; the recorded config keeps the string."""
    model = build(
        fixedpoint=(
            "{'potential_embedding_cls': "
            "'AgnosticChargeBiasedLinearPotentialEmbedding'}"
        )
    )
    block = model.field_dependent_charges_maps[0]

    assert model._fixedpoint_update_config == {
        "potential_embedding_cls": "AgnosticChargeBiasedLinearPotentialEmbedding"
    }
    assert isinstance(
        block.potential_embedding, AgnosticChargeBiasedLinearPotentialEmbedding
    )


def test_nonlinearity_cls_is_accepted_and_ignored():
    """Recorded, not endorsed. `AgnosticEmbeddedOneBodyVariableUpdate._setup`
    discards `nonlinearity_cls` -- `_ = (nonlinearity_cls, num_elements)` -- and
    builds a `RadialMLP` whatever was asked for. The name is still resolved to a
    class on the way in, so the request looks honoured all the way to the block
    that throws it away, and this repository's own polar model builders pass
    `MLPNonLinearity` for it.
    """
    asked = build(fixedpoint="{'nonlinearity_cls': 'MLPNonLinearity'}")
    default = build()

    assert asked._fixedpoint_update_config == {"nonlinearity_cls": "MLPNonLinearity"}
    assert type(asked.field_dependent_charges_maps[0].nonlinearity) is type(
        default.field_dependent_charges_maps[0].nonlinearity
    )
    assert not isinstance(
        asked.field_dependent_charges_maps[0].nonlinearity, MLPNonLinearity
    )


def test_a_block_the_registry_does_not_know_is_refused():
    """The one thing in either config that is checked."""
    with pytest.raises(KeyError):
        build(readout="{'type': 'NoSuchReadout'}")


def test_an_unknown_key_is_discarded_without_a_word():
    """Recorded. `OneBodyMLPFieldReadout._setup` takes `**kwargs` and drops them
    (`_ = kwargs`), so a misspelled option is not a failure and not a warning: the
    model is built as if the flag had been empty, and the only trace is the
    string kept on the model."""
    model = build(readout="{'no_such_option': 1}")

    assert model._field_readout_config == {"no_such_option": 1}
    assert isinstance(model.local_electron_energy, OneBodyMLPFieldReadout)
