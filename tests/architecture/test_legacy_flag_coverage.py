"""Every legacy training flag is accounted for in the v1 configuration.

This is the one test in the shim's suite that needs both trees: the dest set is
re-derived from the frozen tree's own parser rather than written down, so a
flag added to `arg_parser.py` fails here instead of quietly having no v1 home.
That is also why it lives in `tests/architecture` and not beside the shim: the
packages job installs no legacy tree, and a module-level import of `mace` there
fails at collection before any marker can deselect it.

Both imports are skipped rather than assumed, so a job that has one tree and
not the other reports a skip instead of an error it cannot act on.
"""

import pytest

arg_parser = pytest.importorskip(
    "mace.tools.arg_parser", reason="needs the frozen tree's parser"
)
legacy = pytest.importorskip(
    "mace_core.config.legacy", reason="needs the v1 packages installed"
)


def parser_defaults() -> dict[str, object]:
    """Each dest's default, read off the actions.

    Not `parse_args([])`: `--name` is required, so parsing nothing raises, and
    a baseline built by parsing a minimal command line would carry that
    command line's values as if they were defaults.
    """
    parser = arg_parser.build_default_arg_parser()
    return {action.dest: action.default for action in parser._actions}


def parser_dests() -> set[str]:
    """Every dest `mace_run_train` exposes, from the parser itself.

    `help` is argparse's own and is not a setting, so it is the one exclusion
    and it is named rather than filtered by a pattern.
    """
    parser = arg_parser.build_default_arg_parser()
    return {action.dest for action in parser._actions} - {"help"}


def test_every_legacy_dest_has_exactly_one_disposition():
    """A set comparison, so both directions fail.

    A flag with no row is one whose v1 home nobody decided. A row with no flag
    is a decision about something that does not exist, which reads as coverage
    and gives none.
    """
    declared = set(legacy.LEGACY_TRAIN_DESTS)
    actual = parser_dests()
    assert not actual - declared, sorted(actual - declared)
    assert not declared - actual, sorted(declared - actual)


def test_the_dest_count_is_the_one_the_inventory_pinned():
    """184, and it is worth failing on rather than only on the set difference:
    a flag renamed in the same change that adds another would keep the set
    comparison honest and this one is what says the surface moved."""
    assert len(parser_dests()) == 184


def test_the_ten_alias_pairs_are_one_setting_each():
    """`--swa_*` and `--stage_two_*` are two spellings of one dest.

    A table keyed on option strings would claim twenty settings where there are
    ten, so this pins that the frozen tree really does share the dest and that
    the shim is right to key on it.
    """
    parser = arg_parser.build_default_arg_parser()
    by_dest: dict[str, set[str]] = {}
    for action in parser._actions:
        by_dest.setdefault(action.dest, set()).update(action.option_strings)
    # Matched by what the spelling contains, not by what it starts with: one
    # of the ten is `--start_swa` against `--start_stage_two`, so a prefix test
    # finds nine and reads as a discrepancy with the inventory that is not one.
    aliased = {
        dest: options
        for dest, options in by_dest.items()
        if any("swa" in option for option in options)
        and any("stage_two" in option for option in options)
    }
    assert len(aliased) == 10, sorted(aliased)
    for dest in aliased:
        assert dest in legacy.LEGACY_TRAIN_DESTS


def test_a_namespace_from_the_real_parser_carries_across():
    """The shim reads what the parser produces, not what a test hand-built."""
    parser = arg_parser.build_default_arg_parser()
    namespace = parser.parse_args(
        ["--name", "run", "--train_file", "train.xyz", "--r_max", "4.5"]
    )
    values = legacy.from_namespace(namespace, defaults=parser_defaults())
    assert values["runtime"]["name"] == "run"
    assert values["model"]["r_max"] == 4.5
    assert values["data"]["heads"]["default"]["train_file"] == "train.xyz"


def test_a_flag_with_no_v1_home_is_refused_on_a_real_command_line():
    parser = arg_parser.build_default_arg_parser()
    namespace = parser.parse_args(["--name", "run", "--default_dtype", "float32"])
    with pytest.raises(legacy.LegacyFlagError, match="default_dtype"):
        legacy.from_namespace(namespace, defaults=parser_defaults())


def test_the_plainest_real_command_line_validates_into_a_configuration():
    """Not only carried across: validated. A mapping that the schema then
    refuses is a shim that translates nothing, however right its keys look."""
    from mace_core.config.resolved import ResolvedConfig

    parser = arg_parser.build_default_arg_parser()
    namespace = parser.parse_args(["--name", "run", "--train_file", "train.xyz"])
    ResolvedConfig.model_validate(
        legacy.from_namespace(namespace, defaults=parser_defaults())
    )


def test_a_soft_freeze_reaches_the_parameter_group_it_names():
    """`--lr_params_factors` is JSON in a string, keyed `<group>_lr_factor`;
    the groups here are named without the suffix. A factor that does not
    arrive under the group's own name is a freeze that freezes nothing."""
    from mace_core.config.resolved import ResolvedConfig

    parser = arg_parser.build_default_arg_parser()
    namespace = parser.parse_args(
        [
            "--name",
            "run",
            "--train_file",
            "train.xyz",
            "--lr_params_factors",
            '{"embedding_lr_factor": 0.0, "interactions_lr_factor": 1.0, '
            '"products_lr_factor": 0.5, "readouts_lr_factor": 1.0}',
        ]
    )
    config = ResolvedConfig.model_validate(
        legacy.from_namespace(namespace, defaults=parser_defaults())
    )
    assert config.training.scheduler.group_factors == {
        "embedding": 0.0,
        "interactions": 1.0,
        "products": 0.5,
        "readouts": 1.0,
    }


def test_the_default_factors_write_nothing():
    """Every factor at one is no factor at all, and saying so in the recorded
    configuration would make a legacy run look tuned when it was not."""
    parser = arg_parser.build_default_arg_parser()
    namespace = parser.parse_args(["--name", "run", "--train_file", "train.xyz"])
    values = legacy.from_namespace(namespace, defaults=parser_defaults())
    assert "group_factors" not in values.get("training", {}).get("scheduler", {})


class _Reads:
    """A namespace that remembers which dests were read off it."""

    def __init__(self, namespace):
        self._namespace = namespace
        self.read: set[str] = set()

    def __getattr__(self, name):
        self.read.add(name)
        return getattr(self._namespace, name)


def test_every_flag_marked_carried_is_read_by_a_collapse():
    """A merged flag marked as carried is exempt from the refusal, so if no
    collapse reads it, a command line setting it trains without it and nobody
    is told. The collapses read some flags only under a given scheduler,
    optimizer or loss, so every combination is tried."""
    parser = arg_parser.build_default_arg_parser()
    read: set[str] = set()
    for scheduler in legacy._SCHEDULE_KINDS:
        for optimizer in ("adam", "adamw", "schedulefree"):
            for loss in ("weighted", "huber"):
                namespace = _Reads(
                    parser.parse_args(
                        [
                            "--name",
                            "run",
                            "--scheduler",
                            scheduler,
                            "--optimizer",
                            optimizer,
                            "--loss",
                            loss,
                        ]
                    )
                )
                legacy._collapse(namespace, {})
                read |= namespace.read
    carried = {
        dest
        for dest, disposition in legacy.LEGACY_TRAIN_DESTS.items()
        if isinstance(disposition, legacy.Merged) and disposition.applied
    }
    assert carried <= read, sorted(carried - read)


@pytest.mark.parametrize(
    "argv,field,expected",
    [
        (["--freeze", "5"], ("freeze",), 5),
        (["--foundation_model_readout", "False"], ("transfer_readout",), False),
        (["--lora", "True"], ("lora", "enabled"), True),
        (["--lora", "True", "--lora_rank", "8"], ("lora", "rank"), 8),
        (["--lora", "True", "--lora_alpha", "2.5"], ("lora", "alpha"), 2.5),
    ],
)
def test_a_fine_tune_flag_reaches_its_field(argv, field, expected):
    """Validated, not only carried: each lands on the field the fine-tune
    reads, with the value the command line gave."""
    from mace_core.config.resolved import ResolvedConfig

    parser = arg_parser.build_default_arg_parser()
    namespace = parser.parse_args(["--name", "run", "--train_file", "train.xyz", *argv])
    config = ResolvedConfig.model_validate(
        legacy.from_namespace(namespace, defaults=parser_defaults())
    )
    value = config.finetune
    for name in field:
        value = getattr(value, name)
    assert value == expected


def test_the_fine_tune_defaults_are_the_legacy_ones():
    """A command line that sets none of them resolves to what legacy runs
    with, so leaving them out of the mapping loses nothing."""
    from mace_core.config.resolved import ResolvedConfig

    defaults = parser_defaults()
    finetune = ResolvedConfig.model_validate({}).finetune
    assert finetune.freeze == defaults["freeze"]
    assert finetune.transfer_readout == defaults["foundation_model_readout"]
    assert finetune.lora.enabled == defaults["lora"]
    assert finetune.lora.rank == defaults["lora_rank"]
    assert finetune.lora.alpha == defaults["lora_alpha"]


def test_a_hard_and_a_soft_freeze_travel_together():
    """Legacy lets a run freeze by level and scale other groups at once; both
    have to arrive, since each alone is a different run."""
    from mace_core.config.resolved import ResolvedConfig

    parser = arg_parser.build_default_arg_parser()
    namespace = parser.parse_args(
        [
            "--name",
            "run",
            "--train_file",
            "train.xyz",
            "--freeze",
            "1",
            "--lr_params_factors",
            '{"embedding_lr_factor": 1.0, "interactions_lr_factor": 0.5, '
            '"products_lr_factor": 1.0, "readouts_lr_factor": 1.0}',
        ]
    )
    config = ResolvedConfig.model_validate(
        legacy.from_namespace(namespace, defaults=parser_defaults())
    )
    assert config.finetune.freeze == 1
    assert config.training.scheduler.group_factors["interactions"] == 0.5


def test_a_replay_head_flag_is_refused_until_the_flag_port():
    parser = arg_parser.build_default_arg_parser()
    namespace = parser.parse_args(
        ["--name", "run", "--train_file", "train.xyz", "--subselect_pt", "fps"]
    )
    with pytest.raises(legacy.LegacyFlagError, match="--subselect_pt"):
        legacy.from_namespace(namespace, defaults=parser_defaults())
