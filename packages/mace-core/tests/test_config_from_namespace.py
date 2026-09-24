"""Carrying a legacy command line onto the new configuration.

The test that the table covers every flag needs the frozen tree's parser, so it
lives in `tests/architecture`, where both trees are installed. What is here is
the shim's own behaviour, which needs neither.
"""

from argparse import Namespace

import pytest
from mace_core.config.legacy import (
    LEGACY_TRAIN_DESTS,
    Dropped,
    Kept,
    LegacyFlagError,
    Merged,
    Reserved,
    from_namespace,
)
from mace_core.config.resolved import ResolvedConfig


def test_a_kept_flag_lands_on_its_field():
    values = from_namespace(Namespace(r_max=4.5, batch_size=32))
    assert values == {"model": {"r_max": 4.5}, "training": {"batch_size": 32}}
    assert (
        ResolvedConfig.model_validate(
            {**values, "model": {**values.get("model", {}), "model": "an-architecture"}}
        ).model.r_max
        == 4.5
    )


def test_a_namespace_missing_a_dest_contributes_nothing():
    """`mace_prepare_data` shares 22 dests and carries none of the others."""
    assert from_namespace(Namespace(seed=7)) == {"runtime": {"seed": 7}}


def test_a_flag_left_at_its_default_is_not_written_out():
    """So the user dict stays what the user wrote, which is what metadata wants."""
    assert from_namespace(Namespace(r_max=4.5), defaults={"r_max": 4.5}) == {}
    assert from_namespace(Namespace(r_max=3.0), defaults={"r_max": 4.5}) == {
        "model": {"r_max": 3.0}
    }


def test_a_flag_whose_destination_does_not_exist_yet_is_refused():
    """The failure this exists to prevent: a working command line that quietly
    trains something else on the new engine."""
    with pytest.raises(LegacyFlagError, match="default_dtype"):
        from_namespace(
            Namespace(default_dtype="float32"), defaults={"default_dtype": "float64"}
        )


def test_the_refusal_says_what_became_of_the_flag():
    with pytest.raises(LegacyFlagError, match="precision"):
        from_namespace(
            Namespace(default_dtype="float32"), defaults={"default_dtype": "float64"}
        )
    with pytest.raises(LegacyFlagError, match="device-agnostic"):
        from_namespace(Namespace(save_cpu=True), defaults={"save_cpu": False})
    with pytest.raises(LegacyFlagError, match="fine-tuning tickets"):
        from_namespace(
            Namespace(finetune_dipoles_polarizabilities=True),
            defaults={"finetune_dipoles_polarizabilities": False},
        )


def test_several_unreachable_flags_are_reported_together():
    """One run, one list, rather than one error per attempt."""
    with pytest.raises(LegacyFlagError) as caught:
        from_namespace(
            Namespace(
                default_dtype="float32",
                save_cpu=True,
                finetune_dipoles_polarizabilities=True,
            ),
            defaults={
                "default_dtype": "float64",
                "save_cpu": False,
                "finetune_dipoles_polarizabilities": False,
            },
        )
    message = str(caught.value)
    assert message.count("--") >= 3


def test_without_the_defaults_nothing_is_refused():
    """The caller has said it does not mind, and that is a different contract."""
    assert from_namespace(Namespace(default_dtype="float32")) == {}


def test_every_row_says_where_the_value_went():
    """A row that named nothing would read as coverage and give none."""
    for dest, disposition in sorted(LEGACY_TRAIN_DESTS.items()):
        if isinstance(disposition, Kept):
            assert "." in disposition.path, dest
        elif isinstance(disposition, Reserved):
            assert disposition.section and disposition.note, dest
        elif isinstance(disposition, Merged):
            assert disposition.into and disposition.note, dest
        else:
            assert disposition.reason, dest


def test_every_kept_path_reaches_a_real_field():
    """A path with a typo in it writes a key nothing reads.

    Walked against the schema rather than eyeballed: the shim's whole job is
    that a value arrives somewhere, and a dotted path is exactly the kind of
    string that goes stale when a field is renamed. It found the first real
    one, a hyperparameter written straight under a kinds field instead of
    under the kind that owns it.
    """
    unreachable = []
    for dest, disposition in sorted(LEGACY_TRAIN_DESTS.items()):
        if isinstance(disposition, Kept):
            problem = _walk(ResolvedConfig, disposition.path.split("."))
            if problem is not None:
                unreachable.append(f"{dest} -> {disposition.path} (at {problem!r})")
    assert not unreachable, unreachable


def _walk(model, names: list[str]) -> str | None:
    """The first name that names nothing, or `None` if the path resolves."""
    for position, name in enumerate(names):
        fields = getattr(model, "model_fields", {})
        if name in fields:
            model = fields[name].annotation
            continue
        kind = _kind_named(model, name)
        if kind is not None:
            model = kind
            continue
        value = _dict_value_of(model)
        if value is not None:
            # A mapping: this name is a key the user chooses, so anything is a
            # valid one and the rest of the path is checked against its value.
            return _walk(value, names[position + 1 :])
        return name
    return None


def _kind_named(annotation, name: str):
    """The variant of a kinds field whose tag is `name`, if there is one."""
    for arm in getattr(annotation, "__args__", ()):
        tag = getattr(arm, "model_fields", {}).get("kind")
        if tag is not None and name in getattr(tag.annotation, "__args__", ()):
            return arm
    return None


def _dict_value_of(annotation):
    """The value type of a `dict[...]` annotation, if it is one."""
    if getattr(annotation, "__origin__", None) is dict:
        return annotation.__args__[1]
    return None


def test_the_six_dropped_flags_are_the_recorded_ones():
    dropped = sorted(
        dest
        for dest, disposition in LEGACY_TRAIN_DESTS.items()
        if isinstance(disposition, Dropped)
    )
    assert dropped == [
        "field_norm_factor",
        "force_mh_ft_lr",
        "plot_interaction_e",
        "return_electrostatic_potentials",
        "save_cpu",
        "use_so3",
    ]


@pytest.mark.parametrize(("keep_all", "table"), [(True, "foundation"), (False, "data")])
def test_the_element_flag_picks_the_element_table(keep_all, table):
    """Legacy's default shrinks the table to the data's, and v1's keeps the
    foundation model's, so a legacy command line says which it meant."""
    values = from_namespace(Namespace(foundation_model_elements=keep_all))
    assert values["finetune"]["element_table"] == table
