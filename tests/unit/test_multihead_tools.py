"""Unit tests for mace/tools/multihead_tools.py.

Covers the pure/offline pieces: HeadConfig construction and defaults,
dict_head_to_dataclass (head-dict overrides vs. args fallbacks + the
missing-train_file error), prepare_default_head with the real arg parser,
and both branches of prepare_pt_head (neither branch downloads anything).

Also covers what generate_pseudolabels_for_configs does to a configuration's
property *weights*, and what it does when a batch fails, both of which need only
a tiny model built in process.

Not covered here (they require network access or a trained foundation model):
assemble_replay_data (downloads replay xyz) and
apply_pseudolabels_to_pt_head_configs (exercised by the finetuning workflow
tests).
"""

import argparse
import dataclasses

import numpy as np
import pytest
import torch
from ase import Atoms
from e3nn import o3

from mace import modules, tools
from mace.data.utils import config_from_atoms

from mace.data import KeySpecification
from mace.data.utils import update_keyspec_from_kwargs
from mace.tools import build_default_arg_parser
from mace.tools.multihead_tools import (
    HeadConfig,
    dict_head_to_dataclass,
    generate_pseudolabels_for_configs,
    prepare_default_head,
    prepare_pt_head,
    pseudolabel_weight,
)


def parse_minimal_args(extra=None):
    """Parse minimal real CLI args and attach key_specification the way
    run_train.run() does before calling the multihead helpers."""
    argv = ["--name", "test", "--train_file", "train.xyz"]
    if extra:
        argv += extra
    args = build_default_arg_parser().parse_args(argv)
    args.key_specification = KeySpecification()
    update_keyspec_from_kwargs(args.key_specification, vars(args))
    return args


# ---------------------------------------------------------------------------
# HeadConfig
# ---------------------------------------------------------------------------


def test_headconfig_is_dataclass_with_minimal_required_fields():
    keyspec = KeySpecification.from_defaults()
    cfg = HeadConfig(head_name="Default", key_specification=keyspec)
    assert dataclasses.is_dataclass(cfg)
    assert cfg.head_name == "Default"
    assert cfg.key_specification is keyspec


def test_headconfig_defaults_are_none():
    cfg = HeadConfig(head_name="h", key_specification=KeySpecification())
    for f in dataclasses.fields(cfg):
        if f.name in ("head_name", "key_specification"):
            continue
        assert getattr(cfg, f.name) is None, f.name


def test_headconfig_stores_optional_fields():
    cfg = HeadConfig(
        head_name="dft",
        key_specification=KeySpecification(),
        train_file="a.xyz",
        valid_file=["b.xyz", "c.xyz"],
        E0s="average",
        valid_fraction=0.2,
        atomic_numbers=[1, 8],
        mean=1.5,
        std=0.5,
        avg_num_neighbors=10.0,
        compute_avg_num_neighbors=False,
        keep_isolated_atoms=True,
    )
    assert cfg.train_file == "a.xyz"
    assert cfg.valid_file == ["b.xyz", "c.xyz"]
    assert cfg.E0s == "average"
    assert cfg.valid_fraction == 0.2
    assert cfg.atomic_numbers == [1, 8]
    assert cfg.mean == 1.5
    assert cfg.std == 0.5
    assert cfg.avg_num_neighbors == 10.0
    assert cfg.compute_avg_num_neighbors is False
    assert cfg.keep_isolated_atoms is True


# ---------------------------------------------------------------------------
# prepare_default_head
# ---------------------------------------------------------------------------


def test_prepare_default_head_single_default_head():
    args = parse_minimal_args()
    heads = prepare_default_head(args)
    assert list(heads.keys()) == ["Default"]
    head = heads["Default"]
    assert head["train_file"] == "train.xyz"
    assert head["valid_file"] == args.valid_file
    assert head["test_file"] == args.test_file
    assert head["test_dir"] == args.test_dir
    assert head["E0s"] == args.E0s
    assert head["statistics_file"] == args.statistics_file
    assert head["valid_fraction"] == args.valid_fraction
    assert head["config_type_weights"] == args.config_type_weights
    assert head["keep_isolated_atoms"] == args.keep_isolated_atoms
    assert head["key_specification"] is args.key_specification
    # arg-parser defaults flow through into the default head keyspec
    assert head["key_specification"].info_keys["energy"] == "REF_energy"
    assert head["key_specification"].arrays_keys["forces"] == "REF_forces"


def test_prepare_default_head_respects_custom_keys():
    args = parse_minimal_args(extra=["--energy_key", "MY_energy"])
    heads = prepare_default_head(args)
    assert heads["Default"]["key_specification"].info_keys["energy"] == "MY_energy"


# ---------------------------------------------------------------------------
# dict_head_to_dataclass
# ---------------------------------------------------------------------------


def test_dict_head_to_dataclass_head_overrides_args():
    args = parse_minimal_args()
    keyspec = KeySpecification.from_defaults()
    head = {
        "train_file": "head_train.xyz",
        "valid_file": "head_valid.xyz",
        "E0s": "average",
        "valid_fraction": 0.25,
        "mean": 2.0,
        "std": 3.0,
        "avg_num_neighbors": 7.0,
        "compute_avg_num_neighbors": False,
        "atomic_numbers": [1, 8],
        "keep_isolated_atoms": True,
        "key_specification": keyspec,
    }
    cfg = dict_head_to_dataclass(head, "dft", args)
    assert isinstance(cfg, HeadConfig)
    assert cfg.head_name == "dft"
    assert cfg.train_file == "head_train.xyz"
    assert cfg.valid_file == "head_valid.xyz"
    assert cfg.E0s == "average"
    assert cfg.valid_fraction == 0.25
    assert cfg.mean == 2.0
    assert cfg.std == 3.0
    assert cfg.avg_num_neighbors == 7.0
    assert cfg.compute_avg_num_neighbors is False
    assert cfg.atomic_numbers == [1, 8]
    assert cfg.keep_isolated_atoms is True
    assert cfg.key_specification is keyspec
    # test_file/test_dir have no args fallback
    assert cfg.test_file is None
    assert cfg.test_dir is None


def test_dict_head_to_dataclass_falls_back_to_args():
    args = parse_minimal_args()
    head = {"key_specification": args.key_specification}
    cfg = dict_head_to_dataclass(head, "Default", args)
    assert cfg.train_file == args.train_file == "train.xyz"
    assert cfg.valid_file == args.valid_file
    assert cfg.E0s == args.E0s
    assert cfg.valid_fraction == args.valid_fraction
    assert cfg.mean == args.mean
    assert cfg.std == args.std
    assert cfg.avg_num_neighbors == args.avg_num_neighbors
    assert cfg.compute_avg_num_neighbors == args.compute_avg_num_neighbors
    assert cfg.keep_isolated_atoms == args.keep_isolated_atoms


def test_dict_head_to_dataclass_missing_train_file_raises():
    args = parse_minimal_args()
    args.train_file = None  # neither the head nor args provide it
    head = {"key_specification": args.key_specification}
    with pytest.raises(ValueError, match="train file is not set"):
        dict_head_to_dataclass(head, "Default", args)


def test_dict_head_to_dataclass_requires_key_specification():
    args = parse_minimal_args()
    with pytest.raises(KeyError):
        dict_head_to_dataclass({"train_file": "x.xyz"}, "Default", args)


# ---------------------------------------------------------------------------
# prepare_pt_head (both branches are offline; no download happens here)
# ---------------------------------------------------------------------------


def test_prepare_pt_head_foundation_mp_branch():
    args = argparse.Namespace(foundation_model="small", pt_train_file=None)
    pt_keyspec = KeySpecification.from_defaults()
    pt_head = prepare_pt_head(args, pt_keyspec, foundation_model_num_neighbours=25.0)
    assert pt_head["train_file"] == "mp"
    assert pt_head["E0s"] == "foundation"
    assert pt_head["statistics_file"] is None
    assert pt_head["avg_num_neighbors"] == 25.0
    assert pt_head["compute_avg_num_neighbors"] is False
    # the MP branch rewrites the pt keyspec to the raw mptraj key names
    assert pt_keyspec.info_keys["energy"] == "energy"
    assert pt_keyspec.info_keys["stress"] == "stress"
    assert pt_keyspec.arrays_keys["forces"] == "forces"
    assert pt_head["key_specification"] is pt_keyspec


def test_prepare_pt_head_mp_branch_triggered_by_pt_train_file():
    args = argparse.Namespace(foundation_model="/some/local.model", pt_train_file="mp")
    pt_head = prepare_pt_head(
        args, KeySpecification.from_defaults(), foundation_model_num_neighbours=30.0
    )
    assert pt_head["train_file"] == "mp"
    assert pt_head["E0s"] == "foundation"


def test_prepare_pt_head_custom_replay_branch():
    args = argparse.Namespace(
        foundation_model="/some/local.model",
        pt_train_file="my_replay.xyz",
        pt_valid_file="my_replay_valid.xyz",
        statistics_file="stats.json",
        valid_fraction=0.15,
        keep_isolated_atoms=True,
    )
    pt_keyspec = KeySpecification.from_defaults()
    pt_head = prepare_pt_head(args, pt_keyspec, foundation_model_num_neighbours=12.5)
    assert pt_head["train_file"] == "my_replay.xyz"
    assert pt_head["valid_file"] == "my_replay_valid.xyz"
    assert pt_head["E0s"] == "foundation"
    assert pt_head["statistics_file"] == "stats.json"
    assert pt_head["valid_fraction"] == 0.15
    assert pt_head["keep_isolated_atoms"] is True
    assert pt_head["avg_num_neighbors"] == 12.5
    assert pt_head["compute_avg_num_neighbors"] is False
    # custom branch must NOT rewrite the keyspec to mptraj keys
    assert pt_keyspec.info_keys["energy"] == "REF_energy"
    assert pt_keyspec.arrays_keys["forces"] == "REF_forces"
    assert pt_head["key_specification"] is pt_keyspec


# ---------------------------------------------------------------------------
# generate_pseudolabels_for_configs: weights, and what a failed batch does
# ---------------------------------------------------------------------------

TABLE = tools.AtomicNumberTable([1, 8])


def _tiny_model():
    return modules.ScaleShiftMACE(
        r_max=4.0,
        num_bessel=4,
        num_polynomial_cutoff=5,
        max_ell=2,
        interaction_cls=modules.interaction_classes[
            "RealAgnosticResidualInteractionBlock"
        ],
        interaction_cls_first=modules.interaction_classes[
            "RealAgnosticInteractionBlock"
        ],
        num_interactions=2,
        num_elements=2,
        hidden_irreps=o3.Irreps("8x0e"),
        MLP_irreps=o3.Irreps("4x0e"),
        gate=torch.nn.functional.silu,
        atomic_energies=np.array([-1.0, -5.0]),
        avg_num_neighbors=4.0,
        atomic_numbers=TABLE.zs,
        correlation=2,
        atomic_inter_scale=1.0,
        atomic_inter_shift=0.0,
    ).double()


def _configs(count, labelled):
    """`count` waters, with or without the labels a replay file would carry."""
    rng = np.random.default_rng(0)
    keyspec = KeySpecification.from_defaults()
    keyspec.info_keys["energy"] = "REF_energy"
    keyspec.arrays_keys["forces"] = "REF_forces"
    out = []
    for index in range(count):
        atoms = Atoms(
            "H2O",
            positions=[[0, 0, 0], [0.95, 0, 0], [-0.24, 0.93, 0]],
            cell=[8, 8, 8],
            pbc=True,
        )
        atoms.positions += rng.normal(0, 0.04, (3, 3))
        if labelled:
            atoms.info["REF_energy"] = float(-20 + 0.1 * index)
            atoms.arrays["REF_forces"] = rng.normal(0, 0.1, (3, 3))
        out.append(config_from_atoms(atoms, key_specification=keyspec))
    return out


def _generate(configs, model=None, batch_size=4):
    """The graph is built from the process default dtype, so scope it: a float64
    model against a float32 graph fails every batch."""
    previous = torch.get_default_dtype()
    torch.set_default_dtype(torch.float64)
    try:
        return generate_pseudolabels_for_configs(
            model=model if model is not None else _tiny_model(),
            configs=configs,
            z_table=TABLE,
            r_max=4.0,
            device=torch.device("cpu"),
            batch_size=batch_size,
        )
    finally:
        torch.set_default_dtype(previous)


def test_pseudolabel_weight_is_a_floor_not_an_assignment():
    config = _configs(1, labelled=True)[0]
    config.property_weights["energy"] = 0.25
    config.property_weights["forces"] = 0.0

    assert pseudolabel_weight(config, "energy") == 0.25
    assert pseudolabel_weight(config, "forces") == 1.0
    assert pseudolabel_weight(config, "never_set") == 1.0


def test_labels_generated_for_an_unlabelled_set_carry_a_usable_weight():
    """The defect this closes: an unlabelled replay file arrives with every
    weight at zero, so the head trained on nothing and reported a loss of
    exactly zero with no error metrics -- indistinguishable from a perfect fit.
    """
    out = _generate(_configs(8, labelled=False))

    for config in out:
        assert config.properties["energy"] is not None
        assert config.property_weights["energy"] > 0.0
        assert config.property_weights["forces"] > 0.0


def test_a_deliberate_weight_survives_pseudolabelling():
    configs = _configs(8, labelled=True)
    for config in configs:
        config.property_weights["energy"] = 0.25

    out = _generate(configs)

    assert all(config.property_weights["energy"] == 0.25 for config in out)


def test_a_failed_batch_refuses_rather_than_mixing_label_provenance():
    """It used to substitute the file's own labels for the failed batch and
    carry on, so one call could return a set with two levels of theory in it and
    nothing to say which configurations had which."""
    model = _tiny_model()
    calls = {"n": 0}
    real_forward = model.forward

    def fail_on_second(*args, **kwargs):
        calls["n"] += 1
        if calls["n"] == 2:
            raise RuntimeError("simulated transient failure")
        return real_forward(*args, **kwargs)

    model.forward = fail_on_second

    with pytest.raises(RuntimeError, match="batch 2 of 3"):
        _generate(_configs(12, labelled=True), model=model)
