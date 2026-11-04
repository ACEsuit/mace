"""CLI-1 at the frozen input/output boundary and on the committed anchors."""

import argparse
import json

import numpy as np
import pytest
import torch
from ase.io import read, write
from mace_core.clebsch_gordan.conversion import reduced_to_full
from mace_core.clebsch_gordan.irreps import Irreps
from mace_core.cli import main
from mace_core.config.resolved import ResolvedConfig
from mace_core.kernels.descriptors import SymmetricContractionDescriptor
from mace_core.metadata import ConfigRecord
from mace_torch.canonical import canonical_modules
from mace_torch.cli.eval_configs import OUTPUT_COLUMNS
from mace_torch.cli.legacy_flags import DEFAULTS, add_arguments
from mace_torch.cli.model import _rebuild
from mace_torch.deploy.loader import load_deployed
from mace_torch.deploy.neutral_io import import_neutral
from mace_torch.deploy.reference import verify_against_reference
from mace_torch.serialization import canonical_state, load_canonical_state
from mace_torch.train.checkpoint import write_model

from tests.golden.harness import CLOSED_FORM_FP64, FP64_CPU_REFERENCE
from tests.golden.surface_scan import scan_eval_surface
from tests.parity.test_fm_convert_legacy import (
    CATALOGUE,
    FIXTURES,
    GOLDEN,
    extract_here,
    reference,
)


def test_flag_grammar_matches_frozen_parser():
    from mace.tools.arg_parser import build_default_arg_parser

    legacy = build_default_arg_parser()
    native = argparse.ArgumentParser()
    add_arguments(native)
    old = {
        option: action
        for action in legacy._actions
        for option in action.option_strings
        if option not in {"-h", "--help", "--config"}
    }
    new = {
        option: action
        for action in native._actions
        for option in action.option_strings
        if option not in {"-h", "--help", "--config"}
    }
    assert set(old) == set(new)
    for option, action in old.items():
        current = new[option]
        assert (current.dest, current.nargs, type(current)) == (
            action.dest,
            action.nargs,
            type(action),
        ), option
        assert DEFAULTS[action.dest] == action.default, option
        if action.type in (str, int, float):
            assert current.type is action.type, option
        if action.type is not None and "bool" in action.type.__name__.lower():
            for value in ("true", "false"):
                assert current.type(value) == action.type(value), option


def test_all_thirteen_output_columns_are_accounted_for():
    scan, stores = scan_eval_surface()
    assert not scan.unresolved
    assert set(OUTPUT_COLUMNS) == set(stores)
    assert len(stores) == 13
    for name, column in OUTPUT_COLUMNS.items():
        assert column.owner in {"CLI-1", "MAG-1", "ELEC-4"}
        assert ("arrays" if column.per_atom else "info") in stores[name]


@pytest.mark.parametrize(
    "preset,model",
    [
        ("dipole", "AtomicDipolesMACE"),
        ("dipole_polar", "AtomicDielectricMACE"),
    ],
)
@pytest.mark.parametrize("weight", [1.0, 3.25])
@pytest.mark.parametrize("config_weights", [(1.0, 1.0), (2.0, 3.0), (0.0, 3.0)])
@pytest.mark.parametrize("stage_two", [False, True])
@pytest.mark.parametrize("chunked", [False, True])
def test_dipole_cli_presets_preserve_loss_and_gradient(
    fp64, preset, model, weight, config_weights, stage_two, chunked
):
    from mace_core.observables import DEFAULT_CATALOGUE, resolve_requested
    from mace_core.outputs import MACEOutput
    from mace_torch.cli.run_train import parse
    from mace_torch.data import TrainingBatch
    from mace_torch.train import build_loss

    from mace.modules.loss import DipolePolarLoss, DipoleSingleLoss
    from mace.tools.torch_geometric.data import Data

    flags = ["--model", model, "--loss", preset]
    if weight != 1.0:
        flags += [
            "--dipole_weight",
            str(weight),
            "--polarizability_weight",
            str(weight),
        ]
    if stage_two:
        weight *= 2.0
        flags += [
            "--stage_two", "--start_stage_two", "1", "--max_num_epochs", "2",
            "--swa_dipole_weight", str(weight),
            "--swa_polarizability_weight", str(weight),
        ]
    config = parse(flags)
    requested = resolve_requested(config.model.observables, DEFAULT_CATALOGUE)
    shapes = {"dipole": (2, 3)}
    if preset == "dipole_polar":
        shapes["polarizability"] = (2, 3, 3)
    generator = torch.Generator().manual_seed(7)
    targets = {
        name: torch.randn(shape, generator=generator) for name, shape in shapes.items()
    }
    predictions = {
        name: torch.randn(shape, generator=generator, requires_grad=True)
        for name, shape in shapes.items()
    }
    pointers = torch.tensor([0, 2, 5])
    batch = TrainingBatch(
        graph={
            "ptr": pointers,
            "batch": torch.tensor([0, 0, 1, 1, 1]),
            "weight": torch.tensor(config_weights),
        },
        targets=targets,
    )
    output = MACEOutput(
        dipole=predictions["dipole"],
        extras={name: value for name, value in predictions.items() if name != "dipole"},
    )
    stage = config.schedule()[-1]
    loss = build_loss(requested, config.loss, stage)
    if chunked:
        totals = loss.element_counts(batch)
        pieces = []
        for index, size in enumerate((2, 3)):
            part = TrainingBatch(
                graph={"ptr": torch.tensor([0, size]),
                       "weight": batch.graph["weight"][index:index + 1]},
                targets={name: value[index:index + 1] for name, value in targets.items()},
            )
            prediction = MACEOutput(
                dipole=predictions["dipole"][index:index + 1],
                extras={name: value[index:index + 1] for name, value in predictions.items()
                        if name != "dipole"},
            )
            pieces.append(loss(prediction, part, totals))
        native = sum(pieces)
    else:
        native = loss(output, batch)
    legacy_loss = (
        DipoleSingleLoss(dipole_weight=weight)
        if preset == "dipole"
        else DipolePolarLoss(dipole_weight=weight, polarizability_weight=weight)
    )
    legacy = legacy_loss(
        Data(ptr=pointers, weight=torch.tensor(config_weights), **targets), predictions
    )
    tensors = tuple(predictions.values())
    native_gradients = torch.autograd.grad(native, tensors, retain_graph=True)
    legacy_gradients = torch.autograd.grad(legacy, tensors)
    for actual, expected in zip(
        (native, *native_gradients), (legacy, *legacy_gradients), strict=True
    ):
        torch.testing.assert_close(
            actual, expected, atol=CLOSED_FORM_FP64.atol, rtol=CLOSED_FORM_FP64.rtol
        )


@pytest.mark.parametrize("name", ["tiny_mace", "tiny_scaleshift"])
def test_conversion_and_eval_preserve_the_committed_anchor(fp64, tmp_path, name):
    """Use the minimum-norm full representative only to construct the test input.

    Both this input and the CLI output must independently match frozen values;
    no equality of projected weights is claimed.
    """
    imported = import_neutral(
        extract_here(GOLDEN / "models" / f"{name}.model", tmp_path / "neutral"),
        CATALOGUE,
    )
    write_model(tmp_path / "imported", imported.engine, imported.metadata)
    source = load_deployed(tmp_path / "imported.json")
    document = source.config.to_resolved_dict()
    document["model"]["clebsch_gordan_basis"] = "full"
    config = ResolvedConfig.from_dict(document)
    full = _rebuild(source, config, source.heads)
    state = canonical_state(source.engine)
    destination_state = canonical_state(full)
    for path, module in canonical_modules(source.engine).items():
        descriptor = getattr(module, "descriptor", None)
        if not isinstance(descriptor, SymmetricContractionDescriptor):
            continue
        tensors = state[path]
        pieces = iter(tensors["weight"].split(tensors["path_counts"].tolist(), dim=1))
        weights = []
        for order in range(1, descriptor.correlation + 1):
            for _, irrep in Irreps.parse(descriptor.irreps_out):
                piece = next(pieces)
                weights.append(
                    torch.as_tensor(
                        reduced_to_full(
                            piece.numpy(), descriptor.irreps_in, order, irrep
                        ),
                        dtype=piece.dtype,
                    )
                )
        state[path] = {
            "weight": torch.cat(weights, dim=1),
            "path_counts": destination_state[path]["path_counts"],
        }
    load_canonical_state(full, state)
    report = verify_against_reference(
        full,
        reference(name),
        FIXTURES,
        z_table=source.z_table,
        cutoff=source.r_max,
    )
    assert report.passed, report.describe()
    metadata = source.metadata.model_copy(
        update={"config": ConfigRecord.from_config(config)}
    )
    write_model(tmp_path / "full", full, metadata)
    converted = tmp_path / "reduced.json"
    arguments = [
        "model",
        "convert",
        str(tmp_path / "full.json"),
        "--output",
        str(converted),
        "--to-basis",
        "reduced",
        "--reference",
        str(reference(name)),
        "--fixtures",
        str(FIXTURES),
    ]
    assert main(arguments) == 0
    record = load_deployed(converted).metadata.conversion
    assert record and record.projection_verified and record.golden_gate_passed

    # Evaluate the same geometries through the extxyz command and its reader.
    manifest = json.loads((FIXTURES / "manifest.json").read_text())
    snapshots = json.loads(reference(name).read_text())["fixtures"]
    structures = []
    for fixture in snapshots:
        entry = manifest["fixtures"][fixture]
        structures.append(read(FIXTURES / entry["file"]))
    input_file = tmp_path / "input.xyz"
    output_file = tmp_path / "evaluated.xyz"
    write(input_file, structures)
    assert (
        main(
            [
                "eval",
                "--model",
                str(converted),
                "--configs",
                str(input_file),
                "--output",
                str(output_file),
                "--compute_stress",
            ]
        )
        == 0
    )
    for atoms, expected in zip(
        read(output_file, index=":"), snapshots.values(), strict=True
    ):
        for quantity, store in (
            ("energy", atoms.info),
            ("forces", atoms.arrays),
            ("stress", atoms.info),
        ):
            if quantity not in expected["outputs"]:
                assert quantity == "stress" and not atoms.pbc.any()
                assert np.count_nonzero(store["MACE_stress"]) == 0
                continue
            np.testing.assert_allclose(
                store[f"MACE_{quantity}"],
                expected["outputs"][quantity]["value"],
                atol=FP64_CPU_REFERENCE.atol,
                rtol=FP64_CPU_REFERENCE.rtol,
            )

    # A failed golden check must not leave a successful-looking artifact pair.
    bad = json.loads(reference(name).read_text())
    first = next(iter(bad["fixtures"].values()))
    first["outputs"]["energy"]["value"] += 1.0
    bad_file = tmp_path / "bad-reference.json"
    bad_file.write_text(json.dumps(bad))
    arguments[arguments.index("--reference") + 1] = str(bad_file)
    arguments[arguments.index("--output") + 1] = str(tmp_path / "refused.json")
    assert main(arguments) == 2
    assert not (tmp_path / "refused.json").exists()
    assert not (tmp_path / "refused.safetensors").exists()


def test_hierarchical_entry_point_uses_the_launcher_without_framework_imports():
    import subprocess
    import sys

    script = """
import importlib.metadata
import sys
entry, = importlib.metadata.entry_points(group='console_scripts', name='mace')
assert entry.value == 'mace_launcher:main'
sys.argv = ['mace', '--engine', 'v1', '--help']
try:
    entry.load()()
except SystemExit as stopped:
    assert stopped.code == 0
assert not ({'torch', 'jax', 'mace'} & set(sys.modules))
"""
    subprocess.run([sys.executable, "-c", script], check=True)
