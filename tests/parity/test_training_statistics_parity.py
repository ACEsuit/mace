"""Training statistics, their accepted native conventions and model buffers."""

import json

import numpy as np
import pytest
import torch
from ase import Atoms
from ase.io import write
from mace_core.config.resolved import ResolvedConfig
from mace_core.data.backends.memory import InMemoryBackend
from mace_core.data.keys import KeySpecification
from mace_core.data.statistics import compute_statistics
from mace_core.data.xyz import configuration_from_atoms
from mace_core.observables import DEFAULT_CATALOGUE
from mace_torch.cli.run_train import parse
from mace_torch.deploy.loader import load_deployed
from mace_torch.train import run_data_stage, run_model_stage
from mace_torch.train.checkpoint import write_model

from tests.golden.harness import CLOSED_FORM_FP64, FP32, FP64_CPU_REFERENCE

E0S = {1: -1.25, 6: -2.5, 8: -4.0}


def structures():
    positions = np.array(
        [[0, 0, 0], [0.7, 0, 0], [0, 0.7, 0], [0, 0, 0.7], [0.6, 0.6, 0]]
    )
    frames = []
    for index, symbols in enumerate(("H2", "OH2", "CO2", "CH4")):
        atoms = Atoms(symbols, positions=positions[: len(Atoms(symbols))])
        atoms.info["REF_energy"] = sum(E0S[int(z)] for z in atoms.numbers) + (
            1 + 2 * index
        ) * len(atoms)
        atoms.info["dipole"] = np.array([0.5 + index, -0.75, 0.25 * index])
        atoms.arrays["REF_forces"] = (
            np.arange(3 * len(atoms)).reshape(-1, 3) / 10 + index / 4
        )
        frames.append(atoms)
    return frames


def references():
    frames = []
    for number, energy in E0S.items():
        atom = Atoms(numbers=[number], positions=[[0, 0, 0]])
        atom.info.update(REF_energy=energy, config_type="IsolatedAtom")
        atom.arrays["REF_forces"] = np.zeros((1, 3))
        frames.append(atom)
    return frames


def legacy_loader(frames, batch_size=2):
    from mace import data
    from mace.tools import torch_geometric, utils

    configurations = [
        data.config_from_atoms(atoms, data.KeySpecification.from_defaults())
        for atoms in frames
    ]
    graphs = [
        data.AtomicData.from_config(
            config, z_table=utils.AtomicNumberTable(list(E0S)), cutoff=2.5
        )
        for config in configurations
    ]
    return configurations, torch_geometric.dataloader.DataLoader(
        graphs, batch_size=batch_size, shuffle=False
    )


def native_statistics(frames, scaling):
    backend = InMemoryBackend(
        [
            configuration_from_atoms(atoms, KeySpecification.from_defaults())
            for atoms in frames
        ]
    )
    return compute_statistics(backend, list(E0S), 2.5, E0S, scaling=scaling)


def training_config(tmp_path, frames, scaling="rms_forces", e0s=None, extra=()):
    train, valid = tmp_path / "train.xyz", tmp_path / "valid.xyz"
    write(train, frames)
    sentinel = structures()[0]
    sentinel.info["REF_energy"] = 1e6
    sentinel.arrays["REF_forces"][:] = 1e6
    write(valid, [sentinel])
    return parse(
        [
            "--name",
            "statistics",
            "--train_file",
            str(train),
            "--valid_file",
            str(valid),
            "--E0s",
            str(E0S if e0s is None else e0s),
            "--scaling",
            "no_scaling" if scaling == "none" else scaling + "_scaling",
            "--r_max",
            "2.5",
            "--num_channels",
            "4",
            "--max_L",
            "1",
            "--max_ell",
            "1",
            "--num_interactions",
            "2",
            "--correlation",
            "2",
            "--radial_MLP",
            "[4]",
            "--MLP_irreps",
            "4x0e",
            "--batch_size",
            "2",
            "--default_dtype",
            str(torch.get_default_dtype()).removeprefix("torch."),
            *extra,
        ]
    )


def close(actual, expected, tolerance):
    actual, expected = np.asarray(actual), np.asarray(expected)
    assert actual.shape == expected.shape
    assert np.isfinite(actual).all() and np.isfinite(expected).all()
    np.testing.assert_allclose(
        actual, expected, atol=tolerance.atol, rtol=tolerance.rtol
    )


@pytest.mark.parametrize("dtype", [torch.float64, torch.float32])
@pytest.mark.parametrize("scaling", ["rms_forces", "rms_dipoles"])
@pytest.mark.parametrize("batch_size", [1, 3])
def test_rms_statistics_match_legacy(isolated, dtype, scaling, batch_size):
    from mace.modules.utils import (
        compute_avg_num_neighbors,
        compute_mean_rms_energy_forces,
        compute_rms_dipoles,
    )

    torch.set_default_dtype(dtype)
    frames = structures()
    _, loader = legacy_loader(frames, batch_size)
    expected_mean, expected_force_rms = compute_mean_rms_energy_forces(
        loader, np.array(list(E0S.values()))
    )
    expected_rms = (
        np.asarray(expected_force_rms).item()
        if scaling == "rms_forces"
        else compute_rms_dipoles(loader)
    )
    actual = native_statistics(frames, scaling)
    tolerance = FP64_CPU_REFERENCE if dtype == torch.float64 else FP32
    close(actual.mean, np.asarray(expected_mean).item(), tolerance)
    close(actual.std, expected_rms, tolerance)
    close(actual.avg_num_neighbors, compute_avg_num_neighbors(loader), tolerance)


def test_native_population_spread_and_zero_neighbor_atoms_are_explicit(isolated):
    from mace.modules.utils import (
        compute_avg_num_neighbors,
        compute_mean_std_atomic_inter_energy,
    )

    torch.set_default_dtype(torch.float64)
    frames = structures()
    lonely = Atoms("H", positions=[[0, 0, 0]])
    lonely.info["REF_energy"] = E0S[1] + 9
    lonely.arrays["REF_forces"] = np.zeros((1, 3))
    frames.append(lonely)
    actual = native_statistics(frames, "std")
    _, loader = legacy_loader(frames)
    mean, spread = compute_mean_std_atomic_inter_energy(
        loader, np.array(list(E0S.values()))
    )
    # Residual energies per atom are 1, 3, 5, 7, 9; all molecular graphs
    # are complete, while the final atom has no edge.
    close(actual.mean, 5.0, CLOSED_FORM_FP64)
    close(actual.std, np.sqrt(8.0), CLOSED_FORM_FP64)
    close(np.asarray(mean).item(), 5.0, FP64_CPU_REFERENCE)
    close(np.asarray(spread).item(), np.sqrt(10.0), FP64_CPU_REFERENCE)
    close(actual.avg_num_neighbors, 34 / 14, CLOSED_FORM_FP64)
    close(compute_avg_num_neighbors(loader), 34 / 13, FP64_CPU_REFERENCE)
    assert actual.std != np.asarray(spread).item()


@pytest.mark.parametrize("kind", ["literal", "json", "average", "isolated"])
@pytest.mark.parametrize("keep", [False, True])
def test_e0_cli_sources_and_isolated_selection_match_legacy(
    isolated, tmp_path, kind, keep
):
    from mace import data
    from mace.tools import utils
    from mace.tools.scripts_utils import get_atomic_energies

    torch.set_default_dtype(torch.float64)
    frames = ([] if kind == "average" else references()) + structures()
    source = E0S if kind == "literal" else kind
    if kind == "json":
        path = tmp_path / "e0s.json"
        path.write_text(json.dumps(E0S))
        source = str(path)
    config = training_config(
        tmp_path, frames, e0s=source, extra=["--keep_isolated_atoms", str(keep)]
    )
    native = run_data_stage(config, DEFAULT_CATALOGUE)
    extracted, configs = data.load_from_xyz(
        str(tmp_path / "train.xyz"),
        data.KeySpecification.from_defaults(),
        extract_atomic_energies=True,
        keep_isolated_atoms=keep,
    )
    expected = (
        extracted
        if kind == "isolated"
        else get_atomic_energies(
            str(source), configs, utils.AtomicNumberTable(list(E0S))
        )
    )
    close(
        np.array(list(native.e0s.values["default"].values())),
        np.array([expected[z] for z in E0S]),
        FP64_CPU_REFERENCE,
    )
    assert len(native.train_eval_loaders["default"].dataset) == len(configs)
    assert len(configs) == (7 if keep and kind != "average" else 4)


@pytest.mark.parametrize("dtype", [torch.float64, torch.float32])
@pytest.mark.parametrize("scaling", ["rms_forces", "rms_dipoles", "std", "none"])
def test_statistics_reach_model_buffers_and_survive_reload(
    isolated, tmp_path, dtype, scaling
):
    torch.set_default_dtype(dtype)
    frames = structures()
    config = training_config(tmp_path, frames, scaling=scaling)
    data = run_data_stage(config, DEFAULT_CATALOGUE)
    built = run_model_stage(config, data, DEFAULT_CATALOGUE)
    head = built.engine.model.outputs.energy_readout
    measured = native_statistics(frames, "std" if scaling == "none" else scaling)
    tolerance = FP64_CPU_REFERENCE if dtype == torch.float64 else FP32
    close(
        head.scale.detach().numpy(),
        np.array([1.0 if scaling == "none" else measured.std]),
        tolerance,
    )
    close(
        head.shift.detach().numpy(),
        np.array([0.0 if scaling == "none" else measured.mean]),
        tolerance,
    )
    close(
        head.e0_table.detach().numpy(), np.array([list(E0S.values())]), CLOSED_FORM_FP64
    )
    batch = next(iter(data.train_eval_loaders["default"])).graph
    expected = built.engine(batch, compute=("forces",))
    saved = write_model(tmp_path / "model", built.engine, built.metadata)
    loaded = load_deployed(saved)
    actual = loaded.engine(batch, compute=("forces",))
    for name in ("energy", "forces"):
        close(
            actual.get(name).detach().numpy(),
            expected.get(name).detach().numpy(),
            tolerance,
        )


@pytest.mark.parametrize("scaling", ["rms_forces", "std"])
def test_zero_spread_keeps_the_model_trainable_like_legacy(isolated, tmp_path, scaling):
    from mace.modules.utils import (
        compute_mean_rms_energy_forces,
        compute_mean_std_atomic_inter_energy,
    )

    torch.set_default_dtype(torch.float64)
    frames = structures()
    for atoms in frames:
        atoms.arrays["REF_forces"][:] = 0
        if scaling == "std":
            atoms.info["REF_energy"] = sum(
                E0S[int(z)] for z in atoms.numbers
            ) + 2 * len(atoms)
    _, loader = legacy_loader(frames)
    function = (
        compute_mean_rms_energy_forces
        if scaling == "rms_forces"
        else compute_mean_std_atomic_inter_energy
    )
    _, expected_scale = function(loader, np.array(list(E0S.values())))
    close(np.asarray(expected_scale).item(), 1.0, CLOSED_FORM_FP64)
    config = training_config(tmp_path, frames, scaling=scaling)
    data = run_data_stage(config, DEFAULT_CATALOGUE)
    assert data.statistics.std == 0.0
    built = run_model_stage(config, data, DEFAULT_CATALOGUE)
    close(
        built.engine.model.outputs.energy_readout.scale.detach().numpy(),
        np.array([1.0]),
        CLOSED_FORM_FP64,
    )
    batch = next(iter(data.train_eval_loaders["default"])).graph
    built.engine(batch, compute=()).get("energy").sum().backward()
    gradients = [p.grad for p in built.engine.parameters() if p.grad is not None]
    assert gradients and all(torch.isfinite(g).all() for g in gradients)
    assert any(torch.count_nonzero(g) for g in gradients)


@pytest.mark.parametrize("kind", ["foundation", "estimated"])
def test_foundation_e0s_use_live_converted_predictions(isolated, tmp_path, kind):
    from mace_torch.deploy.neutral_io import import_neutral
    from mace_torch.finetune.stages import build

    from mace.data.utils import estimate_e0s_from_foundation
    from mace.tools.utils import AtomicNumberTable
    from tests.parity.anchors import load_anchor
    from tests.parity.test_fm_convert_legacy import GOLDEN, extract_here

    torch.set_default_dtype(torch.float64)
    legacy = load_anchor("tiny_scaleshift.model")
    imported = import_neutral(
        extract_here(GOLDEN / "models/tiny_scaleshift.model", tmp_path / "neutral"),
        DEFAULT_CATALOGUE,
    )
    artifact = write_model(tmp_path / "foundation", imported.engine, imported.metadata)
    frames = structures()
    config = training_config(
        tmp_path, frames, e0s=kind, extra=["--foundation_model", str(artifact)]
    )
    document = config.model_dump(mode="json")
    document["model"] = {"observables": ["energy", "forces"]}
    config = ResolvedConfig.model_validate(document)
    source = dict(
        zip(
            legacy.atomic_numbers.tolist(),
            legacy.atomic_energies_fn.atomic_energies.reshape(-1).tolist(),
            strict=True,
        )
    )
    configs, _ = legacy_loader(frames)
    expected = (
        source
        if kind == "foundation"
        else estimate_e0s_from_foundation(
            legacy, source, configs, AtomicNumberTable(list(E0S)), device="cpu"
        )
    )
    built = build(config)
    actual = built.data.e0s.values["default"]
    close(
        np.array([actual[z] for z in E0S]),
        np.array([expected[z] for z in E0S]),
        FP64_CPU_REFERENCE,
    )
    close(
        built.engine.model.outputs.energy_readout.e0_table.detach().numpy(),
        np.array([[expected[z] for z in E0S]]),
        FP64_CPU_REFERENCE,
    )
    if kind == "estimated":
        assert actual != source
        assert (
            built.data.e0_provenance["default"].solver
            == "foundation_corrected_least_squares"
        )
        assert build(config).data.e0s.values["default"] == actual
    else:
        assert actual == source
