"""Partially labelled batches preserve the frozen loss and mapped gradients."""

import numpy as np
import pytest
import torch
from mace_core.config.loss import LossConfig
from mace_core.data import KeySpecification, configuration_from_atoms
from mace_core.elements import AtomicNumberTable
from mace_core.observables import DEFAULT_CATALOGUE, resolve_requested
from mace_torch.data import GraphDataset, collate_training, target_specs
from mace_torch.physics import DerivativeEngine
from mace_torch.train.loss import build_loss

from tests.golden.anchors import anchor_batch
from tests.golden.harness import CLOSED_FORM_FP64, FP64_CPU_REFERENCE
from tests.parity.anchors import load_anchor
from tests.parity.test_data_backend_parity import frames
from tests.parity.test_fm00_training_step import (
    ENERGY,
    assert_all_parameter_gradients,
    converted,
)

pytestmark = pytest.mark.usefixtures("isolated", "fp64")


@pytest.mark.parametrize("stress", [False, True])
@pytest.mark.parametrize("anchor", ["tiny_mace.model", "tiny_scaleshift.model"])
def test_missing_labels_preserve_loss_and_every_mapped_gradient(anchor, stress):
    from mace.modules.loss import (
        WeightedEnergyForcesLoss,
        WeightedEnergyForcesStressLoss,
    )

    structures = frames(4, 19)
    for index, atoms in enumerate(structures):
        atoms.cell = np.eye(3) * 10
        atoms.pbc = True
        atoms.info["config_weight"] = 0.5 + index / 4
        for name in ("energy", "forces", "stress"):
            atoms.info[f"config_{name}_weight"] = 0.25 + index / 8
    structures[0].info.pop("REF_energy")
    structures[1].arrays.pop("REF_forces")
    structures[2].info.pop("REF_stress")
    structures[3].info.pop("REF_energy")
    structures[3].arrays.pop("REF_forces")
    structures[3].info.pop("REF_stress")
    requested = resolve_requested(
        ["energy", "forces", *(["stress"] if stress else [])], DEFAULT_CATALOGUE
    )
    weights = {"energy": 1.0, "forces": 3.0, **({"stress": 2.0} if stress else {})}
    legacy = load_anchor(anchor)
    native, numbers = converted(legacy)
    legacy_batch = anchor_batch(legacy, structures)
    dataset = GraphDataset(
        [
            configuration_from_atoms(
                atoms, KeySpecification.from_defaults(), head_name="default"
            )
            for atoms in structures
        ],
        cutoff=float(legacy.r_max),
        z_table=AtomicNumberTable(numbers),
        targets=target_specs(requested),
        heads=("default",),
    )
    batch = collate_training(
        [dataset[i] for i in range(len(dataset))],
        z_table=AtomicNumberTable(numbers),
        float_dtype="float64",
    )
    legacy_output = legacy(
        legacy_batch.to_dict(), training=True, compute_force=True, compute_stress=stress
    )
    output = DerivativeEngine(native, ENERGY)(
        batch.graph,
        compute=requested.derivatives,
        training=True,
    )
    predictions = {"energy": output.total_energy, "forces": output.forces}
    if stress:
        predictions["stress"] = output.stress
    for name, value in predictions.items():
        torch.testing.assert_close(
            value,
            legacy_output[name],
            atol=FP64_CPU_REFERENCE.atol,
            rtol=FP64_CPU_REFERENCE.rtol,
        )
    legacy_class = (
        WeightedEnergyForcesStressLoss if stress else WeightedEnergyForcesLoss
    )
    expected = legacy_class(
        **{f"{name}_weight": value for name, value in weights.items()}
    )(
        legacy_batch,
        legacy_output,
    )
    actual = build_loss(requested, LossConfig(weights=weights))(output, batch)
    assert actual.item() > 0
    torch.testing.assert_close(
        actual, expected, atol=CLOSED_FORM_FP64.atol, rtol=CLOSED_FORM_FP64.rtol
    )
    expected.backward()
    actual.backward()
    assert_all_parameter_gradients(legacy, native)


@pytest.mark.parametrize("case", range(10))
def test_historical_key_selections_preserve_labels_through_preparation(tmp_path, case):
    """The ten historical global/head key recipes, at the agreed data boundary.

    Legacy resolves global < head *_key < head info_keys/arrays_keys in
    mace/cli/run_train.py. V1 prepares each head with those resolved keys;
    there is no second mutable key specification inside native training.
    """
    import copy

    from ase.io import read, write
    from mace_core.data import open_dataset
    from mace_torch.cli.preprocess_data import main

    from mace.data import KeySpecification as LegacyKeys
    from mace.data import config_from_atoms
    from mace.data.utils import update_keyspec_from_kwargs
    from tests.parity.test_data_backend_parity import assert_same
    from tests.workflows.test_run_train_allkeys import (
        configs_numbered_keys,
        trial_yamls_and_and_expected,
    )

    document, _, _ = trial_yamls_and_and_expected()[case]
    source = tmp_path / "source.xyz"
    write(source, configs_numbered_keys())
    global_keys = update_keyspec_from_kwargs(LegacyKeys.from_defaults(), document)
    unchanged = copy.deepcopy(global_keys)
    for name, settings in document.get("heads", {"Default": {}}).items():
        keys = update_keyspec_from_kwargs(copy.deepcopy(global_keys), settings)
        keys.update(
            info_keys=settings.get("info_keys", {}),
            arrays_keys=settings.get("arrays_keys", {}),
        )
        output = tmp_path / name
        assert (
            main(
                [
                    "--train_file",
                    str(source),
                    "--valid_file",
                    str(source),
                    "--output",
                    str(output),
                    "--num_process",
                    "1",
                    "--shuffle",
                    "false",
                    "--energy_key",
                    keys.info_keys["energy"],
                    "--forces_key",
                    keys.arrays_keys["forces"],
                    "--E0s",
                    "{1: 0.0, 8: 1.0}",
                ]
            )
            == 0
        )
        expected = [
            config_from_atoms(atoms, key_specification=keys)
            for atoms in read(source, ":")
        ]
        for split in ("train", "valid"):
            actual = list(
                open_dataset(
                    output / split, key_spec=KeySpecification.from_defaults()
                ).iter_range()
            )
            assert len(actual) == len(expected) == 15
            for index, (native, frozen) in enumerate(
                zip(actual, expected, strict=True)
            ):
                assert_same(native, frozen, (case, name, split, index))
        assert global_keys == unchanged
