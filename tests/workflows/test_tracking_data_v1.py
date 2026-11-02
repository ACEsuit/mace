"""Native worker loading and W&B records, using a real offline client."""

import json

import numpy as np
import pytest
from ase.io import read, write

from tests.golden.harness import FP64_CPU_REFERENCE
from tests.workflows.test_run_train_v1 import (
    needs_the_v1_engine, on_v1, tiny_task, training_records, variant,
)

pytestmark = [needs_the_v1_engine, pytest.mark.wandb]


def offline_records(path):
    # Read the local SDK's event stream; no API or online account is used.
    from wandb.proto.wandb_internal_pb2 import Record
    from wandb.sdk.internal.datastore import DataStore

    store = DataStore()
    store.open_for_scan(str(path))
    try:
        while (payload := store.scan_data()) is not None:
            record = Record()
            record.ParseFromString(payload)
            yield record
    finally:
        store.close()


def fields(items):
    return {
        item.key or '.'.join(item.nested_key): json.loads(item.value_json)
        for item in items
    }


def test_offline_tracking_matches_native_training_with_workers(tmp_path, monkeypatch):
    from mace_torch.cli.run_train import parse

    monkeypatch.setenv('WANDB_MODE', 'offline')
    monkeypatch.setenv('WANDB_SILENT', 'true')
    monkeypatch.setenv('WANDB_CONSOLE', 'off')
    source = tiny_task(tmp_path)
    write(tmp_path / 'held-out.xyz', read(tmp_path / 'train.xyz', index='2:4'))
    source = variant(source, {
        'training.max_num_epochs': 1,
        'data.heads.default.test_file': str(tmp_path / 'held-out.xyz'),
    })
    directory = tmp_path / 'tracking'
    directory.mkdir()
    # Supplying client settings must not enable tracking by itself.
    off = tmp_path / 'off'
    reference = on_v1('--config', str(source), '--work_dir', str(off),
                      '--wandb_dir', str(directory), '--wandb_project', 'mace-v1-audit')
    assert reference.returncode == 0, reference.stderr
    assert not list(directory.rglob('*.wandb'))
    on = tmp_path / 'on'
    arguments = ['--config', str(source), '--work_dir', str(on), '--num_workers', '1',
                 '--pin_memory', 'true', '--wandb', '--wandb_dir', str(directory),
                 '--wandb_project', 'mace-v1-audit', '--wandb_entity', 'offline-team',
                 '--wandb_name', 'native-workers']
    expected_config = parse(arguments).to_resolved_dict()
    finished = on_v1(*arguments)
    assert finished.returncode == 0, finished.stderr
    paths = list(directory.glob('wandb/offline-run-*/*.wandb'))
    assert len(paths) == 1
    records = list(offline_records(paths[0]))
    run = next(record.run for record in records if record.WhichOneof('record_type') == 'run')
    assert (run.project, run.entity, run.display_name) == ('mace-v1-audit', 'offline-team', 'native-workers')
    logged_config = fields(run.config.update)
    assert {key: logged_config[key] for key in expected_config} == expected_config
    history = [fields(record.history.item) for record in records if record.WhichOneof('record_type') == 'history']
    assert len(history) == 1 and history[0]['_step'] == 0
    native = [row for row in training_records(on) if row['mode'] == 'valid']
    baseline = [row for row in training_records(off) if row['mode'] == 'valid']
    for actual, expected in zip(native, baseline, strict=True):
        assert actual.keys() == expected.keys()
        for key, value in expected.items():
            if isinstance(value, float):
                np.testing.assert_allclose(actual[key], value, atol=FP64_CPU_REFERENCE.atol, rtol=FP64_CPU_REFERENCE.rtol)
            else:
                assert actual[key] == value
    assert [row['epoch'] for row in native] == [None, 0]
    for key, value in native[-1].items():
        if isinstance(value, float):
            assert history[0][f'valid_default_{key}'] == value
    summary = {}
    for record in records:
        if record.WhichOneof('record_type') == 'summary':
            summary.update(fields(record.summary.update))
    for split in ('train_default', 'valid_default', 'default'):
        assert np.isfinite(summary[f'final_{split}_loss'])
        assert np.isfinite(summary[f'final_{split}_rmse_forces'])
    exits = [record.exit.exit_code for record in records if record.WhichOneof('record_type') == 'exit']
    assert exits == [0]
