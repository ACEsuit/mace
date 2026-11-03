"""CLI preparation, worker writing and training from prepared v2 datasets."""

import json
import subprocess

import numpy as np
import pytest
import yaml
from ase.io import read, write

from tests.helpers import cli_command, preprocess_data
from tests.workflows.test_data_sources_v1 import close, launch
from tests.workflows.test_run_train_v1 import needs_the_v1_engine, tiny_task, training_records

pytestmark = needs_the_v1_engine


def prepare(*arguments):
    done = subprocess.run(
        [*cli_command(preprocess_data), '--engine', 'v1', *arguments],
        capture_output=True, text=True, check=False)
    assert done.returncode == 0, done.stdout + done.stderr


@pytest.fixture(scope='module')
def prepared_runs(tmp_path_factory):
    from mace_core.data import KeySpecification, open_dataset

    tmp = tmp_path_factory.mktemp('native-preprocessing')
    document = yaml.safe_load(tiny_task(tmp).read_text())
    document['training']['max_num_epochs'] = 1
    frames = read(tmp / 'train.xyz', index=':')
    write(tmp / 'valid.xyz', frames[2:6])
    write(tmp / 'first.xyz', frames[:7])
    write(tmp / 'second.xyz', frames[7:])
    settings = dict(train_file=[str(tmp / 'first.xyz'), str(tmp / 'second.xyz')],
                    valid_file=str(tmp / 'valid.xyz'), test_file=str(tmp / 'valid.xyz'),
                    h5_prefix='unused', work_dir=str(tmp), r_max=5.0,
                    seed=1, shuffle=False, compute_statistics=True, shard_size=5)
    path = tmp / 'prepare.yaml'
    path.write_text(yaml.safe_dump(settings))
    records = {}
    for shuffle in (False, True):
        for processes in (1, 2):
            name = f'prepared-{shuffle}-{processes}'
            prepare('--config', str(path), '--h5_prefix', name,
                    '--num_process', str(processes), '--shuffle', str(shuffle))
            keys = KeySpecification.from_defaults()
            records[shuffle, processes] = {
                split: list(open_dataset(tmp / name / split, key_spec=keys,
                                        keep_isolated_atoms=True).iter_range())
                for split in ('train', 'valid', 'test')}
    head = document['data']['heads']['default']
    head.update(valid_file=str(tmp / 'valid.xyz'), test_file=str(tmp / 'valid.xyz'))
    launch(tmp, document, 'raw')
    head.update(train_file=str(tmp / 'prepared-False-2/train'),
                valid_file=str(tmp / 'prepared-False-2/valid'),
                test_file=str(tmp / 'prepared-False-2/test'))
    document['data'].update(statistics_file=str(tmp / 'prepared-False-2/statistics.json'), num_workers=1)
    launch(tmp, document, 'prepared')
    return tmp, records


def test_serial_and_parallel_writers_preserve_order_and_statistics(prepared_runs):
    tmp, records = prepared_runs
    for shuffle in (False, True):
        for split, expected in records[shuffle, 1].items():
            actual = records[shuffle, 2][split]
            assert len(actual) == len(expected) == (14 if split == 'train' else 4)
            for first, second in zip(actual, expected, strict=True):
                np.testing.assert_array_equal(first.positions, second.positions)
                assert first.properties['energy'] == second.properties['energy']
        stats = [(tmp / f'prepared-{shuffle}-{processes}/statistics.json').read_text()
                 for processes in (1, 2)]
        assert stats[0] == stats[1]
    original = [item.properties['energy'] for item in records[False, 1]['train']]
    shuffled = [item.properties['energy'] for item in records[True, 1]['train']]
    assert original[:2] == shuffled[:2]
    assert original[2:] != shuffled[2:]
    assert sorted(original) == sorted(shuffled)
    assert not (tmp / 'unused').exists()


def test_prepared_training_matches_raw_inputs_and_keeps_isolated_references(prepared_runs):
    from safetensors.numpy import load_file

    tmp, _ = prepared_runs
    expected = load_file(tmp / 'raw/tiny.safetensors')
    actual = load_file(tmp / 'prepared/tiny.safetensors')
    assert actual.keys() == expected.keys()
    for name, value in expected.items():
        close(actual[name], value)
    histories = [[row for row in training_records(tmp / name) if row['mode'] == 'valid']
                 for name in ('raw', 'prepared')]
    assert len(histories[0]) == len(histories[1]) == 2
    for first, second in zip(*histories, strict=True):
        assert first.keys() == second.keys()
        for key, value in first.items():
            if isinstance(value, float):
                close(second[key], value)
            else:
                assert second[key] == value
    assert json.loads((tmp / 'prepared-False-2/e0s.json').read_text()) == {'1': -13.6, '8': -2040.0}
