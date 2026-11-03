"""Native CLI training and fine-tuning consume ordered datasets and test directories."""

import json

import numpy as np
import pytest
import yaml
from ase.io import read, write

from tests.golden.harness import FP64_CPU_REFERENCE
from tests.workflows.test_run_train_v1 import needs_the_v1_engine, on_v1, tiny_task, training_records

pytestmark = needs_the_v1_engine


def prepare(source, directory):
    from mace_core.data import KeySpecification, open_dataset
    from mace_core.data_spec.shard_format import write_shards

    keys = KeySpecification.from_defaults()
    items = list(open_dataset(source, key_spec=keys, keep_isolated_atoms=True).iter_range())
    write_shards(items, directory, shard_size=3, key_spec=keys)
    return str(directory)


def launch(tmp_path, document, name):
    path = tmp_path / f'{name}.yaml'
    path.write_text(yaml.safe_dump(document))
    done = on_v1('--config', str(path), '--work_dir', str(tmp_path / name))
    assert done.returncode == 0, done.stderr
    return done


def close(actual, expected):
    np.testing.assert_allclose(actual, expected, atol=FP64_CPU_REFERENCE.atol, rtol=FP64_CPU_REFERENCE.rtol)


@pytest.fixture(scope='module')
def source_runs(tmp_path_factory):
    tmp = tmp_path_factory.mktemp('native-data-sources')
    document = yaml.safe_load(tiny_task(tmp).read_text())
    document['training']['max_num_epochs'] = 1
    frames = read(tmp / 'train.xyz', index=':')
    write(tmp / 'first.xyz', frames[:7])
    write(tmp / 'second.xyz', frames[7:])
    write(tmp / 'held.xyz', frames[2:7])
    files = [str(tmp / 'first.xyz'), prepare(tmp / 'second.xyz', tmp / 'second')]
    head = document['data']['heads']['default']
    head.update(valid_file=str(tmp / 'held.xyz'), test_file=str(tmp / 'held.xyz'))
    launch(tmp, document, 'single')
    directory = tmp / 'tests'
    directory.mkdir()
    write(directory / 'molecules.xyz', frames[2:4])
    prepare(tmp / 'held.xyz', directory / 'crystals')
    document['data'].update(test_dir=str(directory), num_workers=1)
    head.update(train_file=files, valid_file=[str(tmp / 'held.xyz')], test_file=[str(tmp / 'held.xyz')])
    done = launch(tmp, document, 'multiple')
    return tmp, document, done


def test_multiple_sources_match_single_file_training(source_runs):
    from safetensors.numpy import load_file

    tmp, document, done = source_runs
    baseline = [row for row in training_records(tmp / 'single') if row['mode'] == 'valid']
    actual = [row for row in training_records(tmp / 'multiple') if row['mode'] == 'valid']
    assert len(actual) == len(baseline) == 2
    for first, second in zip(actual, baseline, strict=True):
        assert first.keys() == second.keys()
        for key, value in second.items():
            if isinstance(value, float):
                close(first[key], value)
            else:
                assert first[key] == value
    expected = load_file(tmp / 'single/tiny.safetensors')
    for name, tensor in load_file(tmp / 'multiple/tiny.safetensors').items():
        close(tensor, expected[name])
    rows = [line.split('|')[1].strip() for line in (done.stdout + done.stderr).splitlines() if line.startswith('|')]
    assert {'default', 'default_crystals', 'default_molecules'} <= set(rows)
    metadata = json.loads((tmp / 'multiple/tiny.json').read_text())
    resolved = metadata['config']['config']['resolved']
    for split in ('train_file', 'valid_file', 'test_file'):
        assert resolved['data']['heads']['default'][split] == document['data']['heads']['default'][split]


def test_multihead_finetuning_with_lists_filters_test_rows_by_actual_head(source_runs):
    tmp, base, _ = source_runs
    document = json.loads(json.dumps(base))
    source = document['data']['heads']['default']
    document['data']['heads'] = {name: {**source, 'readout_from': 'default'} for name in ('pbe', 'pbe_d3')}
    document['data']['skip_evaluate_heads'] = ['pbe']
    document['finetune'] = {'foundation_model': str(tmp / 'single/tiny.safetensors')}
    done = launch(tmp, document, 'finetune')
    rows = {line.split('|')[1].strip() for line in (done.stdout + done.stderr).splitlines() if line.startswith('|')}
    assert {'train_pbe_d3', 'valid_pbe_d3', 'pbe_d3', 'pbe_d3_crystals', 'pbe_d3_molecules'} <= rows
    assert not {'train_pbe', 'valid_pbe', 'pbe', 'pbe_crystals', 'pbe_molecules'} & rows
    history = training_records(tmp / 'finetune')
    assert {row['head'] for row in history if row['mode'] == 'valid' and row['epoch'] == 0} == {'pbe', 'pbe_d3'}
    assert (tmp / 'finetune/tiny.safetensors').is_file()
