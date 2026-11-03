"""Native selection, replay labelling and explicit reuse produce the same fit."""

import copy
import subprocess

import pytest
import yaml
from ase.io import read, write

from tests.golden.harness import CLOSED_FORM_FP64
from tests.helpers import CLI_DIRECTORY, cli_command, v1_command
from tests.workflows.test_data_sources_v1 import close, launch
from tests.workflows.test_run_train_v1 import needs_the_v1_engine, on_v1, tiny_task, training_records

pytestmark = needs_the_v1_engine


def command(arguments):
    result = subprocess.run(arguments, capture_output=True, text=True, check=False)
    assert result.returncode == 0, result.stdout + result.stderr
    return result


def config_file(directory, document, name):
    path = directory / f'{name}.yaml'
    path.write_text(yaml.safe_dump(document))
    return path


@pytest.fixture(scope='module')
def replay_runs(tmp_path_factory):
    tmp = tmp_path_factory.mktemp('finetuning-steps')
    base = yaml.safe_load(tiny_task(tmp).read_text())
    base['training']['max_num_epochs'] = 1
    launch(tmp, base, 'foundation')
    frames = read(tmp / 'train.xyz', ':')[2:]
    for atoms in frames:
        atoms.info.clear()
        atoms.arrays.pop('REF_forces')
    write(tmp / 'pool-a.xyz', frames[:6])
    write(tmp / 'pool-b.xyz', frames[6:])
    write(tmp / 'held.xyz', frames[:3])
    document = {
        'runtime': {'name': 'fit', 'seed': 11},
        'finetune': {
            'foundation_model': str(tmp / 'foundation/tiny.safetensors'),
            'pseudolabels': {'enabled': True, 'heads': ['memory'], 'batch_size': 2},
        },
        'data': {'heads': {
            'memory': {
                'train_file': [str(tmp / 'pool-a.xyz'), str(tmp / 'pool-b.xyz')],
                'valid_file': str(tmp / 'held.xyz'),
                'subselect': {'method': 'random', 'num_samples': 6},
                'readout_from': 'default', 'weight': 0.4,
            },
            'target': {'train_file': str(tmp / 'train.xyz'), 'readout_from': 'default',
                       'e0s': {'kind': 'isolated_atoms'}},
        }, 'valid_fraction': 0.25, 'pin_memory': False},
        'model': {'observables': ['energy', 'forces']},
        'training': {'max_num_epochs': 1, 'batch_size': 4, 'valid_batch_size': 4,
                     'lr': 0.002, 'scheduler': {'kind': {'kind': 'constant'}}},
        'loss': {'weights': {'energy': 1.0, 'forces': 10.0}},
    }
    launch(tmp, document, 'one-go')
    original = config_file(tmp, document, 'select')
    selected = tmp / 'selected.xyz'
    command([*cli_command(CLI_DIRECTORY / 'fine_tuning_select.py'), '--engine', 'v1',
             '--config', str(original), '--head', 'memory', '--output', str(selected)])
    staged = copy.deepcopy(document)
    staged['data']['heads']['memory'].update(train_file=str(selected), subselect=None)
    path = config_file(tmp, staged, 'label')
    command([*v1_command('data', 'pseudolabel'), '--config', str(path), '--head', 'memory',
             '--output', str(tmp / 'saved-labels')])
    staged['finetune']['pseudolabels'].update(enabled=False, labels_from=str(tmp / 'saved-labels'))
    launch(tmp, staged, 'staged')
    return tmp, staged


def test_selection_and_explicit_reuse_match_one_go_finetuning(replay_runs):
    from safetensors.numpy import load_file

    tmp, _ = replay_runs
    assert len(read(tmp / 'selected.xyz', ':')) == 6
    expected = load_file(tmp / 'one-go/fit.safetensors')
    actual = load_file(tmp / 'staged/fit.safetensors')
    assert actual.keys() == expected.keys()
    for name, value in expected.items():
        close(actual[name], value)
    histories = [[row for row in training_records(tmp / name, name='fit') if row['mode'] == 'valid']
                 for name in ('one-go', 'staged')]
    assert len(histories[0]) == len(histories[1]) == 4
    for first, second in zip(*histories, strict=True):
        assert first.keys() == second.keys()
        for key, value in first.items():
            if isinstance(value, float):
                close(second[key], value)
            else:
                assert second[key] == value
    initial = [row for row in histories[0] if row['head'] == 'memory' and row['epoch'] is None]
    assert len(initial) == 1
    assert initial[0]['loss'] < CLOSED_FORM_FP64.atol
    assert not (tmp / 'staged/pseudolabels').exists()


def test_standalone_labels_cover_training_and_explicit_validation(replay_runs):
    from mace_torch.finetune.pseudolabels import load_pseudolabel_artifact

    tmp, _ = replay_runs
    generated, record = load_pseudolabel_artifact(tmp / 'one-go/pseudolabels/memory')
    saved, provenance = load_pseudolabel_artifact(tmp / 'saved-labels/memory')
    assert record.n_configs == provenance.n_configs == len(generated) == len(saved) == 9
    assert record.dataset_fingerprint == provenance.dataset_fingerprint
    assert record.foundation_fingerprint == provenance.foundation_fingerprint
    for first, second in zip(generated, saved, strict=True):
        assert first.weight == second.weight == 1.0
        for name in ('energy', 'forces'):
            assert first.property_weights[name] == second.property_weights[name] == 1.0
            close(first.properties[name], second.properties[name])


def test_reuse_refuses_changed_selection_before_training(replay_runs):
    tmp, settings = replay_runs
    document = copy.deepcopy(settings)
    changed = read(tmp / 'selected.xyz', ':')
    changed[0].positions[0, 0] += 0.01
    write(tmp / 'changed.xyz', changed)
    document['data']['heads']['memory']['train_file'] = str(tmp / 'changed.xyz')
    path = config_file(tmp, document, 'wrong-labels')
    result = on_v1('--config', str(path), '--work_dir', str(tmp / 'refused'))
    assert result.returncode != 0
    assert 'other structures' in result.stdout + result.stderr
    assert not (tmp / 'refused/fit.safetensors').exists()


def test_fps_selection_runs_from_a_native_checkpoint(replay_runs):
    tmp, settings = replay_runs
    document = copy.deepcopy(settings)
    document['data']['heads']['memory'].update(
        train_file=[str(tmp / 'pool-a.xyz'), str(tmp / 'pool-b.xyz')],
        subselect={'method': 'fps', 'num_samples': 4})
    path = config_file(tmp, document, 'fps')
    output = tmp / 'farthest.xyz'
    command([*cli_command(CLI_DIRECTORY / 'fine_tuning_select.py'), '--engine', 'v1',
             '--config', str(path), '--head', 'memory', '--output', str(output)])
    frames = read(output, ':')
    assert len(frames) == 4
    assert len({tuple(atoms.positions.reshape(-1)) for atoms in frames}) == 4
