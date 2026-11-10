"""Foundation loader options at the native CLI boundary."""

import copy
import json
import subprocess

from tests.helpers import v1_command

import numpy as np
import pytest
import yaml
from ase.io import read
from safetensors.numpy import load_file

from tests.workflows.test_data_sources_v1 import launch
from tests.workflows.test_run_train_v1 import needs_the_v1_engine, on_v1, tiny_task, training_records

pytestmark = needs_the_v1_engine


@pytest.mark.parametrize('foundation', [False, True])
@pytest.mark.parametrize('value', ['expression', '{not a dict', '[1, 2]'])
def test_invalid_foundation_options_stop_before_io(tmp_path, foundation, value):
    sentinel = tmp_path / 'executed'
    if value == 'expression':
        value = "{'device': __import__('pathlib').Path(%r).write_text('executed')}" % str(sentinel)
    arguments = ['--foundation_model_kwargs', value, '--train_file', str(tmp_path / 'missing.xyz'),
                 '--work_dir', str(tmp_path / 'run')]
    if foundation:
        arguments += ['--foundation_model', str(tmp_path / 'missing.safetensors')]
    done = on_v1(*arguments)
    assert done.returncode != 0
    assert '--foundation_model_kwargs' in done.stderr
    assert 'FileNotFoundError' not in done.stderr
    assert not sentinel.exists()
    assert not (tmp_path / 'run').exists()


@pytest.fixture(scope='module')
def foundation_task(tmp_path_factory):
    root = tmp_path_factory.mktemp('foundation-options')
    document = yaml.safe_load(tiny_task(root).read_text())
    document['training']['max_num_epochs'] = 2
    launch(root, document, 'foundation')
    return root, document


@pytest.mark.parametrize('finetune', [False, True])
@pytest.mark.parametrize('empty', ['{}', '{ }'])
def test_empty_foundation_options_preserve_training_and_reload(foundation_task, tmp_path, finetune, empty):
    from mace_torch.calculators import MACECalculator
    from mace_torch.deploy.loader import load_deployed

    root, original = foundation_task
    document = copy.deepcopy(original)
    if finetune:
        document['finetune'] = {'foundation_model': str(root / 'foundation/tiny.safetensors')}
    config = tmp_path / 'run.yaml'
    config.write_text(yaml.safe_dump(document))
    for name, flags in [('omitted', []), ('empty', ['--foundation_model_kwargs', empty])]:
        done = on_v1('--config', str(config), '--work_dir', str(tmp_path / name), *flags)
        assert done.returncode == 0, done.stderr
        records = training_records(tmp_path / name)
        steps = [row for row in records if row['mode'] == 'opt']
        assert steps and {row['epoch'] for row in steps} == {0, 1}
        assert all(np.isfinite(row['loss']) for row in records if 'loss' in row)
    expected = load_file(tmp_path / 'omitted/tiny.safetensors')
    actual = load_file(tmp_path / 'empty/tiny.safetensors')
    assert actual.keys() == expected.keys()
    for name in expected:
        np.testing.assert_array_equal(actual[name], expected[name])
    assert training_records(tmp_path / 'omitted') == training_records(tmp_path / 'empty')
    model = load_deployed(tmp_path / 'empty/tiny.safetensors')
    assert model.heads == ('default',)
    calculator = MACECalculator(models=model)
    for atoms in read(root / 'train.xyz', ':'):
        atoms.calc = calculator
        assert np.isfinite(atoms.get_potential_energy())
        assert np.isfinite(atoms.get_forces()).all()
    metadata = json.loads((tmp_path / 'empty/tiny.json').read_text())
    resolved = metadata['config']['config']['resolved']
    assert 'foundation_model_kwargs' not in resolved['finetune']


@pytest.mark.parametrize('source', ['flags', 'legacy_yaml'])
@pytest.mark.parametrize('options', [{'device': 'cpu'}, {'nonexistent_kwarg': 12345}])
def test_nonempty_foundation_options_are_refused_with_native_fields(foundation_task, tmp_path, source, options):
    root, _ = foundation_task
    values = {'foundation_model': str(root / 'foundation/tiny.safetensors'),
              'foundation_model_kwargs': options, 'train_file': str(tmp_path / 'missing.xyz'),
              'work_dir': str(tmp_path / 'run')}
    if source == 'legacy_yaml':
        config = tmp_path / 'legacy.yaml'
        config.write_text(yaml.safe_dump(values))
        arguments = ['--config', str(config)]
    else:
        arguments = [item for key, value in values.items() for item in (f'--{key}', str(value))]
    if source == 'legacy_yaml':
        done = subprocess.run(v1_command('model', 'migrate', '--output', str(tmp_path/'migrated.yaml'), '--', str(config)),capture_output=True,text=True,check=False)
    else:
        done = on_v1(*arguments)
    assert done.returncode != 0
    for field in ('--foundation_model_kwargs', 'runtime.device', 'precision.model',
                  'model.backend', 'data.heads.<name>.readout_from'):
        assert field in done.stderr
    assert 'does not carry yet' not in done.stderr
    assert 'FileNotFoundError' not in done.stderr
    assert not (tmp_path / 'run').exists()
