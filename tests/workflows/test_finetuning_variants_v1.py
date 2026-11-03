"""Native CLI adaptation, freezing and simultaneous scalar conditioning."""

import copy
import json

import numpy as np
import pytest
import yaml
from ase.io import read, write

from tests.golden.harness import FP64_CPU_REFERENCE
from tests.workflows.test_data_sources_v1 import launch
from tests.workflows.test_run_train_v1 import needs_the_v1_engine, on_v1, tiny_task, training_records

pytestmark = needs_the_v1_engine


@pytest.fixture(scope='module', params=['cpu', pytest.param('cuda', marks=pytest.mark.gpu)])
def adaptation_task(tmp_path_factory, request):
    directory = tmp_path_factory.mktemp('adaptation-' + request.param)
    document = yaml.safe_load(tiny_task(directory).read_text())
    document['runtime']['device'] = request.param
    document['training']['max_num_epochs'] = 2
    launch(directory, document, 'energy')
    return directory, document


def weights(path):
    from mace_torch.deploy.loader import load_deployed

    model = load_deployed(path)
    return {name: value.detach().clone() for name, value in model.engine.named_parameters() if value.numel()}


def record(directory, **values):
    (directory / 'case-evidence.json').write_text(json.dumps(values, indent=2))


def assert_trained(directory):
    rows = training_records(directory)
    steps = [row for row in rows if row['mode'] == 'opt']
    assert steps and {row['epoch'] for row in steps} == {0, 1}
    assert all(np.isfinite(row['loss']) for row in rows if 'loss' in row)
    return len(steps)


@pytest.mark.parametrize('kind', ['hard', 'soft'])
def test_cli_freezing_preserves_weights_while_other_groups_train(adaptation_task, tmp_path, kind):
    import torch

    root, original = adaptation_task
    document = copy.deepcopy(original)
    foundation = root / 'energy/tiny.safetensors'
    document['finetune'] = {'foundation_model': str(foundation)}
    document['data']['heads']['default']['e0s'] = {'kind': 'foundation'}
    if kind == 'hard':
        document['finetune']['freeze'] = 6
    else:
        document['training']['scheduler']['group_factors'] = {'embedding': 0.0}
    before = weights(foundation)
    launch(tmp_path, document, 'fit')
    steps = assert_trained(tmp_path / 'fit')
    after = weights(tmp_path / 'fit/tiny.safetensors')
    assert before.keys() == after.keys()
    markers = {'embedding': 'node_embedding', 'interactions': '.interactions.',
               'products': '.products.', 'readouts': '.readouts.'}
    changes = {}
    for group, marker in markers.items():
        names = [name for name in before if marker in name]
        assert names, group
        moved = [name for name in names if not torch.equal(before[name], after[name])]
        frozen = group == 'embedding' or kind == 'hard' and group != 'readouts'
        assert bool(moved) is not frozen, (group, moved)
        changes[group] = moved
    record(tmp_path, kind=kind, device=document['runtime']['device'], steps=steps, changed=changes)


@pytest.fixture(scope='module')
def dielectric_task(adaptation_task):
    root, original = adaptation_task
    frames = [a for a in read(root / 'train.xyz', ':') if len(a) > 1]
    generator = np.random.default_rng(17)
    for atoms in frames:
        atoms.info['dipole'] = generator.normal(size=3)
        tensor = generator.normal(size=(3, 3))
        atoms.info['polarizability'] = tensor + tensor.T
    source = root / 'dielectric.xyz'
    write(source, frames)
    document = copy.deepcopy(original)
    document['data']['heads']['default'] = {'train_file': str(source)}
    document['model'].update(model='dielectric', observables=['dipole', 'polarizability'],
                             hidden_irreps='0e+1o+2e', max_ell=2,
                             readout={'mlp_irreps': '4x0e+4x1o+4x2e'})
    document['loss'] = {'weights': {'dipole': 1.0, 'polarizability': 1.0}}
    launch(root, document, 'dielectric')
    return root, document


def test_cli_dielectric_finetuning_updates_weights_and_reloads(dielectric_task, tmp_path):
    import torch
    from mace_torch.calculators import MACECalculator
    from mace_torch.deploy.loader import load_deployed

    root, original = dielectric_task
    foundation = root / 'dielectric/tiny.safetensors'
    document = copy.deepcopy(original)
    document['finetune'] = {'foundation_model': str(foundation)}
    document['model'] = {'observables': ['dipole', 'polarizability']}
    before = weights(foundation)
    launch(tmp_path, document, 'fit')
    steps = assert_trained(tmp_path / 'fit')
    path = tmp_path / 'fit/tiny.safetensors'
    after = weights(path)
    assert before.keys() == after.keys()
    changed = [name for name in before if not torch.equal(before[name], after[name])]
    assert changed
    loaded = load_deployed(path, device=document['runtime']['device'])
    assert loaded.heads == ('default',)
    assert loaded.config.model.observables == ('dipole', 'polarizability')
    assert loaded.e0s == {'default': {}}
    calculator = MACECalculator(models=loaded, device=document['runtime']['device'])
    atoms = read(root / 'dielectric.xyz')
    calculator.calculate(atoms)
    for name, shape in [('dipole', (3,)), ('polarizability', (3, 3))]:
        value = calculator.results[name]
        assert value.shape == shape and np.isfinite(value).all()
    record(tmp_path, device=document['runtime']['device'], steps=steps, changed=changed,
           dipole=calculator.results['dipole'].tolist(),
           polarizability=calculator.results['polarizability'].tolist())


def test_cli_dielectric_rejects_an_energy_foundation_before_reading_data(dielectric_task, tmp_path):
    root, original = dielectric_task
    document = copy.deepcopy(original)
    document['model'] = {'observables': ['dipole', 'polarizability']}
    document['finetune'] = {'foundation_model': str(root / 'energy/tiny.safetensors')}
    document['data']['heads']['default']['train_file'] = 'does-not-exist.xyz'
    path = tmp_path / 'invalid.yaml'
    path.write_text(yaml.safe_dump(document))
    done = on_v1('--config', str(path), '--work_dir', str(tmp_path / 'fit'))
    assert done.returncode != 0
    assert 'dipole' in done.stderr and 'does-not-exist' not in done.stderr
    assert not (tmp_path / 'fit/tiny.safetensors').exists()


@pytest.mark.parametrize('storage', ['graph', 'atom'])
def test_cli_conditioning_trains_and_each_feature_changes_predictions(adaptation_task, tmp_path, storage):
    from mace_torch.calculators import MACECalculator

    root, original = adaptation_task
    frames = read(root / 'train.xyz', ':')
    generator = np.random.default_rng(42)
    if storage == 'graph':
        features = [
            {'name': 'elec_temp', 'kind': 'continuous', 'embedding_dim': 2},
            {'name': 'total_charge', 'kind': 'categorical', 'embedding_dim': 2, 'num_classes': 200, 'offset': 100},
            {'name': 'total_spin', 'kind': 'categorical', 'embedding_dim': 2, 'num_classes': 100},
        ]
        for index, atoms in enumerate(frames):
            atoms.info.update(source_temperature=300.0 + 10 * index, total_charge=index % 3 - 1, total_spin=index % 4)
        keys = {'elec_temp': 'source_temperature', 'total_charge': 'total_charge', 'total_spin': 'total_spin'}
    else:
        features = [
            {'name': 'cat', 'kind': 'categorical', 'embedding_dim': 2, 'num_classes': 2, 'per': 'atom'},
            {'name': 'quant', 'kind': 'continuous', 'embedding_dim': 2, 'per': 'atom'},
            {'name': 'quant_v', 'kind': 'continuous', 'embedding_dim': 2, 'input_dim': 2, 'per': 'atom'},
        ]
        for atoms in frames:
            atoms.arrays.update(cat=generator.integers(2, size=len(atoms)),
                                KEY_quant=generator.normal(scale=0.01, size=len(atoms)),
                                quant_v=generator.normal(scale=0.01, size=(len(atoms), 2)))
        keys = {'cat': 'cat', 'quant': 'KEY_quant', 'quant_v': 'quant_v'}
    source = tmp_path / 'conditioned.xyz'
    write(source, frames)
    document = copy.deepcopy(original)
    document['data']['heads']['default']['train_file'] = str(source)
    if storage == 'atom':
        document['data']['embedding_specs'] = {name: {'key': key, 'per': storage} for name, key in keys.items()}
    else:
        document['data']['graph_input_keys'] = {'elec_temp': 'source_temperature'}
    document['model']['graph_features'] = features
    launch(tmp_path, document, 'fit')
    steps = assert_trained(tmp_path / 'fit')
    model = tmp_path / 'fit/tiny.safetensors'
    calculator = MACECalculator(model_paths=model, device=document['runtime']['device'],
                               info_keys={'total_charge': 'total_charge', 'total_spin': 'total_spin'})
    if storage == 'graph':
        assert calculator.info_keys['elec_temp'] == 'source_temperature'
    atoms = frames[2].copy()
    values = atoms.info if storage == 'graph' else atoms.arrays
    for key in keys.values():
        values[key] = (300.0 if key == 'source_temperature' else 0.0) if storage == 'graph' else np.zeros_like(values[key])
    atoms.calc = calculator
    baseline = atoms.get_potential_energy()
    differences = {}
    for name, key in keys.items():
        previous = copy.deepcopy(values[key])
        if storage == 'graph':
            values[key] += 300 if name == 'elec_temp' else 1
        else:
            values[key][0] += 1
        changed = atoms.get_potential_energy()
        assert np.isfinite(changed) and abs(changed - baseline) > FP64_CPU_REFERENCE.atol, name
        differences[name] = float(changed - baseline)
        values[key] = previous
    record(tmp_path, device=document['runtime']['device'], steps=steps, differences=differences)
