"""Gated dielectric training and canonical reload across CPU/CUDA backends."""

import json
import shutil

import numpy as np
import pytest
from ase.io import write

from tests.golden.harness import FP64_ACCELERATED_BACKEND
from tests.workflows.test_data_sources_v1 import launch
# Reuse the historical structures and labels, including isolated atoms.
from tests.workflows.test_run_train_dipole_polar import fixture_fitting_configs
from tests.workflows.test_run_train_v1 import needs_the_v1_engine, training_records

pytestmark = [needs_the_v1_engine, pytest.mark.gpu, pytest.mark.cueq]


@pytest.mark.parametrize('train_backend', ['reference', 'cueq'])
def test_gated_dielectric_training_and_backend_reload(tmp_path, fitting_configs, train_backend):
    from mace_torch.backends.cueq import compiled_operations_available
    from mace_torch.calculators import MACECalculator

    assert compiled_operations_available()
    source = tmp_path / 'fit.xyz'
    prepared = []
    for original in fitting_configs:
        atoms = original.copy()
        for old, new in [('REF_dipoles', 'dipole'), ('REF_polarizability', 'polarizability')]:
            if old in atoms.info:
                atoms.info[new] = atoms.info.pop(old)
        prepared.append(atoms)
    write(source, prepared)
    document = {
        'runtime': {'name': 'fit', 'seed': 5, 'device': 'cuda'},
        'precision': {'model': 'float64'},
        'data': {'heads': {'target': {'train_file': str(source)}},
                 'valid_fraction': 0.1, 'pin_memory': False},
        'model': {'model': 'dielectric', 'backend': train_backend,
                  'observables': ['dipole', 'polarizability'], 'r_max': 3.5,
                  'num_interactions': 2, 'num_channels': 8,
                  'hidden_irreps': '0e+1o+2e', 'max_ell': 2, 'correlation': 3,
                  'readout': {'mlp_irreps': '16x0e+16x1o+16x2e'}},
        'training': {'max_num_epochs': 2, 'batch_size': 5, 'valid_batch_size': 2,
                     'lr': 0.001, 'scheduler': {'kind': 'constant'}},
        'loss': {'weights': {'dipole': 1.0, 'polarizability': 1.0}},
    }
    launch(tmp_path, document, 'trained')
    records = training_records(tmp_path / 'trained', 'fit')
    steps = [row for row in records if row['mode'] == 'opt']
    assert steps and {row['epoch'] for row in steps} == {0, 1}
    assert all(np.isfinite(row['loss']) for row in records if 'loss' in row)
    checkpoint = tmp_path / 'trained/fit.safetensors'
    metadata = json.loads(checkpoint.with_suffix('.json').read_text())
    assert metadata['config']['config']['resolved']['model']['backend'] == train_backend
    rotation, _ = np.linalg.qr(np.random.default_rng(19).normal(size=(3, 3)))
    assert np.linalg.det(rotation) > 0
    baseline = None
    evidence = {}

    def close(actual, expected):
        np.testing.assert_allclose(actual, expected, atol=FP64_ACCELERATED_BACKEND.atol,
                                   rtol=FP64_ACCELERATED_BACKEND.rtol)

    for device, backend in [('cpu', 'reference'), ('cuda', 'reference'), ('cuda', 'cueq')]:
        # Backend choice is build-time metadata; the weight file is identical.
        target = tmp_path / f'{device}-{backend}.safetensors'
        shutil.copyfile(checkpoint, target)
        metadata['config']['config']['resolved']['model']['backend'] = backend
        target.with_suffix('.json').write_text(json.dumps(metadata))
        calculator = MACECalculator(model_paths=target, device=device, head='target')
        called = set()
        ops = {name: module for name, module in calculator.models[0].engine.named_modules()
               if type(module).__name__.startswith('CuEq')}
        if backend == 'cueq':
            assert ops
        hooks = [module.register_forward_hook(lambda m, a, o, name=name: called.add(name))
                 for name, module in ops.items()]
        predictions = []
        try:
            for atoms in fitting_configs:
                calculator.calculate(atoms)
                dipole = calculator.results['dipole'].copy()
                polar = calculator.results['polarizability'].copy()
                assert np.isfinite(dipole).all() and np.isfinite(polar).all()
                predictions.append(np.concatenate([dipole, polar.ravel()]))
                rotated = atoms.copy()
                rotated.positions = atoms.positions @ rotation.T
                rotated.cell = atoms.cell.array @ rotation.T
                calculator.calculate(rotated)
                close(calculator.results['dipole'], rotation @ dipole)
                close(calculator.results['polarizability'], rotation @ polar @ rotation.T)
        finally:
            for hook in hooks:
                hook.remove()
        assert called == ops.keys()
        if baseline is None:
            baseline = np.asarray(predictions)
        close(predictions, baseline)
        evidence[f'{device}-{backend}'] = {
            'max_deviation': float(np.max(np.abs(predictions - baseline))),
            'accelerated_operations_called': sorted(called),
        }
    (tmp_path / 'backend-evidence.json').write_text(json.dumps({
        'train_backend': train_backend, 'steps': len(steps), 'frames': len(fitting_configs),
        'evaluations': evidence,
    }, indent=2))
