"""Published foundations enter native replay training by explicit conversion.

The real MP replay pool and the seeds/sample counts of the historical workflows
are retained. Pickle extraction runs only in the converter's legacy subprocess.
"""

import gc
import hashlib
import json
import os
import sys
from dataclasses import asdict
from pathlib import Path

import numpy as np
import pytest

from tests.golden.foundation_artifacts import ARTIFACTS
from tests.golden.harness import FP32, FP64_CPU_REFERENCE
from tests.workflows.test_data_sources_v1 import launch
from tests.workflows.test_finetuning_pseudolabels import _make_small_real_head_dataset
from tests.workflows.test_run_train_v1 import needs_the_v1_engine, training_records

pytestmark = [needs_the_v1_engine, pytest.mark.network]
GOLDEN = Path(__file__).resolve().parents[1] / 'golden'
HEADS = {
    'mh_0': ('rgd1_b3lyp', 'matpes_r2scan', 'mp_pbe_refit_add', 'omol',
             'spice_wB97M', 'oc20_usemppbe', 'omat_pbe'),
    'mh_1': ('matpes_r2scan', 'mp_pbe_refit_add', 'spice_wB97M',
             'oc20_usemppbe', 'omol', 'omat_pbe'),
    'omol': ('omol',),
}


@pytest.mark.parametrize('device', ['cpu', pytest.param('cuda', marks=pytest.mark.gpu)])
@pytest.mark.parametrize('name,seed,count', [('mh_0', 42, 3), ('mh_1', 43, 3), ('omol', 44, 5)])
def test_published_foundation_replay_starts_at_zero_error(tmp_path, monkeypatch, name, seed, count, device):
    from mace_core.artifacts import resolve_artifact
    from mace_core.observables import DEFAULT_CATALOGUE
    from mace_torch.deploy.legacy import convert_legacy
    from mace_torch.deploy.loader import load_deployed
    from mace_torch.deploy.reference import verify_against_reference
    from mace_torch.finetune.pseudolabels import load_pseudolabel_artifact
    from mace_torch.train.data_stage import graph_inputs_of

    artifact = ARTIFACTS[name]
    source = resolve_artifact(artifact.origin)
    with source.open('rb') as stream:
        digest = hashlib.sha256()
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    assert digest.hexdigest() == artifact.sha256
    monkeypatch.setenv('PYTHONPATH', os.pathsep.join(
        [str(GOLDEN.parents[1]), os.environ.get('PYTHONPATH', '')]))
    conversion = convert_legacy(source, tmp_path / 'foundation', DEFAULT_CATALOGUE,
                                python=sys.executable)
    assert conversion.model.heads == HEADS[name]
    foundation_path = conversion.checkpoint.with_suffix('.safetensors')
    del conversion
    gc.collect()
    model = load_deployed(foundation_path, device=device)
    assert model.heads == HEADS[name]
    source_head = 'omol' if name == 'omol' else 'omat_pbe'
    golden_head = artifact.loader_kwargs.get('head', source_head)
    # FM-2: these published artifacts contain float32 coupling constants even
    # when loaded as float64. The existing FP32 golden row applies to them.
    report = verify_against_reference(
        model.engine, GOLDEN / 'references' / artifact.reference, GOLDEN / 'fixtures',
        z_table=model.z_table, cutoff=model.config.model.r_max,
        head=model.heads.index(golden_head), graph_inputs=graph_inputs_of(model.config.model),
        device=device, atol=FP32.atol, rtol=FP32.rtol)
    assert report.deviations and not report.skipped
    assert report.passed, report.describe()
    (tmp_path / 'golden-report.json').write_text(json.dumps(asdict(report), indent=2))
    # Read the same E0 table consumed by the data stage, before releasing the
    # parent process's copy of this potentially large foundation.
    expected_e0s = model.metadata.heads[source_head].e0.values
    del model
    gc.collect()
    if device == 'cuda':
        import torch
        torch.cuda.empty_cache()

    document = {
        'runtime': {'name': 'fit', 'seed': seed, 'device': device},
        'precision': {'model': 'float64'},
        'finetune': {
            'foundation_model': str(foundation_path),
            'pseudolabels': {'enabled': True, 'heads': ['memory'], 'batch_size': 2},
        },
        'data': {'heads': {
            'memory': {'curated': 'mp', 'readout_from': source_head,
                       'e0s': {'kind': 'foundation', 'head': source_head},
                       'subselect': {'method': 'random', 'num_samples': count}},
            'target': {'train_file': str(_make_small_real_head_dataset(tmp_path)),
                       'readout_from': source_head, 'e0s': {'kind': 'isolated_atoms'}},
        }, 'valid_fraction': 0.3, 'pin_memory': False},
        'model': {'observables': ['energy', 'forces']},
        'training': {'max_num_epochs': 0, 'batch_size': 2, 'valid_batch_size': 1},
        'loss': {'weights': {'energy': 1.0, 'forces': 100.0}},
    }
    launch(tmp_path, document, 'run')
    records = training_records(tmp_path / 'run', 'fit')
    initial = [row for row in records if row['mode'] == 'valid' and row['epoch'] is None]
    assert {row['head'] for row in initial} == {'memory', 'target'}
    replay, = [row for row in initial if row['head'] == 'memory']
    for key in ('loss', 'rmse_energy_per_atom', 'rmse_forces'):
        assert np.isfinite(replay[key]), replay
        assert 0 <= replay[key] <= FP64_CPU_REFERENCE.atol, replay

    labels, provenance = load_pseudolabel_artifact(tmp_path / 'run/pseudolabels/memory')
    assert len(labels) == provenance.n_configs == count
    assert provenance.foundation_head == source_head
    assert provenance.dtype == 'float64'
    assert provenance.device.split(':')[0] == device
    assert all({'energy', 'forces'} <= item.properties.keys() for item in labels)
    sidecar = json.loads((tmp_path / 'run/fit.json').read_text())
    assert set(sidecar['config']['heads']) == {'memory', 'target'}
    assert sidecar['config']['heads']['memory']['e0']['values'] == expected_e0s
    resolved = sidecar['config']['config']['resolved']
    assert resolved['data']['heads']['memory']['curated'] == 'mp'
    assert resolved['data']['heads']['memory']['readout_from'] == source_head
    (tmp_path / 'workflow-evidence.json').write_text(json.dumps({
        'artifact_sha256': artifact.sha256, 'source_heads': HEADS[name],
        'foundation_e0s': expected_e0s, 'initial_metrics': initial,
        'pseudolabel_provenance': provenance.model_dump(mode='json'),
    }, indent=2, default=str))


@pytest.mark.parametrize('device', ['cpu', pytest.param('cuda', marks=pytest.mark.gpu)])
@pytest.mark.parametrize('name', ['mh_1', 'omol'])
def test_published_foundation_trains_updates_weights_and_reloads(tmp_path, monkeypatch, fitting_configs, name, device):
    import torch
    from ase.io import write
    from mace_core.artifacts import resolve_artifact
    from mace_core.config.resolved import ResolvedConfig
    from mace_core.observables import DEFAULT_CATALOGUE
    from mace_torch.calculators import MACECalculator
    from mace_torch.deploy.legacy import convert_legacy
    from mace_torch.deploy.loader import load_deployed
    from mace_torch.deploy.reference import verify_against_reference
    from mace_torch.finetune.stages import build
    from mace_torch.train.data_stage import graph_inputs_of

    artifact = ARTIFACTS[name]
    source = resolve_artifact(artifact.origin)
    with source.open('rb') as stream:
        digest_builder = hashlib.sha256()
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest_builder.update(block)
    digest = digest_builder.hexdigest()
    assert digest == artifact.sha256
    monkeypatch.setenv('PYTHONPATH', os.pathsep.join(
        [str(GOLDEN.parents[1]), os.environ.get('PYTHONPATH', '')]))
    conversion = convert_legacy(source, tmp_path / 'foundation', DEFAULT_CATALOGUE,
                                python=sys.executable)
    foundation_path = conversion.checkpoint.with_suffix('.safetensors')
    del conversion
    gc.collect()
    parent = load_deployed(foundation_path, device=device)
    assert parent.heads == HEADS[name]
    source_head = 'omol' if name == 'omol' else 'omat_pbe'
    report = verify_against_reference(
        parent.engine, GOLDEN / 'references' / artifact.reference, GOLDEN / 'fixtures',
        z_table=parent.z_table, cutoff=parent.config.model.r_max,
        head=parent.heads.index(artifact.loader_kwargs.get('head', source_head)),
        graph_inputs=graph_inputs_of(parent.config.model), device=device,
        atol=FP32.atol, rtol=FP32.rtol)
    assert report.deviations and not report.skipped
    assert report.passed, report.describe()
    elements = tuple(parent.z_table.zs)
    del parent
    gc.collect()
    if device == 'cuda':
        torch.cuda.empty_cache()

    # These are the same 20 periodic waters and two isolated references used by
    # the historical OMOL/mh-1 training workflows; their E0s are both zero.
    source_data = tmp_path / 'fit.xyz'
    write(source_data, fitting_configs)
    document = {
        'runtime': {'name': 'fit', 'seed': 123, 'device': device},
        'precision': {'model': 'float64'},
        'finetune': {'foundation_model': str(foundation_path)},
        'data': {'heads': {'target': {
            'train_file': str(source_data), 'readout_from': source_head,
            'e0s': {'kind': 'isolated_atoms'},
        }}, 'valid_fraction': 0.1, 'pin_memory': False},
        'model': {'observables': ['energy', 'forces']},
        'training': {'max_num_epochs': 2, 'batch_size': 2, 'valid_batch_size': 2,
                     'lr': 0.001, 'scheduler': {'kind': 'constant'}},
        'loss': {'weights': {'energy': 1.0, 'forces': 100.0}},
    }

    def fingerprints(engine):
        return {key: hashlib.sha256(value.detach().cpu().numpy().tobytes()).hexdigest()
                for key, value in engine.named_parameters() if value.numel()}

    built = build(ResolvedConfig.from_dict(document))
    before = fingerprints(built.model)
    assert built.data.e0s.values['target'][1] == built.data.e0s.values['target'][8] == 0.0
    del built
    gc.collect()
    if device == 'cuda':
        torch.cuda.empty_cache()
    launch(tmp_path, document, 'trained')
    records = training_records(tmp_path / 'trained', 'fit')
    updates = [row for row in records if row['mode'] == 'opt']
    assert updates and {row['epoch'] for row in updates} == {0, 1}
    assert all(np.isfinite(row['loss']) for row in records if 'loss' in row)
    loaded = load_deployed(tmp_path / 'trained/fit.safetensors', device=device)
    assert loaded.heads == ('target',)
    assert tuple(loaded.z_table.zs) == elements
    assert loaded.e0s['target'][1] == loaded.e0s['target'][8] == 0.0
    assert loaded.config.data.heads['target'].readout_from == source_head
    after = fingerprints(loaded.engine)
    assert before.keys() == after.keys()
    changed = [key for key in before if before[key] != after[key]]
    for group in ('node_embedding', '.interactions.', '.products.', '.readouts.'):
        assert any(group in key for key in changed), (group, changed)
    calculator = MACECalculator(models=loaded, device=device, head='target')
    predictions = []
    for atoms in fitting_configs:
        atoms.calc = calculator
        energy, forces = atoms.get_potential_energy(), atoms.get_forces()
        assert np.isfinite(energy) and np.isfinite(forces).all()
        assert forces.shape == (len(atoms), 3)
        predictions.append({'energy': energy, 'forces': forces.tolist()})
    assert len(predictions) == 22
    (tmp_path / 'training-evidence.json').write_text(json.dumps({
        'artifact_sha256': digest, 'device': device, 'source_head': source_head,
        'steps': len(updates), 'changed': changed, 'golden': asdict(report),
        'predictions': predictions,
    }, indent=2))
