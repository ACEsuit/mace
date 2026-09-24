"""Fresh QUID/PLA15 float64 evaluation using the validated isolated runtime."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import resource
import sys
import time

p = argparse.ArgumentParser()
p.add_argument('--model', choices=['m', 'l', 'omol'], required=True)
p.add_argument('--checkpoint', type=Path, required=True)
p.add_argument('--output', type=Path, required=True)
args = p.parse_args()
root = Path(__file__).resolve().parent
sys.path.insert(0, str(root / 'runtime'))
for name, sha in json.loads((root / 'runtime-manifest.json').read_text()).items():
    assert hashlib.sha256((root / 'runtime' / name).read_bytes()).hexdigest() == sha
import numpy as np
import torch
from ase import Atoms, units
from mace.calculators import mace_polar, mace_omol
import mace.modules.extensions as ext

assert not args.output.exists(), 'Refusing to overwrite an output'
assert torch.cuda.is_available()
torch.set_num_threads(2)
torch.cuda.set_per_process_memory_fraction(0.60)
sha = hashlib.sha256(args.checkpoint.read_bytes()).hexdigest()
assert sha == {'m': 'fab8b8713c832f31a2a853aaa22fd638be8a369cbf5095e6b3e982a18d10e93a',
               'l': '9f65f8dc6ddaff1d631e299cb531376a7da5e68d1bef04f34a2d5073d5ef114b',
               'omol': '9b64b4fd5153ca578c694abc57806d8111050de6ff652e695c9b525bc4d36469'}[args.model]
input_path = root / 'followup_inputs.json'
data = json.loads(input_path.read_text())
out = dict(model=args.model, dtype='float64', checkpoint_sha256=sha,
           input_sha256=hashlib.sha256(input_path.read_bytes()).hexdigest(),
           runtime_manifest_sha256=hashlib.sha256((root / 'runtime-manifest.json').read_bytes()).hexdigest(),
           torch=torch.__version__, device=torch.cuda.get_device_name(0),
           status='running', cases={}, controls={}, note='Fresh all-case QUID and PLA15 evaluation; no added dispersion')
args.output.parent.mkdir(exist_ok=True, parents=True)

def save():
    out['peak_gpu_GiB'] = torch.cuda.max_memory_allocated()/2**30
    out['peak_host_GiB'] = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/2**20
    tmp = args.output.with_suffix('.tmp')
    with tmp.open('w') as f:
        json.dump(out, f, indent=2, allow_nan=False); f.write('\n'); f.flush(); os.fsync(f.fileno())
    tmp.replace(args.output)

calc = (mace_omol if args.model == 'omol' else mace_polar)(
    model=str(args.checkpoint), device='cuda', default_dtype='float64')

def flags(model, positional, kwargs):
    d = positional[0] if positional else kwargs['data']
    assert len(positional) <= 1 and not bool(torch.any(d['pbc']))
    assert not kwargs.get('use_pbc_evaluator', False)
    kwargs.update(compute_force=False, compute_stress=False, compute_virials=False,
                  compute_displacement=False, compute_edge_forces=False,
                  compute_atomic_stresses=False, compute_hessian=False)
    return positional, kwargs

calc.models[0].register_forward_pre_hook(flags, with_kwargs=True)
native_grid = ext.compute_k_vectors_flat
if args.model != 'omol':
    def empty_grid(cutoff, cell, rcell):
        return cell.new_zeros((0, 3)), cell.new_zeros((0,)), torch.empty(0, dtype=torch.long, device=cell.device), cell.new_zeros((0,))
    ext.compute_k_vectors_flat = empty_grid
cache = {}

def score(frames):
    energies = []
    for f in frames:
        key = hashlib.sha256(json.dumps(f, sort_keys=True).encode()).hexdigest()
        if key not in cache:
            a = Atoms(numbers=f['numbers'], positions=f['positions'], pbc=False)
            a.info.update(charge=f['charge'], spin=f['spin'], external_field=[0., 0., 0.])
            assert (sum(a.numbers)-f['charge']) % 2 == 0 and f['spin'] == 1
            a.calc = calc; calc.reset()
            with torch.no_grad(): cache[key] = float(a.get_potential_energy())
            assert np.isfinite(cache[key])
        energies.append(cache[key])
    return (energies[0]-energies[1]-energies[2])*units.mol/units.kcal, energies

try:
    if args.model == 'omol':
        frames = data['cases']['QUID/F1B1']['frames']; expected = data['omol_control']
    else:
        frames = data['polar_control_frames']; expected = data['polar_controls'][args.model]
    control, _ = score(frames)
    out['controls']['CPU'] = dict(prediction=control, expected=expected, difference=control-expected)
    assert abs(control-expected) < 1e-5
    save()
    # QUID first: small cases finish even if a large PLA active site hits the cap.
    labels = sorted(data['cases'], key=lambda k: (not k.startswith('QUID/'), len(data['cases'][k]['frames'][0]['numbers']), k))
    for label in labels:
        v = data['cases'][label]
        start = time.monotonic()
        pred, energies = score(v['frames'])
        out['cases'][label] = {k:x for k,x in v.items() if k != 'frames'}
        out['cases'][label].update(prediction_kcal_mol=pred, component_energies_eV=energies,
            seconds=time.monotonic()-start, atoms=len(v['frames'][0]['numbers']))
        save()
        print(json.dumps(dict(model=args.model, case=label, prediction=pred, completed=len(out['cases']), peak_gpu_GiB=out['peak_gpu_GiB'])), flush=True)
    groups = {'QUID_equilibrium42': [v for v in out['cases'].values() if v['benchmark']=='QUID' and v['equilibrium']],
              'QUID_dissociation48': [v for v in out['cases'].values() if v['benchmark']=='QUID' and not v['equilibrium'] and not v['duplicate']],
              'QUID_unique90': [v for v in out['cases'].values() if v['benchmark']=='QUID' and not v['duplicate']],
              'QUID_all96': [v for v in out['cases'].values() if v['benchmark']=='QUID'],
              'PLA15': [v for v in out['cases'].values() if v['benchmark']=='PLA15']}
    assert [len(v) for v in groups.values()] == [42, 48, 90, 96, 15]
    out['metrics'] = {k:dict(n=len(g),MAE=sum(abs(v['prediction_kcal_mol']-v['reference_kcal_mol']) for v in g)/len(g)) for k,g in groups.items()}
    out['status']='complete';save()
    print(json.dumps(dict(model=args.model, status=out['status'], metrics=out['metrics'], peak_gpu_GiB=out['peak_gpu_GiB'])),flush=True)
except BaseException as exc:
    out['status']='failed';out['error']=repr(exc);save();raise
finally:
    ext.compute_k_vectors_flat = native_grid
