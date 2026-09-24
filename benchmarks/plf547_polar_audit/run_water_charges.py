"""Audit manuscript water charges with explicit spin and precision diagnostics."""
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
p.add_argument('--dtype',choices=['float32','float64'],required=True)
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
out = dict(model=args.model, dtype=args.dtype, checkpoint_sha256=sha,
           input_sha256=hashlib.sha256(input_path.read_bytes()).hexdigest(),
           runtime_manifest_sha256=hashlib.sha256((root / 'runtime-manifest.json').read_bytes()).hexdigest(),
           torch=torch.__version__, device=torch.cuda.get_device_name(0),
           status='running', cases={}, controls={}, note='Historical paper geometries; singlet physical state and deliberately invalid doublet diagnostic; no added dispersion')
args.output.parent.mkdir(exist_ok=True, parents=True)

def save():
    out['peak_gpu_GiB'] = torch.cuda.max_memory_allocated()/2**30
    out['peak_host_GiB'] = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/2**20
    tmp = args.output.with_suffix('.tmp')
    with tmp.open('w') as f:
        json.dump(out, f, indent=2, allow_nan=False); f.write('\n'); f.flush(); os.fsync(f.fileno())
    tmp.replace(args.output)

calc = (mace_omol if args.model == 'omol' else mace_polar)(
    model=str(args.checkpoint), device='cuda', default_dtype=args.dtype)

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


assert args.model in ['m','l']
assert 'charges' in calc.implemented_properties

def properties(f,charge,spin):
    a=Atoms(numbers=f['numbers'],positions=f['positions'],pbc=False)
    a.info.update(charge=charge,external_field=[0.,0.,0.])
    if spin is not None:a.info['spin']=spin
    # All supplied systems have even electron count. spin=2 is only a diagnostic.
    assert (sum(a.numbers)-charge)%2==0
    a.calc=calc;calc.reset()
    with torch.no_grad():
        e=float(a.get_potential_energy());q=a.get_charges().tolist()
    assert np.isfinite(e) and np.isfinite(q).all()
    assert abs(sum(q)-charge)<(2e-5 if args.dtype=='float32' else 1e-10)
    return dict(energy_eV=e,atomic_charges=q,total_charge=sum(q),spin=spin)

try:
    path=root/'water_charge_inputs.json';data=json.loads(path.read_text())
    out['water_charge_input_sha256']=hashlib.sha256(path.read_bytes()).hexdigest()
    for system,v in data.items():
        explicit=properties(v['frames'][0],v['charge'],1)
        absent=properties(v['frames'][0],v['charge'],None)
        delta=max(abs(x-y) for x,y in zip(explicit['atomic_charges'],absent['atomic_charges']))
        out['controls'][system+'_default_spin']=dict(max_atomic_charge_difference=delta,energy_difference=absent['energy_eV']-explicit['energy_eV'])
        assert delta<1e-9 and abs(absent['energy_eV']-explicit['energy_eV'])<1e-8
        for i,f in enumerate(v['frames']):
            row={}
            for spin in ([1] if system=='neutral' else [1,2]):
                r=properties(f,v['charge'],spin)
                r['stationary_fragment_charge']=sum(r['atomic_charges'][j] for j in v['stationary_indices'])
                row[str(spin)]=r
            out['cases'][f'{system}/{i:03d}']=row;save()
        print(json.dumps(dict(model=args.model,dtype=args.dtype,system=system,completed=len(out['cases']),peak_gpu_GiB=out['peak_gpu_GiB'])),flush=True)
    assert len(out['cases'])==108
    out['status']='complete';save()
except BaseException as exc:
    out['status']='failed';out['error']=repr(exc);save();raise
finally:
    ext.compute_k_vectors_flat=native_grid
