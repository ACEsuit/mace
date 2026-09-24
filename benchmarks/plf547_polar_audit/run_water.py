"""Fresh water-separation float64 evaluation with last-frame energy zero."""
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
           status='running', cases={}, controls={}, note='Fresh water separation; model and DFT energy zero at last reference frame; singlet multiplicity; no added dispersion')
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

def energy(f,charge,spin):
    a=Atoms(numbers=f['numbers'],positions=f['positions'],pbc=False)
    a.info.update(charge=charge,spin=spin,external_field=[0.,0.,0.])
    assert (sum(a.numbers)-charge)%2 == (spin-1)%2
    calc.reset();a.calc=calc
    with torch.no_grad():value=float(a.get_potential_energy())
    assert np.isfinite(value)
    return value

try:
    if args.model=='omol':
        frames=data['cases']['QUID/F1B1']['frames'];expected=data['omol_control']
    else:
        frames=data['polar_control_frames'];expected=data['polar_controls'][args.model]
    e=[energy(f,f['charge'],f['spin']) for f in frames]
    control=(e[0]-e[1]-e[2])*units.mol/units.kcal
    assert abs(control-expected)<1e-5
    out['controls']['CPU']=dict(prediction=control,expected=expected,difference=control-expected)
    path=root/'water_inputs.json';water=json.loads(path.read_text())
    out['water_input_sha256']=hashlib.sha256(path.read_bytes()).hexdigest()
    out['metrics']={}
    for system,v in water.items():
        pred=[]
        for i,f in enumerate(v['frames']):
            value=energy(f,v['charge'],v['spin']);pred.append(value)
            out['cases'][f'{system}/{i:03d}']=dict(energy_eV=value,reference_eV=f['reference_eV'],charge=v['charge'],spin=v['spin'])
            save()
        errors=[]
        for i,(e,f) in enumerate(zip(pred,v['frames'])):
            model_rel=(e-pred[-1])*units.mol/units.kcal
            ref_rel=(f['reference_eV']-v['frames'][-1]['reference_eV'])*units.mol/units.kcal
            out['cases'][f'{system}/{i:03d}'].update(model_relative_kcal_mol=model_rel,reference_relative_kcal_mol=ref_rel)
            errors.append(model_rel-ref_rel)
        out['metrics'][system]=dict(n=len(errors),MAE=float(np.mean(np.abs(errors))),RMSE=float(np.sqrt(np.mean(np.square(errors)))),max_error=float(max(np.abs(errors))))
        save();print(json.dumps(dict(model=args.model,system=system,metrics=out['metrics'][system])),flush=True)
    assert len(out['cases'])==107
    out['status']='complete';save()
except BaseException as exc:
    out['status']='failed';out['error']=repr(exc);save();raise
finally:
    ext.compute_k_vectors_flat=native_grid
