"""Fresh PLF547 ORB evaluation, with paired faulty-element diagnostics.

Uses the public OMol checkpoint and orb-models 0.5.5, ASE calculator defaults
apart from explicitly selected float64 and compile=False. Does not assume that
the historical public table used this library version or precision.
"""
import argparse
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import resource
import time

import numpy as np
import torch
from ase import Atoms, units
from orb_models.forcefield import pretrained
from orb_models.forcefield.calculator import ORBCalculator

p = argparse.ArgumentParser()
p.add_argument('--checkpoint', type=Path, required=True)
p.add_argument('--output', type=Path, required=True)
args = p.parse_args()
assert not args.output.exists()
assert torch.cuda.is_available()
torch.set_num_threads(2)
torch.cuda.set_per_process_memory_fraction(0.6)
root = Path(__file__).resolve().parent
public = json.loads((root/'public_predictions.json').read_text())
trace = next(t for t in public['data'] if t.get('name') == 'orb-v3-consv-omol')
lookup = {c[0].removesuffix('_complex'): dict(prediction=x, reference=y, charged=c[1])
          for x, y, c in zip(trace['x'], trace['y'], trace['customdata'])}
assert len(lookup) == 547
refs = {}
for line in (root/'inputs/reference_energies.txt').read_text().splitlines()[2:]:
    f = line.split()
    if f: refs['_'.join(f[1].split('_')[:2])] = float(f[-1])
assert set(refs) == set(lookup)
assert all(abs(refs[k]-v['reference']) < 1e-10 for k,v in lookup.items())

def frames_for(label, wrong=False):
    lines = (root/'inputs'/f'{label}.pdb').read_text().splitlines()
    q = {s.split()[1]: float(s.split()[2]) for s in lines if s.startswith('REMARK charge')}
    rows = [s for s in lines if s.startswith(('ATOM  ', 'HETATM'))]
    names = [s[12:16].strip() for s in rows]
    symbols = [('Cl' if n.startswith('Cl') and not wrong else n[0]) for n in names]
    xyz = np.array([[float(s[i:i+8]) for i in (30,38,46)] for s in rows]).astype(np.float32).astype(np.float64)
    mask = np.array([s[17:20].strip().upper() == 'UNK' for s in rows])
    a = Atoms(symbols=symbols, positions=xyz, pbc=False)
    frames = [a,a[~mask],a[mask]]
    for a,key in zip(frames,['charge','charge_a','charge_b']):
        a.info.update(charge=q[key],spin=1)
        if not wrong: assert (sum(a.numbers)-q[key]) % 2 == 0
    assert q['charge'] == q['charge_a']+q['charge_b']
    assert any(v != 0 for v in q.values()) == lookup[label]['charged']
    return frames, any(n.startswith('Cl') for n in names)

affected = [k for k in sorted(lookup) if frames_for(k)[1]]
assert len(affected) == 129
out = dict(status='running', cases={}, controls={}, provenance=dict(
    model='orb-v3-consv-omol', precision='float64', compile=False,
    device=torch.cuda.get_device_name(0),
    checkpoint_sha256=hashlib.sha256(args.checkpoint.read_bytes()).hexdigest(),
    checkpoint_url='https://orbitalmaterials-public-models.s3.us-west-1.amazonaws.com/forcefields/orb-v3-conservative-omol-20250820.ckpt',
    inputs_sha256=hashlib.sha256((root/'inputs.tar.gz').read_bytes()).hexdigest(),
    script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
    environment={k:importlib.metadata.version(k) for k in ['orb-models','torch','numpy','ase']},
    coordinate_convention='PDB positions rounded through float32, then float64',
    note='All 547 corrected cases are fresh. Faulty C side is diagnostic only; historical precision/version not presumed.'))

def save():
    out['peak_gpu_GiB'] = torch.cuda.max_memory_allocated()/2**30
    out['peak_host_GiB'] = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/2**20
    args.output.parent.mkdir(parents=True,exist_ok=True)
    tmp = args.output.with_suffix('.tmp')
    with tmp.open('w') as f:
        json.dump(out,f,indent=2,allow_nan=False);f.write('\n');f.flush();os.fsync(f.fileno())
    tmp.replace(args.output)

cache = {}
model = pretrained.orb_v3_conservative_omol(weights_path=str(args.checkpoint),device='cuda',precision='float64',compile=False)
assert next(model.parameters()).dtype == torch.float64
calc = ORBCalculator(model,device='cuda')
assert calc.expects_charge_and_spin

def score(frames):
    energies=[]
    for a in frames:
        key=hashlib.sha256(a.numbers.tobytes()+a.positions.tobytes()+str(a.info).encode()).hexdigest()
        if key not in cache:
            a.calc=calc;calc.reset()
            cache[key]=float(a.get_potential_energy())
            assert np.isfinite(cache[key])
        energies.append(cache[key])
    return (energies[0]-energies[1]-energies[2])*units.mol/units.kcal,energies

try:
    # Translation invariance and cache/state control on an unaffected charged case.
    frames,cl=frames_for('10GS_01');assert not cl
    orig,energies=score(frames)
    shifted=[a.copy() for a in frames]
    for a in shifted: a.translate([1.25,-2.5,0.75])
    moved,_=score(shifted)
    out['controls']['translation']=dict(original=orig,translated=moved,difference=moved-orig)
    assert abs(moved-orig)<1e-6
    save()
    for label in sorted(lookup):
        start=time.monotonic()
        frames,cl=frames_for(label)
        pred,energies=score(frames)
        row=dict(reference_kcal_mol=refs[label],charged=lookup[label]['charged'],affected=cl,
                 corrected_prediction_kcal_mol=pred,corrected_component_energies_eV=energies,
                 public_prediction_kcal_mol=lookup[label]['prediction'])
        if cl:
            wrong,_=frames_for(label,True)
            for a,b in zip(frames,wrong):
                assert np.array_equal(a.positions,b.positions) and a.info == b.info
                assert np.all((a.numbers==b.numbers)|((a.numbers==17)&(b.numbers==6)))
            bad,bad_energies=score(wrong)
            row.update(reconstructed_bad_prediction_kcal_mol=bad,bad_component_energies_eV=bad_energies,
                       bad_minus_public_kcal_mol=bad-lookup[label]['prediction'])
        row['seconds']=time.monotonic()-start
        out['cases'][label]=row;save()
        print(json.dumps(dict(case=label,corrected=pred,completed=len(out['cases']),peak_gpu_GiB=out['peak_gpu_GiB'])),flush=True)
    groups={'Overall':list(out['cases'].values()),'Cl':[v for v in out['cases'].values() if v['affected']],
            'No_Cl':[v for v in out['cases'].values() if not v['affected']],
            'Charged':[v for v in out['cases'].values() if v['charged']],
            'Neutral':[v for v in out['cases'].values() if not v['charged']]}
    out['metrics']={k:dict(n=len(g),corrected_MAE=sum(abs(v['corrected_prediction_kcal_mol']-v['reference_kcal_mol']) for v in g)/len(g),
                         public_MAE=sum(abs(v['public_prediction_kcal_mol']-v['reference_kcal_mol']) for v in g)/len(g)) for k,g in groups.items()}
    out['status']='complete';save();print(json.dumps(out['metrics']),flush=True)
except BaseException as exc:
    out['status']='failed';out['error']=repr(exc);save();raise
