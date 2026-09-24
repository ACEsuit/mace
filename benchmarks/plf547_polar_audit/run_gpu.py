"""Controlled PLF547 Cl/C comparison; CUDA only, atomic progress, pinned sources."""
import argparse
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import resource
import sys
import time


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--size', choices=['m', 'l'], required=True)
    p.add_argument('--phase', choices=['smoke', 'full'], required=True)
    p.add_argument('--checkpoint', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--resume', action='store_true')
    args = p.parse_args()
    root = Path(__file__).resolve().parent
    runtime = root / 'runtime'
    sys.path.insert(0, str(runtime))
    for name, sha in json.loads((root / 'runtime-manifest.json').read_text()).items():
        assert hashlib.sha256((runtime / name).read_bytes()).hexdigest() == sha, name

    import numpy as np
    import torch
    from ase import Atoms, units
    from mace.calculators import mace_polar
    import mace.modules.extensions as ext

    assert torch.cuda.is_available(), 'CUDA GPU required; no CPU fallback'
    torch.set_num_threads(2)
    torch.cuda.set_per_process_memory_fraction(0.60)
    expected_sha = {
        'm': 'fab8b8713c832f31a2a853aaa22fd638be8a369cbf5095e6b3e982a18d10e93a',
        'l': '9f65f8dc6ddaff1d631e299cb531376a7da5e68d1bef04f34a2d5073d5ef114b',
    }[args.size]
    assert hashlib.sha256(args.checkpoint.read_bytes()).hexdigest() == expected_sha
    public = json.loads((root / 'public_predictions.json').read_text())
    trace = next(t for t in public['data'] if t.get('name') == f'mace-polar-1-{args.size}')
    lookup = {c[0].removesuffix('_complex'): dict(prediction=x, reference=y, charged=c[1])
              for x, y, c in zip(trace['x'], trace['y'], trace['customdata'])}
    assert len(lookup) == 547
    cpu = json.loads((root / f'cpu_partial_{args.size}.json').read_text())
    refs = {}
    for line in (root / 'inputs/reference_energies.txt').read_text().splitlines()[2:]:
        fields = line.split()
        if fields:
            refs['_'.join(fields[1].split('_')[:2])] = float(fields[-1])
    assert set(refs) == set(lookup)
    assert all(abs(refs[k] - v['reference']) < 1e-10 for k, v in lookup.items())

    def frames_for(label, wrong=False):
        lines = (root / 'inputs' / f'{label}.pdb').read_text().splitlines()
        q = {s.split()[1]: float(s.split()[2]) for s in lines if s.startswith('REMARK charge')}
        rows = [s for s in lines if s.startswith(('ATOM  ', 'HETATM'))]
        names = [s[12:16].strip() for s in rows]
        symbols = [('Cl' if n.startswith('Cl') and not wrong else n[0]) for n in names]
        xyz = np.array([[float(s[i:i+8]) for i in (30, 38, 46)] for s in rows])
        xyz = xyz.astype(np.float32).astype(np.float64)
        mask = np.array([s[17:20].strip().upper() == 'UNK' for s in rows])
        a = Atoms(symbols=symbols, positions=xyz, pbc=False)
        frames = [a, a[~mask], a[mask]]
        for a, key in zip(frames, ['charge', 'charge_a', 'charge_b']):
            a.info.update(charge=q[key], spin=1, external_field=[0., 0., 0.])
            if not wrong:
                assert (sum(a.numbers) - q[key]) % 2 == 0
        assert q['charge'] == q['charge_a'] + q['charge_b']
        assert any(v != 0 for v in q.values()) == lookup[label]['charged']
        return frames, any(n.startswith('Cl') for n in names)

    affected = [k for k in sorted(lookup) if frames_for(k)[1]]
    assert len(affected) == 129
    # The only deliberate geometry alteration is atomic identity Cl -> C on the
    # diagnostic side. Charges, positions, fragments and multiplicity stay fixed.
    for label in affected:
        correct, _ = frames_for(label)
        wrong, _ = frames_for(label, True)
        for a, b in zip(correct, wrong):
            assert np.array_equal(a.positions, b.positions) and a.info == b.info
            assert np.all((a.numbers == b.numbers) | ((a.numbers == 17) & (b.numbers == 6)))

    provenance = dict(model=f'mace-polar-1-{args.size}', checkpoint_sha256=expected_sha,
                      dtype='float64', device=torch.cuda.get_device_name(0),
                      source_manifest_sha256=hashlib.sha256((root / 'runtime-manifest.json').read_bytes()).hexdigest(),
                      public_sha256=hashlib.sha256((root / 'public_predictions.json').read_bytes()).hexdigest(),
                      inputs_sha256=hashlib.sha256((root / 'inputs.tar.gz').read_bytes()).hexdigest(),
                      environment={k: importlib.metadata.version(k) for k in ['torch', 'e3nn', 'numpy', 'ase', 'scipy']},
                      coordinate_convention='PDB positions rounded through float32 on both sides',
                      public_tolerance_kcal_mol=1e-5,
                      kgrid_change='Only unused nonperiodic k-grid allocation bypassed; original energy expression retained')
    out = dict(provenance=provenance, status='running', cases={}, controls={})
    if args.output.exists():
        if not args.resume:
            raise FileExistsError(f'{args.output}; use --resume explicitly')
        out = json.loads(args.output.read_text())
        assert out['provenance'] == provenance
        out['status'] = 'running'
    args.output.parent.mkdir(parents=True, exist_ok=True)

    def save():
        out['peak_gpu_allocated_GiB'] = torch.cuda.max_memory_allocated() / 2**30
        out['peak_gpu_reserved_GiB'] = torch.cuda.max_memory_reserved() / 2**30
        out['peak_host_RSS_GiB'] = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 2**20
        tmp = args.output.with_suffix('.tmp')
        with tmp.open('w') as f:
            json.dump(out, f, indent=2, allow_nan=False)
            f.write('\n'); f.flush(); os.fsync(f.fileno())
        tmp.replace(args.output)

    calc = mace_polar(model=str(args.checkpoint), device='cuda', default_dtype='float64')
    # Enforce isolated real-space evaluation before supplying an empty k-grid.
    def flags(model, positional, kwargs):
        data = positional[0] if positional else kwargs['data']
        assert len(positional) <= 1, 'Evaluation flags must be passed by keyword'
        assert not bool(torch.any(data['pbc']))
        assert not kwargs.get('use_pbc_evaluator', False)
        kwargs.update(compute_force=False, compute_stress=False, compute_virials=False,
                      compute_displacement=False, compute_edge_forces=False,
                      compute_atomic_stresses=False, compute_hessian=False)
        return positional, kwargs
    calc.models[0].register_forward_pre_hook(flags, with_kwargs=True)
    native_grid = ext.compute_k_vectors_flat

    def empty_grid(cutoff, cell, rcell):
        return (cell.new_zeros((0, 3)), cell.new_zeros((0,)),
                torch.empty(0, dtype=torch.long, device=cell.device), cell.new_zeros((0,)))

    def bounded_native_grid(cutoff, cell, rcell):
        norms = torch.linalg.vector_norm(cell, dim=-1)
        dots = torch.einsum('bij,bij->bi', rcell, cell / norms.unsqueeze(-1))
        maxima = torch.ceil(cutoff / dots).long().max(dim=0).values
        n1, n2, n3 = [int(v) for v in maxima]
        count = 1 + max(n3-1, 0) + max(n2-1, 0)*2*n3 + max(n1-1, 0)*2*n2*2*n3
        assert count <= 2_000_000, f'Native-grid control exceeds allocation guard: {count}'
        return native_grid(cutoff, cell, rcell)

    def energy(a):
        assert not a.pbc.any()
        calc.reset(); a.calc = calc
        with torch.no_grad():
            value = float(a.get_potential_energy())
        assert np.isfinite(value)
        return value

    cache = {}
    def score(frames):
        energies = []
        for a in frames:
            key = hashlib.sha256(a.numbers.tobytes() + a.positions.tobytes()
                                 + str((a.info['charge'], a.info['spin'])).encode()).hexdigest()
            if key not in cache:
                cache[key] = energy(a)
            energies.append(cache[key])
        return (energies[0]-energies[1]-energies[2])*units.mol/units.kcal, energies

    try:
        # Native vs bypassed GPU energy on a small molecule, with strict cap on
        # native grid size. The CPU PLF controls below additionally cover large,
        # charged molecules and the actual benchmark fragment convention.
        water = Atoms('OH2', positions=[[0, 0, 0], [.9572, 0, 0], [-.239987, .927297, 0]])
        water.info.update(charge=0., spin=1, external_field=[0., 0., 0.])
        ext.compute_k_vectors_flat = bounded_native_grid
        native = energy(water)
        ext.compute_k_vectors_flat = empty_grid
        bypass = energy(water)
        assert abs(native - bypass) < 1e-9, (native, bypass)
        out['controls']['native_grid_water'] = dict(native_eV=native, bypass_eV=bypass, difference_eV=bypass-native)
        save()
        for label in ['10GS_01', '2FVD_01', '2XB8_01']:
            frames, cl = frames_for(label); assert not cl
            value, _ = score(frames)
            diff = value - lookup[label]['prediction']
            out['controls'][label] = dict(prediction_kcal_mol=value, public_difference_kcal_mol=diff)
            assert abs(diff) < 1e-5, (label, diff)
            save()
        label = '2OBF_01'
        value, energies = score(frames_for(label)[0])
        cpu_case = cpu['cases'][label]
        delta = value - cpu_case['corrected_prediction_kcal_mol']
        component_delta = max(abs(a-b) for a, b in zip(energies, cpu_case['corrected_component_energies_eV']))
        out['controls']['corrected_Cl_CPU'] = dict(difference_kcal_mol=delta, max_component_difference_eV=component_delta)
        assert abs(delta) < 1e-5 and component_delta < 1e-6, (delta, component_delta)
        save()
        selected = affected if args.phase == 'full' else ['2OBF_01']
        for label in selected:
            if label in out['cases']:
                assert abs(out['cases'][label]['bad_minus_public_kcal_mol']) < 1e-5
                continue
            start = time.monotonic()
            correct, _ = frames_for(label)
            wrong, _ = frames_for(label, True)
            pred, energies = score(correct)
            bad, bad_energies = score(wrong)
            old = lookup[label]
            diff = bad - old['prediction']
            out['cases'][label] = dict(affected=True, reference_kcal_mol=old['reference'], charged=old['charged'],
                public_prediction_kcal_mol=old['prediction'], corrected_prediction_kcal_mol=pred,
                reconstructed_bad_prediction_kcal_mol=bad, bad_minus_public_kcal_mol=diff,
                corrected_component_energies_eV=energies, bad_component_energies_eV=bad_energies,
                seconds=time.monotonic()-start, source='fresh paired GPU calculation')
            save()
            print(json.dumps(dict(model=args.size, case=label, correct=pred, bad_public_delta=diff,
                                  completed=len(out['cases']), peak_gpu_GiB=out['peak_gpu_allocated_GiB'])), flush=True)
            assert abs(diff) < 1e-5, (label, diff)
        if args.phase == 'full':
            for label, old in lookup.items():
                if label not in affected:
                    out['cases'][label] = dict(affected=False, reference_kcal_mol=old['reference'], charged=old['charged'],
                        public_prediction_kcal_mol=old['prediction'], corrected_prediction_kcal_mol=old['prediction'],
                        source='unchanged public prediction')
            assert len(out['cases']) == 547
            groups = {'Overall': list(out['cases'].values()),
                      'Cl': [v for v in out['cases'].values() if v['affected']],
                      'No_Cl': [v for v in out['cases'].values() if not v['affected']],
                      'Charged': [v for v in out['cases'].values() if v['charged']],
                      'Neutral': [v for v in out['cases'].values() if not v['charged']]}
            out['metrics'] = {}
            for name, rows in groups.items():
                out['metrics'][name] = dict(n=len(rows),
                    before_MAE=sum(abs(v['public_prediction_kcal_mol']-v['reference_kcal_mol']) for v in rows)/len(rows),
                    after_MAE=sum(abs(v['corrected_prediction_kcal_mol']-v['reference_kcal_mol']) for v in rows)/len(rows))
        out['status'] = 'complete' if args.phase == 'full' else 'smoke_passed'
        save()
        print(json.dumps(dict(model=args.size, status=out['status'], metrics=out.get('metrics'),
                              peak_gpu_GiB=out['peak_gpu_allocated_GiB'], peak_host_GiB=out['peak_host_RSS_GiB'])), flush=True)
    except BaseException as exc:
        out['status'] = 'failed'
        out['error'] = repr(exc)
        save()
        raise
    finally:
        ext.compute_k_vectors_flat = native_grid


if __name__ == '__main__':
    main()
