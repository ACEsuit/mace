"""The magnetic projection before derivatives, against the frozen wrapper."""

import numpy as np
import pytest
import torch
from ase import Atoms
from mace_core.config.precision import PrecisionConfig
from mace_torch.deploy.loader import load_deployed

from tests.golden.harness import CLOSED_FORM_FP64, FP32, FP64_ACCELERATED_BACKEND
from tests.parity.magnetic_convert import write_magnetic_checkpoint
from tests.parity.test_magnetic_parity import magnetic_surfaces, v1_graph

pytestmark = pytest.mark.magnetic


def close(actual, expected, dtype, device):
    row = FP32 if dtype == torch.float32 else (
        FP64_ACCELERATED_BACKEND if device == "cuda" else CLOSED_FORM_FP64)
    torch.testing.assert_close(actual, expected, atol=row.atol, rtol=row.rtol)


@pytest.fixture
def checkpoint(tmp_path, fp64):
    legacy = magnetic_surfaces().load_anchor()
    path = write_magnetic_checkpoint(legacy, tmp_path, tmp_path / 'unused.xyz')
    return legacy, path


def structure(name):
    if name == "crystal":
        atoms = Atoms("Fe2", scaled_positions=[[0, 0, 0], [0.5, 0.5, 0.5]],
                      cell=np.eye(3) * 2.87 + np.array([[0, 0.05, 0]] * 3), pbc=True)
        atoms.arrays["REF_magmom"] = np.array([[0.2, 0.1, 2.2], [-0.1, 0.3, -2.1]])
        return atoms
    return magnetic_surfaces().magnetic_fixtures()[name]


def moved(graph, dtype, device):
    return {name: value.to(device=device, dtype=dtype if value.is_floating_point() else value.dtype)
            if isinstance(value, torch.Tensor) else value for name, value in graph.items()}


@pytest.mark.parametrize("device", ["cpu", pytest.param("cuda", marks=[pytest.mark.gpu, pytest.mark.cueq])])
@pytest.mark.parametrize("dtype", [torch.float64, torch.float32])
@pytest.mark.parametrize("batched", [False, True])
@pytest.mark.parametrize("name", ["mag_fe3_canted", "mag_fe_atom", "mag_fe_dimer_afm", "crystal"])
def test_projection_matches_legacy_values_and_derivatives(checkpoint, name, batched, dtype, device):
    from mace.modules import TimeReversalSymmetrizedMACE
    from tests.parity.test_accelerated_backend_goldens import on_backend

    legacy, path = checkpoint
    # Legacy creates intermediate tensors in the process default dtype.
    # The fp64 fixture restores it after this case; both models use dtype here.
    torch.set_default_dtype(dtype)
    if device == 'cuda':
        path = on_backend(path, 'cueq')
    native = load_deployed(path, device=device).with_precision(
        PrecisionConfig(model=str(dtype).split('.')[-1]), device=device)
    projected = native.with_time_reversal(batched=batched)
    called = set()
    hooks = [module.register_forward_hook(lambda module, args, output: called.add(type(module).__name__))
             for module in projected.engine.modules() if type(module).__name__.startswith('CuEq')]
    assert next(projected.engine.parameters()).dtype == dtype
    assert native.config.model.magnetic.time_reversal == 'none'
    legacy = TimeReversalSymmetrizedMACE(legacy.to(device=device, dtype=dtype), batched=batched)
    atoms = structure(name)
    old_graph = moved(magnetic_surfaces().build_batch(legacy, atoms), dtype, device)
    expected = legacy(old_graph, training=True, compute_force=True,
                      compute_magforces=True, compute_stress=True, compute_virials=True)
    graph = moved(v1_graph(atoms, list(native.z_table.zs), native.r_max,
                          atoms.arrays['REF_magmom']), dtype, device)
    wanted = ('forces', 'magforces', 'stress', 'virials')
    result = projected.engine(graph, compute=wanted, training=True)
    for hook in hooks:
        hook.remove()
    if device == 'cuda':
        assert {'CuEqLinear', 'CuEqChannelwiseTPConv', 'CuEqSymmetricContraction', 'CuEqFullyConnectedTP'} <= called
    for actual, key in ((result.total_energy, 'energy'), (result.node_energies, 'node_energy'),
                        (result.forces, 'forces'), (result.stress, 'stress'), (result.virials, 'virials'),
                        (result.extras['magforces'], 'magforces')):
        close(actual, expected[key], dtype, device)
    reversed_result = projected.compute({**graph, 'magmom': -graph['magmom']}, compute=wanted)
    close(result.total_energy, reversed_result.total_energy, dtype, device)
    close(result.forces, reversed_result.forces, dtype, device)
    close(result.extras['magforces'], -reversed_result.extras['magforces'], dtype, device)
    assert not graph['magmom'].requires_grad and not graph['positions'].requires_grad


@pytest.mark.parametrize('batched', [False, True])
def test_projected_hessian_and_edge_forces_match_legacy(checkpoint, batched):
    from mace.modules import TimeReversalSymmetrizedMACE

    legacy, path = checkpoint
    atoms = structure('mag_fe3_canted')
    native = load_deployed(path).with_time_reversal(batched=batched)
    old = TimeReversalSymmetrizedMACE(legacy, batched=batched)
    expected = old(magnetic_surfaces().build_batch(legacy, atoms), training=True,
                   compute_force=True, compute_hessian=True, compute_edge_forces=True)
    graph = v1_graph(atoms, list(native.z_table.zs), native.r_max, atoms.arrays['REF_magmom'])
    result = native.compute(graph, compute=('forces', 'hessian', 'edge_forces', 'atomic_stresses', 'atomic_virials'))
    close(result.extras['hessian'], expected['hessian'], torch.float64, 'cpu')
    # V1 defines edge forces as -dE/dv; frozen legacy reports +dE/dv.
    close(result.extras['edge_forces'], -expected['edge_forces'], torch.float64, 'cpu')
    assert result.extras['hessian'].shape == (3 * len(atoms), len(atoms), 3)


@pytest.mark.parametrize('batched', [False, True])
@pytest.mark.parametrize('device', ['cpu', pytest.param('cuda', marks=[pytest.mark.gpu, pytest.mark.cueq])])
def test_force_loss_parameter_gradients_match_legacy(checkpoint, batched, device):
    from mace.modules import TimeReversalSymmetrizedMACE

    legacy, path = checkpoint
    if device == 'cuda':
        from tests.parity.test_accelerated_backend_goldens import on_backend
        path = on_backend(path, 'cueq')
    legacy = legacy.to(device)
    atoms = structure('mag_fe3_canted')
    native = load_deployed(path, device=device).with_time_reversal(batched=batched)
    old = TimeReversalSymmetrizedMACE(legacy, batched=batched)
    expected = old(moved(magnetic_surfaces().build_batch(legacy, atoms), torch.float64, device), training=True,
                   compute_force=True, compute_magforces=True)
    graph = moved(v1_graph(atoms, list(native.z_table.zs), native.r_max, atoms.arrays['REF_magmom']), torch.float64, device)
    result = native.engine(graph, compute=('forces', 'magforces'), training=True)
    (result.forces.square().sum() + result.extras['magforces'].square().sum()).backward()
    (expected['forces'].square().sum() + expected['magforces'].square().sum()).backward()
    for source, block in zip(legacy.interactions, native.model.backbone.interactions, strict=True):
        for old_mlp, new_mlp in ((source.conv_tp_weights, block.edge_weights),
                                 (source.conv_tp_weights_magmom, block.moment_weights),
                                 (source.density_fn, block.density)):
            for index, weight in enumerate(new_mlp.weights):
                close(weight.grad, getattr(old_mlp, f'layer{index}').weight.grad, torch.float64, device)


@pytest.mark.parametrize('batched', [False, True])
@pytest.mark.parametrize('name', ['mag_fe_atom', 'mag_fe_dimer_afm', 'mag_fe_dimer_fm'])
def test_projected_fixed_point_matches_legacy(checkpoint, batched, name):
    from mace.modules import TimeReversalSymmetrizedMACE
    from mace.modules.extensions import MagneticSCFMACE
    from mace_core.config import FixedPointSpec
    from mace_torch.physics import FixedPointDriver
    from tests.golden.harness import FP64_CPU_REFERENCE

    legacy, path = checkpoint
    atoms = structure(name)
    old = MagneticSCFMACE(TimeReversalSymmetrizedMACE(legacy, batched=batched),
                         n_scf_step=200, scf_tol=1e-10)
    expected = old(magnetic_surfaces().build_batch(legacy, atoms), compute_force=True)
    native = load_deployed(path).with_time_reversal(batched=batched)
    driver = FixedPointDriver(native.engine, FixedPointSpec(variable='magmom', max_iter=200,
                             tolerance=1e-10, require_convergence=False))
    graph = v1_graph(atoms, list(native.z_table.zs), native.r_max, atoms.arrays['REF_magmom'])
    result = driver(graph, compute=('forces', 'magforces'))
    for actual, target in ((result.total_energy, expected['energy']),
                           (result.forces, expected['forces']),
                           (result.extras['converged_magmom'], expected['equilibrated_magmom'])):
        torch.testing.assert_close(actual, target, atol=FP64_CPU_REFERENCE.atol, rtol=FP64_CPU_REFERENCE.rtol)
