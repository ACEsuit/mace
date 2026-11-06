"""DEP-2 native-runtime probe on the converted, frozen ScaleShift anchor."""

import json
import os
import re
import shlex
import shutil
import subprocess
import sys
from copy import deepcopy
from pathlib import Path

import numpy as np
import pytest
import torch
from ase.io import read
from torch import nn

from tests.golden.harness import FP64_CPU_REFERENCE

ROOT = Path(__file__).resolve().parents[3]
TINY = ROOT / "packages/mace-jax/tests/fixtures/tiny.json"
INPUTS = ("positions", "shifts", "edge_index", "node_attrs", "node_atomic_numbers")


class Energy(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, positions, shifts, edge_index, node_attrs, numbers):
        graph = dict(
            positions=positions,
            shifts=shifts,
            edge_index=edge_index,
            node_attrs=node_attrs,
            node_atomic_numbers=numbers,
            batch=torch.zeros_like(numbers),
            ptr=numbers.new_tensor([0, len(numbers)]),
            head=numbers.new_zeros(1),
        )
        return self.model(graph).total_energy.sum()


class EnergyForces(nn.Module):
    def __init__(self, model, route):
        super().__init__()
        self.energy = Energy(model)
        self.route = route

    def forward(self, positions, shifts, edge_index, node_attrs, numbers):
        args = (positions, shifts, edge_index, node_attrs, numbers)
        if self.route == "func":
            grads, energy = torch.func.grad_and_value(self.energy, argnums=(0, 1))(
                *args
            )
        else:
            energy = self.energy(*args)
            grads = torch.autograd.grad(energy, (positions, shifts))
        return energy.detach(), -grads[0].detach(), grads[1].detach()


@pytest.fixture(scope="module")
def cases(tmp_path_factory):
    """Move the atoms and change their ordering at equal input shapes."""
    from mace_torch.calculators import MACECalculator
    from mace_torch.deploy.bundle import export_lammps

    bundle = export_lammps(
        TINY,
        tmp_path_factory.mktemp("native") / "anchor.mace-deploy",
        halo_exchange=False,
        compile_regions=False,
    )
    calc = MACECalculator(bundle, device="cpu", default_dtype="float64")
    original = read(ROOT / "tests/golden/fixtures/water_cluster.xyz")
    displaced = original.copy()
    displaced.positions += np.sin(
        np.linspace(0, 0.006, original.positions.size)
    ).reshape(-1, 3)
    samples = []
    for atoms in (original, displaced, displaced[::-1]):
        graph, _ = calc._graph(atoms, padded=False)
        inputs = tuple(graph[key].contiguous() for key in INPUTS)
        calc.calculate(atoms)
        samples.append(
            (inputs, float(calc.results["energy"]), calc.results["forces"].copy())
        )
    frozen = json.loads(
        (
            ROOT / "tests/golden/references/tiny_scaleshift_e3nn_cpu_fp64.json"
        ).read_text()
    )["fixtures"]["water_cluster"]["outputs"]
    for value, key in zip(samples[0][1:], ("energy", "forces"), strict=True):
        np.testing.assert_allclose(
            value, frozen[key]["value"], atol=FP64_CPU_REFERENCE.atol, rtol=0
        )
    assert np.max(np.abs(samples[0][2] - samples[1][2])) > 1e-4
    assert not torch.equal(samples[0][0][4], samples[2][0][4])
    return calc.models[0].model.eval().requires_grad_(False), samples


def grad_inputs(args):
    return tuple(t.detach().requires_grad_(i < 2) for i, t in enumerate(args))


def check_outputs(out, sample):
    from torch._subclasses.fake_tensor import FakeTensor

    args, expected_energy, expected_forces = sample
    assert all(not isinstance(tensor, FakeTensor) for tensor in out), (
        "FakeTensor output"
    )
    energy, forces, edge_grad = (tensor.detach().cpu().numpy() for tensor in out)
    tol = FP64_CPU_REFERENCE.atol
    np.testing.assert_allclose(energy, expected_energy, atol=tol, rtol=0)
    np.testing.assert_allclose(forces, expected_forces, atol=tol, rtol=0)
    # The force on the sender is +dE/dv for v=r_receiver-r_sender+shift.
    reconstructed = np.zeros_like(forces)
    np.add.at(reconstructed, args[2][0].cpu(), edge_grad)
    np.add.at(reconstructed, args[2][1].cpu(), -edge_grad)
    np.testing.assert_allclose(reconstructed, expected_forces, atol=tol, rtol=0)


class DirectCaptureUnavailable(Exception):
    pass


@pytest.mark.parametrize("route", ["autograd", "func"])
@pytest.mark.xfail(
    raises=DirectCaptureUnavailable, strict=True, reason="DEP-2 T1 promotion probe"
)
def test_export_energy_forces(cases, route):
    model, samples = cases
    wrapped = EnergyForces(model, route)
    inputs = grad_inputs(samples[0][0])
    for sample in samples:
        check_outputs(EnergyForces(model, "autograd")(*grad_inputs(sample[0])), sample)
    try:
        with torch._dynamo.config.patch(trace_autograd_ops=True):
            ep = torch.export.export(wrapped, inputs, strict=True)
        with torch.no_grad():
            for sample in samples:
                check_outputs(ep.module()(*sample[0]), sample)
    except (RuntimeError, AssertionError, AttributeError) as error:
        known = (
            "FakeTensor output",
            "doesn't have storage",
            "setup_context",
            "autograd.grad consumed",
            "trace_autograd_ops",
            "functorch transform",
        )
        if not any(reason in str(error) for reason in known):
            raise
        raise DirectCaptureUnavailable(str(error)) from error


@pytest.mark.slow
@pytest.mark.parametrize("device", ["cpu", pytest.param("cuda", marks=pytest.mark.gpu)])
def test_precomputed_gradient_runs_in_native_cpp(cases, tmp_path, device):
    """Compile both derivatives at export time, then load them in standalone C++."""
    from mace_torch.kernels import ops
    from mace_torch.kernels.contraction import symmetric_contraction_grouped
    from torch.fx.experimental.proxy_tensor import make_fx
    from torch.utils.cpp_extension import include_paths, library_paths

    compiler = shlex.split(os.environ.get("CXX", "c++"))
    if not shutil.which(compiler[0]) or sys.platform not in {"darwin", "linux"}:
        pytest.skip("native probe requires a C++ compiler on Linux or macOS")
    model, samples = cases
    model = deepcopy(model).to(device)
    samples = [(tuple(x.to(device) for x in args), e, f) for args, e, f in samples]
    example = samples[0][0]
    joint = make_fx(EnergyForces(model, "autograd"))(*grad_inputs(example))
    # The reference custom operators are registered in Python. Inline their
    # existing ATen implementations at export time. _init_fn is experimental
    # and belongs only to this promotion probe, outside the runtime API.
    names = (
        "segment_sum",
        "channelwise_tp_conv",
        "equivariant_linear",
        "symmetric_contraction",
    )
    lowerings = {
        getattr(torch.ops.mace, name).default: getattr(ops, name)._init_fn
        for name in names
    }
    lowerings[torch.ops.mace.symmetric_contraction_grouped.default] = (
        symmetric_contraction_grouped._init_fn
    )
    aten = make_fx(joint, decomposition_table=lowerings)(*example)
    program = torch.export.export(aten, example, strict=True)
    assert not any(str(node.target).startswith("mace.") for node in program.graph.nodes)
    with torch.no_grad():
        for sample in samples:
            check_outputs(program.module()(*sample[0]), sample)
    package = tmp_path / "energy-forces.pt2"
    torch._inductor.aoti_compile_and_package(
        program, package_path=str(package), inductor_configs={"cpp.threads": 1}
    )
    exe = tmp_path / "native"
    command = [*compiler, "-std=c++20", "-O1"]
    command += [f"-I{path}" for path in include_paths()]
    for path in library_paths():
        command += [f"-L{path}", f"-Wl,-rpath,{path}"]
    libraries = ["-ltorch", "-ltorch_cpu", "-lc10"]
    if device == "cuda":
        libraries.extend(["-Wl,--no-as-needed", "-ltorch_cuda", "-lc10_cuda"])
    subprocess.run(
        [
            *command,
            str(Path(__file__).with_name("native_capture.cpp")),
            *libraries,
            "-o",
            str(exe),
        ],
        check=True,
    )
    geometries = tmp_path / "samples.bin"
    with geometries.open("wb") as stream:
        for args, energy, forces in samples:
            np.array(
                [len(args[0]), len(args[1]), args[3].shape[1]], dtype=np.int64
            ).tofile(stream)
            for tensor in args:
                tensor.detach().cpu().numpy().tofile(stream)
            np.asarray(energy, dtype=np.float64).tofile(stream)
            forces.tofile(stream)
    env = os.environ.copy()
    for name in ("PYTHONPATH", "PYTHONHOME"):
        env.pop(name, None)
    env["DYLD_PRINT_LIBRARIES" if sys.platform == "darwin" else "LD_DEBUG"] = (
        "1" if sys.platform == "darwin" else "libs"
    )
    result = subprocess.run(
        [
            str(exe),
            str(package),
            str(geometries),
            str(len(samples)),
            str(FP64_CPU_REFERENCE.atol),
            device,
        ],
        env=env,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stdout + result.stderr[-4000:]
    assert not re.search(
        r"/(?:libpython[0-9]|libtorch_python|Python\.framework/.*/Python\b)",
        result.stderr,
    )
    print(result.stdout)
