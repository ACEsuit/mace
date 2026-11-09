"""The ML-IAP callback ABI, with owned atoms, changing edges and a halo adjoint."""

from dataclasses import replace
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from conftest import assert_close
from mace_core.edges import plan_edges
from mace_core.graph.padding import FixedPadding
from mace_torch.calculators import MACECalculator
from mace_torch.deploy.bundle import export_lammps
from mace_torch.deploy.mliap import MACEMLIAPUnified
from test_mace_torch_calculator import water
from test_mace_torch_cli_commands import artifact as artifact


class PlainData(SimpleNamespace):
    @property
    def eatoms(self):
        raise AttributeError("write-only eatoms")

    @eatoms.setter
    def eatoms(self, value):
        self.site = value.copy()

    def update_pair_forces(self, value):
        assert value.dtype == np.float64 and value.flags.c_contiguous
        self.pair_forces = value.copy()


class HaloData(PlainData):
    def forward_exchange(self, send, receive, width):
        assert send.is_contiguous() and send.shape == (self.ntotal, width)
        receive.copy_(send[torch.arange(self.ntotal) % self.nlocal])
        self.forward_calls += 1

    def reverse_exchange(self, send, receive, width):
        receive.zero_()
        receive[: self.nlocal] = send[: self.nlocal] + send[self.nlocal :]
        self.reverse_calls += 1


def data_for(calculator, atoms, halo=False):
    graph, _ = calculator._graph(atoms, padded=False)
    order = torch.arange(graph["edge_index"].shape[1] - 1, -1, -1)
    sender, receiver = graph["edge_index"][:, order]
    vectors = (
        graph["positions"][receiver]
        - graph["positions"][sender]
        + graph["shifts"][order]
    )
    elements = graph["node_attrs"].argmax(-1).cpu().numpy()
    cls = HaloData if halo else PlainData
    return cls(
        nlocal=len(atoms),
        ntotal=len(atoms) * (2 if halo else 1),
        nlistatoms=len(atoms),
        natomneigh=len(atoms),
        npairs=len(sender),
        elems=np.tile(elements, 2 if halo else 1),
        pair_i=receiver.cpu().numpy(),
        pair_j=sender.cpu().numpy() + (len(atoms) if halo else 0),
        rij=-vectors.cpu().numpy(),
        iatoms=np.arange(len(atoms))[::-1].copy(),
        forward_calls=0,
        reverse_calls=0,
        pair_forces=np.zeros((len(sender), 3)),
    )


def atomic_forces(data):
    result = np.zeros((data.nlocal, 3))
    np.add.at(result, data.pair_i % data.nlocal, data.pair_forces)
    np.add.at(result, data.pair_j % data.nlocal, -data.pair_forces)
    return result


@pytest.mark.parametrize("halo", [False, True])
@pytest.mark.parametrize("padded", [False, True])
def test_callback_matches_calculator_after_neighbor_changes(artifact, halo, padded):
    bundle = export_lammps(
        artifact,
        halo_exchange=halo,
        compile_regions=False,
        dtype="float64" if torch.get_default_dtype() == torch.float64 else "float32",
        pad_spec=FixedPadding(nodes_budget=16, edges_budget=64, graphs_budget=2)
        if padded
        else None,
    )
    adapter = MACEMLIAPUnified(bundle)
    calculator = MACECalculator(bundle)
    for distance in (1.0, 1.3, 12.0):
        atoms = water()
        atoms.positions[0, 0] += distance
        calculator.calculate(atoms)
        data = data_for(calculator, atoms, halo)
        # Force an order-sensitive backend contract, including stable runs.
        if adapter.deployed is None:
            adapter._setup(data, torch.device("cpu"))
        adapter.edge_plan = plan_edges({"sender_runs", "receiver_runs"}, ())
        identity = adapter.deployed
        with torch.no_grad():
            adapter.compute_forces(data)
        assert adapter.deployed is identity and adapter.data is None
        assert_close(data.energy, calculator.results["energy"])
        assert_close(data.site[::-1], calculator.results["energies"])
        assert_close(atomic_forces(data), calculator.results["forces"])
        assert data.forward_calls == data.reverse_calls == (1 if halo else 0)
        assert not any(p.requires_grad for p in adapter.engine.parameters())


def test_missing_exchange_and_incomplete_ghost_shell_fail_before_setup(artifact):
    atoms = water()
    calc = MACECalculator(artifact)
    data = data_for(calc, atoms)
    halo = MACEMLIAPUnified(export_lammps(artifact, halo_exchange=True))
    with pytest.raises(RuntimeError, match=r"PKG_KOKKOS=yes.*single-layer"):
        halo.compute_forces(data)
    assert halo.deployed is None
    halo.manifest = halo.manifest.__class__(
        **{**halo.manifest.__dict__, "halo_exchange": False}
    )
    data.ntotal += 1
    with pytest.raises(RuntimeError, match="EXISTS 1"):
        halo.compute_forces(data)


@pytest.mark.parametrize("override", [None, "MACE_ALLOW_CPU", "MACE_FORCE_CPU"])
def test_cpu_guard_and_environment_controls(artifact, monkeypatch, caplog, override):
    monkeypatch.setenv("MACE_TIME", "true")
    monkeypatch.setenv("MACE_PROFILE", "yes")
    monkeypatch.setenv("MACE_PROFILE_START", "1")
    monkeypatch.setenv("MACE_PROFILE_END", "2")
    if override:
        monkeypatch.setenv(override, "t")
    adapter = MACEMLIAPUnified(
        export_lammps(artifact, halo_exchange=False, compile_regions=False)
    )
    data = data_for(MACECalculator(artifact), water())
    data.__class__ = type("KokkosData", (PlainData,), {"__module__": "kokkos"})
    if override is None:
        with pytest.raises(ValueError, match=r"GPU requested.*MACE_ALLOW_CPU"):
            adapter.compute_forces(data)
        assert adapter.deployed is None
    else:
        with caplog.at_level("INFO"):
            adapter.compute_forces(data)
        assert "Timer - ML-IAP step" in caplog.text
        with pytest.raises(SystemExit) as stop:
            adapter.compute_forces(data)
        assert stop.value.code == 0


def test_compiled_regions_keep_exchange_outside_capture(artifact, monkeypatch):
    compile_model = torch.compile
    calls = []

    def compile_region(forward, **kwargs):
        calls.append(forward)
        return compile_model(forward, backend="aot_eager", **kwargs)

    monkeypatch.setattr(torch, "compile", compile_region)
    bundle = export_lammps(artifact, halo_exchange=True, compile_regions=True)
    adapter = MACEMLIAPUnified(bundle)
    calculator = MACECalculator(bundle)
    for factor in (1.0, 1.2):
        atoms = water()
        atoms.positions *= factor
        calculator.calculate(atoms)
        data = data_for(calculator, atoms, halo=True)
        adapter.compute_forces(data)
        assert_close(atomic_forces(data), calculator.results["forces"])
        assert data.forward_calls == data.reverse_calls == 1
    assert len(calls) == 4
    assert all(
        layer.forward.compiled is not None
        for layer in (
            *adapter.deployed.model.backbone.interactions,
            *adapter.deployed.model.backbone.products,
        )
    )


def test_isolated_atoms_and_empty_domain(artifact):
    bundle = export_lammps(artifact, halo_exchange=False, compile_regions=False)
    adapter = MACEMLIAPUnified(bundle)
    calc = MACECalculator(bundle)
    atoms = water()
    atoms.positions *= 100
    calc.calculate(atoms)
    data = data_for(calc, atoms)
    adapter.compute_forces(data)
    assert data.npairs == 0
    assert_close(data.energy, calc.results["energy"])
    assert_close(data.site[::-1], calc.results["energies"])
    data.nlocal = data.nlistatoms = data.natomneigh = 0
    data.iatoms = None
    del data.site
    # Retain the ghost shell on an empty owner rank.
    data.natomneigh = data.ntotal
    adapter.compute_forces(data)
    assert data.energy == 0 and not hasattr(data, "site")
    data.ntotal = data.natomneigh = 0
    data.elems = np.empty(0, dtype=int)
    adapter.compute_forces(data)
    assert data.energy == 0 and not hasattr(data, "site")


def test_inference_accepts_a_first_order_backend(artifact, monkeypatch):
    from mace_torch.backends.reference import ReferenceBackend

    capabilities = ReferenceBackend.capabilities
    monkeypatch.setattr(
        ReferenceBackend,
        "capabilities",
        lambda backend: replace(capabilities(backend), supports_double_backward=False),
    )
    bundle = export_lammps(artifact, halo_exchange=False, compile_regions=False)
    adapter = MACEMLIAPUnified(bundle)
    calc = MACECalculator(bundle)
    atoms = water()
    calc.calculate(atoms)
    data = data_for(calc, atoms)
    adapter.compute_forces(data)
    assert_close(atomic_forces(data), calc.results["forces"])


def test_compiler_failure_falls_back_once(artifact, monkeypatch, caplog):
    from torch._dynamo.exc import Unsupported

    attempts = []

    def unavailable(forward, **kwargs):
        def fail(*args, **kwargs):
            attempts.append(forward)
            raise Unsupported("test compiler unavailable")

        return fail

    monkeypatch.setattr(torch, "compile", unavailable)
    adapter = MACEMLIAPUnified(export_lammps(artifact, halo_exchange=False))
    calc = MACECalculator(artifact)
    atoms = water()
    calc.calculate(atoms)
    for _ in range(2):
        data = data_for(calc, atoms)
        adapter.compute_forces(data)
        assert_close(atomic_forces(data), calc.results["forces"])
    assert len(attempts) == 4
    assert "test compiler unavailable" in caplog.text


def test_exported_head_reaches_mliap_graph(tmp_path):
    from mace_core.config.resolved import ResolvedConfig
    from mace_core.observables import DEFAULT_CATALOGUE
    from mace_torch.train import run_data_stage, run_model_stage, write_model
    from test_mace_torch_full_batch import configuration

    doc = configuration(tmp_path, max_num_epochs=1).to_resolved_dict()
    doc["data"]["heads"] = dict.fromkeys(
        ("pbe", "scan"), doc["data"]["heads"]["default"]
    )
    config = ResolvedConfig.model_validate(doc)
    data = run_data_stage(config, DEFAULT_CATALOGUE)
    built = run_model_stage(config, data, DEFAULT_CATALOGUE)
    checkpoint = write_model(tmp_path / "multi", built.engine, built.metadata)
    bundle = export_lammps(
        checkpoint, halo_exchange=False, head="scan", compile_regions=False
    )
    adapter = MACEMLIAPUnified(bundle)
    calc = MACECalculator(checkpoint, head="scan")
    atoms = water()
    calc.calculate(atoms)
    data = data_for(calc, atoms)
    adapter.compute_forces(data)
    assert adapter.head == 1
    assert_close(data.energy, calc.results["energy"])
    assert_close(atomic_forces(data), calc.results["forces"])
