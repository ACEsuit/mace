import os

import numpy as np
import pytest
import torch
import torch._dynamo as dynamo
from ase import build
from torch.testing import assert_close

from mace import data as mace_data
from mace.calculators.lammps_mliap_mace import LAMMPS_MLIAP_MACE
from mace.calculators.mace import MACECalculator
from mace.tools import compile as mace_compile
from tests.test_compile import create_mace


@pytest.mark.skipif(os.name == "nt", reason="Not supported on Windows")
def test_ase_padding_matches_unpadded():
    atoms = build.bulk("C", "diamond", a=3.567, cubic=True).repeat((2, 2, 2))

    calc_ref = MACECalculator(models=create_mace("cpu"), device="cpu")
    ref_batch = calc_ref._atoms_to_batch(atoms)
    calc_pad = MACECalculator(
        models=create_mace("cpu"),
        device="cpu",
        pad_num_atoms=len(atoms) + 4,
        pad_num_edges=int(ref_batch["edge_index"].shape[1]) + 64,
    )

    atoms_ref = atoms.copy()
    atoms_ref.calc = calc_ref
    energy_ref = atoms_ref.get_potential_energy()
    forces_ref = atoms_ref.get_forces()
    stress_ref = atoms_ref.get_stress()

    atoms_pad = atoms.copy()
    atoms_pad.calc = calc_pad
    energy_pad = atoms_pad.get_potential_energy()
    forces_pad = atoms_pad.get_forces()
    stress_pad = atoms_pad.get_stress()

    assert_close(
        torch.tensor(energy_ref), torch.tensor(energy_pad), rtol=1e-6, atol=1e-6
    )
    assert_close(
        torch.tensor(forces_ref), torch.tensor(forces_pad), rtol=1e-5, atol=1e-5
    )
    assert_close(
        torch.tensor(stress_ref), torch.tensor(stress_pad), rtol=1e-5, atol=1e-5
    )


@pytest.mark.skipif(os.name == "nt", reason="Not supported on Windows")
def test_ase_padding_fixed_batch_shape():
    atoms_small = build.bulk("C", "diamond", a=3.567, cubic=True).repeat((1, 1, 1))
    atoms_large = build.bulk("C", "diamond", a=3.567, cubic=True).repeat((2, 2, 1))

    probe_calc = MACECalculator(models=create_mace("cpu"), device="cpu")
    large_batch = probe_calc._atoms_to_batch(atoms_large)
    target_atoms = len(atoms_large) + 4
    target_edges = int(large_batch["edge_index"].shape[1]) + 32

    calc_pad = MACECalculator(
        models=create_mace("cpu"),
        device="cpu",
        pad_num_atoms=target_atoms,
        pad_num_edges=target_edges,
    )

    batch_small = calc_pad._atoms_to_batch(atoms_small)
    batch_large = calc_pad._atoms_to_batch(atoms_large)

    assert int(batch_small["positions"].shape[0]) == target_atoms
    assert int(batch_large["positions"].shape[0]) == target_atoms
    assert int(batch_small["edge_index"].shape[1]) == target_edges
    assert int(batch_large["edge_index"].shape[1]) == target_edges
    assert int(batch_small["ptr"].numel() - 1) == 2
    assert int(batch_large["ptr"].numel() - 1) == 2


@pytest.mark.skipif(os.name == "nt", reason="Not supported on Windows")
def test_ase_padding_pads_unknown_node_and_edge_fields():
    atoms = build.bulk("C", "diamond", a=3.567, cubic=True)
    calc = MACECalculator(models=create_mace("cpu"), device="cpu")
    keyspec = mace_data.KeySpecification(
        info_keys=calc.info_keys, arrays_keys=calc.arrays_keys
    )
    config = mace_data.config_from_atoms(
        atoms, key_specification=keyspec, head_name=calc.head
    )
    real_graph = mace_data.AtomicData.from_config(
        config,
        z_table=calc.z_table,
        cutoff=calc.r_max,
        heads=calc.available_heads,
    )

    real_num_atoms = int(real_graph["node_attrs"].shape[0])
    real_num_edges = int(real_graph["edge_index"].shape[1])
    real_graph["custom_node_scalar"] = torch.ones(real_num_atoms, 1)
    real_graph["custom_edge_vec"] = torch.ones(real_num_edges, 2)

    fake_graph = mace_data.build_fake_padding_graph(
        real_graph, num_atoms=3, num_edges=7, r_max=calc.r_max
    )

    assert fake_graph["custom_node_scalar"].shape == (3, 1)
    assert fake_graph["custom_edge_vec"].shape == (7, 2)
    assert_close(fake_graph["custom_node_scalar"], torch.zeros(3, 1))
    assert_close(fake_graph["custom_edge_vec"], torch.zeros(7, 2))


class DummyLAMMPSData:
    def __init__(self, nlocal=8, nghost=2, npairs=20):
        self.nlocal = nlocal
        self.ntotal = nlocal + nghost
        self.npairs = npairs
        self.elems = np.zeros((nlocal,), dtype=np.int64)
        self.rij = np.random.randn(npairs, 3).astype(np.float32)
        self.pair_i = np.random.randint(0, nlocal, size=(npairs,), dtype=np.int64)
        self.pair_j = np.random.randint(0, nlocal, size=(npairs,), dtype=np.int64)
        self.eatoms = np.zeros((nlocal,), dtype=np.float64)
        self.energy = 0.0
        self.pair_forces_written = None

    def forward_exchange(self, feats, out, vec_len):
        out.copy_(feats)

    def reverse_exchange(self, grad, gout, vec_len):
        gout.copy_(grad)

    def update_pair_forces_gpu(self, pair_forces):
        self.pair_forces_written = pair_forces.detach().clone()


@pytest.mark.skipif(os.name == "nt", reason="Not supported on Windows")
def test_mliap_padding_shapes_and_graph_breaks():
    model = mace_compile.prepare(create_mace)("cpu")
    mliap = LAMMPS_MLIAP_MACE(model, pad_num_atoms=12, pad_num_pairs=40)
    data = DummyLAMMPSData()
    species = torch.as_tensor(data.elems, dtype=torch.int64)
    batch = mliap._prepare_batch(
        data, data.nlocal, data.ntotal - data.nlocal, data.npairs, species
    )

    assert batch["node_attrs"].shape[0] == 12
    assert batch["edge_index"].shape[1] == 40
    assert batch["vectors"].shape[0] == 40
    assert batch["natoms"] == (12, 2)

    explanation = dynamo.explain(mliap.model)(batch)
    assert explanation.graph_break_count == 0

    mliap.compute_forces(data)
    assert data.pair_forces_written is not None
    assert data.pair_forces_written.shape == (data.npairs, 3)


@pytest.mark.skipif(os.name == "nt", reason="Not supported on Windows")
def test_mliap_pair_padding_fallback_without_fake_atoms():
    model = mace_compile.prepare(create_mace)("cpu")
    data = DummyLAMMPSData(nlocal=8, nghost=2, npairs=20)
    mliap = LAMMPS_MLIAP_MACE(model, pad_num_atoms=8, pad_num_pairs=40)
    species = torch.as_tensor(data.elems, dtype=torch.int64)
    batch = mliap._prepare_batch(
        data, data.nlocal, data.ntotal - data.nlocal, data.npairs, species
    )

    assert batch["node_attrs"].shape[0] == data.nlocal
    assert batch["edge_index"].shape[1] == data.npairs
    assert batch["vectors"].shape[0] == data.npairs
    assert batch["natoms"] == (data.nlocal, data.ntotal - data.nlocal)


@pytest.mark.skipif(os.name == "nt", reason="Not supported on Windows")
def test_mliap_no_padding_keeps_single_graph_behavior():
    model = mace_compile.prepare(create_mace)("cpu")
    data = DummyLAMMPSData(nlocal=8, nghost=2, npairs=20)
    mliap = LAMMPS_MLIAP_MACE(model, pad_num_atoms=0, pad_num_pairs=0)
    species = torch.as_tensor(data.elems, dtype=torch.int64)
    batch = mliap._prepare_batch(
        data, data.nlocal, data.ntotal - data.nlocal, data.npairs, species
    )

    assert batch["lammps_class"] is data
    assert batch["natoms"] == (data.nlocal, data.ntotal - data.nlocal)
    assert int(batch["batch"].max()) == 0

    mliap.model(batch)
    assert batch["head"].shape == (1,)
    assert batch["total_charge"].shape == (1,)
    assert batch["total_spin"].shape == (1,)
