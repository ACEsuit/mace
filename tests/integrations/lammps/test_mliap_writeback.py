"""Contract tier: the ML-IAP energy/force writeback, on both couplings.

LAMMPS ships two ML-IAP Python couplings with *different* writeback APIs, and
the difference is invisible until the model has already run:

* ``src/KOKKOS/mliap_unified_couple_kokkos.pyx`` — ``eatoms`` has a getter
  (a view sized by ``nlistatoms``, copied into in place) and there is an
  ``update_pair_forces_gpu`` that takes a device pointer.
* ``src/ML-IAP/mliap_unified_couple.pyx`` — ``eatoms`` is a
  ``write_only_property`` (``property(fget=None, fset=...)``) and only
  ``update_pair_forces`` exists, taking a contiguous host ``double[:, ::1]``.

conda-forge builds every CPU ``lammps`` with ``PKG_KOKKOS=OFF``, so the plain
coupling is what the nightly real tier actually gets — and reading ``eatoms``
there raises ``property 'eatoms' of 'MLIAPDataPy' object has no getter``.
The stubs below mirror both property shapes so the two branches stay covered
without a LAMMPS binary; the real tier can only ever exercise one of them.
"""

import numpy as np
import pytest
import torch

from mace.calculators.lammps_mliap_mace import LAMMPS_MLIAP_MACE
from tests.integrations.lammps._harness import StubMACE


class _PlainData:
    """The non-KOKKOS MLIAPDataPy: write-only eatoms/energy, host forces."""

    def __init__(self, natoms, nlistatoms=None):
        self.nlocal = natoms
        self.nlistatoms = natoms if nlistatoms is None else nlistatoms
        self.eatoms_written = None
        self.energy_written = None
        self.forces_written = None

    def _set_eatoms(self, value):
        # Cython's `double[:] value_view = value` accepts only a real float64
        # buffer -- a torch tensor or a float32 array raises there, so reject
        # them here too instead of quietly recording something LAMMPS would
        # not have taken.
        assert isinstance(value, np.ndarray), f"eatoms got {type(value)}"
        assert value.dtype == np.float64, value.dtype
        assert value.shape == (self.nlistatoms,), value.shape
        self.eatoms_written = value.copy()

    def _set_energy(self, value):
        assert isinstance(value, float), f"energy got {type(value)}"
        self.energy_written = value

    eatoms = property(fget=None, fset=_set_eatoms)
    energy = property(fget=None, fset=_set_energy)

    def update_pair_forces(self, fij):
        assert isinstance(fij, np.ndarray), f"forces got {type(fij)}"
        assert fij.dtype == np.float64, fij.dtype
        assert fij.flags["C_CONTIGUOUS"], "update_pair_forces needs double[:, ::1]"
        self.forces_written = fij.copy()


class _KokkosData:
    """The KOKKOS MLIAPDataPy: an eatoms view to copy into, forces by pointer."""

    def __init__(self, natoms):
        self.nlocal = natoms
        self.nlistatoms = natoms
        self.eatoms = torch.zeros(natoms, dtype=torch.float64)
        self.energy = None
        self.forces_written = None

    def update_pair_forces_gpu(self, fij):
        assert isinstance(fij, torch.Tensor), f"forces got {type(fij)}"
        assert fij.dtype == torch.float64, fij.dtype
        self.forces_written = fij


def _writeback(data, unified, natoms=3, npairs=4):
    atom_energies = torch.arange(1.0, natoms + 1, dtype=unified.dtype)
    pair_forces = torch.arange(3.0 * npairs, dtype=unified.dtype).reshape(npairs, 3)
    unified._update_lammps_data(  # pylint: disable=protected-access
        data, atom_energies, pair_forces, natoms
    )
    return atom_energies.double(), pair_forces.double()


@pytest.mark.parametrize("dtype", [torch.float64, torch.float32])
def test_plain_coupling_writes_through_the_host_api(dtype):
    # The regression: this path used to read data.eatoms, which the non-KOKKOS
    # coupling does not expose, so the nightly real tier failed here with the
    # model already evaluated.
    unified = LAMMPS_MLIAP_MACE(StubMACE(1, dtype=dtype))
    data = _PlainData(natoms=3)

    energies, forces = _writeback(data, unified)

    assert np.allclose(data.eatoms_written, energies.numpy())
    assert data.energy_written == pytest.approx(energies.sum().item())
    assert np.allclose(data.forces_written, forces.numpy())


def test_kokkos_coupling_copies_into_the_eatoms_view():
    unified = LAMMPS_MLIAP_MACE(StubMACE(1))
    data = _KokkosData(natoms=3)
    view = data.eatoms

    energies, forces = _writeback(data, unified)

    # In place: LAMMPS owns that buffer, so rebinding the attribute instead of
    # copying would drop the energies on the floor.
    assert view.data_ptr() == data.eatoms.data_ptr()
    assert torch.allclose(view, energies)
    assert data.energy == pytest.approx(energies.sum().item())
    assert torch.allclose(data.forces_written, forces)


def test_float32_model_hands_lammps_float64():
    # LAMMPS reads both buffers as double regardless of the model's dtype.
    unified = LAMMPS_MLIAP_MACE(StubMACE(1, dtype=torch.float32))
    assert unified.dtype == torch.float32
    data = _KokkosData(natoms=3)

    _writeback(data, unified)

    assert data.forces_written.dtype == torch.float64


def test_atom_count_mismatch_is_actionable():
    # The write-only setter fills exactly nlistatoms doubles and Cython rejects
    # a length mismatch from inside the .pyx, naming neither count.
    unified = LAMMPS_MLIAP_MACE(StubMACE(1))
    data = _PlainData(natoms=3, nlistatoms=4)

    with pytest.raises(RuntimeError, match="nlistatoms"):
        _writeback(data, unified)
