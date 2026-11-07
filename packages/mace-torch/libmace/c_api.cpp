// The cmace_* signatures and column-major Fortran mapping follow libmace
// e3413ae658530654894b6bac6a62a764a74c5115/src/mace_c_api.cpp and mace.f90.
#include "mace.h"
#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <stdexcept>

namespace {
// The established C/Fortran ABI has no error result. Fail explicitly instead
// of unwinding a C++ exception through a foreign-language caller.
template <class Function> auto checked(Function action) -> decltype(action()) {
  try {
    return action();
  } catch (const std::exception &error) {
    std::fprintf(stderr, "libmace: %s\n", error.what());
    std::abort();
  }
}
bool boolean(int value) {
  if (value != 0 && value != 1)
    throw std::invalid_argument("boolean flags must be 0 or 1");
  return value != 0;
}
MACE &model(CMace *value) {
  if (!value)
    throw std::invalid_argument("null libmace model");
  return *value;
}
} // namespace

CMace *cmace_init(const char *path) {
  return checked([&] {
    if (!path)
      throw std::invalid_argument("null native.pt2 path");
    return new MACE(path);
  });
}
void cmace_finalise(CMace *self) { delete self; }
void cmace_reload(CMace *self) {
  checked([&] { model(self).reload(); });
}
void cmace_print(CMace *self) {
  checked([&] { model(self).print(); });
}
void cmace_calculate(CMace *self, int calc_virial, int n_atoms,
                     const double *cell, const int *pbc,
                     const int *atomic_numbers, const double *positions,
                     double *total_energy, double *node_energy, double *forces,
                     double *virial) {
  checked([&] {
    if (n_atoms <= 0 || !cell || !pbc || !atomic_numbers || !positions ||
        !total_energy || !node_energy || !forces || !virial)
      throw std::invalid_argument(
          "libmace requires nonempty input/output arrays");
    std::array<std::array<double, 3>, 3> lattice;
    std::array<bool, 3> periodic;
    for (int axis = 0; axis < 3; ++axis) {
      std::copy_n(cell + 3 * axis, 3, lattice[axis].begin());
      periodic[axis] = boolean(pbc[axis]);
    }
    std::vector<std::array<double, 3>> points(n_atoms);
    for (int atom = 0; atom < n_atoms; ++atom)
      std::copy_n(positions + 3 * atom, 3, points[atom].begin());
    const auto result = model(self).calculate(
        lattice, periodic, points,
        std::vector<int>(atomic_numbers, atomic_numbers + n_atoms),
        boolean(calc_virial));
    *total_energy = result.total_energy;
    std::copy(result.node_energy.begin(), result.node_energy.end(),
              node_energy);
    std::copy(result.virial.begin(), result.virial.end(), virial);
    for (int atom = 0; atom < n_atoms; ++atom)
      std::copy(result.forces[atom].begin(), result.forces[atom].end(),
                forces + 3 * atom);
  });
}
