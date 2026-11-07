// Source-compatible C++ and C entry points from stenczelt/libmace, revision
// e3413ae658530654894b6bac6a62a764a74c5115/include/mace.h.
// The implementation consumes V1's native.pt2. No framework type crosses the
// ABI.
#ifndef MACE_V1_LIBMACE_H
#define MACE_V1_LIBMACE_H

#ifdef __cplusplus
#include <array>
#include <memory>
#include <string>
#include <vector>

struct CalculationResult {
  double total_energy = 0;
  std::vector<double> node_energy;
  std::vector<std::array<double, 3>> forces;
  // eV, in libmace order xx, yy, zz, xy, xz, yz; virial = -V * stress.
  std::array<double, 6> virial{};
};

class MACE {
public:
  explicit MACE(const std::string &model_path);
  void reload();
  void print() const;
  CalculationResult
  calculate(const std::array<std::array<double, 3>, 3> &cell,
            const std::array<bool, 3> &pbc,
            const std::vector<std::array<double, 3>> &positions,
            const std::vector<int> &atomic_numbers, const bool &calc_virial);

private:
  struct Runtime;
  std::string model_path;
  std::shared_ptr<Runtime> runtime;
};

extern "C" {
typedef MACE CMace;
#else
typedef struct CMace CMace;
#endif

CMace *cmace_init(const char *model_path);
void cmace_finalise(CMace *self);
void cmace_reload(CMace *self);
void cmace_print(CMace *self);
// C: cell rows and [n_atoms,3] positions/forces, contiguous doubles in Å.
// Fortran: cell columns and (3,n_atoms), as in the unmodified upstream module.
// Energies are eV, forces eV/Å. A disabled virial is returned as six zeros.
void cmace_calculate(CMace *self, int calc_virial, int n_atoms,
                     const double *cell, const int *pbc,
                     const int *atomic_numbers, const double *positions,
                     double *total_energy, double *node_energy, double *forces,
                     double *virial);

#ifdef __cplusplus
}
#endif
#endif
