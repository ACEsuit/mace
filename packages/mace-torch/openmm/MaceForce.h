// OpenMM's public Force/CustomCPPForceImpl interface, with V1 native.pt2.
#ifndef MACE_OPENMM_FORCE_H
#define MACE_OPENMM_FORCE_H
#include <openmm/Force.h>
#include <string>
#include <vector>

namespace mace {
class MaceForceImpl;
class MaceForce final : public OpenMM::Force {
public:
  // Atomic numbers follow the System particle order. Geometry uses OpenMM's
  // nm; the native model keeps its canonical eV/Angstrom contract.
  MaceForce(std::string package, std::vector<int> atomicNumbers, bool periodic);
  bool usesPeriodicBoundaryConditions() const override { return periodic_; }

protected:
  OpenMM::ForceImpl *createImpl() const override;

private:
  friend class MaceForceImpl;
  std::string package_;
  std::vector<int> numbers_;
  bool periodic_;
};
} // namespace mace
#endif
