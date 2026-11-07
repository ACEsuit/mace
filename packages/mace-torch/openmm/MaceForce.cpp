// Uses OpenMM's platform-independent C++ force API (CustomCPPForceImpl).
// https://github.com/openmm/openmm/blob/8.6.1/openmmapi/include/openmm/internal/CustomCPPForceImpl.h
// OpenMM owns force distribution, device context selection and force groups.
#include "MaceForce.h"
#include "mace.h"
#include <memory>
#include <openmm/OpenMMException.h>
#include <openmm/System.h>
#include <openmm/internal/ContextImpl.h>
#include <openmm/internal/CustomCPPForceImpl.h>
#include <utility>

namespace mace {
class MaceForceImpl final : public OpenMM::CustomCPPForceImpl {
public:
  explicit MaceForceImpl(const MaceForce &owner)
      : CustomCPPForceImpl(owner), owner_(owner) {}
  const OpenMM::Force &getOwner() const override { return owner_; }
  void initialize(OpenMM::ContextImpl &context) override {
    if (context.getSystem().getNumParticles() != int(owner_.numbers_.size()))
      throw OpenMM::OpenMMException(
          "MaceForce atomic numbers must match all System particles");
    model_ = std::make_unique<MACE>(owner_.package_);
    CustomCPPForceImpl::initialize(context);
  }
  double computeForce(OpenMM::ContextImpl &context,
                      const std::vector<OpenMM::Vec3> &positions,
                      std::vector<OpenMM::Vec3> &forces) override {
    std::array<std::array<double, 3>, 3> cell{};
    if (owner_.periodic_) {
      OpenMM::Vec3 vectors[3];
      context.getPeriodicBoxVectors(vectors[0], vectors[1], vectors[2]);
      for (int i = 0; i < 3; ++i)
        for (int j = 0; j < 3; ++j)
          cell[i][j] = 10 * vectors[i][j];
    }
    std::vector<std::array<double, 3>> coordinates(positions.size());
    for (size_t i = 0; i < positions.size(); ++i)
      for (int j = 0; j < 3; ++j)
        coordinates[i][j] = 10 * positions[i][j];
    const auto pbc = owner_.periodic_;
    const auto result = model_->calculate(cell, {pbc, pbc, pbc}, coordinates,
                                          owner_.numbers_, false);
    // Same ASE conversion as the existing V1 OpenMM callback (CODATA 2014).
    constexpr double energyFactor = 96.48533288249877;
    forces.resize(positions.size());
    for (size_t i = 0; i < positions.size(); ++i)
      for (int j = 0; j < 3; ++j)
        forces[i][j] = result.forces[i][j] * (10 * energyFactor);
    return result.total_energy * energyFactor;
  }

private:
  const MaceForce &owner_;
  std::unique_ptr<MACE> model_;
};

MaceForce::MaceForce(std::string package, std::vector<int> numbers,
                     bool periodic)
    : package_(std::move(package)), numbers_(std::move(numbers)),
      periodic_(periodic) {
  if (numbers_.empty())
    throw OpenMM::OpenMMException(
        "MaceForce requires at least one atomic number");
  for (int number : numbers_)
    if (number < 1 || number > 118)
      throw OpenMM::OpenMMException(
          "MaceForce atomic numbers must be in [1,118]");
}
OpenMM::ForceImpl *MaceForce::createImpl() const {
  return new MaceForceImpl(*this);
}
} // namespace mace
