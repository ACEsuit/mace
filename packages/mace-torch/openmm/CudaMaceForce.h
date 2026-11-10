#pragma once
#include <memory>
#include <string>
#include <vector>
namespace OpenMM {
class ContextImpl;
}
namespace mace {
class CudaMaceForce {
public:
  virtual ~CudaMaceForce() = default;
  virtual double execute(OpenMM::ContextImpl &, bool includeForces,
                         bool includeEnergy) = 0;
};
std::unique_ptr<CudaMaceForce>
create_cuda_force(OpenMM::ContextImpl &, const std::string &package,
                  const std::vector<int> &numbers);
} // namespace mace
