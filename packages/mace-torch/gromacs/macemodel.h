// GROMACS INNPotModel implementation over V1's existing native libmace ABI.
// Interface: GROMACS 121090014570a53a17ea391bcddae45e5ea05eb4.
#ifndef MACE_GROMACS_MODEL_H
#define MACE_GROMACS_MODEL_H
#include "mace.h"
#include "nnpotmodel.h"
#include <memory>
namespace gmx {
struct NNPotParameters;
void validateMaceParameters(const NNPotParameters &parameters);
class NativeMaceModel final : public INNPotModel {
public:
  NativeMaceModel(const NNPotParameters &parameters, const MpiComm &comm);
  void evaluateModel(gmx_enerdata_t *, ArrayRef<RVec>, ArrayRef<const int>,
                     ArrayRef<const int>, ArrayRef<const std::string>,
                     ArrayRef<RVec>, ArrayRef<int>, ArrayRef<int>,
                     ArrayRef<RVec>, ArrayRef<RVec>, ArrayRef<real>, real,
                     ArrayRef<const LinkFrontierAtom>, matrix *,
                     PbcType *) override;
  bool outputsForces() const override { return true; }
  const matrix &virial() const { return virial_; }

private:
  std::unique_ptr<MACE> model_;
  const MpiComm &comm_;
  matrix virial_{};
};
} // namespace gmx
#endif
