// CUDA platform integration. OpenMM and ATen own different CUDA contexts on
// the same device. Explicit context scopes and synchronization preserve their
// ownership, while coordinates and forces are exchanged entirely on the GPU.
#include "CudaMaceForce.h"
#include "mace.h"
#include <cstring>
#include <cuda.h>
#include <openmm/OpenMMException.h>
#include <openmm/cuda/CudaContext.h>
#include <openmm/cuda/CudaPlatform.h>
#include <openmm/internal/ContextImpl.h>

namespace mace {
namespace {
void check(CUresult status) {
  if (status == CUDA_SUCCESS)
    return;
  const char *message = nullptr;
  cuGetErrorString(status, &message);
  throw OpenMM::OpenMMException(std::string("MACE CUDA: ") +
                                (message ? message : "driver error"));
}
class ContextScope {
public:
  explicit ContextScope(CUcontext context) { check(cuCtxPushCurrent(context)); }
  ~ContextScope() {
    CUcontext previous;
    cuCtxPopCurrent(&previous);
  }
};
constexpr double energyFactor = 96.48533288249877;

const char *source = R"CUDA(
extern "C" __global__ void copyPositions(const real4 *posq,
    const float4 *correction, const int *order, double *positions,
    const int4 *offsets, double lx, double ly, double lz,
    double xy, double xz, double yz) {
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= NUM_ATOMS) return;
  const real4 p = posq[i];
  double x = p.x, y = p.y, z = p.z;
#ifdef USE_MIXED_PRECISION
  const float4 c = correction[i];
  x += (double)c.x; y += (double)c.y; z += (double)c.z;
#endif
  const int4 image = offsets[i];
  x -= image.x*lx + image.y*xy + image.z*xz;
  y -= image.y*ly + image.z*yz;
  z -= image.z*lz;
  const int atom = order[i];
  positions[3*atom] = 10*x;
  positions[3*atom+1] = 10*y;
  positions[3*atom+2] = 10*z;
}
extern "C" __global__ void addForces(const double *forces,
    const int *order, unsigned long long *destination) {
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= NUM_ATOMS) return;
  const int atom = order[i];
  for (int a = 0; a < 3; ++a)
    destination[i + a*PADDED_NUM_ATOMS] += (unsigned long long)
        ((long long)(forces[3*atom+a] * 964.8533288249877 * 4294967296.0));
}
)CUDA";

class CudaForce final : public CudaMaceForce {
public:
  CudaForce(OpenMM::CudaContext &context, const std::string &package,
            const std::vector<int> &numbers)
      : cu_(context), numbers_(numbers) {
    if (cu_.getDeviceIndex() != 0)
      throw OpenMM::OpenMMException("MACE native CUDA uses visible device 0; "
                                    "select it with CUDA_VISIBLE_DEVICES");
    check(cuDevicePrimaryCtxRetain(&primary_, cu_.getDevice()));
    try {
      {
        ContextScope scope(primary_);
        model_ = std::make_unique<MACE>(package);
        if (model_->accepts_cuda_positions())
          check(cuMemAlloc(&positions_, sizeof(double) * 3 * numbers.size()));
      }
      if (positions_) {
        ContextScope scope(cu_.getContext());
        offsets_ = std::make_unique<OpenMM::CudaArray>();
        offsets_->initialize<OpenMM::mm_int4>(cu_, numbers.size(),
                                              "MACE cell offsets");
        const std::map<std::string, std::string> defines{
            {"NUM_ATOMS", std::to_string(numbers.size())},
            {"PADDED_NUM_ATOMS", std::to_string(cu_.getPaddedNumAtoms())}};
        module_ = cu_.createModule(source, defines, "--fmad=false");
        check(cuModuleGetFunction(&copy_, module_, "copyPositions"));
        check(cuModuleGetFunction(&add_, module_, "addForces"));
      }
    } catch (...) {
      cleanup();
      throw;
    }
  }
  ~CudaForce() override { cleanup(); }
  bool supported() const { return positions_ != 0; }

  double execute(OpenMM::ContextImpl &context, bool includeForces,
                 bool includeEnergy) override {
    std::array<std::array<double, 3>, 3> cell{};
    OpenMM::Vec3 box[3];
    context.getPeriodicBoxVectors(box[0], box[1], box[2]);
    for (int a = 0; a < 3; ++a)
      for (int b = 0; b < 3; ++b)
        cell[a][b] = 10 * box[a][b];
    {
      ContextScope scope(cu_.getContext());
      auto posq = cu_.getPosq().getDevicePointer();
      CUdeviceptr correction = cu_.getUseMixedPrecision()
                                   ? cu_.getPosqCorrection().getDevicePointer()
                                   : 0;
      auto order = cu_.getAtomIndexArray().getDevicePointer();
      // OpenMM keeps image counters on the host. Upload only when they change;
      // coordinates and forces never leave the device during force evaluation.
      const auto &images = cu_.getPosCellOffsets();
      if (cached_images_.size() != numbers_.size() ||
          std::memcmp(cached_images_.data(), images.data(),
                      numbers_.size() * sizeof(OpenMM::mm_int4)) != 0) {
        cached_images_.assign(images.begin(), images.begin() + numbers_.size());
        offsets_->upload(cached_images_);
      }
      auto offsets = offsets_->getDevicePointer();
      double lx = box[0][0], ly = box[1][1], lz = box[2][2], xy = box[1][0],
             xz = box[2][0], yz = box[2][1];
      void *args[] = {&posq, &correction, &order, &positions_, &offsets, &lx,
                      &ly,   &lz,         &xy,    &xz,         &yz};
      cu_.executeKernel(copy_, args, numbers_.size());
      check(cuCtxSynchronize());
    }
    ContextScope primaryScope(primary_);
    const auto result = model_->calculate_cuda(
        cell, reinterpret_cast<const double *>(positions_), numbers_);
    check(cuCtxSynchronize());
    if (includeForces) {
      ContextScope scope(cu_.getContext());
      const double *forces = result.forces();
      auto order = cu_.getAtomIndexArray().getDevicePointer();
      auto destination = cu_.getForce().getDevicePointer();
      void *args[] = {&forces, &order, &destination};
      cu_.executeKernel(add_, args, numbers_.size());
      check(cuCtxSynchronize());
    }
    return includeEnergy ? energyFactor * result.energy() : 0;
  }

private:
  void cleanup() noexcept {
    if (!primary_)
      return;
    CUcontext previous;
    if (cuCtxPushCurrent(cu_.getContext()) == CUDA_SUCCESS) {
      offsets_.reset();
      if (module_)
        cuModuleUnload(module_);
      cuCtxPopCurrent(&previous);
    }
    if (cuCtxPushCurrent(primary_) == CUDA_SUCCESS) {
      model_.reset();
      if (positions_)
        cuMemFree(positions_);
      cuCtxPopCurrent(&previous);
    }
    cuDevicePrimaryCtxRelease(cu_.getDevice());
    primary_ = nullptr;
    positions_ = 0;
    module_ = nullptr;
  }
  OpenMM::CudaContext &cu_;
  std::vector<int> numbers_;
  CUcontext primary_ = nullptr;
  CUdeviceptr positions_ = 0;
  CUmodule module_ = nullptr;
  CUfunction copy_ = nullptr, add_ = nullptr;
  std::unique_ptr<MACE> model_;
  std::unique_ptr<OpenMM::CudaArray> offsets_;
  std::vector<OpenMM::mm_int4> cached_images_;
};
} // namespace

extern "C" CudaMaceForce *
mace_create_cuda_force(OpenMM::ContextImpl &context, const std::string &package,
                       const std::vector<int> &numbers) {
  auto &data = *static_cast<OpenMM::CudaPlatform::PlatformData *>(
      context.getPlatformData());
  if (data.contexts.size() != 1)
    throw OpenMM::OpenMMException(
        "MACE native CUDA requires one OpenMM device");
  auto force = std::make_unique<CudaForce>(*data.contexts[0], package, numbers);
  if (!force->supported())
    return nullptr;
  return force.release();
}
} // namespace mace
