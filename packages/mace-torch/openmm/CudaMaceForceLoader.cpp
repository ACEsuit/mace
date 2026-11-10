#include "CudaMaceForce.h"
#include <dlfcn.h>
#include <filesystem>
#include <openmm/OpenMMException.h>

namespace mace {
namespace {
using Factory = CudaMaceForce *(*)(OpenMM::ContextImpl &, const std::string &,
                                   const std::vector<int> &);

Factory load_factory() {
  static const char library_anchor = 0;
  Dl_info location{};
  if (!dladdr(&library_anchor, &location) || !location.dli_fname)
    throw OpenMM::OpenMMException("Cannot locate the MACE OpenMM library");
  const auto path = std::filesystem::path(location.dli_fname).parent_path() /
                    MACE_OPENMM_CUDA_MODULE;
  // Keep the module loaded for all force instances, like OpenMM's own plugins.
  // RTLD_LOCAL is essential: CUDA and OpenCL export overlapping private names.
  void *module = dlopen(path.c_str(), RTLD_NOW | RTLD_LOCAL);
  if (!module)
    throw OpenMM::OpenMMException("Cannot load MACE CUDA adapter " +
                                  path.string() + ": " + dlerror());
  auto factory =
      reinterpret_cast<Factory>(dlsym(module, "mace_create_cuda_force"));
  if (!factory)
    throw OpenMM::OpenMMException("Invalid MACE CUDA adapter " + path.string() +
                                  ": " + dlerror());
  return factory;
}
} // namespace

std::unique_ptr<CudaMaceForce>
create_cuda_force(OpenMM::ContextImpl &context, const std::string &package,
                  const std::vector<int> &numbers) {
  static const auto factory = load_factory();
  return std::unique_ptr<CudaMaceForce>(factory(context, package, numbers));
}
} // namespace mace
