// Optional torch entry point for the same preprocessing kernel as libmace.
#include "cuda_neighbors.h"
#include <torch/library.h>

namespace {
std::tuple<at::Tensor, at::Tensor, at::Tensor>
periodic(const at::Tensor &positions, at::ArrayRef<double> values,
         double cutoff, int64_t capacity, int64_t padding_shift) {
  TORCH_CHECK(values.size() == 9, "CUDA neighbours require a 3x3 cell");
  TORCH_CHECK(capacity >= 0 && capacity <= INT_MAX,
              "CUDA neighbour capacity is outside the int32 range");
  std::array<std::array<double, 3>, 3> cell;
  std::copy(values.begin(), values.end(), cell.front().data());
  auto result = mace_native::periodic_cuda_neighbors(positions, cell, cutoff,
                                                     capacity, padding_shift);
  return {result.edge_index, result.unit_shifts, result.count};
}
} // namespace
TORCH_LIBRARY(mace_neighbors, library) {
  library.def(
      "periodic(Tensor positions, float[] cell, float cutoff, int capacity, "
      "int padding_shift) -> (Tensor, Tensor, Tensor)");
}
TORCH_LIBRARY_IMPL(mace_neighbors, CUDA, library) {
  library.impl("periodic", &periodic);
}
