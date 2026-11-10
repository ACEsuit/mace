// Discrete neighbour search outside the differentiable model graph.
#pragma once
#include <ATen/ATen.h>
#include <array>

namespace mace_native {
struct CudaNeighbors {
  at::Tensor edge_index;
  at::Tensor unit_shifts;
  at::Tensor count;
};
CudaNeighbors
periodic_cuda_neighbors(const at::Tensor &positions,
                        const std::array<std::array<double, 3>, 3> &cell,
                        double cutoff, int capacity, int64_t padding_shift);
} // namespace mace_native
