// A periodic cell list with deterministic atom and edge ordering. All distance
// tests use double precision, including for float32 models. No position or
// neighbour array is copied to the host.
#include "cuda_neighbors.h"
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAException.h>
#include <c10/cuda/CUDAGuard.h>
#include <climits>
#include <cmath>
#include <cstdio>
#include <limits>

namespace mace_native {
namespace {
__device__ void require(bool condition, const char *message) {
  if (!condition) {
    printf("MACE CUDA neighbours: %s\n", message);
    asm volatile("trap;");
  }
}

struct Grid {
  double cell[9], inverse[9], cutoff2;
  int bins[3], reach[3], count, capacity;
};

__device__ int floor_div(int value, int divisor) {
  return value >= 0 ? value / divisor : -((-value + divisor - 1) / divisor);
}

__global__ void assign_cells(const double *positions, int64_t *keys,
                             int64_t *images, int *counts, Grid grid) {
  const int atom = blockIdx.x * blockDim.x + threadIdx.x;
  if (atom >= grid.count)
    return;
  int bin[3];
  for (int a = 0; a < 3; ++a) {
    double fraction = 0;
    for (int b = 0; b < 3; ++b) {
      require(isfinite(positions[3 * atom + b]), "nonfinite position");
      fraction += positions[3 * atom + b] * grid.inverse[3 * b + a];
    }
    const double image = floor(fraction);
    require(fabs(image) < 0x1p60, "periodic image exceeds integer range");
    images[3 * atom + a] = static_cast<int64_t>(image);
    bin[a] = min(grid.bins[a] - 1,
                 static_cast<int>((fraction - image) * grid.bins[a]));
  }
  const int slot = (bin[0] * grid.bins[1] + bin[1]) * grid.bins[2] + bin[2];
  keys[atom] = static_cast<int64_t>(slot) * (grid.count + 1) + atom;
  atomicAdd(counts + slot, 1);
}

template <bool Write>
__global__ void enumerate_edges(const double *positions, const int64_t *keys,
                                const int64_t *sorted_keys,
                                const int64_t *images, const int *bin_starts,
                                const int64_t *edge_starts, int *degrees,
                                int64_t *index, int64_t *shifts, Grid grid) {
  const int receiver = blockIdx.x * blockDim.x + threadIdx.x;
  if (receiver >= grid.count)
    return;
  if constexpr (Write) {
    // Refuse overflow on the device before an incomplete graph can be used.
    require(edge_starts[grid.count] <= grid.capacity,
            "capacity exceeded; re-export with more edges");
  }
  int slot = keys[receiver] / (grid.count + 1), bin[3];
  bin[2] = slot % grid.bins[2];
  slot /= grid.bins[2];
  bin[1] = slot % grid.bins[1];
  bin[0] = slot / grid.bins[1];
  int degree = 0;
  for (int x = -grid.reach[0]; x <= grid.reach[0]; ++x)
    for (int y = -grid.reach[1]; y <= grid.reach[1]; ++y)
      for (int z = -grid.reach[2]; z <= grid.reach[2]; ++z) {
        const int unwrapped[3] = {bin[0] + x, bin[1] + y, bin[2] + z};
        int target[3], image[3];
        for (int a = 0; a < 3; ++a) {
          image[a] = floor_div(unwrapped[a], grid.bins[a]);
          target[a] = unwrapped[a] - image[a] * grid.bins[a];
        }
        const int target_bin =
            (target[0] * grid.bins[1] + target[1]) * grid.bins[2] + target[2];
        for (int p = bin_starts[target_bin]; p < bin_starts[target_bin + 1];
             ++p) {
          const int sender = sorted_keys[p] % (grid.count + 1);
          int64_t shift[3];
          for (int a = 0; a < 3; ++a)
            shift[a] =
                images[3 * sender + a] - images[3 * receiver + a] - image[a];
          if (sender == receiver && shift[0] == 0 && shift[1] == 0 &&
              shift[2] == 0)
            continue;
          double squared = 0;
          for (int a = 0; a < 3; ++a) {
            double vector =
                positions[3 * receiver + a] - positions[3 * sender + a];
            for (int b = 0; b < 3; ++b)
              vector += shift[b] * grid.cell[3 * b + a];
            squared += vector * vector;
          }
          if (squared >= grid.cutoff2)
            continue;
          if constexpr (Write) {
            const int edge = edge_starts[receiver] + degree;
            index[edge] = sender;
            index[grid.capacity + edge] = receiver;
            for (int a = 0; a < 3; ++a)
              shifts[3 * edge + a] = shift[a];
          }
          ++degree;
        }
      }
  if constexpr (!Write)
    degrees[receiver] = degree;
}

Grid make_grid(const std::array<std::array<double, 3>, 3> &cell, double cutoff,
               int count, int capacity) {
  Grid grid{};
  grid.count = count;
  grid.capacity = capacity;
  grid.cutoff2 = cutoff * cutoff;
  for (int i = 0; i < 3; ++i)
    for (int j = 0; j < 3; ++j)
      grid.cell[3 * i + j] = cell[i][j];
  const auto &a = cell[0], &b = cell[1], &c = cell[2];
  const double volume = a[0] * (b[1] * c[2] - b[2] * c[1]) -
                        a[1] * (b[0] * c[2] - b[2] * c[0]) +
                        a[2] * (b[0] * c[1] - b[1] * c[0]);
  TORCH_CHECK(std::isfinite(volume) && volume > 0 && std::isfinite(cutoff) &&
                  cutoff > 0,
              "CUDA neighbours require a finite right-handed periodic cell and "
              "positive cutoff");
  for (int i = 0; i < 3; ++i)
    for (int j = 0; j < 3; ++j)
      grid.inverse[3 * i + j] =
          (cell[(j + 1) % 3][(i + 1) % 3] * cell[(j + 2) % 3][(i + 2) % 3] -
           cell[(j + 1) % 3][(i + 2) % 3] * cell[(j + 2) % 3][(i + 1) % 3]) /
          volume;
  double radius[3];
  for (int i = 0; i < 3; ++i) {
    double norm = 0;
    for (int j = 0; j < 3; ++j)
      norm += grid.inverse[3 * j + i] * grid.inverse[3 * j + i];
    radius[i] = cutoff * std::sqrt(norm);
    grid.bins[i] =
        std::max(1, int(std::min(double(count), std::floor(1 / radius[i]))));
  }
  // Bound empty-bin storage in sparse cells without changing the search radius.
  while (int64_t(grid.bins[0]) * grid.bins[1] * grid.bins[2] > 8LL * count) {
    int axis = 0;
    for (int i = 1; i < 3; ++i)
      if (grid.bins[i] > grid.bins[axis])
        axis = i;
    grid.bins[axis] = std::max(1, grid.bins[axis] / 2);
  }
  for (int i = 0; i < 3; ++i)
    grid.reach[i] = int(std::ceil(std::nextafter(
        radius[i] * grid.bins[i], std::numeric_limits<double>::infinity())));
  return grid;
}
} // namespace

CudaNeighbors
periodic_cuda_neighbors(const at::Tensor &positions,
                        const std::array<std::array<double, 3>, 3> &cell,
                        double cutoff, int capacity, int64_t padding_shift) {
  TORCH_CHECK(
      positions.is_cuda() && positions.scalar_type() == at::kDouble &&
          positions.dim() == 2 && positions.size(1) == 3 &&
          positions.is_contiguous(),
      "CUDA neighbour positions must be contiguous [N,3] float64 on CUDA");
  TORCH_CHECK(positions.size(0) <= INT_MAX, "too many CUDA neighbour atoms");
  const c10::cuda::CUDAGuard device_guard(positions.device());
  const int count = positions.size(0);
  TORCH_CHECK(count > 0 && capacity >= 0,
              "nonpositive native neighbour capacity");
  Grid grid = make_grid(cell, cutoff, count, capacity);
  const auto integers = positions.options().dtype(at::kInt);
  const auto longs = positions.options().dtype(at::kLong);
  auto keys = at::empty({count}, longs), images = at::empty({count, 3}, longs);
  const int bins = grid.bins[0] * grid.bins[1] * grid.bins[2];
  auto counts = at::zeros({bins}, integers);
  const auto stream = at::cuda::getCurrentCUDAStream();
  const int blocks = (count + 127) / 128;
  assign_cells<<<blocks, 128, 0, stream>>>(
      positions.data_ptr<double>(), keys.data_ptr<int64_t>(),
      images.data_ptr<int64_t>(), counts.data_ptr<int>(), grid);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  auto sorted_keys = std::get<0>(at::sort(keys));
  auto bin_starts =
      at::cat({at::zeros({1}, integers), at::cumsum(counts, 0, at::kInt)});
  auto degrees = at::empty({count}, integers);
  enumerate_edges<false><<<blocks, 128, 0, stream>>>(
      positions.data_ptr<double>(), keys.data_ptr<int64_t>(),
      sorted_keys.data_ptr<int64_t>(), images.data_ptr<int64_t>(),
      bin_starts.data_ptr<int>(), nullptr, degrees.data_ptr<int>(), nullptr,
      nullptr, grid);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  auto starts =
      at::cat({at::zeros({1}, longs), at::cumsum(degrees, 0, at::kLong)});
  // Eager callers can request the exact allocation with one scalar readback.
  // Native exported models supply a fixed capacity and never take this branch.
  if (capacity == 0) {
    const auto live = starts.select(0, count).item<int64_t>();
    TORCH_CHECK(live <= INT_MAX, "too many CUDA neighbour edges");
    capacity = live;
    grid.capacity = capacity;
  }
  auto index = at::full({2, capacity}, count - 1, longs);
  auto shifts = at::zeros({capacity, 3}, longs);
  shifts.select(1, 0).fill_(padding_shift);
  enumerate_edges<true><<<blocks, 128, 0, stream>>>(
      positions.data_ptr<double>(), keys.data_ptr<int64_t>(),
      sorted_keys.data_ptr<int64_t>(), images.data_ptr<int64_t>(),
      bin_starts.data_ptr<int>(), starts.data_ptr<int64_t>(), nullptr,
      index.data_ptr<int64_t>(), shifts.data_ptr<int64_t>(), grid);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return {index, shifts, starts.select(0, count)};
}
} // namespace mace_native
