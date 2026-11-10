#include "cuda_neighbors.h"
#include <iostream>

int main(int argc, char **argv) {
  try {
    const int capacity = argc > 1 ? std::stoi(argv[1]) : 65536;
    int count;
    double cutoff;
    std::cin >> count >> cutoff;
    std::array<std::array<double, 3>, 3> cell{};
    for (auto &row : cell)
      for (auto &value : row)
        std::cin >> value;
    auto positions = at::empty({count, 3}, at::kDouble);
    for (int i = 0; i < 3 * count; ++i)
      std::cin >> positions.data_ptr<double>()[i];
    if (!std::cin)
      throw std::runtime_error("incomplete geometry");
    auto result = mace_native::periodic_cuda_neighbors(
        positions.to(at::kCUDA), cell, cutoff, capacity, 100);
    const int edges = result.count.item<int>();
    auto index = result.edge_index.to(at::kCPU).contiguous();
    auto shifts = result.unit_shifts.to(at::kCPU).contiguous();
    const auto edge = index.accessor<int64_t, 2>();
    const auto shift = shifts.accessor<int64_t, 2>();
    for (int i = 0; i < edges; ++i)
      std::cout << edge[0][i] << ' ' << edge[1][i] << ' ' << shift[i][0] << ' '
                << shift[i][1] << ' ' << shift[i][2] << '\n';
    return 0;
  } catch (const std::exception &error) {
    std::cerr << error.what() << '\n';
    return 2;
  }
}
