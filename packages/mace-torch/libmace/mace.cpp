// libmace's public geometry/Fortran interface over V1's native AOT package.
// Neighbours come from the pinned upstream libmace implementation. The model
// and edge derivative are the same exported functions used by native LAMMPS.
#include "mace.h"
#include "neighbour_list.h"

#include <ATen/ATen.h>
#include <c10/core/DeviceGuard.h>
#include <torch/csrc/inductor/aoti_package/model_package_loader.h>
#include <torch/library.h>
#ifdef MACE_HAS_CUEQ
#include <cuda_runtime_api.h>
#endif

#include <cmath>
#include <iostream>
#include <mutex>
#include <sstream>
#include <unordered_map>

using Loader = torch::inductor::AOTIModelPackageLoader;

// Every atom is owned here. libmace's Fortran MPI wrapper evaluates the entire
// structure on its root and broadcasts results; it does not decompose space.
TORCH_LIBRARY(mace_native, m) {
  m.def("halo(Tensor features) -> Tensor",
        [](const at::Tensor &values) { return values.clone(); });
  m.def("halo_adjoint(Tensor adjoint) -> Tensor",
        [](const at::Tensor &values) { return values.clone(); });
}

struct MACE::Runtime {
  std::unique_ptr<Loader> loader;
  at::Device device{at::kCPU};
  at::ScalarType dtype;
  int nodes, edges;
  double cutoff;
  std::unordered_map<int, int> species;
  std::mutex evaluation;

  explicit Runtime(const std::string &path) {
    const auto meta = Loader::load_metadata_from_package(path, "model");
    TORCH_CHECK(meta.at("mace_native_version") == "1",
                "libmace requires a V1 native package version 1");
    TORCH_CHECK(meta.count("mace_atomic_numbers"),
                "native.pt2 lacks atomic numbers; re-export with current V1");
    nodes = std::stoi(meta.at("mace_nodes"));
    edges = std::stoi(meta.at("mace_edges"));
    cutoff = std::stod(meta.at("mace_cutoff"));
    const auto precision = meta.at("mace_dtype");
    TORCH_CHECK(precision == "float64" || precision == "float32",
                "native package dtype must be float64 or float32");
    dtype = precision == "float64" ? at::kDouble : at::kFloat;
    const auto backend = meta.at("mace_backend");
    TORCH_CHECK(backend == "reference" || backend == "cueq",
                "unsupported native package backend: ", backend);
#ifndef MACE_HAS_CUEQ
    TORCH_CHECK(backend != "cueq",
                "cuEq package requires libmace built with CUEQ_OPS_DIR");
#endif
    std::istringstream numbers(meta.at("mace_atomic_numbers"));
    for (int number; numbers >> number;) {
      TORCH_CHECK(number > 0 && number <= 118 && !species.count(number),
                  "invalid or duplicate native atomic number: ", number);
      species.emplace(number, species.size());
    }
    TORCH_CHECK(nodes >= 3 && edges >= 1 && std::isfinite(cutoff) &&
                    cutoff > 0 && !species.empty(),
                "invalid native capacity, cutoff or species");
    device = at::Device(meta.at("mace_device"));
    TORCH_CHECK(device.is_cpu() || device.is_cuda(),
                "native package device must be cpu or cuda");
    if (device.is_cuda()) {
      TORCH_CHECK(at::getNumGPUs() > 0, "CUDA package needs a visible GPU");
      device = at::Device(at::kCUDA, 0);
#ifdef MACE_HAS_CUEQ
      cudaDeviceProp properties{};
      const auto status = cudaGetDeviceProperties(&properties, 0);
      TORCH_CHECK(status == cudaSuccess, cudaGetErrorString(status));
      const auto sm =
          std::to_string(properties.major) + std::to_string(properties.minor);
      TORCH_CHECK(
          meta.at("AOTI_COMPUTE_CAPABILITY") == sm,
          "native package targets another GPU; re-export on this device");
#endif
    }
    loader = std::make_unique<Loader>(path, "model", false, 1, device.index());
  }
};

MACE::MACE(const std::string &path) : model_path(path) { reload(); }
void MACE::reload() { runtime = std::make_shared<Runtime>(model_path); }
void MACE::print() const {
  std::cout << "V1 libmace AOT runtime: " << runtime->device << ", cutoff "
            << runtime->cutoff << " Angstrom, " << runtime->nodes
            << " node slots and " << runtime->edges << " edge slots\n";
}

CalculationResult
MACE::calculate(const std::array<std::array<double, 3>, 3> &cell,
                const std::array<bool, 3> &pbc,
                const std::vector<std::array<double, 3>> &positions,
                const std::vector<int> &atomic_numbers,
                const bool &calc_virial) {
  auto &model = *runtime;
  std::lock_guard<std::mutex> lock(model.evaluation);
  c10::DeviceGuard device_guard(model.device);
  at::NoGradGuard no_grad;
  const auto count = positions.size();
  TORCH_CHECK(count > 0 && count == atomic_numbers.size(),
              "positions and atomic_numbers must have the same nonzero length");
  TORCH_CHECK(
      count < static_cast<size_t>(model.nodes), "libmace needs ", count + 1,
      " node slots including padding; re-export with a larger capacity");
  for (const auto &row : cell)
    for (double value : row)
      TORCH_CHECK(std::isfinite(value), "cell contains a nonfinite value");
  auto elements = at::zeros({model.nodes}, at::kLong);
  for (size_t atom = 0; atom < count; ++atom) {
    const auto found = model.species.find(atomic_numbers[atom]);
    TORCH_CHECK(found != model.species.end(), "unknown atomic number ",
                atomic_numbers[atom], " for this native package");
    elements.data_ptr<int64_t>()[atom] = found->second;
    for (double value : positions[atom])
      TORCH_CHECK(std::isfinite(value), "positions contain a nonfinite value");
  }
  // The upstream API takes an atomic radius and sums the two radii for a
  // pair. Half the model cutoff selects exactly the required neighbourhood.
  const auto neighbors =
      calc_mace_neighbour_list(cell, pbc, positions, 0.5 * model.cutoff);
  auto index = at::full({2, model.edges}, model.nodes - 1, at::kLong);
  auto vectors = at::zeros({model.edges, 3}, at::kDouble);
  vectors.select(1, 0).fill_(2 * model.cutoff);
  auto edge = index.accessor<int64_t, 2>();
  auto vector = vectors.accessor<double, 2>();
  int live_edges = 0;
  for (size_t item = 0; item < neighbors.edge_index.size(); ++item) {
    const int sender = neighbors.edge_index[item][0];
    const int receiver = neighbors.edge_index[item][1];
    std::array<double, 3> displacement{};
    double squared = 0;
    for (int axis = 0; axis < 3; ++axis) {
      displacement[axis] = positions[receiver][axis] - positions[sender][axis];
      for (int lattice = 0; lattice < 3; ++lattice)
        displacement[axis] += neighbors.unit_shifts[item][lattice] *
                              neighbors.extended_cell[lattice][axis];
      squared += displacement[axis] * displacement[axis];
    }
    // Keep the radial boundary explicit when rebuilding the edge vectors.
    if (squared >= model.cutoff * model.cutoff)
      continue;
    TORCH_CHECK(live_edges < model.edges, "libmace exceeds ", model.edges,
                " edge slots; re-export with a larger capacity");
    edge[0][live_edges] = sender;
    edge[1][live_edges] = receiver;
    for (int axis = 0; axis < 3; ++axis)
      vector[live_edges][axis] = displacement[axis];
    ++live_edges;
  }
  auto owned = at::zeros({model.nodes}, at::kDouble);
  owned.narrow(0, 0, count).fill_(1);
  const auto output = model.loader->run(
      {vectors.to(model.device, model.dtype), index.to(model.device),
       elements.to(model.device), owned.to(model.device, model.dtype)});
  const auto site = output.at(0).to(at::kCPU, at::kDouble).contiguous();
  const auto gradient = output.at(1).to(at::kCPU, at::kDouble).contiguous();
  TORCH_CHECK(site.sizes() == at::IntArrayRef({model.nodes}) &&
                  gradient.sizes() == at::IntArrayRef({model.edges, 3}),
              "invalid native package output shapes");
  CalculationResult result;
  result.node_energy.assign(site.data_ptr<double>(),
                            site.data_ptr<double>() + count);
  result.forces.resize(count, {0, 0, 0});
  for (double energy : result.node_energy)
    result.total_energy += energy;
  const auto grad = gradient.accessor<double, 2>();
  constexpr int left[6] = {0, 1, 2, 0, 0, 1};
  constexpr int right[6] = {0, 1, 2, 1, 2, 2};
  for (int item = 0; item < live_edges; ++item) {
    for (int axis = 0; axis < 3; ++axis) {
      result.forces[edge[0][item]][axis] += grad[item][axis];
      result.forces[edge[1][item]][axis] -= grad[item][axis];
    }
    if (calc_virial)
      for (int component = 0; component < 6; ++component) {
        const int a = left[component], b = right[component];
        result.virial[component] -= 0.5 * (vector[item][a] * grad[item][b] +
                                           vector[item][b] * grad[item][a]);
      }
  }
  return result;
}
