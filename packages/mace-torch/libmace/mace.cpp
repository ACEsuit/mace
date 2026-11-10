// libmace's public geometry/Fortran interface over V1's native AOT package.
// CPU neighbours use upstream libmace; full periodic CUDA systems stay on GPU.
// The exported functions are shared with native LAMMPS.
#include "mace.h"
#include "../native/package_metadata.h"
#ifdef MACE_HAS_CUDA_NEIGHBORS
#include "../native/cuda_neighbors.h"
#endif
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

struct DeviceCalculation::Outputs {
  std::vector<at::Tensor> values;
};

const double *DeviceCalculation::forces() const {
  TORCH_CHECK(outputs_, "empty CUDA calculation");
  return outputs_->values.at(2).data_ptr<double>();
}

double DeviceCalculation::energy() const {
  TORCH_CHECK(outputs_, "empty CUDA calculation");
  return outputs_->values.at(0).item<double>();
}

namespace {
CalculationResult periodic_host_result(const std::vector<at::Tensor> &output,
                                       int count, bool calc_virial) {
  const auto energy = output.at(0).to(at::kCPU, at::kDouble).contiguous();
  const auto site = output.at(1).to(at::kCPU, at::kDouble).contiguous();
  const auto forces = output.at(2).to(at::kCPU, at::kDouble).contiguous();
  TORCH_CHECK(energy.numel() == 1 && site.sizes() == at::IntArrayRef({count}) &&
                  forces.sizes() == at::IntArrayRef({count, 3}) &&
                  output.at(3).sizes() == at::IntArrayRef({3, 3}),
              "invalid periodic output shapes");
  CalculationResult result;
  result.total_energy = energy.item<double>();
  result.node_energy.assign(site.data_ptr<double>(),
                            site.data_ptr<double>() + count);
  result.forces.resize(count);
  std::copy_n(forces.data_ptr<double>(), 3 * count,
              result.forces.front().data());
  if (calc_virial) {
    const auto virial = output.at(3).to(at::kCPU, at::kDouble).contiguous();
    constexpr int left[6] = {0, 1, 2, 0, 0, 1}, right[6] = {0, 1, 2, 1, 2, 2};
    const auto w = virial.accessor<double, 2>();
    for (int i = 0; i < 6; ++i)
      result.virial[i] = w[left[i]][right[i]];
  }
  return result;
}
} // namespace

CalculationResult DeviceCalculation::to_host(bool calc_virial) const {
  TORCH_CHECK(outputs_, "empty CUDA calculation");
  return periodic_host_result(outputs_->values, outputs_->values.at(1).numel(),
                              calc_virial);
}

struct MACE::Runtime {
  std::unique_ptr<Loader> loader;
  std::unique_ptr<mace_native::PackageMetadata> package;
  at::Device device{at::kCPU};
  at::ScalarType dtype;
  int nodes, edges;
  double cutoff;
  std::unordered_map<int, int> species;
  std::mutex evaluation;
  std::vector<int> cached_numbers;
  at::Tensor cuda_elements, cuda_cell;
  std::array<std::array<double, 3>, 3> cached_cell{};

#ifdef MACE_HAS_CUDA_NEIGHBORS
  std::vector<at::Tensor>
  evaluate_cuda(const at::Tensor &positions,
                const std::array<std::array<double, 3>, 3> &cell,
                const std::vector<int> &numbers) {
    package->validate_system(cell, {true, true, true}, numbers.size());
    if (numbers != cached_numbers) {
      auto elements = at::empty({nodes}, at::kLong);
      for (int i = 0; i < nodes; ++i) {
        const auto found = species.find(numbers[i]);
        TORCH_CHECK(found != species.end(), "unknown atomic number ",
                    numbers[i]);
        elements.data_ptr<int64_t>()[i] = found->second;
      }
      cuda_elements = elements.to(device);
      cached_numbers = numbers;
    }
    if (!cuda_cell.defined() || cell != cached_cell) {
      cuda_cell =
          at::from_blob(const_cast<std::array<double, 3> *>(cell.data()),
                        {3, 3}, at::kDouble)
              .to(device, dtype);
      cached_cell = cell;
    }
    auto neighbors = mace_native::periodic_cuda_neighbors(
        positions, cell, cutoff, edges, package->padding_unit_shift());
    return loader->run({neighbors.unit_shifts.to(dtype), neighbors.edge_index,
                        cuda_elements, positions.to(dtype), cuda_cell});
  }
#endif

  explicit Runtime(const std::string &path) {
    const auto meta = Loader::load_metadata_from_package(path, "model");
    this->package = std::make_unique<mace_native::PackageMetadata>(meta);
    const auto &package = *this->package;
    nodes = package.nodes;
    edges = package.edges;
    cutoff = package.cutoff;
    dtype = package.dtype == "float64" ? at::kDouble : at::kFloat;
    for (int number : package.atomic_numbers) {
      species.emplace(number, species.size());
    }
    device = at::Device(package.device);
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
  model.package->validate_system(cell, pbc, count);
  TORCH_CHECK(
      model.package->periodic_system ||
          count < static_cast<size_t>(model.nodes),
      "libmace needs ", count + 1,
      " node slots including padding; re-export with a larger capacity");
  for (const auto &row : cell)
    for (double value : row)
      TORCH_CHECK(std::isfinite(value), "cell contains a nonfinite value");
  const bool periodic = model.package->periodic_system;
#ifdef MACE_HAS_CUDA_NEIGHBORS
  if (periodic && model.device.is_cuda()) {
    auto points =
        at::from_blob(const_cast<std::array<double, 3> *>(positions.data()),
                      {model.nodes, 3}, at::kDouble)
            .to(model.device);
    return periodic_host_result(
        model.evaluate_cuda(points, cell, atomic_numbers), count, calc_virial);
  }
#endif
  auto elements = at::zeros({model.nodes}, at::kLong);
  for (size_t atom = 0; atom < count; ++atom) {
    const auto found = model.species.find(atomic_numbers[atom]);
    TORCH_CHECK(found != model.species.end(), "unknown atomic number ",
                atomic_numbers[atom], " for this native package");
    elements.data_ptr<int64_t>()[atom] = found->second;
    for (double value : positions[atom])
      TORCH_CHECK(std::isfinite(value), "positions contain a nonfinite value");
  }
  at::Tensor index, vectors;
  int live_edges = 0;
  {
    // The upstream API takes an atomic radius and sums the two radii for a
    // pair. Half the model cutoff selects exactly the required neighbourhood.
    const auto neighbors =
        calc_mace_neighbour_list(cell, pbc, positions, 0.5 * model.cutoff);
    index = at::full({2, model.edges}, model.nodes - 1, at::kLong);
    vectors = at::zeros({model.edges, 3}, at::kDouble);
    vectors.select(1, 0).fill_(periodic ? model.package->padding_unit_shift()
                                        : 2 * model.cutoff);
    auto edge = index.accessor<int64_t, 2>();
    auto vector = vectors.accessor<double, 2>();
    for (size_t item = 0; item < neighbors.edge_index.size(); ++item) {
      const int sender = neighbors.edge_index[item][0];
      const int receiver = neighbors.edge_index[item][1];
      std::array<double, 3> displacement{};
      double squared = 0;
      for (int axis = 0; axis < 3; ++axis) {
        displacement[axis] =
            positions[receiver][axis] - positions[sender][axis];
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
        vector[live_edges][axis] =
            periodic ? neighbors.unit_shifts[item][axis] : displacement[axis];
      ++live_edges;
    }
  }
  auto owned = at::zeros({model.nodes}, at::kDouble);
  owned.narrow(0, 0, count).fill_(1);
  if (model.package->periodic_system) {
    auto points =
        at::from_blob(const_cast<std::array<double, 3> *>(positions.data()),
                      {model.nodes, 3}, at::kDouble)
            .to(model.device, model.dtype);
    auto box = at::from_blob(const_cast<std::array<double, 3> *>(cell.data()),
                             {3, 3}, at::kDouble)
                   .to(model.device, model.dtype);
    const auto output = model.loader->run(
        {vectors.to(model.device, model.dtype), index.to(model.device),
         elements.to(model.device), points, box});
    return periodic_host_result(output, count, calc_virial);
  }
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
  const auto edge = index.accessor<int64_t, 2>();
  const auto vector = vectors.accessor<double, 2>();
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

bool MACE::accepts_cuda_positions() const {
#ifdef MACE_HAS_CUDA_NEIGHBORS
  return runtime->device.is_cuda() && runtime->package->periodic_system;
#else
  return false;
#endif
}

DeviceCalculation
MACE::calculate_cuda(const std::array<std::array<double, 3>, 3> &cell,
                     const double *positions,
                     const std::vector<int> &atomic_numbers) {
  TORCH_CHECK(accepts_cuda_positions(),
              "CUDA coordinates require a periodic CUDA package and runtime");
#ifdef MACE_HAS_CUDA_NEIGHBORS
  auto &model = *runtime;
  std::lock_guard<std::mutex> lock(model.evaluation);
  const c10::DeviceGuard guard(model.device);
  TORCH_CHECK(positions, "null CUDA positions");
  auto points = at::from_blob(
      const_cast<double *>(positions),
      {static_cast<int64_t>(atomic_numbers.size()), 3},
      at::TensorOptions().device(model.device).dtype(at::kDouble));
  DeviceCalculation result;
  result.outputs_ = std::make_shared<DeviceCalculation::Outputs>();
  result.outputs_->values = model.evaluate_cuda(points, cell, atomic_numbers);
  auto &forces = result.outputs_->values.at(2);
  forces = forces.to(at::kDouble).contiguous();
  return result;
#else
  return {};
#endif
}
