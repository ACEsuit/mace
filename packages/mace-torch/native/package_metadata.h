// The native.pt2 metadata contract shared by the LAMMPS and libmace runtimes.
#pragma once

#include <array>
#include <cmath>
#include <sstream>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <vector>

namespace mace_native {

struct PackageMetadata {
  int nodes, edges, comm_width;
  double cutoff;
  std::string dtype, backend, device;
  std::vector<int> atomic_numbers;
  std::vector<std::string> species;
  bool periodic_system;
  std::array<std::array<double, 3>, 3> cell{};

  explicit PackageMetadata(
      const std::unordered_map<std::string, std::string> &values) {
    const auto require = [](bool condition, const std::string &message) {
      if (!condition)
        throw std::runtime_error(message);
    };
    const auto integer = [&](const std::string &key) {
      size_t end;
      int value = std::stoi(values.at(key), &end);
      require(end == values.at(key).size(), "invalid native integer: " + key);
      return value;
    };
    require(values.at("mace_native_version") == "1.0",
            "unsupported native package version; expected 1.0");
    nodes = integer("mace_nodes");
    edges = integer("mace_edges");
    comm_width = integer("mace_comm_width");
    const auto evaluation = values.at("mace_evaluation");
    require(evaluation == "local_edges" || evaluation == "periodic_system",
            "invalid native evaluation interface");
    periodic_system = evaluation == "periodic_system";
    if (periodic_system) {
      require(comm_width == 0, "periodic systems do not use a halo exchange");
      std::istringstream lattice(values.at("mace_cell"));
      for (auto &row : cell)
        for (auto &value : row)
          require(bool(lattice >> value) && std::isfinite(value),
                  "invalid periodic native cell");
      std::string extra;
      require(!(lattice >> extra),
              "periodic native cell must contain nine values");
      const auto &a = cell[0], &b = cell[1], &c = cell[2];
      const double volume = a[0] * (b[1] * c[2] - b[2] * c[1]) -
                            a[1] * (b[0] * c[2] - b[2] * c[0]) +
                            a[2] * (b[0] * c[1] - b[1] * c[0]);
      require(volume > 0, "periodic native cell must be right-handed");
    }
    size_t end;
    cutoff = std::stod(values.at("mace_cutoff"), &end);
    require(end == values.at("mace_cutoff").size() && std::isfinite(cutoff) &&
                cutoff > 0 && nodes >= 3 && edges >= 1 && comm_width >= 0,
            "invalid native capacity, communication width or cutoff");
    dtype = values.at("mace_dtype");
    require(dtype == "float64" || dtype == "float32",
            "native package dtype must be float64 or float32");
    backend = values.at("mace_backend");
    require(backend == "reference" || backend == "cueq",
            "unsupported native package backend: " + backend);
#ifndef MACE_HAS_CUEQ
    require(backend != "cueq",
            "cuEq package requires a runtime built with CUEQ_OPS_DIR");
#endif
    device = values.at("mace_device");
    const auto colon = device.find(':');
    const auto type = device.substr(0, colon);
    require(type == "cpu" || type == "cuda",
            "native package device must be cpu or cuda");
    if (colon != std::string::npos) {
      const auto index = device.substr(colon + 1);
      require(!index.empty() &&
                  index.find_first_not_of("0123456789") == std::string::npos,
              "invalid native device index");
    }
    std::istringstream numbers(values.at("mace_atomic_numbers"));
    std::unordered_set<int> seen_numbers;
    for (std::string token; numbers >> token;) {
      int number = std::stoi(token, &end);
      require(end == token.size() && number > 0 && number <= 118 &&
                  seen_numbers.insert(number).second,
              "invalid or duplicate native atomic number");
      atomic_numbers.push_back(number);
    }
    require(numbers.eof() && !atomic_numbers.empty(),
            "invalid native atomic numbers");
    std::istringstream symbols(values.at("mace_species"));
    std::unordered_set<std::string> seen_symbols;
    for (std::string symbol; symbols >> symbol;) {
      require(seen_symbols.insert(symbol).second, "duplicate native species");
      species.push_back(symbol);
    }
    require(species.size() == atomic_numbers.size(),
            "native species and atomic numbers must have the same size");
  }

  double padding_unit_shift() const {
    const auto &a = cell[0];
    return std::ceil(2 * cutoff /
                     std::sqrt(a[0] * a[0] + a[1] * a[1] + a[2] * a[2])) +
           1;
  }

  void validate_system(const std::array<std::array<double, 3>, 3> &current,
                       const std::array<bool, 3> &pbc, size_t count) const {
    if (!periodic_system)
      return;
    if (count != static_cast<size_t>(nodes) || !pbc[0] || !pbc[1] || !pbc[2])
      throw std::runtime_error("Polar native package requires its exported "
                               "atom count and full periodicity");
    for (int i = 0; i < 3; ++i)
      for (int j = 0; j < 3; ++j)
        if (!std::isfinite(current[i][j]) ||
            std::abs(current[i][j] - cell[i][j]) > 1e-8)
          throw std::runtime_error("Polar native package requires its exported "
                                   "fixed cell; re-export for a changed cell");
  }
};

} // namespace mace_native
