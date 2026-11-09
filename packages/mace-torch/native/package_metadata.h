// The native.pt2 metadata contract shared by the LAMMPS and libmace runtimes.
#pragma once

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
      require(!index.empty() && index.find_first_not_of("0123456789") ==
                                   std::string::npos,
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
};

} // namespace mace_native
