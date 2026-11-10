// Exercise the shared metadata parser without libtorch or an MD engine.
#include "package_metadata.h"
#include <iostream>

int main(int argc, char **argv) {
  std::unordered_map<std::string, std::string> values{
      {"mace_native_version", "1.0"},
      {"mace_nodes", "8"},
      {"mace_evaluation", "local_edges"},
      {"mace_edges", "32"},
      {"mace_comm_width", "16"},
      {"mace_cutoff", "5.0"},
      {"mace_dtype", "float64"},
      {"mace_backend", "reference"},
      {"mace_device", "cpu"},
      {"mace_atomic_numbers", "1 6 8"},
      {"mace_species", "H C O"}};
  for (int i = 1; i + 1 < argc; i += 2)
    values[argv[i]] = argv[i + 1];
  try {
    const mace_native::PackageMetadata package(values);
    std::cout << package.nodes << ' ' << package.edges << ' '
              << package.comm_width << ' ' << package.cutoff << ' '
              << package.dtype << ' ' << package.device << '\n';
    for (size_t i = 0; i < package.species.size(); ++i)
      std::cout << package.species[i] << ':' << package.atomic_numbers[i]
                << ' ';
    return 0;
  } catch (const std::exception &error) {
    std::cerr << error.what();
    return 1;
  }
}
