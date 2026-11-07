// Standalone consumer of both public interfaces. No torch or Python headers.
#include "mace.h"
#include <algorithm>
#include <iomanip>
#include <iostream>
#include <stdexcept>

int main(int argc, char **argv) {
  if (argc == 2 && std::string(argv[1]) == "--version") {
    std::cout << "mace-libmace-aoti-v1\n";
    return 0;
  }
  try {
    if (argc < 2)
      throw std::runtime_error("expected native.pt2");
    const bool use_c = argc == 3 && std::string(argv[2]) == "c";
    auto cpp = use_c ? nullptr : std::make_unique<MACE>(argv[1]);
    CMace *c = use_c ? cmace_init(argv[1]) : nullptr;
    int count, virial, reload;
    std::cout << std::scientific << std::setprecision(17);
    while (std::cin >> count >> virial >> reload) {
      if (count <= 0)
        throw std::runtime_error("nonpositive atom count");
      std::array<std::array<double, 3>, 3> cell;
      std::array<int, 3> periodic;
      std::array<bool, 3> pbc;
      std::vector<int> numbers(count);
      std::vector<std::array<double, 3>> positions(count);
      for (auto &row : cell)
        for (auto &value : row)
          std::cin >> value;
      for (int axis = 0; axis < 3; ++axis) {
        std::cin >> periodic[axis];
        pbc[axis] = periodic[axis] != 0;
      }
      for (auto &number : numbers)
        std::cin >> number;
      for (auto &row : positions)
        for (auto &value : row)
          std::cin >> value;
      if (!std::cin)
        throw std::runtime_error("incomplete geometry");
      CalculationResult result;
      if (use_c) {
        if (reload)
          cmace_reload(c);
        result.node_energy.resize(count);
        result.forces.resize(count);
        cmace_calculate(c, virial, count, cell[0].data(), periodic.data(),
                        numbers.data(), positions[0].data(),
                        &result.total_energy, result.node_energy.data(),
                        result.forces[0].data(), result.virial.data());
      } else {
        if (reload)
          cpp->reload();
        result = cpp->calculate(cell, pbc, positions, numbers, virial != 0);
      }
      std::cout << "RESULT " << result.total_energy;
      for (auto value : result.node_energy)
        std::cout << ' ' << value;
      for (auto row : result.forces)
        for (auto value : row)
          std::cout << ' ' << value;
      for (auto value : result.virial)
        std::cout << ' ' << value;
      std::cout << '\n';
    }
    cmace_finalise(c);
    return 0;
  } catch (const std::exception &error) {
    std::cerr << error.what() << '\n';
    return 2;
  }
}
