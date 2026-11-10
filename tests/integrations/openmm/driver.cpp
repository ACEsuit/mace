// Native OpenMM application, including real integration steps and force groups.
#include "MaceForce.h"
#include <chrono>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <memory>
#include <openmm/Context.h>
#include <openmm/Platform.h>
#include <openmm/State.h>
#include <openmm/System.h>
#include <openmm/VerletIntegrator.h>
#include <stdexcept>

int main(int argc, char **argv) {
  if (argc == 2 && std::string(argv[1]) == "--version") {
    std::cout << "mace-openmm-native-v1\n";
    return 0;
  }
  try {
    if (argc < 3)
      throw std::runtime_error("expected native.pt2 platform [steps]");
    const int steps = argc > 3 ? std::stoi(argv[3]) : 0;
    const bool benchmark = argc > 4 && std::string(argv[4]) == "benchmark";
    OpenMM::Platform::loadPluginsFromDirectory(
        OpenMM::Platform::getDefaultPluginsDirectory());
    std::unique_ptr<OpenMM::System> system;
    std::unique_ptr<OpenMM::VerletIntegrator> integrator;
    std::unique_ptr<OpenMM::Context> context;
    std::vector<int> previousNumbers;
    bool previousPbc = false;
    int count, ignored1, ignored2;
    std::cout << std::scientific << std::setprecision(17);
    while (std::cin >> count >> ignored1 >> ignored2) {
      if (count <= 0)
        throw std::runtime_error("nonpositive atom count");
      OpenMM::Vec3 box[3];
      for (auto &row : box)
        for (int j = 0; j < 3; ++j) {
          std::cin >> row[j];
          row[j] /= 10;
        }
      int periodic[3];
      for (auto &value : periodic)
        std::cin >> value;
      if (periodic[0] != periodic[1] || periodic[1] != periodic[2])
        throw std::runtime_error(
            "OpenMM native supports full or no periodicity");
      std::vector<int> numbers(count);
      for (auto &value : numbers)
        std::cin >> value;
      std::vector<OpenMM::Vec3> positions(count);
      for (auto &row : positions)
        for (int j = 0; j < 3; ++j) {
          std::cin >> row[j];
          row[j] /= 10;
        }
      if (!std::cin)
        throw std::runtime_error("incomplete geometry");
      if (!context || previousNumbers != numbers ||
          previousPbc != bool(periodic[0])) {
        context.reset();
        integrator.reset();
        system = std::make_unique<OpenMM::System>();
        for (int number : numbers)
          system->addParticle(number == 1    ? 1.008
                              : number == 14 ? 28.0855
                                             : 15.999);
        auto force = new mace::MaceForce(argv[1], numbers, periodic[0]);
        force->setForceGroup(3);
        system->addForce(force);
        if (periodic[0])
          system->setDefaultPeriodicBoxVectors(box[0], box[1], box[2]);
        integrator = std::make_unique<OpenMM::VerletIntegrator>(0.0001);
        auto &platform = OpenMM::Platform::getPlatformByName(argv[2]);
        std::map<std::string, std::string> properties;
        if (platform.getName() == "CUDA") {
          const char *precision = std::getenv("MACE_OPENMM_PRECISION");
          properties["Precision"] = precision ? precision : "double";
        }
        context = std::make_unique<OpenMM::Context>(*system, *integrator,
                                                    platform, properties);
        previousNumbers = numbers;
        previousPbc = periodic[0];
      }
      if (periodic[0])
        context->setPeriodicBoxVectors(box[0], box[1], box[2]);
      context->setPositions(positions);
      if (context->getState(OpenMM::State::Energy, false, 1)
              .getPotentialEnergy() != 0)
        throw std::runtime_error("force group exclusion failed");
      if (benchmark)
        integrator->step(10);
      const auto start = std::chrono::steady_clock::now();
      if (benchmark) {
        integrator->step(steps);
        std::cerr << "BENCHMARK_MS "
                  << 1000 *
                         std::chrono::duration<double>(
                             std::chrono::steady_clock::now() - start)
                             .count() /
                         steps
                  << '\n';
      }
      for (int step = 0; step <= (benchmark ? 0 : steps); ++step) {
        if (step)
          integrator->step(1);
        const auto state =
            context->getState(OpenMM::State::Energy | OpenMM::State::Forces |
                              OpenMM::State::Positions);
        std::cout << "RESULT "
                  << state.getPotentialEnergy() / 96.48533288249877;
        for (const auto &row : state.getPositions())
          for (int j = 0; j < 3; ++j)
            std::cout << ' ' << row[j] * 10;
        for (const auto &row : state.getForces())
          for (int j = 0; j < 3; ++j)
            std::cout << ' ' << row[j] / 964.8533288249877;
        std::cout << '\n';
      }
      std::cerr << "ELAPSED "
                << std::chrono::duration<double>(
                       std::chrono::steady_clock::now() - start)
                       .count()
                << '\n';
    }
    return 0;
  } catch (const std::exception &error) {
    std::cerr << error.what() << '\n';
    return 2;
  }
}
