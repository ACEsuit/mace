// Standalone reader for the DEP-2 CPU capture probe. No embedded interpreter.
#include <ATen/ATen.h>
#include <torch/csrc/inductor/aoti_package/model_package_loader.h>
#include <cstdint>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

at::Tensor read_tensor(std::ifstream& stream, std::initializer_list<int64_t> shape,
                       at::ScalarType dtype) {
    auto tensor = at::empty(shape, at::TensorOptions().dtype(dtype).device(at::kCPU));
    if (!stream.read(static_cast<char*>(tensor.data_ptr()), tensor.nbytes()))
        throw std::runtime_error("Incomplete sample");
    return tensor;
}

int main(int argc, char** argv) {
    if (argc != 6) return 2;
    torch::inductor::AOTIModelPackageLoader model(argv[1]);
    std::ifstream samples(argv[2], std::ios::binary);
    const double tolerance = std::stod(argv[4]);
    for (int step = 0; step < std::stoi(argv[3]); ++step) {
        int64_t shape[3];
        if (!samples.read(reinterpret_cast<char*>(shape), sizeof(shape))) return 2;
        const auto [n, edges, species] = shape;
        std::vector<at::Tensor> inputs{
            read_tensor(samples, {n, 3}, at::kDouble),
            read_tensor(samples, {edges, 3}, at::kDouble),
            read_tensor(samples, {2, edges}, at::kLong),
            read_tensor(samples, {n, species}, at::kDouble),
            read_tensor(samples, {n}, at::kLong)
        };
        auto expected_energy = read_tensor(samples, {}, at::kDouble);
        auto expected_forces = read_tensor(samples, {n, 3}, at::kDouble);
        auto device_inputs = inputs;
        for (auto& tensor : device_inputs) tensor = tensor.to(at::Device(argv[5]));
        auto out = model.run(device_inputs);
        for (auto& tensor : out) tensor = tensor.cpu();
        auto reconstructed = at::zeros_like(expected_forces);
        reconstructed.index_add_(0, inputs[2][0], out.at(2));
        reconstructed.index_add_(0, inputs[2][1], -out.at(2));
        double de = (out.at(0) - expected_energy).abs().item<double>();
        double df = (out.at(1) - expected_forces).abs().max().item<double>();
        double dp = (reconstructed - expected_forces).abs().max().item<double>();
        std::cout << "case " << step << " |dE| " << de << " max |dF| " << df
                  << " from edge gradients " << dp << std::endl;
        if (!(de <= tolerance && df <= tolerance && dp <= tolerance)) return 1;
    }
}
