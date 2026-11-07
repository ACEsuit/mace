"""Create an isolated diagnostic libmace source; never edit the production runtime.

Synchronization separates model time from D2H. These diagnostics are reported
apart from uninstrumented wall-time measurements.
"""

import argparse
from pathlib import Path

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("source", type=Path)
parser.add_argument("output", type=Path)
args = parser.parse_args()
source = args.source.read_text()
source = source.replace(
    "#include <cmath>", "#include <cmath>\n#include <chrono>\n#include <cstdlib>"
)
source = source.replace(
    "  const auto neighbors =",
    "  const auto profile_start = std::chrono::steady_clock::now();\n  const auto neighbors =",
    1,
)
source = source.replace(
    "  auto index = at::full",
    "  const auto profile_neighbors = std::chrono::steady_clock::now();\n  auto index = at::full",
    1,
)
old = """  const auto output = model.loader->run(
      {vectors.to(model.device, model.dtype), index.to(model.device),
       elements.to(model.device), owned.to(model.device, model.dtype)});"""
new = """  const auto profile_prepare = std::chrono::steady_clock::now();
  const std::vector<at::Tensor> inputs = {
      vectors.to(model.device, model.dtype), index.to(model.device),
      elements.to(model.device), owned.to(model.device, model.dtype)};
  const auto profile_input = std::chrono::steady_clock::now();
  const auto output = model.loader->run(inputs);
#ifdef MACE_HAS_CUEQ
  if (model.device.is_cuda()) cudaDeviceSynchronize();
#endif
  const auto profile_model = std::chrono::steady_clock::now();"""
assert old in source
source = source.replace(old, new)
source = source.replace(
    "  TORCH_CHECK(site.sizes()",
    "  const auto profile_output = std::chrono::steady_clock::now();\n  TORCH_CHECK(site.sizes()",
    1,
)
source = source.replace(
    "  return result;",
    """  const auto profile_finish = std::chrono::steady_clock::now();
  if (std::getenv("MACE_PROFILE")) {
    const auto ms = [](auto first, auto last) { return std::chrono::duration<double, std::milli>(last-first).count(); };
    std::cerr << "MACE_PROFILE " << count << ' '
              << ms(profile_start, profile_neighbors) << ' '
              << ms(profile_neighbors, profile_prepare) << ' '
              << ms(profile_prepare, profile_input) << ' '
              << ms(profile_input, profile_model) << ' '
              << ms(profile_model, profile_output) << ' '
              << ms(profile_output, profile_finish) << '\\n';
  }
  return result;""",
    1,
)
args.output.write_text(source)
