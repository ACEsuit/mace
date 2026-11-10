# Hardware, precision and kernel backends

CPU reference kernels are always available. GPU execution needs a compatible
PyTorch installation and the chosen backend's optional dependencies. Select
the device and backend explicitly in configuration. CUDA is also PyTorch's
device spelling on ROCm. cuEquivariance is NVIDIA-only; OpenEquivariance has
vendor-specific JIT toolchain requirements. A wheel's runtime libraries do not
replace a compiler toolkit needed by a JIT backend.

Native checkpoints store canonical weights independently of backend layout.
The backend binds an operation at model construction. This removes the old
runtime e3nn↔cueq model conversion workflow. Precision has separate model and
accumulation settings, with explicit per-operation overrides; consult
[configuration](../reforge/configuration.md) before overriding them.

The `gpu`, `cueq` and `oeq` markers describe test requirements. `network`
permits downloads; `schedulefree` and `wandb` cover optional optimizer/tracking
clients. A required capability unavailable on a CI host is an error. See
[capabilities](capabilities.md) for the full marker-to-guide map and
[backend contracts](../reforge/extending_plugin.md) when implementing a plugin.
