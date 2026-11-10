# mace-backend-example

A separately installed Torch backend and a template for implementing one. It
implements the five dispatched operation classes using its own Torch code and
custom operators. Installing its entry point makes it discoverable without
adding its name to MACE's implementation. This example distribution is never
published; repository CI exercises it against the native packages.

The [V1 backend guide](../../docs/reforge/kernel_backends.md) describes the
factory API, canonical weights, precision and validation. The
[performance contracts](../../docs/reforge/performance_contracts.md) cover
optional fusion APIs. This example implements individual operations and leaves
fusion to the caller's fallback.

## Install and exercise locally

From the repository root, in an environment with a compatible Torch install:

```sh
python -m pip install -e packages/mace-core -e packages/mace-torch -e packages/mace-backend-example
python -m pytest packages/mace-backend-example/tests
```

```python
from mace_core.kernels import get_backend
from mace_core.kernels.protocol import KernelBackend
from mace_torch.backends.conformance import run_backend_conformance

backend = get_backend("example", framework="torch")
assert isinstance(backend, KernelBackend)
results = run_backend_conformance(backend, precision="float64", compile_ops=True)
assert results and all(result.built for result in results)
```

CUDA capture is a separate opt-in check: pass `device="cuda", cuda_graphs=True`
on a GPU runner. The package tests skip CUDA locally when unavailable and fail
instead if `MACE_REQUIRE_CAPS` promises `gpu`.

## Adapt the contract

`backend.py` depends on `mace_core`, Torch, NumPy and its own operators. Its
runtime code imports no `mace_torch` implementation. Tests use the public Torch
conformance and model helpers.

- `KernelBackend` requires `name`, `capabilities()` and the five dispatched
  `make_<op>(kernel_spec)` methods. It also includes `make_spherical_harmonics`,
  `make_radial_basis`, `make_radial_mlp` and `make_interaction_layer`; this example
  returns `None` from those four methods.
- `BackendCapabilities` declares supported ops, devices, precisions, layouts,
  bases and derivatives. This example uses `mul_ir` activations, supports both
  `reduced` and `full`, and rejects convolution multiplicities and non-scalar
  skip attributes that it cannot compute. Its `supports` override enforces
  those shape limits; new backends can override `unsupported_reason` to also
  supply a diagnostic. Direct unsupported requests raise
  `UnsupportedKernelSpecError`.
- A backend may accelerate only some ops. Declare that subset and use Torch's
  `resolve_backend` for per-op reference fallback. `None` declines an optional
  factory; `NotImplemented` is not supported. `get_backend` returns the raw
  backend without fallback.
- `INTERNAL_WEIGHT_OPS` lists four weight-owning classes: linear, symmetric
  contraction, fully connected tensor product and radial MLP. The example
  implements the first three and leaves the MLP to the reference. Weight owners
  provide canonical save/load and seeded initialization. Linear/skip path
  normalization is applied during computation, preserving raw checkpoint weights.

## Register the implementation

```toml
[project.entry-points."mace.kernel_backends.torch"]
example = "mace_backend_example:ExampleBackend"
```

The target is a zero-argument factory; this class serves as one. Rename the
distribution, import package, entry-point name and custom-op namespace for your
backend. Conflicting targets registered under the same name raise an error;
identical duplicate registrations are coalesced.

## Implement and validate kernels

`ops.py` registers custom operators for convolution, contraction and reduction.
Each has a fake implementation with shapes derived from inputs/host metadata,
and an autograd rule written in differentiable Torch operations. The linear and
fully connected product modules use ordinary Torch operations, which need no
additional custom-op registration. An opaque GPU implementation must preserve
these compiler and derivative contracts.

The conformance tests compare values, input/weight gradients, equivariance where
applicable, canonical round-trips and declared unsupported cases. FP64 cases
also run `gradgradcheck` when double backward is declared. Compile and CUDA
capture checks run only when requested. The default cases cover the five
dispatched op classes; optional helpers and additional shapes need their own
tests. See the package tests for model-level energy, forces, stress and
force-loss weight-gradient checks.
