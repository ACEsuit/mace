# Implementing a V1 kernel backend

This guide describes the API in this V1 checkout. The installable template is
`packages/mace-backend-example`: its runtime code depends on `mace_core`, Torch,
NumPy and its own kernels. Its tests exercise registration, numerical contracts
and model integration without adding the backend's name to MACE's implementation.

`mace_core.kernels` owns framework-free specifications, capabilities, canonical
weight conventions and discovery. A backend implements operators for one
framework. The Torch integration below uses `mace_torch.backends`; JAX has a
separate entry-point group and V1's JAX package supports inference only.

## Installation and discovery

From the repository root, with a compatible Torch installation:

```sh
python -m pip install -e packages/mace-core -e packages/mace-torch -e packages/mace-backend-example
```

An external distribution declares a **zero-argument factory** in its metadata.
A class with a zero-argument constructor is sufficient:

```toml
[project.entry-points."mace.kernel_backends.torch"]
example = "mace_backend_example:ExampleBackend"
```

Use `mace.kernel_backends.jax` for a JAX implementation. Rename the package,
entry-point name and custom-op namespace when adapting the example. Conflicting
targets under the same name raise `DuplicateBackendError`; identical duplicate
registrations are coalesced. Missing or unimportable selected backends raise
`BackendNotAvailableError`.

```python
from mace_core.kernels import available_backends, get_backend
from mace_core.kernels.protocol import KernelBackend

discovered = {item.name: item for item in available_backends(framework="torch")}
assert discovered["example"].loaded, discovered["example"].reason
backend = get_backend("example", framework="torch")
assert isinstance(backend, KernelBackend)
```

`available_backends` includes failed imports with their `reason`. `get_backend`
returns the raw backend. Torch model construction uses `resolve_backend` to
compose it with reference fallback and, when supplied, precision policy.
Installing a backend makes it discoverable; selecting it is a separate step.

## Factories and capabilities

`KernelBackend` is a runtime-checkable structural protocol. It requires `name`,
`capabilities()` and the factories below. A factory receives the corresponding
immutable specification from `mace_core.kernels.specs`; specifications contain
shape and numerical policy, without weight tensors.

| Factory | Specification | Returned Torch operator call |
| --- | --- | --- |
| `make_linear` | `LinearSpec` | `op(features)` |
| `make_channelwise_tp_conv` | `ChannelwiseTPConvSpec` | `op(node_features, edge_attributes, radial_weights, sender, receiver, num_nodes)` |
| `make_symmetric_contraction` | `SymmetricContractionSpec` | `op(features, element)` |
| `make_fully_connected_tp` | `FullyConnectedTPSpec` | `op(features, attributes)` |
| `make_segment_reduce` | `SegmentReduceSpec` | `op(values, index, num_segments)` |
| `make_spherical_harmonics` | `SphericalHarmonicsSpec` | `op(directions)` or factory returns `None` |
| `make_radial_basis` | `RadialBasisSpec` | `op(lengths, envelope=None)` or factory returns `None` |
| `make_radial_mlp` | `RadialMLPSpec` | `op(radial)` or factory returns `None` |
| `make_interaction_layer` | tuple of specifications | stateless span, or `None`; see below |

The first five names form `DISPATCHED_OPS`. The next three form
`REFERENCE_ONLY_OPS`: despite that name, accelerated overrides are supported.
The example declares all five dispatched ops and returns `None` from the other
four factories. A backend accelerating a subset declares only that subset in
`BackendCapabilities.ops`; the composite supplies other operations.

Capabilities also declare `devices`, `dtypes`, `max_lmax` (zero means unbounded),
`weight_layouts`, `activation_layouts`, `native_activation_layout`, `products`,
`bases`, `supports_double_backward`, `wide_accumulation`, `edge_features` and
`spec_version`. Use `KERNEL_SPEC_VERSION` from `mace_core.kernels.canonical`
(currently `"1.0"`). Other versions are rejected.

For shape-specific limits, subclass `BackendCapabilities` and override
`unsupported_reason(spec) -> str | None`: preserve the parent's checks, then
return a diagnostic for additional restrictions. `supports(spec)` derives the
boolean answer. Direct unsupported requests to dispatched factories must raise
`UnsupportedKernelSpecError`; `capabilities.require(spec, name)` supplies that
check when using `unsupported_reason`.

Torch's composite falls back per operation when the primary does not declare
the op, does not support the specification, or returns `None`. Factories that
advertise support must build a working operator. **`NotImplemented` is not a
decline sentinel.** Exceptions propagate rather than triggering fallback.
`CompositeBackend.decisions` and `report()` record the choices made at build time.

```python
import torch
from mace_core.kernels.specs import LinearSpec
from mace_torch.backends.composite import resolve_backend

backend = resolve_backend("example")
spec = LinearSpec(irreps_in="2x0e", irreps_out="3x0e", precision="float64")
operation = backend.make_linear(spec)
operation.initialize_weights(seed=7)
features = torch.ones(4, 2, dtype=torch.float64)
assert operation(features).shape == (4, 3)
```

Convolution returns receiver-summed **node** outputs with `num_nodes` rows,
including zero messages for isolated nodes. Its feature blocks follow
`spec.irreps_paths`, which enumerates coupling paths. The radial network supplies
external convolution weights. Harmonic conventions, radial families and cutoffs
are fixed by their specifications. A radial override must keep the cutoff on
physical distances when a distance transform is used; expose `kernel_spec`,
`cutoff`, `canonical_components` and `span_weights` as the radial protocol states.

## Canonical weights, basis and precision

`INTERNAL_WEIGHT_OPS` contains linear, symmetric contraction, fully connected
tensor product and radial MLP. Their operators implement `InternalWeights`:

- `to_canonical()` returns the canonical tensor mapping.
- `load_canonical(state)` restores it, converting private storage if needed.
- `initialize_weights(seed)` performs the canonical draw with the caller's
  per-op seed. Use a local generator and the scales in
  `mace_core.kernels.canonical`; zero-only initialization is invalid.

Canonical checkpoint weights use `mul_ir` independently of activation layout.
Linear and skip weights retain their raw values; path normalization is applied
in the forward computation. Preserve contraction path labels and
`canonical_metadata`, and expose logical component paths through
`canonical_components` when an operator contains several canonical parts.
Constants affecting model values belong to canonical state as well.

`ModelConfig.clebsch_gordan_basis` defaults to `reduced` and also accepts explicit
`full`. The reference and example support both. A backend may reject a basis and
use reference fallback, but must preserve the model's selected basis. Native
checkpoint loading retains it. The explicit legacy neutral importer projects
full weights to reduced in FP64; that conversion is specific to the importer.
CG coefficients, path order and normalization come from `mace_core`'s
first-principles construction, independently of optional kernel libraries.

The composite resolves one activation layout for the operator chain. The
reference accepts both `mul_ir` and `ir_mul`, allowing fallback within the same
layout. `weight_layouts` describes accepted weight layouts, not private storage.

With a `PrecisionConfig`, `resolve_backend` wraps factories in `PrecisionBackend`.
`precision` names the compute dtype and `accumulate` can impose a wider floor.
Without `wide_accumulation`, the wrapper promotes compute and parameters to that
floor, with casts at the interface. A backend declaring wide accumulation must
honour the accumulator dtype itself. TF32 is eligible only for float32 linear
and fully connected products and requires `products` support. Unavailable TF32,
including after promotion to float64, becomes IEEE with a build diagnostic.
See [configuration](configuration.md) for user-facing precision policy.

## Autograd and compilation

Ordinary differentiable Torch modules are valid backend operators. The example's
linear and fully connected product use them. Its convolution, contraction and
reduction demonstrate `torch.library.custom_op`, fake and autograd registrations
for an opaque operator boundary.

A fake implementation derives output shapes from inputs and host metadata such
as `num_nodes`; it must avoid reading device values. A backend declaring double
backward must make its backward differentiable. Forces/stress need input
derivatives, and training on them needs second derivatives reaching weights.
Add kernel-specific `gradcheck` and `gradgradcheck` cases where appropriate.

Resolve dispatch, precision, packing and edge order before execution. GPU kernels
use the framework's current stream and keep inputs on device. Avoid registry
calls, host transfers, global dtype/product switches and edge sorting inside the
hot path. JAX kernels need their own tracing and differentiation integration;
Torch registrations and conformance do not validate them.

## Fusion APIs

The [performance contracts](performance_contracts.md) specify member ordering,
density normalization, precision and distributed boundaries in detail.

| Contract | Weight lifetime | Factory result |
| --- | --- | --- |
| `make_interaction_layer(kernel_specs)` | Live member tensors supplied on every call; supports training when declared | Stateless module or `None` |
| Optional `prepare_inference(spec, weights)` | Independent snapshot prepared from canonical weights | `InferencePreparation(operation, tensors)` or `None` |

### Live-weight edge spans

`mace_core.kernels.spans.EdgeConvolutionSpan` matches radial basis, spherical
harmonics, optional distance transform, radial MLP, optional density MLP and
convolution. Use `EdgeConvolutionSpan.from_specs(kernel_specs)` to match a tuple;
return `None` for an unsupported tuple, including the empty tuple.

The returned Torch module has no parameters or buffers. Its call is
`forward(node_features, edges, sender, receiver, num_nodes, member_weights,
edge_runs)`, with `edges: EdgeInputs`, a tuple of live member tensors and
`edge_runs: EdgeRuns | None`. Member `span_weights()` exposes differentiable
tensors; checkpoint `canonical_state()` detaches tensors and is unsuitable here.
Declare `SpanCapabilities(double_backward=..., weight_gradients=...)` from
`mace_torch.backends.spans`; `validate_span` enforces statelessness.

The result is the message tensor, or `(message, density)` with density of shape
`[num_nodes, 1]`. Request edge views through `capabilities.edge_features`
(`receiver_runs`, `sender_runs`, `sorted_by_sender`); the caller prepares them
outside the forward. Both run views can coexist on `EdgeInputs`. Paths requiring
unsupported weight gradients or second derivatives retain the member operations.

### Prepared inference

The optional Torch `InferenceBackend` extension lives in
`mace_torch.backends.inference`; it adds no required method to `KernelBackend`.
Its descriptors in `mace_core.kernels.inference` are `ProductReadoutSpec` and
`InteractionFusionSpec`. The former describes contraction, mixing, optional
residual and optional linear readout; it currently requires the reduced basis.
The latter covers supported plain/residual/density interactions plus that product.

The public helper `prepare_inference(backend, spec, weights)` returns a
`PreparedInference` or `None`. `weights` maps `spec.weight_roles` to each member's
`canonical_state`; the helper gives the backend detached copies. The returned
operation is stateless, declares `weight_gradients=False`, and receives
`(inputs, tensors)`. The helper owns copies of derived tensors as nonpersistent
buffers; the caller invokes `prepared(inputs)` with tuples ordered by
`spec.input_names` and receives `spec.output_names`.

Preparation happens outside execution. Geometry and topology remain live;
runtime input autograd must remain enabled for forces and stress. Reprepare
after changing weights, device, dtype or specifications. The prepared snapshot
refuses training, device/dtype conversion and canonical checkpoint saving; save
and train the source model. Missing/declined preparation retains original ops,
while errors propagate. Existing builders and native exporters do not activate
this API automatically; its caller must integrate selection and fallback.

## Verification and coverage

Run the example's tests locally:

```sh
python -m pytest packages/mace-backend-example/tests
```

For individual operators, use the installed backend and inspect built cases:

```python
from mace_torch.backends.conformance import run_backend_conformance

results = run_backend_conformance(
    "example", device="cpu", precision="float64",
    activation_layout="mul_ir", compile_ops=True,
)
assert results and all(result.built for result in results)
```

The default seven cases cover the five dispatched op classes, including three
linear variants. The helper checks reference values, input and weight gradients,
canonical round-trips, equivariance where applicable and capability honesty.
`cases=[...]` customizes specifications for these five classes. Optional radial
and harmonic overrides need separate tests. FP64 cases run `gradgradcheck` only
when `supports_double_backward` is declared.

`compile_ops=True` enables full-graph compilation and size-change checks. CUDA
capture is separately enabled with `device="cuda", cuda_graphs=True` under a GPU
capability gate. Capture checks forward replay; model derivatives still need
integration tests. Exercise each supported dtype/layout/device combination and
representative unsupported shapes. For partial backends, assert the intended
subset of `result.built` values so an all-declined run cannot pass unnoticed.

Fusion uses separate helpers:

- `mace_torch.backends.span_conformance.run_span_conformance(backend_name,
  device="cpu", precision="float64", activation_layout="mul_ir", training=False,
  compile_spans=False, cuda_graphs=False)` exercises the built-in edge span cases.
  It has no `cases` argument. Inspect built results and provide dedicated tests
  for additional descriptors or shapes.
- `mace_torch.backends.inference_conformance.run_inference_conformance(prepared,
  reference, cases, precision="float64")` takes a nonempty sequence of input
  tuples and an independent reference callable. It checks values, input
  derivatives, declared second derivatives and immutable prepared buffers/inputs.
  Equivariance, compile/export, native integration and CUDA capture need separate
  tests for the concrete implementation.

At model level, check canonical checkpoint round-trips, energy/forces/stress and
force-loss parameter gradients for the affected paths. The example package
contains these checks. Preserve frozen references and numerical tolerances.
Electrostatics solvers use a separate per-framework registry; a non-bit-identical
solver change affects model state and cannot be hidden by kernel fallback.
