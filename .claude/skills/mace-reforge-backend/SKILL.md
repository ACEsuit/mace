---
name: mace-reforge-backend
description: Implement or review a MACE V1 kernel backend, including registration, capability negotiation, canonical weights, precision, live-weight spans, prepared inference and conformance. Use for kernel factories, dispatch, weight layout or Clebsch-Gordan basis changes.
---

# MACE V1 kernel backends

Read the [implementation guide](../../../docs/reforge/kernel_backends.md) for
the current API and runnable checks. Start from the separately registered
[example backend](../../../packages/mace-backend-example/README.md). For fusion,
read [performance contracts](../../../docs/reforge/performance_contracts.md).
These describe the API implemented in this V1 checkout; historical RFCs and
feature-plugin sketches may describe interfaces that were never implemented.

## Registration and dispatch

- A framework supplies tensors, differentiation and compilation. A backend
  supplies operators for that framework. Register a zero-argument factory in
  `mace.kernel_backends.torch` or `mace.kernel_backends.jax`.
- Runtime backend code depends on `mace_core`, its framework and its own kernels.
  Keep model and training implementation imports out of it. Torch conformance
  helpers belong in backend tests; they do not validate JAX implementations.
- `KernelBackend` is a structural, runtime-checkable protocol. Implement all
  its factory methods; optional overrides can return `None`. It is not a
  `Generic[TensorT]` class.
- Declare implemented ops, devices, dtypes, bases, layouts, products, derivative
  support and edge metadata in `BackendCapabilities`. For shape restrictions,
  override `unsupported_reason(spec)`, preserving the parent's checks.
- `get_backend` returns the raw registered backend. Torch's `resolve_backend`
  adds per-op reference fallback and optional precision handling at build time.
  Undeclared ops, unsupported specs and `None` use the reference. `NotImplemented`
  is not a decline sentinel. Factory exceptions propagate; a missing or broken
  selected backend is an error.
- Capabilities must agree with factories: declared supported specs build an op;
  direct requests for unsupported dispatched specs raise
  `UnsupportedKernelSpecError`. Partial acceleration is valid.
- Resolve operators and activation layout during construction. Keep registry
  lookup, packing, sorting and device-to-host reads out of `forward`.

## Operators, weights and numerics

- Ordinary differentiable Torch modules are valid. Opaque kernels need
  `torch.library.custom_op`, fake/meta and autograd registrations. Declaring
  double backward requires a differentiable backward; force/stress training
  needs this support. Use the framework's current GPU stream.
- Convolution returns receiver-summed node values, including isolated nodes.
  Its output follows `spec.irreps_paths`; its weights are external radial outputs.
- `INTERNAL_WEIGHT_OPS` includes `linear`, `symmetric_contraction`,
  `fully_connected_tp` and `radial_mlp`. Implement `to_canonical`,
  `load_canonical` and `initialize_weights(seed)` for these. Preserve canonical
  component names, constants and contraction path metadata.
- Checkpoints use canonical `mul_ir` weights independently of activation layout.
  Use `mace_core.kernels.canonical` for order, initialization and normalization.
  Linear/skip weights are raw; apply path normalization in the computation.
  Initialize with the per-op seed and canonical scale; zero-only weights are invalid.
- `reduced` is the default CG basis; V1 also accepts explicit `full`. Preserve
  the configured and saved basis. Installed dependencies cannot select it.
  Legacy neutral import explicitly projects to reduced in FP64; ordinary native
  loading does not silently perform that conversion. Prepared product regions
  currently require reduced. CG coefficients and path order come from `mace_core`.
- Honour resolved compute/accumulation precision and product policy. Without
  `wide_accumulation`, the precision wrapper promotes compute and weights to
  the accumulation floor. TF32 applies only to float32 linear and fully
  connected products; it requires a declared capability. Never change global
  dtype or matrix-product flags in execution.

## Two fusion lifetimes

- `make_interaction_layer(kernel_specs)` negotiates stateless spans. The standard
  descriptor is `EdgeConvolutionSpan`; unmatched tuples return `None`. The op
  owns no parameters/buffers and receives live member tensors. Use `span_weights`
  for differentiable tensors; checkpoint `canonical_state` detaches them.
  Declare `SpanCapabilities(double_backward=..., weight_gradients=...)`.
- Optional Torch `prepare_inference(spec, weights)` handles `ProductReadoutSpec`
  and `InteractionFusionSpec`. Return `InferencePreparation(operation, tensors)`
  or `None`. The public helper owns a detached snapshot of derived tensors.
  Reprepare after weight/device/dtype/spec changes. Keep geometry live and
  input derivatives available for forces; execution must retain autograd.
  Existing builders/exporters do not activate this API automatically.

## Verification

Run `run_backend_conformance` for supported device/dtype/layout combinations.
Inspect `built` results: a suite of declined cases proves no implementation.
The default cases cover the five dispatched op classes. Add dedicated tests
for optional radial/harmonic overrides and unsupported shapes. Compilation
requires `compile_ops=True`; CUDA capture requires `device="cuda"` and
`cuda_graphs=True` on a GPU runner. FP64 double-backward checks run only for
backends declaring that capability.

Use `run_span_conformance` for live spans and `run_inference_conformance` for
prepared regions; their signatures and coverage differ. Add model-level
energy/force/stress, canonical round-trip and force-loss weight-gradient checks
for the affected paths. Keep frozen references and tolerances unchanged.

Electrostatics has a separate per-framework registry and solver contract. A
solver change that is not bit-identical affects model state and must be recorded;
kernel fallback cannot silently replace the configured solver.
