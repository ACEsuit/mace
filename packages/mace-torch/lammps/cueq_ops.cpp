// Native bindings to NVIDIA's cuEquivariance 0.10.0 CUDA library (libcue_ops).
// Layouts and operand ordering follow the vendor's uniform_1d,
// fused_tensor_product and segmented_transpose torch interfaces. Python is
// used at build time to read their exact schemas, never by the pair style.
#include <ATen/ATen.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAStream.h>
#include <equivariance/fused_tensor_product.cuh>
#include <equivariance/segmented_transpose.cuh>
#include <equivariance/uniform1d/api.hh>
#include <torch/library.h>

namespace {
namespace args {
#include "cueq_args.inc"
}
using at::Tensor;
using Stack = torch::jit::Stack;
using Op = c10::OperatorHandle;
namespace tp = kernelcatcher::tensor_product;
namespace u1 = kernelcatcher::equivariance::uniform_1d;
using kernelcatcher::utils::BatchDimension;
using kernelcatcher::utils::Datatype;

Datatype vendor_type(at::ScalarType type) {
  switch (type) {
  case at::kFloat:
    return Datatype::kFloat32;
  case at::kDouble:
    return Datatype::kFloat64;
  case at::kInt:
    return Datatype::kInt32;
  case at::kLong:
    return Datatype::kInt64;
  default:
    TORCH_CHECK(false, "unsupported cuEq native dtype ", type);
  }
}

std::vector<int> ints(const c10::IValue &arg) {
  auto values = arg.toIntVector();
  return std::vector<int>(values.begin(), values.end());
}

void uniform(const Op &, Stack *stack) {
  using A = args::uniform_1d;
  auto &s = *stack;
  auto input = s[A::tensors].toTensorVector();
  c10::cuda::CUDAGuard guard(input.at(0).device());
  int extent = s[A::operand_extent].toInt(), inputs = s[A::num_inputs].toInt(),
      results = s[A::num_outputs].toInt(),
      index_count = s[A::num_index].toInt();
  auto shape = ints(s[A::buffer_dim]),
       segments = ints(s[A::buffer_num_segments]),
       layout = ints(s[A::batch_dim]), index_map = ints(s[A::index_buffer]),
       types = ints(s[A::dtypes]);
  int batch = s[A::batch_size].toInt();
  auto resolve_size = [&](int n) {
    if (batch == -2)
      batch = n;
    TORCH_CHECK(n == batch, "cuEq incompatible batch sizes");
  };
  for (int i = inputs; i < inputs + index_count; ++i)
    resolve_size(input.at(i).size(0));
  for (int i = 0; i < inputs; ++i) {
    int n = input.at(i).size(0);
    if (layout[i] == -2)
      layout[i] = (n == 1 ? 0 : 1);
    if (layout[i] == 1)
      resolve_size(n);
  }
  if (batch == -2)
    batch = 1;
  for (int &mode : layout) {
    if (mode == -2 || (batch == 1 && mode == 0))
      mode = 1;
  }
  std::vector<std::vector<BatchDimension>> batches;
  for (int mode : layout)
    batches.push_back({mode == 1   ? BatchDimension::kBatched
                       : mode == 0 ? BatchDimension::kShared
                                   : BatchDimension::kIndexed});
  std::vector<u1::Dimension> dimensions;
  for (int mode : shape)
    dimensions.push_back(static_cast<u1::Dimension>(mode));
  std::vector<std::vector<int>> indices;
  for (int i = 0; i < inputs + results; ++i)
    indices.push_back({index_map.at(i)});
  // Each index buffer has one entry per batch item. Its own index is -1.
  for (int i = 0; i < index_count; ++i) {
    batches.push_back({BatchDimension::kBatched});
    indices.push_back({-1});
  }
  std::vector<int> index_sizes(index_map.begin() + inputs + results,
                               index_map.end());
  std::vector<Tensor> outputs, buffers;
  for (int i = 0; i < inputs; ++i)
    buffers.push_back(input[i].contiguous());
  for (int i = inputs; i < inputs + results; ++i) {
    auto dtype = types[i] < 0 ? s[A::math_dtype].toScalarType()
                              : input.at(types[i]).scalar_type();
    int rows = layout[i] == 1   ? batch
               : layout[i] == 0 ? 1
                                : index_sizes.at(index_map[i]);
    auto tensor = at::empty({rows, segments[i] * (shape[i] ? extent : 1)},
                            input[0].options().dtype(dtype));
    if (layout[i] != 1)
      tensor.zero_(); // scatter/reduced outputs accumulate
    outputs.push_back(tensor);
    buffers.push_back(tensor);
  }
  for (int i = inputs; i < inputs + index_count; ++i)
    buffers.push_back(input[i].contiguous());
  std::vector<Datatype> dtypes;
  std::vector<void *> pointers;
  std::vector<size_t> bytes;
  for (auto &tensor : buffers) {
    dtypes.push_back(vendor_type(tensor.scalar_type()));
    pointers.push_back(tensor.data_ptr());
    bytes.push_back(tensor.nbytes());
  }
  auto operand_ids = ints(s[A::operations]);
  std::vector<std::vector<int>> operations;
  size_t start = 0;
  for (int length : ints(s[A::num_operands])) {
    operations.emplace_back(operand_ids.begin() + start,
                            operand_ids.begin() + start + length);
    start += length;
  }
  u1::run_uniform_1d_cuda(
      s[A::name].toStringRef(), vendor_type(s[A::math_dtype].toScalarType()),
      extent, inputs, results, index_count, dimensions, segments, batches,
      indices, index_sizes, dtypes, operations, ints(s[A::num_paths]),
      ints(s[A::path_indices_start]), ints(s[A::path_coefficients_start]),
      ints(s[A::path_indices]), s[A::path_coefficients].toDoubleVector(),
      {batch}, pointers, bytes, false, true,
      c10::cuda::getCurrentCUDAStream().stream());
  s.clear();
  s.emplace_back(std::move(outputs));
}

void transpose(const Op &, Stack *stack) {
  using A = args::segmented_transpose;
  auto &s = *stack;
  auto x = s[A::tensor].toTensor().contiguous(),
       info = s[A::segment_info].toTensor().contiguous();
  c10::cuda::CUDAGuard guard(x.device());
  auto out = at::empty_like(x);
  AT_DISPATCH_FLOATING_TYPES(x.scalar_type(), "mace_cueq_transpose", [&] {
    tp::segmented_transpose(out.data_ptr<scalar_t>(), x.data_ptr<scalar_t>(),
                            info.data_ptr<int>(), info.size(0), x.size(0),
                            x.size(1), s[A::input_contiguous_as_info].toBool(),
                            c10::cuda::getCurrentCUDAStream().stream());
  });
  s.clear();
  s.emplace_back(out);
}

template <class M>
tp::tp_info<M> path_info(const Stack &s, int start, int which) {
  auto csr = s[start + which].toTensor(),
       offsets = s[start + 4 + which].toTensor(),
       cg = s[start + 8 + which].toTensor();
  return {csr.data_ptr<int>(), offsets.data_ptr<int>(),
          static_cast<const M *>(cg.const_data_ptr()),
          static_cast<int>(csr.numel() - 1), static_cast<int>(cg.numel())};
}

// The wheel's integer IDs describe the tensor-product polynomial's connection
// pattern.
tp::ConnectionModeT connection(int mode) {
  using C = tp::ConnectionModeT;
  static const std::vector<C> modes = {
      C::kUUU, C::kUVUV, C::kUUVV, C::kU1U1, C::k1V1V, C::kUVU, C::kU1W,
      C::kU1W, C::k1VW,  C::kUVU,  C::kUVV,  C::kUUW,  C::kUVW};
  return modes.at(mode);
}

template <bool backward>
using FusedArgs = std::conditional_t<backward, args::fused_tensor_product_bwd,
                                     args::fused_tensor_product>;

template <class T, class M, bool backward> void fused(Stack &s) {
  using A = FusedArgs<backward>;
  Tensor a = s[A::in0].toTensor().contiguous(),
         b = s[A::in1].toTensor().contiguous();
  c10::cuda::CUDAGuard guard(a.device());
  Tensor w = s[A::in2].isNone() ? Tensor() : s[A::in2].toTensor().contiguous();
  auto mode = connection(s[A::connection_mode].toInt());
  Tensor d =
      backward
          ? s[args::fused_tensor_product_bwd::grad_out].toTensor().contiguous()
          : Tensor();
  int stride = backward ? d.size(1)
                        : s[args::fused_tensor_product::output_stride].toInt();
  tp::tp_data_sizes sizes{
      std::max({a.size(0), b.size(0), w.defined() ? w.size(0) : int64_t(1)}),
      a.size(0) == 1,
      b.size(0) == 1,
      !w.defined() || w.size(0) == 1,
      static_cast<int>(a.size(1)),
      static_cast<int>(b.size(1)),
      w.defined() ? static_cast<int>(w.size(1)) : 0,
      stride};
  auto stream = c10::cuda::getCurrentCUDAStream().stream();
  const T *weight = w.defined() ? w.data_ptr<T>() : nullptr;
  if constexpr (!backward) {
    Tensor out = at::empty({sizes.batch_size, stride}, a.options());
    tp::fused_tensor_product_fwd<T, T, T, T, M>(
        out.template data_ptr<T>(), a.data_ptr<T>(), b.data_ptr<T>(), weight,
        mode, path_info<M>(s, A::tp_path_csr_offsets_fwd, 0), sizes, stream);
    s.clear();
    s.emplace_back(out);
  } else {
    std::vector<Tensor> grads;
    auto dest = [&](const Tensor &like, int flag) -> T * {
      if (!s[flag].toBool() || !like.defined())
        return nullptr;
      grads.push_back(at::zeros_like(like));
      return grads.back().data_ptr<T>();
    };
    T *ga = dest(a, A::needs_grad_in0), *gb = dest(b, A::needs_grad_in1),
      *gw = dest(w, A::needs_grad_in2);
    tp::fused_tensor_product_bwd<T, T, T, T, M>(
        ga, gb, gw, d.data_ptr<T>(), a.data_ptr<T>(), b.data_ptr<T>(), weight,
        mode, path_info<M>(s, A::tp_path_csr_offsets_fwd, 1),
        path_info<M>(s, A::tp_path_csr_offsets_fwd, 2),
        path_info<M>(s, A::tp_path_csr_offsets_fwd, 3), sizes, stream);
    s.clear();
    s.emplace_back(std::move(grads));
  }
}
template <bool backward> void dispatch_fused(Stack *stack) {
  auto &s = *stack;
  if (s[FusedArgs<backward>::in0].toTensor().scalar_type() == at::kDouble)
    fused<double, double, backward>(s);
  else if (s[backward ? static_cast<int>(args::fused_tensor_product_bwd::
                                             tp_path_cg_values_dgrad_in0)
                      : static_cast<int>(
                            args::fused_tensor_product::tp_path_cg_values_fwd)]
               .toTensor()
               .scalar_type() == at::kLong)
    fused<float, double, backward>(s);
  else
    fused<float, float, backward>(s);
}
void fused_forward(const Op &, Stack *s) { dispatch_fused<false>(s); }
void fused_backward(const Op &, Stack *s) { dispatch_fused<true>(s); }
TORCH_LIBRARY(cuequivariance, m) {
#include "cueq_schemas.inc"
}
} // namespace
