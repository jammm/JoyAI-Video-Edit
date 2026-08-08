/*
 * Pack the image, text, and cached Q/K/V segments used by JoyOmni joint
 * attention into contiguous tensors.  Text Q/K RMSNorm is fused into the
 * copy, replacing two multi-kernel PyTorch norms plus five torch.cat kernels
 * in every DiT block.
 *
 * The streaming fast path is batch=1, head_dim=128.  One native wave owns one
 * head row: wave64 lanes copy one bf16 pair each on gfx950, while CUDA warp32
 * lanes copy two pairs.  Strided V views from the QKV projection are accepted.
 */
#include <ATen/cuda/Exceptions.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAStream.h>
#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <torch/all.h>

#include <tuple>

#include "joyomni_ops.h"

namespace joyomni_ops {
namespace {

__device__ __forceinline__ float wave_sum(float value) {
#if defined(__HIP_PLATFORM_AMD__)
#pragma unroll
  for (int offset = warpSize / 2; offset > 0; offset >>= 1) {
    value += __shfl_xor(value, offset);
  }
#else
#pragma unroll
  for (int offset = 16; offset > 0; offset >>= 1) {
    value += __shfl_xor_sync(0xffffffffu, value, offset, 32);
  }
#endif
  return value;
}

template <int HeadDim>
__device__ __forceinline__ const __nv_bfloat16* row_ptr(
    const __nv_bfloat16* base, int64_t row, int heads,
    int64_t stride_l, int64_t stride_h) {
  const int64_t token = row / heads;
  const int64_t head = row - token * heads;
  return base + token * stride_l + head * stride_h;
}

template <int HeadDim>
__device__ __forceinline__ void copy_row(
    __nv_bfloat16* dst, const __nv_bfloat16* src, int lane) {
#pragma unroll
  for (int dim = lane; dim < HeadDim; dim += warpSize) {
    dst[dim] = src[dim];
  }
}

template <int HeadDim>
__device__ __forceinline__ void rmsnorm_copy_row(
    __nv_bfloat16* dst, const __nv_bfloat16* src,
    const __nv_bfloat16* weight, float eps, int lane) {
  float sum_sq = 0.0f;
#pragma unroll
  for (int dim = lane; dim < HeadDim; dim += warpSize) {
    const float value = static_cast<float>(src[dim]);
    sum_sq += value * value;
  }
  sum_sq = wave_sum(sum_sq);
  const float inv_rms = rsqrtf(sum_sq / static_cast<float>(HeadDim) + eps);
#pragma unroll
  for (int dim = lane; dim < HeadDim; dim += warpSize) {
    const float value = static_cast<float>(src[dim]);
    // Match the model's RMSNorm implementation: normalize and cast back to
    // bf16 first, then apply the bf16 affine weight (a second rounding).
    const __nv_bfloat16 normalized = __nv_bfloat16(value * inv_rms);
    dst[dim] = __nv_bfloat16(
        static_cast<float>(normalized) * static_cast<float>(weight[dim]));
  }
}

template <int HeadDim>
__global__ void fusedJointQKVPack(
    const __nv_bfloat16* __restrict__ img_q,
    const __nv_bfloat16* __restrict__ img_k,
    const __nv_bfloat16* __restrict__ img_v,
    const __nv_bfloat16* __restrict__ txt_q,
    const __nv_bfloat16* __restrict__ txt_k,
    const __nv_bfloat16* __restrict__ txt_v,
    const __nv_bfloat16* __restrict__ cached_k,
    const __nv_bfloat16* __restrict__ cached_v,
    const __nv_bfloat16* __restrict__ txt_q_weight,
    const __nv_bfloat16* __restrict__ txt_k_weight,
    __nv_bfloat16* __restrict__ out_q,
    __nv_bfloat16* __restrict__ out_k,
    __nv_bfloat16* __restrict__ out_v,
    int img_len, int txt_len, int cache_len, int heads, float eps,
    int64_t img_q_sl, int64_t img_q_sh,
    int64_t img_k_sl, int64_t img_k_sh,
    int64_t img_v_sl, int64_t img_v_sh,
    int64_t txt_q_sl, int64_t txt_q_sh,
    int64_t txt_k_sl, int64_t txt_k_sh,
    int64_t txt_v_sl, int64_t txt_v_sh,
    int64_t cache_k_sl, int64_t cache_k_sh,
    int64_t cache_v_sl, int64_t cache_v_sh) {
  const int waves_per_block = blockDim.x / warpSize;
  const int wave = threadIdx.x / warpSize;
  const int lane = threadIdx.x % warpSize;
  const int64_t row =
      static_cast<int64_t>(blockIdx.x) * waves_per_block + wave;

  const int64_t img_rows = static_cast<int64_t>(img_len) * heads;
  const int64_t txt_rows = static_cast<int64_t>(txt_len) * heads;
  const int64_t cache_rows = static_cast<int64_t>(cache_len) * heads;
  const int64_t q_rows = img_rows + txt_rows;
  const int64_t kv_rows = cache_rows + img_rows + txt_rows;
  const int64_t total_rows = q_rows + 2 * kv_rows;
  if (row >= total_rows) return;

  if (row < q_rows) {
    __nv_bfloat16* dst = out_q + row * HeadDim;
    if (row < img_rows) {
      copy_row<HeadDim>(
          dst, row_ptr<HeadDim>(img_q, row, heads, img_q_sl, img_q_sh), lane);
    } else {
      const int64_t txt_row = row - img_rows;
      rmsnorm_copy_row<HeadDim>(
          dst, row_ptr<HeadDim>(txt_q, txt_row, heads, txt_q_sl, txt_q_sh),
          txt_q_weight, eps, lane);
    }
    return;
  }

  const int64_t kv_row = (row - q_rows) % kv_rows;
  const bool is_value = row >= q_rows + kv_rows;
  __nv_bfloat16* dst =
      (is_value ? out_v : out_k) + kv_row * HeadDim;

  if (kv_row < cache_rows) {
    const __nv_bfloat16* src = is_value
        ? row_ptr<HeadDim>(cached_v, kv_row, heads, cache_v_sl, cache_v_sh)
        : row_ptr<HeadDim>(cached_k, kv_row, heads, cache_k_sl, cache_k_sh);
    copy_row<HeadDim>(dst, src, lane);
    return;
  }

  const int64_t current_row = kv_row - cache_rows;
  if (current_row < img_rows) {
    const __nv_bfloat16* src = is_value
        ? row_ptr<HeadDim>(img_v, current_row, heads, img_v_sl, img_v_sh)
        : row_ptr<HeadDim>(img_k, current_row, heads, img_k_sl, img_k_sh);
    copy_row<HeadDim>(dst, src, lane);
    return;
  }

  const int64_t txt_row = current_row - img_rows;
  if (is_value) {
    copy_row<HeadDim>(
        dst, row_ptr<HeadDim>(txt_v, txt_row, heads, txt_v_sl, txt_v_sh), lane);
  } else {
    rmsnorm_copy_row<HeadDim>(
        dst, row_ptr<HeadDim>(txt_k, txt_row, heads, txt_k_sl, txt_k_sh),
        txt_k_weight, eps, lane);
  }
}

template <int HeadDim>
__global__ void fusedCachedKVPack(
    const __nv_bfloat16* __restrict__ key0,
    const __nv_bfloat16* __restrict__ value0,
    const __nv_bfloat16* __restrict__ key1,
    const __nv_bfloat16* __restrict__ value1,
    const float* __restrict__ cos,
    const float* __restrict__ sin,
    __nv_bfloat16* __restrict__ out_key,
    __nv_bfloat16* __restrict__ out_value,
    int len0, int len1, int heads,
    int64_t key0_sl, int64_t key0_sh,
    int64_t value0_sl, int64_t value0_sh,
    int64_t key1_sl, int64_t key1_sh,
    int64_t value1_sl, int64_t value1_sh) {
  const int waves_per_block = blockDim.x / warpSize;
  const int wave = threadIdx.x / warpSize;
  const int lane = threadIdx.x % warpSize;
  const int64_t row =
      static_cast<int64_t>(blockIdx.x) * waves_per_block + wave;
  const int64_t rows0 = static_cast<int64_t>(len0) * heads;
  const int64_t rows1 = static_cast<int64_t>(len1) * heads;
  const int64_t total_rows = rows0 + rows1;
  if (row >= 2 * total_rows) return;

  const bool is_value = row >= total_rows;
  const int64_t kv_row = row % total_rows;
  const bool use_second = kv_row >= rows0;
  const int64_t source_row = use_second ? kv_row - rows0 : kv_row;
  const __nv_bfloat16* source = nullptr;
  if (is_value) {
    source = use_second
        ? row_ptr<HeadDim>(value1, source_row, heads, value1_sl, value1_sh)
        : row_ptr<HeadDim>(value0, source_row, heads, value0_sl, value0_sh);
    copy_row<HeadDim>(out_value + kv_row * HeadDim, source, lane);
    return;
  }

  source = use_second
      ? row_ptr<HeadDim>(key1, source_row, heads, key1_sl, key1_sh)
      : row_ptr<HeadDim>(key0, source_row, heads, key0_sl, key0_sh);
  __nv_bfloat16* destination = out_key + kv_row * HeadDim;
  const int64_t token = kv_row / heads;
  const int64_t freq_offset = token * HeadDim;
#pragma unroll
  for (int pair = lane; pair < HeadDim / 2; pair += warpSize) {
    const int dim = pair * 2;
    const float x0 = static_cast<float>(source[dim]);
    const float x1 = static_cast<float>(source[dim + 1]);
    const float c = cos[freq_offset + dim];
    const float s = sin[freq_offset + dim];
    destination[dim] = __nv_bfloat16(x0 * c - x1 * s);
    destination[dim + 1] = __nv_bfloat16(x0 * s + x1 * c);
  }
}

void check_qkv_tensor(
    const torch::Tensor& tensor, const char* name, int64_t batch,
    int64_t heads, int64_t head_dim) {
  TORCH_CHECK(tensor.is_cuda(), name, " must be a CUDA tensor");
  TORCH_CHECK(tensor.scalar_type() == torch::kBFloat16,
              name, " must be bfloat16");
  TORCH_CHECK(tensor.dim() == 4, name, " must be [B,L,H,D]");
  TORCH_CHECK(tensor.size(0) == batch && tensor.size(2) == heads &&
                  tensor.size(3) == head_dim,
              name, " has incompatible shape ", tensor.sizes());
  TORCH_CHECK(tensor.stride(3) == 1, name, " last dimension must be contiguous");
}

}  // namespace

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor> fused_joint_qkv_pack(
    const torch::Tensor& img_q, const torch::Tensor& img_k,
    const torch::Tensor& img_v, const torch::Tensor& txt_q,
    const torch::Tensor& txt_k, const torch::Tensor& txt_v,
    const c10::optional<torch::Tensor>& cached_k_opt,
    const c10::optional<torch::Tensor>& cached_v_opt,
    const torch::Tensor& txt_q_weight, const torch::Tensor& txt_k_weight,
    double eps) {
  TORCH_CHECK(img_q.dim() == 4, "img_q must be [B,L,H,D]");
  const int64_t batch = img_q.size(0);
  const int64_t img_len = img_q.size(1);
  const int64_t heads = img_q.size(2);
  const int64_t head_dim = img_q.size(3);
  TORCH_CHECK(batch == 1, "fused_joint_qkv_pack fast path requires batch=1");
  TORCH_CHECK(head_dim == 128, "fused_joint_qkv_pack requires head_dim=128");
  TORCH_CHECK(img_len > 0 && heads > 0, "image length and heads must be positive");

  check_qkv_tensor(img_q, "img_q", batch, heads, head_dim);
  check_qkv_tensor(img_k, "img_k", batch, heads, head_dim);
  check_qkv_tensor(img_v, "img_v", batch, heads, head_dim);
  TORCH_CHECK(img_k.size(1) == img_len && img_v.size(1) == img_len,
              "image Q/K/V lengths must match");

  const int64_t txt_len = txt_q.size(1);
  check_qkv_tensor(txt_q, "txt_q", batch, heads, head_dim);
  check_qkv_tensor(txt_k, "txt_k", batch, heads, head_dim);
  check_qkv_tensor(txt_v, "txt_v", batch, heads, head_dim);
  TORCH_CHECK(txt_len > 0 && txt_k.size(1) == txt_len && txt_v.size(1) == txt_len,
              "text Q/K/V lengths must match and be positive");

  JO_CHECK_INPUT(txt_q_weight, torch::kBFloat16);
  JO_CHECK_INPUT(txt_k_weight, torch::kBFloat16);
  TORCH_CHECK(txt_q_weight.dim() == 1 && txt_q_weight.numel() == head_dim,
              "txt_q_weight must be [head_dim]");
  TORCH_CHECK(txt_k_weight.dim() == 1 && txt_k_weight.numel() == head_dim,
              "txt_k_weight must be [head_dim]");

  TORCH_CHECK(cached_k_opt.has_value() == cached_v_opt.has_value(),
              "cached_k and cached_v must either both be present or both be absent");
  int64_t cache_len = 0;
  const torch::Tensor* cached_k = nullptr;
  const torch::Tensor* cached_v = nullptr;
  if (cached_k_opt.has_value()) {
    cached_k = &cached_k_opt.value();
    cached_v = &cached_v_opt.value();
    check_qkv_tensor(*cached_k, "cached_k", batch, heads, head_dim);
    check_qkv_tensor(*cached_v, "cached_v", batch, heads, head_dim);
    cache_len = cached_k->size(1);
    TORCH_CHECK(cached_v->size(1) == cache_len, "cached K/V lengths must match");
  }

  const c10::cuda::CUDAGuard guard(img_q.device());
  const auto options = img_q.options();
  auto out_q = torch::empty({batch, img_len + txt_len, heads, head_dim}, options);
  auto out_k = torch::empty(
      {batch, cache_len + img_len + txt_len, heads, head_dim}, options);
  auto out_v = torch::empty_like(out_k);

  const __nv_bfloat16* cache_k_ptr = cached_k
      ? reinterpret_cast<const __nv_bfloat16*>(cached_k->data_ptr()) : nullptr;
  const __nv_bfloat16* cache_v_ptr = cached_v
      ? reinterpret_cast<const __nv_bfloat16*>(cached_v->data_ptr()) : nullptr;

  constexpr int block_size = 256;
#if defined(__HIP_PLATFORM_AMD__)
  constexpr int native_wave_size = 64;
#else
  constexpr int native_wave_size = 32;
#endif
  constexpr int waves_per_block = block_size / native_wave_size;
  const int64_t q_rows = (img_len + txt_len) * heads;
  const int64_t kv_rows = (cache_len + img_len + txt_len) * heads;
  const int64_t total_rows = q_rows + 2 * kv_rows;
  const int blocks = static_cast<int>((total_rows + waves_per_block - 1) /
                                      waves_per_block);
  auto stream = at::cuda::getCurrentCUDAStream(img_q.get_device());

  fusedJointQKVPack<128><<<blocks, block_size, 0, stream>>>(
      reinterpret_cast<const __nv_bfloat16*>(img_q.data_ptr()),
      reinterpret_cast<const __nv_bfloat16*>(img_k.data_ptr()),
      reinterpret_cast<const __nv_bfloat16*>(img_v.data_ptr()),
      reinterpret_cast<const __nv_bfloat16*>(txt_q.data_ptr()),
      reinterpret_cast<const __nv_bfloat16*>(txt_k.data_ptr()),
      reinterpret_cast<const __nv_bfloat16*>(txt_v.data_ptr()),
      cache_k_ptr, cache_v_ptr,
      reinterpret_cast<const __nv_bfloat16*>(txt_q_weight.data_ptr()),
      reinterpret_cast<const __nv_bfloat16*>(txt_k_weight.data_ptr()),
      reinterpret_cast<__nv_bfloat16*>(out_q.data_ptr()),
      reinterpret_cast<__nv_bfloat16*>(out_k.data_ptr()),
      reinterpret_cast<__nv_bfloat16*>(out_v.data_ptr()),
      static_cast<int>(img_len), static_cast<int>(txt_len),
      static_cast<int>(cache_len), static_cast<int>(heads),
      static_cast<float>(eps),
      img_q.stride(1), img_q.stride(2),
      img_k.stride(1), img_k.stride(2),
      img_v.stride(1), img_v.stride(2),
      txt_q.stride(1), txt_q.stride(2),
      txt_k.stride(1), txt_k.stride(2),
      txt_v.stride(1), txt_v.stride(2),
      cached_k ? cached_k->stride(1) : 0,
      cached_k ? cached_k->stride(2) : 0,
      cached_v ? cached_v->stride(1) : 0,
      cached_v ? cached_v->stride(2) : 0);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return std::make_tuple(out_q, out_k, out_v);
}

std::tuple<torch::Tensor, torch::Tensor> fused_cached_kv_pack(
    const torch::Tensor& key0, const torch::Tensor& value0,
    const c10::optional<torch::Tensor>& key1_opt,
    const c10::optional<torch::Tensor>& value1_opt,
    const torch::Tensor& cos, const torch::Tensor& sin) {
  TORCH_CHECK(key0.dim() == 4, "key0 must be [B,L,H,D]");
  const int64_t batch = key0.size(0);
  const int64_t len0 = key0.size(1);
  const int64_t heads = key0.size(2);
  const int64_t head_dim = key0.size(3);
  TORCH_CHECK(batch == 1, "fused_cached_kv_pack requires batch=1");
  TORCH_CHECK(head_dim == 128, "fused_cached_kv_pack requires head_dim=128");
  TORCH_CHECK(len0 > 0 && heads > 0, "cache length and heads must be positive");
  check_qkv_tensor(key0, "key0", batch, heads, head_dim);
  check_qkv_tensor(value0, "value0", batch, heads, head_dim);
  TORCH_CHECK(value0.size(1) == len0, "key0/value0 lengths must match");

  TORCH_CHECK(key1_opt.has_value() == value1_opt.has_value(),
              "key1 and value1 must either both be present or both be absent");
  int64_t len1 = 0;
  const torch::Tensor* key1 = nullptr;
  const torch::Tensor* value1 = nullptr;
  if (key1_opt.has_value()) {
    key1 = &key1_opt.value();
    value1 = &value1_opt.value();
    check_qkv_tensor(*key1, "key1", batch, heads, head_dim);
    check_qkv_tensor(*value1, "value1", batch, heads, head_dim);
    len1 = key1->size(1);
    TORCH_CHECK(len1 > 0 && value1->size(1) == len1,
                "key1/value1 lengths must match and be positive");
  }

  JO_CHECK_INPUT(cos, torch::kFloat32);
  JO_CHECK_INPUT(sin, torch::kFloat32);
  TORCH_CHECK(cos.sizes() == sin.sizes(), "cos/sin shapes must match");
  TORCH_CHECK(
      cos.numel() == (len0 + len1) * head_dim,
      "cos/sin must contain [total_cache_len, head_dim], got ", cos.sizes());

  const c10::cuda::CUDAGuard guard(key0.device());
  auto out_key = torch::empty(
      {batch, len0 + len1, heads, head_dim}, key0.options());
  auto out_value = torch::empty_like(out_key);

  constexpr int block_size = 256;
#if defined(__HIP_PLATFORM_AMD__)
  constexpr int native_wave_size = 64;
#else
  constexpr int native_wave_size = 32;
#endif
  constexpr int waves_per_block = block_size / native_wave_size;
  const int64_t total_rows = (len0 + len1) * heads * 2;
  const int blocks = static_cast<int>(
      (total_rows + waves_per_block - 1) / waves_per_block);
  auto stream = at::cuda::getCurrentCUDAStream(key0.get_device());

  fusedCachedKVPack<128><<<blocks, block_size, 0, stream>>>(
      reinterpret_cast<const __nv_bfloat16*>(key0.data_ptr()),
      reinterpret_cast<const __nv_bfloat16*>(value0.data_ptr()),
      key1 ? reinterpret_cast<const __nv_bfloat16*>(key1->data_ptr()) : nullptr,
      value1 ? reinterpret_cast<const __nv_bfloat16*>(value1->data_ptr()) : nullptr,
      reinterpret_cast<const float*>(cos.data_ptr()),
      reinterpret_cast<const float*>(sin.data_ptr()),
      reinterpret_cast<__nv_bfloat16*>(out_key.data_ptr()),
      reinterpret_cast<__nv_bfloat16*>(out_value.data_ptr()),
      static_cast<int>(len0), static_cast<int>(len1), static_cast<int>(heads),
      key0.stride(1), key0.stride(2), value0.stride(1), value0.stride(2),
      key1 ? key1->stride(1) : 0, key1 ? key1->stride(2) : 0,
      value1 ? value1->stride(1) : 0, value1 ? value1->stride(2) : 0);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return std::make_tuple(out_key, out_value);
}

}  // namespace joyomni_ops
