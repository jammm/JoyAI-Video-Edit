/*
 * gfx950 row-wise dynamic OCP FP8 e4m3 quantization.
 *
 * input:    [M, K] bf16/fp16, contiguous
 * output_q: [M, K] torch.float8_e4m3fn, contiguous
 * output_s: [M] or [M, 1] float32, contiguous
 *
 * One block owns a row.  Four wave64s reduce the absolute maximum, publish
 * one scale, then quantize the row.  The launch geometry is intentionally
 * fixed for the JoyOmni K=4096/16384 projections: M supplies thousands of
 * independent blocks, so extra blocks are more useful than larger CTAs.
 */
#include <ATen/hip/HIPContext.h>
#include <c10/hip/HIPGuard.h>
#include <c10/hip/HIPStream.h>
#include <hip/hip_bf16.h>
#include <hip/hip_fp16.h>
#include <hip/hip_fp8.h>

#include "joyomni_ops_hip.h"

namespace joyomni_ops {
namespace {

constexpr float kFp8Max = 448.0f;

__device__ __forceinline__ float wave_max(float value) {
#pragma unroll
  for (int offset = warpSize / 2; offset > 0; offset >>= 1) {
    value = fmaxf(value, __shfl_down(value, offset));
  }
  return value;
}

__device__ __forceinline__ float block_max(float value) {
  __shared__ float partial[16];
  const int lane = threadIdx.x % warpSize;
  const int wave = threadIdx.x / warpSize;
  value = wave_max(value);
  if (lane == 0) partial[wave] = value;
  __syncthreads();
  if (wave == 0) {
    const int waves = (blockDim.x + warpSize - 1) / warpSize;
    value = lane < waves ? partial[lane] : 0.0f;
    value = wave_max(value);
  }
  return value;  // valid on thread 0
}

template <typename T>
__global__ void per_token_quant(
    const T* __restrict__ input,
    __hip_fp8_e4m3* __restrict__ output,
    float* __restrict__ scales,
    int64_t hidden_dim) {
  const int64_t row = blockIdx.x;
  const T* in = input + row * hidden_dim;
  __hip_fp8_e4m3* out = output + row * hidden_dim;

  float local_max = 0.0f;
  for (int64_t col = threadIdx.x; col < hidden_dim; col += blockDim.x) {
    local_max = fmaxf(local_max, fabsf(static_cast<float>(in[col])));
  }
  local_max = block_max(local_max);

  __shared__ float row_scale;
  if (threadIdx.x == 0) {
    row_scale = fmaxf(local_max, 1.0e-8f) / kFp8Max;
    scales[row] = row_scale;
  }
  __syncthreads();
  const float inv_scale = 1.0f / row_scale;

  for (int64_t col = threadIdx.x; col < hidden_dim; col += blockDim.x) {
    float value = static_cast<float>(in[col]) * inv_scale;
    value = fmaxf(-kFp8Max, fminf(kFp8Max, value));
    out[col] = __hip_fp8_e4m3(value);
  }
}

template <typename T>
void launch(const torch::Tensor& input, torch::Tensor& output_q, torch::Tensor& output_s) {
  const int64_t rows = input.size(0);
  const int64_t hidden_dim = input.size(1);
  auto stream = at::cuda::getCurrentCUDAStream();
  const unsigned threads = hidden_dim >= 8192 ? 512 : 256;
  hipLaunchKernelGGL(
      (per_token_quant<T>), dim3((unsigned)rows), dim3(threads), 0, stream,
      static_cast<const T*>(input.data_ptr()),
      static_cast<__hip_fp8_e4m3*>(output_q.data_ptr()),
      static_cast<float*>(output_s.data_ptr()), hidden_dim);
}

}  // namespace

void sgl_per_token_quant_fp8(
    torch::Tensor input, torch::Tensor output_q, torch::Tensor output_s) {
  JO_CHECK_CUDA(input);
  JO_CHECK_CUDA(output_q);
  JO_CHECK_CUDA(output_s);
  JO_CHECK_CONTIGUOUS(input);
  JO_CHECK_CONTIGUOUS(output_q);
  JO_CHECK_CONTIGUOUS(output_s);
  TORCH_CHECK(input.dim() == 2, "input must be [M, K]");
  TORCH_CHECK(output_q.sizes() == input.sizes(), "output_q shape must match input");
  TORCH_CHECK(output_q.scalar_type() == c10::ScalarType::Float8_e4m3fn,
              "output_q must be torch.float8_e4m3fn");
  TORCH_CHECK(output_s.scalar_type() == torch::kFloat32,
              "output_s must be float32");
  TORCH_CHECK(output_s.numel() == input.size(0),
              "output_s must contain one scale per row");
  TORCH_CHECK(input.device() == output_q.device() && input.device() == output_s.device(),
              "input and outputs must be on the same device");

  const c10::cuda::CUDAGuard guard(input.device());
  if (input.scalar_type() == torch::kBFloat16) {
    launch<__hip_bfloat16>(input, output_q, output_s);
  } else if (input.scalar_type() == torch::kFloat16) {
    launch<__half>(input, output_q, output_s);
  } else {
    TORCH_CHECK(false, "input must be bfloat16 or float16");
  }
}

}  // namespace joyomni_ops
