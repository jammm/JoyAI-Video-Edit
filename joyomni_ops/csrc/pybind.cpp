// Single Torch library registration for joyomni_ops.
// Op names are exposed as torch.ops.joyomni_ops.<name> and mirror the sgl_kernel
// signatures the JoyOmni pipeline uses, so the Python shim is a thin rename.
//
// Define JOYOMNI_OPS_NO_FP8 to build without the cutlass FP8 GEMM (lets the 3
// light kernels compile on toolchains without cutlass / CUDA<12.8).
#include <torch/all.h>
#include <torch/library.h>

namespace joyomni_ops {

void fused_qk_norm_rope_3d_paired(
    torch::Tensor& q, torch::Tensor& k, int64_t seq_len, int64_t num_heads, double eps,
    torch::Tensor& q_weight, torch::Tensor& k_weight, torch::Tensor& cos, torch::Tensor& sin,
    const c10::optional<torch::Tensor>& k_pre_rope_opt);

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor> fused_joint_qkv_pack(
    const torch::Tensor& img_q, const torch::Tensor& img_k,
    const torch::Tensor& img_v, const torch::Tensor& txt_q,
    const torch::Tensor& txt_k, const torch::Tensor& txt_v,
    const c10::optional<torch::Tensor>& cached_k_opt,
    const c10::optional<torch::Tensor>& cached_v_opt,
    const torch::Tensor& txt_q_weight, const torch::Tensor& txt_k_weight,
    double eps);

std::tuple<torch::Tensor, torch::Tensor> fused_cached_kv_pack(
    const torch::Tensor& key0, const torch::Tensor& value0,
    const c10::optional<torch::Tensor>& key1_opt,
    const c10::optional<torch::Tensor>& value1_opt,
    const torch::Tensor& cos, const torch::Tensor& sin);

torch::Tensor fused_norm_scale_shift(
    const torch::Tensor& x, const c10::optional<torch::Tensor>& gamma_opt,
    const c10::optional<torch::Tensor>& beta_opt, const torch::Tensor& scale,
    const torch::Tensor& shift, int64_t norm_type, double eps);

torch::Tensor rmsnorm(const torch::Tensor& x, const torch::Tensor& weight, double eps);

#if !defined(JOYOMNI_OPS_NO_FP8) || defined(JOYOMNI_OPS_ROCM_QUANT)
void sgl_per_token_quant_fp8(torch::Tensor input, torch::Tensor output_q, torch::Tensor output_s);
#endif

#ifndef JOYOMNI_OPS_NO_FP8
torch::Tensor fp8_scaled_mm(
    const torch::Tensor& mat_a, const torch::Tensor& mat_b, const torch::Tensor& scales_a,
    const torch::Tensor& scales_b, const torch::ScalarType& out_dtype,
    const c10::optional<torch::Tensor>& bias);
#endif

}  // namespace joyomni_ops

TORCH_LIBRARY(joyomni_ops, m) {
  m.def(
      "fused_qk_norm_rope_3d_paired(Tensor! q, Tensor! k, int seq_len, int num_heads, float eps, "
      "Tensor q_weight, Tensor k_weight, Tensor cos, Tensor sin, Tensor? k_pre_rope) -> ()");
  m.def(
      "fused_joint_qkv_pack(Tensor img_q, Tensor img_k, Tensor img_v, "
      "Tensor txt_q, Tensor txt_k, Tensor txt_v, Tensor? cached_k, Tensor? cached_v, "
      "Tensor txt_q_weight, Tensor txt_k_weight, float eps) -> (Tensor, Tensor, Tensor)");
  m.def(
      "fused_cached_kv_pack(Tensor key0, Tensor value0, Tensor? key1, Tensor? value1, "
      "Tensor cos, Tensor sin) -> (Tensor, Tensor)");
  m.def(
      "fused_norm_scale_shift(Tensor x, Tensor? gamma_opt, Tensor? beta_opt, "
      "Tensor scale, Tensor shift, int norm_type, float eps) -> Tensor");
  m.def("rmsnorm(Tensor x, Tensor weight, float eps) -> Tensor");
#if !defined(JOYOMNI_OPS_NO_FP8) || defined(JOYOMNI_OPS_ROCM_QUANT)
  m.def("sgl_per_token_quant_fp8(Tensor input, Tensor! output_q, Tensor! output_s) -> ()");
#endif
#ifndef JOYOMNI_OPS_NO_FP8
  m.def(
      "fp8_scaled_mm(Tensor mat_a, Tensor mat_b, Tensor scales_a, Tensor scales_b, "
      "ScalarType out_dtype, Tensor? bias) -> Tensor");
#endif
}

TORCH_LIBRARY_IMPL(joyomni_ops, CUDA, m) {
  m.impl("fused_qk_norm_rope_3d_paired", &joyomni_ops::fused_qk_norm_rope_3d_paired);
  m.impl("fused_joint_qkv_pack", &joyomni_ops::fused_joint_qkv_pack);
  m.impl("fused_cached_kv_pack", &joyomni_ops::fused_cached_kv_pack);
  m.impl("fused_norm_scale_shift", &joyomni_ops::fused_norm_scale_shift);
  m.impl("rmsnorm", &joyomni_ops::rmsnorm);
#if !defined(JOYOMNI_OPS_NO_FP8) || defined(JOYOMNI_OPS_ROCM_QUANT)
  m.impl("sgl_per_token_quant_fp8", &joyomni_ops::sgl_per_token_quant_fp8);
#endif
#ifndef JOYOMNI_OPS_NO_FP8
  m.impl("fp8_scaled_mm", &joyomni_ops::fp8_scaled_mm);
#endif
}
