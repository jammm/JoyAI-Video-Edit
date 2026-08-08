from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn

try:
    from joyomni_ops import fp8_scaled_mm, has_fp8, has_quant, sgl_per_token_quant_fp8
    _EXTENSION_FP8_OK = bool(has_fp8())
    _EXTENSION_QUANT_OK = bool(has_quant())
except Exception:
    fp8_scaled_mm = None
    sgl_per_token_quant_fp8 = None
    _EXTENSION_FP8_OK = False
    _EXTENSION_QUANT_OK = False


FP8_DTYPE = torch.float8_e4m3fn
FP8_MAX = torch.finfo(FP8_DTYPE).max
_ROCM_SCALED_MM_OK = torch.version.hip is not None and hasattr(torch, "_scaled_mm")
_ROCM_NATIVE_QUANT_OK = _ROCM_SCALED_MM_OK and _EXTENSION_QUANT_OK
_COMPILED_QUANT_FAILED = False


def available() -> bool:
    return _EXTENSION_FP8_OK or _ROCM_SCALED_MM_OK


def backend() -> str:
    if _ROCM_SCALED_MM_OK:
        return "torch._scaled_mm/hipBLASLt"
    if _EXTENSION_FP8_OK:
        return "joyomni_ops/CUTLASS"
    return "unavailable"


def _quantize_per_token_impl(x: torch.Tensor):
    """Row-wise dynamic FP8 quantization; Inductor fuses this on ROCm."""
    absmax = x.abs().amax(dim=1, keepdim=True).float().clamp_min(1e-8)
    scale = absmax / FP8_MAX
    q = (x.float() / scale).clamp(-FP8_MAX, FP8_MAX).to(FP8_DTYPE)
    return q, scale


if _ROCM_SCALED_MM_OK:
    # Dynamic token counts otherwise produce excessive recompilation in the
    # streaming loop.  The reduction, scaling, clamp and FP8 cast become one
    # gfx950 Triton kernel.
    _quantize_per_token_compiled = torch.compile(
        _quantize_per_token_impl, fullgraph=True, dynamic=True
    )
else:
    _quantize_per_token_compiled = _quantize_per_token_impl


def _quantize_per_token_rocm(x: torch.Tensor):
    global _COMPILED_QUANT_FAILED
    # The native wave64 path wins decisively for JoyOmni's three K=4096
    # projections.  Triton's wider reduction remains faster for the single
    # K=16384 MLP down-projection, so retain that specialized compiled path.
    if _ROCM_NATIVE_QUANT_OK and x.shape[1] < 8192:
        q = torch.empty_like(x, dtype=FP8_DTYPE)
        scale = torch.empty((x.shape[0], 1), device=x.device, dtype=torch.float32)
        sgl_per_token_quant_fp8(x, q, scale)
        return q, scale
    if not _COMPILED_QUANT_FAILED:
        try:
            return _quantize_per_token_compiled(x)
        except Exception:
            # A TheRock/Triton mismatch should not make inference impossible;
            # eager ops still execute on the GPU and _scaled_mm remains fused.
            _COMPILED_QUANT_FAILED = True
    return _quantize_per_token_impl(x)


def _quantize_weight_per_channel(w_bf16: torch.Tensor):
    w_kn = w_bf16.t().contiguous()
    K, N = w_kn.shape
    absmax = w_kn.abs().amax(dim=0).to(torch.float32).clamp_min(1e-8)
    scale = absmax / FP8_MAX
    w_q_rm = (w_kn.float() / scale.unsqueeze(0)).clamp(-FP8_MAX, FP8_MAX).to(FP8_DTYPE)
    w_q_cm = w_q_rm.t().contiguous().t()
    assert w_q_cm.shape == (K, N) and w_q_cm.stride() == (1, K)
    return w_q_cm, scale.reshape(1, N).contiguous()


class Fp8Linear(nn.Module):

    def __init__(self, weight_fp8: torch.Tensor, weight_scale: torch.Tensor,
                 bias: Optional[torch.Tensor], out_dtype: torch.dtype = torch.bfloat16):
        super().__init__()
        self.register_buffer("weight_fp8", weight_fp8, persistent=False)
        self.register_buffer("weight_scale", weight_scale, persistent=False)
        if bias is not None:
            self.register_buffer("bias", bias.contiguous(), persistent=False)
        else:
            self.bias = None
        self.out_dtype = out_dtype
        K, N = weight_fp8.shape
        self.in_features = K
        self.out_features = N

    @classmethod
    def from_linear(cls, lin: nn.Linear, out_dtype: torch.dtype = torch.bfloat16) -> "Fp8Linear":
        assert available(), "joyomni_ops fp8 ops not available; cannot build Fp8Linear"
        w = lin.weight.data
        w_q, w_s = _quantize_weight_per_channel(w.to(torch.bfloat16))
        b = None
        if lin.bias is not None:
            b = lin.bias.data.to(dtype=out_dtype)
        return cls(w_q, w_s, b, out_dtype=out_dtype)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        orig_shape = x.shape
        x_2d = x.reshape(-1, orig_shape[-1]).contiguous()
        M, K = x_2d.shape
        assert K == self.in_features, f"in features mismatch {K} vs {self.in_features}"
        if _ROCM_SCALED_MM_OK:
            x_q, x_scale = _quantize_per_token_rocm(x_2d)
            # gfx950 dispatches this row-wise FP8 GEMM to hipBLASLt/CK.  A is
            # row-major, B has column-major stride (1, K), and scales use the
            # required (M,1)/(1,N) outer-vector layout.
            y = torch._scaled_mm(
                x_q,
                self.weight_fp8,
                x_scale,
                self.weight_scale,
                bias=self.bias,
                out_dtype=self.out_dtype,
                use_fast_accum=True,
            )
        else:
            x_q = torch.empty((M, K), device=x_2d.device, dtype=FP8_DTYPE)
            x_scale = torch.empty((M, 1), device=x_2d.device, dtype=torch.float32)
            sgl_per_token_quant_fp8(x_2d, x_q, x_scale)
            y = fp8_scaled_mm(
                x_q, self.weight_fp8, x_scale, self.weight_scale,
                out_dtype=self.out_dtype, bias=self.bias
            )
        return y.reshape(*orig_shape[:-1], self.out_features)
