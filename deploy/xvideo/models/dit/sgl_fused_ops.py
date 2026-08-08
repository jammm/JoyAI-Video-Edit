from __future__ import annotations

from typing import Optional, Tuple

import torch
import torch.nn.functional as F


_AVAILABLE: Optional[bool] = None
_FUSED_NORM_SCALE_SHIFT = None
_FUSED_QK_NORM_ROPE_3D = None
_FUSED_JOINT_QKV_PACK = None
_FUSED_CACHED_KV_PACK = None
_RMSNORM = None


def available() -> bool:
    return _try_import()


def _try_import() -> bool:
    global _AVAILABLE, _FUSED_NORM_SCALE_SHIFT, _FUSED_QK_NORM_ROPE_3D
    global _FUSED_JOINT_QKV_PACK, _FUSED_CACHED_KV_PACK, _RMSNORM
    if _AVAILABLE is not None:
        return _AVAILABLE
    try:
        from joyomni_ops import (
            fused_cached_kv_pack,
            fused_joint_qkv_pack,
            fused_norm_scale_shift,
            fused_qk_norm_rope_3d_paired,
            rmsnorm,
        )
    except Exception:
        _AVAILABLE = False
        return _AVAILABLE
    _RMSNORM = rmsnorm
    _FUSED_NORM_SCALE_SHIFT = fused_norm_scale_shift
    _FUSED_QK_NORM_ROPE_3D = fused_qk_norm_rope_3d_paired
    _FUSED_JOINT_QKV_PACK = fused_joint_qkv_pack
    _FUSED_CACHED_KV_PACK = fused_cached_kv_pack
    _AVAILABLE = True
    return _AVAILABLE


def fused_layernorm_modulate(
    x: torch.Tensor,
    shift: torch.Tensor,
    scale: torch.Tensor,
    *,
    weight: Optional[torch.Tensor] = None,
    bias: Optional[torch.Tensor] = None,
    eps: float = 1e-6,
) -> torch.Tensor:
    B, L, D = x.shape

    if _try_import():
        x_2d = x.reshape(-1, D)
        # adaLN produces one modulation vector per batch item. Online
        # streaming is B=1, so let the native kernel broadcast [1, D] over
        # all token rows instead of materializing two [L, D] copies per norm.
        if scale.dim() == 2 and scale.shape == (B, D):
            if B == 1 or L == 1:
                scale_2d = scale.reshape(B, D).contiguous()
                shift_2d = shift.reshape(B, D).contiguous()
            else:
                scale_2d = scale[:, None, :].expand(B, L, D).reshape(-1, D).contiguous()
                shift_2d = shift[:, None, :].expand(B, L, D).reshape(-1, D).contiguous()
        else:
            if scale.dim() == 2:
                scale = scale.unsqueeze(1)
                shift = shift.unsqueeze(1)
            if scale.shape[1] == 1 and L != 1:
                scale = scale.expand(B, L, D)
                shift = shift.expand(B, L, D)
            scale_2d = scale.reshape(-1, D).contiguous()
            shift_2d = shift.reshape(-1, D).contiguous()

        out = _FUSED_NORM_SCALE_SHIFT(
            x_2d, weight, bias, scale_2d, shift_2d, "layer", eps
        )
        return out.reshape(B, L, D)

    # GPU fallback.  Under torch.compile, layer_norm and the following
    # modulation are fused by Inductor/Triton instead of forcing a CPU path.
    out = F.layer_norm(x, (D,), weight=weight, bias=bias, eps=eps)
    return out * (1 + scale) + shift


def prepare_fused_rope(
    freqs_cis: Tuple[torch.Tensor, torch.Tensor],
    *,
    head_dim: int,
    device: torch.device,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Pack full-width RoPE once per forward for all transformer layers."""
    cos, sin = freqs_cis
    while cos.dim() > 2:
        if cos.shape[0] != 1:
            raise RuntimeError(
                f"freqs_cis cos has non-singleton leading dim: {cos.shape}"
            )
        cos = cos.squeeze(0)
        sin = sin.squeeze(0)
    if cos.shape[-1] == head_dim:
        cos = cos[..., ::2]
        sin = sin[..., ::2]
    if cos.shape[-1] != head_dim // 2:
        raise RuntimeError(
            f"freqs_cis width must be {head_dim} or {head_dim // 2}, got {cos.shape[-1]}"
        )
    return (
        cos.to(device=device, dtype=torch.bfloat16).contiguous(),
        sin.to(device=device, dtype=torch.bfloat16).contiguous(),
    )


def fused_qk_norm_rope_3d(
    q: torch.Tensor,
    k: torch.Tensor,
    q_norm_weight: torch.Tensor,
    k_norm_weight: torch.Tensor,
    freqs_cis: Tuple[torch.Tensor, torch.Tensor],
    *,
    eps: float = 1e-6,
    return_k_pre_rope: bool = False,
) -> Tuple[torch.Tensor, ...]:
    B, L, H, D = q.shape
    cos, sin = freqs_cis
    cos = cos.to(q.device)
    sin = sin.to(q.device)

    if _try_import() and q.dtype == torch.bfloat16:
        q = q.contiguous()
        k = k.contiguous()
        q_r = q.view(B, L * H, D)
        k_r = k.view(B, L * H, D)

        while cos.dim() > 2:
            if cos.shape[0] != 1:
                raise RuntimeError(
                    f"freqs_cis cos has non-singleton leading dim: {cos.shape}"
                )
            cos = cos.squeeze(0)
            sin = sin.squeeze(0)
        if cos.shape[-1] == D:
            cos = cos[..., ::2].contiguous()
            sin = sin[..., ::2].contiguous()
        else:
            cos = cos.contiguous()
            sin = sin.contiguous()
        cos_bf16 = cos.to(torch.bfloat16)
        sin_bf16 = sin.to(torch.bfloat16)
        qw = q_norm_weight.to(torch.bfloat16)
        kw = k_norm_weight.to(torch.bfloat16)
        k_pre_rope = torch.empty_like(k) if return_k_pre_rope else None

        _FUSED_QK_NORM_ROPE_3D(
            q_r, k_r, L, H, eps, qw, kw, cos_bf16, sin_bf16,
            k_pre_rope.view(B, L * H, D) if k_pre_rope is not None else None,
        )
        if k_pre_rope is not None:
            return q, k, k_pre_rope
        return q, k

    # Keep the fallback entirely on the accelerator.  This path also permits
    # fp16, which is useful for debugging builds without the native extension.
    q = _torch_rmsnorm(q, q_norm_weight, eps)
    k = _torch_rmsnorm(k, k_norm_weight, eps)
    k_pre_rope = k if return_k_pre_rope else None
    while cos.dim() > 2:
        if cos.shape[0] != 1:
            raise RuntimeError(f"freqs_cis cos has non-singleton leading dim: {cos.shape}")
        cos = cos.squeeze(0)
        sin = sin.squeeze(0)
    if cos.shape[-1] == D // 2:
        cos = cos.repeat_interleave(2, dim=-1)
        sin = sin.repeat_interleave(2, dim=-1)
    if cos.shape != (L, D):
        raise RuntimeError(
            f"freqs_cis must reduce to {(L, D)} (or half-width), got {tuple(cos.shape)}"
        )
    cos = cos.reshape(1, L, 1, D).float()
    sin = sin.reshape(1, L, 1, D).float()

    def apply(x: torch.Tensor) -> torch.Tensor:
        x_float = x.float()
        pairs = x_float.reshape(*x.shape[:-1], -1, 2)
        rotated = torch.stack((-pairs[..., 1], pairs[..., 0]), dim=-1).flatten(-2)
        return (x_float * cos + rotated * sin).to(dtype=x.dtype)

    q, k = apply(q), apply(k)
    if k_pre_rope is not None:
        return q, k, k_pre_rope
    return q, k


def _torch_rmsnorm(
    x: torch.Tensor, weight: torch.Tensor, eps: float
) -> torch.Tensor:
    # FP32 accumulation matches the native kernels; the cast and multiply stay
    # on the same GPU and are fusible by torch.compile.
    xf = x.float()
    out = xf * torch.rsqrt(xf.square().mean(dim=-1, keepdim=True) + eps)
    return (out * weight.float()).to(dtype=x.dtype)


def rmsnorm_qk_bf16(x: torch.Tensor, weight: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    assert x.dtype == torch.bfloat16, f"rmsnorm_qk_bf16 needs bf16, got {x.dtype}"
    orig_shape = x.shape
    D = orig_shape[-1]
    if not _try_import() or _RMSNORM is None:
        return _torch_rmsnorm(x, weight, eps)
    x_flat = x.reshape(-1, D).contiguous()
    w = weight.to(dtype=torch.bfloat16)
    out = _RMSNORM(x_flat, w, eps)
    return out.reshape(orig_shape)


def fused_joint_qkv_pack(
    img_q: torch.Tensor,
    img_k: torch.Tensor,
    img_v: torch.Tensor,
    txt_q: torch.Tensor,
    txt_k: torch.Tensor,
    txt_v: torch.Tensor,
    cached_k: Optional[torch.Tensor],
    cached_v: Optional[torch.Tensor],
    txt_q_weight: torch.Tensor,
    txt_k_weight: torch.Tensor,
    eps: float = 1e-6,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Fused B=1,D=128 joint Q/K/V pack, with text Q/K RMSNorm."""
    fast_path = (
        _try_import()
        and _FUSED_JOINT_QKV_PACK is not None
        and img_q.dtype == torch.bfloat16
        and img_q.ndim == 4
        and img_q.shape[0] == 1
        and img_q.shape[-1] == 128
        and txt_q.dtype == torch.bfloat16
        and (cached_k is None or cached_k.dtype == torch.bfloat16)
    )
    if fast_path:
        return _FUSED_JOINT_QKV_PACK(
            img_q, img_k, img_v, txt_q, txt_k, txt_v,
            cached_k, cached_v,
            txt_q_weight.to(dtype=torch.bfloat16),
            txt_k_weight.to(dtype=torch.bfloat16),
            eps,
        )

    txt_q = _torch_rmsnorm(txt_q, txt_q_weight, eps).to(txt_v)
    txt_k = _torch_rmsnorm(txt_k, txt_k_weight, eps).to(txt_v)
    q = torch.cat((img_q, txt_q), dim=1)
    k = torch.cat((img_k, txt_k), dim=1)
    v = torch.cat((img_v, txt_v), dim=1)
    if cached_k is not None:
        k = torch.cat((cached_k, k), dim=1)
        v = torch.cat((cached_v, v), dim=1)
    return q, k, v


def fused_cached_kv_pack(
    keys: Tuple[torch.Tensor, ...],
    values: Tuple[torch.Tensor, ...],
    freqs_cis: Tuple[torch.Tensor, torch.Tensor],
) -> Optional[Tuple[torch.Tensor, torch.Tensor]]:
    """Fast rotation+packing for one or two pre-RoPE B=1,D=128 caches."""
    if len(keys) not in (1, 2) or len(keys) != len(values):
        return None
    key0, value0 = keys[0], values[0]
    fast_path = (
        _try_import()
        and _FUSED_CACHED_KV_PACK is not None
        and key0.dtype == torch.bfloat16
        and key0.ndim == 4
        and key0.shape[0] == 1
        and key0.shape[-1] == 128
    )
    if not fast_path:
        return None
    cos, sin = freqs_cis
    while cos.ndim > 2:
        if cos.shape[0] != 1:
            return None
        cos = cos.squeeze(0)
        sin = sin.squeeze(0)
    cos = cos.to(device=key0.device, dtype=torch.float32).contiguous()
    sin = sin.to(device=key0.device, dtype=torch.float32).contiguous()
    key1 = keys[1] if len(keys) == 2 else None
    value1 = values[1] if len(values) == 2 else None
    return _FUSED_CACHED_KV_PACK(
        key0, value0, key1, value1, cos, sin
    )


def fused_add_gate(
    residual: torch.Tensor, x: torch.Tensor, gate: torch.Tensor
) -> torch.Tensor:
    return torch.addcmul(residual, x, gate.unsqueeze(1))
