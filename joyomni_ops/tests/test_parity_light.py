"""Numerical parity: joyomni_ops vs sgl_kernel for the 3 light ops.
Run on a CUDA GPU with both packages importable:
  python tests/test_parity_light.py
"""
import torch
import joyomni_ops as jo

try:
    import sgl_kernel as sk
    HAVE_SGL = True
except Exception as e:  # noqa: BLE001
    HAVE_SGL = False
    print(f"[warn] sgl_kernel not importable ({e!r}); running self-consistency only")

dev = "cuda"
torch.manual_seed(0)


def report(name, a, b):
    # Compare at the kernel output dtype: an FP32 reference can differ by one
    # normal bf16 rounding step even when the kernel is correct.
    b = b.to(a.dtype)
    err = (a.float() - b.float()).abs()
    d = err.max().item()
    rel = (err.mean() / b.float().abs().mean().clamp_min(1e-9)).item()
    print(f"{name:34s} max_diff={d:.3e} rel={rel:.3e}  {'OK' if rel < 1e-4 else 'FAIL'}")
    return d, rel


def test_rmsnorm():
    M, N = 512, 4096
    x = torch.randn(M, N, device=dev, dtype=torch.bfloat16)
    w = torch.randn(N, device=dev, dtype=torch.bfloat16)
    out = jo.rmsnorm(x, w, 1e-6)
    # reference in fp32
    xf = x.float()
    ref = (xf * torch.rsqrt(xf.pow(2).mean(-1, keepdim=True) + 1e-6)) * w.float()
    report("rmsnorm vs fp32-ref", out, ref)
    if HAVE_SGL:
        report("rmsnorm vs sgl", out, sk.rmsnorm(x, w, 1e-6))


def test_fused_norm_scale_shift():
    M, N = 512, 4096
    x = torch.randn(M, N, device=dev, dtype=torch.bfloat16)
    gamma = torch.randn(N, device=dev, dtype=torch.bfloat16)
    beta = torch.randn(N, device=dev, dtype=torch.bfloat16)
    scale = torch.randn(M, N, device=dev, dtype=torch.bfloat16)
    shift = torch.randn(M, N, device=dev, dtype=torch.bfloat16)
    out = jo.fused_norm_scale_shift(x, gamma, beta, scale, shift, "layer", 1e-5)
    # fp32 ref: LayerNorm then modulate
    xf = x.float()
    mean = xf.mean(-1, keepdim=True)
    var = (xf - mean).pow(2).mean(-1, keepdim=True)
    ln = (xf - mean) * torch.rsqrt(var + 1e-5) * gamma.float() + beta.float()
    ref = ln * (1 + scale.float()) + shift.float()
    report("norm_scale_shift vs fp32-ref", out, ref)
    if HAVE_SGL:
        report("norm_scale_shift vs sgl", out, sk.fused_norm_scale_shift(x, gamma, beta, scale, shift, "layer", 1e-5))

    scale_b = scale[:1].contiguous()
    shift_b = shift[:1].contiguous()
    out_b = jo.fused_norm_scale_shift(
        x, gamma, beta, scale_b, shift_b, "layer", 1e-5
    )
    ref_b = ln * (1 + scale_b.float()) + shift_b.float()
    report("norm_scale_shift broadcast", out_b, ref_b)


def test_qk_norm_rope():
    B, seq_len, num_heads, head_dim = 1, 128, 8, 128
    M = seq_len * num_heads
    q = torch.randn(B, M, head_dim, device=dev, dtype=torch.bfloat16)
    k = torch.randn(B, M, head_dim, device=dev, dtype=torch.bfloat16)
    qw = torch.randn(head_dim, device=dev, dtype=torch.bfloat16)
    kw = torch.randn(head_dim, device=dev, dtype=torch.bfloat16)
    cos = torch.randn(seq_len, head_dim // 2, device=dev, dtype=torch.bfloat16)
    sin = torch.randn(seq_len, head_dim // 2, device=dev, dtype=torch.bfloat16)
    if HAVE_SGL:
        q_j, k_j = q.clone(), k.clone()
        q_s, k_s = q.clone(), k.clone()
        jo.fused_qk_norm_rope_3d_paired(q_j, k_j, seq_len, num_heads, 1e-6, qw, kw, cos, sin)
        sk.fused_qk_norm_rope_3d_paired(q_s, k_s, seq_len, num_heads, 1e-6, qw, kw, cos, sin)
        report("qk_norm_rope q vs sgl", q_j, q_s)
        report("qk_norm_rope k vs sgl", k_j, k_s)
    else:
        q_j, k_j = q.clone(), k.clone()
        k_pre = torch.empty_like(k_j)
        jo.fused_qk_norm_rope_3d_paired(
            q_j, k_j, seq_len, num_heads, 1e-6, qw, kw, cos, sin, k_pre
        )
        # fp32 ref
        def ref_one(x, w):
            xf = x.float().view(B, seq_len, num_heads, head_dim)
            xn = xf * torch.rsqrt(xf.pow(2).mean(-1, keepdim=True) + 1e-6) * w.float()
            c = cos.float().view(seq_len, 1, head_dim // 2)
            s = sin.float().view(seq_len, 1, head_dim // 2)
            x1 = xn[..., 0::2]
            x2 = xn[..., 1::2]
            o = torch.empty_like(xn)
            o[..., 0::2] = x1 * c - x2 * s
            o[..., 1::2] = x1 * s + x2 * c
            return o.view(B, M, head_dim)
        report("qk_norm_rope q vs fp32-ref", q_j, ref_one(q, qw))
        report("qk_norm_rope k vs fp32-ref", k_j, ref_one(k, kw))
        kf = k.float()
        k_pre_ref = (
            kf * torch.rsqrt(kf.pow(2).mean(-1, keepdim=True) + 1e-6) * kw.float()
        )
        report("qk_norm pre-rope k", k_pre, k_pre_ref)


def test_per_token_quant_fp8():
    if not jo.has_quant():
        print("per_token_quant_fp8              (not built)")
        return
    M, N = 67, 4096
    x = torch.randn(M, N, device=dev, dtype=torch.bfloat16)
    x[0].zero_()
    q = torch.empty_like(x, dtype=torch.float8_e4m3fn)
    scale = torch.empty((M, 1), device=dev, dtype=torch.float32)
    jo.sgl_per_token_quant_fp8(x, q, scale)
    fp8_max = torch.finfo(torch.float8_e4m3fn).max
    ref_scale = x.abs().amax(dim=1, keepdim=True).float().clamp_min(1e-8) / fp8_max
    ref_q = (x.float() / ref_scale).clamp(-fp8_max, fp8_max).to(torch.float8_e4m3fn)
    scale_rel = ((scale - ref_scale).abs().max() / ref_scale.abs().max().clamp_min(1e-9)).item()
    q_rel = (
        (q.float() * scale - ref_q.float() * ref_scale).abs().mean()
        / (ref_q.float() * ref_scale).abs().mean().clamp_min(1e-9)
    ).item()
    print(f"{'per_token_quant vs torch-ref':34s} scale_rel={scale_rel:.3e} q_rel={q_rel:.3e}  "
          f"{'OK' if scale_rel < 1e-5 and q_rel < 1e-5 else 'FAIL'}")
    assert scale_rel < 1e-5 and q_rel < 1e-5


def test_fused_joint_qkv_pack():
    B, img_len, txt_len, cache_len, heads, dim = 1, 17, 11, 9, 4, 128
    # Slice Q/K/V from interleaved projections so V exercises the same
    # non-contiguous row stride used by the real transformer.
    img_qkv = torch.randn(
        B, img_len, 3, heads, dim, device=dev, dtype=torch.bfloat16
    )
    txt_qkv = torch.randn(
        B, txt_len, 3, heads, dim, device=dev, dtype=torch.bfloat16
    )
    img_q, img_k, img_v = img_qkv.unbind(dim=2)
    txt_q, txt_k, txt_v = txt_qkv.unbind(dim=2)
    cached_k = torch.randn(
        B, cache_len, heads, dim, device=dev, dtype=torch.bfloat16
    )
    cached_v = torch.randn_like(cached_k)
    qw = torch.randn(dim, device=dev, dtype=torch.bfloat16)
    kw = torch.randn(dim, device=dev, dtype=torch.bfloat16)

    out_q, out_k, out_v = jo.fused_joint_qkv_pack(
        img_q, img_k, img_v, txt_q, txt_k, txt_v,
        cached_k, cached_v, qw, kw, 1e-6,
    )

    def norm(x, w):
        xf = x.float()
        normalized = (
            xf * torch.rsqrt(xf.square().mean(-1, keepdim=True) + 1e-6)
        ).to(torch.bfloat16)
        return normalized * w

    ref_q = torch.cat((img_q, norm(txt_q, qw)), dim=1)
    ref_k = torch.cat((cached_k, img_k, norm(txt_k, kw)), dim=1)
    ref_v = torch.cat((cached_v, img_v, txt_v), dim=1)
    report("joint_qkv_pack q", out_q, ref_q)
    report("joint_qkv_pack k", out_k, ref_k)
    report("joint_qkv_pack v", out_v, ref_v)
    assert torch.equal(out_v, ref_v)


def test_fused_cached_kv_pack():
    B, len0, len1, heads, dim = 1, 13, 7, 4, 128
    key0 = torch.randn(B, len0, heads, dim, device=dev, dtype=torch.bfloat16)
    value0 = torch.randn_like(key0)
    key1 = torch.randn(B, len1, heads, dim, device=dev, dtype=torch.bfloat16)
    value1 = torch.randn_like(key1)
    cos_pair = torch.randn(len0 + len1, dim // 2, device=dev)
    sin_pair = torch.randn_like(cos_pair)
    cos = cos_pair.repeat_interleave(2, dim=-1).contiguous()
    sin = sin_pair.repeat_interleave(2, dim=-1).contiguous()
    out_k, out_v = jo.fused_cached_kv_pack(
        key0, value0, key1, value1, cos, sin
    )
    source_k = torch.cat((key0, key1), dim=1).float()
    pairs = source_k.reshape(B, len0 + len1, heads, dim // 2, 2)
    rotated = torch.stack((-pairs[..., 1], pairs[..., 0]), dim=-1).flatten(-2)
    ref_k = (
        source_k * cos.view(1, len0 + len1, 1, dim)
        + rotated * sin.view(1, len0 + len1, 1, dim)
    ).to(torch.bfloat16)
    ref_v = torch.cat((value0, value1), dim=1)
    report("cached_kv_pack k", out_k, ref_k)
    report("cached_kv_pack v", out_v, ref_v)
    assert torch.equal(out_v, ref_v)


if __name__ == "__main__":
    print("has_fp8:", jo.has_fp8())
    test_rmsnorm()
    test_fused_norm_scale_shift()
    test_qk_norm_rope()
    test_per_token_quant_fp8()
    test_fused_joint_qkv_pack()
    test_fused_cached_kv_pack()
