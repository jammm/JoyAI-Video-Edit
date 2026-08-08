# joyomni_ops

Minimal, self-contained CUDA/ROCm operator library for the JoyOmni V2V DiT
pipeline, extracted from
[sgl-kernel](https://github.com/sgl-project/sglang/tree/main/sgl-kernel).

**No `sglang` / `sgl_kernel` / `flashinfer` runtime dependency.** Only the ops the
pipeline actually uses, in one small `.so` (~1 MB without the FP8 GEMM, tens of MB
with it — versus sgl-kernel's ~350 MB `common_ops.so`).

## Ops

| op | description | deps |
|---|---|---|
| `fused_qk_norm_rope_3d_paired` | fused RMSNorm(q,k) + interleaved 3D RoPE (bf16, in-place) | none |
| `fused_norm_scale_shift` | `Norm(x; γ, β) * (1 + scale) + shift` (Layer/RMS, adaLN) | none |
| `rmsnorm` | `(x / RMS(x)) * weight` | none (self-written) |
| `sgl_per_token_quant_fp8` | dynamic per-token bf16/fp16 → fp8_e4m3 quant | CUDA only; self-written |
| `fp8_scaled_mm` | FP8 per-token × per-channel scaled GEMM | CUDA only; CUTLASS |

The first three kernels are hand-written and support both CUDA warp32 and ROCm
wave64. On NVIDIA, the quantizer is also native and `fp8_scaled_mm` uses
CUTLASS. On ROCm, `xvideo.models.dit.fp8_linear` instead fuses dynamic row-wise
quantization with TorchInductor/Triton and dispatches FP8 E4M3 GEMM through
`torch._scaled_mm` to hipBLASLt/Composable Kernel. No CPU fallback is used.

## Build

```bash
# Full build (needs a cutlass checkout; CUDA >= 12.8 for Blackwell SASS):
JOYOMNI_OPS_CUTLASS_DIR=/path/to/cutlass pip install .

# Light build, no FP8 GEMM / no cutlass:
JOYOMNI_OPS_NO_FP8=1 pip install .
```

### ROCm / gfx950

From the repository root, install PyTorch from a compatible ROCm distribution
first. For TheRock wheels,
its `rocm-sdk` packages provide `hipcc`, headers, libraries, and device bitcode
inside the Python environment. Install the accelerator-neutral application
requirements separately so pip never replaces that torch/Triton stack:

```bash
python -m pip install -r requirements-rocm.txt
JOYOMNI_OPS_ROCM_ARCHS=gfx950 python -m pip install --no-build-isolation ./joyomni_ops
```

`setup.py` detects `torch.version.hip`, hipifies only the three portable light
kernels, and excludes the NVIDIA-only quantizer/CUTLASS translation units. The
PyTorch `CUDA` dispatcher key and `torch.cuda` APIs are expected on ROCm; those
names are part of PyTorch's compatibility surface rather than CUDA-only code.

Override the target list with a semicolon-separated value such as
`JOYOMNI_OPS_ROCM_ARCHS="gfx942;gfx950"`. A gfx950-only build avoids carrying
irrelevant code objects and is recommended for MI350-class deployment.

### GPU architectures

Chosen automatically from the local nvcc:
- always: `sm_80`, `sm_89`, `sm_90`
- CUDA ≥ 12.8: also `sm_100a` (B200) and `sm_120a` (RTX 5090)
- on CUDA < 12.8 an `sm_90` PTX is embedded so the driver JITs for Blackwell
  (correctness only; build on CUDA ≥ 12.8 for tuned Blackwell SASS)

Override with `JOYOMNI_OPS_CUDA_ARCHS="90;100a;120a"`.

cutlass tag matching the reference build: `57e3cfb47a2d9e0d46eb6335c3dc411498efa198`.

## Usage

```python
import joyomni_ops as jo
jo.rmsnorm(x, weight, eps)
jo.fused_norm_scale_shift(x, gamma, beta, scale, shift, "layer", eps)
jo.fused_qk_norm_rope_3d_paired(q, k, seq_len, num_heads, eps, qw, kw, cos, sin)  # in-place
jo.sgl_per_token_quant_fp8(x_2d, out_q, out_s)
y = jo.fp8_scaled_mm(x_q, w_fp8, x_scale, w_scale, out_dtype, bias)
```

The last two direct extension calls are NVIDIA-only. The application-level
`Fp8Linear` automatically chooses `torch._scaled_mm` on ROCm.

## Tests

```bash
python tests/test_parity_light.py          # parity vs sgl_kernel (quick)
python tests/test_bench.py --dtype bf16    # speed + accuracy vs pure-torch (also --dtype fp16)
```

On multi-GPU ROCm hosts, select one physical GPU with
`ROCR_VISIBLE_DEVICES=<index>` alone. It becomes logical `cuda:0` to PyTorch;
combining ROCr, HIP, and CUDA visibility masks can double-filter the device set.

## Benchmarks

Measured on NVIDIA B20Z (sm_100), bf16, 4096×4096 shapes, CUDA-event timed.
"vs torch" = accuracy against a pure-PyTorch (fp32-accumulate) reference;
"vs sgl" = against the original sgl_kernel op; speedup vs the equivalent eager
PyTorch ops.

| op | accuracy vs torch (rel) | accuracy vs sgl (max_abs) | speedup vs torch |
|---|---|---|---|
| `rmsnorm` | 1e-8 (equivalent) | 1.5e-2 (bf16 rounding) | **5.7×** |
| `fused_norm_scale_shift` | 3e-8 (equivalent) | **0** | **11.6×** |
| `fused_qk_norm_rope_3d_paired` | 6e-8 (equivalent) | **0** | **10.6×** |
| `fp8_scaled_mm` (mm only) | 3.7% (fp8 quant) | **0** | **1.6×** |
| fp8 linear (dyn-quant + mm) | 3.7% | **0** | ~0.95× |

Notes:
- The 3 fused kernels are numerically **equivalent** to eager PyTorch (rel ~1e-8);
  the speedup comes from fusing several ops into one kernel launch.
- fp8's ~3.7% error is the **inherent fp8 quantization loss**, not a bug. The mm
  itself is 1.6× faster; the "quant + mm" row includes per-call activation
  quantization — in production the weight is pre-quantized and activation quant
  can be fused upstream, so the mm-only figure is the relevant one.
- fp16 results are within noise of bf16 (fp8 mm ~1.6×, fused ops 5.7–11.8×).

## License

Apache-2.0. Portions derive from sgl-kernel (SGLang Team) and NVIDIA
TensorRT-LLM / cutlass; original copyright headers retained in the sources.
