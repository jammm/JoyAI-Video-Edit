# Deployment Guide

This guide describes how to deploy JoyAI-Video-Edit from the code repository plus the released Hugging Face weights.

## Runtime Layout

The server is launched from `deploy/`. Runtime weights are expected under `deploy/deps/`:

```text
deploy/
|-- run_server.sh
|-- static/
|-- xvideo/
`-- deps/
    |-- checkpoints/
    |   |-- JoyAI-Video-Edit/
    |   |   |-- dit/
    |   |   |   `-- joyai_video_edit_dit_0804.pth
    |   |   `-- vae/
    |   |       |-- config.json
    |   |       `-- diffusion_pytorch_model.safetensors
    |   |-- MiMo-VL-7B-RL-2508/
    |   |-- face_detection_yunet_2023mar.onnx
    |   |-- yolov8n.pt
    |   `-- yolov8n.onnx
    `-- cache/
        |-- torchinductor/
        |-- triton/
        `-- nv_compute/
```

## 1. Prepare Environment

Create a Python 3.10 environment and install CUDA-enabled PyTorch plus the serving dependencies:

```bash
conda create -n joyai-video-edit python=3.10 -y
conda activate joyai-video-edit

python -m pip install --upgrade pip setuptools wheel
python -m pip install -r requirements.txt
```

The fused DiT CUDA kernels (RMSNorm, adaLN modulate, 3D-RoPE QK-norm, and the
FP8 quant/GEMM) are provided by the vendored `joyomni_ops` package, not by
sgl-kernel. Build and install it from the repo root:

```bash
# Full build (FP8 GEMM included): point at a cutlass checkout, tag
# 57e3cfb47a2d9e0d46eb6335c3dc411498efa198, CUDA >= 12.8 for Blackwell SASS.
JOYOMNI_OPS_CUTLASS_DIR=/path/to/cutlass python -m pip install ./joyomni_ops

# Light build (no FP8 GEMM, no cutlass needed) — set JOYOMNI_FP8_IMG=0 at runtime:
# JOYOMNI_OPS_NO_FP8=1 python -m pip install ./joyomni_ops
```

See `joyomni_ops/README.md` for op list, GPU-arch selection, and benchmarks.

### AMD ROCm / gfx950

Use a ROCm PyTorch build that supports the target GPU. The accelerator-neutral
dependency file deliberately omits torch, torchvision, torchaudio, Triton,
flash-attn, NVIDIA CUTLASS, and CUDA wheels, so an existing ROCm stack is not
silently replaced:

```bash
# Install/activate the desired ROCm or TheRock torch environment first.
python -m pip install -r requirements-rocm.txt
JOYOMNI_OPS_ROCM_ARCHS=gfx950 python -m pip install --no-build-isolation ./joyomni_ops
```

On ROCm the native extension builds wave64-aware RMSNorm, adaLN modulation,
Q/K norm + RoPE, and per-token OCP E4M3 quantization kernels. The runtime uses
the native quantizer for `K=4096` image- and text-stream projections and the
faster TorchInductor/Triton implementation for `K=16384` projections.
`torch._scaled_mm(..., use_fast_accum=True)` uses the gfx950
hipBLASLt/Composable Kernel path. NVIDIA flash-attn is optional; PyTorch SDPA
selects the available ROCm AOTriton/CK backend.

Select a physical AMD GPU with only `ROCR_VISIBLE_DEVICES` and continue to use
logical `cuda:0` in PyTorch. Set `GPU_SELECTOR` to a free physical index or,
preferably, its ROCr UUID:

```bash
env -u HIP_VISIBLE_DEVICES -u CUDA_VISIBLE_DEVICES \
  ROCR_VISIBLE_DEVICES="$GPU_SELECTOR" JOYOMNI_DEVICE=cuda:0 \
  bash deploy/run_server.sh --height 720 --width 1248 \
    --num-inference-steps 2 --no-use-pe \
    --exact-global-sink-kv --kv-reset-frames 0 \
    --no-freeze-kv-on-static --scene-cut-threshold 25
```

Do not combine the ROCr mask with `HIP_VISIBLE_DEVICES` or
`CUDA_VISIBLE_DEVICES`; multiple masks can filter the device list twice. The
launcher therefore does not inject a CUDA visibility default when a ROCr mask
is already present.

Verify the key runtime imports:

```bash
python - <<'PY'
import torch
import cv2
import av
import joyomni_ops

print("torch", torch.__version__, "cuda", torch.version.cuda, "hip", torch.version.hip)
print("accelerator available:", torch.cuda.is_available())
print("joyomni_ops native fp8:", joyomni_ops.has_fp8())
print("joyomni_ops native quant:", joyomni_ops.has_quant())
print("cv2", cv2.__version__)
PY
```

`joyomni_ops.has_fp8()` is expected to be false for a ROCm light extension;
that flag describes the vendored CUTLASS GEMM, not quantization.
`joyomni_ops.has_quant()` and the application-level `Fp8Linear.available()`
should both be true on gfx950 when the native quantizer and
`torch._scaled_mm` are available.

With `ROCR_VISIBLE_DEVICES` set, the launcher selects the qualified MI350X
preset: pseudo-context causal VAE, event-ordered stage streams on one physical
GPU, final-denoise K/V reuse, a one-time exact all-40-layer refresh for the
permanent global sink, a 24-layer clean refresh for later bounded tail chunks,
and one dynamic full graph around the 40-block DiT core. It accepted 1,440
frames at 24.0 FPS and returned every complete temporal chunk at 23.883
source-window FPS over 60 seconds. With FP8 image and text projections, the
same hybrid policy also sustained 30.983 source-window FPS during a 60-second
31 FPS backend-headroom run, with no drops or queue growth. Periodic K/V reset,
temporal-ID capping, and experimental motion-based K/V freezing default off.
Extreme full-frame discontinuities trigger a safe drain/reset so a hard cut is
not blended with the prior scene. Set
`JOYOMNI_CACHE_LAST_DENOISE_KV=0` for the slower all-40-layer exact clean-cache
policy on every chunk. `JOYOMNI_CLEAN_KV_PREFIX_LAYERS=0` selects the fastest,
gentler editing policy for tail chunks.

`JOYOMNI_FP8_IMG=1`, `JOYOMNI_FP8_TXT=1`, `JOYOMNI_STATEFUL_VAE=0`,
`JOYOMNI_EXPLICIT_STREAMS=1`, and `JOYOMNI_CLEAN_KV_PREFIX_LAYERS=24` are the
qualified gfx950 defaults. All streams still target the one logical `cuda:0`;
this is neither multi-GPU execution nor CPU model offload.

The ROCm launcher also defaults to
`JOYOMNI_DIT_COMPILE_MODE=default`,
`JOYOMNI_DIT_COMPILE_FULLGRAPH=1`, and
`JOYOMNI_DIT_COMPILE_DYNAMIC=1`. The cache dictionaries stay eager while the
complete tensor core is captured as one graph. First startup can take several
minutes; keep `deploy/deps/cache/` and judge only warmed performance. Set the
mode to an empty string for an eager A/B run. Max-autotune and static graph
replay were measured and rejected because both were slower on gfx950.

The launcher does not contain private paths or API credentials. Activate your environment before launching, or pass the conda entrypoint through environment variables:

```bash
JOYAI_CONDA_SH=/path/to/conda/etc/profile.d/conda.sh \
JOYAI_CONDA_ENV=joyai-video-edit \
bash deploy/run_server.sh
```

For local private settings, copy `.env.example` to `.env`, fill in your local values, and source it before launching:

```bash
cp .env.example .env
set -a
source .env
set +a
bash deploy/run_server.sh
```

For prompt enhancement, pass an OpenAI-compatible endpoint at runtime instead of editing source files:

```bash
PE_API_KEY=<your-api-key> \
PE_BASE_URL=https://your-openai-compatible-endpoint/v1 \
PE_MODEL=<your-model-name> \
bash deploy/run_server.sh
```

If these variables are not set, prompt enhancement falls back to the raw user prompt.

### Tested Environment

The public deployment package was tested with a single NVIDIA B200 GPU:

| Item | Version / Configuration |
| --- | --- |
| GPU | 1 x NVIDIA B200 |
| CUDA runtime | `12.8` |
| Python | `3.10` |
| PyTorch | `2.9.1+cu128` |
| Transformers / Diffusers | `4.57.0` / `0.36.0` |
| FastAPI / Uvicorn | `0.117.1` / `0.37.0` |
| OpenCV / PyAV | `opencv-python 5.0.0.93` / `av 13.1.0` |
| Attention / kernel packages | `flash-attn-4 4.0.0b13`, `joyomni_ops 0.1.0` (vendored, built from source), `triton 3.5.1`, `nvidia-cutlass-dsl 4.5.1` |

The ROCm port was additionally qualified on one AMD Instinct MI350X
(`gfx950`, 256 CUs) selected from an eight-GPU host:

| Item | Version / Configuration |
| --- | --- |
| ROCm SDK | TheRock `10.1.0a20260807` |
| Python / PyTorch | `3.12.3` / `2.13.0+rocm10.1.0a20260807` |
| Triton / rocprofv3 | `3.8.0` / `1.3.5` |
| Precision paths | BF16 activations, OCP E4M3 image/text projections, hipBLASLt/CK scaled GEMM, AOTriton/CK SDPA |
| Live result | 1248x720 H.264, two denoising steps, dynamic full-graph DiT, exact 40-layer permanent sink + 24-layer tail refresh, 31.0 input / 30.983 source-window / 30.933 receive-window output FPS over 60 seconds |

For the qualified MI350X live preset, launch with `--num-inference-steps 2`;
the launcher supplies the one-time exact global sink plus hybrid 24-layer tail
refresh by default on ROCm. After the cache-boundary and full-graph work, even
the stricter policy that refreshes all 40 layers on every chunk sustained the
24 FPS gate for 60 seconds: mean/p95/max service time was
289.1/295.4/301.4 ms per eight-frame chunk. The older 18.7 FPS measurement
predates that optimization. The browser remains capped at the upstream 24 FPS
despite the measured greater-than-30-FPS backend capacity, preserving
continuity headroom. Full
reproduction commands and profiling results are in [`HANDOFF.md`](HANDOFF.md).

## 2. Clone JoyAI-Video-Edit Weights

From the code repository root:

```bash
cd deploy
cd deps/checkpoints
git lfs install
git clone https://huggingface.co/jdopensource/JoyAI-Video-Edit
```

This should create:

```text
deps/checkpoints/JoyAI-Video-Edit/dit/joyai_video_edit_dit_0804.pth
deps/checkpoints/JoyAI-Video-Edit/vae/config.json
deps/checkpoints/JoyAI-Video-Edit/vae/diffusion_pytorch_model.safetensors
```

## 3. Prepare External Dependencies

The released JoyAI-Video-Edit weight repo only contains the DiT and VAE weights. The following dependencies are still required at runtime:

| Dependency | Expected path | Notes |
| --- | --- | --- |
| MiMo-VL-7B-RL-2508 | `deploy/deps/checkpoints/MiMo-VL-7B-RL-2508` | Text and visual condition encoder. Download from the upstream MiMo-VL model repository. |
| YuNet face detector | `deploy/deps/checkpoints/face_detection_yunet_2023mar.onnx` | Used by startup and online face gates. If missing, the face gate is disabled by the server. |
| YOLOv8n person detector | `deploy/deps/checkpoints/yolov8n.pt` | Preferred ROCm PyTorch path for online person presence checks. The fixed-320 ONNX path is retained only as a CPU fallback. |

Example text-encoder download:

```bash
cd deploy
hf download XiaomiMiMo/MiMo-VL-7B-RL-2508 \
  --repo-type model \
  --local-dir deps/checkpoints/MiMo-VL-7B-RL-2508
```

Place the ONNX detector files at the paths shown above, or remove the corresponding flags from `run_server.sh` if you intentionally want to run without those gates.

## 4. Launch Server

From `deploy/`:

```bash
bash run_server.sh
```

The default launcher:

- binds to `0.0.0.0:8080`;
- uses `cuda:0` by default;
- places DiT, VAE encode/decode, pseudo encode, and postprocess on the same device by default;
- enables persistent TorchInductor, Triton, and CUDA caches under `deploy/deps/cache/`;
- leaves session recording disabled so the two extra CPU encoders cannot
  interrupt real-time playback.

Open the UI:

```text
http://<server-ip>:8080/
```

## 5. Common Overrides

Enable session recording with an explicit directory:

```bash
JOYOMNI_RECORD_DIR=/path/to/recordings bash run_server.sh
```

Change download re-encode quality:

```bash
JOYOMNI_DOWNLOAD_CRF=8 bash run_server.sh
```

Append additional server flags after the script:

```bash
bash run_server.sh --port 7860 --profile-timings
```

Use custom checkpoint locations:

```bash
JOYAI_DIT_CKPT=/path/to/joyai_video_edit_dit_0804.pth \
JOYAI_VAE_CKPT=/path/to/vae \
JOYAI_TEXT_ENCODER_CKPT=/path/to/MiMo-VL-7B-RL-2508 \
bash run_server.sh
```

## 6. Sanity Checks

Check files:

```bash
test -f deploy/deps/checkpoints/JoyAI-Video-Edit/dit/joyai_video_edit_dit_0804.pth
test -f deploy/deps/checkpoints/JoyAI-Video-Edit/vae/diffusion_pytorch_model.safetensors
test -d deploy/deps/checkpoints/MiMo-VL-7B-RL-2508
```

Check server health after launch:

```bash
curl http://127.0.0.1:8080/health
```

The first launch can take several minutes because PyTorch, Triton, accelerator
kernels, VAE paths, and all dynamic DiT cache states compile and warm up. On
ROCm, the launcher also warms all 49 reference-image buckets so an unseen
reference shape cannot trigger a 25-29 second compile during a prompt switch.
This is intentional: warmed throughput and predictable interaction are the
deployment objectives. Wait for `/health`, and keep `deploy/deps/cache/` stable
across restarts to reuse compile artifacts.
