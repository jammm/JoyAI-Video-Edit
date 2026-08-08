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
the native quantizer for the three `K=4096` image-stream projections and the
faster TorchInductor/Triton implementation for the `K=16384` projection.
`torch._scaled_mm(..., use_fast_accum=True)` uses the gfx950
hipBLASLt/Composable Kernel path. NVIDIA flash-attn is optional; PyTorch SDPA
selects the available ROCm AOTriton/CK backend.

Select a physical AMD GPU with only `ROCR_VISIBLE_DEVICES` and continue to use
logical `cuda:0` in PyTorch:

```bash
env -u HIP_VISIBLE_DEVICES -u CUDA_VISIBLE_DEVICES \
  ROCR_VISIBLE_DEVICES=2 JOYOMNI_DEVICE=cuda:0 \
  bash deploy/run_server.sh --height 720 --width 1248 \
    --num-inference-steps 2 --no-use-pe
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
preset: stateful causal VAE, serialized default-stream submission, final-denoise
K/V reuse, and an exact 18-layer clean-KV refresh. This sustained 24 decoded
FPS at 1248x720 with two denoising steps for 60 seconds. Set
`JOYOMNI_CACHE_LAST_DENOISE_KV=0` for the slower all-40-layer exact clean-cache
policy. `JOYOMNI_CLEAN_KV_PREFIX_LAYERS=0` selects the fastest, gentler editing
policy; values above 18 were not real-time in the qualified workload.

`JOYOMNI_EXPLICIT_STREAMS=1` remains available for event-ordered per-stage
streams, but `0` is the qualified gfx950 setting because rocprof showed no
throughput benefit from the extra queues for this pipeline.

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
| Precision paths | BF16 activations, OCP E4M3 image projections, hipBLASLt/CK scaled GEMM, AOTriton/CK SDPA |
| Live result | 1248x720 H.264, two denoising steps, hybrid 18-layer clean-KV refresh, 24.0 decoded FPS over 60 seconds |

For the qualified MI350X live preset, launch with `--num-inference-steps 2`;
the launcher supplies the hybrid 18-layer clean-KV refresh by default on ROCm.
The one-step preset measured 24.067 FPS, while the slower exact all-40-layer
clean-cache policy measured about 18.7 FPS at this resolution. Full
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

The first launch can be slow because PyTorch, Triton, CUDA kernels, VAE paths, and DiT attention kernels need to compile and warm up. Keep `deploy/deps/cache/` stable across restarts to reuse compile artifacts.
