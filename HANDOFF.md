# JoyAI Video Edit ROCm/gfx950 handoff

Prepared 2026-08-07 from upstream commit
`231aab0d32f62fefc853cf9a046b8f29b4a39dfd`.

This branch contains the ROCm/gfx950 port, native wave64
kernels, streaming runtime fixes, and a synthetic streaming benchmark. The
original transfer archive omitted model weights, Python environments, compiler
caches, recordings, native build products, and Git metadata; those reproducible
assets have now been restored on the destination host. No changes were pushed
upstream.

The transfer archive also contains a snapshot of the source user's Codex data
directory (normally `${CODEX_HOME:-$HOME/.codex}`) as its top-level `.codex/`
directory, as explicitly requested for machine handoff. That snapshot includes
Codex configuration, session/state history, installed plugins/binaries, and
`auth.json`. Treat the ZIP as a credential-bearing secret: transfer it only over
a trusted channel, restrict access on the destination, and delete the transfer
copy when the handoff is complete.

## Path and device conventions

The commands below do not depend on a particular username, mount point, host,
or GPU ordinal. Start in the root of the cloned repository and define these
values once for the current machine:

```bash
export REPO_ROOT="$(pwd -P)"
export VENV_ROOT="${VENV_ROOT:-${REPO_ROOT}/.venv}"
export GPU_SELECTOR="<physical-index-or-rocr-uuid>"
export SERVICE_HOST="${SERVICE_HOST:-127.0.0.1}"
export SERVICE_PORT="${SERVICE_PORT:-8080}"
export ARTIFACT_ROOT="${ARTIFACT_ROOT:-${REPO_ROOT}/profiles}"
```

Replace the `GPU_SELECTOR` placeholder with a free physical-device index or,
preferably, its ROCr UUID. All repository, environment, checkpoint, cache,
profile, and TLS paths in this handoff are derived from these variables or are
relative to `REPO_ROOT`.

## Current status

Completed and validated on an MI350X (`gfx950`):

- ROCm/TheRock PyTorch, torchvision, torchaudio, Triton, hipBLASLt/CK, AOTriton
  SDPA, OCP FP8 scaled GEMM, and `rocprofv3` smoke tests.
- A ROCm-safe `joyomni_ops` build path with wave64-aware native RMSNorm,
  layernorm/modulation, and paired Q/K norm + 3D RoPE kernels.
- A gfx950 FP8 path using a native wave64 row-wise E4M3FN quantizer for the
  three `K=4096` projections, compiled Triton quantization for the `K=16384`
  projection, and `torch._scaled_mm`; NVIDIA CUTLASS/flash-attn are optional
  and not imported on ROCm.
- GPU-only fallbacks; no model CPU offload was added.
- Correct 720x1248 VAE warmup geometry and removal of redundant streaming VAE
  work, scheduler/device synchronization, repeated RGB/JPEG conversions, and
  per-stage synchronization.
- Direct H.264/JPEG delivery, single-consumer ordered output, a locked PyAV
  encoder lifecycle, bounded result backpressure, safer worker shutdown, and
  output-pump error handling.
- A serialized single-GPU submission path, which is the qualified default.
  Experimental event-ordered stage streams remain available with
  `JOYOMNI_EXPLICIT_STREAMS=1`, but are not enabled by default because compiled
  stateful VAE execution was not reliable on non-default streams.
- Static/unit/protocol checks and native op parity/shape sweeps.

Destination-host qualification is complete. The qualification machine had
eight 256-CU AMD Instinct MI350X (`gfx950`) cards, but every model run was
isolated to one selected physical GPU and saw it as the sole logical `cuda:0`.
No physical index, UUID, PCI address, or host path is a deployment requirement.
The full model uses about 88 GiB of that GPU's HBM and does not offload model
work to the CPU.

The final qualified live preset uses two denoising steps, an 18-layer clean K/V
refresh, `--no-use-pe`, `--max-temporal-ids 8`, `--no-online-gate`, and no
session recording. It passed a 60-second 1248x720 HTTPS/WSS H.264 run at 24.0
input and decoded output FPS with zero protocol/decode errors, zero
backpressure skips, no scheduled drops, and no pending frames. Mean/p95 server
chunk time was 300.7/310.7 ms, mean/p95 end-to-end latency was 567.9/696.0 ms,
and the largest packet gap was 375.8 ms, within the browser's adaptive one-second
decoded-frame buffer. The slower exact all-40-layer clean-cache policy produced
about 18.7 FPS and therefore does not meet the 24 FPS live acceptance target.

## 1. Create the ROCm environment

Python 3.12 was used for qualification. `VENV_ROOT` may point inside or outside
the checkout; generated environments are not part of the repository:

```bash
cd "$REPO_ROOT"
python3 -m venv "$VENV_ROOT"
source "$VENV_ROOT/bin/activate"
python -m pip install --upgrade pip setuptools wheel

python -m pip install \
  --index-url https://rocm.nightlies.amd.com/whl-multi-arch/ \
  "rocm[libraries,devel,device-gfx950]" \
  "torch[device-gfx950]" \
  "torchvision[device-gfx950]" \
  torchaudio

python -m pip install \
  --index-url https://rocm.nightlies.amd.com/whl-multi-arch/ \
  "rocm[profiler]"

rocm-sdk init
rocm-sdk path --root

python -m pip install -r requirements-rocm.txt
JOYOMNI_OPS_ROCM_ARCHS=gfx950 \
  python -m pip install --no-build-isolation ./joyomni_ops
python -m pip check
```

The 2026-08-07 validation used ROCm/TheRock `10.1.0a20260807`, PyTorch
`2.13.0+rocm10.1.0a20260807`, torchvision `0.28.0` with the same ROCm suffix,
Triton 3.8.0, and `rocprofv3` 1.3.5. The current SDK command is
`rocm-sdk init`; these wheels do not provide the older `rocm-init` executable.

Check the selected physical GPU. Use only `ROCR_VISIBLE_DEVICES`; PyTorch still
names the visible logical device `cuda:0` on ROCm:

```bash
env -u HIP_VISIBLE_DEVICES -u CUDA_VISIBLE_DEVICES \
  ROCR_VISIBLE_DEVICES="$GPU_SELECTOR" python - <<'PY'
import torch
p = torch.cuda.get_device_properties(0)
print(torch.cuda.device_count(), p.name, p.gcnArchName, hex(p.pci_bus_id))
print("torch", torch.__version__, "HIP", torch.version.hip)
PY
```

Do not combine ROCr, HIP, and CUDA visibility masks; multiple masks can filter
the device set twice. UUID selection avoids numbering ambiguity. The exact
index or UUID is intentionally not embedded in this handoff because it is a
property of the destination host, not the repository.

## 2. Download runtime assets

The models are public and ungated. From the repository root:

```bash
source "$VENV_ROOT/bin/activate"
cd "$REPO_ROOT/deploy"

hf download jdopensource/JoyAI-Video-Edit \
  --revision 56e4a5dfb14f054bffa03dfc36ef5562e68cb988 \
  --local-dir deps/checkpoints/JoyAI-Video-Edit

hf download XiaomiMiMo/MiMo-VL-7B-RL-2508 \
  --revision 4bfb270765825d2fa059011deb4c96fdd579be6f \
  --include '*.json' '*.txt' '*.safetensors' \
  --local-dir deps/checkpoints/MiMo-VL-7B-RL-2508

curl -fL \
  https://media.githubusercontent.com/media/opencv/opencv_zoo/main/models/face_detection_yunet/face_detection_yunet_2023mar.onnx \
  -o deps/checkpoints/face_detection_yunet_2023mar.onnx
curl -fL \
  https://github.com/ultralytics/assets/releases/download/v8.4.0/yolov8n.onnx \
  -o deps/checkpoints/yolov8n.onnx
curl -fL \
  https://github.com/ultralytics/assets/releases/download/v8.4.0/yolov8n.pt \
  -o deps/checkpoints/yolov8n.pt
```

Expected large-file checksums and sizes:

| Asset | Bytes | SHA-256 |
| --- | ---: | --- |
| JoyAI DiT `joyai_video_edit_dit_0804.pth` | 32,527,711,955 | `acc90774bb72c80ffb2b2c93f7ef539da4f993a47c6d44d7423fba2aff8f1aa6` |
| JoyAI VAE `diffusion_pytorch_model.safetensors` | 1,534,679,470 | `150315748d7c3307cdae2819ee651b32d58385668ca0c4db3d3dcd6e63b77e86` |
| MiMo shard 1 | - | `f93e07524d843d63c080eef4fd43d7c5b98ac7a17e8c56e48edfe89297d6bff3` |
| MiMo shard 2 | - | `9715248b5ff4357d2deb23669c83131f35934c981e88f648a9769976e533412e` |
| MiMo shard 3 | - | `5b8db223f443f8ed88b65017c65aaa9e201e386b942676e75bf1651202e2181f` |
| MiMo shard 4 | - | `8bcbeaac2b402096c0629435395003fa66eb14dfba682d3af02b4398001f5b0f` |
| YuNet ONNX | 232,589 | `8f2383e4dd3cfbb4553ea8718107fc0423210dc964f9f4280604804ed2552fa4` |
| YOLOv8n ONNX | 12,851,049 | `b2bc52f40e8e1c532427d5bde3575a5d5b571b739fab2c6df443733ed1589cbd` |

Core runtime assets total about 47.2 GiB. YuNet and YOLO are optional; their
gates disable themselves when the corresponding files are absent.
The official JoyAI DiT/VAE checkpoint repository declares Apache-2.0, while
the pinned MiMo-VL checkpoint repository declares MIT. Detector assets remain
third-party components and retain their own licensing terms.

## 3. Launch on one free GPU

From the repository root, with the environment active:

```bash
cd "$REPO_ROOT"
env -u HIP_VISIBLE_DEVICES -u CUDA_VISIBLE_DEVICES \
  ROCR_VISIBLE_DEVICES="$GPU_SELECTOR" \
  JOYOMNI_DEVICE=cuda:0 \
  JOYAI_HOST="$SERVICE_HOST" JOYAI_PORT="$SERVICE_PORT" \
  bash deploy/run_server.sh \
    --height 720 --width 1248 \
    --num-inference-steps 2 \
    --no-use-pe \
    --max-temporal-ids 8 \
    --no-online-gate \
    --profile-timings
```

All DiT, text-encoder, VAE encode/decode/pseudo roles, and postprocessing stay
on the selected GPU. The first launch compiles/autotunes kernels; retain
`deploy/deps/cache` after a successful warmup. On gfx950 the launcher defaults
to the qualified stateful-VAE, serialized-stream,
hybrid-KV preset (`JOYOMNI_CACHE_LAST_DENOISE_KV=1` and
`JOYOMNI_CLEAN_KV_PREFIX_LAYERS=18`). Set
`JOYOMNI_CACHE_LAST_DENOISE_KV=0` for the slower exact all-layer cache policy.

Check health locally:

```bash
curl -f "http://${SERVICE_HOST}:${SERVICE_PORT}/health"
```

## 4. Exercise the streaming path

The included benchmark speaks the real `/ws` protocol, negotiates H.264 with a
JPEG fallback, applies acknowledgement backpressure, validates decoded frames,
samples `/debug`, and writes a compact JSON report:

```bash
cd "$REPO_ROOT"
source "$VENV_ROOT/bin/activate"
mkdir -p "$ARTIFACT_ROOT"
python deploy/benchmark_streaming.py \
  --url "ws://${SERVICE_HOST}:${SERVICE_PORT}/ws" \
  --width 1248 --height 720 --fps 24 \
  --num-inference-steps 2 \
  --max-temporal-ids 8 \
  --cache-last-denoise-kv --clean-kv-prefix-layers 18 \
  --warmup-seconds 10 --measure-seconds 60 \
  --output-json "$ARTIFACT_ROOT/streaming.json"
```

The UI is capped at 24 FPS. The live acceptance gate is sustained decoded
output of at least 23.5 FPS for 60 seconds at 1248x720 and two denoising steps,
with no protocol/decode errors or sustained queue/drop growth. The local
validation artifact (intentionally excluded from Git) is
`$ARTIFACT_ROOT/streaming-continuity-https-h264-60s.json`: 24.0 decoded FPS,
zero errors/drops/skips, and bounded queues over the 60-second measurement.

## 5. Profile after warmup

Dynamic attach is not reliable with this PyTorch/rocprofiler combination. Wrap
the server with selected-region profiling, then let the streaming session open
and close the capture interval after warmup:

```bash
cd "$REPO_ROOT"
source "$VENV_ROOT/bin/activate"
mkdir -p "$ARTIFACT_ROOT"
env -u HIP_VISIBLE_DEVICES -u CUDA_VISIBLE_DEVICES \
  ROCR_VISIBLE_DEVICES="$GPU_SELECTOR" \
  JOYOMNI_ROCPROF_SELECTED_REGIONS=1 \
  rocprofv3 --kernel-trace --memory-copy-trace --marker-trace \
    --stats --group-by-queue --selected-regions -f csv \
    -d "$ARTIFACT_ROOT/rocprof" -- \
    bash deploy/run_server.sh --height 720 --width 1248 \
      --num-inference-steps 2 --no-use-pe --max-temporal-ids 8 \
      --no-online-gate --profile-timings
```

The baseline full-model trace recorded 392,998 dispatches, all on the selected
gfx950 agent. Kernel time was dominated by FP8 GEMM (~23%), VAE convolution
(~22.5%), attention (~12.4%), and other GEMM (~9.5%); device-copy time was
negligible. It also proved that the original worker threads shared one default
stream. Explicit event-ordered streams were implemented and tested from that
evidence, but did not improve the qualified workload; the ROCm preset therefore
uses serialized default-stream submission.

## 6. Remote browser access

The browser captures its own camera with `getUserMedia` and sends WebCodecs
H.264 (JPEG fallback) over WebSocket; this is not WebRTC and the server does not
need `/dev/video`.

Remote camera capture requires a browser-trusted HTTPS/WSS origin. Plain
`http://<remote-host>:<port>` can render the page but is not a secure context,
so camera access is normally rejected. Production use should place a trusted
TLS reverse proxy in front of the loopback-bound service and use a
client-resolvable DNS name. Neither a hostname nor a certificate is embedded in
the repository.

For a controlled local-network demo, a self-signed proxy can be created without
putting host-specific paths in the checkout. Keep `SERVICE_HOST=127.0.0.1`, set
`PUBLIC_ORIGIN_HOST` to the address clients use, and set `TLS_SAN` to either an
IP or DNS subject alternative name:

```bash
export PUBLIC_ORIGIN_HOST="<client-reachable-host-or-ip>"
export PUBLIC_BIND_ADDRESS="<server-listen-address>"
export TLS_SAN="<IP:server-address-or-DNS:server-name>"
export TLS_ROOT="${TLS_ROOT:-${ARTIFACT_ROOT}/tls}"

mkdir -p "$TLS_ROOT"
openssl req -x509 -newkey rsa:2048 -sha256 -nodes -days 7 \
  -keyout "$TLS_ROOT/server.key" \
  -out "$TLS_ROOT/server.crt" \
  -subj "/CN=JoyAI local demo" \
  -addext "subjectAltName=${TLS_SAN}"

socat \
  "OPENSSL-LISTEN:${SERVICE_PORT},bind=${PUBLIC_BIND_ADDRESS},reuseaddr,fork,cert=${TLS_ROOT}/server.crt,key=${TLS_ROOT}/server.key,verify=0" \
  "TCP:${SERVICE_HOST}:${SERVICE_PORT}"
```

Open `https://${PUBLIC_ORIGIN_HOST}:${SERVICE_PORT}`. A self-signed certificate
requires an explicit browser exception or development policy and is not a
production trust solution. Any public tunnel is deployment infrastructure, not
repository configuration; do not record an ephemeral tunnel URL in this file.

## Validation record

Native extension validation completed before the machine became contended:

- gfx950 build and hipify succeeded with zero unsupported CUDA calls.
- RMSNorm, layernorm/modulation, and Q/K+RoPE parity passed in float32, float16,
  and bfloat16 over representative widths/head sizes.
- At the runtime RMS shape (`M=1560`, `N=4096`), native RMSNorm was 9.43 us vs
  17.64 us for compiled PyTorch, and native layernorm/modulation was 11.78 us
  vs 18.12 us.
- At `B=1,L=1560,H=32,D=128`, paired native Q/K+RoPE was 26.69 us vs 34.49 us.
- Native gfx950 FP8 quantization matched the reference at relative error below
  `9e-8`; the final extension parity suite passed all five native operations.
- ROCm SDPA and OCP E4M3FN scaled GEMM produced finite bfloat16 output.
- Static Python compilation, focused CPU units, scheduler parity, PyAV H.264
  round trips, mocked WebSocket flow, bounded-queue behavior, and concurrent
  encoder lifecycle tests passed.
- Full warmup, the 60-second live benchmark, a full selected-region rocprof
  capture, and a public HTTPS/WSS protocol run all completed successfully on
  the destination host.
