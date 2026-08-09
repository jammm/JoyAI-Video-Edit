# JoyAI Video Edit ROCm/gfx950 handoff

Prepared 2026-08-08 from upstream commit
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
- Long-motion ghosting fix: the permanent global-sink chunk stores clean K/V
  at all 40 layers, while later bounded tail chunks use a 24-layer clean
  refresh. Periodic sink replacement, temporal-ID capping, and motion-based
  stale-tail reuse are disabled by default because each could preserve an old
  pose.
- Extreme full-frame discontinuities are detected with a small luma MAD. The
  old causal pipeline is quiesced, its pre-cut partial chunk is padded and
  drained in order, and the cut frame becomes a clean one-frame sink. A failed
  teardown never creates an overlapping replacement session.
- Direct H.264/JPEG delivery, single-consumer ordered output, a locked PyAV
  encoder lifecycle, bounded result backpressure, safer worker shutdown, and
  output-pump error handling.
- Event-ordered encode, DiT, decode, pseudo-encode, and postprocess streams on
  one physical GPU. The qualified pseudo-context VAE path avoids the compiled
  stateful VAE replay issue, and independent per-session GPU generators make
  its asynchronous VAE sampling deterministic.
- Static/unit/protocol checks and native op parity/shape sweeps.

Destination-host qualification is complete. The qualification machine had
eight 256-CU AMD Instinct MI350X (`gfx950`) cards, but every model run was
isolated to one selected physical GPU and saw it as the sole logical `cuda:0`.
No physical index, UUID, PCI address, or host path is a deployment requirement.
The full model uses about 88 GiB of that GPU's HBM and does not offload model
work to the CPU.

The final qualified live preset uses two denoising steps, exact all-layer clean
K/V for the one permanent sink, a 24-layer clean refresh for later chunks,
window-relative temporal IDs, `--kv-reset-frames 0`,
`--no-freeze-kv-on-static`, `--scene-cut-threshold 25`, `--no-use-pe`,
`--no-online-gate`, and no session recording. Its 60-second 1248x720 H.264 run
accepted and acknowledged all 1,440 inputs at 24.0 FPS with zero
protocol/decode errors, zero backpressure skips, zero scheduled drops, and a
maximum pending depth of four frames. It returned all 1,433 frames belonging
to complete temporal chunks (23.883 source-window FPS; the final seven inputs
remain an intentionally incomplete chunk). Mean/p95 server chunk residence was
431.8/515.0 ms, mean/p95 end-to-end latency was 681.1/855.8 ms, and the largest
packet gap was 435.4 ms, within the browser's adaptive one-second decoded-frame
buffer. The slower policy that refreshes all 40 layers on every chunk produced
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
    --exact-global-sink-kv \
    --kv-reset-frames 0 \
    --no-freeze-kv-on-static \
    --scene-cut-threshold 25 \
    --no-online-gate
```

All DiT, text-encoder, VAE encode/decode/pseudo roles, and postprocessing stay
on the selected GPU. The first launch compiles/autotunes kernels; retain
`deploy/deps/cache` after a successful warmup. On gfx950 the launcher defaults
to the qualified pseudo-context VAE, event-ordered single-GPU stream,
hybrid-KV preset (`JOYOMNI_STATEFUL_VAE=0`,
`JOYOMNI_EXPLICIT_STREAMS=1`, `JOYOMNI_CACHE_LAST_DENOISE_KV=1`, and
`JOYOMNI_CLEAN_KV_PREFIX_LAYERS=24`). Set
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
  --exact-global-sink-kv \
  --kv-reset-frames 0 \
  --no-freeze-kv-on-static --scene-cut-threshold 25 \
  --cache-last-denoise-kv --clean-kv-prefix-layers 24 \
  --no-profile-timings \
  --warmup-seconds 0 --measure-seconds 60 \
  --output-json "$ARTIFACT_ROOT/streaming.json"
```

The UI is capped at 24 FPS. The live acceptance gate is sustained decoded
output of at least 23.5 FPS for 60 seconds at 1248x720 and two denoising steps,
with no protocol/decode errors or sustained queue/drop growth. The local
validation artifact (intentionally excluded from Git) is
`$ARTIFACT_ROOT/sdpa-scene-reset-prefix24-60s.json`: 24.0 input FPS and 23.883
source-window output FPS, zero errors/drops/skips, and bounded queues over the
60-second measurement.
Application stage timing is deliberately disabled for this acceptance run;
`--profile-timings` inserts cross-thread GPU events and is intended only for a
separate diagnostic run, not user-facing continuity or throughput numbers.

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
      --num-inference-steps 2 --no-use-pe \
      --no-online-gate --profile-timings
```

The baseline full-model trace recorded 392,998 dispatches, all on the selected
gfx950 agent. Kernel time was dominated by FP8 GEMM (~23%), VAE convolution
(~22.5%), attention (~12.4%), and other GEMM (~9.5%); device-copy time was
negligible. It also proved that the original worker threads shared one default
stream. The final pseudo-context path uses explicit event-ordered stage streams
on that same physical GPU; events preserve dependencies without model
offloading or multi-GPU execution.

## 6. Long-motion quality and attention follow-up

The ghosting regression was reproduced with the high-motion final 15 seconds
of the original reference demo:
`https://github.com/user-attachments/assets/bca232c9-75df-46f9-b366-14cfa2651994`.
The old hybrid policy left the permanent sink's later layers at a noisy denoise
state forever. A false-static decision could then reuse that pose as another
tail anchor. The final policy stores the sink from the clean latent at all
layers exactly once, refreshes 24 leading layers for bounded tail chunks, and
disables periodic reset, temporal-ID capping, and stale-tail freezing.

The apparent catastrophic trail at source frame 32 was separately identified
as an actual presentation cut: 64x36 luma MAD jumps to 56.59 between frames 31
and 32, while ordinary high motion in the rest of the clip stays below 7.5.
Feeding the cut through an old causal chunk creates the expected double
exposure, including with BF16 image projections and JPEG transport. The final
cut path instead drained source frames 0-31, inserted one duplicate of frame 31
to complete the temporal chunk, and generated frame 32 from a fresh sink. The
full 360-input qualification saw exactly this one cut, no false resets, no
drops or codec errors, and a largest output gap of 864.1 ms, still below the
default one-second browser buffer.

The final 24-layer capture had 353 unique mapped outputs through source index
352; the last seven source frames intentionally remain an incomplete live
chunk. Over adjacent moving pairs, optical-flow EPE was 0.587826 px, flow
cosine 0.859630, motion-compensated luma L1 0.018673, and stale-edge rate
0.031510. Relative to the otherwise identical 18-layer capture, the 24-layer
setting improved flow cosine (0.844737 to 0.859630), p95 stale-edge rate
(0.083503 to 0.079524), and luma L1 (0.018741 to 0.018673), with a small EPE
tradeoff (0.583516 to 0.587826). It was selected because the 60-second test
proved that the additional exact refresh still met the live throughput gate.
The official demo's edited side has different rendering and crop geometry, so
it was used as a visual target rather than as uncalibrated pixel/flow ground
truth. Use `deploy/analyze_streaming_quality.py` to reproduce the temporal
analysis from a benchmark packet capture.

ROCm/AITER PR #4627 was retested at commit
`41c4093bc92531c83959cae481a1445a615ff2af`. Its current gfx950 MHA v4 path is
dense and quantized; the PR documentation defers sparse kernels. Focused
upstream tests passed 34 cases and failed the two FP4-value compile-parity
cases, so those formats were excluded. On real JoyAI activations,
INT8-Q/K+FP8-V passed the per-layer gate at observed `Q=3467` and growing K/V
lengths, while MXFP6/FP8 failed early-layer tail-cosine gates.

The decisive end-to-end test included JoyAI's non-contiguous Q/K/V views, a
native wave64 quantizer, packing, and the AITER launch at `Q=3467`, 32 heads,
head dimension 128, and K/V lengths 3467 through 11267. The AITER kernel alone
was 1.61-1.91x faster than SDPA, but the complete path was only 0.678-0.974x
SDPA and never won at any tested length. Transformer-like inputs also missed
the accuracy gate (roughly 0.9983 global cosine and 0.058 relative L2), with
sink-heavy inputs worse. AITER was therefore rejected for production; the
qualified backend remains PyTorch ROCm SDPA (AOTriton/CK dispatch). No AITER
runtime dependency or experimental attention code is required by this branch.

## 7. Remote browser access

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

Native and runtime validation record:

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
- Full warmup, the prefix-24 60-second live benchmark, the 360-input motion/cut
  capture, a full selected-region rocprof capture, and a controlled HTTPS/WSS
  protocol run all completed successfully on the destination host.
