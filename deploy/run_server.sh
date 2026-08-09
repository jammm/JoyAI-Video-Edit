#!/bin/bash
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$HERE"

if [ -n "${JOYAI_CONDA_SH:-}" ]; then
  source "$JOYAI_CONDA_SH"
  while [ "${CONDA_SHLVL:-0}" -gt 0 ]; do conda deactivate; done
  conda activate "${JOYAI_CONDA_ENV:-joyai-video-edit}"
  hash -r
elif [ -n "${CONDA_PREFIX:-}" ] || [ -n "${VIRTUAL_ENV:-}" ]; then
  :
elif command -v conda >/dev/null 2>&1; then
  source "$(conda info --base)/etc/profile.d/conda.sh"
  conda activate "${JOYAI_CONDA_ENV:-joyai-video-edit}"
  hash -r
else
  echo "No Python environment is active. Activate one first, or set JOYAI_CONDA_SH and JOYAI_CONDA_ENV." >&2
  exit 1
fi

# ---- compile cache (MUST be exported before python imports torch) ----
export TORCHINDUCTOR_CACHE_DIR="${TORCHINDUCTOR_CACHE_DIR:-$HERE/deps/cache/torchinductor}"
export TRITON_CACHE_DIR="${TRITON_CACHE_DIR:-$HERE/deps/cache/triton}"
export CUDA_CACHE_PATH="${CUDA_CACHE_PATH:-$HERE/deps/cache/nv_compute}"
export TORCHINDUCTOR_FX_GRAPH_CACHE="${TORCHINDUCTOR_FX_GRAPH_CACHE:-1}"
mkdir -p "$TORCHINDUCTOR_CACHE_DIR" "$TRITON_CACHE_DIR" "$CUDA_CACHE_PATH"

export PYTHONUNBUFFERED=1
export PYTHONPATH="$HERE"

# PyTorch intentionally retains its `cuda` API/device spelling on ROCm. Select
# an AMD physical GPU with ROCR_VISIBLE_DEVICES alone; combining ROCr and HIP/
# CUDA masks can filter the logical device set twice. Keep the NVIDIA default
# only when the ROCr mask is absent.
if [ -z "${ROCR_VISIBLE_DEVICES:-}" ]; then
  export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
fi
DEVICE="${JOYOMNI_DEVICE:-cuda:0}"

export JOYOMNI_FP8_IMG="${JOYOMNI_FP8_IMG:-1}"
export JOYOMNI_FP8_TXT="${JOYOMNI_FP8_TXT:-0}"
export JOYOMNI_TUNABLEOP_FILE="${JOYOMNI_TUNABLEOP_FILE:-$HERE/deps/cache/tunableop_results_gfx950.csv}"

# Qualified MI350X/gfx950 streaming preset.  Direct Python callers retain the
# conservative exact-cache defaults; the ROCm launcher selects the sustained
# two-step real-time policy validated by deploy/benchmark_streaming.py.
if [ -n "${ROCR_VISIBLE_DEVICES:-}" ]; then
  export JOYOMNI_STATEFUL_VAE="${JOYOMNI_STATEFUL_VAE:-0}"
  export JOYOMNI_EXPLICIT_STREAMS="${JOYOMNI_EXPLICIT_STREAMS:-1}"
  export JOYOMNI_CACHE_LAST_DENOISE_KV="${JOYOMNI_CACHE_LAST_DENOISE_KV:-1}"
  export JOYOMNI_CLEAN_KV_PREFIX_LAYERS="${JOYOMNI_CLEAN_KV_PREFIX_LAYERS:-24}"
fi

# ---- recording ----
# Keep the real-time path free of two additional CPU video encoders by
# default. Set JOYOMNI_RECORD_DIR explicitly to enable session recording.
RECORD_DIR="${JOYOMNI_RECORD_DIR:-}"
DOWNLOAD_CRF="${JOYOMNI_DOWNLOAD_CRF:-8}"

# ---- weights (all vendored under deps/) ----
CHECKPOINT_ROOT="${JOYAI_CHECKPOINT_ROOT:-$HERE/deps/checkpoints}"
JOYAI_WEIGHTS_ROOT="${JOYAI_WEIGHTS_ROOT:-$CHECKPOINT_ROOT/JoyAI-Video-Edit}"
DIT_CKPT="${JOYAI_DIT_CKPT:-$JOYAI_WEIGHTS_ROOT/dit/joyai_video_edit_dit_0804.pth}"
VAE_CKPT="${JOYAI_VAE_CKPT:-$JOYAI_WEIGHTS_ROOT/vae}"
TE_CKPT="${JOYAI_TEXT_ENCODER_CKPT:-$CHECKPOINT_ROOT/MiMo-VL-7B-RL-2508}"
FACE_ONNX="${JOYAI_FACE_DETECTOR_ONNX:-$CHECKPOINT_ROOT/face_detection_yunet_2023mar.onnx}"
PERSON_PT="${JOYAI_PERSON_DETECTOR_PT:-$CHECKPOINT_ROOT/yolov8n.pt}"
PERSON_ONNX="${JOYAI_PERSON_DETECTOR_ONNX:-$CHECKPOINT_ROOT/yolov8n.onnx}"
HOST="${JOYAI_HOST:-0.0.0.0}"
PORT="${JOYAI_PORT:-8080}"

python xvideo/serving/serve_joyomni_streaming.py \
  --dit-ckpt          "$DIT_CKPT" \
  --vae-ckpt          "$VAE_CKPT" \
  --text-encoder-ckpt "$TE_CKPT" \
  --face-detector-onnx   "$FACE_ONNX" \
  --person-detector-pt   "$PERSON_PT" \
  --person-detector-onnx "$PERSON_ONNX" \
  --record-dir "$RECORD_DIR" \
  --download-crf "$DOWNLOAD_CRF" \
  --device "$DEVICE" \
  --vae-encode-device "$DEVICE" \
  --vae-decode-device "$DEVICE" \
  --vae-pseudo-device "$DEVICE" \
  --postprocess-device "$DEVICE" \
  --host "$HOST" --port "$PORT" \
  "$@"
