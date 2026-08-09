from __future__ import annotations

import os
import queue
import threading
import time
from contextlib import nullcontext
from dataclasses import dataclass
from typing import Any

import numpy as np
import torch
from diffusers.utils.torch_utils import randn_tensor
from einops import rearrange
from PIL import Image

from xvideo.config import ExpConfig, generate_video_image_bucket
from xvideo.models.models import load_dit, load_pipeline, build_vae
from xvideo.models.pipeline import (
    PRECISION_TO_TYPE,
)
from xvideo.utils import _dynamic_resize_from_bucket, seed_everything

DEFAULT_REFERENCE_IMG_IV2V_BASESIZE = 512

def _autocast_ctx(device_type: str, dtype: torch.dtype, enabled: bool):
    if device_type in {"cuda", "cpu"}:
        return torch.autocast(device_type=device_type, dtype=dtype, enabled=enabled)
    return nullcontext()


_FULL_WARMUP_CHUNKS = 8


def _env_enabled(name: str, default: bool = False) -> bool:
    value = os.environ.get(name)
    if value is None:
        return default
    return value.lower() not in {"0", "false", "no", "off"}

def _vae_compile_module():
    from xvideo.models.vae import vae_compile as module
    return module

@dataclass(frozen=True)
class _ProfileTimer:
    wall_start: float
    start_event: torch.cuda.Event | None = None
    device: torch.device | None = None


@dataclass(frozen=True)
class _PendingCudaTimer:
    key: str
    start_event: torch.cuda.Event
    end_event: torch.cuda.Event


_PENDING_CUDA_TIMERS_KEY = "_pending_cuda_timers"

def _profile_timer_start(
    device: torch.device | str | None = None,
    *,
    use_cuda_event: bool = True,
) -> _ProfileTimer:
    if not use_cuda_event or device is None or not torch.cuda.is_available():
        return _ProfileTimer(wall_start=time.perf_counter())
    device_obj = torch.device(device)
    if device_obj.type != "cuda":
        return _ProfileTimer(wall_start=time.perf_counter())
    start_event = torch.cuda.Event(enable_timing=True)
    start_event.record(torch.cuda.current_stream(device_obj))
    return _ProfileTimer(
        wall_start=time.perf_counter(),
        start_event=start_event,
        device=device_obj,
    )

@dataclass(frozen=True)
class StreamingSettings:
    height: int = 480
    width: int = 832
    num_inference_steps: int = 2
    seed: int = 42
    max_sequence_length: int | None = None
    enable_denormalization: bool | None = None
    max_temporal_ids: int | None = None

    freeze_kv_on_static: bool = False
    static_diff_thresh: float = 2.0
    store_clean_self_only: bool = True
    cache_last_denoise_kv: bool | None = None
    clean_kv_prefix_layers: int | None = None
    exact_global_sink_kv: bool = True
    profile_timings: bool = False

@dataclass
class StreamingChunkResult:
    profile: dict[str, Any]
    source_metas: list[dict[str, Any]]
    elapsed: float

    frames: list[Image.Image] | None = None
    pixels: torch.Tensor | None = None
    jpegs: list[bytes] | None = None

@dataclass
class _ChunkJob:
    chunk_idx: int
    source_frames: list[Image.Image]
    source_metas: list[dict[str, Any]]
    profile: dict[str, Any]
    total_started: float

    frozen_anchor_id: int | None = None

@dataclass
class _EncodedChunk:
    job: _ChunkJob
    ref_chunk_latent: torch.Tensor
    ready_event: torch.cuda.Event | None = None

@dataclass
class _DenoisedChunk:
    job: _ChunkJob
    current_chunk_latents: torch.Tensor
    ready_event: torch.cuda.Event | None = None

@dataclass
class _DecodedPixelsChunk:
    job: _ChunkJob
    decoded_pixels: torch.Tensor
    ready_event: torch.cuda.Event | None = None

def _module_device(module: torch.nn.Module) -> torch.device:
    if hasattr(module, "device"):
        return torch.device(module.device)
    try:
        return next(module.parameters()).device
    except StopIteration:
        return torch.device("cpu")

def _load_vae_for_device(cfg: ExpConfig, device: torch.device) -> torch.nn.Module:
    vae = build_vae(cfg, device)
    vae = vae.to(device)
    params = cfg.vae_arch_config.get("params", {}) if isinstance(cfg.vae_arch_config, dict) else {}
    if params.get("enable_feature_caching", False) and hasattr(vae, "enable_feature_caching"):
        vae.enable_feature_caching()
    vae.requires_grad_(False)
    vae.eval()
    return vae


def _vae_latent_hw(vae: torch.nn.Module, height: int, width: int) -> tuple[int, int]:
    spatial_factor = int(getattr(vae, "ffactor_spatial", 0) or 0)
    if spatial_factor <= 0:
        raise ValueError("VAE must expose a positive ffactor_spatial")
    canonical_height, canonical_width = int(height), int(width)
    stem = getattr(vae, "stem", None)
    if (
        stem is not None and
        canonical_height % int(stem.stride) == 0 and
        canonical_width % int(stem.stride) == 0
    ):
        canonical_height = canonical_height * int(stem.group) // int(stem.stride)
        canonical_width = canonical_width * int(stem.group) // int(stem.stride)
    return canonical_height // spatial_factor, canonical_width // spatial_factor

class JoyOmniRuntime:
    def __init__(
        self,
        cfg: ExpConfig,
        pipeline: Any,
        device: torch.device,
        *,
        decode_vae: torch.nn.Module | None = None,
        pseudo_encode_vae: torch.nn.Module | None = None,
        postprocess_device: torch.device | None = None,
        stateful_vae: bool = False,
    ):
        self.cfg = cfg
        self.pipeline = pipeline
        self.device = device
        self.decode_vae = decode_vae or pipeline.vae
        self.pseudo_encode_vae = pseudo_encode_vae or self.decode_vae
        self.postprocess_device = postprocess_device or _module_device(self.pseudo_encode_vae)
        self.stateful_vae = bool(stateful_vae)
        self.dit_lock = threading.Lock()

    @classmethod
    def load(
        cls,
        dit_ckpt: str | None,
        *,
        vae_ckpt: str | None = None,
        text_encoder_ckpt: str | None = None,
        device: str | torch.device | None = None,
        vae_device: str | torch.device | None = None,
        vae_encode_device: str | torch.device | None = None,
        vae_decode_device: str | torch.device | None = None,
        vae_pseudo_device: str | torch.device | None = None,
        postprocess_device: str | torch.device | None = None,
        seed: int = 42,
        warmup_height: int = 480,
        warmup_width: int = 832,
    ) -> "JoyOmniRuntime":
        device_obj = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        encode_device_arg = vae_encode_device if vae_encode_device is not None else vae_device
        decode_device_arg = vae_decode_device if vae_decode_device is not None else vae_device
        pseudo_device_arg = vae_pseudo_device if vae_pseudo_device is not None else decode_device_arg
        vae_encode_device_obj = torch.device(encode_device_arg) if encode_device_arg is not None else None
        vae_decode_device_obj = torch.device(decode_device_arg) if decode_device_arg is not None else None
        vae_pseudo_device_obj = torch.device(pseudo_device_arg) if pseudo_device_arg is not None else None
        stateful_vae = _env_enabled("JOYOMNI_STATEFUL_VAE", default=False)
        postprocess_default = (
            vae_decode_device_obj if stateful_vae else vae_pseudo_device_obj
        )
        postprocess_device_obj = (
            torch.device(postprocess_device)
            if postprocess_device is not None else postprocess_default
        )
        seed_everything(seed)

        cfg = ExpConfig()
        if vae_ckpt is not None:
            cfg.vae_arch_config["pretrained"] = vae_ckpt
        if text_encoder_ckpt is not None:
            cfg.text_encoder_arch_config["params"]["text_encoder_ckpt"] = text_encoder_ckpt
        if dit_ckpt is not None:
            cfg.dit_ckpt = dit_ckpt

        cfg.training_mode = False

        dit = load_dit(cfg, device=device_obj)
        if getattr(dit.config, "causal", False):
            dit.config.use_inference_kv_cache = True
        dit.requires_grad_(False)
        dit.eval()

        try:
            from xvideo.models.dit import warmup_attention_backend as _warmup_fa
            _heads = int(getattr(dit.config, "heads_num", 0) or 0)
            _hidden = int(getattr(dit.config, "hidden_size", 0) or 0)
            if _heads > 0 and _hidden > 0:
                _warmup_fa(_heads, _hidden // _heads, device=device_obj)
        except Exception as _fa_exc:  # noqa: BLE001
            print(f"#####[STREAM] FA4 warmup skipped: {_fa_exc!r}")

        pipeline = load_pipeline(cfg, dit, device_obj)
        pipeline.vae.requires_grad_(False)
        pipeline.vae.eval()
        if vae_encode_device_obj is not None:
            pipeline.vae = pipeline.vae.to(vae_encode_device_obj)
            print(f"#####[STREAM] moved encode VAE to {vae_encode_device_obj}")

        decode_vae = pipeline.vae
        if vae_decode_device_obj is not None:
            decode_vae = _load_vae_for_device(cfg, vae_decode_device_obj)
            print(f"#####[STREAM] loaded decode VAE on {vae_decode_device_obj}")
        pseudo_encode_vae = decode_vae
        if not stateful_vae and vae_pseudo_device_obj is not None:
            pseudo_encode_vae = _load_vae_for_device(cfg, vae_pseudo_device_obj)
            print(f"#####[STREAM] loaded pseudo encode VAE on {vae_pseudo_device_obj}")
        elif stateful_vae:
            print("#####[STREAM] stateful causal VAE enabled; pseudo encoder eliminated")

        if hasattr(pipeline, "set_progress_bar_config"):
            pipeline.set_progress_bar_config(disable=True)
        pipeline.transformer.eval()

        _orientations = [(warmup_height, warmup_width)]
        if (
            _env_enabled("JOYOMNI_WARMUP_BOTH_ORIENTATIONS", default=True)
            and (warmup_width, warmup_height) != (warmup_height, warmup_width)
        ):
            _orientations.append((warmup_width, warmup_height))

        try:
            _vc = _vae_compile_module()
            _lat_c = int(getattr(decode_vae, "latent_channels", 0) or 0)
            _fspa = int(getattr(decode_vae, "ffactor_spatial", 0) or 0)
            _ddev = _module_device(decode_vae)
            if _lat_c > 0 and _fspa > 0:
                for (_wh, _ww) in _orientations:
                    _latent_h, _latent_w = _vae_latent_hw(decode_vae, _wh, _ww)
                    if stateful_vae:
                        _vc.warmup_stream_decode(
                            decode_vae, _lat_c, _latent_h, _latent_w,
                            device=_ddev, dtype=PRECISION_TO_TYPE[cfg.vae_precision],
                        )
                    else:
                        _vc.warmup_decode(
                            decode_vae, _lat_c,
                            _latent_h, _latent_w,
                            device=_ddev, dtype=PRECISION_TO_TYPE[cfg.vae_precision],
                        )
        except Exception as _vc_exc:
            print(f"#####[STREAM] VAE compile warmup skipped: {_vc_exc!r}")

        try:
            _vc = _vae_compile_module()
            _vae_dt = PRECISION_TO_TYPE[cfg.vae_precision]
            _vae_ac = (_vae_dt != torch.float32)

            _src_vae = pipeline.vae
            for (_wh, _ww) in _orientations:
                if stateful_vae:
                    _vc.warmup_stream_encode(
                        _src_vae, 3, _wh, _ww,
                        device=_module_device(_src_vae), dtype=_vae_dt,
                        autocast=_vae_ac,
                    )
                else:
                    _vc.warmup_encode(
                        _src_vae, 3, _wh, _ww,
                        device=_module_device(_src_vae), dtype=_vae_dt,
                        temporal_lens=(1, 1 + int(getattr(_src_vae, "ffactor_temporal", 8) or 8)),
                        autocast=_vae_ac,
                    )
                    _vc.warmup_encode(
                        pseudo_encode_vae, 3, _wh, _ww,
                        device=_module_device(pseudo_encode_vae), dtype=_vae_dt,
                        temporal_lens=(1,),
                        autocast=_vae_ac,
                    )

            _ref_basesize = getattr(cfg, "ref_image_basesize", DEFAULT_REFERENCE_IMG_IV2V_BASESIZE)
            _ref_cfgs = generate_video_image_bucket(
                img_basesize=_ref_basesize, bs_img=1, bs_vid=0, bs_mimg=0, bs_mvid=0,
            )
            _ref_hw = sorted({(c[3], c[4]) for c in _ref_cfgs})
            _vc.maybe_setup_encode_dynamic(_src_vae)
            _vc.warmup_encode_dynamic(
                _src_vae, 3, _ref_hw,
                device=_module_device(_src_vae), dtype=_vae_dt,
                temporal_lens=(1,),
                autocast=False,
            )
        except Exception as _ve_exc:
            print(f"#####[STREAM] VAE encode compile warmup skipped: {_ve_exc!r}")

        runtime = cls(
            cfg=cfg,
            pipeline=pipeline,
            device=device_obj,
            decode_vae=decode_vae,
            pseudo_encode_vae=pseudo_encode_vae,
            postprocess_device=postprocess_device_obj,
            stateful_vae=stateful_vae,
        )

        try:
            for (_wh, _ww) in _orientations:
                runtime.warmup_full_pipeline(height=_wh, width=_ww)
        except Exception as _wexc:
            print(f"#####[STREAM] full-pipeline warmup error (non-fatal): {_wexc!r}")

        # Compile only after eager warmup: by this point the lazy FP8
        # projections already exist. Mutable Python KV dictionaries remain
        # outside this compiled 40-block tensor boundary.
        _dit_compile_mode = os.environ.get("JOYOMNI_DIT_COMPILE_MODE", "").strip()
        if _dit_compile_mode:
            try:
                _dit_compile_fullgraph = _env_enabled(
                    "JOYOMNI_DIT_COMPILE_FULLGRAPH", default=False
                )
                _dit_compile_dynamic = _env_enabled(
                    "JOYOMNI_DIT_COMPILE_DYNAMIC", default=False
                )
                _dit_recompile_limit = int(
                    os.environ.get("JOYOMNI_DIT_RECOMPILE_LIMIT", "64")
                )
                _dit_accumulated_recompile_limit = int(
                    os.environ.get(
                        "JOYOMNI_DIT_ACCUMULATED_RECOMPILE_LIMIT",
                        str(max(256, _dit_recompile_limit * 16)),
                    )
                )
                # Set both the current and legacy spellings and pass the
                # per-compile value explicitly; older compile contexts can
                # otherwise retain PyTorch's default of 8.
                torch._dynamo.config.recompile_limit = _dit_recompile_limit
                torch._dynamo.config.cache_size_limit = _dit_recompile_limit
                torch._dynamo.config.accumulated_recompile_limit = (
                    _dit_accumulated_recompile_limit
                )
                torch._dynamo.config.accumulated_cache_size_limit = (
                    _dit_accumulated_recompile_limit
                )
                print(
                    "#####[STREAM] compiling DiT block core "
                    f"mode={_dit_compile_mode!r} fullgraph={_dit_compile_fullgraph} "
                    f"dynamic={_dit_compile_dynamic} "
                    f"recompile_limit={torch._dynamo.config.recompile_limit} "
                    "accumulated_recompile_limit="
                    f"{torch._dynamo.config.accumulated_recompile_limit}",
                    flush=True,
                )
                pipeline.transformer._compiled_double_blocks = torch.compile(
                    pipeline.transformer._run_double_blocks,
                    mode=_dit_compile_mode,
                    fullgraph=_dit_compile_fullgraph,
                    dynamic=_dit_compile_dynamic,
                    recompile_limit=_dit_recompile_limit,
                    isolate_recompiles=True,
                )
                _dit_compile_warmup_chunks = int(
                    os.environ.get("JOYOMNI_DIT_COMPILE_WARMUP_CHUNKS", "3")
                )
                for (_wh, _ww) in _orientations:
                    # Inductor autotuning runs kernels on the live device.
                    # Feed one chunk at a time so another stage cannot launch
                    # VAE work concurrently and corrupt a tuning measurement.
                    runtime.warmup_full_pipeline(
                        height=_wh,
                        width=_ww,
                        num_chunks=_dit_compile_warmup_chunks,
                        serial_chunks=True,
                        raise_on_error=True,
                    )
                runtime._dit_core_compiled = True
                print("#####[STREAM] compiled DiT block-core warmup complete", flush=True)
            except Exception as _dit_compile_exc:
                print(
                    "#####[STREAM] DiT block-core compile failed; restoring eager core: "
                    f"{_dit_compile_exc!r}",
                    flush=True,
                )
                pipeline.transformer._compiled_double_blocks = None
                runtime._dit_core_compiled = False

        return runtime

    def create_v2v_session(
        self,
        prompt: str,
        *,
        settings: StreamingSettings | None = None,
        ref_image: Image.Image | None = None,
    ) -> "JoyOmniV2VStreamingSession":
        return JoyOmniV2VStreamingSession(
            runtime=self,
            prompt=prompt,
            settings=settings or StreamingSettings(),
            ref_image=ref_image,
        )

    def warmup_full_pipeline(
        self,
        *,
        height: int = 480,
        width: int = 832,
        num_chunks: int = _FULL_WARMUP_CHUNKS,
        prompt: str = "warmup",
        num_inference_steps: int = 2,
        max_temporal_ids: int | None = None,
        freeze_kv_on_static: bool = False,
        static_diff_thresh: float = 2.0,
        serial_chunks: bool = False,
        raise_on_error: bool = False,
    ) -> None:
        t0 = time.time()

        ffactor_t = int(self.pipeline.vae_scale_factor_temporal)
        n_frames = 1 + max(0, num_chunks - 1) * ffactor_t + ffactor_t
        print(f"#####[STREAM] full-pipeline warmup: {num_chunks} chunks (~{n_frames} frames) at {width}x{height}")
        session = None
        try:
            settings = StreamingSettings(
                height=height,
                width=width,
                num_inference_steps=num_inference_steps,
                max_temporal_ids=max_temporal_ids,
                freeze_kv_on_static=freeze_kv_on_static,
                static_diff_thresh=static_diff_thresh,
                profile_timings=False,
            )
            session = self.create_v2v_session(
                prompt=prompt, settings=settings,
            )
            rng = np.random.default_rng(0)
            completed = 0
            submitted = 0
            if serial_chunks:
                # The causal stream emits its first latent chunk from one
                # source frame and each subsequent chunk from ffactor_t more.
                # Waiting after every group prevents cross-stage work from
                # overlapping compiler/autotuner GPU probes.
                for chunk_idx in range(num_chunks):
                    frames_this_chunk = 1 if chunk_idx == 0 else ffactor_t
                    for _ in range(frames_this_chunk):
                        arr = rng.integers(
                            0, 256, size=(height, width, 3), dtype=np.uint8
                        )
                        frame = Image.fromarray(arr, mode="RGB")
                        submitted += 1
                        results = session.push_frame(
                            frame,
                            frame_meta={"seq": submitted, "t_capture_ms": 0.0},
                        )
                        completed += len(results)
                    while completed <= chunk_idx:
                        result = session.wait_async_result(timeout=600.0)
                        if result is None:
                            raise TimeoutError(
                                f"serial warmup timed out after chunk {chunk_idx}"
                            )
                        completed += 1
            else:
                for i in range(n_frames):
                    arr = rng.integers(
                        0, 256, size=(height, width, 3), dtype=np.uint8
                    )
                    frame = Image.fromarray(arr, mode="RGB")
                    results = session.push_frame(
                        frame, frame_meta={"seq": i + 1, "t_capture_ms": 0.0}
                    )
                    completed += len(results)
                    if completed >= num_chunks:
                        break

            deadline = time.time() + 120.0
            while completed < num_chunks and time.time() < deadline:
                r = session.wait_async_result(timeout=0.5)
                if r is not None:
                    completed += 1
            print(f"#####[STREAM] full-pipeline warmup done: {completed} chunks in {time.time() - t0:.1f}s")
        except Exception as exc:
            print(f"#####[STREAM] full-pipeline warmup skipped/failed: {exc!r}")
            if raise_on_error:
                raise
        finally:
            if session is not None:
                try:
                    session.close()
                except Exception:
                    pass

class JoyOmniV2VStreamingSession:
    def __init__(
        self,
        *,
        runtime: JoyOmniRuntime,
        prompt: str,
        settings: StreamingSettings,
        ref_image: Image.Image | None = None,
    ):
        self.runtime = runtime
        self.pipeline = runtime.pipeline

        self.decode_vae = runtime.decode_vae
        self.pseudo_encode_vae = runtime.pseudo_encode_vae
        self.postprocess_device = runtime.postprocess_device
        self.stateful_vae = runtime.stateful_vae
        self.cfg = runtime.cfg
        self.raw_prompt = prompt
        self.prompt = prompt
        self.settings = settings
        self.ref_image = ref_image.convert("RGB") if ref_image is not None else None
        self.ref_image_latent: torch.Tensor | None = None
        # Experimental latency/quality trade-off: retain the target-token K/V
        # already produced by the final denoise step instead of running a third
        # transformer pass over the clean latent.  Exact clean-cache storage
        # remains the default until the streaming quality gate is qualified.
        cache_last_denoise_kv = (
            _env_enabled("JOYOMNI_CACHE_LAST_DENOISE_KV", False)
            if settings.cache_last_denoise_kv is None
            else bool(settings.cache_last_denoise_kv)
        )
        self._cache_last_denoise_kv = (
            cache_last_denoise_kv
            and settings.store_clean_self_only
            and settings.num_inference_steps >= 2
        )
        clean_prefix_value = (
            os.environ.get("JOYOMNI_CLEAN_KV_PREFIX_LAYERS", "0")
            if settings.clean_kv_prefix_layers is None
            else settings.clean_kv_prefix_layers
        )
        self._clean_kv_prefix_layers = int(clean_prefix_value)
        num_transformer_layers = len(self.pipeline.transformer.double_blocks)
        if not 0 <= self._clean_kv_prefix_layers <= num_transformer_layers:
            raise ValueError(
                "clean KV prefix layers must be between 0 and "
                f"{num_transformer_layers}, got {self._clean_kv_prefix_layers}."
            )
        if not self._cache_last_denoise_kv:
            self._clean_kv_prefix_layers = 0
        self._exact_global_sink_kv = bool(settings.exact_global_sink_kv)
        self.device = self.pipeline.transformer.device
        self.generator = torch.Generator(device=self.device).manual_seed(settings.seed)
        # Do not draw VAE noise from the process-global generator. Encode and
        # pseudo-encode can execute on independent GPU streams, and their
        # scheduling order must not change the latent sequence for a fixed
        # session seed.
        seed_modulus = (1 << 63) - 1
        encode_seed = (int(settings.seed) + 1) % seed_modulus
        pseudo_seed = (int(settings.seed) + 2) % seed_modulus
        self._vae_encode_generator = torch.Generator(
            device=_module_device(self.pipeline.vae)
        ).manual_seed(encode_seed)
        self._vae_pseudo_generator = torch.Generator(
            device=_module_device(self.pseudo_encode_vae)
        ).manual_seed(pseudo_seed)

        self._prev_static_gray: np.ndarray | None = None
        self._static_anchor_id: int | None = None

        self.target_dtype = PRECISION_TO_TYPE[self.cfg.dit_precision]
        self.vae_dtype = PRECISION_TO_TYPE[self.cfg.vae_precision]
        self.vae_autocast_enabled = self.vae_dtype != torch.float32
        self.autocast_enabled = self.target_dtype != torch.float32
        self.device_type = self.device.type if isinstance(self.device, torch.device) else str(self.device).split(":", 1)[0]

        self.chunk_size = self.pipeline._resolve_streaming_chunk_size(None, 1)
        if self.chunk_size != 1:
            raise ValueError(
                "The online v2v session currently supports transformer chunk_size=1 only; "
                f"got {self.chunk_size}."
            )

        self.ffactor_t = int(self.pipeline.vae_scale_factor_temporal)
        self.latent_channels = int(self.pipeline.vae.config.latent_channels)
        self.latent_h, self.latent_w = _vae_latent_hw(
            self.pipeline.vae,
            settings.height,
            settings.width,
        )
        self.local_window_size = int(getattr(self.pipeline.transformer.config, "local_window_size", 1))
        self.global_sink_chunk = self.pipeline._resolve_global_sink_chunk(None, self.pipeline.transformer)
        self.enable_denormalization = (
            self.cfg.enable_denormalization
            if settings.enable_denormalization is None
            else settings.enable_denormalization
        )

        self.initialized = False
        self.chunk_idx = 0
        self.pending_frames: list[Image.Image] = []
        self.pending_metas: list[dict[str, Any]] = []
        self.prev_source_frame: torch.Tensor | None = None
        self.ref_image_kv_prefilled = False

        self.streaming_cond_embeds: torch.Tensor | None = None
        self.streaming_cond_mask: torch.Tensor | None = None
        self.last_chunk_profile: dict[str, Any] | None = None

        self._encode_queue: queue.Queue[_ChunkJob | None] | None = None
        self._dit_queue: queue.Queue[_EncodedChunk | None] | None = None
        self._decode_queue: queue.Queue[_DenoisedChunk | None] | None = None

        self._pseudo_queue: queue.Queue[_DecodedPixelsChunk | None] | None = None
        self._postprocess_queue: queue.Queue[_DecodedPixelsChunk | None] | None = None
        self._result_queue: queue.Queue[StreamingChunkResult] | None = None
        self._workers: list[threading.Thread] = []
        self._worker_error: BaseException | None = None
        self._worker_error_lock = threading.Lock()
        self._pseudo_latent_condition = threading.Condition()
        self._pseudo_latent_chunk_idx: int | None = None
        self._pseudo_latent: torch.Tensor | None = None
        self._pseudo_latent_ready_event: torch.cuda.Event | None = None
        # Compiled stateful VAE graphs currently hang when first replayed from
        # a non-default ROCm stream (the decode worker stalls while every
        # upstream queue fills). Keep the proven default-stream execution path
        # unless explicit streams are deliberately enabled for qualification.
        self._use_explicit_streams = (
            self.device_type == "cuda" and
            os.environ.get("JOYOMNI_EXPLICIT_STREAMS", "0").lower()
            not in {"0", "false", "no", "off"}
        )
        self._encode_stream = self._new_stage_stream(_module_device(self.pipeline.vae))
        self._dit_stream = self._new_stage_stream(self.device)
        self._decode_stream = self._new_stage_stream(_module_device(self.decode_vae))
        self._pseudo_stream = (
            None if self.stateful_vae
            else self._new_stage_stream(_module_device(self.pseudo_encode_vae))
        )
        self._postprocess_stream = self._new_stage_stream(self.postprocess_device)
        self._debug_lock = threading.Lock()
        self._debug_started_at = time.time()
        self._debug_worker_states: dict[str, dict[str, Any]] = {}
        self._debug_counters: dict[str, int] = {
            "submitted_chunks": 0,
            "encoded_chunks": 0,
            "denoised_chunks": 0,
            "decoded_pixel_chunks": 0,
            "pseudo_encoded_chunks": 0,
            "postprocessed_chunks": 0,
            "returned_results": 0,
            "dropped_results": 0,
        }

    def _new_stage_stream(self, device: torch.device | str) -> torch.cuda.Stream | None:
        device_obj = torch.device(device)
        if not self._use_explicit_streams or device_obj.type != "cuda":
            return None
        return torch.cuda.Stream(device=device_obj)

    @staticmethod
    def _stage_stream_context(stream: torch.cuda.Stream | None):
        return torch.cuda.stream(stream) if stream is not None else nullcontext()

    @staticmethod
    def _record_stage_ready(stream: torch.cuda.Stream | None) -> torch.cuda.Event | None:
        if stream is None:
            return None
        ready = torch.cuda.Event()
        ready.record(stream)
        return ready

    @staticmethod
    def _wait_stage_ready(
        stream: torch.cuda.Stream | None,
        ready: torch.cuda.Event | None,
        tensor: torch.Tensor | None = None,
    ) -> None:
        if stream is None or ready is None:
            return
        stream.wait_event(ready)
        if tensor is not None and tensor.device.type == "cuda":
            # The caching allocator otherwise only knows the producer stream.
            # Keep the storage alive until this consumer stream has finished.
            tensor.record_stream(stream)

    def _prime_stage_streams(self) -> None:
        """Make each worker stream wait for session initialization."""
        streams = [
            self._encode_stream,
            self._dit_stream,
            self._decode_stream,
            self._pseudo_stream,
            self._postprocess_stream,
        ]
        by_device: dict[torch.device, list[torch.cuda.Stream]] = {}
        for stream in streams:
            if stream is not None:
                by_device.setdefault(torch.device(stream.device), []).append(stream)
        for device, device_streams in by_device.items():
            initialized = torch.cuda.Event()
            initialized.record(torch.cuda.current_stream(device))
            for stream in device_streams:
                stream.wait_event(initialized)

    def _synchronize_stage_devices(self) -> None:
        devices = {
            torch.device(stream.device)
            for stream in (
                self._encode_stream,
                self._dit_stream,
                self._decode_stream,
                self._pseudo_stream,
                self._postprocess_stream,
            )
            if stream is not None
        }
        for device in devices:
            torch.cuda.synchronize(device)

    def _timer_start(
        self,
        device: torch.device | str | None = None,
        *,
        use_cuda_event: bool = True,
    ) -> _ProfileTimer | float:
        if self.settings.profile_timings:
            return _profile_timer_start(device, use_cuda_event=use_cuda_event)
        return time.perf_counter()

    def _timer_record(
        self,
        profile: dict[str, Any],
        key: str,
        started: _ProfileTimer | float,
    ) -> None:
        if not self.settings.profile_timings:
            return
        if isinstance(started, _ProfileTimer):
            if started.start_event is not None and started.device is not None:
                end_event = torch.cuda.Event(enable_timing=True)
                end_event.record(torch.cuda.current_stream(started.device))
                profile.setdefault(_PENDING_CUDA_TIMERS_KEY, []).append(
                    _PendingCudaTimer(key, started.start_event, end_event)
                )
                return
            elapsed = time.perf_counter() - started.wall_start
        else:
            elapsed = time.perf_counter() - float(started)
        profile[key] = float(profile.get(key, 0.0)) + elapsed

    @staticmethod
    def _resolve_cuda_profile_timers(profile: dict[str, Any]) -> None:
        pending = profile.pop(_PENDING_CUDA_TIMERS_KEY, [])
        if not pending:
            return
        # Resolve at the chunk boundary, after the output D2H transfer has
        # already established completion. Synchronizing every individual stage
        # at record time would serialize the otherwise asynchronous GPU pipe.
        for timer in pending:
            timer.end_event.synchronize()
        for timer in pending:
            elapsed = float(timer.start_event.elapsed_time(timer.end_event)) / 1000.0
            profile[timer.key] = float(profile.get(timer.key, 0.0)) + elapsed

    def _record_queue_wait(
        self,
        profile: dict[str, Any],
        key: str,
        started: float,
    ) -> None:
        if not self.settings.profile_timings:
            return
        profile[key] = time.perf_counter() - started

    @property
    def frames_per_next_chunk(self) -> int:
        return 1 if self.chunk_idx == 0 else self.ffactor_t

    def _clear_vae_feature_caches(self) -> None:
        seen: set[int] = set()
        for vae in (self.pipeline.vae, self.decode_vae, self.pseudo_encode_vae):
            if vae is None or id(vae) in seen:
                continue
            seen.add(id(vae))
            clear_cache = getattr(vae, "clear_cache", None)
            if callable(clear_cache):
                clear_cache()

    @torch.no_grad()
    def push_frame(
        self,
        frame: Image.Image,
        frame_meta: dict[str, Any] | None = None,
        *,
        drain_results: bool = True,
    ) -> list[StreamingChunkResult]:
        self._raise_worker_error_if_needed()
        frame = self._resize_frame(frame)
        meta = frame_meta or {"seq": self.chunk_idx + len(self.pending_frames) + 1, "t_capture_ms": time.time() * 1000.0}
        results: list[StreamingChunkResult] = []

        if not self.initialized:
            self._initialize(frame)
            self._start_async_workers()
            results.extend(self._submit_or_process_chunk([frame], [meta]))
            if drain_results:
                results.extend(self._drain_async_results())
            self._raise_worker_error_if_needed()
            return results

        self.pending_frames.append(frame)
        self.pending_metas.append(meta)
        while len(self.pending_frames) >= self.ffactor_t:
            chunk_frames = self.pending_frames[: self.ffactor_t]
            chunk_metas = self.pending_metas[: self.ffactor_t]
            del self.pending_frames[: self.ffactor_t]
            del self.pending_metas[: self.ffactor_t]
            results.extend(self._submit_or_process_chunk(chunk_frames, chunk_metas))
        if drain_results:
            results.extend(self._drain_async_results())
        self._raise_worker_error_if_needed()
        return results

    def flush_pending_for_reset(self) -> int:
        """Submit a final padded chunk before an intentional state reset.

        A cut can arrive after one to seven pre-cut frames have accumulated.
        Dropping those frames creates a visible hole in the downlink.  Pad the
        partial chunk with its final pre-cut frame so the causal VAE never sees
        the new scene, while allowing the client to play every accepted frame
        (plus less than one padded chunk at the boundary).
        """
        self._raise_worker_error_if_needed()
        if not self.initialized or not self.pending_frames:
            return 0

        source_count = len(self.pending_frames)
        chunk_frames = list(self.pending_frames)
        chunk_metas = list(self.pending_metas)
        self.pending_frames.clear()
        self.pending_metas.clear()

        final_frame = chunk_frames[-1]
        final_meta = chunk_metas[-1] if chunk_metas else {}
        while len(chunk_frames) < self.ffactor_t:
            chunk_frames.append(final_frame)
            chunk_metas.append(dict(final_meta))

        self._submit_or_process_chunk(chunk_frames, chunk_metas)
        return source_count

    def close(self) -> list[StreamingChunkResult]:
        # Quiesce the pipeline before mutating feature/KV caches.  Workers may
        # still be draining queued chunks during warmup, restart, or a client
        # disconnect; clearing live cache state races their model forwards.
        completed = self._stop_async_workers()
        self._clear_vae_feature_caches()
        self.pipeline.transformer.reset_inference_kv_cache()
        return completed

    @torch.no_grad()
    def _initialize(self, first_frame: Image.Image) -> None:
        if self.initialized:
            return

        if not getattr(self.pipeline.transformer.config, "causal", False):
            raise ValueError("Online streaming requires a causal transformer config.")
        self.pipeline.transformer.config.use_inference_kv_cache = True
        self.pipeline._interrupt = False

        self.ref_image_latent = self._encode_ref_image_latent()

        self._encode_streaming_prompt(first_frame)

        self._clear_vae_feature_caches()
        self.pipeline.transformer.reset_inference_kv_cache()
        if self.ref_image_latent is not None:
            self.pipeline._prefill_static_reference_kv_cache(
                self.pipeline.transformer,
                prompt_embeds=self.streaming_cond_embeds,
                prompt_embeds_mask=self.streaming_cond_mask,
                reference_image_latents=self.ref_image_latent,
                transformer_dtype=self.target_dtype,
            )
            self.ref_image_kv_prefilled = True

        self.initialized = True

    @torch.no_grad()
    def _encode_streaming_prompt(self, anchor_frame: Image.Image) -> None:
        prompt_image = self._resize_frame(anchor_frame)
        prompt = f"<|im_start|>user\n<image>\n{self.prompt}<|im_end|>\n"

        max_sequence_length = self.settings.max_sequence_length or int(self.cfg.text_token_max_length)
        prompt_embeds, prompt_mask = self.pipeline.encode_prompt(
            prompt=[prompt],
            images=[prompt_image],
            device=self.device,
            num_videos_per_prompt=1,
            max_sequence_length=max_sequence_length,
            template_type="video",
        )

        self.streaming_cond_embeds = prompt_embeds
        self.streaming_cond_mask = prompt_mask

    @torch.no_grad()
    def _encode_ref_image_latent(self) -> torch.Tensor | None:
        if self.ref_image is None:
            return None

        ref_image_basesize = getattr(self.cfg, "ref_image_basesize", DEFAULT_REFERENCE_IMG_IV2V_BASESIZE)
        ref_image_bucket_configs = generate_video_image_bucket(
            img_basesize=ref_image_basesize,
            bs_img=1,
            bs_vid=0,
            bs_mimg=0,
            bs_mvid=0,
        )
        resized_image, _ = _dynamic_resize_from_bucket(
            self.ref_image,
            bucket_configs=ref_image_bucket_configs,
            num_frames=1,
            num_items=1,
            return_bucket=True,
        )
        pixel = torch.from_numpy(np.asarray(resized_image))
        pixel = rearrange(pixel, "h w c -> c h w")
        if pixel.dtype != torch.uint8:
            pixel = pixel.clamp(0, 255).to(torch.uint8)
        pixel_normalized = pixel.to(torch.float32) / 127.5 - 1.0

        ref_img_tensor = rearrange(pixel_normalized, "c h w -> 1 c 1 h w")
        encode_device = _module_device(self.pipeline.vae)
        ref_img_encoded = ref_img_tensor.to(device=encode_device, dtype=self.vae_dtype)

        _vc = _vae_compile_module()
        if self.enable_denormalization:
            encoded = _vc.encode_via_dynamic(self.pipeline.vae, ref_img_encoded)
            if hasattr(encoded, "latent_dist"):
                ref_img_latent = encoded.latent_dist.sample(
                    generator=self._vae_encode_generator
                )
                ref_img_latent = self.pipeline.normalize_latents(ref_img_latent)
            elif torch.is_tensor(encoded):
                posterior = self.pipeline.vae.encode(ref_img_encoded, return_posterior=True)
                ref_img_latent = self.pipeline.normalize_latents(
                    posterior.sample(generator=self._vae_encode_generator)
                )
            else:
                raise TypeError(f"Unsupported VAE encode output type for ref image: {type(encoded)}")
        else:
            ref_img_latent = _vc.encode_via_dynamic(self.pipeline.vae, ref_img_encoded)
            if hasattr(ref_img_latent, "latent_dist"):
                ref_img_latent = ref_img_latent.latent_dist.sample(
                    generator=self._vae_encode_generator
                )
            if not torch.is_tensor(ref_img_latent):
                raise TypeError(f"Unsupported VAE encode output type for ref image: {type(ref_img_latent)}")

        return ref_img_latent[:, :, :1].to(device=self.device, dtype=self.target_dtype)

    @staticmethod
    def _chunk_last_frame_gray(source_frames: list[Image.Image]) -> np.ndarray | None:
        if not source_frames:
            return None
        frame = source_frames[-1].convert("L").resize((64, 36))
        return np.asarray(frame, dtype=np.float32)

    def _update_static_anchor(self, chunk_idx: int, source_frames: list[Image.Image]) -> int | None:
        gray = self._chunk_last_frame_gray(source_frames)
        if not self.settings.freeze_kv_on_static or chunk_idx == 0 or gray is None:
            self._prev_static_gray = gray if gray is not None else self._prev_static_gray
            self._static_anchor_id = None
            return None

        anchor_id: int | None = None
        if self._prev_static_gray is not None:
            mad = float(np.mean(np.abs(gray - self._prev_static_gray)))
            if mad < self.settings.static_diff_thresh:
                if self._static_anchor_id is None:
                    self._static_anchor_id = chunk_idx - 1
                    print(
                        f"#####[FREEZE-KV] static detected at chunk_idx={chunk_idx} "
                        f"(mad={mad:.3f} < {self.settings.static_diff_thresh}), "
                        f"anchor={self._static_anchor_id}",
                        flush=True,
                    )
                anchor_id = self._static_anchor_id
            else:
                if self._static_anchor_id is not None:
                    print(
                        f"#####[FREEZE-KV] motion resumed at chunk_idx={chunk_idx} "
                        f"(mad={mad:.3f}), anchor released",
                        flush=True,
                    )
                self._static_anchor_id = None
                anchor_id = None

        self._prev_static_gray = gray
        return anchor_id

    def _submit_or_process_chunk(
        self,
        source_frames: list[Image.Image],
        source_metas: list[dict[str, Any]],
    ) -> list[StreamingChunkResult]:
        self._submit_async_chunk(source_frames, source_metas)
        return []

    def _new_profile(self, chunk_idx: int, input_frames: int) -> dict[str, Any]:
        return {
            "chunk_idx": chunk_idx,
            "input_frames": input_frames,
            "steps": int(self.settings.num_inference_steps),
            "profile_timings": int(self.settings.profile_timings),
            "profile_timing_mode": "deferred_cuda_events" if self.settings.profile_timings else "wall_only",
            "gpu_stream_mode": "explicit_per_stage" if self._use_explicit_streams else "default",
            "stateful_vae": int(self.stateful_vae),
            "kv_store_strategy": (
                (
                    (
                        f"last_denoise_clean_prefix_{self._clean_kv_prefix_layers}"
                        + ("_exact_global_sink" if self._exact_global_sink_kv else "")
                    )
                    if self._clean_kv_prefix_layers
                    else (
                        "last_denoise_approx_exact_global_sink"
                        if self._exact_global_sink_kv
                        else "last_denoise_approx"
                    )
                )
                if self._cache_last_denoise_kv
                else "clean_exact"
            ),
            "dit_device": str(self.device),
            "vae_encode_device": str(_module_device(self.pipeline.vae)),
            "vae_decode_device": str(_module_device(self.decode_vae)),
            "vae_pseudo_device": str(_module_device(self.pseudo_encode_vae)),
            "postprocess_device": str(self.postprocess_device),
        }

    def _clean_kv_store_plan(self, active_chunk_id: int) -> tuple[int, bool]:
        """Return the exact clean-store depth and whether this is the sink refresh."""
        exact_global_sink = bool(
            self._cache_last_denoise_kv
            and self._exact_global_sink_kv
            and self.global_sink_chunk
            and active_chunk_id == 0
        )
        layers = (
            len(self.pipeline.transformer.double_blocks)
            if exact_global_sink
            else self._clean_kv_prefix_layers
        )
        return int(layers), exact_global_sink

    def _denoise_chunk(
        self,
        ref_chunk_latent: torch.Tensor,
        *,
        profile: dict[str, Any],
        chunk_idx: int,
        frozen_anchor_id: int | None = None,
    ) -> torch.Tensor:
        noise_shape = (1, self.latent_channels, self.chunk_size, self.latent_h, self.latent_w)
        current_chunk_latents = randn_tensor(
            noise_shape,
            generator=self.generator,
            device=self.device,
            dtype=self.target_dtype,
        )

        total_latent_frames = chunk_idx + 1
        chunk_windows = self.pipeline._get_chunk_windows(
            total_latent_frames=total_latent_frames,
            chunk_size=self.chunk_size,
            window_size=self.local_window_size,
            global_sink_chunk=self.global_sink_chunk,
        )
        chunk_window = chunk_windows[-1]
        selected_chunk_ids = chunk_window["selected_chunk_ids"]
        history_chunk_ids = selected_chunk_ids[:-1]
        active_chunk_id = selected_chunk_ids[-1]

        gather_chunk_ids = selected_chunk_ids
        if (
            frozen_anchor_id is not None and
            history_chunk_ids and
            history_chunk_ids[-1] != frozen_anchor_id
        ):
            old_tail = history_chunk_ids[-1]

            fz_history = history_chunk_ids[:-1]
            anchor_is_dup = frozen_anchor_id in fz_history
            fz_history_with_anchor = fz_history if anchor_is_dup else fz_history + [frozen_anchor_id]
            fz_selected = fz_history_with_anchor + [active_chunk_id]

            disguised_tail = active_chunk_id - 1
            fz_gather_head = fz_history_with_anchor[:-1]
            degenerate = (
                anchor_is_dup or
                disguised_tail < 0 or
                disguised_tail in fz_gather_head
            )
            if not degenerate:
                history_chunk_ids = fz_history_with_anchor
                selected_chunk_ids = fz_selected
                gather_chunk_ids = fz_gather_head + [disguised_tail, active_chunk_id]
                print(
                    f"#####[FREEZE-KV] chunk_idx={chunk_idx} tail {old_tail}->anchor "
                    f"{frozen_anchor_id} (KV), pos-disguise->{disguised_tail}, "
                    f"kv_window={selected_chunk_ids} pos_window={gather_chunk_ids}",
                    flush=True,
                )
            else:
                print(
                    f"#####[FREEZE-KV] chunk_idx={chunk_idx} anchor={frozen_anchor_id} "
                    f"degenerate (dup_or_collision), falling back to normal path",
                    flush=True,
                )

        window_ids = self.pipeline._gather_window_temporal_ids(
            gather_chunk_ids,
            self.chunk_size,
            total_latent_frames,
            self.device,
            max_temporal_ids=self.settings.max_temporal_ids,
        )
        current_chunk_temporal_ids = window_ids[-self.chunk_size:]
        cached_temporal_ids = window_ids[: -self.chunk_size]
        if cached_temporal_ids.numel() == 0:
            cached_temporal_ids = None

        self.pipeline.scheduler.set_timesteps(self.settings.num_inference_steps, device=self.device)
        timesteps_for_chunk = self.pipeline.scheduler.timesteps
        current_temporal_ids_batch = current_chunk_temporal_ids.unsqueeze(0).expand(
            current_chunk_latents.shape[0], -1
        )
        cached_temporal_ids_batch = (
            cached_temporal_ids.unsqueeze(0).expand(current_chunk_latents.shape[0], -1)
            if cached_temporal_ids is not None
            else None
        )

        for step_idx, timestep in enumerate(timesteps_for_chunk):
            store_from_this_step = (
                self._cache_last_denoise_kv
                and step_idx == len(timesteps_for_chunk) - 1
            )
            autocast_context = _autocast_ctx(
                self.device_type, self.target_dtype, self.autocast_enabled
            )
            with autocast_context:
                latent_model_input = current_chunk_latents.to(self.target_dtype)
                cache_memory_ids = [
                    self.pipeline._kv_cache_memory_id("clean", cid) for cid in history_chunk_ids
                ]
                if self.ref_image_kv_prefilled:
                    cache_memory_ids.append(self.pipeline._kv_cache_memory_id("ref_image"))

                t_expand = timestep.repeat(latent_model_input.shape[0])
                with self.pipeline.transformer.cache_context("cond"):
                    started = self._timer_start(self.device)
                    noise_pred = self.pipeline.transformer(
                        hidden_states=latent_model_input,
                        timestep=t_expand,
                        encoder_hidden_states=self.streaming_cond_embeds,
                        encoder_hidden_states_mask=self.streaming_cond_mask,
                        ref_video_latent=ref_chunk_latent,
                        current_temporal_ids=current_temporal_ids_batch,
                        cached_temporal_ids=cached_temporal_ids_batch,
                        kv_cache_mode="reuse_store" if store_from_this_step else "reuse",
                        kv_cache_scope="cond",
                        kv_cache_chunk_id=active_chunk_id,
                        kv_cache_selected_chunk_ids=cache_memory_ids,
                        kv_cache_pre_rope=True,
                        kv_cache_store_current_only=store_from_this_step,
                    )[0]
                    self._timer_record(profile, "dit_forward_cond_s", started)
                    self._timer_record(
                        profile, f"dit_forward_cond_step_{step_idx}_s", started
                    )

                current_chunk_latents = self.pipeline.scheduler.step(
                    noise_pred,
                    timestep,
                    current_chunk_latents,
                    return_dict=False,
                )[0]

        if self._cache_last_denoise_kv:
            # The active cache was written as part of the last denoise forward.
            # Optionally overwrite an exact prefix from the final clean latent;
            # untouched later layers retain their last-denoise entries. Chunk 0
            # is different: global-sink attention retains it for the entire
            # session, so leaving its later layers at a noisy denoise state
            # permanently injects the first pose and accumulates ghosting. Pay
            # for one full clean store at startup; tail chunks remain hybrid
            # and are evicted after the bounded local window advances.
            clean_store_layers, exact_global_sink = self._clean_kv_store_plan(active_chunk_id)
            profile["clean_kv_store_layers"] = int(clean_store_layers)
            profile["exact_global_sink_refresh"] = int(exact_global_sink)
            if clean_store_layers:
                started = self._timer_start(self.device, use_cuda_event=False)
                self._timer_record(profile, "kv_store_setup_s", started)
                started = self._timer_start(self.device)
                self.pipeline._store_clean_chunk_kv_cache(
                    self.pipeline.transformer,
                    clean_chunk_latents=current_chunk_latents.to(self.target_dtype),
                    chunk_temporal_ids=current_chunk_temporal_ids.unsqueeze(0).expand(
                        current_chunk_latents.shape[0], -1
                    ),
                    prompt_embeds=self.streaming_cond_embeds,
                    prompt_embeds_mask=self.streaming_cond_mask,
                    active_chunk_id=active_chunk_id,
                    history_chunk_ids=[],
                    pre_rope=True,
                    cached_temporal_ids=None,
                    store_mode="store",
                    store_prefix_layers=clean_store_layers,
                )
                self._timer_record(profile, "kv_store_forward_s", started)
            # `_evict_after_store` immediately applies the next-window policy.
            return current_chunk_latents

        keep_before_store = {self.pipeline._kv_cache_memory_id("clean", cid) for cid in history_chunk_ids}
        if self.ref_image_kv_prefilled:
            keep_before_store.add(self.pipeline._kv_cache_memory_id("ref_image"))
        self.pipeline.transformer.evict_kv_cache_chunks(keep_before_store)

        started = self._timer_start(self.device, use_cuda_event=False)
        store_self_only = self.settings.store_clean_self_only
        store_history_chunk_ids = [] if store_self_only else history_chunk_ids
        store_mode = "store" if store_self_only else "reuse_store"
        store_cached_temporal_ids = None if store_self_only else cached_temporal_ids
        self._timer_record(profile, "kv_store_setup_s", started)

        started = self._timer_start(self.device)
        self.pipeline._store_clean_chunk_kv_cache(
            self.pipeline.transformer,
            clean_chunk_latents=current_chunk_latents.to(self.target_dtype),
            chunk_temporal_ids=current_chunk_temporal_ids.unsqueeze(0).expand(
                current_chunk_latents.shape[0], -1
            ),
            prompt_embeds=self.streaming_cond_embeds,
            prompt_embeds_mask=self.streaming_cond_mask,
            active_chunk_id=active_chunk_id,
            history_chunk_ids=store_history_chunk_ids,
            pre_rope=True,
            cached_temporal_ids=store_cached_temporal_ids,
            store_mode=store_mode,
        )
        self._timer_record(profile, "kv_store_forward_s", started)
        return current_chunk_latents

    def _evict_after_store(
        self, chunk_idx: int, profile: dict[str, Any], frozen_anchor_id: int | None = None
    ) -> None:
        next_selected = self._next_selected_chunk_ids(chunk_idx)
        keep_after_store = {self.pipeline._kv_cache_memory_id("clean", cid) for cid in next_selected}
        if self.ref_image_kv_prefilled:
            keep_after_store.add(self.pipeline._kv_cache_memory_id("ref_image"))

        if frozen_anchor_id is not None:
            keep_after_store.add(self.pipeline._kv_cache_memory_id("clean", frozen_anchor_id))
        self.pipeline.transformer.evict_kv_cache_chunks(keep_after_store)

    def _finish_chunk_profile(
        self,
        profile: dict[str, Any],
        total_started: float,
    ) -> None:
        if self.settings.profile_timings:
            self._resolve_cuda_profile_timers(profile)
        if self.settings.profile_timings:
            self._timer_record(profile, "total_server_chunk_s", total_started)
        else:
            profile["total_server_chunk_s"] = time.perf_counter() - total_started
        if self.settings.profile_timings:
            profile["dit_denoise_s"] = float(profile.get("dit_forward_cond_s", 0.0))
            profile["kv_store_s"] = (
                float(profile.get("kv_store_setup_s", 0.0)) +
                float(profile.get("kv_store_forward_s", 0.0))
            )
        self.last_chunk_profile = profile

    def _log_chunk_profile(self, profile: dict[str, Any], input_frames: int, output_frames: int) -> None:
        elapsed = float(profile["total_server_chunk_s"])
        if self.settings.profile_timings:
            print(
                f"#####[STREAM] chunk={profile['chunk_idx']} in_frames={input_frames} "
                f"out_frames={output_frames} elapsed={elapsed:.3f}s "
                f"upload={float(profile.get('frames_to_tensor_gpu_s', 0.0)):.3f}s "
                f"vae_enc={float(profile.get('vae_encode_s', 0.0)):.3f}s "
                f"dit={float(profile.get('dit_denoise_s', 0.0)):.3f}s "
                f"kv_store={float(profile.get('kv_store_s', 0.0)):.3f}s "
                f"vae_dec={float(profile.get('vae_decode_s', 0.0)):.3f}s "
                f"pack={float(profile.get('pack_frames_s', 0.0)):.3f}s "
                f"pack_gpu={float(profile.get('pack_frames_gpu_s', 0.0)):.3f}s"
            )

            print(
                f"#####[STREAM-QWAIT] chunk={profile['chunk_idx']} "
                f"enc={float(profile.get('q_wait_encode_s', 0.0)) * 1000:.0f}ms "
                f"dit={float(profile.get('q_wait_dit_s', 0.0)) * 1000:.0f}ms "
                f"dec={float(profile.get('q_wait_decode_s', 0.0)) * 1000:.0f}ms "
                f"pseudo={float(profile.get('q_wait_pseudo_s', 0.0)) * 1000:.0f}ms "
                f"post={float(profile.get('q_wait_postprocess_s', 0.0)) * 1000:.0f}ms",
                flush=True,
            )
        else:
            print(
                f"#####[STREAM] chunk={profile['chunk_idx']} in_frames={input_frames} "
                f"out_frames={output_frames} elapsed={elapsed:.3f}s profile_timings=off"
            )

    @staticmethod
    def _align_source_metas(
        source_metas: list[dict[str, Any]],
        output_count: int,
    ) -> list[dict[str, Any]]:
        if not source_metas:
            return [{} for _ in range(output_count)]
        if len(source_metas) >= output_count:
            return source_metas[-output_count:]
        return [source_metas[-1]] * output_count

    def _start_async_workers(self) -> None:
        if self._encode_queue is not None:
            return
        self._prime_stage_streams()
        self._encode_queue = queue.Queue(maxsize=4)
        self._dit_queue = queue.Queue(maxsize=4)
        self._decode_queue = queue.Queue(maxsize=4)
        self._pseudo_queue = None if self.stateful_vae else queue.Queue(maxsize=4)
        self._postprocess_queue = queue.Queue(maxsize=4)
        # Raw 720p RGB chunks are about 21 MiB each.  Bound this handoff so a
        # slow/disconnected WebSocket cannot grow host RAM without limit.
        # Dropping the oldest completed chunk preserves realtime latency; each
        # subsequently encoded H.264 chunk starts with an IDR frame.
        self._result_queue = queue.Queue(maxsize=4)
        self._workers = [
            threading.Thread(target=self._encode_worker, name="joyomni-vae-encode", daemon=True),
            threading.Thread(target=self._dit_worker, name="joyomni-dit-denoise", daemon=True),
            threading.Thread(target=self._decode_worker, name="joyomni-vae-decode", daemon=True),
            threading.Thread(target=self._postprocess_worker, name="joyomni-postprocess", daemon=True),
        ]
        if not self.stateful_vae:
            self._workers.insert(
                3,
                threading.Thread(
                    target=self._pseudo_worker,
                    name="joyomni-pseudo-encode",
                    daemon=True,
                ),
            )
        for worker in self._workers:
            worker.start()

    def _stop_async_workers(self) -> list[StreamingChunkResult]:
        if self._encode_queue is None:
            return []
        try:
            self._encode_queue.put(None, timeout=0.1)
        except queue.Full:
            if self._worker_error is None:
                self._encode_queue.put(None)
        for worker in self._workers:
            worker.join()
        # Worker functions enqueue asynchronously.  Quiesce every stage before
        # close() clears model feature/KV caches or releases queued tensors.
        self._synchronize_stage_devices()
        completed = self._drain_async_results()
        self._encode_queue = None
        self._dit_queue = None
        self._decode_queue = None
        self._pseudo_queue = None
        self._postprocess_queue = None
        self._result_queue = None
        self._workers = []
        return completed

    def _set_worker_error(self, exc: BaseException) -> None:
        import traceback as _tb
        print(f"#####[STREAM-WORKER-ERROR] {exc!r}", flush=True)
        _tb.print_exc()
        with self._worker_error_lock:
            if self._worker_error is None:
                self._worker_error = exc
        with self._pseudo_latent_condition:
            self._pseudo_latent_condition.notify_all()

    def _raise_worker_error_if_needed(self) -> None:
        with self._worker_error_lock:
            exc = self._worker_error
            self._worker_error = None
        if exc is not None:
            raise RuntimeError(f"async streaming worker failed: {exc!r}") from exc

    @staticmethod
    def _queue_size(q: queue.Queue | None) -> int | None:
        if q is None:
            return None
        try:
            return q.qsize()
        except NotImplementedError:
            return None

    def _set_debug_state(self, worker: str, state: str, chunk_idx: int | None = None) -> None:
        with self._debug_lock:
            self._debug_worker_states[worker] = {
                "state": state,
                "chunk_idx": chunk_idx,
                "at": time.time(),
            }

    def _inc_debug_counter(self, key: str, amount: int = 1) -> None:
        with self._debug_lock:
            self._debug_counters[key] = int(self._debug_counters.get(key, 0)) + amount

    def debug_snapshot(self) -> dict[str, Any]:
        now = time.time()
        with self._debug_lock:
            worker_states = {
                name: {
                    **state,
                    "age_s": now - float(state.get("at", now)),
                }
                for name, state in self._debug_worker_states.items()
            }
            counters = dict(self._debug_counters)
        return {
            "age_s": now - self._debug_started_at,
            "initialized": self.initialized,
            "has_ref_image": self.ref_image is not None,
            "chunk_idx_next": self.chunk_idx,
            "pending_frames": len(self.pending_frames),
            "frames_per_next_chunk": self.frames_per_next_chunk,
            "raw_prompt": self.raw_prompt,
            "prompt": self.prompt,
            "queues": {
                "encode": self._queue_size(self._encode_queue),
                "dit": self._queue_size(self._dit_queue),
                "decode": self._queue_size(self._decode_queue),
                "pseudo": self._queue_size(self._pseudo_queue),
                "postprocess": self._queue_size(self._postprocess_queue),
                "result": self._queue_size(self._result_queue),
            },
            "queue_maxsize": {
                "encode": self._encode_queue.maxsize if self._encode_queue is not None else None,
                "dit": self._dit_queue.maxsize if self._dit_queue is not None else None,
                "decode": self._decode_queue.maxsize if self._decode_queue is not None else None,
                "pseudo": self._pseudo_queue.maxsize if self._pseudo_queue is not None else None,
                "postprocess": self._postprocess_queue.maxsize if self._postprocess_queue is not None else None,
                "result": self._result_queue.maxsize if self._result_queue is not None else None,
            },
            "workers": {
                worker.name: {
                    "alive": worker.is_alive(),
                    "state": worker_states.get(worker.name.replace("joyomni-", ""), {}),
                }
                for worker in self._workers
            },
            "worker_states": worker_states,
            "counters": counters,
            "pseudo_latent_chunk_idx": self._pseudo_latent_chunk_idx,
            "last_chunk_profile": self.last_chunk_profile,
            "devices": {
                "dit": str(self.device),
                "vae_encode": str(_module_device(self.pipeline.vae)),
                "vae_decode": str(_module_device(self.decode_vae)),
                "vae_pseudo": str(_module_device(self.pseudo_encode_vae)),
                "postprocess": str(self.postprocess_device),
            },
        }

    def _submit_async_chunk(
        self,
        source_frames: list[Image.Image],
        source_metas: list[dict[str, Any]],
    ) -> None:
        if self._encode_queue is None:
            self._start_async_workers()
        assert self._encode_queue is not None
        chunk_idx = self.chunk_idx

        frozen_anchor_id = self._update_static_anchor(chunk_idx, source_frames)
        job = _ChunkJob(
            chunk_idx=chunk_idx,
            source_frames=source_frames,
            source_metas=source_metas,
            profile=self._new_profile(chunk_idx, len(source_frames)),
            total_started=time.perf_counter(),
            frozen_anchor_id=frozen_anchor_id,
        )
        self._set_debug_state("submit", "put_encode", chunk_idx)

        while True:
            self._raise_worker_error_if_needed()
            try:
                self._encode_queue.put(job, timeout=1.0)
                break
            except queue.Full:
                continue
        self._inc_debug_counter("submitted_chunks")
        self.chunk_idx += 1

    def _drain_async_results(self) -> list[StreamingChunkResult]:
        if self._result_queue is None:
            return []
        results: list[StreamingChunkResult] = []
        while True:
            try:
                results.append(self._result_queue.get_nowait())
            except queue.Empty:
                break
        return results

    def wait_async_result(self, timeout: float = 0.05) -> StreamingChunkResult | None:
        self._raise_worker_error_if_needed()
        if self._result_queue is None:
            time.sleep(timeout)
            return None
        try:
            result = self._result_queue.get(timeout=timeout)
            self._inc_debug_counter("returned_results")
            return result
        except queue.Empty:
            self._raise_worker_error_if_needed()
            return None

    def _encode_worker(self) -> None:
        assert self._encode_queue is not None
        assert self._dit_queue is not None
        while True:
            self._set_debug_state("vae-encode", "wait_encode_queue")
            _q_started = time.perf_counter()
            job = self._encode_queue.get()
            if job is None:
                self._set_debug_state("vae-encode", "stop")
                self._dit_queue.put(None)
                return
            self._record_queue_wait(job.profile, "q_wait_encode_s", _q_started)
            try:
                self._set_debug_state("vae-encode", "encode", job.chunk_idx)
                with self._stage_stream_context(self._encode_stream):
                    started = self._timer_start()
                    ref_chunk_latent = self._encode_reference_chunk(
                        job.source_frames,
                        profile=job.profile,
                        chunk_idx=job.chunk_idx,
                    )
                    self._timer_record(job.profile, "reference_prepare_s", started)
                    ready_event = self._record_stage_ready(self._encode_stream)
                self._set_debug_state("vae-encode", "put_dit_queue", job.chunk_idx)
                self._dit_queue.put(_EncodedChunk(
                    job=job,
                    ref_chunk_latent=ref_chunk_latent,
                    ready_event=ready_event,
                ))
                self._inc_debug_counter("encoded_chunks")
            except BaseException as exc:
                self._set_debug_state("vae-encode", "error", job.chunk_idx)
                self._set_worker_error(exc)
                self._dit_queue.put(None)
                return

    def _dit_worker(self) -> None:
        assert self._dit_queue is not None
        assert self._decode_queue is not None
        while True:
            self._set_debug_state("dit-denoise", "wait_dit_queue")
            _q_started = time.perf_counter()
            encoded = self._dit_queue.get()
            if encoded is None:
                self._set_debug_state("dit-denoise", "stop")
                self._decode_queue.put(None)
                return
            self._record_queue_wait(encoded.job.profile, "q_wait_dit_s", _q_started)
            try:
                with self.runtime.dit_lock:
                    self._set_debug_state("dit-denoise", "denoise", encoded.job.chunk_idx)
                    with self._stage_stream_context(self._dit_stream):
                        self._wait_stage_ready(
                            self._dit_stream, encoded.ready_event, encoded.ref_chunk_latent
                        )
                        current_chunk_latents = self._denoise_chunk(
                            encoded.ref_chunk_latent,
                            profile=encoded.job.profile,
                            chunk_idx=encoded.job.chunk_idx,
                            frozen_anchor_id=encoded.job.frozen_anchor_id,
                        )
                        self._evict_after_store(
                            encoded.job.chunk_idx,
                            encoded.job.profile,
                            frozen_anchor_id=encoded.job.frozen_anchor_id,
                        )
                        ready_event = self._record_stage_ready(self._dit_stream)
                self._set_debug_state("dit-denoise", "put_decode_queue", encoded.job.chunk_idx)
                self._decode_queue.put(
                    _DenoisedChunk(
                        job=encoded.job,
                        current_chunk_latents=current_chunk_latents,
                        ready_event=ready_event,
                    )
                )
                self._inc_debug_counter("denoised_chunks")
            except BaseException as exc:
                self._set_debug_state("dit-denoise", "error", encoded.job.chunk_idx)
                self._set_worker_error(exc)
                self._decode_queue.put(None)
                return

    def _decode_worker(self) -> None:
        assert self._decode_queue is not None
        assert self._postprocess_queue is not None
        if not self.stateful_vae:
            assert self._pseudo_queue is not None
        while True:
            self._set_debug_state("vae-decode", "wait_decode_queue")
            _q_started = time.perf_counter()
            denoised = self._decode_queue.get()
            if denoised is None:
                self._set_debug_state("vae-decode", "stop")
                if self._pseudo_queue is not None:
                    self._pseudo_queue.put(None)
                self._postprocess_queue.put(None)
                return
            self._record_queue_wait(denoised.job.profile, "q_wait_decode_s", _q_started)
            try:
                self._set_debug_state("vae-decode", "decode_pixels", denoised.job.chunk_idx)
                with self._stage_stream_context(self._decode_stream):
                    self._wait_stage_ready(
                        self._decode_stream,
                        denoised.ready_event,
                        denoised.current_chunk_latents,
                    )
                    decoded_pixels = self._decode_chunk_pixels(
                        denoised.current_chunk_latents,
                        profile=denoised.job.profile,
                        chunk_idx=denoised.job.chunk_idx,
                    )
                    ready_event = self._record_stage_ready(self._decode_stream)

                if self._pseudo_queue is not None:
                    self._set_debug_state(
                        "vae-decode", "put_pseudo_queue", denoised.job.chunk_idx
                    )
                    self._pseudo_queue.put(
                        _DecodedPixelsChunk(
                            job=denoised.job,
                            decoded_pixels=decoded_pixels,
                            ready_event=ready_event,
                        )
                    )
                self._set_debug_state("vae-decode", "put_postprocess_queue", denoised.job.chunk_idx)
                self._postprocess_queue.put(
                    _DecodedPixelsChunk(
                        job=denoised.job,
                        decoded_pixels=decoded_pixels,
                        ready_event=ready_event,
                    )
                )
                self._inc_debug_counter("decoded_pixel_chunks")
            except BaseException as exc:
                self._set_debug_state("vae-decode", "error", denoised.job.chunk_idx)
                self._set_worker_error(exc)
                if self._pseudo_queue is not None:
                    self._pseudo_queue.put(None)
                self._postprocess_queue.put(None)
                return

    def _pseudo_worker(self) -> None:
        assert self._pseudo_queue is not None
        while True:
            self._set_debug_state("pseudo-encode", "wait_pseudo_queue")
            _q_started = time.perf_counter()
            decoded = self._pseudo_queue.get()
            if decoded is None:
                self._set_debug_state("pseudo-encode", "stop")
                return
            self._record_queue_wait(decoded.job.profile, "q_wait_pseudo_s", _q_started)
            try:
                self._set_debug_state("pseudo-encode", "pseudo_encode", decoded.job.chunk_idx)
                with self._stage_stream_context(self._pseudo_stream):
                    self._wait_stage_ready(
                        self._pseudo_stream, decoded.ready_event, decoded.decoded_pixels
                    )
                    self._encode_next_decode_pseudo_latent(
                        decoded.decoded_pixels,
                        profile=decoded.job.profile,
                        chunk_idx=decoded.job.chunk_idx,
                    )
                self._inc_debug_counter("pseudo_encoded_chunks")
            except BaseException as exc:
                self._set_debug_state("pseudo-encode", "error", decoded.job.chunk_idx)

                self._set_worker_error(exc)
                return

    def _postprocess_worker(self) -> None:
        assert self._postprocess_queue is not None
        assert self._result_queue is not None
        while True:
            self._set_debug_state("postprocess", "wait_postprocess_queue")
            _q_started = time.perf_counter()
            decoded = self._postprocess_queue.get()
            if decoded is None:
                self._set_debug_state("postprocess", "stop")
                return
            self._record_queue_wait(decoded.job.profile, "q_wait_postprocess_s", _q_started)
            try:
                self._set_debug_state("postprocess", "pack_frames", decoded.job.chunk_idx)
                with self._stage_stream_context(self._postprocess_stream):
                    self._wait_stage_ready(
                        self._postprocess_stream, decoded.ready_event, decoded.decoded_pixels
                    )
                    packed = self._pack_decoded_pixels(
                        decoded.decoded_pixels,
                        profile=decoded.job.profile,
                        chunk_idx=decoded.job.chunk_idx,
                    )

                n_out = int(packed.shape[0])
                self._finish_chunk_profile(
                    decoded.job.profile,
                    decoded.job.total_started,
                )
                self._log_chunk_profile(
                    decoded.job.profile,
                    len(decoded.job.source_frames),
                    n_out,
                )
                self._set_debug_state("postprocess", "put_result_queue", decoded.job.chunk_idx)
                result = StreamingChunkResult(
                    pixels=packed,
                    profile=decoded.job.profile,
                    source_metas=self._align_source_metas(
                        decoded.job.source_metas,
                        n_out,
                    ),
                    elapsed=float(decoded.job.profile.get("total_server_chunk_s", 0.0)),
                )
                while True:
                    try:
                        self._result_queue.put_nowait(result)
                        break
                    except queue.Full:
                        try:
                            self._result_queue.get_nowait()
                            self._inc_debug_counter("dropped_results")
                        except queue.Empty:
                            continue
                self._inc_debug_counter("postprocessed_chunks")
            except BaseException as exc:
                self._set_debug_state("postprocess", "error", decoded.job.chunk_idx)
                self._set_worker_error(exc)
                return

    @torch.no_grad()
    def _encode_reference_chunk(
        self,
        source_frames: list[Image.Image],
        *,
        profile: dict[str, Any] | None = None,
        chunk_idx: int | None = None,
    ) -> torch.Tensor:
        chunk_idx = self.chunk_idx if chunk_idx is None else chunk_idx
        encode_device = _module_device(self.pipeline.vae)
        prepare_wall_started = time.perf_counter()
        prepare_gpu_started = self._timer_start(encode_device)
        if self.stateful_vae:
            # XVAE's encoder is causal and already exposes exactly the cache
            # needed between its 1-frame prologue and subsequent 8-frame
            # chunks.  Feeding only new frames is mathematically equivalent to
            # encoding the full sequence, and avoids recomputing the overlap
            # frame plus every causal feature-cache prologue.
            source_window = self._frames_to_tensor(
                source_frames[:1] if chunk_idx == 0 else source_frames,
                device=encode_device,
                dtype=self.target_dtype,
            )
        elif chunk_idx == 0:
            source_window = self._frames_to_tensor(
                source_frames[:1],
                device=encode_device,
                dtype=self.target_dtype,
            )
        else:
            if self.prev_source_frame is None:
                raise RuntimeError("Missing previous source frame for streaming VAE encode.")
            if len(source_frames) != self.ffactor_t:
                raise ValueError(
                    f"Expected {self.ffactor_t} frames after the first chunk, got {len(source_frames)}."
                )
            new_frames = self._frames_to_tensor(
                source_frames,
                device=encode_device,
                dtype=self.target_dtype,
            )
            source_window = torch.cat([self.prev_source_frame, new_frames], dim=2)

        # Retain the overlap frame where the encoder runs. Keeping it on CPU
        # caused every subsequent chunk to resend one full-resolution float
        # frame and perform the temporal concatenation on the host.
        if not self.stateful_vae:
            self.prev_source_frame = source_window[:, :, -1:].detach().clone()
        if profile is not None:
            profile["frames_to_tensor_s"] = float(profile.get("frames_to_tensor_s", 0.0)) + (
                time.perf_counter() - prepare_wall_started
            )
            self._timer_record(profile, "frames_to_tensor_gpu_s", prepare_gpu_started)

        _vc = _vae_compile_module()
        if self.stateful_vae:
            _vc.maybe_setup_stream_encode(self.pipeline.vae)
        else:
            _vc.maybe_setup_encode(self.pipeline.vae)
        source_window = _vc.prep_input(source_window)
        started = self._timer_start(encode_device)

        _enc_dev_type = torch.device(encode_device).type
        _enc_ctx = _autocast_ctx(_enc_dev_type, self.vae_dtype, self.vae_autocast_enabled)
        with _enc_ctx:
            if self.stateful_vae:
                posterior = self.pipeline.vae.encode_stream(source_window).latent_dist
                ref_latent = posterior.sample(generator=self._vae_encode_generator)
                if self.enable_denormalization:
                    ref_latent = self.pipeline.normalize_latents(ref_latent)
            else:
                ref_latent = self.pipeline._sample_vae_latents(
                    source_window,
                    enable_denormalization=self.enable_denormalization,
                    last_only=True,
                    generator=self._vae_encode_generator,
                )
        if profile is not None:
            self._timer_record(profile, "vae_encode_s", started)
        ref_latent = ref_latent[:, :, -self.chunk_size:].to(device=self.device, dtype=self.target_dtype)
        return ref_latent

    @torch.no_grad()
    def _wait_decode_pseudo_latent(
        self,
        chunk_idx: int,
        decode_device: torch.device,
    ) -> torch.Tensor:
        while True:
            with self._pseudo_latent_condition:
                if self._pseudo_latent_chunk_idx == chunk_idx and self._pseudo_latent is not None:
                    pseudo_latent = self._pseudo_latent
                    ready_event = self._pseudo_latent_ready_event
                    self._pseudo_latent = None
                    self._pseudo_latent_chunk_idx = None
                    self._pseudo_latent_ready_event = None
                    break
                self._pseudo_latent_condition.wait(timeout=0.05)
            self._raise_worker_error_if_needed()
        self._wait_stage_ready(
            self._decode_stream, ready_event, pseudo_latent
        )
        return pseudo_latent.to(device=decode_device, dtype=self.target_dtype)

    def _store_decode_pseudo_latent(
        self,
        chunk_idx: int,
        pseudo_latent: torch.Tensor,
        ready_event: torch.cuda.Event | None,
    ) -> None:
        with self._pseudo_latent_condition:
            self._pseudo_latent = pseudo_latent.detach()
            self._pseudo_latent_chunk_idx = chunk_idx
            self._pseudo_latent_ready_event = ready_event
            self._pseudo_latent_condition.notify_all()

    @torch.no_grad()
    def _decode_chunk_pixels(
        self,
        current_chunk_latents: torch.Tensor,
        *,
        profile: dict[str, Any] | None = None,
        chunk_idx: int | None = None,
    ) -> torch.Tensor:
        chunk_idx = self.chunk_idx if chunk_idx is None else chunk_idx
        decode_vae = self.decode_vae
        vae_device = _module_device(decode_vae)
        chunk_lat_flat = current_chunk_latents
        if self.enable_denormalization:
            chunk_lat_flat = self.pipeline.denormalize_latents(chunk_lat_flat)

        vae_device_type = vae_device.type
        # The scheduler intentionally accumulates in fp32, while the compiled
        # VAE decode graphs are warmed in the configured VAE precision. Cast
        # before entering the graph so the first live chunk does not compile a
        # second float32 specialization.
        chunk_lat_flat = chunk_lat_flat.to(device=vae_device, dtype=self.vae_dtype)

        if not self.stateful_vae and chunk_idx > 0:
            self._set_debug_state("vae-decode", "wait_pseudo_latent", chunk_idx)
            pseudo_latent = self._wait_decode_pseudo_latent(chunk_idx, vae_device)
            self._set_debug_state("vae-decode", "cat_pseudo_latent", chunk_idx)
            decode_input = torch.cat([pseudo_latent, chunk_lat_flat], dim=2)
            del pseudo_latent
        else:
            decode_input = chunk_lat_flat

        _vc = _vae_compile_module()
        if self.stateful_vae:
            _vc.maybe_setup_stream_decode(decode_vae)
        else:
            _vc.maybe_setup_decode(decode_vae)
        decode_input = _vc.prep_input(decode_input)

        vae_ctx = _autocast_ctx(vae_device_type, self.vae_dtype, self.vae_autocast_enabled)
        with vae_ctx:
            started = self._timer_start(vae_device)
            if self.stateful_vae:
                chunk_decoded = decode_vae.decode_stream(decode_input, return_dict=False)[0]
            else:
                chunk_decoded = decode_vae.decode(decode_input, return_dict=False)[0]
            if profile is not None:
                self._timer_record(profile, "vae_decode_s", started)

        if not self.stateful_vae and chunk_idx > 0:
            window_pixels = self.chunk_size * self.ffactor_t
            chunk_decoded = chunk_decoded[:, :, -window_pixels:]
        return chunk_decoded.detach()

    @torch.no_grad()
    def _encode_next_decode_pseudo_latent(
        self,
        decoded_pixels: torch.Tensor,
        *,
        profile: dict[str, Any] | None = None,
        chunk_idx: int,
    ) -> None:
        pseudo_vae = self.pseudo_encode_vae
        pseudo_device = _module_device(pseudo_vae)
        pseudo_device_type = pseudo_device.type
        prev_pixels = decoded_pixels[:, :, -1:].detach().to(device=pseudo_device, dtype=self.vae_dtype)

        _vc = _vae_compile_module()
        _vc.maybe_setup_encode(pseudo_vae)
        prev_pixels = _vc.prep_input(prev_pixels)
        vae_ctx = _autocast_ctx(pseudo_device_type, self.vae_dtype, self.vae_autocast_enabled)
        with vae_ctx:
            pseudo_enc = pseudo_vae.encode(prev_pixels)
            if hasattr(pseudo_enc, "latent_dist"):
                pseudo_latent = pseudo_enc.latent_dist.sample(
                    generator=self._vae_pseudo_generator
                )
            else:
                pseudo_latent = pseudo_enc
        self._set_debug_state("pseudo-encode", "store_pseudo_latent", chunk_idx)
        ready_event = self._record_stage_ready(self._pseudo_stream)
        self._store_decode_pseudo_latent(chunk_idx + 1, pseudo_latent, ready_event)
        del prev_pixels, pseudo_enc, pseudo_latent

    @torch.no_grad()
    def _pack_decoded_pixels(
        self,
        decoded_pixels: torch.Tensor,
        *,
        profile: dict[str, Any] | None = None,
        chunk_idx: int,
    ) -> torch.Tensor:
        post_device = self.postprocess_device
        wall_started = time.perf_counter()
        gpu_started = self._timer_start(post_device)

        chunk_decoded = decoded_pixels.to(device=post_device)
        frames_u8 = (
            (chunk_decoded / 2 + 0.5)
            .clamp(0, 1)
            .mul_(255.0)
            .round_()
            .clamp_(0, 255)
            .to(torch.uint8)[0]
            .permute(1, 2, 3, 0)
            .contiguous()
        )
        # Hand RGB pixels to the server so it can encode directly to the
        # negotiated wire codec. Previously every chunk was JPEG encoded here;
        # H.264 clients then decoded those JPEGs and encoded them a second time.
        frames_cpu = frames_u8.cpu()
        if profile is not None:
            profile["pack_frames_s"] = float(profile.get("pack_frames_s", 0.0)) + (
                time.perf_counter() - wall_started
            )
            profile["output_rgb_bytes"] = int(frames_cpu.numel())
            self._timer_record(profile, "pack_frames_gpu_s", gpu_started)
        return frames_cpu

    def _next_selected_chunk_ids(self, chunk_idx: int | None = None) -> list[int]:
        chunk_idx = self.chunk_idx if chunk_idx is None else chunk_idx
        next_total = chunk_idx + 2
        windows = self.pipeline._get_chunk_windows(
            total_latent_frames=next_total,
            chunk_size=self.chunk_size,
            window_size=self.local_window_size,
            global_sink_chunk=self.global_sink_chunk,
        )
        return list(windows[-1]["selected_chunk_ids"])

    def _resize_frame(self, frame: Image.Image) -> Image.Image:
        frame = frame.convert("RGB")
        if frame.size == (self.settings.width, self.settings.height):
            return frame
        resampling = getattr(Image, "Resampling", Image).BICUBIC
        return frame.resize((self.settings.width, self.settings.height), resampling)

    def _frames_to_tensor(
        self,
        frames: list[Image.Image],
        *,
        device: torch.device | str | None = None,
        dtype: torch.dtype = torch.float32,
    ) -> torch.Tensor:
        arrays = [np.asarray(self._resize_frame(frame), dtype=np.uint8) for frame in frames]
        pixel = torch.from_numpy(np.stack(arrays, axis=0))
        pixel = rearrange(pixel, "t h w c -> 1 c t h w")
        if device is not None:
            device_obj = torch.device(device)
            # Transfer compact uint8 pixels first, then normalize on-device.
            # This cuts host work and H2D bytes versus materializing float32
            # video tensors on CPU.
            pixel = pixel.to(device=device_obj, non_blocking=device_obj.type == "cuda")
            pixel = pixel.to(torch.float32)
        else:
            pixel = pixel.to(torch.float32)
        pixel = pixel / 127.5 - 1.0
        return pixel.to(dtype=dtype)
