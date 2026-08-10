from __future__ import annotations

import argparse
import asyncio
import base64
import io
import json
import os
import queue
import sys
import tempfile
import threading
import time
from concurrent.futures import Future, ThreadPoolExecutor
from contextlib import asynccontextmanager
from functools import partial
from fractions import Fraction
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse, Response
from PIL import Image, ImageChops, ImageStat
import uvicorn

from xvideo.config import generate_video_image_bucket
from xvideo.serving.pe import DEFAULT_MODEL as DEFAULT_PE_MODEL
from xvideo.serving.joyomni_streaming import (
    JoyOmniRuntime,
    StreamingSettings,
    _module_device,
)

DEFAULT_DIT_CKPT = str(REPO_ROOT / "deps" / "checkpoints" / "JoyAI-Video-Edit" / "dit" / "joyai_video_edit_dit_0804.pth")
DEFAULT_FACE_DETECTOR_ONNX = str(REPO_ROOT / "deps" / "checkpoints" / "face_detection_yunet_2023mar.onnx")
DEFAULT_PERSON_DETECTOR_ONNX = str(REPO_ROOT / "deps" / "checkpoints" / "yolov8n.onnx")
DEFAULT_PERSON_DETECTOR_PT = str(REPO_ROOT / "deps" / "checkpoints" / "yolov8n.pt")


_ROCPROF_SELECTED_REGIONS = os.environ.get(
    "JOYOMNI_ROCPROF_SELECTED_REGIONS", ""
).lower() in {"1", "true", "yes", "on"}
_ROCPROF_ROCTX = None


def _rocprof_selected_region(*, resume: bool) -> None:
    """Control rocprofv3 ``--selected-regions`` around a live session.

    This is opt-in because these ROCTx control calls affect every profiler in
    the process.  Keeping startup and model warmup outside the selected region
    makes a short kernel trace representative of steady-state streaming.
    """
    if not _ROCPROF_SELECTED_REGIONS:
        return

    import ctypes

    global _ROCPROF_ROCTX
    if _ROCPROF_ROCTX is None:
        _ROCPROF_ROCTX = ctypes.CDLL("librocprofiler-sdk-roctx.so.1")
        for name in ("roctxProfilerPause", "roctxProfilerResume"):
            fn = getattr(_ROCPROF_ROCTX, name)
            fn.argtypes = [ctypes.c_uint64]
            fn.restype = ctypes.c_int

    name = "roctxProfilerResume" if resume else "roctxProfilerPause"
    status = int(getattr(_ROCPROF_ROCTX, name)(0))
    if status != 0:
        raise RuntimeError(f"{name}(0) failed with status {status}")
    print(f"#####[ROCPROF] selected region {'resumed' if resume else 'paused'}", flush=True)


def _configure_rocm_tunableop() -> None:
    """Load an offline gfx950 GEMM selection table before model warmup."""
    filename = os.environ.get("JOYOMNI_TUNABLEOP_FILE", "").strip()
    if not filename:
        return

    import torch

    if torch.version.hip is None:
        return
    path = Path(filename)
    if not path.is_file():
        print(f"#####[TUNABLEOP] table not found; using library defaults: {path}", flush=True)
        return

    from torch.cuda import tunable

    tunable.set_filename(str(path), insert_device_ordinal=False)
    tunable.enable(True)
    tunable.tuning_enable(False)
    if not tunable.read_file(str(path)):
        raise RuntimeError(f"TunableOp rejected gfx950 table {path}")
    print(
        f"#####[TUNABLEOP] loaded {len(tunable.get_results())} gfx950 GEMM selections from {path}",
        flush=True,
    )


class SessionGate:
    def __init__(self) -> None:
        self._order: list[int] = []
        self._events: dict[int, asyncio.Event] = {}
        self._next = 0

    def enqueue(self) -> int:
        self._next += 1
        ticket = self._next
        self._order.append(ticket)
        self._events[ticket] = asyncio.Event()
        return ticket

    def position(self, ticket: int) -> int:
        return self._order.index(ticket) if ticket in self._order else -1

    def is_holder(self, ticket: int) -> bool:
        return bool(self._order) and self._order[0] == ticket

    async def wait(self, ticket: int) -> None:
        ev = self._events.get(ticket)
        if ev is None:
            return
        await ev.wait()
        ev.clear()

    def release(self, ticket: int) -> None:
        self._events.pop(ticket, None)
        if ticket in self._order:
            self._order.remove(ticket)
        for ev in self._events.values():
            ev.set()


REF_IMAGE_DIR = REPO_ROOT / "rv2v_reference"
REF_IMAGE_FILES = {
    "hat": "4e481f7a-2443-4935-a841-af6113cc4236.png",
    "scarf": "5e178546-3ebf-40df-bb86-01613dd96c3b.png",
    "pink_tee": "1c182f2f-32cf-4825-904e-64c69aed2e31.png",
    "orange_glasses": "486b9561-e73d-45ca-bb2d-2a47998a0a73.png",
}

def _load_ref_images() -> dict[str, str]:
    out: dict[str, str] = {}
    for name, fname in REF_IMAGE_FILES.items():
        path = REF_IMAGE_DIR / fname
        try:
            data = path.read_bytes()
        except OSError as exc:
            print(f"#####[STREAM] ref image {name!r} unavailable ({path}): {exc!r}")
            continue
        mime = "image/jpeg" if path.suffix.lower() in (".jpg", ".jpeg") else "image/png"
        out[name] = f"data:{mime};base64," + base64.b64encode(data).decode("ascii")
    return out


_REF_IMAGES_CACHE: dict[str, str] | None = None

def _ref_images_cached() -> dict[str, str]:
    global _REF_IMAGES_CACHE
    if _REF_IMAGES_CACHE is None:
        _REF_IMAGES_CACHE = _load_ref_images()
    return _REF_IMAGES_CACHE


def _warm_reference_prefill_paths(runtime: JoyOmniRuntime, args: argparse.Namespace) -> None:
    """Compile every reference-image bucket before the first browser session.

    Reference KV prefill is a separate dynamic DiT path. Its first unseen
    latent sequence length can otherwise spend tens of seconds compiling after
    a prompt switch. Startup cost is preferable to an interactive stall, so
    exercise all 49 supported reference buckets on the persistent frame-submit
    thread after the final process-wide PyTorch state has been established.
    """
    base_size = int(getattr(runtime.cfg, "ref_image_basesize", 512))
    buckets = generate_video_image_bucket(
        img_basesize=base_size,
        bs_img=1,
        bs_vid=0,
        bs_mimg=0,
        bs_mvid=0,
    )
    reference_hw = sorted({(int(item[3]), int(item[4])) for item in buckets})
    settings = StreamingSettings(
        height=args.height,
        width=args.width,
        num_inference_steps=args.num_inference_steps,
        max_temporal_ids=args.max_temporal_ids,
        freeze_kv_on_static=args.freeze_kv_on_static,
        static_diff_thresh=args.static_diff_thresh,
        exact_global_sink_kv=bool(args.exact_global_sink_kv),
        profile_timings=False,
    )
    anchor = Image.new("RGB", (args.width, args.height), (127, 127, 127))
    started = time.perf_counter()
    slow_paths = 0
    print(
        f"#####[STREAM] reference-prefill warmup: {len(reference_hw)} buckets",
        flush=True,
    )
    for index, (height, width) in enumerate(reference_hw, start=1):
        session = runtime.create_v2v_session(
            prompt=args.prompt,
            settings=settings,
            ref_image=Image.new("RGB", (width, height), (127, 127, 127)),
        )
        path_started = time.perf_counter()
        try:
            # push_frame performs reference initialization and establishes the
            # normal worker/stream lifecycle. close(drop_pending=True) then
            # cancels the synthetic output chunk while still synchronizing the
            # GPU before it clears the warmup session's caches.
            session.push_frame(
                anchor,
                frame_meta={"seq": 1, "t_capture_ms": 0.0},
                drain_results=False,
            )
        finally:
            session.close(drop_pending=True)
        elapsed = time.perf_counter() - path_started
        if elapsed >= 1.0:
            slow_paths += 1
            print(
                f"#####[STREAM] reference-prefill bucket {index}/{len(reference_hw)} "
                f"{height}x{width} warmed in {elapsed:.1f}s",
                flush=True,
            )
    print(
        "#####[STREAM] reference-prefill warmup complete: "
        f"{len(reference_hw)} buckets, {slow_paths} cold paths, "
        f"{time.perf_counter() - started:.1f}s",
        flush=True,
    )


_INDEX_HTML_PATH = Path(__file__).resolve().parents[2] / "static" / "index.html"

def _load_index_html() -> str:
    return _INDEX_HTML_PATH.read_text(encoding="utf-8")

def _decode_image(data: bytes) -> Image.Image:
    with Image.open(io.BytesIO(data)) as image:
        return image.convert("RGB")


def _h264_available() -> bool:
    try:
        import av  # noqa: F401
        av.codec.Codec("h264", "r")
        return True
    except Exception:
        return False


class _UplinkH264Decoder:
    def __init__(self) -> None:
        import av
        self._ctx = av.codec.CodecContext.create("h264", "r")
        self._np = __import__("numpy")

    def decode(self, data: bytes) -> Image.Image | None:
        import av
        packet = av.packet.Packet(data)
        frames = self._ctx.decode(packet)
        if not frames:
            return None
        arr = frames[-1].to_ndarray(format="rgb24")
        return Image.fromarray(arr, mode="RGB")

    def close(self) -> None:
        try:
            self._ctx.close()
        except Exception:
            pass


class _DownlinkH264Encoder:
    def __init__(self, width: int, height: int, fps: int = 24, bitrate: int | None = None) -> None:
        import av
        import fractions
        self._ctx = av.codec.CodecContext.create("libx264", "w")
        self._ctx.width = int(width)
        self._ctx.height = int(height)
        self._ctx.pix_fmt = "yuv420p"
        self._ctx.time_base = fractions.Fraction(1, max(1, int(fps)))
        if bitrate:
            self._ctx.bit_rate = int(bitrate)
        self._ctx.options = {
            "preset": "ultrafast",
            "tune": "zerolatency",
            "profile": "baseline",
            "g": str(max(1, int(fps) * 2)),
        }
        self._av = av
        self._pts = 0
        self._lock = threading.Lock()
        self._closed = False
        from av.video.frame import PictureType
        self._I_TYPE = PictureType.I

    def encode(self, frames_u8: list) -> tuple[list[bytes], list[bool]]:
        # PyAV/libx264 contexts are stateful and not thread-safe.  A cancelled
        # asyncio.to_thread call can continue in the worker after the coroutine
        # is gone, so close() must also share this lifetime lock.
        with self._lock:
            if self._closed:
                raise RuntimeError("H.264 encoder is closed")
            return self._encode_locked(frames_u8)

    def _encode_locked(self, frames_u8: list) -> tuple[list[bytes], list[bool]]:
        import numpy as np
        packets: list[bytes] = []
        keys: list[bool] = []
        first = True
        for a in frames_u8:
            if a is None:
                continue
            arr = np.ascontiguousarray(a)
            vf = self._av.VideoFrame.from_ndarray(arr, format="rgb24").reformat(format="yuv420p")
            vf.pts = self._pts
            self._pts += 1
            # Force an IDR keyframe on the FIRST frame of every encode() call.
            # One encode() call == one chunk (the pump encodes a whole chunk at
            # once), so this puts a keyframe at each chunk boundary (~8 frames).
            # Rationale for keyframe-per-chunk rather than the two extremes:
            #  - libx264 reused for the whole session emits a keyframe only on
            #    the very first frame; if the browser VideoDecoder misses that
            #    single early keyframe it drops every delta forever -> black.
            #  - Forcing EVERY frame to a keyframe kills inter-frame compression:
            #    at 1248x720 that is ~24 Mbps (3x normal h264, worse than JPEG),
            #    which saturates the uplink and makes latency climb unbounded.
            # Per-chunk keyframes: recover within one chunk (~0.3s) AND keep
            # delta compression (~8 Mbps at 720p).
            if first:
                vf.pict_type = self._I_TYPE
                first = False
            for pkt in self._ctx.encode(vf):
                packets.append(bytes(pkt))
                keys.append(bool(pkt.is_keyframe))
        return packets, keys

    def close(self) -> None:
        with self._lock:
            if self._closed:
                return
            self._closed = True
            try:
                for _ in self._ctx.encode(None):
                    pass
            except Exception:
                pass
            try:
                self._ctx.close()
            except Exception:
                pass

def _decode_ref_image(value: str | None) -> Image.Image | None:
    if not value:
        return None
    data = value.split(",", 1)[1] if value.startswith("data:") and "," in value else value
    return _decode_image(base64.b64decode(data))


_FACE_DETECTOR: dict[str, Any] = {}
_FACE_DETECTOR_MAX_SIDE = 640

def _get_face_detector(onnx_path: str, score_thresh: float):
    if not onnx_path:
        return None
    if onnx_path in _FACE_DETECTOR:
        return _FACE_DETECTOR[onnx_path] or None
    det = False
    try:
        if not os.path.exists(onnx_path):
            print(f"#####[FACE-GATE] YuNet weight not found: {onnx_path} -> gate DISABLED", flush=True)
        else:
            import cv2
            det = cv2.FaceDetectorYN.create(onnx_path, "", (320, 320), float(score_thresh), 0.3, 5000)
            print(f"#####[FACE-GATE] YuNet loaded: {onnx_path} (score>={score_thresh})", flush=True)
    except Exception as exc:
        print(f"#####[FACE-GATE] failed to load YuNet ({exc!r}) -> gate DISABLED", flush=True)
        det = False
    _FACE_DETECTOR[onnx_path] = det
    return det or None

def _face_detector_input(image: Image.Image):
    """Build a bounded detector-only image without changing the model input."""
    import cv2
    import numpy as np

    rgb = np.asarray(image if image.mode == "RGB" else image.convert("RGB"))
    bgr = np.ascontiguousarray(rgb[:, :, ::-1])
    height, width = bgr.shape[:2]
    longest = max(height, width)
    if longest > _FACE_DETECTOR_MAX_SIDE:
        scale = _FACE_DETECTOR_MAX_SIDE / float(longest)
        resized = (
            max(2, int(round(width * scale))),
            max(2, int(round(height * scale))),
        )
        bgr = cv2.resize(bgr, resized, interpolation=cv2.INTER_AREA)
    return bgr

def _check_face_gate(image: Image.Image, *, onnx_path: str,
                     score_thresh: float, min_below_ratio: float,
                     count_min_ratio: float = 0.45):
    det = _get_face_detector(onnx_path, score_thresh)
    if det is None:
        return (None, None, 0)
    bgr = _face_detector_input(image)
    h, w = bgr.shape[:2]
    det.setScoreThreshold(float(score_thresh))
    det.setInputSize((w, h))
    _, faces = det.detect(bgr)
    frame_min = float(min(w, h))

    best = None
    best_area = -1.0
    conf_shorts = []
    if faces is not None:
        for f in faces:
            if float(f[-1]) < float(score_thresh):
                continue
            fw, fh = float(f[2]), float(f[3])
            conf_shorts.append(min(fw, fh))
            if fw * fh > best_area:
                best_area = fw * fh
                best = (float(f[0]), float(f[1]), fw, fh)
    if best is None:
        return ("no_face", None, 0)
    fx, fy, fw, fh = best

    _best_short = min(fw, fh)
    _count_thr = max(0.05 * frame_min, float(count_min_ratio) * _best_short)
    n_faces = sum(1 for s in conf_shorts if s >= _count_thr)
    cx = (fx + fw / 2.0) / float(w)
    cy = (fy + fh / 2.0) / float(h)
    room_below = 1.0 - (fy + fh) / float(h)
    if room_below < float(min_below_ratio):
        return ("too_close", None, n_faces)
    return (None, (cx, cy, min(fw, fh) / frame_min), n_faces)

def _face_present(image: Image.Image, *, onnx_path: str, score_thresh: float, min_ratio: float = 0.0, edge_margin: float = 0.0) -> bool:
    det = _get_face_detector(onnx_path, score_thresh)
    if det is None:
        return True
    bgr = _face_detector_input(image)
    h, w = bgr.shape[:2]
    det.setInputSize((w, h))
    _, faces = det.detect(bgr)
    if faces is None:
        return False
    _min_side = float(min_ratio) * float(min(w, h))
    _mx = float(edge_margin) * float(w)
    _my = float(edge_margin) * float(h)
    for f in faces:
        if float(f[-1]) < float(score_thresh):
            continue
        fx, fy, fw, fh = float(f[0]), float(f[1]), float(f[2]), float(f[3])
        if _min_side > 0.0 and min(fw, fh) < _min_side:
            continue
        if edge_margin > 0.0 and (fx < _mx or fy < _my or fx + fw > w - _mx or fy + fh > h - _my):
            continue
        return True
    return False


_PERSON_NET: dict[str, Any] = {}
_PERSON_YOLO: dict[tuple[str, str, str], Any] = {}
_PERSON_YOLO_LOCK = threading.Lock()


def _get_person_yolo(pt_path: str, *, device: str, compile_mode: str):
    """Load one native PyTorch YOLO model for the selected accelerator."""
    if not pt_path:
        return None
    key = (pt_path, device, compile_mode)
    if key in _PERSON_YOLO:
        return _PERSON_YOLO[key] or None

    model = False
    try:
        if not os.path.exists(pt_path):
            print(f"#####[PERSON-GATE] YOLO PyTorch weight not found: {pt_path}", flush=True)
        else:
            from ultralytics import YOLO

            model = YOLO(pt_path)
            print(
                f"#####[PERSON-GATE] YOLO PyTorch loaded: {pt_path} "
                f"(device={device}, compile={compile_mode or 'off'})",
                flush=True,
            )
    except Exception as exc:
        print(f"#####[PERSON-GATE] failed to load PyTorch YOLO ({exc!r})", flush=True)
        model = False
    _PERSON_YOLO[key] = model
    return model or None

def _get_person_net(onnx_path: str):
    if not onnx_path:
        return None
    if onnx_path in _PERSON_NET:
        return _PERSON_NET[onnx_path] or None
    net = False
    try:
        if not os.path.exists(onnx_path):
            print(f"#####[PERSON-GATE] YOLO weight not found: {onnx_path} -> presence passthrough DISABLED", flush=True)
        else:
            import cv2
            net = cv2.dnn.readNetFromONNX(onnx_path)
            print(f"#####[PERSON-GATE] YOLO loaded: {onnx_path}", flush=True)
    except Exception as exc:
        print(f"#####[PERSON-GATE] failed to load YOLO ({exc!r}) -> DISABLED", flush=True)
        net = False
    _PERSON_NET[onnx_path] = net
    return net or None

def _person_present(
    image: Image.Image,
    *,
    onnx_path: str,
    pt_path: str = "",
    device: str = "cuda:0",
    compile_mode: str = "",
    conf: float,
) -> bool:
    # Prefer Ultralytics' native PyTorch model. PyTorch keeps the `cuda`
    # spelling on ROCm. Ultralytics predictor/model state is mutable; serialize
    # access in case WebSocket sessions overlap during teardown.
    if pt_path:
        with _PERSON_YOLO_LOCK:
            yolo = _get_person_yolo(pt_path, device=device, compile_mode=compile_mode)
            if yolo is not None:
                try:
                    import numpy as np

                    rgb = np.ascontiguousarray(
                        np.asarray(image if image.mode == "RGB" else image.convert("RGB"))
                    )
                    result = yolo.predict(
                        source=rgb,
                        imgsz=320,
                        device=device,
                        quantize=16 if device != "cpu" else None,
                        classes=[0],
                        conf=float(conf),
                        max_det=1,
                        rect=False,
                        verbose=False,
                        compile=compile_mode or False,
                    )[0]
                    return result.boxes is not None and len(result.boxes) > 0
                except Exception as exc:
                    key = (pt_path, device, compile_mode)
                    print(
                        f"#####[PERSON-GATE] PyTorch YOLO inference failed ({exc!r}) "
                        "-> falling back",
                        flush=True,
                    )
                    _PERSON_YOLO[key] = False

    net = _get_person_net(onnx_path)
    if net is None:
        return True
    try:
        import numpy as np
        import cv2
        rgb = np.ascontiguousarray(np.asarray(image if image.mode == "RGB" else image.convert("RGB")))
        blob = cv2.dnn.blobFromImage(rgb, 1.0 / 255.0, (320, 320), swapRB=False, crop=False)
        net.setInput(blob)
        out = net.forward()
        return float(out[0, 4, :].max()) >= float(conf)
    except Exception as exc:
        # The upstream gate expects a fixed-320 export, while common public
        # YOLOv8n ONNX files are fixed at 640.  A detector mismatch must not
        # terminate the WebSocket session: person monitoring is optional and
        # already fails open when the weight is absent or cannot be loaded.
        print(f"#####[PERSON-GATE] YOLO inference failed ({exc!r}) -> DISABLED", flush=True)
        _PERSON_NET[onnx_path] = False
        return True


def _warm_person_detector(args: argparse.Namespace) -> None:
    """Compile and warm the optional detector before accepting WebSockets."""
    if not args.person_detector_pt or not os.path.exists(args.person_detector_pt):
        return
    started = time.perf_counter()
    _person_present(
        Image.new("RGB", (320, 320)),
        onnx_path=args.person_detector_onnx,
        pt_path=args.person_detector_pt,
        device=args.device,
        compile_mode=args.person_detector_compile,
        conf=float(args.person_gate_conf),
    )
    print(
        f"#####[PERSON-GATE] ROCm detector warmup done in {time.perf_counter() - started:.2f}s",
        flush=True,
    )

def _enhance_prompt_sync(
    *,
    raw_prompt: str,
    ref_image: Image.Image | None,
    pe_frame: Image.Image | None,
    pe_model: str | None,
) -> dict[str, Any]:
    started = time.time()
    task_type = "v2v"
    enhanced_prompt = raw_prompt
    error = None
    model = pe_model or DEFAULT_PE_MODEL
    try:
        from xvideo.serving.pe import PromptEnhancer

        enhancer = PromptEnhancer(model=pe_model)
        model = enhancer.model or model
        enhanced = enhancer(
            task_type,
            raw_prompt,
            video=[pe_frame] if pe_frame is not None else None,
            images=[ref_image] if ref_image is not None else None,
        )
        if isinstance(enhanced, str) and enhanced.strip():
            enhanced_prompt = enhanced.strip()
    except Exception as exc:
        error = repr(exc)
    report = {
        "enabled": True,
        "task_type": task_type,
        "model": model,
        "raw_prompt": raw_prompt,
        "enhanced_prompt": enhanced_prompt,
        "elapsed_s": time.time() - started,
        "fallback": enhanced_prompt == raw_prompt,
        "cached": False,
    }
    if error is not None:
        report["error"] = error
    return report

class _SegmentedRecorder:
    def __init__(
        self,
        *,
        prefix: Path,
        fps: int,
        codec: str,
        bitrate: int,
        segment_seconds: int,
        queue_max: int = 64,
    ) -> None:
        self._prefix = prefix
        self._fps = max(1, int(fps))
        self._codec = codec
        self._bitrate = int(bitrate)
        self._segment_frames = max(1, int(segment_seconds) * self._fps)
        self._q: "queue.Queue[Any]" = queue.Queue(maxsize=max(1, queue_max))
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, name=f"rec-{prefix.name[:12]}", daemon=True)
        self._width: int | None = None
        self._height: int | None = None

        self.frames_written = 0
        self.frames_dropped_recording = 0
        self.segments = 0
        self.last_error: str | None = None
        self._started = False

    def start(self) -> None:
        self._thread.start()
        self._started = True

    def submit(self, item: Any) -> None:
        if not self._started or self._stop.is_set():
            return
        try:
            self._q.put_nowait(item)
        except queue.Full:
            self.frames_dropped_recording += 1

    def stop(self, timeout: float = 5.0) -> None:
        if not self._started:
            return
        if self._stop.is_set():
            self._thread.join(timeout=timeout)
            return
        self._stop.set()
        try:
            self._q.put_nowait(None)
        except queue.Full:
            pass
        self._thread.join(timeout=timeout)

    def _to_image(self, item: Any) -> "Image.Image | None":
        if isinstance(item, Image.Image):
            return item
        if isinstance(item, (bytes, bytearray)):
            try:
                img = Image.open(io.BytesIO(item))
                img.load()
                return img.convert("RGB") if img.mode != "RGB" else img
            except Exception:
                return None

        try:
            import numpy as np
            if isinstance(item, np.ndarray):
                return Image.fromarray(np.ascontiguousarray(item), mode="RGB")
        except Exception:
            return None
        return None

    def _open_segment(self):
        import av
        path = self._prefix.parent / f"{self._prefix.name}_{self.segments:04d}.mp4"
        output = av.open(str(path), mode="w")
        stream = output.add_stream(self._codec, rate=self._fps)
        stream.width = int(self._width)
        stream.height = int(self._height)
        stream.pix_fmt = "yuv420p"
        stream.bit_rate = self._bitrate
        stream.codec_context.options = {
            "preset": "ultrafast",
            "tune": "zerolatency",
            "g": str(self._fps * 2),
        }
        self.segments += 1
        return output, stream

    def _close_segment(self, output, stream) -> None:
        if output is None or stream is None:
            return
        try:
            for packet in stream.encode():
                output.mux(packet)
        except Exception:
            pass
        try:
            output.close()
        except Exception:
            pass

    def _run(self) -> None:
        try:
            import av
        except Exception as exc:
            self.last_error = f"pyav import failed: {exc!r}"
            return
        output = stream = None
        seg_idx = 0
        time_base = Fraction(1, self._fps)
        try:
            while True:
                try:
                    item = self._q.get(timeout=0.2)
                except queue.Empty:
                    if self._stop.is_set():
                        break
                    continue
                if item is None:
                    break
                image = self._to_image(item)
                if image is None:
                    continue
                if self._width is None:
                    self._width, self._height = int(image.width), int(image.height)
                if output is None:
                    output, stream = self._open_segment()
                    seg_idx = 0
                try:
                    frame = av.VideoFrame.from_image(image).reformat(format="yuv420p")
                    frame.pts = seg_idx
                    frame.time_base = time_base
                    for packet in stream.encode(frame):
                        output.mux(packet)
                    self.frames_written += 1
                    seg_idx += 1
                except Exception:
                    pass
                if seg_idx >= self._segment_frames:
                    self._close_segment(output, stream)
                    output = stream = None
        finally:
            self._close_segment(output, stream)

def _encode_jpeg(image: Image.Image, quality: int) -> bytes:
    buf = io.BytesIO()
    image.save(buf, format="JPEG", quality=quality, optimize=False)
    return buf.getvalue()

def _optional_positive_int(value: Any, *, name: str) -> int | None:
    if value is None or value == "":
        return None
    parsed = int(value)
    if parsed <= 0:
        raise ValueError(f"{name} must be a positive integer, got {parsed}.")
    return parsed

def create_app(args: argparse.Namespace) -> FastAPI:
    def get_runtime() -> JoyOmniRuntime:
        if app.state.runtime is not None:
            return app.state.runtime
        with app.state.runtime_lock:
            if app.state.runtime is None:
                runtime = JoyOmniRuntime.load(
                    args.dit_ckpt,
                    vae_ckpt=args.vae_ckpt,
                    text_encoder_ckpt=args.text_encoder_ckpt,
                    device=args.device,
                    vae_device=args.vae_device,
                    vae_encode_device=args.vae_encode_device,
                    vae_decode_device=args.vae_decode_device,
                    vae_pseudo_device=args.vae_pseudo_device,
                    postprocess_device=args.postprocess_device,
                    seed=args.seed,
                    warmup_height=args.height,
                    warmup_width=args.width,
                )
                app.state.runtime = runtime
        return app.state.runtime

    @asynccontextmanager
    async def lifespan(_app: FastAPI):
        try:
            if args.preload:
                runtime = get_runtime()
                # Ultralytics adjusts PyTorch's process-wide CPU thread count
                # while loading. Dynamo includes that global state in compiled
                # VAE guards, so warming the detector *after* the video graphs
                # invalidated them and made the first real session spend
                # 10-20 seconds recompiling. Establish the final global state
                # first, then warm the live pipeline on the persistent submit
                # thread.
                _warm_person_detector(args)
                # Live frame submission must stay on one persistent host
                # thread. Warming on the main thread and then entering ROCm
                # through arbitrary asyncio-pool threads caused a one-time
                # 30-50 second specialization in the first browser session.
                live_warmup = partial(
                    runtime.warmup_full_pipeline,
                    height=args.height,
                    width=args.width,
                    num_chunks=(
                        3 if getattr(runtime, "_dit_core_compiled", False) else 8
                    ),
                    prompt=args.prompt,
                    num_inference_steps=args.num_inference_steps,
                    max_temporal_ids=args.max_temporal_ids,
                    freeze_kv_on_static=args.freeze_kv_on_static,
                    static_diff_thresh=args.static_diff_thresh,
                    serial_chunks=getattr(runtime, "_dit_core_compiled", False),
                )
                await asyncio.get_running_loop().run_in_executor(
                    _app.state.frame_executor,
                    live_warmup,
                )
                if os.environ.get(
                    "JOYOMNI_WARMUP_REFERENCE_PATHS", "0"
                ).lower() not in {"0", "false", "no", "off"}:
                    await asyncio.get_running_loop().run_in_executor(
                        _app.state.frame_executor,
                        _warm_reference_prefill_paths,
                        runtime,
                        args,
                    )
            yield
        finally:
            await asyncio.to_thread(
                _app.state.frame_executor.shutdown,
                wait=True,
                cancel_futures=True,
            )

    app = FastAPI(lifespan=lifespan)
    app.state.args = args
    app.state.runtime = None
    app.state.runtime_lock = threading.Lock()
    app.state.frame_executor = ThreadPoolExecutor(
        max_workers=1,
        thread_name_prefix="joyomni-frame-submit",
    )
    app.state.inference_lock = threading.Lock()
    app.state.active_session = None
    app.state.ws_debug = {}

    app.state.session_gate = SessionGate()

    app.state.last_recording_dir = None

    @app.get("/")
    def index() -> HTMLResponse:
        server_defaults = {
            "width": args.width,
            "height": args.height,
            "num_inference_steps": args.num_inference_steps,
            "kv_reset_frames": args.kv_reset_frames,
            "output_quality": args.output_quality,
            "online_gate": args.online_gate,
            "person_count_reedit": args.person_count_reedit,
            "require_face": args.require_face,
            "static_diff_thresh": args.static_diff_thresh,
            "scene_cut_threshold": args.scene_cut_threshold,
            "freeze_kv_on_static": args.freeze_kv_on_static,
            "profile_timings": args.profile_timings,
            "use_pe": args.use_pe,
            "max_temporal_ids": args.max_temporal_ids,
            "exact_global_sink_kv": args.exact_global_sink_kv,

            "record_enabled": bool(args.record_dir),
        }
        html = _load_index_html().replace("__SERVER_DEFAULTS__", json.dumps(server_defaults))
        return HTMLResponse(html)

    @app.get("/ref-images")
    def ref_images() -> JSONResponse:
        return JSONResponse(_ref_images_cached())

    @app.get("/health")
    def health() -> JSONResponse:
        return JSONResponse(
            {
                "ok": True,
                "runtime_loaded": app.state.runtime is not None,
                "dit_ckpt": args.dit_ckpt,
                "device": str(app.state.runtime.device) if app.state.runtime is not None else args.device,
                "vae_device": str(_module_device(app.state.runtime.pipeline.vae)) if app.state.runtime is not None else args.vae_device,
                "vae_encode_device": (
                    str(_module_device(app.state.runtime.pipeline.vae))
                    if app.state.runtime is not None
                    else (args.vae_encode_device or args.vae_device)
                ),
                "vae_decode_device": (
                    str(_module_device(app.state.runtime.decode_vae))
                    if app.state.runtime is not None
                    else (args.vae_decode_device or args.vae_device)
                ),
                "vae_pseudo_device": (
                    str(_module_device(app.state.runtime.pseudo_encode_vae))
                    if app.state.runtime is not None
                    else (args.vae_pseudo_device or args.vae_decode_device or args.vae_device)
                ),
                "postprocess_device": (
                    str(app.state.runtime.postprocess_device)
                    if app.state.runtime is not None
                    else (args.postprocess_device or args.vae_pseudo_device or args.vae_decode_device or args.vae_device)
                ),
                "use_pe": args.use_pe,
                "pe_model": args.pe_model or DEFAULT_PE_MODEL,
                "kv_reset_frames": args.kv_reset_frames,
                "max_temporal_ids": args.max_temporal_ids,
                "exact_global_sink_kv": args.exact_global_sink_kv,
                "freeze_kv_on_static": args.freeze_kv_on_static,
                "static_diff_thresh": args.static_diff_thresh,
                "scene_cut_threshold": args.scene_cut_threshold,
            }
        )

    @app.get("/debug")
    def debug() -> JSONResponse:
        session = getattr(app.state, "active_session", None)
        session_debug = session.debug_snapshot() if session is not None else None
        return JSONResponse(
            {
                "ok": True,
                "ts": time.time(),
                "runtime_loaded": app.state.runtime is not None,
                "ws": getattr(app.state, "ws_debug", {}),
                "session": session_debug,
            }
        )

    @app.post("/load")
    def load() -> JSONResponse:
        started = time.time()
        get_runtime()
        return JSONResponse({"ok": True, "elapsed": time.time() - started})

    @app.get("/download_last")
    async def download_last() -> Response:
        if args.record_dir is None:
            return JSONResponse({"error": "Recording is not enabled (--record-dir is unset)."}, status_code=404)
        rec_dir = getattr(app.state, "last_recording_dir", None)
        if not rec_dir:
            return JSONResponse({"error": "No downloadable result yet. Send an edit first."}, status_code=404)
        base = Path(rec_dir)
        segments = sorted(base.glob("output_*.mp4"))
        if not segments:
            return JSONResponse({"error": "Recording file has not been generated yet. Try again later."}, status_code=404)
        download_name = f"joyomni_{base.name}.mp4"
        crf = int(args.download_crf)
        reencode = crf >= 0

        if not reencode and len(segments) == 1:
            return FileResponse(
                str(segments[0]), media_type="video/mp4", filename=download_name
            )
        try:
            import imageio_ffmpeg
            ffmpeg_exe = imageio_ffmpeg.get_ffmpeg_exe()
        except Exception as exc:
            return JSONResponse({"error": f"ffmpeg unavailable: {exc!r}"}, status_code=500)
        list_path = out_path = None
        try:
            fd_list, list_path = tempfile.mkstemp(suffix=".txt", prefix="rv2v_cat_")
            with os.fdopen(fd_list, "w", encoding="utf-8") as f:
                for seg in segments:
                    f.write(f"file '{seg.as_posix()}'\n")
            fd_out, out_path = tempfile.mkstemp(suffix=".mp4", prefix="rv2v_dl_")
            os.close(fd_out)
            if reencode:
                enc_args = [
                    "-c:v", "libx264", "-preset", str(args.download_preset),
                    "-crf", str(crf), "-pix_fmt", "yuv420p",
                ]
            else:
                enc_args = ["-c", "copy"]
            proc = await asyncio.create_subprocess_exec(
                ffmpeg_exe, "-y", "-f", "concat", "-safe", "0", "-i", list_path,
                *enc_args, "-movflags", "+faststart", out_path,
                stdout=asyncio.subprocess.DEVNULL, stderr=asyncio.subprocess.PIPE,
            )
            _, stderr = await proc.communicate()
            if proc.returncode != 0:
                tail = (stderr or b"").decode("utf-8", "replace")[-800:]
                return JSONResponse({"error": f"ffmpeg failed: {tail}"}, status_code=500)
            with open(out_path, "rb") as f:
                data = f.read()
            return Response(
                content=data, media_type="video/mp4",
                headers={"Content-Disposition": f'attachment; filename="{download_name}"'},
            )
        except Exception as exc:
            return JSONResponse({"error": f"download encode error: {exc!r}"}, status_code=500)
        finally:
            for p in (list_path, out_path):
                if p:
                    try:
                        os.unlink(p)
                    except OSError:
                        pass

    @app.websocket("/ws")
    async def websocket_endpoint(websocket: WebSocket) -> None:
        await websocket.accept()
        runtime = get_runtime()
        session = None
        ticket = None
        frames_in = 0
        frames_out = 0
        session_prompt = args.prompt
        session_settings: StreamingSettings | None = None
        ref_image: Image.Image | None = None
        face_gate_pending = False
        pe_defer = False
        presence_monitor = False
        face_required = bool(args.require_face) and bool(args.online_gate)
        count_monitor = bool(args.person_count_reedit) and bool(args.online_gate)

        fg_score = float(args.face_gate_score)
        fg_min_below = float(args.face_gate_min_below_ratio)
        fg_stable = int(args.face_gate_stable_frames)
        fg_absent = int(args.presence_absent_frames)

        gate_state = {"count": 0, "cx": None, "cy": None, "absent": 0, "passthrough": False,
                      "absent_hold": False, "present": 0, "person_check_i": 0, "person_last": True,
                      "subject_count": None, "cand": None, "cand_n": 0, "recount": False}
        kv_reset_frames = max(0, int(args.kv_reset_frames or 0))

        output_quality = max(1, min(100, int(args.output_quality)))
        max_temporal_ids = args.max_temporal_ids
        freeze_kv_on_static = args.freeze_kv_on_static
        static_diff_thresh = args.static_diff_thresh
        scene_cut_threshold = max(0.0, float(args.scene_cut_threshold))
        frames_since_session_reset = 0
        reset_count = 0
        next_frame_meta: dict[str, Any] | None = None
        uplink_codec = "jpeg"
        uplink_decoder: _UplinkH264Decoder | None = None
        downlink_codec = "jpeg"
        downlink_encoder: _DownlinkH264Encoder | None = None
        send_lock = asyncio.Lock()
        stop_output_pump = asyncio.Event()
        output_task: asyncio.Task[None] | None = None
        rocprof_region_active = False
        loop = asyncio.get_running_loop()
        first_frame_receive_mono: float | None = None
        last_frame_receive_mono: float | None = None
        person_monitor_pool = ThreadPoolExecutor(
            max_workers=1,
            thread_name_prefix="joyomni-person-monitor",
        )
        person_monitor_future: Future[dict[str, Any]] | None = None
        person_monitor_epoch = 0
        person_monitor_last_chunk = 0
        person_monitor_last_submit_frame = 0
        person_monitor_next_at = 0.0

        rec_input: _SegmentedRecorder | None = None
        rec_output: _SegmentedRecorder | None = None
        rec_seq = 0
        rec_base: Path | None = None
        ws_debug: dict[str, Any] = {
            "connected_at": time.time(),
            "frames_in": 0,
            "frames_out": 0,
            "input_fps": 0.0,
            "input_interval_ms": None,
            "frame_process_ms": None,
            "output_bytes": 0,
            "chunk_results_sent": 0,
            "last_receive_at": None,
            "last_send_at": None,
            "last_message_type": None,
            "kv_reset_frames": kv_reset_frames,
            "max_temporal_ids": max_temporal_ids,
            "freeze_kv_on_static": freeze_kv_on_static,
            "static_diff_thresh": static_diff_thresh,
            "scene_cut_threshold": scene_cut_threshold,
            "scene_cut_count": 0,
            "scene_cut_last_mad": None,
            "kv_reset_count": reset_count,
            "frames_since_session_reset": frames_since_session_reset,
            "send_state": "idle",
            "person_monitor_state": "idle",
            "person_monitor_submitted": 0,
            "person_monitor_completed": 0,
        }

        def _run_person_monitor(
            frame: Image.Image,
            epoch: int,
            *,
            monitor_presence: bool,
            monitor_face: bool,
            monitor_count: bool,
        ) -> dict[str, Any]:
            """Run one latest-frame monitor sample without blocking frame ACKs."""
            started = time.perf_counter()
            body_present = True
            face_present = True
            face_count: int | None = None

            if monitor_presence:
                body_present = _person_present(
                    frame,
                    onnx_path=args.person_detector_onnx,
                    pt_path=args.person_detector_pt,
                    device=args.device,
                    compile_mode=args.person_detector_compile,
                    conf=float(args.person_gate_conf),
                )
                if monitor_face and body_present:
                    face_present = _face_present(
                        frame,
                        onnx_path=args.face_detector_onnx,
                        score_thresh=fg_score,
                        min_ratio=float(args.face_present_min_ratio),
                        edge_margin=float(args.face_present_edge_margin),
                    )

            if monitor_count:
                _, _, face_count = _check_face_gate(
                    frame,
                    onnx_path=args.face_detector_onnx,
                    score_thresh=fg_score,
                    min_below_ratio=fg_min_below,
                    count_min_ratio=float(args.count_face_min_ratio),
                )

            return {
                "epoch": epoch,
                "body_present": body_present,
                "face_present": face_present,
                "face_count": face_count,
                "elapsed_ms": (time.perf_counter() - started) * 1000.0,
            }

        async def _send_json(payload: dict[str, Any]) -> None:
            async with send_lock:
                ws_debug["send_state"] = f"json:{payload.get('type')}"
                await websocket.send_json(payload)
                ws_debug["last_send_at"] = time.time()
                ws_debug["send_state"] = "idle"

        def _encode_downlink(frames_u8: list):
            if downlink_codec == "h264" and downlink_encoder is not None:
                return downlink_encoder.encode(frames_u8)
            jpegs = [_encode_jpeg(Image.fromarray(a), output_quality) for a in frames_u8 if a is not None]
            return jpegs, [False] * len(jpegs)

        async def _send_encoded_frames(
            encoded_frames: list[bytes],
            source_metas: list[dict[str, Any]],
            profile: dict[str, Any],
            server_elapsed: float,
            *,
            keys: list[bool] | None = None,
            rec_frames: list | None = None,
        ) -> int:
            nonlocal frames_out
            if not encoded_frames:
                return 0

            if gate_state.get("absent_hold"):
                return 0
            count = len(encoded_frames)
            if not source_metas:
                source_metas = [{} for _ in range(count)]

            # DIAG: expose the real key flags being sent so /debug can confirm
            # whether the downlink is emitting keyframes (black-screen debug).
            try:
                ws_debug["last_keys"] = [bool(k) for k in (keys or [])][:16]
                ws_debug["last_keys_true"] = int(sum(1 for k in (keys or []) if k))
                ws_debug["last_keys_len"] = int(len(keys or []))
                ws_debug["downlink_codec_live"] = downlink_codec
            except Exception:
                pass

            _prof = bool(profile.get("profile_timings"))
            _send_t0 = time.perf_counter() if _prof else 0.0

            if _prof:
                _seen: set[int] = set()
                _dec_ms = 0.0
                for _m in source_metas:
                    if id(_m) in _seen:
                        continue
                    _seen.add(id(_m))
                    _dec_ms += float(_m.get("jpeg_decode_ms", 0.0) or 0.0)
                if _dec_ms > 0.0:
                    profile["jpeg_decode_ms"] = _dec_ms

            async with send_lock:
                ws_debug["send_state"] = f"chunk_start:{profile.get('chunk_idx')}"
                await websocket.send_json(
                    {
                        "type": "chunk_start",
                        "count": count,
                        "elapsed": server_elapsed,
                        "frames_in": frames_in,
                        "source_seq_start": source_metas[0].get("seq"),
                        "source_seq_end": source_metas[-1].get("seq"),
                    }
                )
                ws_debug["last_send_at"] = time.time()
                ws_debug["send_state"] = "idle"

            for idx, encoded in enumerate(encoded_frames):
                source_meta = source_metas[min(idx, len(source_metas) - 1)]
                async with send_lock:
                    ws_debug["send_state"] = f"chunk_frame:{profile.get('chunk_idx')}:{idx}"
                    await websocket.send_json(
                        {
                            "type": "output_frame",
                            "index": idx,
                            "count": count,
                            "source_seq": source_meta.get("seq"),
                            "t_capture_ms": source_meta.get("t_capture_ms"),
                            "server_elapsed": server_elapsed,
                            "profile": profile,
                            "codec": downlink_codec,

                            "key": bool(keys[idx]) if (keys and idx < len(keys)) else False,
                        }
                    )
                    await websocket.send_bytes(encoded)
                    frames_out += 1
                    ws_debug["frames_out"] = frames_out
                    ws_debug["output_bytes"] = int(ws_debug.get("output_bytes", 0)) + len(encoded)
                    ws_debug["last_send_at"] = time.time()
                    ws_debug["send_state"] = "idle"

                    _rec_o = rec_output
                    if _rec_o is not None:
                        _rec_o.submit(rec_frames[idx] if (rec_frames is not None and idx < len(rec_frames)) else encoded)
                        ws_debug["rec_out_written"] = _rec_o.frames_written
                        ws_debug["rec_out_dropped"] = _rec_o.frames_dropped_recording

            next_chunk_needs = (
                session.frames_per_next_chunk - len(session.pending_frames)
                if session is not None
                else 0
            )

            if _prof:
                profile["ws_send_s"] = time.perf_counter() - _send_t0

                _recv_ms = source_metas[-1].get("t_server_recv_ms")
                if _recv_ms:
                    profile["server_residence_s"] = max(
                        0.0, time.time() - float(_recv_ms) / 1000.0
                    )

            _chunk_done_msg = {
                "type": "chunk_done",
                "count": count,
                "frames_in": frames_in,
                "frames_out": frames_out,
                "next_chunk_needs": next_chunk_needs,
            }
            if _prof:
                _chunk_done_msg["ws_send_s"] = profile.get("ws_send_s")
                _chunk_done_msg["server_residence_s"] = profile.get("server_residence_s")
                _chunk_done_msg["chunk_idx"] = profile.get("chunk_idx")
            async with send_lock:
                ws_debug["send_state"] = f"chunk_done:{profile.get('chunk_idx')}"
                await websocket.send_json(_chunk_done_msg)
                ws_debug["chunk_results_sent"] = int(ws_debug.get("chunk_results_sent", 0)) + 1
                ws_debug["last_send_at"] = time.time()
                ws_debug["send_state"] = "idle"

            if _prof:
                print(
                    f"#####[SERVER] chunk={profile.get('chunk_idx')} count={count} "
                    f"jpeg_dec={float(profile.get('jpeg_decode_ms', 0.0)):.0f}ms "
                    f"pack={float(profile.get('pack_frames_s', 0.0)) * 1000:.0f}ms "
                    f"wire_enc={float(profile.get('downlink_encode_s', 0.0)) * 1000:.0f}ms "
                    f"ws_send={float(profile.get('ws_send_s', 0.0)) * 1000:.0f}ms "
                    f"residence={float(profile.get('server_residence_s', 0.0)) * 1000:.0f}ms",
                    flush=True,
                )
            return count

        async def _send_chunk_result(
            result,
            *,
            fallback_meta: dict[str, Any] | None = None,
            fallback_elapsed: float = 0.0,
        ) -> int:
            server_elapsed = float(result.elapsed or fallback_elapsed)
            profile = result.profile
            _prof = bool(profile.get("profile_timings"))

            def _record_encode_time(elapsed_s: float, *, transcode: bool = False) -> None:
                if not _prof:
                    return
                profile["downlink_encode_s"] = float(profile.get("downlink_encode_s", 0.0)) + elapsed_s
                if downlink_codec == "h264":
                    key = "h264_transcode_s" if transcode else "h264_encode_s"
                else:
                    key = "jpeg_encode_s"
                profile[key] = float(profile.get(key, 0.0)) + elapsed_s

            rec_frames = None
            encoded_frames: list[bytes]
            keys: list[bool]

            if result.pixels is not None:
                def _enc_pixels(pixels=result.pixels):
                    import numpy as np
                    value = pixels.detach() if hasattr(pixels, "detach") else pixels
                    value = value.cpu() if hasattr(value, "cpu") else value
                    value = value.numpy() if hasattr(value, "numpy") else np.asarray(value)
                    if value.ndim != 4 or value.shape[-1] != 3:
                        raise ValueError(
                            f"expected output pixels shaped [T,H,W,3], got {value.shape}"
                        )
                    frames_u8 = [
                        np.ascontiguousarray(frame, dtype=np.uint8) for frame in value
                    ]
                    enc, frame_keys = _encode_downlink(frames_u8)
                    return frames_u8, enc, frame_keys

                encode_started = time.perf_counter()
                rec_frames, encoded_frames, keys = await asyncio.to_thread(_enc_pixels)
                _record_encode_time(time.perf_counter() - encode_started)
            elif result.frames:
                output_frames = result.frames

                def _enc_frames(imgs=output_frames):
                    import numpy as np
                    frames_u8 = [
                        np.ascontiguousarray(np.asarray(im.convert("RGB"), dtype=np.uint8))
                        for im in imgs
                    ]
                    enc, frame_keys = _encode_downlink(frames_u8)
                    return frames_u8, enc, frame_keys

                encode_started = time.perf_counter()
                rec_frames, encoded_frames, keys = await asyncio.to_thread(_enc_frames)
                _record_encode_time(time.perf_counter() - encode_started)
            elif result.jpegs:
                if downlink_codec == "h264" and downlink_encoder is not None:
                    # Compatibility for results produced by an older session:
                    # current sessions hand off RGB pixels and skip this lossy
                    # JPEG decode/re-encode round trip.
                    def _transcode(jpegs=result.jpegs):
                        import cv2
                        import numpy as np
                        rgb = []
                        for encoded in jpegs:
                            bgr = cv2.imdecode(
                                np.frombuffer(encoded, dtype=np.uint8),
                                cv2.IMREAD_COLOR,
                            )
                            if bgr is None:
                                raise ValueError("failed to decode legacy JPEG output")
                            rgb.append(np.ascontiguousarray(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)))
                        enc, frame_keys = downlink_encoder.encode(rgb)
                        return rgb, enc, frame_keys

                    encode_started = time.perf_counter()
                    rec_frames, encoded_frames, keys = await asyncio.to_thread(_transcode)
                    _record_encode_time(time.perf_counter() - encode_started, transcode=True)
                else:
                    encoded_frames = result.jpegs
                    keys = [False] * len(encoded_frames)
            else:
                return 0

            output_count = len(encoded_frames)
            if result.source_metas:
                source_metas = result.source_metas
            elif fallback_meta is not None:
                source_metas = [fallback_meta] * output_count
            else:
                source_metas = [{} for _ in range(output_count)]

            return await _send_encoded_frames(
                encoded_frames,
                source_metas,
                profile,
                server_elapsed,
                keys=keys,
                rec_frames=rec_frames,
            )

        async def _output_pump(session_ref) -> None:
            while not stop_output_pump.is_set():
                try:
                    result = await asyncio.to_thread(session_ref.wait_async_result, 0.05)
                except Exception as exc:
                    await _send_json({"type": "error", "message": repr(exc)})
                    break
                if result is None:
                    continue
                try:
                    await _send_chunk_result(result)
                except WebSocketDisconnect:
                    stop_output_pump.set()
                    break
                except Exception as exc:
                    # Do not leave a live inference session producing raw RGB
                    # chunks after its codec/socket output path has failed.
                    ws_debug["output_error"] = repr(exc)
                    stop_output_pump.set()
                    try:
                        await _send_json({"type": "error", "message": repr(exc)})
                    except Exception:
                        pass
                    try:
                        await websocket.close(code=1011)
                    except Exception:
                        pass
                    break

        def _create_session():
            if session_settings is None:
                raise RuntimeError("streaming settings are not initialized")
            return runtime.create_v2v_session(
                prompt=session_prompt,
                settings=session_settings,
                ref_image=ref_image,
            )

        def _session_health_error(session_ref) -> str | None:
            if session_ref is None:
                return None
            try:
                snapshot = session_ref.debug_snapshot()
            except Exception:
                return None

            workers = snapshot.get("workers") or {}
            errored = []
            dead = []
            for name, info in workers.items():
                state = info.get("state") or {}
                state_name = state.get("state")
                if state_name == "error":
                    errored.append(f"{name}@chunk={state.get('chunk_idx')}")
                if info.get("alive") is False:
                    dead.append(f"{name}:{state_name or 'unknown'}")
            if errored:
                return "streaming pipeline worker error: " + ", ".join(errored)

            queues = snapshot.get("queues") or {}
            queue_maxsize = snapshot.get("queue_maxsize") or {}
            encode_depth = queues.get("encode")
            encode_max = queue_maxsize.get("encode")
            encode_full = (
                isinstance(encode_depth, int) and
                isinstance(encode_max, int) and
                encode_max > 0 and
                encode_depth >= encode_max
            )
            if encode_full and dead:
                return (
                    f"streaming pipeline stuck: encode queue full "
                    f"({encode_depth}/{encode_max}) and workers not alive: " +
                    ", ".join(dead)
                )
            return None

        def _close_session_sync(session_ref, flush_pending: bool, drop_pending: bool):
            padded_source_frames = 0
            if flush_pending:
                padded_source_frames = session_ref.flush_pending_for_reset()
            completed = session_ref.close(drop_pending=drop_pending)
            return padded_source_frames, completed

        async def _close_session_safely(
            session_ref,
            reason: str,
            *,
            flush_pending: bool = False,
            drop_pending: bool = False,
        ) -> tuple[bool, int, list[Any]]:
            try:
                padded_source_frames, completed = await asyncio.wait_for(
                    loop.run_in_executor(
                        app.state.frame_executor,
                        _close_session_sync,
                        session_ref,
                        flush_pending,
                        drop_pending,
                    ),
                    timeout=max(0.1, float(args.session_close_timeout_s)),
                )
                return True, int(padded_source_frames), list(completed)
            except asyncio.TimeoutError:
                ws_debug["close_timeout"] = reason
                print(
                    f"#####[WS-GUARD] session close timed out after "
                    f"{args.session_close_timeout_s:.1f}s reason={reason}",
                    flush=True,
                )
                return False, 0, []
            except Exception as exc:
                ws_debug["close_error"] = repr(exc)
                print(f"#####[WS-GUARD] session close failed reason={reason}: {exc!r}", flush=True)
                return False, 0, []

        async def _stop_output_task() -> None:
            nonlocal output_task
            stop_output_pump.set()
            if output_task is not None:
                try:
                    await asyncio.wait_for(output_task, timeout=1.0)
                except asyncio.TimeoutError:
                    output_task.cancel()
                except Exception:
                    pass
            output_task = None

        async def _stop_person_monitor() -> None:
            nonlocal person_monitor_epoch, person_monitor_future
            person_monitor_epoch += 1
            if person_monitor_future is not None:
                person_monitor_future.cancel()
            # A running HIP call cannot be cancelled. Let the one bounded
            # sample finish before model/session teardown mutates GPU state.
            await asyncio.to_thread(
                person_monitor_pool.shutdown,
                wait=True,
                cancel_futures=True,
            )
            person_monitor_future = None

        def _start_recorders() -> None:
            nonlocal rec_input, rec_output, rec_seq, rec_base
            # An empty --record-dir explicitly disables recording. This keeps
            # the live demo from running two additional CPU encoders when the
            # priority is uninterrupted low-latency playback.
            if not args.record_dir:
                return
            _stop_recorders()
            try:
                rec_seq += 1
                base = Path(args.record_dir) / f"{int(time.time())}_{rec_seq}"
                base.mkdir(parents=True, exist_ok=True)
                common = dict(
                    fps=int(args.record_fps),
                    codec=str(args.record_codec),
                    bitrate=int(args.record_bitrate),
                    segment_seconds=int(args.record_segment_seconds),
                )
                rec_input = _SegmentedRecorder(prefix=base / "input", **common)
                rec_output = _SegmentedRecorder(prefix=base / "output", **common)
                rec_input.start()
                rec_output.start()
                rec_base = base
                ws_debug["rec_dir"] = str(base)
                print(f"#####[REC] recording -> {base}", flush=True)

                _write_prompt_sidecar()
            except Exception as exc:
                ws_debug["rec_error"] = repr(exc)
                print(f"#####[REC] start failed: {exc!r} -> recording OFF", flush=True)
                rec_input = None
                rec_output = None
                rec_base = None

        def _stop_recorders() -> None:
            nonlocal rec_input, rec_output, rec_base
            for _rec in (rec_input, rec_output):
                if _rec is not None:
                    try:
                        _rec.stop()
                    except Exception as exc:
                        ws_debug["rec_error"] = f"stop: {exc!r}"
            rec_input = None
            rec_output = None
            rec_base = None

        def _write_prompt_sidecar() -> None:
            if rec_base is None:
                return
            try:
                report = pe_report if isinstance(pe_report, dict) else None
                enhanced = report.get("enhanced_prompt") if report else None
                doc = {"raw_prompt": raw_session_prompt}

                if enhanced and enhanced != raw_session_prompt:
                    doc["enhanced_prompt"] = enhanced
                (rec_base / "prompts.json").write_text(
                    json.dumps(doc, ensure_ascii=False, indent=2), encoding="utf-8"
                )
            except Exception as exc:
                ws_debug["rec_error"] = f"prompts: {exc!r}"

        def _start_output_task(session_ref) -> None:
            nonlocal output_task
            if session_settings is not None:
                stop_output_pump.clear()
                output_task = asyncio.create_task(_output_pump(session_ref))

        async def _reset_session(reason: str) -> None:
            nonlocal session, frames_since_session_reset, reset_count
            if session is None:
                return

            await _stop_output_task()
            closed, padded_source_frames, completed = await _close_session_safely(
                session,
                reason,
                flush_pending=(reason == "scene_cut"),
            )
            if not closed:
                # wait_for cannot stop a native worker that is already inside
                # a HIP call.  Never create a second session over model state
                # which the timed-out close may still be mutating.
                raise RuntimeError(f"could not safely reset session ({reason})")

            # The output pump is deliberately stopped before close so it
            # cannot race the stateful encoder.  close() returns every result
            # produced while its workers drain; preserve their order on the
            # wire before beginning the new causal scene.
            for result in completed:
                await _send_chunk_result(result)
            if padded_source_frames:
                ws_debug["scene_cut_flushed_source_frames"] = padded_source_frames
            reset_count += 1
            session = _create_session()
            app.state.active_session = session
            frames_since_session_reset = 0
            ws_debug["kv_reset_count"] = reset_count
            ws_debug["frames_since_session_reset"] = frames_since_session_reset
            await _send_json(
                {
                    "type": "session_reset",
                    "reason": reason,
                    "reset_count": reset_count,
                    "frames_in": frames_in,
                    "kv_reset_frames": kv_reset_frames,
                    "max_temporal_ids": max_temporal_ids,
                    "freeze_kv_on_static": freeze_kv_on_static,
                    "static_diff_thresh": static_diff_thresh,
                    "scene_cut_threshold": scene_cut_threshold,
                    "frames_per_next_chunk": session.frames_per_next_chunk,
                }
            )
            _start_output_task(session)

        try:
            gate = app.state.session_gate
            ticket = gate.enqueue()
            while not gate.is_holder(ticket):
                pos = gate.position(ticket)
                await _send_json({"type": "queue_position", "position": pos, "ahead": pos})
                try:
                    await asyncio.wait_for(gate.wait(ticket), timeout=2.0)
                except asyncio.TimeoutError:
                    pass
            await _send_json({"type": "session_granted"})

            app.state.ws_debug = ws_debug

            HOLDER_IDLE_TIMEOUT_S = 10.0
            # Do not evict a connected client merely because it has filled the
            # bounded input window and is waiting for an in-flight GPU chunk.
            # This also lets the one-time live-session graph specialization
            # finish. A genuinely wedged backend is still released eventually.
            HOLDER_INFLIGHT_TIMEOUT_S = 120.0
            last_activity = time.monotonic()
            last_frames_out = frames_out
            while True:
                if frames_out != last_frames_out:
                    last_frames_out = frames_out
                    last_activity = time.monotonic()
                holder_timeout_s = (
                    HOLDER_INFLIGHT_TIMEOUT_S
                    if session is not None and frames_in > frames_out
                    else HOLDER_IDLE_TIMEOUT_S
                )
                if time.monotonic() - last_activity >= holder_timeout_s:
                    try:
                        await _send_json(
                            {
                                "type": "session_timeout",
                                "message": f"released after {holder_timeout_s:.0f}s idle; reconnect to re-queue",
                            }
                        )
                    except Exception:
                        pass
                    break
                try:
                    message = await asyncio.wait_for(websocket.receive(), timeout=2.0)
                except asyncio.TimeoutError:
                    continue
                if "text" in message and message["text"] is not None:
                    payload = json.loads(message["text"])
                    msg_type = payload.get("type")
                    if msg_type == "start":
                        restart_started = time.perf_counter()
                        restart_profile: dict[str, Any] = {
                            "had_live_session": bool(session is not None),
                        }
                        print(f"#####[RESTART] 'start' received (session {'live' if session is not None else 'none'})", flush=True)
                        last_activity = time.monotonic()
                        if session is not None:
                            pre_close_snapshot = session.debug_snapshot()
                            restart_profile["pending_frames_before"] = int(
                                pre_close_snapshot.get("pending_frames") or 0
                            )
                            restart_profile["queues_before"] = dict(
                                pre_close_snapshot.get("queues") or {}
                            )
                            restart_profile["worker_states_before"] = dict(
                                pre_close_snapshot.get("worker_states") or {}
                            )
                            print("#####[RESTART] tearing down prior session: stop_output_task", flush=True)
                            phase_started = time.perf_counter()
                            await _stop_output_task()
                            restart_profile["stop_output_s"] = time.perf_counter() - phase_started
                            print("#####[RESTART] close_session", flush=True)
                            phase_started = time.perf_counter()
                            closed, padded_source_frames, completed = await _close_session_safely(
                                session,
                                "restart",
                                flush_pending=False,
                                drop_pending=True,
                            )
                            restart_profile["close_session_s"] = time.perf_counter() - phase_started
                            restart_profile["close_profile"] = dict(
                                getattr(session, "last_close_profile", None) or {}
                            )
                            restart_profile["padded_source_frames"] = padded_source_frames
                            restart_profile["completed_chunks"] = len(completed)
                            if not closed:
                                await _send_json(
                                    {
                                        "type": "error",
                                        "message": "prior session did not close safely",
                                    }
                                )
                                break
                            # Preserve every result that the intentional full
                            # teardown had to finish.  Previously close() paid
                            # this cost and then discarded the completed old-
                            # prompt chunks, starving the browser's continuity
                            # buffer for the entire restart interval.
                            phase_started = time.perf_counter()
                            drained_frames = 0
                            for result in completed:
                                drained_frames += await _send_chunk_result(result)
                            restart_profile["send_completed_s"] = time.perf_counter() - phase_started
                            restart_profile["completed_frames_sent"] = drained_frames
                            print("#####[RESTART] prior session closed OK", flush=True)
                            if getattr(app.state, "active_session", None) is session:
                                app.state.active_session = None
                            session = None

                        raw_session_prompt = str(payload.get("prompt", args.prompt))
                        session_prompt = raw_session_prompt
                        try:
                            ref_image = _decode_ref_image(payload.get("ref_image"))
                        except Exception as exc:
                            await _send_json({"type": "error", "message": f"failed to decode ref image: {exc!r}"})
                            continue
                        kv_reset_frames = max(0, int(payload.get("kv_reset_frames", args.kv_reset_frames)))
                        output_quality = max(1, min(100, int(payload.get("output_quality", args.output_quality))))
                        try:
                            max_temporal_ids = _optional_positive_int(
                                payload.get("max_temporal_ids", args.max_temporal_ids),
                                name="max_temporal_ids",
                            )
                        except Exception as exc:
                            await _send_json({"type": "error", "message": f"invalid max temporal ids: {exc!r}"})
                            continue

                        freeze_kv_on_static = bool(
                            payload.get("freeze_kv_on_static", args.freeze_kv_on_static)
                        )
                        static_diff_thresh = float(
                            payload.get("static_diff_thresh", args.static_diff_thresh)
                        )
                        scene_cut_threshold = max(
                            0.0,
                            float(payload.get("scene_cut_threshold", args.scene_cut_threshold)),
                        )
                        use_pe = bool(payload.get("use_pe", args.use_pe))

                        face_gate_pending = bool(payload.get("gate_enabled", True))

                        presence_monitor = bool(args.online_gate) and bool(
                            payload.get("no_person_blank", True)
                        )

                        face_required = bool(args.online_gate) and bool(payload.get("require_face", True))

                        count_monitor = (
                            bool(args.person_count_reedit) and
                            bool(args.online_gate) and
                            bool(payload.get("person_count_reedit", True))
                        )

                        fg_score = float(payload.get("fg_score", args.face_gate_score))
                        fg_min_below = float(payload.get("fg_min_below_ratio", args.face_gate_min_below_ratio))
                        fg_stable = max(1, int(payload.get("fg_stable_frames", args.face_gate_stable_frames)))
                        fg_absent = max(1, int(args.presence_absent_frames))
                        fg_return = max(1, int(args.presence_return_frames))
                        gate_state["count"] = 0
                        gate_state["cx"] = None
                        gate_state["cy"] = None
                        gate_state["absent"] = 0
                        gate_state["passthrough"] = False
                        gate_state["absent_hold"] = False
                        gate_state["present"] = 0
                        gate_state["person_check_i"] = 0
                        gate_state["person_last"] = True
                        gate_state["face_last"] = True
                        gate_state["hold_reason"] = "no_person"
                        gate_state["body_miss"] = 0
                        gate_state["subject_count"] = None
                        gate_state["cand"] = None
                        gate_state["cand_n"] = 0
                        gate_state["recount"] = False
                        gate_state["settle_gray"] = None
                        gate_state["pe_anchor"] = None
                        gate_state["scene_gray"] = None
                        gate_state["settle_ax"] = None
                        gate_state["settle_ay"] = None
                        person_monitor_epoch += 1
                        if person_monitor_future is not None and person_monitor_future.cancel():
                            person_monitor_future = None
                        person_monitor_last_chunk = int(ws_debug.get("chunk_results_sent", 0))
                        person_monitor_last_submit_frame = frames_in
                        person_monitor_next_at = 0.0
                        ws_debug["person_monitor_state"] = "idle"
                        if face_gate_pending:
                            print(f"#####[FACE-GATE] armed (mode=upper[all], score={fg_score}, min_below={fg_min_below}, stable={fg_stable}, absent={fg_absent})", flush=True)
                        pe_report = None
                        pe_defer = False
                        if use_pe:
                            cached_enhanced_prompt = str(payload.get("enhanced_prompt") or "").strip()
                            if cached_enhanced_prompt:
                                session_prompt = cached_enhanced_prompt
                                pe_report = {
                                    "enabled": True,
                                    "task_type": "rv2v" if ref_image is not None else "v2v",
                                    "model": args.pe_model or DEFAULT_PE_MODEL,
                                    "raw_prompt": raw_session_prompt,
                                    "enhanced_prompt": session_prompt,
                                    "elapsed_s": 0.0,
                                    "fallback": session_prompt == raw_session_prompt,
                                    "cached": True,
                                }
                                ws_debug["pe_report"] = pe_report
                                await _send_json({"type": "prompt_enhanced", **pe_report})
                            else:
                                pe_defer = True
                        session_settings = StreamingSettings(
                            height=int(payload.get("height", args.height)),
                            width=int(payload.get("width", args.width)),
                            num_inference_steps=int(payload.get("num_inference_steps", args.num_inference_steps)),
                            seed=int(payload.get("seed", args.seed)),
                            max_temporal_ids=max_temporal_ids,
                            freeze_kv_on_static=freeze_kv_on_static,
                            static_diff_thresh=static_diff_thresh,
                            cache_last_denoise_kv=(
                                bool(payload["cache_last_denoise_kv"])
                                if "cache_last_denoise_kv" in payload
                                else None
                            ),
                            clean_kv_prefix_layers=(
                                int(payload["clean_kv_prefix_layers"])
                                if "clean_kv_prefix_layers" in payload
                                else None
                            ),
                            exact_global_sink_kv=(
                                bool(payload["exact_global_sink_kv"])
                                if "exact_global_sink_kv" in payload
                                else bool(args.exact_global_sink_kv)
                            ),
                            profile_timings=bool(payload.get("profile_timings", args.profile_timings)),
                        )

                        print("#####[RESTART] creating new session", flush=True)
                        phase_started = time.perf_counter()
                        session = _create_session()
                        restart_profile["create_session_s"] = time.perf_counter() - phase_started
                        print("#####[RESTART] new session created; sending 'started'", flush=True)
                        app.state.active_session = session
                        frames_since_session_reset = 0
                        reset_count = 0
                        ws_debug["kv_reset_frames"] = kv_reset_frames
                        ws_debug["use_pe"] = use_pe
                        ws_debug["pe_model"] = args.pe_model or DEFAULT_PE_MODEL
                        ws_debug["max_temporal_ids"] = max_temporal_ids
                        ws_debug["freeze_kv_on_static"] = freeze_kv_on_static
                        ws_debug["static_diff_thresh"] = static_diff_thresh
                        ws_debug["scene_cut_threshold"] = scene_cut_threshold
                        ws_debug["scene_cut_count"] = 0
                        ws_debug["scene_cut_last_mad"] = None
                        ws_debug["kv_reset_count"] = reset_count
                        ws_debug["frames_since_session_reset"] = frames_since_session_reset
                        ws_debug["has_ref_image"] = ref_image is not None
                        ws_debug["last_message_type"] = "start"
                        phase_started = time.perf_counter()
                        if uplink_decoder is not None:
                            uplink_decoder.close()
                            uplink_decoder = None
                        _req_uplink = str(payload.get("uplink_codec", "jpeg")).lower()
                        if args.uplink_codec == "jpeg":
                            uplink_codec = "jpeg"
                        elif _req_uplink == "h264" and _h264_available():
                            try:
                                uplink_decoder = _UplinkH264Decoder()
                                uplink_codec = "h264"
                            except Exception as _uexc:
                                print(f"#####[UPLINK] h264 decoder init failed, fallback jpeg: {_uexc!r}", flush=True)
                                uplink_codec = "jpeg"
                        else:
                            uplink_codec = "jpeg"
                        ws_debug["uplink_codec"] = uplink_codec
                        if downlink_encoder is not None:
                            downlink_encoder.close()
                            downlink_encoder = None
                        _req_downlink = str(payload.get("downlink_codec", "jpeg")).lower()
                        if args.downlink_codec == "jpeg":
                            downlink_codec = "jpeg"
                        elif _req_downlink == "h264" and _h264_available():
                            try:
                                downlink_encoder = _DownlinkH264Encoder(
                                    session_settings.width, session_settings.height,
                                    fps=max(1, int(args.downlink_fps)),
                                )
                                downlink_codec = "h264"
                            except Exception as _dexc:
                                print(f"#####[DOWNLINK] h264 encoder init failed, fallback jpeg: {_dexc!r}", flush=True)
                                downlink_codec = "jpeg"
                        else:
                            downlink_codec = "jpeg"
                        ws_debug["downlink_codec"] = downlink_codec
                        restart_profile["codec_setup_s"] = time.perf_counter() - phase_started
                        restart_profile["total_to_started_s"] = time.perf_counter() - restart_started
                        ws_debug["last_restart_profile"] = dict(restart_profile)
                        print(
                            "#####[RESTART-TIMING] "
                            + " ".join(
                                f"{key}={value:.3f}s"
                                for key, value in restart_profile.items()
                                if isinstance(value, float)
                            )
                            + f" completed_chunks={restart_profile.get('completed_chunks', 0)}"
                            + f" pending_frames={restart_profile.get('pending_frames_before', 0)}"
                            + f" queue_depth={sum(v for v in (restart_profile.get('queues_before') or {}).values() if isinstance(v, int))}",
                            flush=True,
                        )
                        await _send_json(
                            {
                                "type": "started",
                                "frames_per_next_chunk": session.frames_per_next_chunk,
                                "height": session_settings.height,
                                "width": session_settings.width,
                                "ref_image": ref_image is not None,
                                "kv_reset_frames": kv_reset_frames,
                                "use_pe": use_pe,
                                "pe_model": args.pe_model or DEFAULT_PE_MODEL,
                                "max_temporal_ids": max_temporal_ids,
                                "exact_global_sink_kv": session_settings.exact_global_sink_kv,
                                "freeze_kv_on_static": freeze_kv_on_static,
                                "static_diff_thresh": static_diff_thresh,
                                "scene_cut_threshold": scene_cut_threshold,
                                "uplink_codec": uplink_codec,
                                "downlink_codec": downlink_codec,
                                "restart_profile": restart_profile,
                            }
                        )
                        _start_recorders()
                        _start_output_task(session)
                        if not rocprof_region_active:
                            _rocprof_selected_region(resume=True)
                            rocprof_region_active = True
                    elif msg_type == "stop":
                        ws_debug["last_message_type"] = "stop"

                        await _stop_output_task()
                        _stopped_rec = rec_base
                        await asyncio.to_thread(_stop_recorders)
                        if rocprof_region_active:
                            _rocprof_selected_region(resume=False)
                            rocprof_region_active = False
                        if _stopped_rec is not None:
                            app.state.last_recording_dir = str(_stopped_rec)
                        break
                    elif msg_type == "finalize_recording":
                        ws_debug["last_message_type"] = "finalize_recording"
                        if args.record_dir is None:
                            await _send_json({"type": "recording_finalized", "ok": False,
                                              "message": "Recording is not enabled (--record-dir is unset)."})
                            continue

                        await _stop_output_task()
                        finalized = rec_base
                        await asyncio.to_thread(_stop_recorders)
                        if finalized is not None:
                            app.state.last_recording_dir = str(finalized)
                        await _send_json({
                            "type": "recording_finalized",
                            "ok": finalized is not None,
                            "message": None if finalized is not None
                            else "No downloadable result yet. Send an edit first.",
                        })
                        continue
                    elif msg_type == "frame_meta":
                        ws_debug["last_message_type"] = "frame_meta"
                        next_frame_meta = {
                            "seq": int(payload.get("seq", frames_in + 1)),
                            "t_capture_ms": float(payload.get("t_capture_ms", time.time() * 1000.0)),
                        }
                    elif msg_type == "ping":
                        await _send_json({"type": "pong", "t": payload.get("t")})
                        continue
                    elif msg_type == "set_output_quality":
                        try:
                            output_quality = max(1, min(100, int(payload.get("value", output_quality))))
                        except (TypeError, ValueError):
                            pass
                        continue
                    else:
                        await _send_json({"type": "error", "message": f"unknown message type: {msg_type}"})
                    continue

                if "bytes" not in message or message["bytes"] is None:
                    continue
                if session is None:
                    await _send_json({"type": "error", "message": "send start JSON before frames"})
                    continue

                frame_bytes = message["bytes"]
                frames_in += 1
                last_activity = time.monotonic()
                receive_mono = time.monotonic()
                if first_frame_receive_mono is None:
                    first_frame_receive_mono = receive_mono
                if last_frame_receive_mono is not None:
                    interval_ms = (receive_mono - last_frame_receive_mono) * 1000.0
                    previous_ms = ws_debug.get("input_interval_ms")
                    ws_debug["input_interval_ms"] = (
                        interval_ms if previous_ms is None
                        else float(previous_ms) * 0.9 + interval_ms * 0.1
                    )
                last_frame_receive_mono = receive_mono
                receive_elapsed = receive_mono - first_frame_receive_mono
                if receive_elapsed > 0.0:
                    ws_debug["input_fps"] = (frames_in - 1) / receive_elapsed
                ws_debug["frames_in"] = frames_in
                ws_debug["last_receive_at"] = time.time()
                ws_debug["last_message_type"] = "frame_bytes"

                # Lightweight per-frame receive ack. The client's uplink
                # backpressure is pending = sentFrames - backendAckedFrames, and
                # backendAckedFrames only advances from messages carrying
                # frames_in. Those were previously emitted only when a chunk was
                # produced (every ~8 frames, after inference). At 720p the
                # produce/ack cadence lags per-frame uplink, so pending climbs to
                # MAX_BACKEND_PENDING_FRAMES (32) and the client throttles the
                # uplink down to ~1fps. Acking every received frame lets pending
                # track real "sent but not yet received" instead of the much
                # slower chunk-production rate.
                await _send_json({"type": "frame_ack", "frames_in": frames_in})

                frame_meta = next_frame_meta or {
                    "seq": frames_in,
                    "t_capture_ms": time.time() * 1000.0,
                }
                frame_meta["t_server_recv_ms"] = time.time() * 1000.0
                next_frame_meta = None

                if kv_reset_frames > 0 and frames_since_session_reset >= kv_reset_frames:
                    face_gate_pending = True
                    gate_state["count"] = 0
                    gate_state["cx"] = None
                    gate_state["cy"] = None
                    gate_state["settle_gray"] = None
                    gate_state["pe_anchor"] = None
                    gate_state["settle_ax"] = None
                    gate_state["settle_ay"] = None
                    gate_state["subject_count"] = None
                    gate_state["cand"] = None
                    gate_state["cand_n"] = 0

                    if pe_report and pe_report.get("enhanced_prompt"):
                        session_prompt = pe_report["enhanced_prompt"]
                    await _reset_session("kv_reset_frames")
                    continue

                _prof_on = session_settings is not None and session_settings.profile_timings

                def _decode_uplink(data: bytes):
                    if uplink_codec == "h264" and uplink_decoder is not None:
                        return uplink_decoder.decode(data)
                    return _decode_image(data)

                def _run_frame():
                    nonlocal person_monitor_future
                    nonlocal person_monitor_last_chunk
                    nonlocal person_monitor_last_submit_frame
                    nonlocal person_monitor_next_at

                    if _prof_on:
                        _dec_t0 = time.perf_counter()
                        frame = _decode_uplink(frame_bytes)
                        frame_meta["jpeg_decode_ms"] = (time.perf_counter() - _dec_t0) * 1000.0
                    else:
                        frame = _decode_uplink(frame_bytes)
                    if frame is None:
                        return None

                    # A causal one-latent/8-frame chunk will deliberately
                    # blend history across an abrupt edit or camera-source
                    # switch. Detect only extreme full-frame discontinuities
                    # here; ordinary body/camera motion in the qualification
                    # clip stays far below this MAD threshold. The event-loop
                    # path recreates the bounded streaming state and replays
                    # this frame as the new one-frame sink.
                    if scene_cut_threshold > 0.0:
                        scene_gray = frame.convert("L").resize(
                            (64, 36), Image.Resampling.BILINEAR
                        )
                        previous_scene_gray = gate_state.get("scene_gray")
                        gate_state["scene_gray"] = scene_gray
                        if previous_scene_gray is not None:
                            scene_mad = float(
                                ImageStat.Stat(
                                    ImageChops.difference(previous_scene_gray, scene_gray)
                                ).mean[0]
                            )
                            ws_debug["scene_cut_last_mad"] = scene_mad
                            if (
                                scene_mad >= scene_cut_threshold
                                and not face_gate_pending
                                and bool(getattr(session, "initialized", False))
                            ):
                                return ("__scene_cut__", frame, frame_meta, scene_mad)

                    if face_gate_pending:
                        if True:
                            _reason, _center, _nf = _check_face_gate(
                                frame,
                                onnx_path=args.face_detector_onnx,
                                score_thresh=fg_score,
                                min_below_ratio=fg_min_below,
                            )
                            if _reason is not None:
                                gate_state["count"] = 0
                                gate_state["settle_ax"] = None
                                gate_state["settle_ay"] = None
                                return _reason
                            if _center is not None:
                                _cx, _cy, _csz = _center

                                if _nf < 2 and abs(_cx - 0.5) > float(args.face_gate_center_margin):
                                    gate_state["count"] = 0
                                    gate_state["settle_ax"] = None
                                    gate_state["settle_ay"] = None
                                    gate_state["settle_asz"] = None
                                    gate_state["cx"] = _cx
                                    gate_state["cy"] = _cy
                                    gate_state["csz"] = _csz
                                    return "off_center"
                                _pcx, _pcy, _pcsz = gate_state["cx"], gate_state["cy"], gate_state.get("csz")
                                _eps = float(args.face_gate_move_eps)
                                _cap = float(args.face_gate_settle_drift)

                                _szeps = _eps * 0.5

                                _still = (_pcx is not None and abs(_cx - _pcx) <= _eps and abs(_cy - _pcy) <= _eps and
                                          _pcsz is not None and abs(_csz - _pcsz) <= _szeps)
                                _ax, _ay, _asz = gate_state.get("settle_ax"), gate_state.get("settle_ay"), gate_state.get("settle_asz")
                                if (_still and _ax is not None and abs(_cx - _ax) <= _cap and abs(_cy - _ay) <= _cap and
                                        _asz is not None and abs(_csz - _asz) <= _cap):
                                    gate_state["count"] += 1
                                else:
                                    gate_state["count"] = 1
                                    gate_state["settle_ax"] = _cx
                                    gate_state["settle_ay"] = _cy
                                    gate_state["settle_asz"] = _csz
                                gate_state["cx"] = _cx
                                gate_state["cy"] = _cy
                                gate_state["csz"] = _csz
                                if gate_state["count"] < fg_stable:
                                    return "settling"

                        if pe_defer:
                            gate_state["pe_anchor"] = frame
                            return "__gate_pe__"

                    if (presence_monitor or count_monitor) and not face_gate_pending:
                        stride = max(1, int(args.person_check_stride))
                        fresh_monitor_result: dict[str, Any] | None = None

                        # Never wait for the detector on the receive/ACK path.
                        # A same-GPU detector can legitimately wait behind a
                        # compiled video graph; blocking here used to stop new
                        # frames from reaching the next 8-frame chunk.
                        if person_monitor_future is not None and person_monitor_future.done():
                            try:
                                candidate = person_monitor_future.result()
                                if int(candidate.get("epoch", -1)) == person_monitor_epoch:
                                    fresh_monitor_result = candidate
                                    ws_debug["person_monitor_completed"] = int(
                                        ws_debug.get("person_monitor_completed", 0)
                                    ) + 1
                                    ws_debug["person_monitor_ms"] = float(
                                        candidate.get("elapsed_ms", 0.0)
                                    )
                                    ws_debug["person_monitor_state"] = "idle"
                            except Exception as exc:
                                # Continuous monitoring is a safety feature, so
                                # a detector failure remains fail-open instead
                                # of terminating an otherwise healthy stream.
                                ws_debug["person_monitor_error"] = repr(exc)
                                ws_debug["person_monitor_state"] = "error"
                                print(
                                    f"#####[PERSON-GATE] async monitor failed ({exc!r}) -> fail-open",
                                    flush=True,
                                )
                            finally:
                                person_monitor_future = None

                        if fresh_monitor_result is not None:
                            gate_state["person_last"] = bool(
                                fresh_monitor_result.get("body_present", True)
                            )
                            gate_state["face_last"] = bool(
                                fresh_monitor_result.get("face_present", True)
                            )

                            if count_monitor and fresh_monitor_result.get("face_count") is not None:
                                _n = int(fresh_monitor_result["face_count"])
                                sample_frames = stride
                                if gate_state["subject_count"] is None:
                                    gate_state["subject_count"] = _n
                                    gate_state["cand"] = None
                                    gate_state["cand_n"] = 0
                                elif _n > gate_state["subject_count"]:
                                    if _n == gate_state["cand"]:
                                        gate_state["cand_n"] += sample_frames
                                    else:
                                        gate_state["cand"] = _n
                                        gate_state["cand_n"] = sample_frames
                                    if gate_state["cand_n"] >= int(args.person_count_change_frames):
                                        gate_state["recount"] = True
                                        gate_state["subject_count"] = _n
                                        gate_state["cand"] = None
                                        gate_state["cand_n"] = 0
                                elif _n < gate_state["subject_count"]:
                                    if _n == gate_state["cand"]:
                                        gate_state["cand_n"] += sample_frames
                                    else:
                                        gate_state["cand"] = _n
                                        gate_state["cand_n"] = sample_frames
                                    if gate_state["cand_n"] >= int(args.person_count_change_frames):
                                        gate_state["subject_count"] = _n
                                        gate_state["cand"] = None
                                        gate_state["cand_n"] = 0
                                else:
                                    gate_state["cand"] = None
                                    gate_state["cand_n"] = 0

                        now_mono = time.monotonic()
                        completed_chunks = int(ws_debug.get("chunk_results_sent", 0))
                        chunk_ready = (
                            completed_chunks > person_monitor_last_chunk
                            and frames_in - person_monitor_last_submit_frame >= stride
                        )
                        hold_ready = (
                            bool(gate_state.get("absent_hold"))
                            and now_mono >= person_monitor_next_at
                        )
                        if person_monitor_future is None and (chunk_ready or hold_ready):
                            monitor_frame = frame.copy()
                            person_monitor_future = person_monitor_pool.submit(
                                _run_person_monitor,
                                monitor_frame,
                                person_monitor_epoch,
                                monitor_presence=presence_monitor,
                                monitor_face=face_required,
                                monitor_count=count_monitor,
                            )
                            person_monitor_last_chunk = completed_chunks
                            person_monitor_last_submit_frame = frames_in
                            person_monitor_next_at = now_mono + stride / 24.0
                            ws_debug["person_monitor_submitted"] = int(
                                ws_debug.get("person_monitor_submitted", 0)
                            ) + 1
                            ws_debug["person_monitor_state"] = "running"

                        if presence_monitor:
                            _body_here = bool(gate_state["person_last"])
                            _face_here = bool(gate_state.get("face_last", True))
                            _present = _body_here and _face_here
                            _reason_now = "no_person" if not _body_here else ("no_face" if not _face_here else "")

                            body_flip = max(1, int(args.person_body_flip_frames))
                            if _body_here:
                                gate_state["body_miss"] = 0
                            else:
                                gate_state["body_miss"] = gate_state.get("body_miss", 0) + 1
                            if _present:
                                gate_state["absent"] = 0
                                if gate_state.get("absent_hold"):
                                    gate_state["present"] = gate_state.get("present", 0) + 1
                                    if gate_state["present"] >= fg_return:
                                        gate_state["absent_hold"] = False
                                        gate_state["present"] = 0
                                        print("#####[PERSON-GATE] subject returned (stable) -> re-run startup gate (reset)", flush=True)
                                        return ("__person_returned__",)

                            else:
                                gate_state["present"] = 0
                                gate_state["absent"] += 1

                                if _reason_now == "no_person" and gate_state.get("body_miss", 0) < body_flip:
                                    _reason_now = "no_face" if face_required else ""
                                gate_state["hold_reason"] = _reason_now or gate_state.get("hold_reason") or "no_person"
                                if not gate_state.get("absent_hold") and gate_state["absent"] >= fg_absent:
                                    gate_state["absent_hold"] = True

                                    try:
                                        if session is not None:
                                            session.pending_frames.clear()
                                            session.pending_metas.clear()
                                    except Exception:
                                        pass
                                    print(f"#####[PERSON-GATE] {gate_state['hold_reason']} for {gate_state['absent']} frames -> black-hold", flush=True)
                            if gate_state.get("absent_hold"):
                                return ("__no_person__", gate_state.get("hold_reason", "no_person"))

                    if pe_defer and not face_gate_pending:
                        gate_state["pe_anchor"] = frame
                        return "__gate_pe__"

                    health_error = _session_health_error(session)
                    if health_error:
                        raise RuntimeError(health_error)

                    acquired = app.state.inference_lock.acquire(
                        timeout=max(0.1, float(args.inference_lock_timeout_s))
                    )
                    if not acquired:
                        raise TimeoutError(
                            f"inference lock timeout after {args.inference_lock_timeout_s:.1f}s"
                        )
                    try:
                        health_error = _session_health_error(session)
                        if health_error:
                            raise RuntimeError(health_error)

                        _rec_i = rec_input
                        if _rec_i is not None:
                            _rec_i.submit(frame)
                            ws_debug["rec_in_written"] = _rec_i.frames_written
                            ws_debug["rec_in_dropped"] = _rec_i.frames_dropped_recording
                        # The output pump is the sole consumer of the async
                        # result queue.  Draining here as well lets adjacent
                        # chunks race through the same stateful H.264 encoder,
                        # which can reorder PTS and interleave wire messages.
                        return session.push_frame(
                            frame,
                            frame_meta=frame_meta,
                            drain_results=False,
                        )
                    finally:
                        app.state.inference_lock.release()

                started = time.time()
                try:
                    chunk_results = await asyncio.wait_for(
                        loop.run_in_executor(app.state.frame_executor, _run_frame),
                        timeout=max(0.1, float(args.push_frame_timeout_s)),
                    )
                except asyncio.TimeoutError:
                    msg = f"push_frame timeout after {args.push_frame_timeout_s:.1f}s"
                    print(f"#####[WS-GUARD] {msg}", flush=True)
                    await _send_json({"type": "error", "message": msg})
                    break
                except Exception as exc:
                    await _send_json({"type": "error", "message": repr(exc)})
                    break
                frame_process_ms = (time.time() - started) * 1000.0
                previous_process_ms = ws_debug.get("frame_process_ms")
                ws_debug["frame_process_ms"] = (
                    frame_process_ms if previous_process_ms is None
                    else float(previous_process_ms) * 0.9 + frame_process_ms * 0.1
                )
                if chunk_results is None:
                    continue
                if (
                    isinstance(chunk_results, tuple)
                    and chunk_results
                    and chunk_results[0] == "__scene_cut__"
                ):
                    _, cut_frame, cut_meta, scene_mad = chunk_results
                    ws_debug["scene_cut_count"] = int(
                        ws_debug.get("scene_cut_count", 0)
                    ) + 1
                    ws_debug["scene_cut_last_mad"] = float(scene_mad)
                    print(
                        f"#####[SCENE-CUT] MAD={float(scene_mad):.2f} >= "
                        f"{scene_cut_threshold:.2f}; resetting causal state",
                        flush=True,
                    )
                    await _reset_session("scene_cut")
                    reset_session = session

                    def _replay_scene_cut_frame():
                        acquired = app.state.inference_lock.acquire(
                            timeout=max(0.1, float(args.inference_lock_timeout_s))
                        )
                        if not acquired:
                            raise TimeoutError(
                                "inference lock timeout while replaying scene-cut frame"
                            )
                        try:
                            return reset_session.push_frame(
                                cut_frame,
                                frame_meta=cut_meta,
                                drain_results=False,
                            )
                        finally:
                            app.state.inference_lock.release()

                    try:
                        chunk_results = await asyncio.wait_for(
                            loop.run_in_executor(
                                app.state.frame_executor,
                                _replay_scene_cut_frame,
                            ),
                            timeout=max(0.1, float(args.push_frame_timeout_s)),
                        )
                    except Exception as exc:
                        await _send_json(
                            {
                                "type": "error",
                                "message": f"scene-cut replay failed: {exc!r}",
                            }
                        )
                        break
                if isinstance(chunk_results, tuple) and chunk_results and isinstance(chunk_results[0], str):
                    sentinel = chunk_results[0]
                    if sentinel == "__no_person__":
                        _hold_reason = chunk_results[1] if len(chunk_results) > 1 else "no_person"
                        await _send_json({"type": "no_person", "reason": _hold_reason, "frames_in": frames_in})
                        continue
                    if sentinel == "__person_returned__":
                        face_gate_pending = True
                        gate_state["count"] = 0
                        gate_state["cx"] = None
                        gate_state["cy"] = None
                        gate_state["settle_gray"] = None
                        gate_state["pe_anchor"] = None
                        gate_state["settle_ax"] = None
                        gate_state["settle_ay"] = None
                        gate_state["subject_count"] = None
                        gate_state["cand"] = None
                        gate_state["cand_n"] = 0

                        if pe_report and pe_report.get("enhanced_prompt"):
                            session_prompt = pe_report["enhanced_prompt"]
                        await _reset_session("person_returned")
                        continue
                if isinstance(chunk_results, str) and chunk_results == "__gate_pe__":
                    anchor = gate_state.get("pe_anchor")
                    gate_state["pe_anchor"] = None
                    await _send_json({"type": "pe_running", "frames_in": frames_in})
                    try:
                        pe_report = await asyncio.wait_for(
                            asyncio.to_thread(
                                _enhance_prompt_sync,
                                raw_prompt=raw_session_prompt,
                                ref_image=ref_image,
                                pe_frame=anchor,
                                pe_model=args.pe_model,
                            ),
                            timeout=max(1.0, float(args.pe_timeout_s)),
                        )
                        enhanced = str(pe_report.get("enhanced_prompt") or raw_session_prompt)

                        def _swap_prompt(_sess=session, _txt=enhanced, _anchor=anchor):
                            with app.state.inference_lock:
                                _sess.prompt = _txt
                                _sess._encode_streaming_prompt(_anchor)
                        await asyncio.to_thread(_swap_prompt)
                        ws_debug["pe_report"] = pe_report
                        await _send_json({"type": "prompt_enhanced", **pe_report})
                    except Exception as exc:
                        _why = "timeout" if isinstance(exc, asyncio.TimeoutError) else repr(exc)
                        print(f"#####[PE] deferred enhance failed: {_why} -> raw prompt", flush=True)
                        await _send_json({"type": "prompt_enhanced", "enabled": True,
                                          "raw_prompt": raw_session_prompt, "enhanced_prompt": raw_session_prompt,
                                          "fallback": True, "error": True, "elapsed_s": 0.0})
                    pe_defer = False
                    _write_prompt_sidecar()
                    continue
                if isinstance(chunk_results, str):
                    await _send_json({"type": "waiting_face", "reason": chunk_results, "frames_in": frames_in})
                    continue
                if face_gate_pending:
                    face_gate_pending = False
                    print(f"#####[FACE-GATE] passed (mode=upper) -> editing starts", flush=True)
                frames_since_session_reset += 1
                ws_debug["frames_since_session_reset"] = frames_since_session_reset
                if gate_state.get("recount"):
                    gate_state["recount"] = False
                    print("#####[PERSON-GATE] person count changed -> re-edit (reset)", flush=True)

                    if pe_report and pe_report.get("enhanced_prompt"):
                        session_prompt = pe_report["enhanced_prompt"]
                    await _reset_session("person_count_changed")
                    continue
                elapsed = time.time() - started
                if not chunk_results:
                    await _send_json(
                        {
                            "type": "accepted",
                            "frames_in": frames_in,
                            "frames_out": frames_out,
                            "next_chunk_needs": session.frames_per_next_chunk - len(session.pending_frames),
                        }
                    )
                    continue

                last_count = 0
                for result in chunk_results:
                    last_count = await _send_chunk_result(
                        result,
                        fallback_meta=frame_meta,
                        fallback_elapsed=elapsed,
                    )
                if last_count == 0:
                    await _send_json(
                        {
                            "type": "accepted",
                            "frames_in": frames_in,
                            "frames_out": frames_out,
                            "next_chunk_needs": session.frames_per_next_chunk - len(session.pending_frames),
                        }
                    )
        except WebSocketDisconnect:
            pass
        except RuntimeError as exc:
            if "disconnect" not in str(exc).lower():
                raise
        finally:
            try:
                await _stop_person_monitor()
                await _stop_output_task()
                _final_rec = rec_base
                await asyncio.to_thread(_stop_recorders)
                if _final_rec is not None:
                    app.state.last_recording_dir = str(_final_rec)
                if session is not None:
                    await _close_session_safely(session, "finally")
                if getattr(app.state, "active_session", None) is session:
                    app.state.active_session = None
                if uplink_decoder is not None:
                    uplink_decoder.close()
                    uplink_decoder = None
                if downlink_encoder is not None:
                    downlink_encoder.close()
                    downlink_encoder = None
                if rocprof_region_active:
                    _rocprof_selected_region(resume=False)
                    rocprof_region_active = False
                ws_debug["closed_at"] = time.time()
                ws_debug["send_state"] = "closed"
            finally:
                if ticket is not None:
                    app.state.session_gate.release(ticket)

    return app

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Serve JoyOmni online v2v streaming inference.")
    parser.add_argument("--dit-ckpt", type=str, default=DEFAULT_DIT_CKPT)
    parser.add_argument("--vae-ckpt", type=str, default=None, help="Override VAE checkpoint dir (else uses the config default).")
    parser.add_argument("--text-encoder-ckpt", type=str, default=None, help="Override text-encoder checkpoint dir (else uses the config default).")
    parser.add_argument("--host", type=str, default="127.0.0.1")
    parser.add_argument("--port", type=int, default=7860)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--vae-device", type=str, default=None)
    parser.add_argument("--vae-encode-device", type=str, default=None)
    parser.add_argument("--vae-decode-device", type=str, default=None)
    parser.add_argument("--vae-pseudo-device", type=str, default=None)
    parser.add_argument("--postprocess-device", type=str, default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--height", type=int, default=720)
    parser.add_argument("--width", type=int, default=1248)
    parser.add_argument("--num-inference-steps", type=int, default=2)
    parser.add_argument("--use-pe", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--pe-model", type=str, default=None)
    parser.add_argument("--pe-timeout-s", type=float, default=20.0, help="Hard wall-clock cap for deferred prompt-enhancement. On timeout the session degrades to the RAW prompt and starts editing, so a slow/hung PE endpoint (bad network / provider stall) can never wedge the client in the prompt-enhancement state.")

    parser.add_argument("--face-detector-onnx", type=str, default=DEFAULT_FACE_DETECTOR_ONNX, help="YuNet ONNX weight for the face-presence gate. Missing -> gate disabled (edits run unconditionally).")
    parser.add_argument("--face-gate-score", type=float, default=0.35, help="Min YuNet confidence to count as a face. Lower = detects motion-blurred faces (fewer transient drops), but more false positives.")
    parser.add_argument("--face-present-min-ratio", type=float, default=0.15, help="Mid-session presence: a detected face counts as 'present' only if its short side is >= this fraction of the frame short side. A too-small/partial face (subject sat down so only the top of the head shows) counts as no-face -> black-hold, instead of letting the model t2v-hallucinate a person. 0 = any face counts. Higher = stricter (black out sooner when the face gets small/far). Normal editing faces measure ~0.37, so 0.15 has a wide margin.")
    parser.add_argument("--face-present-edge-margin", type=float, default=0.0, help="Mid-session presence: a face whose box comes within this fraction of ANY frame border counts as a HALF/partial face (turned/leaned out) -> no-face -> black-hold, so the model never edits a half-face frame (which it fills in as a t2v hallucination). 0 = no edge check (default: disabled -- the face-box edge check false-blacked too eagerly when a face merely neared a border). Set e.g. 0.02 to re-enable a lenient check.")
    parser.add_argument("--face-gate-min-below-ratio", type=float, default=0.20, help="mode=upper (garment): min fraction of frame HEIGHT that must be below the face (torso room). Bigger -> stricter (must back up more).")

    parser.add_argument("--face-gate-center-margin", type=float, default=0.28, help="Max |face-center-x - 0.5| (fraction of width) for a SINGLE subject to count as centered. Bigger -> more lenient. SKIPPED entirely when 2+ comparable faces are present (side-by-side people can't be centered). Note: motion/stability is enforced separately by --face-gate-move-eps + --face-gate-settle-drift, so this does not affect the swing-into-frame ghost fix.")
    parser.add_argument("--face-gate-move-eps", type=float, default=0.02, help="Max per-frame face-center movement (fraction of frame) to count as 'still'. Bigger -> tolerates more motion. Pairs with --face-gate-settle-drift (cumulative) so a slow glide can't creep through frame-by-frame.")
    parser.add_argument("--face-gate-settle-drift", type=float, default=0.05, help="Max CUMULATIVE face-center wander (fraction of frame) allowed across the whole settle streak. Closes the 'slow continuous glide' hole where every per-frame step is < move-eps but they sum to a big slide (swing-into-frame motion baked into chunk0 -> ghost/duplicate person). Smaller = must hold more still. Complements --face-gate-move-eps (per-frame) + --face-gate-stable-frames (streak length).")
    parser.add_argument("--face-gate-stable-frames", type=int, default=12, help="Consecutive centered+still frames required before editing starts (~24fps send rate, so 12 ≈ 0.5s).")
    parser.add_argument("--online-gate", action=argparse.BooleanOptionalAction, default=True, help="Master switch for MID-SESSION behavior (no-person black-hold + person-count re-edit). On (default) = presence/count monitoring runs for ALL sessions once editing begins. --no-online-gate to disable and make the inference path identical to the base commit.")
    parser.add_argument("--presence-absent-frames", type=int, default=12, help="Consecutive not-present frames (body missing, OR face too small / half-out per --face-present-*) before the output goes black. Small = stop FAST (less T2V leak on a quick sit-down / turn-away); larger = tolerate a brief occlusion / head-turn without black-holding. ~24fps, 12 ≈ 0.5s.")
    parser.add_argument("--presence-return-frames", type=int, default=24, help="Consecutive present (body+face) frames required to LEAVE the black-hold and re-run the startup gate. Separate from --presence-absent-frames so entry stays fast (black out quickly) while exit is well de-bounced: a face flickering through finger gaps while hands cover the face won't bounce no_face<->settling. ~24fps, 24 ≈ 1s.")
    parser.add_argument("--person-count-change-frames", type=int, default=24, help="Consecutive frames a NEW face count must hold before re-editing (reset chunk0) so people who enter later get edited. Debounce vs transient miscounts (sway / motion blur / a background face flickering in). ~24fps, 24 ≈ 1s.")
    parser.add_argument("--count-face-min-ratio", type=float, default=0.45, help="For person-count-change: a face counts as an additional subject only if its short side is >= this fraction of the MAIN (largest/foreground) face's short side. Excludes far-smaller BACKGROUND people (e.g. a coworker behind the subject) that otherwise flip the count and trigger spurious re-edits. Higher = stricter (ignore more background). Default 0.45.")
    parser.add_argument("--person-count-reedit", action=argparse.BooleanOptionalAction, default=True, help="Re-edit (reset chunk0) when the head count changes -- ALL modes incl. style. --no-person-count-reedit to disable.")
    parser.add_argument("--require-face", action=argparse.BooleanOptionalAction, default=True, help="Also black-hold when a body is present but NO face is detected (hand over face / turned away). --no-require-face to only gate on body.")
    parser.add_argument("--person-detector-pt", type=str, default=DEFAULT_PERSON_DETECTOR_PT, help="Native YOLOv8n PyTorch weight for accelerated person presence. Preferred over ONNX when present.")
    parser.add_argument("--person-detector-compile", type=str, default="max-autotune-no-cudagraphs", help="torch.compile mode for native YOLO; pass an empty string to disable.")
    parser.add_argument("--person-detector-onnx", type=str, default=DEFAULT_PERSON_DETECTOR_ONNX, help="Fixed-320 YOLOv8n ONNX fallback for cv2.dnn. Missing or incompatible -> person monitoring fails open.")
    parser.add_argument("--person-gate-conf", type=float, default=0.4, help="Min YOLO person-class score to count the person as present.")
    parser.add_argument(
        "--person-check-stride",
        type=int,
        default=8,
        help=(
            "Run continuous person/face/count monitoring every Nth frame during editing "
            "(default: 8, or 3 Hz at 24 fps). The startup face gate remains per-frame."
        ),
    )
    parser.add_argument("--person-body-flip-frames", type=int, default=6, help="Consecutive body-misses before the client reason flips to no_person. Below this, a lone YOLO dip (a hand/object over the face also clips the torso) keeps the current reason -- normally show_full_face -- so the hint doesn't strobe no_face<->no_person. Reason-only de-bounce; the black-hold timing (--presence-absent-frames) is unaffected. ~24fps, 6 ≈ 0.25s. Higher = more reluctant to ever show no_person; 1 = report no_person on the first miss (old behavior).")
    parser.add_argument("--output-quality", type=int, default=60)
    parser.add_argument("--uplink-codec", type=str, default="auto", choices=["auto", "jpeg"],
                        help="Uplink frame codec. 'auto' honors the client's request (H.264 if it supports WebCodecs, else JPEG). 'jpeg' forces the legacy per-frame JPEG path.")
    parser.add_argument("--downlink-codec", type=str, default="auto", choices=["auto", "jpeg"],
                        help="Downlink frame codec. 'auto' honors the client's request (H.264 if it supports WebCodecs VideoDecoder, else JPEG). 'jpeg' forces the legacy per-frame JPEG path.")
    parser.add_argument("--downlink-fps", type=int, default=24,
                        help="Framerate hint for the downlink H.264 encoder (affects GOP/keyframe interval only).")
    parser.add_argument("--prompt", type=str, default="Keep the person and scene temporally consistent while applying the requested edit.")
    parser.add_argument("--profile-timings", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--kv-reset-frames", type=int, default=0, help="Recreate streaming state every N input frames; 0 (default) keeps the bounded cache continuous and avoids promoting an in-motion frame to a new global sink.")
    parser.add_argument("--max-temporal-ids", type=int, default=None)
    parser.add_argument("--freeze-kv-on-static", action=argparse.BooleanOptionalAction, default=False, help="Experimental stale-tail anchor reuse. Disabled by default because false-static decisions cause visible pose ghosting and do not reduce model work.")
    parser.add_argument("--static-diff-thresh", type=float, default=0.5)
    parser.add_argument(
        "--scene-cut-threshold",
        type=float,
        default=25.0,
        help=(
            "Reset causal state when consecutive 64x36 luma frames exceed this "
            "mean absolute difference; 0 disables hard-cut detection."
        ),
    )
    parser.add_argument("--exact-global-sink-kv", action=argparse.BooleanOptionalAction, default=True, help="Store all layers of the permanent global-sink chunk from the final clean latent once; later bounded tail chunks retain the fast hybrid policy.")
    parser.add_argument("--preload", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--push-frame-timeout-s", type=float, default=60.0, help="Max seconds a single frame submission may block before releasing the WS session gate. The default covers one-time ROCm graph specialization; steady-state submissions remain sub-second.")
    parser.add_argument("--session-close-timeout-s", type=float, default=5.0, help="Best-effort session cleanup timeout during WS teardown.")
    parser.add_argument("--inference-lock-timeout-s", type=float, default=5.0, help="Max seconds to wait for the process-wide inference lock.")

    parser.add_argument("--record-dir", type=str, default=None, help="Directory to record input/output mp4s into (per-session subfolder). Off if unset.")
    parser.add_argument("--record-fps", type=int, default=24, help="Recording time base (fps) for muxed segments.")
    parser.add_argument("--record-codec", type=str, default="libx264", help="Recording video codec (PyAV/ffmpeg name).")
    parser.add_argument("--record-bitrate", type=int, default=8_000_000, help="Recording target bitrate in bits/sec.")
    parser.add_argument("--record-segment-seconds", type=int, default=300, help="Roll to a new mp4 segment every N seconds of recorded frames.")
    parser.add_argument("--download-crf", type=int, default=8, help="CRF for the downloaded final video (libx264, yuv420p). Matches Bernini's export_to_video (-crf 8): constant-quality, rate floats with content. Lower = higher quality/bigger. Set <0 to skip re-encode and stream the raw recording.")
    parser.add_argument("--download-preset", type=str, default="medium", help="libx264 preset for the download re-encode (quality/speed trade-off). The realtime recording uses ultrafast; the download re-encode can afford 'medium' for a smaller file at the same CRF.")
    args = parser.parse_args()

    return args

def main() -> None:
    args = parse_args()
    _configure_rocm_tunableop()
    app = create_app(args)
    uvicorn.run(app, host=args.host, port=args.port, log_level="info", ws_max_size=32 * 1024 * 1024)


if __name__ == "__main__":
    main()
