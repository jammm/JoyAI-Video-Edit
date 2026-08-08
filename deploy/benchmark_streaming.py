#!/usr/bin/env python3
"""Synthetic end-to-end benchmark for the JoyAI streaming WebSocket API.

The client intentionally follows the browser demo's wire protocol: a JSON
``frame_meta`` message immediately precedes every binary input frame, and the
number of unacknowledged frames is bounded using the server's ``frames_in``
acknowledgements.  H.264 Annex-B is used in both directions by default; pass
``--jpeg-fallback`` to exercise the JPEG compatibility path.

Example:

    python deploy/benchmark_streaming.py --warmup-seconds 10 \
        --measure-seconds 60 --output-json profiles/streaming.json
"""

from __future__ import annotations

import argparse
import asyncio
import contextlib
import io
import json
import math
import ssl
import sys
import time
from collections import Counter, deque
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit, urlunsplit

import aiohttp
import av
import numpy as np
import websockets
from PIL import Image


DEFAULT_URL = "ws://127.0.0.1:8080/ws"
DEFAULT_PROMPT = "Turn the scene into a cinematic watercolor painting."


def _positive_float(value: str) -> float:
    parsed = float(value)
    if not math.isfinite(parsed) or parsed <= 0:
        raise argparse.ArgumentTypeError("must be a finite number greater than zero")
    return parsed


def _nonnegative_float(value: str) -> float:
    parsed = float(value)
    if not math.isfinite(parsed) or parsed < 0:
        raise argparse.ArgumentTypeError("must be a finite number greater than or equal to zero")
    return parsed


def _positive_int(value: str) -> int:
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("must be greater than zero")
    return parsed


def _nonnegative_int(value: str) -> int:
    parsed = int(value)
    if parsed < 0:
        raise argparse.ArgumentTypeError("must be greater than or equal to zero")
    return parsed


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="milliseconds")


def _round(value: float | int | None, digits: int = 3) -> float | int | None:
    if value is None:
        return None
    if isinstance(value, int):
        return value
    return round(float(value), digits)


def _distribution(values: list[float], *, digits: int = 3) -> dict[str, Any]:
    finite = np.asarray([v for v in values if math.isfinite(v)], dtype=np.float64)
    if finite.size == 0:
        return {"count": 0}
    return {
        "count": int(finite.size),
        "mean": _round(float(finite.mean()), digits),
        "p50": _round(float(np.percentile(finite, 50)), digits),
        "p95": _round(float(np.percentile(finite, 95)), digits),
        "max": _round(float(finite.max()), digits),
    }


def _debug_url(ws_url: str) -> str:
    parsed = urlsplit(ws_url)
    scheme = "https" if parsed.scheme == "wss" else "http"
    path = parsed.path or "/ws"
    if path.endswith("/ws"):
        path = path[:-3] + "/debug"
    else:
        path = "/debug"
    return urlunsplit((scheme, parsed.netloc, path, "", ""))


def _write_buffer_size(websocket: Any) -> int:
    try:
        transport = websocket.transport
        return int(transport.get_write_buffer_size())
    except Exception:
        return 0


class SyntheticFrames:
    """Cheap deterministic moving RGB test pattern."""

    def __init__(self, width: int, height: int) -> None:
        self.width = width
        self.height = height
        self._x = np.arange(width, dtype=np.uint16)[None, :]
        self._y = np.arange(height, dtype=np.uint16)[:, None]

    def frame(self, index: int) -> np.ndarray:
        image = np.empty((self.height, self.width, 3), dtype=np.uint8)
        image[..., 0] = (self._x + index * 5) & 0xFF
        image[..., 1] = (self._y * 2 + index * 3) & 0xFF
        image[..., 2] = ((self._x // 2 + self._y // 2) + index * 7) & 0xFF

        # A pair of moving blocks supplies unambiguous temporal motion while
        # remaining fully reproducible across machines.
        box_w = max(16, self.width // 12)
        box_h = max(16, self.height // 10)
        x0 = (index * 17) % max(1, self.width - box_w + 1)
        y0 = (index * 11) % max(1, self.height - box_h + 1)
        image[y0 : y0 + box_h, x0 : x0 + box_w] = (250, 40, 30)
        x1 = (self.width - box_w - index * 9) % max(1, self.width - box_w + 1)
        y1 = (index * 5 + self.height // 3) % max(1, self.height - box_h + 1)
        image[y1 : y1 + box_h, x1 : x1 + box_w] = (20, 230, 245)
        return image


class MovingImageFrames:
    """Deterministic slow pan over a photographic validation image."""

    def __init__(self, path: Path, width: int, height: int) -> None:
        with Image.open(path) as source:
            source = source.convert("RGB")
            # Retain an 8% crop margin so the source moves without wraparound.
            canvas_width = max(width + 2, int(round(width * 1.08)))
            canvas_height = max(height + 2, int(round(height * 1.08)))
            scale = max(canvas_width / source.width, canvas_height / source.height)
            resized = source.resize(
                (max(canvas_width, int(round(source.width * scale))),
                 max(canvas_height, int(round(source.height * scale)))),
                Image.Resampling.LANCZOS,
            )
        self._image = np.asarray(resized, dtype=np.uint8)
        self.width = width
        self.height = height
        self._max_x = self._image.shape[1] - width
        self._max_y = self._image.shape[0] - height

    def frame(self, index: int) -> np.ndarray:
        # Independent periods avoid a short looping diagonal trajectory.
        x = int(round((np.sin(index * (2.0 * np.pi / 241.0)) + 1.0) * 0.5 * self._max_x))
        y = int(round((np.sin(index * (2.0 * np.pi / 173.0) + 0.7) + 1.0) * 0.5 * self._max_y))
        return np.ascontiguousarray(self._image[y:y + self.height, x:x + self.width])


class H264Encoder:
    """Low-latency Annex-B encoder mirroring the browser's WebCodecs setup."""

    def __init__(self, width: int, height: int, fps: float) -> None:
        if width % 2 or height % 2:
            raise ValueError("H.264 yuv420p requires even width and height")
        from fractions import Fraction
        from av.video.frame import PictureType

        fps_int = max(1, int(round(fps)))
        self._ctx = av.codec.CodecContext.create("libx264", "w")
        self._ctx.width = width
        self._ctx.height = height
        self._ctx.pix_fmt = "yuv420p"
        self._ctx.time_base = Fraction(1, fps_int)
        self._ctx.framerate = Fraction(fps_int, 1)
        self._key_interval = max(1, fps_int * 2)
        self._ctx.options = {
            "preset": "ultrafast",
            "tune": "zerolatency",
            "profile": "baseline",
            "g": str(self._key_interval),
            "keyint_min": str(self._key_interval),
            "sc_threshold": "0",
            "bf": "0",
            "repeat_headers": "1",
        }
        self._ctx.open()
        self._pts = 0
        self._i_type = PictureType.I

    def encode(self, image: np.ndarray) -> list[tuple[bytes, bool]]:
        frame = av.VideoFrame.from_ndarray(np.ascontiguousarray(image), format="rgb24")
        frame.pts = self._pts
        if self._pts % self._key_interval == 0:
            frame.pict_type = self._i_type
        self._pts += 1
        return [(bytes(packet), bool(packet.is_keyframe)) for packet in self._ctx.encode(frame)]

    def close(self) -> None:
        with contextlib.suppress(Exception):
            # zerolatency/bf=0 should leave no delayed frames. Flush only to
            # release libavcodec state; flushed packets aren't sent after the
            # benchmark's fixed measurement window.
            list(self._ctx.encode(None))
        with contextlib.suppress(Exception):
            self._ctx.close()


class OutputValidator:
    """Decode output off the event loop so wire-receive timing stays honest."""

    def __init__(self, codec: str, width: int, height: int) -> None:
        self.codec = codec
        self.width = width
        self.height = height
        self._ctx: av.codec.CodecContext | None = None

    def decode(self, payload: bytes) -> list[tuple[int, int]]:
        if self.codec == "h264":
            if self._ctx is None:
                self._ctx = av.codec.CodecContext.create("h264", "r")
            frames = self._ctx.decode(av.packet.Packet(payload))
            return [(int(frame.width), int(frame.height)) for frame in frames]
        with Image.open(io.BytesIO(payload)) as image:
            image.load()
            return [(int(image.width), int(image.height))]

    def flush(self) -> list[tuple[int, int]]:
        if self._ctx is None:
            return []
        with contextlib.suppress(Exception):
            return [(int(frame.width), int(frame.height)) for frame in self._ctx.decode(None)]
        return []

    def close(self) -> None:
        if self._ctx is not None:
            with contextlib.suppress(Exception):
                self._ctx.close()
            self._ctx = None


@dataclass(slots=True)
class Config:
    url: str
    width: int
    height: int
    fps: float
    warmup_seconds: float
    measure_seconds: float
    prompt: str
    input_image: Path | None
    output_json: Path | None
    capture_output_dir: Path | None
    jpeg_fallback: bool
    no_output_decode: bool
    num_inference_steps: int
    max_temporal_ids: int
    max_pending: int
    jpeg_quality: int
    handshake_timeout: float
    drain_timeout: float
    send_timeout: float
    debug_interval: float
    insecure: bool
    online_monitoring: bool
    profile_timings: bool
    cache_last_denoise_kv: bool | None
    clean_kv_prefix_layers: int | None


@dataclass
class BenchmarkState:
    started_mono: float = 0.0
    measure_start_mono: float = 0.0
    measure_end_mono: float = 0.0
    finished_mono: float = 0.0
    negotiated: dict[str, Any] = field(default_factory=dict)
    sent: dict[int, dict[str, Any]] = field(default_factory=dict)
    acked_frames: int = 0
    ack_latencies_ms: list[float] = field(default_factory=list)
    max_pending_frames: int = 0
    max_write_buffer_bytes: int = 0
    encode_ms: list[float] = field(default_factory=list)
    send_ms: list[float] = field(default_factory=list)
    schedule_drops: int = 0
    backpressure_skips: int = 0
    encoder_no_packet: int = 0
    encoder_extra_packets: int = 0
    input_bytes: int = 0
    input_keyframes: int = 0
    input_keyframe_seqs: list[int] = field(default_factory=list)
    output_packets: list[dict[str, Any]] = field(default_factory=list)
    capture_outputs: bool = False
    captured_output_payloads: list[tuple[int | None, str, bytes]] = field(
        default_factory=list,
        repr=False,
    )
    output_bytes: int = 0
    output_keyframes: int = 0
    output_keyframe_source_seqs: list[int] = field(default_factory=list)
    output_meta_queue: deque[dict[str, Any]] = field(default_factory=deque)
    binary_without_meta: int = 0
    decoded_frames: int = 0
    decoded_resolutions: Counter[str] = field(default_factory=Counter)
    decode_errors: list[str] = field(default_factory=list)
    decode_queue_max: int = 0
    decode_queue_drops: int = 0
    chunks: list[dict[str, Any]] = field(default_factory=list)
    current_chunk: dict[str, Any] | None = None
    message_counts: Counter[str] = field(default_factory=Counter)
    latest_next_chunk_needs: int | None = None
    latest_server_frames_out: int = 0
    latest_server_frames_in: int = 0
    debug_samples: list[dict[str, Any]] = field(default_factory=list)
    debug_errors: list[str] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)
    fatal: asyncio.Event = field(default_factory=asyncio.Event)
    last_output_mono: float = 0.0

    def note_ack(self, frames_in: Any, now: float) -> None:
        try:
            new_ack = int(frames_in)
        except (TypeError, ValueError):
            return
        if new_ack <= self.acked_frames:
            return
        for seq in range(self.acked_frames + 1, new_ack + 1):
            record = self.sent.get(seq)
            if record is not None:
                latency_ms = (now - float(record["sent_mono"])) * 1000.0
                record["ack_latency_ms"] = latency_ms
                self.ack_latencies_ms.append(latency_ms)
        self.acked_frames = new_ack
        self.latest_server_frames_in = max(self.latest_server_frames_in, new_ack)


def _jpeg_encode(image: np.ndarray, quality: int) -> bytes:
    output = io.BytesIO()
    Image.fromarray(image, mode="RGB").save(output, format="JPEG", quality=quality)
    return output.getvalue()


async def _wait_for_handshake(
    websocket: Any,
    wanted: str,
    state: BenchmarkState,
    timeout: float,
) -> dict[str, Any]:
    deadline = time.monotonic() + timeout
    while True:
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            raise TimeoutError(f"timed out waiting for {wanted!r}")
        raw = await asyncio.wait_for(websocket.recv(), timeout=remaining)
        if not isinstance(raw, str):
            raise RuntimeError(f"received binary data while waiting for {wanted!r}")
        payload = json.loads(raw)
        msg_type = str(payload.get("type", "unknown"))
        state.message_counts[msg_type] += 1
        state.note_ack(payload.get("frames_in"), time.monotonic())
        if msg_type == wanted:
            return payload
        if msg_type == "queue_position":
            print(
                f"waiting for server slot (ahead={payload.get('ahead', payload.get('position'))})",
                file=sys.stderr,
                flush=True,
            )
            continue
        if msg_type in {"error", "session_timeout"}:
            raise RuntimeError(str(payload.get("message") or msg_type))


def _ssl_context(config: Config) -> ssl.SSLContext | bool | None:
    if not config.url.startswith("wss://"):
        return None
    if not config.insecure:
        return None
    context = ssl.create_default_context()
    context.check_hostname = False
    context.verify_mode = ssl.CERT_NONE
    return context


async def _decode_worker(
    queue: asyncio.Queue[bytes | None],
    validator: OutputValidator,
    state: BenchmarkState,
    executor: ThreadPoolExecutor,
) -> None:
    loop = asyncio.get_running_loop()
    while True:
        payload = await queue.get()
        try:
            if payload is None:
                return
            try:
                resolutions = await loop.run_in_executor(executor, validator.decode, payload)
                state.decoded_frames += len(resolutions)
                for width, height in resolutions:
                    state.decoded_resolutions[f"{width}x{height}"] += 1
                    if width != validator.width or height != validator.height:
                        state.decode_errors.append(
                            f"unexpected decoded resolution {width}x{height}; "
                            f"expected {validator.width}x{validator.height}"
                        )
            except Exception as exc:
                state.decode_errors.append(repr(exc))
        finally:
            queue.task_done()


def _record_chunk_start(payload: dict[str, Any], state: BenchmarkState, now: float) -> None:
    chunk = {
        "ordinal": len(state.chunks),
        "start_recv_mono": now,
        "done_recv_mono": None,
        "count": payload.get("count"),
        "source_seq_start": payload.get("source_seq_start"),
        "source_seq_end": payload.get("source_seq_end"),
        "server_elapsed_s": payload.get("elapsed"),
        "profile": {},
    }
    state.chunks.append(chunk)
    state.current_chunk = chunk


def _record_output_meta(payload: dict[str, Any], state: BenchmarkState) -> None:
    state.output_meta_queue.append(payload)
    chunk = state.current_chunk
    if chunk is None:
        return
    profile = payload.get("profile")
    if isinstance(profile, dict):
        chunk["profile"] = dict(profile)
        if profile.get("chunk_idx") is not None:
            chunk["chunk_idx"] = profile.get("chunk_idx")
    if payload.get("server_elapsed") is not None:
        chunk["server_elapsed_s"] = payload.get("server_elapsed")


def _record_chunk_done(payload: dict[str, Any], state: BenchmarkState, now: float) -> None:
    chunk = state.current_chunk
    if chunk is not None:
        chunk["done_recv_mono"] = now
        chunk["chunk_idx"] = payload.get("chunk_idx", chunk.get("chunk_idx"))
        chunk["ws_send_s"] = payload.get("ws_send_s")
        chunk["server_residence_s"] = payload.get("server_residence_s")
        profile = chunk.setdefault("profile", {})
        for key in ("ws_send_s", "server_residence_s"):
            if payload.get(key) is not None:
                profile[key] = payload[key]
    state.current_chunk = None


async def _receiver(
    websocket: Any,
    state: BenchmarkState,
    decode_queue: asyncio.Queue[bytes | None] | None,
) -> None:
    try:
        async for raw in websocket:
            now = time.monotonic()
            if isinstance(raw, str):
                try:
                    payload = json.loads(raw)
                except json.JSONDecodeError as exc:
                    state.errors.append(f"invalid server JSON: {exc}")
                    continue
                msg_type = str(payload.get("type", "unknown"))
                state.message_counts[msg_type] += 1
                state.note_ack(payload.get("frames_in"), now)
                try:
                    state.latest_server_frames_out = max(
                        state.latest_server_frames_out, int(payload.get("frames_out", 0) or 0)
                    )
                except (TypeError, ValueError):
                    pass
                if payload.get("next_chunk_needs") is not None:
                    with contextlib.suppress(TypeError, ValueError):
                        state.latest_next_chunk_needs = int(payload["next_chunk_needs"])

                if msg_type == "chunk_start":
                    _record_chunk_start(payload, state, now)
                elif msg_type == "output_frame":
                    _record_output_meta(payload, state)
                elif msg_type == "chunk_done":
                    _record_chunk_done(payload, state, now)
                elif msg_type == "error":
                    state.errors.append(str(payload.get("message") or payload))
                    state.fatal.set()
                elif msg_type == "session_timeout":
                    state.errors.append(str(payload.get("message") or "server session timed out"))
                    state.fatal.set()
                continue

            payload_bytes = bytes(raw)
            if state.output_meta_queue:
                meta = state.output_meta_queue.popleft()
            else:
                meta = {}
                state.binary_without_meta += 1
            recv_epoch_ms = time.time() * 1000.0
            source_seq: int | None = None
            with contextlib.suppress(TypeError, ValueError):
                if meta.get("source_seq") is not None:
                    source_seq = int(meta["source_seq"])
            latency_ms: float | None = None
            with contextlib.suppress(TypeError, ValueError):
                capture_ms = float(meta.get("t_capture_ms"))
                latency_ms = max(0.0, recv_epoch_ms - capture_ms)
            key = bool(meta.get("key"))
            state.output_packets.append(
                {
                    "recv_mono": now,
                    "source_seq": source_seq,
                    "latency_ms": latency_ms,
                    "bytes": len(payload_bytes),
                    "key": key,
                    "codec": str(meta.get("codec") or state.negotiated.get("downlink_codec") or "unknown"),
                }
            )
            state.output_bytes += len(payload_bytes)
            state.last_output_mono = now
            if state.capture_outputs:
                state.captured_output_payloads.append(
                    (
                        source_seq,
                        str(
                            meta.get("codec")
                            or state.negotiated.get("downlink_codec")
                            or "unknown"
                        ).lower(),
                        payload_bytes,
                    )
                )
            if key:
                state.output_keyframes += 1
                if source_seq is not None:
                    state.output_keyframe_source_seqs.append(source_seq)
            if decode_queue is not None:
                try:
                    decode_queue.put_nowait(payload_bytes)
                    state.decode_queue_max = max(state.decode_queue_max, decode_queue.qsize())
                except asyncio.QueueFull:
                    state.decode_queue_drops += 1
    except asyncio.CancelledError:
        raise
    except websockets.exceptions.ConnectionClosedOK:
        pass
    except websockets.exceptions.ConnectionClosed as exc:
        if not state.finished_mono:
            state.errors.append(f"WebSocket closed unexpectedly: {exc}")
            state.fatal.set()
    except Exception as exc:
        state.errors.append(f"receiver failed: {exc!r}")
        state.fatal.set()


async def _fetch_debug(
    session: aiohttp.ClientSession,
    url: str,
) -> dict[str, Any]:
    async with session.get(url) as response:
        response.raise_for_status()
        value = await response.json()
        if not isinstance(value, dict):
            raise TypeError("/debug did not return a JSON object")
        return value


async def _debug_sampler(
    config: Config,
    state: BenchmarkState,
    stop: asyncio.Event,
) -> None:
    url = _debug_url(config.url)
    connector = aiohttp.TCPConnector(ssl=False if config.insecure else None)
    timeout = aiohttp.ClientTimeout(total=max(1.0, min(5.0, config.debug_interval * 2.0)))
    async with aiohttp.ClientSession(connector=connector, timeout=timeout) as session:
        while not stop.is_set():
            try:
                sample = await _fetch_debug(session, url)
                sample["client_sample_mono"] = time.monotonic()
                state.debug_samples.append(sample)
            except asyncio.CancelledError:
                raise
            except Exception as exc:
                if isinstance(exc, aiohttp.ClientResponseError):
                    text = f"HTTP {exc.status}: {exc.message}"
                else:
                    text = f"{type(exc).__name__}: {exc}"
                if not state.debug_errors or state.debug_errors[-1] != text:
                    state.debug_errors.append(text)
            try:
                await asyncio.wait_for(stop.wait(), timeout=config.debug_interval)
            except asyncio.TimeoutError:
                pass


async def _send_pair(websocket: Any, meta: dict[str, Any], payload: bytes, timeout: float) -> float:
    started = time.perf_counter()

    async def _send() -> None:
        await websocket.send(json.dumps(meta, separators=(",", ":")))
        await websocket.send(payload)

    await asyncio.wait_for(_send(), timeout=timeout)
    return (time.perf_counter() - started) * 1000.0


async def _sender(
    websocket: Any,
    config: Config,
    state: BenchmarkState,
    codec: str,
) -> None:
    source = (
        MovingImageFrames(config.input_image, config.width, config.height)
        if config.input_image is not None
        else SyntheticFrames(config.width, config.height)
    )
    encoder = H264Encoder(config.width, config.height, config.fps) if codec == "h264" else None
    interval = 1.0 / config.fps
    run_start = time.monotonic()
    state.started_mono = run_start
    state.measure_start_mono = run_start + config.warmup_seconds
    state.measure_end_mono = state.measure_start_mono + config.measure_seconds
    next_tick = run_start
    source_index = 0
    next_seq = 1

    try:
        while next_tick < state.measure_end_mono:
            if state.fatal.is_set():
                break
            now = time.monotonic()
            if now < next_tick:
                try:
                    await asyncio.wait_for(state.fatal.wait(), timeout=next_tick - now)
                    break
                except asyncio.TimeoutError:
                    pass
                now = time.monotonic()

            if now - next_tick >= interval:
                missed = int((now - next_tick) // interval)
                state.schedule_drops += missed
                source_index += missed
                next_tick += missed * interval
            if next_tick >= state.measure_end_mono or now >= state.measure_end_mono:
                break

            pending = max(0, (next_seq - 1) - state.acked_frames)
            state.max_pending_frames = max(state.max_pending_frames, pending)
            state.max_write_buffer_bytes = max(
                state.max_write_buffer_bytes, _write_buffer_size(websocket)
            )
            if pending >= config.max_pending:
                state.backpressure_skips += 1
                source_index += 1
                next_tick += interval
                continue

            encode_started = time.perf_counter()
            image = source.frame(source_index)
            if codec == "h264":
                assert encoder is not None
                packets = encoder.encode(image)
            else:
                packets = [(_jpeg_encode(image, config.jpeg_quality), True)]
            state.encode_ms.append((time.perf_counter() - encode_started) * 1000.0)
            if not packets:
                state.encoder_no_packet += 1
            if len(packets) > 1:
                state.encoder_extra_packets += len(packets) - 1

            for encoded, is_key in packets:
                seq = next_seq
                next_seq += 1
                sent_mono = time.monotonic()
                capture_ms = time.time() * 1000.0
                phase = "warmup" if sent_mono < state.measure_start_mono else "measure"
                meta = {"type": "frame_meta", "seq": seq, "t_capture_ms": capture_ms}
                try:
                    send_ms = await _send_pair(websocket, meta, encoded, config.send_timeout)
                except Exception as exc:
                    state.errors.append(f"send failed at seq={seq}: {exc!r}")
                    state.fatal.set()
                    return
                state.send_ms.append(send_ms)
                state.sent[seq] = {
                    "sent_mono": sent_mono,
                    "capture_ms": capture_ms,
                    "phase": phase,
                    "bytes": len(encoded),
                    "key": is_key,
                    "source_index": source_index,
                    "encode_ms": state.encode_ms[-1],
                    "send_ms": send_ms,
                }
                state.input_bytes += len(encoded)
                if is_key:
                    state.input_keyframes += 1
                    state.input_keyframe_seqs.append(seq)
                pending = seq - state.acked_frames
                state.max_pending_frames = max(state.max_pending_frames, pending)
                state.max_write_buffer_bytes = max(
                    state.max_write_buffer_bytes, _write_buffer_size(websocket)
                )

            source_index += 1
            next_tick += interval
    finally:
        if encoder is not None:
            encoder.close()


def _server_queues_idle(sample: dict[str, Any] | None) -> bool:
    if not sample:
        return False
    session = sample.get("session")
    if not isinstance(session, dict):
        return False
    queues = session.get("queues")
    if not isinstance(queues, dict) or not queues:
        return False
    try:
        if any(int(value or 0) != 0 for value in queues.values()):
            return False
    except (TypeError, ValueError):
        return False
    states = session.get("worker_states")
    if isinstance(states, dict):
        for worker, value in states.items():
            if worker == "submit":
                continue
            if not isinstance(value, dict):
                continue
            worker_state = str(value.get("state", ""))
            if worker_state and not worker_state.startswith("wait_") and worker_state not in {"stop", "error"}:
                return False
    return True


async def _drain(config: Config, state: BenchmarkState) -> None:
    deadline = time.monotonic() + config.drain_timeout
    # One second of silence after every sampled server queue becomes empty is
    # enough to retain completed chunks without waiting for an intentionally
    # partial final chunk. Without /debug (e.g. a mock server), use a slightly
    # more conservative three-second quiet period.
    while time.monotonic() < deadline and not state.fatal.is_set():
        now = time.monotonic()
        all_acked = state.acked_frames >= len(state.sent)
        idle = _server_queues_idle(state.debug_samples[-1] if state.debug_samples else None)
        quiet_for = now - (state.last_output_mono or state.measure_end_mono)
        if all_acked and idle and quiet_for >= 1.0:
            return
        if all_acked and not state.debug_samples and quiet_for >= 3.0:
            return
        await asyncio.sleep(0.1)


def _measured_sequences(state: BenchmarkState) -> set[int]:
    return {seq for seq, record in state.sent.items() if record.get("phase") == "measure"}


def _chunk_is_measured(chunk: dict[str, Any], measured: set[int], state: BenchmarkState) -> bool:
    if measured:
        low, high = min(measured), max(measured)
        try:
            start = int(chunk.get("source_seq_start"))
            end = int(chunk.get("source_seq_end"))
            return end >= low and start <= high
        except (TypeError, ValueError):
            pass
    done = chunk.get("done_recv_mono")
    return bool(done is not None and state.measure_start_mono <= float(done) < state.measure_end_mono)


def _profile_summary(chunks: list[dict[str, Any]]) -> dict[str, Any]:
    values: dict[str, list[float]] = {}
    skip = {
        "chunk_idx",
        "input_frames",
        "output_frames",
        "profile_timings",
        "profile_timing_mode",
        "frozen_anchor_id",
    }
    for chunk in chunks:
        profile = chunk.get("profile")
        if not isinstance(profile, dict):
            continue
        for key, raw in profile.items():
            if key in skip or isinstance(raw, bool) or not isinstance(raw, (int, float)):
                continue
            value = float(raw)
            if key.endswith("_s"):
                out_key = f"{key[:-2]}_ms"
                value *= 1000.0
            elif key.endswith("_ms"):
                out_key = key
            else:
                continue
            values.setdefault(out_key, []).append(value)
    return {key: _distribution(series) for key, series in sorted(values.items())}


def _queue_summary(samples: list[dict[str, Any]]) -> dict[str, Any]:
    maxima: dict[str, int] = {}
    configured: dict[str, int | None] = {}
    pending_frames_max = 0
    for sample in samples:
        session = sample.get("session")
        if not isinstance(session, dict):
            continue
        with contextlib.suppress(TypeError, ValueError):
            pending_frames_max = max(pending_frames_max, int(session.get("pending_frames", 0) or 0))
        queues = session.get("queues")
        if isinstance(queues, dict):
            for key, value in queues.items():
                with contextlib.suppress(TypeError, ValueError):
                    maxima[str(key)] = max(maxima.get(str(key), 0), int(value or 0))
        sizes = session.get("queue_maxsize")
        if isinstance(sizes, dict):
            for key, value in sizes.items():
                with contextlib.suppress(TypeError, ValueError):
                    configured[str(key)] = None if value is None else int(value)
    return {
        "server_max_observed": maxima,
        "server_configured_max": configured,
        "server_pending_frames_max": pending_frames_max,
    }


def _last_debug_evidence(samples: list[dict[str, Any]]) -> dict[str, Any] | None:
    if not samples:
        return None
    sample = samples[-1]
    session = sample.get("session") if isinstance(sample.get("session"), dict) else {}
    ws = sample.get("ws") if isinstance(sample.get("ws"), dict) else {}
    return {
        "runtime_loaded": sample.get("runtime_loaded"),
        "devices": session.get("devices"),
        "counters": session.get("counters"),
        "ws_frames_in": ws.get("frames_in"),
        "ws_frames_out": ws.get("frames_out"),
        "ws_output_bytes": ws.get("output_bytes"),
        "ws_last_keys": ws.get("last_keys"),
        "ws_downlink_codec": ws.get("downlink_codec_live", ws.get("downlink_codec")),
    }


def build_summary(config: Config, state: BenchmarkState, started_at: str) -> dict[str, Any]:
    measured = _measured_sequences(state)
    measured_input_records = [state.sent[seq] for seq in sorted(measured)]
    measured_outputs = [p for p in state.output_packets if p.get("source_seq") in measured]
    receive_window_outputs = [
        p
        for p in state.output_packets
        if state.measure_start_mono <= float(p["recv_mono"]) < state.measure_end_mono
    ]
    measured_chunks = [c for c in state.chunks if _chunk_is_measured(c, measured, state)]
    chunk_server_ms = [
        float(c["server_elapsed_s"]) * 1000.0
        for c in measured_chunks
        if isinstance(c.get("server_elapsed_s"), (int, float))
    ]
    chunk_residence_ms = [
        float(c["server_residence_s"]) * 1000.0
        for c in measured_chunks
        if isinstance(c.get("server_residence_s"), (int, float))
    ]
    chunk_wire_ms = [
        float(c["ws_send_s"]) * 1000.0
        for c in measured_chunks
        if isinstance(c.get("ws_send_s"), (int, float))
    ]
    latency_ms = [float(p["latency_ms"]) for p in measured_outputs if p.get("latency_ms") is not None]
    receive_times = sorted(float(p["recv_mono"]) for p in receive_window_outputs)
    receive_gap_ms = [
        (later - earlier) * 1000.0
        for earlier, later in zip(receive_times, receive_times[1:])
    ]
    warmup_sent = sum(1 for record in state.sent.values() if record.get("phase") == "warmup")
    duration = config.measure_seconds
    queues = _queue_summary(state.debug_samples)
    queues["client_decode_max_observed"] = state.decode_queue_max
    queues["client_decode_drops"] = state.decode_queue_drops
    requested_codec = "jpeg" if config.jpeg_fallback else "h264"

    output_codec_counts = Counter(str(p.get("codec", "unknown")) for p in state.output_packets)
    protocol_ok = state.binary_without_meta == 0 and len(state.output_meta_queue) == 0
    decoded_ok = config.no_output_decode or (
        state.decoded_frames > 0 and not state.decode_errors and state.decode_queue_drops == 0
    )
    output_seen = bool(state.output_packets)
    ok = not state.errors and protocol_ok and output_seen and decoded_ok

    return {
        "ok": ok,
        "started_at": started_at,
        "config": {
            "url": config.url,
            "debug_url": _debug_url(config.url),
            "width": config.width,
            "height": config.height,
            "target_fps": config.fps,
            "warmup_seconds": config.warmup_seconds,
            "measure_seconds": config.measure_seconds,
            "prompt": config.prompt,
            "input_source": str(config.input_image) if config.input_image is not None else "synthetic",
            "cache_last_denoise_kv": config.cache_last_denoise_kv,
            "clean_kv_prefix_layers": config.clean_kv_prefix_layers,
            "num_inference_steps": config.num_inference_steps,
            "max_temporal_ids": config.max_temporal_ids,
            "max_pending": config.max_pending,
            "online_monitoring": config.online_monitoring,
            "profile_timings": config.profile_timings,
            "output_decode": not config.no_output_decode,
            "capture_output_dir": (
                str(config.capture_output_dir)
                if config.capture_output_dir is not None
                else None
            ),
        },
        "wall_seconds": _round(state.finished_mono - state.started_mono if state.finished_mono else None),
        "input": {
            "fps": _round(len(measured) / duration),
            "measured_frames": len(measured),
            "warmup_frames": warmup_sent,
            "total_frames": len(state.sent),
            "acked_frames": state.acked_frames,
            "bytes": state.input_bytes,
            "encode_ms": _distribution(
                [float(record["encode_ms"]) for record in measured_input_records]
            ),
            "send_ms": _distribution(
                [float(record["send_ms"]) for record in measured_input_records]
            ),
            "ack_latency_ms": _distribution(
                [
                    float(record["ack_latency_ms"])
                    for record in measured_input_records
                    if record.get("ack_latency_ms") is not None
                ]
            ),
        },
        "output": {
            "fps": _round(len(receive_window_outputs) / duration),
            "receive_window_frames": len(receive_window_outputs),
            "source_window_fps": _round(len(measured_outputs) / duration),
            "source_window_frames": len(measured_outputs),
            "total_packets": len(state.output_packets),
            "bytes": state.output_bytes,
            "end_to_end_latency_ms": _distribution(latency_ms),
            # Output arrives in model-sized bursts. This distribution makes
            # the longest wire silence explicit so the browser jitter buffer
            # can be qualified against it rather than relying on average FPS.
            "packet_interarrival_ms": _distribution(receive_gap_ms),
            "decoded_frames": state.decoded_frames,
            "decoded_resolutions": dict(sorted(state.decoded_resolutions.items())),
            "decode_errors": state.decode_errors[:20],
            "captured_packets": len(state.captured_output_payloads),
        },
        "chunks": {
            "measured": len(measured_chunks),
            "total": len(state.chunks),
            "server_elapsed_ms": _distribution(chunk_server_ms),
            "server_residence_ms": _distribution(chunk_residence_ms),
            "wire_send_ms": _distribution(chunk_wire_ms),
            "profile_ms": _profile_summary(measured_chunks),
        },
        "backpressure_and_drops": {
            "schedule_drops": state.schedule_drops,
            "backend_backpressure_skips": state.backpressure_skips,
            "max_pending_frames": state.max_pending_frames,
            "final_pending_frames": max(0, len(state.sent) - state.acked_frames),
            "max_websocket_write_buffer_bytes": state.max_write_buffer_bytes,
            "encoder_no_packet_frames": state.encoder_no_packet,
            "encoder_extra_packets": state.encoder_extra_packets,
            "binary_without_meta": state.binary_without_meta,
            "output_meta_without_binary": len(state.output_meta_queue),
        },
        "queues": queues,
        "codec_evidence": {
            "requested_uplink": requested_codec,
            "requested_downlink": requested_codec,
            "negotiated_uplink": state.negotiated.get("uplink_codec"),
            "negotiated_downlink": state.negotiated.get("downlink_codec"),
            "input_keyframes": state.input_keyframes,
            "input_keyframe_seqs": state.input_keyframe_seqs[:16],
            "output_keyframes": state.output_keyframes,
            "output_keyframe_source_seqs": state.output_keyframe_source_seqs[:16],
            "output_codec_packets": dict(sorted(output_codec_counts.items())),
        },
        "server_debug": {
            "samples": len(state.debug_samples),
            "sample_errors": state.debug_errors[:10],
            "final": _last_debug_evidence(state.debug_samples),
        },
        "message_counts": dict(sorted(state.message_counts.items())),
        "errors": state.errors[:20],
    }


async def run_benchmark(config: Config) -> dict[str, Any]:
    started_at = _utc_now()
    state = BenchmarkState()
    state.capture_outputs = config.capture_output_dir is not None
    requested_codec = "jpeg" if config.jpeg_fallback else "h264"
    connect_kwargs: dict[str, Any] = {
        "open_timeout": config.handshake_timeout,
        "close_timeout": 5.0,
        "ping_interval": None,
        "max_size": None,
        "max_queue": 256,
        "write_limit": 64 * 1024 * 1024,
    }
    context = _ssl_context(config)
    if context is not None:
        connect_kwargs["ssl"] = context

    receiver_task: asyncio.Task[None] | None = None
    debug_task: asyncio.Task[None] | None = None
    decode_task: asyncio.Task[None] | None = None
    decode_queue: asyncio.Queue[bytes | None] | None = None
    validator: OutputValidator | None = None
    executor: ThreadPoolExecutor | None = None
    debug_stop = asyncio.Event()

    try:
        async with websockets.connect(config.url, **connect_kwargs) as websocket:
            await _wait_for_handshake(websocket, "session_granted", state, config.handshake_timeout)
            start_payload = {
                "type": "start",
                "prompt": config.prompt,
                "width": config.width,
                "height": config.height,
                "num_inference_steps": config.num_inference_steps,
                "max_temporal_ids": config.max_temporal_ids,
                "seed": 42,
                "use_pe": False,
                "gate_enabled": False,
                "no_person_blank": config.online_monitoring,
                "require_face": config.online_monitoring,
                "person_count_reedit": config.online_monitoring,
                "kv_reset_frames": 0,
                "freeze_kv_on_static": False,
                "profile_timings": config.profile_timings,
                "uplink_codec": requested_codec,
                "downlink_codec": requested_codec,
                "output_quality": 80,
            }
            if config.cache_last_denoise_kv is not None:
                start_payload["cache_last_denoise_kv"] = config.cache_last_denoise_kv
            if config.clean_kv_prefix_layers is not None:
                start_payload["clean_kv_prefix_layers"] = config.clean_kv_prefix_layers
            await asyncio.wait_for(
                websocket.send(json.dumps(start_payload, separators=(",", ":"))),
                timeout=config.send_timeout,
            )
            state.negotiated = await _wait_for_handshake(
                websocket, "started", state, config.handshake_timeout
            )
            uplink_codec = str(state.negotiated.get("uplink_codec") or "jpeg").lower()
            downlink_codec = str(state.negotiated.get("downlink_codec") or "jpeg").lower()
            if uplink_codec not in {"h264", "jpeg"}:
                raise RuntimeError(f"unsupported negotiated uplink codec: {uplink_codec!r}")
            if downlink_codec not in {"h264", "jpeg"}:
                raise RuntimeError(f"unsupported negotiated downlink codec: {downlink_codec!r}")
            print(
                f"stream started: uplink={uplink_codec}, downlink={downlink_codec}, "
                f"{config.width}x{config.height}@{config.fps:g}",
                file=sys.stderr,
                flush=True,
            )

            if not config.no_output_decode:
                validator = OutputValidator(downlink_codec, config.width, config.height)
                decode_queue = asyncio.Queue(maxsize=256)
                executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="stream-output-decode")
                decode_task = asyncio.create_task(
                    _decode_worker(decode_queue, validator, state, executor),
                    name="output-validator",
                )

            receiver_task = asyncio.create_task(
                _receiver(websocket, state, decode_queue), name="websocket-receiver"
            )
            debug_task = asyncio.create_task(
                _debug_sampler(config, state, debug_stop), name="debug-sampler"
            )
            await _sender(websocket, config, state, uplink_codec)
            if not state.fatal.is_set():
                await _drain(config, state)
            state.finished_mono = time.monotonic()

            with contextlib.suppress(Exception):
                await asyncio.wait_for(
                    websocket.send(json.dumps({"type": "stop"})), timeout=config.send_timeout
                )
            if receiver_task is not None:
                with contextlib.suppress(asyncio.TimeoutError):
                    await asyncio.wait_for(receiver_task, timeout=5.0)
    except Exception as exc:
        state.errors.append(repr(exc))
        state.fatal.set()
        if not state.started_mono:
            state.started_mono = time.monotonic()
        state.finished_mono = time.monotonic()
    finally:
        debug_stop.set()
        if debug_task is not None:
            with contextlib.suppress(asyncio.TimeoutError, asyncio.CancelledError):
                await asyncio.wait_for(debug_task, timeout=3.0)
            if not debug_task.done():
                debug_task.cancel()
        if receiver_task is not None and not receiver_task.done():
            receiver_task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await receiver_task
        if decode_queue is not None and decode_task is not None:
            with contextlib.suppress(asyncio.TimeoutError):
                await asyncio.wait_for(decode_queue.join(), timeout=10.0)
            with contextlib.suppress(asyncio.QueueFull):
                decode_queue.put_nowait(None)
            with contextlib.suppress(asyncio.TimeoutError, asyncio.CancelledError):
                await asyncio.wait_for(decode_task, timeout=5.0)
            if not decode_task.done():
                decode_task.cancel()
        if validator is not None and executor is not None:
            loop = asyncio.get_running_loop()
            try:
                flushed = await loop.run_in_executor(executor, validator.flush)
                state.decoded_frames += len(flushed)
                for width, height in flushed:
                    state.decoded_resolutions[f"{width}x{height}"] += 1
            except Exception as exc:
                state.decode_errors.append(f"decoder flush: {exc!r}")
            with contextlib.suppress(Exception):
                await loop.run_in_executor(executor, validator.close)
            executor.shutdown(wait=True, cancel_futures=True)
        if not state.finished_mono:
            state.finished_mono = time.monotonic()

    if config.capture_output_dir is not None:
        config.capture_output_dir.mkdir(parents=True, exist_ok=True)
        manifest = []
        for ordinal, (source_seq, codec, payload) in enumerate(
            state.captured_output_payloads
        ):
            extension = "jpg" if codec in {"jpg", "jpeg"} else "h264"
            sequence_label = "none" if source_seq is None else f"{source_seq:06d}"
            filename = f"{ordinal:05d}-seq-{sequence_label}.{extension}"
            (config.capture_output_dir / filename).write_bytes(payload)
            manifest.append(
                {
                    "ordinal": ordinal,
                    "source_seq": source_seq,
                    "codec": codec,
                    "bytes": len(payload),
                    "file": filename,
                }
            )
        (config.capture_output_dir / "manifest.json").write_text(
            json.dumps(manifest, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )

    return build_summary(config, state, started_at)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Benchmark JoyAI's live WebSocket pipeline with deterministic synthetic "
            "frames and backend-ack backpressure."
        )
    )
    parser.add_argument("--url", default=DEFAULT_URL, help=f"WebSocket URL (default: {DEFAULT_URL})")
    parser.add_argument("--width", type=_positive_int, default=1248)
    parser.add_argument("--height", type=_positive_int, default=720)
    parser.add_argument("--fps", type=_positive_float, default=24.0)
    parser.add_argument("--warmup-seconds", type=_nonnegative_float, default=10.0)
    parser.add_argument("--measure-seconds", type=_positive_float, default=60.0)
    parser.add_argument("--prompt", default=DEFAULT_PROMPT)
    parser.add_argument(
        "--input-image",
        type=Path,
        help="Use a deterministic slow pan over this image instead of the synthetic pattern",
    )
    parser.add_argument("--output-json", type=Path, help="Also write the JSON summary to this path")
    parser.add_argument(
        "--capture-output-dir",
        type=Path,
        help="Save received output packets plus a source-sequence manifest",
    )
    parser.add_argument(
        "--jpeg-fallback",
        action="store_true",
        help="Request JPEG uplink/downlink instead of H.264 (server fallback is still honored)",
    )
    parser.add_argument(
        "--no-output-decode",
        action="store_true",
        help="Measure output packets without decoding them for codec/resolution validation",
    )
    parser.add_argument("--num-inference-steps", type=_positive_int, default=2)
    parser.add_argument("--max-temporal-ids", type=_positive_int, default=8)
    parser.add_argument("--max-pending", type=_positive_int, default=32)
    parser.add_argument("--jpeg-quality", type=_positive_int, default=85)
    parser.add_argument("--handshake-timeout", type=_positive_float, default=900.0)
    parser.add_argument("--drain-timeout", type=_positive_float, default=30.0)
    parser.add_argument("--send-timeout", type=_positive_float, default=10.0)
    parser.add_argument("--debug-interval", type=_positive_float, default=1.0)
    parser.add_argument(
        "--online-monitoring",
        action="store_true",
        help="Exercise continuous person/face/count monitoring without the startup hold-still gate",
    )
    parser.add_argument(
        "--profile-timings",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Collect server CUDA-event stage timings (disable for an uninstrumented throughput run)",
    )
    cache_group = parser.add_mutually_exclusive_group()
    cache_group.add_argument(
        "--cache-last-denoise-kv",
        dest="cache_last_denoise_kv",
        action="store_true",
        default=None,
        help="Store target K/V from the final denoise call (fast approximate cache)",
    )
    cache_group.add_argument(
        "--exact-clean-kv",
        dest="cache_last_denoise_kv",
        action="store_false",
        help="Run the separate exact clean-latent KV storage pass",
    )
    parser.add_argument(
        "--clean-kv-prefix-layers",
        type=_nonnegative_int,
        help="With fast caching, overwrite this many leading layers from the exact clean latent",
    )
    parser.add_argument(
        "--insecure",
        action="store_true",
        help="Disable TLS certificate verification for wss:// and /debug (test systems only)",
    )
    return parser


def _config_from_args(args: argparse.Namespace, parser: argparse.ArgumentParser) -> Config:
    parsed = urlsplit(args.url)
    if parsed.scheme not in {"ws", "wss"} or not parsed.netloc:
        parser.error("--url must be a ws:// or wss:// URL with a host")
    if not 1 <= args.jpeg_quality <= 100:
        parser.error("--jpeg-quality must be between 1 and 100")
    if not args.jpeg_fallback and (args.width % 2 or args.height % 2):
        parser.error("H.264 mode requires even --width and --height")
    if args.input_image is not None and not args.input_image.is_file():
        parser.error(f"--input-image does not exist or is not a file: {args.input_image}")
    return Config(
        url=args.url,
        width=args.width,
        height=args.height,
        fps=args.fps,
        warmup_seconds=args.warmup_seconds,
        measure_seconds=args.measure_seconds,
        prompt=args.prompt,
        input_image=args.input_image,
        output_json=args.output_json,
        capture_output_dir=args.capture_output_dir,
        jpeg_fallback=args.jpeg_fallback,
        no_output_decode=args.no_output_decode,
        num_inference_steps=args.num_inference_steps,
        max_temporal_ids=args.max_temporal_ids,
        max_pending=args.max_pending,
        jpeg_quality=args.jpeg_quality,
        handshake_timeout=args.handshake_timeout,
        drain_timeout=args.drain_timeout,
        send_timeout=args.send_timeout,
        debug_interval=args.debug_interval,
        insecure=args.insecure,
        online_monitoring=args.online_monitoring,
        profile_timings=args.profile_timings,
        cache_last_denoise_kv=args.cache_last_denoise_kv,
        clean_kv_prefix_layers=args.clean_kv_prefix_layers,
    )


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    config = _config_from_args(parser.parse_args(argv), parser)
    try:
        summary = asyncio.run(run_benchmark(config))
    except KeyboardInterrupt:
        print('{"ok":false,"errors":["interrupted"]}')
        return 130
    encoded = json.dumps(summary, sort_keys=True, separators=(",", ":"))
    print(encoded)
    if config.output_json is not None:
        config.output_json.parent.mkdir(parents=True, exist_ok=True)
        config.output_json.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return 0 if summary.get("ok") else 1


if __name__ == "__main__":
    raise SystemExit(main())
