#!/usr/bin/env python3
"""Compare captured streaming output with its input and the reference demo.

The benchmark stores one H.264 access unit per output frame plus a manifest that
maps it back to the input-frame index.  This tool keeps that mapping intact,
builds a source/capture/reference contact sheet, and reports CPU-only temporal
metrics over moving regions.  It intentionally does not load any model or use a
GPU, so quality captures can be analyzed while the single model GPU stays live.
"""

from __future__ import annotations

import argparse
import json
import math
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import av
import cv2
import numpy as np


@dataclass(frozen=True)
class Crop:
    left: int
    top: int
    right: int
    bottom: int

    def apply(self, image: np.ndarray) -> np.ndarray:
        height, width = image.shape[:2]
        if not (
            0 <= self.left < self.right <= width
            and 0 <= self.top < self.bottom <= height
        ):
            raise ValueError(
                f"crop {self} is outside a {width}x{height} frame"
            )
        return image[self.top : self.bottom, self.left : self.right]


def _crop(value: str) -> Crop:
    try:
        values = tuple(int(part.strip()) for part in value.split(","))
    except ValueError as exc:
        raise argparse.ArgumentTypeError("crop must contain four integers") from exc
    if len(values) != 4:
        raise argparse.ArgumentTypeError("crop must be LEFT,TOP,RIGHT,BOTTOM")
    crop = Crop(*values)
    if crop.left < 0 or crop.top < 0 or crop.right <= crop.left or crop.bottom <= crop.top:
        raise argparse.ArgumentTypeError("crop bounds must describe a positive rectangle")
    return crop


def _positive_float(value: str) -> float:
    parsed = float(value)
    if not math.isfinite(parsed) or parsed <= 0:
        raise argparse.ArgumentTypeError("value must be a positive finite number")
    return parsed


def _decode_capture(capture_dir: Path) -> dict[int, np.ndarray]:
    manifest_path = capture_dir / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if not isinstance(manifest, list) or not manifest:
        raise ValueError(f"empty or invalid capture manifest: {manifest_path}")

    decoded: dict[int, np.ndarray] = {}
    h264 = av.codec.CodecContext.create("h264", "r")
    pending: deque[dict] = deque()
    try:
        for entry in sorted(manifest, key=lambda item: int(item["ordinal"])):
            source_index = entry.get("source_index")
            if source_index is None:
                continue
            payload = (capture_dir / entry["file"]).read_bytes()
            codec = str(entry.get("codec", "")).lower()
            if codec in {"jpg", "jpeg"}:
                image = cv2.imdecode(np.frombuffer(payload, np.uint8), cv2.IMREAD_COLOR)
                if image is None:
                    raise ValueError(f"failed to decode {entry['file']}")
                decoded[int(source_index)] = image
                continue
            if codec != "h264":
                raise ValueError(f"unsupported captured codec {codec!r}")
            pending.append(entry)
            for frame in h264.decode(av.Packet(payload)):
                if not pending:
                    raise ValueError("H.264 decoder produced an unassociated frame")
                owner = pending.popleft()
                decoded[int(owner["source_index"])] = frame.to_ndarray(format="bgr24")
        for frame in h264.decode(None):
            if not pending:
                raise ValueError("H.264 flush produced an unassociated frame")
            owner = pending.popleft()
            decoded[int(owner["source_index"])] = frame.to_ndarray(format="bgr24")
    finally:
        h264.close()
    if pending:
        raise ValueError(f"H.264 decoder did not emit {len(pending)} captured frames")
    return decoded


def _decode_selected_video_frames(
    path: Path,
    indices: Iterable[int],
    *,
    crop: Crop | None = None,
    output_size: tuple[int, int] | None = None,
) -> dict[int, np.ndarray]:
    wanted = set(int(index) for index in indices)
    if not wanted:
        return {}
    if min(wanted) < 0:
        raise ValueError("video frame indices cannot be negative")
    result: dict[int, np.ndarray] = {}
    with av.open(str(path)) as container:
        stream = container.streams.video[0]
        stream.thread_type = "AUTO"
        for index, frame in enumerate(container.decode(stream)):
            if index not in wanted:
                if index > max(wanted):
                    break
                continue
            image = frame.to_ndarray(format="bgr24")
            if crop is not None:
                image = crop.apply(image)
            if output_size is not None and image.shape[1::-1] != output_size:
                image = cv2.resize(image, output_size, interpolation=cv2.INTER_AREA)
            result[index] = image
            if len(result) == len(wanted):
                break
    missing = sorted(wanted - result.keys())
    if missing:
        preview = ", ".join(str(value) for value in missing[:8])
        raise ValueError(f"video {path} ended before frame(s) {preview}")
    return result


def _video_rate(path: Path) -> float:
    with av.open(str(path)) as container:
        rate = container.streams.video[0].average_rate
        if rate is None:
            raise ValueError(f"video has no average frame rate: {path}")
        return float(rate)


def _decode_reference(
    path: Path,
    source_indices: list[int],
    *,
    source_fps: float,
    start_seconds: float,
    source_crop: Crop,
    output_crop: Crop,
    output_size: tuple[int, int],
) -> tuple[dict[int, np.ndarray], dict[int, np.ndarray]]:
    reference_fps = _video_rate(path)
    frame_to_source_indices: dict[int, list[int]] = {}
    for source_index in source_indices:
        frame_index = round((start_seconds + source_index / source_fps) * reference_fps)
        frame_to_source_indices.setdefault(frame_index, []).append(source_index)
    frames = _decode_selected_video_frames(path, frame_to_source_indices)
    reference_source: dict[int, np.ndarray] = {}
    reference_output: dict[int, np.ndarray] = {}
    for frame_index, mapped_indices in frame_to_source_indices.items():
        frame = frames[frame_index]
        source = cv2.resize(source_crop.apply(frame), output_size, interpolation=cv2.INTER_AREA)
        output = cv2.resize(output_crop.apply(frame), output_size, interpolation=cv2.INTER_AREA)
        for source_index in mapped_indices:
            reference_source[source_index] = source
            reference_output[source_index] = output
    return reference_source, reference_output


def _stats(values: list[float]) -> dict[str, float | int]:
    if not values:
        return {"count": 0}
    array = np.asarray(values, dtype=np.float64)
    return {
        "count": int(array.size),
        "mean": round(float(array.mean()), 6),
        "p50": round(float(np.percentile(array, 50)), 6),
        "p95": round(float(np.percentile(array, 95)), 6),
        "max": round(float(array.max()), 6),
    }


def _gray_small(image: np.ndarray, width: int) -> np.ndarray:
    height = max(1, round(image.shape[0] * width / image.shape[1]))
    resized = cv2.resize(image, (width, height), interpolation=cv2.INTER_AREA)
    return cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)


def _temporal_metrics(
    source: dict[int, np.ndarray],
    output: dict[int, np.ndarray],
    *,
    analysis_width: int,
    motion_threshold: float,
) -> dict[str, object]:
    common = sorted(source.keys() & output.keys())
    flow_epe: list[float] = []
    flow_cosine: list[float] = []
    magnitude_ratio: list[float] = []
    warped_luma_l1: list[float] = []
    stale_edge_rate: list[float] = []
    motion_coverage: list[float] = []
    pair_metrics: list[dict[str, float | int | None]] = []

    for previous_index, current_index in zip(common, common[1:]):
        if current_index != previous_index + 1:
            continue
        source_previous = _gray_small(source[previous_index], analysis_width)
        source_current = _gray_small(source[current_index], analysis_width)
        output_previous = _gray_small(output[previous_index], analysis_width)
        output_current = _gray_small(output[current_index], analysis_width)

        source_forward = cv2.calcOpticalFlowFarneback(
            source_previous, source_current, None, 0.5, 3, 21, 3, 5, 1.2, 0
        )
        output_forward = cv2.calcOpticalFlowFarneback(
            output_previous, output_current, None, 0.5, 3, 21, 3, 5, 1.2, 0
        )
        source_magnitude = np.linalg.norm(source_forward, axis=2)
        output_magnitude = np.linalg.norm(output_forward, axis=2)
        moving = source_magnitude >= motion_threshold
        motion_coverage.append(float(moving.mean()))
        if not np.any(moving):
            continue

        delta = output_forward - source_forward
        pair_epe = float(np.linalg.norm(delta, axis=2)[moving].mean())
        flow_epe.append(pair_epe)
        dot = np.sum(output_forward * source_forward, axis=2)
        cosine = dot / np.maximum(output_magnitude * source_magnitude, 1e-4)
        pair_cosine = float(np.clip(cosine[moving], -1.0, 1.0).mean())
        flow_cosine.append(pair_cosine)
        pair_magnitude_ratio = float(
            np.median(output_magnitude[moving] / np.maximum(source_magnitude[moving], 1e-3))
        )
        magnitude_ratio.append(pair_magnitude_ratio)

        source_backward = cv2.calcOpticalFlowFarneback(
            source_current, source_previous, None, 0.5, 3, 21, 3, 5, 1.2, 0
        )
        grid_y, grid_x = np.mgrid[: source_current.shape[0], : source_current.shape[1]]
        map_x = (grid_x + source_backward[..., 0]).astype(np.float32)
        map_y = (grid_y + source_backward[..., 1]).astype(np.float32)
        warped_previous = cv2.remap(
            output_previous,
            map_x,
            map_y,
            interpolation=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_REFLECT101,
        )
        pair_warped_luma_l1 = (
            float(np.abs(output_current.astype(np.float32) - warped_previous)[moving].mean())
            / 255.0
        )
        warped_luma_l1.append(pair_warped_luma_l1)

        source_previous_edges = cv2.Canny(source_previous, 70, 140) > 0
        source_current_edges = cv2.Canny(source_current, 70, 140) > 0
        output_current_edges = cv2.Canny(output_current, 70, 140) > 0
        distance_previous = cv2.distanceTransform(
            (~source_previous_edges).astype(np.uint8), cv2.DIST_L2, 3
        )
        distance_current = cv2.distanceTransform(
            (~source_current_edges).astype(np.uint8), cv2.DIST_L2, 3
        )
        moving_dilated = cv2.dilate(moving.astype(np.uint8), np.ones((5, 5), np.uint8)) > 0
        evaluated_edges = output_current_edges & moving_dilated
        edge_count = int(evaluated_edges.sum())
        pair_stale_edge_rate = None
        if edge_count:
            stale = (
                evaluated_edges
                & (distance_previous + 1.0 < distance_current)
                & (distance_previous <= 3.0)
            )
            pair_stale_edge_rate = float(stale.sum() / edge_count)
            stale_edge_rate.append(pair_stale_edge_rate)

        pair_metrics.append(
            {
                "previous_source_index": previous_index,
                "source_index": current_index,
                "motion_coverage": round(float(moving.mean()), 6),
                "flow_endpoint_error_px": round(pair_epe, 6),
                "flow_direction_cosine": round(pair_cosine, 6),
                "flow_magnitude_ratio": round(pair_magnitude_ratio, 6),
                "motion_compensated_luma_l1": round(pair_warped_luma_l1, 6),
                "stale_edge_rate": (
                    round(pair_stale_edge_rate, 6)
                    if pair_stale_edge_rate is not None
                    else None
                ),
            }
        )

    worst_stale = sorted(
        (item for item in pair_metrics if item["stale_edge_rate"] is not None),
        key=lambda item: float(item["stale_edge_rate"]),
        reverse=True,
    )[:12]
    worst_epe = sorted(
        pair_metrics,
        key=lambda item: float(item["flow_endpoint_error_px"]),
        reverse=True,
    )[:12]

    return {
        "analysis_width": analysis_width,
        "motion_threshold_px_at_analysis_scale": motion_threshold,
        "adjacent_frame_pairs": len(flow_epe),
        "motion_coverage": _stats(motion_coverage),
        "flow_endpoint_error_px": _stats(flow_epe),
        "flow_direction_cosine": _stats(flow_cosine),
        "flow_magnitude_ratio": _stats(magnitude_ratio),
        "motion_compensated_luma_l1": _stats(warped_luma_l1),
        "stale_edge_rate": _stats(stale_edge_rate),
        "worst_stale_edge_pairs": worst_stale,
        "worst_flow_endpoint_pairs": worst_epe,
        "interpretation": {
            "flow_endpoint_error_px": "lower is better",
            "flow_direction_cosine": "higher is better",
            "flow_magnitude_ratio": "closer to 1 is better; near 0 indicates stuck output",
            "motion_compensated_luma_l1": "lower is better when motion transfer remains near 1",
            "stale_edge_rate": "lower is better; detects output edges closer to the prior pose",
        },
    }


def _label(image: np.ndarray, text: str, width: int) -> np.ndarray:
    height = round(image.shape[0] * width / image.shape[1])
    tile = cv2.resize(image, (width, height), interpolation=cv2.INTER_AREA)
    overlay = tile.copy()
    cv2.rectangle(overlay, (0, 0), (width, 28), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.72, tile, 0.28, 0, tile)
    cv2.putText(tile, text, (8, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (255, 255, 255), 1, cv2.LINE_AA)
    return tile


def _write_contact_sheet(
    source: dict[int, np.ndarray],
    capture: dict[int, np.ndarray],
    reference: dict[int, np.ndarray] | None,
    *,
    fps: float,
    sample_indices: list[int],
    output_path: Path,
    tile_width: int,
    columns: int = 5,
) -> None:
    if not sample_indices:
        raise ValueError("contact sheet needs at least one sample frame")
    columns = min(columns, len(sample_indices))
    blocks: list[np.ndarray] = []
    for offset in range(0, len(sample_indices), columns):
        group = sample_indices[offset : offset + columns]
        rows = []
        kinds = [("source", source), ("capture", capture)]
        if reference is not None:
            kinds.append(("official", reference))
        for kind, frames in kinds:
            tiles = [
                _label(frames[index], f"{kind}  t={index / fps:.1f}s  i={index}", tile_width)
                for index in group
            ]
            while len(tiles) < columns:
                tiles.append(np.zeros_like(tiles[0]))
            rows.append(np.hstack(tiles))
        blocks.append(np.vstack(rows))
    sheet = np.vstack(blocks)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(output_path), sheet, [cv2.IMWRITE_JPEG_QUALITY, 92]):
        raise OSError(f"failed to write contact sheet: {output_path}")


def _contact_sheet(
    source: dict[int, np.ndarray],
    capture: dict[int, np.ndarray],
    reference: dict[int, np.ndarray] | None,
    *,
    fps: float,
    output_path: Path,
    tile_width: int,
) -> list[int]:
    common = sorted(source.keys() & capture.keys())
    if not common:
        raise ValueError("no source/capture frames overlap")
    last_second = int(common[-1] / fps)
    sample_indices = []
    for second in range(last_second + 1):
        target = round(second * fps)
        sample_indices.append(min(common, key=lambda index: abs(index - target)))
    _write_contact_sheet(
        source,
        capture,
        reference,
        fps=fps,
        sample_indices=sample_indices,
        output_path=output_path,
        tile_width=tile_width,
    )
    return sample_indices


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--capture-dir", type=Path, required=True)
    parser.add_argument("--source-video", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--fps", type=_positive_float, default=24.0)
    parser.add_argument("--analysis-width", type=int, default=312)
    parser.add_argument("--motion-threshold", type=_positive_float, default=0.75)
    parser.add_argument("--tile-width", type=int, default=320)
    parser.add_argument("--reference-video", type=Path)
    parser.add_argument("--reference-start", type=float, default=140.0)
    parser.add_argument(
        "--reference-source-crop", type=_crop, default=Crop(35, 183, 928, 699)
    )
    parser.add_argument(
        "--reference-output-crop", type=_crop, default=Crop(1002, 183, 1918, 699)
    )
    return parser


def main() -> int:
    args = build_parser().parse_args()
    if args.analysis_width < 64 or args.tile_width < 64:
        raise SystemExit("analysis and tile widths must be at least 64 pixels")
    for path in (args.capture_dir, args.source_video):
        if not path.exists():
            raise SystemExit(f"path does not exist: {path}")
    if args.reference_video is not None and not args.reference_video.is_file():
        raise SystemExit(f"reference video does not exist: {args.reference_video}")

    capture = _decode_capture(args.capture_dir)
    source_indices = sorted(capture)
    source = _decode_selected_video_frames(
        args.source_video,
        source_indices,
        output_size=next(iter(capture.values())).shape[1::-1],
    )
    reference_source = None
    reference_output = None
    if args.reference_video is not None:
        reference_source, reference_output = _decode_reference(
            args.reference_video,
            source_indices,
            source_fps=args.fps,
            start_seconds=args.reference_start,
            source_crop=args.reference_source_crop,
            output_crop=args.reference_output_crop,
            output_size=next(iter(capture.values())).shape[1::-1],
        )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    capture_metrics = _temporal_metrics(
        source,
        capture,
        analysis_width=args.analysis_width,
        motion_threshold=args.motion_threshold,
    )
    contact_sheet_path = args.output_dir / "contact-sheet.jpg"
    sampled = _contact_sheet(
        source,
        capture,
        reference_output,
        fps=args.fps,
        output_path=contact_sheet_path,
        tile_width=args.tile_width,
    )
    report: dict[str, object] = {
        "capture_dir": str(args.capture_dir),
        "source_video": str(args.source_video),
        "capture_frames": len(capture),
        "source_index_first": source_indices[0],
        "source_index_last": source_indices[-1],
        "fps": args.fps,
        "sampled_source_indices": sampled,
        "contact_sheet": str(contact_sheet_path),
        "capture_metrics": capture_metrics,
    }
    worst_pairs = capture_metrics["worst_stale_edge_pairs"]
    peak_indices: list[int] = []
    for pair in worst_pairs[:6]:
        for key in ("previous_source_index", "source_index"):
            index = int(pair[key])
            if index not in peak_indices:
                peak_indices.append(index)
    if peak_indices:
        peaks_path = args.output_dir / "worst-stale-pairs.jpg"
        _write_contact_sheet(
            source,
            capture,
            reference_output,
            fps=args.fps,
            sample_indices=peak_indices,
            output_path=peaks_path,
            tile_width=args.tile_width,
            columns=4,
        )
        report["worst_stale_pair_indices"] = peak_indices
        report["worst_stale_contact_sheet"] = str(peaks_path)
    worst_flow_pairs = capture_metrics["worst_flow_endpoint_pairs"]
    flow_peak_indices: list[int] = []
    for pair in worst_flow_pairs[:6]:
        for key in ("previous_source_index", "source_index"):
            index = int(pair[key])
            if index not in flow_peak_indices:
                flow_peak_indices.append(index)
    if flow_peak_indices:
        flow_peaks_path = args.output_dir / "worst-flow-pairs.jpg"
        _write_contact_sheet(
            source,
            capture,
            reference_output,
            fps=args.fps,
            sample_indices=flow_peak_indices,
            output_path=flow_peaks_path,
            tile_width=args.tile_width,
            columns=4,
        )
        report["worst_flow_pair_indices"] = flow_peak_indices
        report["worst_flow_contact_sheet"] = str(flow_peaks_path)
    if reference_source is not None and reference_output is not None:
        report["reference_video"] = str(args.reference_video)
        report["reference_start_seconds"] = args.reference_start
        report["official_reference_metrics"] = _temporal_metrics(
            reference_source,
            reference_output,
            analysis_width=args.analysis_width,
            motion_threshold=args.motion_threshold,
        )
        report["official_reference_note"] = (
            "Diagnostic only: the edited panel has different rendering and crop "
            "geometry, so use its contact-sheet row as the visual target rather "
            "than treating cross-panel flow as calibrated ground truth."
        )
    report_path = args.output_dir / "quality.json"
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
