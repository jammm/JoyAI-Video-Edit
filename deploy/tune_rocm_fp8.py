"""Offline-tune JoyOmni's steady-state OCP-FP8 GEMMs on one ROCm GPU.

Run with the same one-GPU ``ROCR_VISIBLE_DEVICES`` mask used by the server.
PyTorch validates the output against its ROCm, library, and gfx ISA versions.
"""

from __future__ import annotations

import argparse
import gc
import time
from pathlib import Path

import torch
from torch.cuda import tunable

from xvideo.models.dit.fp8_linear import (
    _quantize_per_token_rocm,
    _quantize_weight_per_channel,
)


SHAPES = (
    (1024, 4096, 12288),
    (1024, 4096, 4096),
    (1024, 4096, 16384),
    (1024, 16384, 4096),
    (1560, 4096, 12288),
    (1560, 4096, 4096),
    (1560, 4096, 16384),
    (1560, 16384, 4096),
    (3120, 4096, 12288),
    (3120, 4096, 4096),
    (3120, 4096, 16384),
    (3120, 16384, 4096),
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(__file__).resolve().parent
        / "deps/cache/tunableop_results_gfx950.csv",
    )
    parser.add_argument("--duration-ms", type=int, default=10)
    parser.add_argument("--iterations", type=int, default=30)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if torch.version.hip is None:
        raise RuntimeError("This tuner requires a ROCm PyTorch build")
    props = torch.cuda.get_device_properties(0)
    if not str(props.gcnArchName).startswith("gfx950"):
        raise RuntimeError(f"Expected gfx950, got {props.gcnArchName}")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    tunable.set_filename(str(args.output), insert_device_ordinal=False)
    tunable.enable(True)
    tunable.tuning_enable(True)
    tunable.record_untuned_enable(False)
    tunable.set_max_tuning_duration(args.duration_ms)
    tunable.set_max_tuning_iterations(args.iterations)

    for m, k, n in SHAPES:
        x = torch.randn(m, k, device="cuda", dtype=torch.bfloat16)
        weight = torch.randn(n, k, device="cuda", dtype=torch.bfloat16)
        bias = torch.randn(n, device="cuda", dtype=torch.bfloat16)
        x_q, x_scale = _quantize_per_token_rocm(x)
        weight_q, weight_scale = _quantize_weight_per_channel(weight)
        started = time.perf_counter()
        torch._scaled_mm(
            x_q,
            weight_q,
            x_scale,
            weight_scale,
            bias=bias,
            out_dtype=torch.bfloat16,
            use_fast_accum=True,
        )
        torch.cuda.synchronize()
        print(
            f"M={m} K={k} N={n}: {time.perf_counter() - started:.2f}s",
            flush=True,
        )
        del x, weight, bias, x_q, x_scale, weight_q, weight_scale
        gc.collect()
        torch.cuda.empty_cache()

    print(f"tuned {len(tunable.get_results())} shapes -> {args.output}")


if __name__ == "__main__":
    main()
