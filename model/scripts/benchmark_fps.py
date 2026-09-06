#!/usr/bin/env python3
"""Model-size ablation: accuracy vs. speed across YOLOv8 variants and devices.

This mirrors Table II of the review paper - the accuracy/FPS trade-off that
decides whether a model can run on an edge device (Raspberry Pi / Jetson) in a
vehicle or must sit on a server.

Reports, for each checkpoint and device: parameter count, achieved FPS, and
milliseconds per frame. Pair it with evaluate.py for the accuracy column.

Usage:
    python model/scripts/benchmark_fps.py \
        --weights model/weights/crackcatch.pt model/weights/yolov8s.pt \
        --devices cpu mps
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]


def available_devices() -> list[str]:
    devices = ["cpu"]
    try:
        import torch

        if torch.cuda.is_available():
            devices.append("0")
        if torch.backends.mps.is_available():
            devices.append("mps")
    except Exception:
        pass
    return devices


def bench(weights: Path, device: str, imgsz: int, runs: int, warmup: int) -> dict:
    from ultralytics import YOLO

    model = YOLO(str(weights))
    frame = np.random.default_rng(0).integers(0, 255, (imgsz, imgsz, 3), dtype=np.uint8)

    for _ in range(warmup):
        model.predict(frame, imgsz=imgsz, device=device, verbose=False)

    started = time.perf_counter()
    for _ in range(runs):
        model.predict(frame, imgsz=imgsz, device=device, verbose=False)
    elapsed = time.perf_counter() - started

    try:
        params = sum(p.numel() for p in model.model.parameters())
    except Exception:
        params = 0

    return {
        "weights": weights.name,
        "device": device,
        "imgsz": imgsz,
        "params_millions": round(params / 1e6, 2),
        "runs": runs,
        "ms_per_frame": round(1000.0 * elapsed / runs, 2),
        "fps": round(runs / elapsed, 2) if elapsed > 0 else 0.0,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--weights", nargs="+",
                        default=[str(REPO_ROOT / "model" / "weights" / "crackcatch.pt")])
    parser.add_argument("--devices", nargs="+", default=None,
                        help="Default: every device available on this machine")
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--runs", type=int, default=30)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--output", default=str(REPO_ROOT / "docs" / "benchmark.json"))
    args = parser.parse_args()

    devices = args.devices or available_devices()
    rows: list[dict] = []

    print("=" * 78)
    print("CrackCatch - speed / size ablation")
    print("=" * 78)
    print(f"{'checkpoint':<26}{'device':<8}{'params(M)':>11}{'ms/frame':>11}{'FPS':>9}")
    print("-" * 78)

    for weight_path in args.weights:
        path = Path(weight_path).expanduser().resolve()
        if not path.exists():
            print(f"{path.name:<26}{'-':<8}{'missing':>11}", file=sys.stderr)
            continue
        for device in devices:
            try:
                row = bench(path, device, args.imgsz, args.runs, args.warmup)
            except Exception as exc:
                print(f"{path.name:<26}{device:<8}  failed: {exc}", file=sys.stderr)
                continue
            rows.append(row)
            print(f"{row['weights']:<26}{row['device']:<8}"
                  f"{row['params_millions']:>11.2f}{row['ms_per_frame']:>11.2f}"
                  f"{row['fps']:>9.2f}")

    if not rows:
        print("error: nothing was benchmarked", file=sys.stderr)
        return 1

    print("-" * 78)
    print("Read alongside docs/metrics.json for the accuracy column: a smaller")
    print("model that runs on the vehicle can beat a larger one that cannot.")

    output = Path(args.output).resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(
        {"benchmarked_at": datetime.now(timezone.utc).isoformat(), "results": rows},
        indent=2,
    ))
    print(f"Wrote {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
