#!/usr/bin/env python3
"""Evaluate a CrackCatch checkpoint and emit the metrics the report needs.

Produces exactly what the project scope asks to report:
  * mAP@0.5 and mAP@0.5:0.95
  * precision and recall **per class**
  * a confusion matrix (written as PNG by Ultralytics, plus a JSON copy)
  * achieved inference FPS on the current device

Everything is written to a JSON file as well as printed, so the numbers in the
report and the presentation can be copied from one source of truth rather than
retyped from a terminal.

Usage:
    python model/scripts/evaluate.py \
        --data data/synthetic_yolo/data.yaml \
        --weights model/weights/crackcatch.pt
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


def resolve_device(requested: str) -> str:
    if requested != "auto":
        return requested
    try:
        import torch

        if torch.cuda.is_available():
            return "0"
        if torch.backends.mps.is_available():
            return "mps"
    except Exception:
        pass
    return "cpu"


def measure_fps(model, imgsz: int, device: str, runs: int = 30) -> dict[str, float]:
    """Measure achieved single-image inference throughput.

    Reported rather than claimed - the scope explicitly asks for the achieved
    figure. This is pure model inference (pre/post-processing included, batch
    of one), which is the number that matters for a live dashcam feed.
    """
    dummy = np.random.default_rng(0).integers(0, 255, (imgsz, imgsz, 3), dtype=np.uint8)
    for _ in range(5):  # warm up: first call includes lazy init and allocation
        model.predict(dummy, imgsz=imgsz, device=device, verbose=False)

    started = time.perf_counter()
    for _ in range(runs):
        model.predict(dummy, imgsz=imgsz, device=device, verbose=False)
    elapsed = time.perf_counter() - started
    return {
        "device": device,
        "runs": runs,
        "ms_per_frame": round(1000.0 * elapsed / runs, 2),
        "fps": round(runs / elapsed, 2) if elapsed > 0 else 0.0,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data", required=True)
    parser.add_argument("--weights", default=str(REPO_ROOT / "model" / "weights" / "crackcatch.pt"))
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--split", default="val", choices=["val", "test", "train"])
    parser.add_argument("--fps-runs", type=int, default=30)
    parser.add_argument("--also-cpu-fps", action="store_true",
                        help="Additionally measure CPU FPS, for the CPU/GPU table")
    parser.add_argument("--output", default=str(REPO_ROOT / "docs" / "metrics.json"))
    args = parser.parse_args()

    weights = Path(args.weights).expanduser().resolve()
    if not weights.exists():
        print(f"error: checkpoint not found: {weights}", file=sys.stderr)
        print("Train one first: python model/scripts/train.py --data <data.yaml>",
              file=sys.stderr)
        return 2

    data_path = Path(args.data).expanduser().resolve()
    if not data_path.exists():
        print(f"error: dataset config not found: {data_path}", file=sys.stderr)
        return 2

    from ultralytics import YOLO

    device = resolve_device(args.device)
    model = YOLO(str(weights))

    print("=" * 68)
    print("CrackCatch - model evaluation")
    print("=" * 68)
    print(f"  weights : {weights}")
    print(f"  dataset : {data_path}  (split: {args.split})")
    print(f"  device  : {device}")
    print("=" * 68)

    metrics = model.val(
        data=str(data_path),
        imgsz=args.imgsz,
        batch=args.batch,
        device=device,
        split=args.split,
        plots=True,
        verbose=True,
        # Keep evaluation artefacts beside the training runs instead of
        # letting Ultralytics drop a runs/ directory at the repo root.
        project=str(REPO_ROOT / "model" / "runs"),
        name=f"val_{Path(args.weights).stem}",
        exist_ok=True,
    )

    names = model.names if isinstance(model.names, dict) else dict(enumerate(model.names))
    box = metrics.box

    per_class: dict[str, dict[str, float]] = {}
    try:
        class_indices = list(getattr(box, "ap_class_index", [])) or list(names.keys())
        for position, class_id in enumerate(class_indices):
            label = names.get(int(class_id), str(class_id))
            per_class[label] = {
                "precision": round(float(box.p[position]), 4),
                "recall": round(float(box.r[position]), 4),
                "mAP50": round(float(box.ap50[position]), 4),
                "mAP50_95": round(float(box.ap[position]), 4),
            }
    except Exception as exc:  # pragma: no cover - shape varies across versions
        print(f"warning: per-class metrics unavailable ({exc})", file=sys.stderr)

    confusion: list[list[float]] | None = None
    try:
        confusion = metrics.confusion_matrix.matrix.tolist()
    except Exception:
        pass

    speed = measure_fps(model, args.imgsz, device, args.fps_runs)
    cpu_speed = None
    if args.also_cpu_fps and device != "cpu":
        cpu_speed = measure_fps(model, args.imgsz, "cpu", max(5, args.fps_runs // 3))

    report = {
        "evaluated_at": datetime.now(timezone.utc).isoformat(),
        "weights": str(weights),
        "dataset": str(data_path),
        "split": args.split,
        "imgsz": args.imgsz,
        "overall": {
            "mAP50": round(float(box.map50), 4),
            "mAP50_95": round(float(box.map), 4),
            "precision": round(float(box.mp), 4),
            "recall": round(float(box.mr), 4),
        },
        "per_class": per_class,
        "confusion_matrix": confusion,
        "confusion_matrix_labels": [names[k] for k in sorted(names)] + ["background"],
        "speed": speed,
        "speed_cpu": cpu_speed,
        "ultralytics_speed_ms": getattr(metrics, "speed", None),
        "plots_dir": str(getattr(metrics, "save_dir", "")),
    }

    print("\n" + "-" * 68)
    print("RESULTS")
    print("-" * 68)
    print(f"  mAP@0.5      : {report['overall']['mAP50']:.4f}")
    print(f"  mAP@0.5:0.95 : {report['overall']['mAP50_95']:.4f}")
    print(f"  precision    : {report['overall']['precision']:.4f}")
    print(f"  recall       : {report['overall']['recall']:.4f}")
    if per_class:
        print("\n  Per class:")
        print(f"    {'class':<12}{'P':>9}{'R':>9}{'mAP50':>9}{'mAP50-95':>11}")
        for label, values in per_class.items():
            print(f"    {label:<12}{values['precision']:>9.4f}{values['recall']:>9.4f}"
                  f"{values['mAP50']:>9.4f}{values['mAP50_95']:>11.4f}")
    print(f"\n  Inference    : {speed['fps']:.1f} FPS "
          f"({speed['ms_per_frame']:.1f} ms/frame) on {speed['device']}")
    if cpu_speed:
        print(f"  Inference CPU: {cpu_speed['fps']:.1f} FPS "
              f"({cpu_speed['ms_per_frame']:.1f} ms/frame)")
    if report["plots_dir"]:
        print(f"\n  Plots (confusion matrix, PR curves): {report['plots_dir']}")

    output = Path(args.output).resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2))
    print(f"  Metrics JSON: {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
