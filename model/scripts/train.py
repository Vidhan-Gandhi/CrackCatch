#!/usr/bin/env python3
"""Train the CrackCatch YOLOv8 detector (pipeline stage 3).

Transfer-learns from a pretrained COCO checkpoint, which is the right choice
given the project's stated constraint of no dedicated GPU: the backbone
already knows edges, texture and shading, so only the head needs to learn what
a pothole looks like, and useful results arrive in tens of epochs rather than
hundreds.

Reproducibility (a stated non-functional requirement):
  * ``--seed`` is passed to Ultralytics, which seeds torch, numpy and random;
  * ``deterministic=True`` by default;
  * the train/val split is fixed by the dataset preparation script, not
    re-drawn here, so re-running training cannot leak val images into train.

Device selection is automatic: CUDA if present, else Apple Silicon MPS, else
CPU. CPU training works and is the documented fallback - it is slower, not
broken.

Examples:
    # Real data (after downloading RDD2022)
    python model/scripts/train.py --data data/rdd2022_yolo/data.yaml --epochs 60

    # Offline pipeline check on synthetic data
    python model/scripts/train.py --data data/synthetic_yolo/data.yaml --epochs 20
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUTPUT = REPO_ROOT / "model" / "weights" / "crackcatch.pt"


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


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--data", required=True, help="Path to a YOLO data.yaml")
    parser.add_argument("--base", default="yolov8n.pt",
                        help="Pretrained checkpoint to fine-tune (yolov8n.pt or yolov8s.pt)")
    parser.add_argument("--epochs", type=int, default=60)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--patience", type=int, default=20,
                        help="Early-stopping patience in epochs (0 disables)")
    parser.add_argument("--project", default=str(REPO_ROOT / "model" / "runs"))
    parser.add_argument("--name", default=None)
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT),
                        help="Where to copy the best checkpoint for serving")
    parser.add_argument("--no-promote", action="store_true",
                        help="Train but do not overwrite the served checkpoint")
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()

    data_path = Path(args.data).expanduser().resolve()
    if not data_path.exists():
        print(f"error: dataset config not found: {data_path}", file=sys.stderr)
        print("Run data/scripts/prepare_rdd2022.py or make_synthetic_dataset.py first.",
              file=sys.stderr)
        return 2

    try:
        from ultralytics import YOLO
    except ImportError:
        print("error: ultralytics is not installed. pip install -r requirements.txt",
              file=sys.stderr)
        return 2

    device = resolve_device(args.device)
    run_name = args.name or f"crackcatch_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

    print("=" * 68)
    print("CrackCatch - YOLOv8 training (pipeline stage 3)")
    print("=" * 68)
    print(f"  dataset : {data_path}")
    print(f"  base    : {args.base}")
    print(f"  epochs  : {args.epochs}   imgsz: {args.imgsz}   batch: {args.batch}")
    print(f"  device  : {device}   seed: {args.seed}")
    if device == "cpu":
        print("  NOTE: training on CPU. This works but is slow; expect minutes")
        print("        per epoch. Reduce --epochs or --imgsz to iterate faster.")
    print("=" * 68)

    model = YOLO(args.base)
    results = model.train(
        data=str(data_path),
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        device=device,
        workers=args.workers,
        seed=args.seed,
        deterministic=True,
        patience=args.patience,
        project=args.project,
        name=run_name,
        exist_ok=True,
        resume=args.resume,
        # Augmentation tuned for road imagery: horizontal flips are valid
        # (a pothole is symmetric), vertical flips are not (they would put
        # road surface in the sky). Modest HSV jitter stands in for the
        # lighting variation between morning and afternoon surveys.
        fliplr=0.5,
        flipud=0.0,
        degrees=5.0,
        translate=0.1,
        scale=0.4,
        hsv_h=0.015,
        hsv_s=0.6,
        hsv_v=0.4,
        mosaic=1.0,
        verbose=True,
    )

    run_dir = Path(results.save_dir) if hasattr(results, "save_dir") else Path(args.project) / run_name
    best = run_dir / "weights" / "best.pt"
    if not best.exists():
        print(f"error: training finished but {best} is missing", file=sys.stderr)
        return 1

    summary = {
        "trained_at": datetime.now(timezone.utc).isoformat(),
        "dataset": str(data_path),
        "base_checkpoint": args.base,
        "epochs": args.epochs,
        "imgsz": args.imgsz,
        "batch": args.batch,
        "device": device,
        "seed": args.seed,
        "run_dir": str(run_dir),
    }
    try:
        metrics = getattr(results, "results_dict", {}) or {}
        summary["metrics"] = {k: float(v) for k, v in metrics.items()
                              if isinstance(v, (int, float))}
    except Exception:
        pass

    if not args.no_promote:
        output = Path(args.output).resolve()
        output.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(best, output)
        summary["promoted_to"] = str(output)
        print(f"\nPromoted best checkpoint -> {output}")
        print("The backend and pipeline will pick this up automatically.")
    else:
        print(f"\nBest checkpoint left at {best} (not promoted)")

    (run_dir / "crackcatch_summary.json").write_text(json.dumps(summary, indent=2))
    print(f"Run summary: {run_dir / 'crackcatch_summary.json'}")
    print("\nNext: python model/scripts/evaluate.py --data",
          str(data_path), "--weights", args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
