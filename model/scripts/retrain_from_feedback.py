#!/usr/bin/env python3
"""Stage 8 - continuous model improvement from dashboard feedback.

Closes the loop drawn in the pipeline diagram: detections an authority user
has reviewed on the dashboard become training data, the model is re-fine-tuned
on the accumulated corpus, the new checkpoint is evaluated against the current
one, and it is promoted **only if it is actually better**.

    export -> retrain -> evaluate -> promote

What counts as a training signal (and why):
  * ``Verified`` / ``Scheduled`` / ``Repaired`` - a human confirmed the defect
    is real, so the box is a true positive.
  * ``Rejected`` - a human confirmed it is not, so the image becomes a hard
    negative (kept with an empty label file). These matter more than extra
    positives: they teach the model to stop firing on tar patches and shadows,
    which is the dominant false-positive mode.
  * ``review_label`` - the class an authority corrected it to, overriding the
    model's own prediction.

Untouched ``New`` detections are deliberately excluded. Training on the
model's own unreviewed output is self-distillation of its existing bias, and
would make the model more confident without making it more correct.

This is a documented, runnable script rather than an automatic scheduled job -
promoting a model into a public-safety workflow should stay a human decision.

Usage:
    # 1. Export what the dashboard has reviewed
    python model/scripts/retrain_from_feedback.py export --output data/feedback_yolo

    # 2. Retrain, evaluate against the incumbent, and promote if better
    python model/scripts/retrain_from_feedback.py retrain \
        --dataset data/feedback_yolo/data.yaml --epochs 30

    # Or run the whole loop
    python model/scripts/retrain_from_feedback.py all --epochs 30
"""

from __future__ import annotations

import argparse
import asyncio
import json
import random
import shutil
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "backend"))
sys.path.insert(0, str(REPO_ROOT / "model"))

CLASS_NAMES = ["pothole", "crack"]
CLASS_INDEX = {name: i for i, name in enumerate(CLASS_NAMES)}
POSITIVE_STATUSES = {"Verified", "Scheduled", "Repaired"}


# --------------------------------------------------------------------- export


async def fetch_reviewed() -> list[dict]:
    from app.core.config import get_settings
    from app.db.mongo import database
    from app.db.repository import DefectRepository

    settings = get_settings()
    await database.connect(settings)
    if database.backend != "mongodb":
        print(
            "warning: connected to the in-memory fallback, which holds no "
            "history. Start MongoDB (`docker compose up -d mongo`) so the "
            "dashboard's reviewed detections are actually available.",
            file=sys.stderr,
        )
    repository = DefectRepository(database.db)
    records = await repository.export_for_retraining(include_rejected=True)
    await database.close()
    return records


def to_yolo_row(defect: dict, image_w: int, image_h: int) -> str | None:
    """Convert a stored defect's bbox into a normalised YOLO row."""
    box = defect.get("bbox") or {}
    try:
        x1, y1 = float(box["x1"]), float(box["y1"])
        x2, y2 = float(box["x2"]), float(box["y2"])
    except (KeyError, TypeError, ValueError):
        return None

    x1, x2 = sorted((max(0.0, x1), min(float(image_w), x2)))
    y1, y2 = sorted((max(0.0, y1), min(float(image_h), y2)))
    width, height = x2 - x1, y2 - y1
    if width < 2 or height < 2:
        return None

    label = defect.get("review_label") or defect.get("defect_class")
    class_id = CLASS_INDEX.get(label)
    if class_id is None:
        return None

    return (
        f"{class_id} {(x1 + x2) / 2 / image_w:.6f} {(y1 + y2) / 2 / image_h:.6f} "
        f"{width / image_w:.6f} {height / image_h:.6f}"
    )


def export(output: Path, val_split: float, seed: int) -> dict:
    """Write a YOLO dataset from the dashboard's reviewed detections."""
    import cv2

    from app.core.config import get_settings

    settings = get_settings()
    snapshot_dir = settings.snapshot_dir

    records = asyncio.run(fetch_reviewed())
    if not records:
        print(
            "No reviewed detections found. Verify or reject some defects on the "
            "dashboard first - the loop needs human decisions to learn from.",
            file=sys.stderr,
        )
        return {"images": 0}

    # Group by snapshot: one image may carry several reviewed boxes.
    by_snapshot: dict[str, list[dict]] = {}
    for record in records:
        snapshot = record.get("snapshot_path")
        if snapshot:
            by_snapshot.setdefault(snapshot, []).append(record)

    if not by_snapshot:
        print("Reviewed detections exist but none have a stored snapshot image.",
              file=sys.stderr)
        return {"images": 0}

    for split in ("train", "val"):
        (output / "images" / split).mkdir(parents=True, exist_ok=True)
        (output / "labels" / split).mkdir(parents=True, exist_ok=True)

    items = sorted(by_snapshot.items())
    random.Random(seed).shuffle(items)
    val_count = int(len(items) * val_split)
    splits = {"val": items[:val_count], "train": items[val_count:]}

    stats = {"images": 0, "positives": 0, "hard_negatives": 0, "missing": 0,
             "by_class": {name: 0 for name in CLASS_NAMES}}

    for split, entries in splits.items():
        for snapshot, defects in entries:
            source = snapshot_dir / snapshot
            if not source.exists():
                stats["missing"] += 1
                continue
            image = cv2.imread(str(source))
            if image is None:
                stats["missing"] += 1
                continue
            height, width = image.shape[:2]

            rows: list[str] = []
            for defect in defects:
                if defect.get("status") == "Rejected":
                    continue  # contributes an empty label = hard negative
                if defect.get("status") not in POSITIVE_STATUSES:
                    continue
                row = to_yolo_row(defect, width, height)
                if row:
                    rows.append(row)
                    stats["by_class"][CLASS_NAMES[int(row.split()[0])]] += 1

            stem = Path(snapshot).stem
            shutil.copy2(source, output / "images" / split / f"{stem}.jpg")
            (output / "labels" / split / f"{stem}.txt").write_text(
                "\n".join(rows) + ("\n" if rows else "")
            )
            stats["images"] += 1
            if rows:
                stats["positives"] += len(rows)
            else:
                stats["hard_negatives"] += 1

    names_block = "\n".join(f"  {i}: {n}" for i, n in enumerate(CLASS_NAMES))
    (output / "data.yaml").write_text(
        "# CrackCatch feedback dataset - authority-reviewed detections.\n"
        f"# Exported: {datetime.now(timezone.utc).isoformat()}\n"
        f"path: {output.resolve()}\n"
        "train: images/train\n"
        "val: images/val\n"
        f"nc: {len(CLASS_NAMES)}\n"
        f"names:\n{names_block}\n"
    )
    (output / "export_stats.json").write_text(json.dumps(stats, indent=2))

    print(f"Exported {stats['images']} reviewed images -> {output}")
    print(f"  confirmed boxes : {stats['positives']} {stats['by_class']}")
    print(f"  hard negatives  : {stats['hard_negatives']} (rejected detections)")
    if stats["missing"]:
        print(f"  snapshots missing on disk: {stats['missing']}")
    return stats


# -------------------------------------------------------------------- retrain


def read_map50(metrics_path: Path) -> float | None:
    try:
        return float(json.loads(metrics_path.read_text())["overall"]["mAP50"])
    except Exception:
        return None


def retrain(dataset: Path, epochs: int, base: Path, imgsz: int, batch: int,
            promote_threshold: float) -> int:
    """Fine-tune from the incumbent checkpoint, then promote only if better."""
    python = sys.executable
    served = REPO_ROOT / "model" / "weights" / "crackcatch.pt"
    candidate_dir = REPO_ROOT / "model" / "weights" / "candidates"
    candidate_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    candidate = candidate_dir / f"crackcatch_{stamp}.pt"

    # --- baseline: how good is the model we are already serving? ---
    incumbent_map = None
    if served.exists():
        print("\n[1/4] Evaluating the incumbent checkpoint for comparison...")
        baseline_metrics = candidate_dir / f"incumbent_{stamp}.json"
        result = subprocess.run(
            [python, str(REPO_ROOT / "model" / "scripts" / "evaluate.py"),
             "--data", str(dataset), "--weights", str(served),
             "--output", str(baseline_metrics)],
            cwd=REPO_ROOT,
        )
        if result.returncode == 0:
            incumbent_map = read_map50(baseline_metrics)
            print(f"      incumbent mAP@0.5 = {incumbent_map}")
    else:
        print("\n[1/4] No incumbent checkpoint - this run establishes the baseline.")

    # --- train ---
    print("\n[2/4] Fine-tuning on the feedback dataset...")
    train_base = served if served.exists() else base
    result = subprocess.run(
        [python, str(REPO_ROOT / "model" / "scripts" / "train.py"),
         "--data", str(dataset), "--base", str(train_base),
         "--epochs", str(epochs), "--imgsz", str(imgsz), "--batch", str(batch),
         "--name", f"feedback_{stamp}", "--no-promote",
         "--output", str(candidate)],
        cwd=REPO_ROOT,
    )
    if result.returncode != 0:
        print("error: training failed; nothing was promoted.", file=sys.stderr)
        return result.returncode

    run_dir = REPO_ROOT / "model" / "runs" / f"feedback_{stamp}"
    best = run_dir / "weights" / "best.pt"
    if not best.exists():
        print(f"error: expected checkpoint {best} was not produced.", file=sys.stderr)
        return 1
    shutil.copy2(best, candidate)

    # --- evaluate the candidate ---
    print("\n[3/4] Evaluating the candidate checkpoint...")
    candidate_metrics = candidate_dir / f"candidate_{stamp}.json"
    result = subprocess.run(
        [python, str(REPO_ROOT / "model" / "scripts" / "evaluate.py"),
         "--data", str(dataset), "--weights", str(candidate),
         "--output", str(candidate_metrics)],
        cwd=REPO_ROOT,
    )
    if result.returncode != 0:
        print("error: evaluation failed; nothing was promoted.", file=sys.stderr)
        return result.returncode
    candidate_map = read_map50(candidate_metrics)

    # --- promote, but only on evidence ---
    print("\n[4/4] Promotion decision")
    print(f"      incumbent mAP@0.5 : {incumbent_map}")
    print(f"      candidate mAP@0.5 : {candidate_map}")

    if candidate_map is None:
        print("      -> NOT promoted (candidate metrics unavailable)")
        return 1
    if incumbent_map is not None and candidate_map < incumbent_map + promote_threshold:
        print(f"      -> NOT promoted: needs at least +{promote_threshold} mAP@0.5 "
              "over the incumbent to justify swapping a deployed model.")
        print(f"      Candidate retained at {candidate}")
        return 0

    shutil.copy2(candidate, served)
    print(f"      -> PROMOTED to {served}")
    print("      Restart the backend (or `docker compose restart backend`) to load it.")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)

    export_parser = sub.add_parser("export", help="Export reviewed detections")
    export_parser.add_argument("--output", default=str(REPO_ROOT / "data" / "feedback_yolo"))
    export_parser.add_argument("--val-split", type=float, default=0.2)
    export_parser.add_argument("--seed", type=int, default=42)

    retrain_parser = sub.add_parser("retrain", help="Retrain, evaluate and maybe promote")
    retrain_parser.add_argument("--dataset", required=True)
    retrain_parser.add_argument("--epochs", type=int, default=30)
    retrain_parser.add_argument("--imgsz", type=int, default=640)
    retrain_parser.add_argument("--batch", type=int, default=16)
    retrain_parser.add_argument("--base", default=str(REPO_ROOT / "model" / "weights" / "yolov8n.pt"))
    retrain_parser.add_argument("--promote-threshold", type=float, default=0.005,
                                help="Minimum mAP@0.5 gain required to promote")

    all_parser = sub.add_parser("all", help="export + retrain in one go")
    all_parser.add_argument("--output", default=str(REPO_ROOT / "data" / "feedback_yolo"))
    all_parser.add_argument("--epochs", type=int, default=30)
    all_parser.add_argument("--imgsz", type=int, default=640)
    all_parser.add_argument("--batch", type=int, default=16)
    all_parser.add_argument("--val-split", type=float, default=0.2)
    all_parser.add_argument("--seed", type=int, default=42)
    all_parser.add_argument("--base", default=str(REPO_ROOT / "model" / "weights" / "yolov8n.pt"))
    all_parser.add_argument("--promote-threshold", type=float, default=0.005)

    args = parser.parse_args()

    if args.command == "export":
        stats = export(Path(args.output).resolve(), args.val_split, args.seed)
        return 0 if stats["images"] else 1

    if args.command == "retrain":
        return retrain(Path(args.dataset).resolve(), args.epochs, Path(args.base),
                       args.imgsz, args.batch, args.promote_threshold)

    output = Path(args.output).resolve()
    stats = export(output, args.val_split, args.seed)
    if not stats["images"]:
        return 1
    return retrain(output / "data.yaml", args.epochs, Path(args.base),
                   args.imgsz, args.batch, args.promote_threshold)


if __name__ == "__main__":
    raise SystemExit(main())
