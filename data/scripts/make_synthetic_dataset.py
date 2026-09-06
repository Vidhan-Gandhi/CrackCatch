#!/usr/bin/env python3
"""Generate a labelled synthetic road-damage dataset in YOLO format.

PURPOSE AND HONEST FRAMING
--------------------------
RDD2022 is the project's real dataset and ``prepare_rdd2022.py`` converts it.
But RDD2022 is a multi-gigabyte download, so a fresh clone cannot train
anything until the user has fetched it. This script renders a synthetic
dataset - procedural asphalt, perspective-projected potholes and cracks, with
exact ground-truth boxes - so that:

  * ``model/scripts/train.py`` can be executed and verified end to end;
  * the demo has genuine two-class YOLOv8 weights rather than a CV fallback;
  * the evaluation harness has a val split to report mAP on.

A model trained only on this data will NOT generalise to real Indian roads.
Synthetic asphalt has none of the confounders that make road-damage detection
hard: wet patches, tar repairs, shadows from overhead cables, painted
markings, manhole covers, debris. Any metric computed on this dataset measures
whether the training pipeline works, not whether the detector is fit for
deployment. Real numbers require RDD2022 - see docs/MODEL_CARD.md.

Usage:
    python data/scripts/make_synthetic_dataset.py \
        --output data/synthetic_yolo --train 400 --val 100
"""

from __future__ import annotations

import argparse
import math
import random
from pathlib import Path

import cv2
import numpy as np

CLASS_NAMES = ["pothole", "crack"]
POTHOLE, CRACK = 0, 1


def asphalt(width: int, height: int, rng: np.random.Generator) -> np.ndarray:
    """Procedural tarmac: fine grain plus low-frequency mottling."""
    fine = rng.normal(96, 14, (height, width)).astype(np.float32)
    coarse = cv2.resize(
        rng.normal(0, 20, (max(2, height // 14), max(2, width // 14))).astype(np.float32),
        (width, height),
        interpolation=cv2.INTER_CUBIC,
    )
    grey = np.clip(fine + coarse, 15, 215).astype(np.uint8)
    return cv2.cvtColor(grey, cv2.COLOR_GRAY2BGR)


def apply_lighting(image: np.ndarray, rng: random.Random) -> np.ndarray:
    """Simulate the sun/shadow split that CLAHE in stage 2 has to survive."""
    height, width = image.shape[:2]
    xx = np.linspace(0, 1, width, dtype=np.float32)[None, :]
    yy = np.linspace(0, 1, height, dtype=np.float32)[:, None]
    phase = rng.uniform(0, math.pi * 2)
    gain = 0.62 + 0.55 * (0.5 + 0.5 * np.sin(xx * rng.uniform(1.5, 4.0) * math.pi + phase))
    gain = gain * (0.85 + 0.3 * yy)  # slightly brighter close to the camera
    return np.clip(image.astype(np.float32) * gain[..., None], 0, 255).astype(np.uint8)


def draw_pothole(
    image: np.ndarray, cx: int, cy: int, radius: int, rng: random.Random
) -> tuple[int, int, int, int]:
    """Draw an irregular dark depression; return its tight bounding box."""
    points = []
    lobes = rng.randint(7, 11)
    squash = rng.uniform(0.55, 0.85)          # foreshortening on the road plane
    for i in range(lobes):
        angle = 2 * math.pi * i / lobes
        r = radius * rng.uniform(0.7, 1.15)
        points.append((cx + r * math.cos(angle), cy + r * squash * r / max(r, 1) * math.sin(angle) * 1.0))
    polygon = np.array(
        [(int(px), int(cy + (py - cy) * squash)) for px, py in points], dtype=np.int32
    )

    overlay = image.copy()
    cv2.fillPoly(overlay, [polygon], (rng.randint(14, 34),) * 3)
    alpha = rng.uniform(0.78, 0.95)
    cv2.addWeighted(overlay, alpha, image, 1 - alpha, 0, image)

    # A broken, slightly brighter rim where light catches the torn edge.
    cv2.polylines(
        image, [polygon], True, (rng.randint(130, 175),) * 3,
        max(1, radius // 8), cv2.LINE_AA,
    )
    x, y, w, h = cv2.boundingRect(polygon)
    return x, y, x + w, y + h


def draw_crack(
    image: np.ndarray, x0: int, y0: int, length: int, angle_deg: float, rng: random.Random
) -> tuple[int, int, int, int]:
    """Draw a jagged linear crack; return its tight bounding box."""
    angle = math.radians(angle_deg)
    points = [(x0, y0)]
    step = max(6, length // rng.randint(6, 12))
    travelled = 0
    x, y = float(x0), float(y0)
    while travelled < length:
        wobble = math.radians(rng.uniform(-22, 22))
        x += step * math.cos(angle + wobble)
        y += step * math.sin(angle + wobble) * 0.45  # flattened by perspective
        points.append((int(x), int(y)))
        travelled += step

    polyline = np.array(points, dtype=np.int32)
    thickness = max(2, int(length * rng.uniform(0.012, 0.035)))
    overlay = image.copy()
    cv2.polylines(overlay, [polyline], False, (rng.randint(10, 30),) * 3, thickness, cv2.LINE_AA)
    alpha = rng.uniform(0.75, 0.95)
    cv2.addWeighted(overlay, alpha, image, 1 - alpha, 0, image)

    # Hairline branches, which is what distinguishes a crack from a scratch.
    for _ in range(rng.randint(0, 2)):
        idx = rng.randrange(len(points))
        bx, by = points[idx]
        blen = int(length * rng.uniform(0.15, 0.35))
        bang = angle + math.radians(rng.choice([-1, 1]) * rng.uniform(35, 75))
        cv2.line(
            image, (bx, by),
            (int(bx + blen * math.cos(bang)), int(by + blen * math.sin(bang) * 0.45)),
            (rng.randint(20, 45),) * 3, max(1, thickness // 2), cv2.LINE_AA,
        )

    x1, y1, w, h = cv2.boundingRect(polyline)
    pad = thickness
    return x1 - pad, y1 - pad, x1 + w + pad, y1 + h + pad


def clamp_box(box, width, height):
    x1, y1, x2, y2 = box
    x1, y1 = max(0, min(x1, width - 1)), max(0, min(y1, height - 1))
    x2, y2 = max(0, min(x2, width - 1)), max(0, min(y2, height - 1))
    return x1, y1, x2, y2


def render(width: int, height: int, seed: int) -> tuple[np.ndarray, list[tuple[int, tuple]]]:
    """Render one image plus its ground-truth boxes."""
    rng = random.Random(seed)
    nrng = np.random.default_rng(seed)

    image = asphalt(width, height, nrng)

    # Optional lane marking - a bright line the model must learn to ignore.
    if rng.random() < 0.55:
        lane_x = rng.randint(int(width * 0.05), int(width * 0.95))
        cv2.line(image, (lane_x, 0), (lane_x + rng.randint(-40, 40), height),
                 (198, 198, 192), rng.randint(4, 10), cv2.LINE_AA)

    annotations: list[tuple[int, tuple]] = []
    placed: list[tuple[int, int, int, int]] = []

    def overlaps(box) -> bool:
        x1, y1, x2, y2 = box
        for px1, py1, px2, py2 in placed:
            if not (x2 < px1 or x1 > px2 or y2 < py1 or y1 > py2):
                return True
        return False

    # A defect that is drawn but not annotated would teach the model that a
    # real pothole is background, so a rejected placement is rolled back off
    # the canvas rather than merely skipped in the label file.
    def try_place(class_id: int, paint, validate=lambda box: True) -> None:
        snapshot = image.copy()
        box = clamp_box(paint(), width, height)
        if overlaps(box) or not validate(box):
            image[:] = snapshot        # undo the drawing
            return
        placed.append(box)
        annotations.append((class_id, box))

    for _ in range(rng.randint(1, 3)):
        radius = rng.randint(int(min(width, height) * 0.05), int(min(width, height) * 0.17))
        cx = rng.randint(radius + 5, width - radius - 5)
        cy = rng.randint(int(height * 0.35) + radius, height - radius - 5)
        try_place(POTHOLE, lambda: draw_pothole(image, cx, cy, radius, rng))

    for _ in range(rng.randint(1, 3)):
        length = rng.randint(int(width * 0.16), int(width * 0.45))
        x0 = rng.randint(5, max(6, width - length - 5))
        y0 = rng.randint(int(height * 0.35), height - 20)
        angle = rng.uniform(-30, 30)
        try_place(
            CRACK,
            lambda: draw_crack(image, x0, y0, length, angle, rng),
            validate=lambda box: (box[2] - box[0]) >= 12 and (box[3] - box[1]) >= 4,
        )

    image = apply_lighting(image, rng)
    image = cv2.GaussianBlur(image, (3, 3), 0)
    if rng.random() < 0.3:  # occasional motion blur, as in real dashcam frames
        k = rng.choice([5, 7])
        kernel = np.zeros((k, k), np.float32)
        kernel[k // 2, :] = 1.0 / k
        image = cv2.filter2D(image, -1, kernel)

    return image, annotations


def to_yolo(box, width, height) -> str | None:
    x1, y1, x2, y2 = box
    bw, bh = x2 - x1, y2 - y1
    if bw < 4 or bh < 3:
        return None
    return (
        f"{(x1 + x2) / 2 / width:.6f} {(y1 + y2) / 2 / height:.6f} "
        f"{bw / width:.6f} {bh / height:.6f}"
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--output", default="data/synthetic_yolo")
    parser.add_argument("--train", type=int, default=400)
    parser.add_argument("--val", type=int, default=100)
    parser.add_argument("--width", type=int, default=640)
    parser.add_argument("--height", type=int, default=640)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    output = Path(args.output).resolve()
    counts = {"train": args.train, "val": args.val}
    totals = {name: 0 for name in CLASS_NAMES}

    for split, count in counts.items():
        image_dir = output / "images" / split
        label_dir = output / "labels" / split
        image_dir.mkdir(parents=True, exist_ok=True)
        label_dir.mkdir(parents=True, exist_ok=True)

        # Disjoint seed ranges keep train and val independent for a given seed.
        offset = args.seed * 100_000 + (0 if split == "train" else 50_000)
        for i in range(count):
            image, annotations = render(args.width, args.height, offset + i)
            rows = []
            for class_id, box in annotations:
                yolo = to_yolo(box, args.width, args.height)
                if yolo:
                    rows.append(f"{class_id} {yolo}")
                    totals[CLASS_NAMES[class_id]] += 1
            if not rows:
                continue
            cv2.imwrite(str(image_dir / f"{split}_{i:05d}.jpg"), image,
                        [int(cv2.IMWRITE_JPEG_QUALITY), 92])
            (label_dir / f"{split}_{i:05d}.txt").write_text("\n".join(rows) + "\n")

    names_block = "\n".join(f"  {i}: {n}" for i, n in enumerate(CLASS_NAMES))
    (output / "data.yaml").write_text(
        "# SYNTHETIC dataset - pipeline verification only, NOT a benchmark.\n"
        "# Generated by data/scripts/make_synthetic_dataset.py\n"
        f"# seed: {args.seed}\n"
        f"path: {output}\n"
        "train: images/train\n"
        "val: images/val\n"
        f"nc: {len(CLASS_NAMES)}\n"
        f"names:\n{names_block}\n"
    )

    print(f"Wrote synthetic dataset to {output}")
    print(f"  train images: {len(list((output / 'images' / 'train').glob('*.jpg')))}")
    print(f"  val images:   {len(list((output / 'images' / 'val').glob('*.jpg')))}")
    print(f"  boxes: {totals}")
    print("\nREMINDER: synthetic data verifies the training pipeline; it does not")
    print("measure real-world accuracy. Use RDD2022 for reportable metrics.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
