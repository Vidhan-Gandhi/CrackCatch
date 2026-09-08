#!/usr/bin/env python3
"""Convert RDD2022 (Pascal VOC XML) into YOLO format with a seeded split.

RDD2022 is the primary dataset named in the project scope. It ships as
per-country folders, each with ``images/`` and ``annotations/xmls/`` in Pascal
VOC form. This script maps its damage-type labels onto CrackCatch's classes,
writes YOLO ``.txt`` labels, and produces a reproducible train/val split.

Label mapping (documented in docs/MODEL_CARD.md):
    D00 longitudinal crack  -> crack
    D10 transverse crack    -> crack
    D20 alligator crack     -> crack
    D40 pothole             -> pothole
    D43/D44 (line/crosswalk blur) -> ignored by default; these are marking
        defects, not structural damage, and including them hurts precision on
        the two classes we actually act on.

Reproducibility: the split is driven by a seeded RNG (``--seed``, default 42)
and is deterministic given the same file listing, satisfying the project's
"seed-controlled train/val split" requirement.

The dataset is NOT redistributed with this repository. Download it from
https://github.com/sekilab/RoadDamageDetector and point ``--source`` at it.

Usage:
    python data/scripts/prepare_rdd2022.py \
        --source ~/datasets/RDD2022 --output data/rdd2022_yolo \
        --countries India Japan --val-split 0.2
"""

from __future__ import annotations

import argparse
import random
import shutil
import sys
import xml.etree.ElementTree as ET
from collections import Counter
from pathlib import Path

#: RDD2022 damage code -> CrackCatch class index. Order defines the YOLO ids.
CLASS_NAMES = ["pothole", "crack"]
LABEL_MAP = {
    "D00": "crack",
    "D10": "crack",
    "D20": "crack",
    "D40": "pothole",
}
IGNORED_LABELS = {"D43", "D44", "D50", "D0w0"}


def parse_voc(xml_path: Path) -> tuple[int, int, list[tuple[str, float, float, float, float]]]:
    """Return ``(width, height, [(label, x1, y1, x2, y2), ...])``."""
    root = ET.parse(str(xml_path)).getroot()
    size = root.find("size")
    if size is None:
        raise ValueError(f"{xml_path}: no <size> element")
    width = int(float(size.findtext("width", "0")))
    height = int(float(size.findtext("height", "0")))

    boxes = []
    for obj in root.findall("object"):
        label = (obj.findtext("name") or "").strip()
        box = obj.find("bndbox")
        if box is None:
            continue
        try:
            x1 = float(box.findtext("xmin", "0"))
            y1 = float(box.findtext("ymin", "0"))
            x2 = float(box.findtext("xmax", "0"))
            y2 = float(box.findtext("ymax", "0"))
        except ValueError:
            continue
        boxes.append((label, x1, y1, x2, y2))
    return width, height, boxes


def to_yolo_line(class_id: int, x1: float, y1: float, x2: float, y2: float,
                 width: int, height: int) -> str | None:
    """Convert an absolute VOC box to a normalised YOLO row."""
    if width <= 0 or height <= 0:
        return None
    x1, x2 = sorted((max(0.0, x1), min(float(width), x2)))
    y1, y2 = sorted((max(0.0, y1), min(float(height), y2)))
    box_w, box_h = x2 - x1, y2 - y1
    if box_w <= 1 or box_h <= 1:
        return None  # degenerate after clipping
    cx = (x1 + x2) / 2.0 / width
    cy = (y1 + y2) / 2.0 / height
    return f"{class_id} {cx:.6f} {cy:.6f} {box_w / width:.6f} {box_h / height:.6f}"


def find_country_dirs(source: Path, countries: list[str] | None) -> list[Path]:
    """Locate per-country folders, tolerating the several RDD2022 layouts."""
    candidates: list[Path] = []
    for child in sorted(source.iterdir()):
        if not child.is_dir():
            continue
        if countries and child.name not in countries:
            continue
        # Layouts seen in the wild: <country>/train/{images,annotations} or
        # <country>/{images,annotations}.
        for base in (child / "train", child):
            if (base / "images").is_dir():
                candidates.append(base)
                break
    return candidates


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--source", required=True, help="Root of the extracted RDD2022")
    parser.add_argument("--output", default="data/rdd2022_yolo")
    parser.add_argument("--countries", nargs="*", default=None,
                        help="Subset to convert, e.g. India Japan. Default: all found.")
    parser.add_argument("--val-split", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--copy", action="store_true",
                        help="Copy images instead of symlinking (uses more disk).")
    parser.add_argument("--include-negatives", action="store_true",
                        help="Keep images with no usable boxes as hard negatives.")
    args = parser.parse_args()

    source = Path(args.source).expanduser().resolve()
    if not source.is_dir():
        print(f"error: --source {source} is not a directory", file=sys.stderr)
        return 2

    country_dirs = find_country_dirs(source, args.countries)
    if not country_dirs:
        print(
            f"error: no country folders with an images/ subdirectory under {source}.\n"
            "Expected e.g. <source>/India/train/images/.",
            file=sys.stderr,
        )
        return 2

    output = Path(args.output).resolve()
    for split in ("train", "val"):
        (output / "images" / split).mkdir(parents=True, exist_ok=True)
        (output / "labels" / split).mkdir(parents=True, exist_ok=True)

    rng = random.Random(args.seed)
    class_index = {name: i for i, name in enumerate(CLASS_NAMES)}

    records: list[tuple[Path, list[str], str]] = []
    label_counts: Counter[str] = Counter()
    skipped_labels: Counter[str] = Counter()
    images_without_boxes = 0

    for country_dir in country_dirs:
        image_dir = country_dir / "images"
        xml_dir = country_dir / "annotations" / "xmls"
        if not xml_dir.is_dir():
            xml_dir = country_dir / "annotations"
        country = country_dir.parent.name if country_dir.name == "train" else country_dir.name

        for image_path in sorted(image_dir.glob("*.jpg")):
            xml_path = xml_dir / f"{image_path.stem}.xml"
            if not xml_path.exists():
                continue
            try:
                width, height, boxes = parse_voc(xml_path)
            except (ET.ParseError, ValueError):
                continue

            lines: list[str] = []
            for label, x1, y1, x2, y2 in boxes:
                mapped = LABEL_MAP.get(label)
                if mapped is None:
                    if label not in IGNORED_LABELS:
                        skipped_labels[label] += 1
                    continue
                line = to_yolo_line(class_index[mapped], x1, y1, x2, y2, width, height)
                if line:
                    lines.append(line)
                    label_counts[mapped] += 1

            if not lines:
                images_without_boxes += 1
                if not args.include_negatives:
                    continue
            records.append((image_path, lines, country))

    if not records:
        print("error: no usable image/annotation pairs were produced.", file=sys.stderr)
        return 1

    # Deterministic split: sort first so filesystem order cannot leak in.
    records.sort(key=lambda r: str(r[0]))
    rng.shuffle(records)
    val_count = int(len(records) * args.val_split)
    splits = {"val": records[:val_count], "train": records[val_count:]}

    for split, items in splits.items():
        for image_path, lines, country in items:
            stem = f"{country}_{image_path.stem}"
            destination = output / "images" / split / f"{stem}.jpg"
            if args.copy:
                shutil.copy2(image_path, destination)
            else:
                if destination.exists() or destination.is_symlink():
                    destination.unlink()
                destination.symlink_to(image_path)
            (output / "labels" / split / f"{stem}.txt").write_text(
                "\n".join(lines) + ("\n" if lines else "")
            )

    names_block = "\n".join(f"  {i}: {n}" for i, n in enumerate(CLASS_NAMES))
    (output / "data.yaml").write_text(
        f"# CrackCatch dataset generated by data/scripts/prepare_rdd2022.py\n"
        f"# Source: {source}\n"
        f"# Seed: {args.seed}  Val split: {args.val_split}\n"
        f"path: {output}\n"
        f"train: images/train\n"
        f"val: images/val\n"
        f"nc: {len(CLASS_NAMES)}\n"
        f"names:\n{names_block}\n"
    )

    print(f"Converted {len(records)} images from {len(country_dirs)} country folder(s)")
    print(f"  train: {len(splits['train'])}   val: {len(splits['val'])}")
    print(f"  boxes: {dict(label_counts)}")
    if images_without_boxes:
        action = "kept as negatives" if args.include_negatives else "skipped"
        print(f"  images with no usable boxes: {images_without_boxes} ({action})")
    if skipped_labels:
        print(f"  unmapped labels ignored: {dict(skipped_labels)}")
    print(f"\nWrote {output / 'data.yaml'}")
    print(f"Next: python model/scripts/train.py --data {output / 'data.yaml'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
