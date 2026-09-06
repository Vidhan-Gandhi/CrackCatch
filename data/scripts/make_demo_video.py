#!/usr/bin/env python3
"""Generate a synthetic dashcam clip for pipeline smoke tests and demos.

WHY THIS EXISTS: a fresh clone of this repository has no road footage in it
(video is not something to commit to git, and redistributing RDD2022 imagery
would violate its terms). Without a sample input, nothing in the repo can be
run. This script synthesises a short clip - a perspective road surface with
procedurally placed dark blobs and linear cracks approaching the camera - so
that `make demo` works immediately after `git clone`.

HONESTY NOTE: this footage is SYNTHETIC. It is a plumbing test for stages 1-7,
not an evaluation set. No accuracy number in this project is computed on it -
see docs/MODEL_CARD.md for how real metrics are obtained from RDD2022. Records
produced from it are tagged `source_ref="demo_drive.mp4"` so they are trivially
distinguishable from real detections.

Usage:
    python data/scripts/make_demo_video.py --output data/samples/demo_drive.mp4
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path

import cv2
import numpy as np


def asphalt_texture(width: int, height: int, rng: np.random.Generator) -> np.ndarray:
    """Grainy grey base that looks enough like tarmac to exercise the filters."""
    base = rng.normal(loc=95, scale=13, size=(height, width)).astype(np.float32)
    coarse = cv2.resize(
        rng.normal(loc=0, scale=18, size=(height // 12 + 1, width // 12 + 1)).astype(
            np.float32
        ),
        (width, height),
        interpolation=cv2.INTER_CUBIC,
    )
    grey = np.clip(base + coarse, 20, 210).astype(np.uint8)
    return cv2.cvtColor(grey, cv2.COLOR_GRAY2BGR)


def draw_scene(
    width: int,
    height: int,
    horizon: int,
    rng: np.random.Generator,
    sun_phase: float,
) -> np.ndarray:
    """Sky + road plane + lane markings, with a moving lighting gradient."""
    frame = np.zeros((height, width, 3), dtype=np.uint8)

    # Sky: a flat hazy band. The detector's road ROI excludes it anyway.
    frame[:horizon] = (168, 152, 132)

    road = asphalt_texture(width, height - horizon, rng)

    # Lighting gradient: simulates driving past buildings casting shade, which
    # is exactly the condition CLAHE in stage 2 exists to handle.
    yy, xx = np.mgrid[0 : road.shape[0], 0 : road.shape[1]].astype(np.float32)
    shade = 0.72 + 0.42 * (0.5 + 0.5 * np.sin(xx / width * 3.0 * math.pi + sun_phase))
    road = np.clip(road.astype(np.float32) * shade[..., None], 0, 255).astype(np.uint8)

    frame[horizon:] = road

    # Perspective lane edges converging on the vanishing point.
    vanish = (width // 2, horizon)
    for offset in (-1, 1):
        cv2.line(
            frame,
            vanish,
            (int(width // 2 + offset * width * 0.95), height),
            (205, 205, 200),
            3,
            cv2.LINE_AA,
        )
    return frame


def project(
    u_norm: float, depth: float, width: int, height: int, horizon: int
) -> tuple[int, int, float]:
    """Place a point at lateral offset ``u_norm`` and ``depth`` in [0, 1].

    ``depth`` 0 = at the camera, 1 = at the horizon. Returns ``(x, y, scale)``
    where ``scale`` shrinks with distance, giving correct-looking perspective.
    """
    d = float(np.clip(depth, 0.02, 1.0))
    y = int(horizon + (height - horizon) * (1.0 - d) ** 1.6)
    shrink = (1.0 - d) ** 1.6
    x = int(width / 2 + u_norm * width * 0.42 * (0.12 + 0.88 * shrink))
    return x, y, max(0.04, shrink)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", default="data/samples/demo_drive.mp4")
    parser.add_argument("--seconds", type=float, default=20.0)
    parser.add_argument("--fps", type=int, default=15)
    parser.add_argument("--width", type=int, default=1280)
    parser.add_argument("--height", type=int, default=720)
    parser.add_argument("--potholes", type=int, default=9)
    parser.add_argument("--cracks", type=int, default=7)
    parser.add_argument("--seed", type=int, default=7)
    args = parser.parse_args()

    rng = np.random.default_rng(args.seed)
    width, height = args.width, args.height
    horizon = int(height * 0.42)
    total_frames = int(args.seconds * args.fps)

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)

    writer = cv2.VideoWriter(
        str(output), cv2.VideoWriter_fourcc(*"mp4v"), args.fps, (width, height)
    )
    if not writer.isOpened():
        raise SystemExit(
            f"Could not open a video writer for {output}. Your OpenCV build may "
            "lack mp4v support - try --output with a .avi extension."
        )

    # Each defect enters at the horizon and travels toward the camera.
    defects = []
    for i in range(args.potholes):
        defects.append(
            {
                "kind": "pothole",
                "u": float(rng.uniform(-0.75, 0.75)),
                "enter": (i + 0.5) * total_frames / (args.potholes + 1),
                "size": float(rng.uniform(0.055, 0.14)),
                "darkness": float(rng.uniform(0.30, 0.60)),
            }
        )
    for i in range(args.cracks):
        defects.append(
            {
                "kind": "crack",
                "u": float(rng.uniform(-0.8, 0.8)),
                "enter": (i + 0.5) * total_frames / (args.cracks + 1) + 6,
                "size": float(rng.uniform(0.14, 0.30)),
                # Cracks are drawn with strong contrast: a faint hairline is
                # invisible once the clip is JPEG-compressed by the video
                # encoder, and this asset exists to exercise BOTH detection
                # classes end-to-end.
                "darkness": float(rng.uniform(0.70, 0.92)),
                "angle": float(rng.uniform(-38, 38)),
            }
        )

    #: Frames a defect takes to travel from horizon to camera.
    travel = args.fps * 3.2

    for index in range(total_frames):
        frame = draw_scene(width, height, horizon, rng, index * 0.06)

        for defect in defects:
            age = index - defect["enter"]
            if age < 0 or age > travel:
                continue
            depth = 1.0 - age / travel          # 1 at horizon -> 0 at camera
            x, y, scale = project(defect["u"], depth, width, height, horizon)
            if y >= height + 60 or scale <= 0.05:
                continue

            overlay = frame.copy()
            if defect["kind"] == "pothole":
                axis_a = max(4, int(defect["size"] * width * scale))
                axis_b = max(3, int(axis_a * rng.uniform(0.62, 0.92)))
                cv2.ellipse(
                    overlay, (x, y), (axis_a, axis_b), 0, 0, 360, (18, 20, 24), -1
                )
                # Bright rim: light catching the broken asphalt edge.
                cv2.ellipse(
                    overlay,
                    (x, y),
                    (axis_a, axis_b),
                    0,
                    0,
                    360,
                    (150, 150, 148),
                    max(1, axis_a // 9),
                )
            else:
                length = max(8, int(defect["size"] * width * scale * 2.4))
                theta = math.radians(defect["angle"])
                dx, dy = int(length * math.cos(theta)), int(length * math.sin(theta) / 3)
                cv2.line(
                    overlay,
                    (x - dx, y - dy),
                    (x + dx, y + dy),
                    (14, 15, 18),
                    max(2, int(11 * scale) + 2),
                    cv2.LINE_AA,
                )

            cv2.addWeighted(overlay, defect["darkness"], frame, 1 - defect["darkness"], 0, frame)

        frame = cv2.GaussianBlur(frame, (3, 3), 0)
        cv2.putText(
            frame,
            "SYNTHETIC DEMO FOOTAGE - not a real road survey",
            (14, height - 16),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (235, 235, 235),
            1,
            cv2.LINE_AA,
        )
        writer.write(frame)

    writer.release()
    print(f"Wrote {output} ({total_frames} frames @ {args.fps} fps, {args.seconds}s)")
    print("Reminder: synthetic footage - for pipeline testing only.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
