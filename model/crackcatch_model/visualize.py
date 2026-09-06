"""Annotated snapshot rendering.

Produces the bounding-box overlay images the dashboard shows on a defect's
detail page. Colours are keyed to severity and match the frontend's map-pin
palette so a snapshot and its map marker read as the same thing.
"""

from __future__ import annotations

from typing import Iterable, Sequence

import cv2
import numpy as np

from crackcatch_model.types import DefectRecord, Severity

#: BGR, matching the frontend severity palette in frontend/src/lib/theme.js.
SEVERITY_COLOURS: dict[Severity, tuple[int, int, int]] = {
    Severity.MINOR: (86, 180, 233),     # blue
    Severity.MODERATE: (0, 165, 245),   # amber
    Severity.SEVERE: (60, 60, 220),     # red
}

_FONT = cv2.FONT_HERSHEY_SIMPLEX


def _put_label(
    image: np.ndarray,
    text: str,
    origin: tuple[int, int],
    colour: tuple[int, int, int],
    scale: float = 0.5,
    thickness: int = 1,
) -> None:
    """Draw text on a filled background chip so it stays readable."""
    x, y = origin
    (text_w, text_h), baseline = cv2.getTextSize(text, _FONT, scale, thickness)
    top = max(0, y - text_h - baseline - 4)
    cv2.rectangle(
        image, (x, top), (x + text_w + 8, top + text_h + baseline + 4), colour, -1
    )
    cv2.putText(
        image,
        text,
        (x + 4, top + text_h + 2),
        _FONT,
        scale,
        (255, 255, 255),
        thickness,
        cv2.LINE_AA,
    )


def annotate(
    frame: np.ndarray,
    records: Sequence[DefectRecord],
    show_priority: bool = True,
    show_size: bool = True,
) -> np.ndarray:
    """Return a copy of ``frame`` with each record's box and label drawn."""
    canvas = frame.copy()
    if canvas.ndim == 2:
        canvas = cv2.cvtColor(canvas, cv2.COLOR_GRAY2BGR)

    height, width = canvas.shape[:2]
    thickness = max(2, int(round(min(width, height) / 400)))
    scale = max(0.45, min(width, height) / 1400.0)

    for record in records:
        colour = SEVERITY_COLOURS.get(record.severity, (200, 200, 200))
        x1, y1, x2, y2 = (int(round(v)) for v in record.bbox.as_xyxy())
        cv2.rectangle(canvas, (x1, y1), (x2, y2), colour, thickness)

        parts = [
            record.defect_class.value,
            record.severity.value,
            f"{record.confidence:.0%}",
        ]
        if show_size and record.size and record.size.area_m2 is not None:
            parts.append(f"{record.size.area_m2:.2f}m2")
        if show_priority:
            parts.append(f"P{record.priority_score:.0f}")
        _put_label(canvas, " | ".join(parts), (x1, y1), colour, scale, thickness // 2 or 1)

    return canvas


def draw_horizon(
    frame: np.ndarray, horizon_v: float, colour: tuple[int, int, int] = (120, 120, 120)
) -> np.ndarray:
    """Debug aid: draw the calibration's horizon line on a frame."""
    canvas = frame.copy()
    y = int(round(horizon_v))
    if 0 <= y < canvas.shape[0]:
        cv2.line(canvas, (0, y), (canvas.shape[1], y), colour, 1, cv2.LINE_AA)
        cv2.putText(
            canvas, "horizon", (8, max(14, y - 6)), _FONT, 0.5, colour, 1, cv2.LINE_AA
        )
    return canvas


def severity_legend(width: int = 320, height: int = 110) -> np.ndarray:
    """A small standalone legend image, used in the exported PDF report."""
    canvas = np.full((height, width, 3), 255, dtype=np.uint8)
    cv2.putText(canvas, "Severity", (10, 22), _FONT, 0.6, (30, 30, 30), 1, cv2.LINE_AA)
    for row, severity in enumerate(Severity):
        y = 44 + row * 22
        cv2.rectangle(canvas, (12, y - 11), (32, y + 3), SEVERITY_COLOURS[severity], -1)
        cv2.putText(
            canvas, severity.value, (42, y), _FONT, 0.5, (30, 30, 30), 1, cv2.LINE_AA
        )
    return canvas


def stack_before_after(
    before: np.ndarray, after: np.ndarray, labels: tuple[str, str] = ("Before", "After")
) -> np.ndarray:
    """Side-by-side composite for the repair before/after verification view."""
    target_h = min(before.shape[0], after.shape[0], 720)

    def _fit(image: np.ndarray) -> np.ndarray:
        scale = target_h / image.shape[0]
        return cv2.resize(image, (int(image.shape[1] * scale), target_h))

    left, right = _fit(before), _fit(after)
    gap = np.full((target_h, 8, 3), 255, dtype=np.uint8)
    composite = np.hstack([left, gap, right])
    _put_label(composite, labels[0], (10, 30), (60, 60, 60), 0.7, 2)
    _put_label(composite, labels[1], (left.shape[1] + 18, 30), (60, 60, 60), 0.7, 2)
    return composite
