"""Exportable defect reports - CSV and PDF.

The scope calls for "an exportable report (PDF/CSV) of defects for a date
range - useful for demo and for a real municipal workflow". The PDF is laid
out as a ward work-order: a summary block, a severity/priority breakdown, and
a table ordered by repair priority so the top of page one is what a crew
should fix first.
"""

from __future__ import annotations

import csv
import io
import logging
from datetime import datetime, timezone
from typing import Any, Sequence

logger = logging.getLogger(__name__)

CSV_COLUMNS = [
    "id",
    "defect_class",
    "severity",
    "severity_score",
    "priority_score",
    "priority_band",
    "status",
    "confidence",
    "latitude",
    "longitude",
    "gps_source",
    "area_m2",
    "size_reliable",
    "road_type",
    "detected_at",
    "repaired_at",
    "source_type",
    "source_ref",
]


def _row(defect: dict[str, Any]) -> dict[str, Any]:
    location = defect.get("location") or {}
    size = defect.get("size") or {}
    return {
        "id": defect.get("_id", ""),
        "defect_class": defect.get("defect_class", ""),
        "severity": defect.get("severity", ""),
        "severity_score": round(float(defect.get("severity_score", 0.0)), 3),
        "priority_score": round(float(defect.get("priority_score", 0.0)), 2),
        "priority_band": defect.get("priority_band", ""),
        "status": defect.get("status", ""),
        "confidence": round(float(defect.get("confidence", 0.0)), 3),
        "latitude": location.get("latitude", ""),
        "longitude": location.get("longitude", ""),
        "gps_source": location.get("source", ""),
        "area_m2": size.get("area_m2", ""),
        "size_reliable": size.get("reliable", ""),
        "road_type": defect.get("road_type", ""),
        "detected_at": _fmt(defect.get("detected_at")),
        "repaired_at": _fmt(defect.get("repaired_at")),
        "source_type": defect.get("source_type", ""),
        "source_ref": defect.get("source_ref", ""),
    }


def _fmt(value: Any) -> str:
    if isinstance(value, datetime):
        return value.strftime("%Y-%m-%d %H:%M:%S")
    return str(value or "")


def build_csv(defects: Sequence[dict[str, Any]]) -> bytes:
    buffer = io.StringIO()
    writer = csv.DictWriter(buffer, fieldnames=CSV_COLUMNS, extrasaction="ignore")
    writer.writeheader()
    for defect in defects:
        writer.writerow(_row(defect))
    return buffer.getvalue().encode("utf-8")


def build_pdf(
    defects: Sequence[dict[str, Any]],
    title: str = "CrackCatch Road Damage Report",
    ward: str | None = None,
    date_from: datetime | None = None,
    date_to: datetime | None = None,
    max_rows: int = 300,
) -> bytes:
    """Render a municipal work-order PDF, ordered by repair priority."""
    from reportlab.lib import colors
    from reportlab.lib.enums import TA_LEFT
    from reportlab.lib.pagesizes import A4, landscape
    from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
    from reportlab.lib.units import mm
    from reportlab.platypus import (
        PageBreak,
        Paragraph,
        SimpleDocTemplate,
        Spacer,
        Table,
        TableStyle,
    )

    buffer = io.BytesIO()
    document = SimpleDocTemplate(
        buffer,
        pagesize=landscape(A4),
        leftMargin=12 * mm,
        rightMargin=12 * mm,
        topMargin=12 * mm,
        bottomMargin=12 * mm,
        title=title,
        author="CrackCatch",
    )

    styles = getSampleStyleSheet()
    small = ParagraphStyle(
        "small", parent=styles["Normal"], fontSize=8, leading=10, alignment=TA_LEFT
    )
    story: list[Any] = []

    story.append(Paragraph(title, styles["Title"]))
    subtitle_parts = []
    if ward:
        subtitle_parts.append(f"Ward: {ward}")
    if date_from or date_to:
        subtitle_parts.append(
            f"Period: {_fmt(date_from) or 'start'} to {_fmt(date_to) or 'now'}"
        )
    subtitle_parts.append(
        f"Generated: {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')}"
    )
    story.append(Paragraph(" &nbsp;|&nbsp; ".join(subtitle_parts), small))
    story.append(Spacer(1, 6 * mm))

    # --- summary ---
    total = len(defects)
    by_severity = {s: 0 for s in ("Minor", "Moderate", "Severe")}
    by_status: dict[str, int] = {}
    priorities: list[float] = []
    for defect in defects:
        by_severity[defect.get("severity", "Minor")] = (
            by_severity.get(defect.get("severity", "Minor"), 0) + 1
        )
        by_status[defect.get("status", "New")] = by_status.get(defect.get("status", "New"), 0) + 1
        priorities.append(float(defect.get("priority_score", 0.0)))

    summary = [
        ["Total defects", str(total)],
        ["Severe", str(by_severity.get("Severe", 0))],
        ["Moderate", str(by_severity.get("Moderate", 0))],
        ["Minor", str(by_severity.get("Minor", 0))],
        [
            "Mean repair priority",
            f"{sum(priorities) / len(priorities):.1f} / 100" if priorities else "n/a",
        ],
        [
            "Open (not repaired)",
            str(sum(v for k, v in by_status.items() if k != "Repaired")),
        ],
        ["Repaired", str(by_status.get("Repaired", 0))],
    ]
    summary_table = Table(summary, colWidths=[60 * mm, 40 * mm])
    summary_table.setStyle(
        TableStyle(
            [
                ("FONTSIZE", (0, 0), (-1, -1), 9),
                ("GRID", (0, 0), (-1, -1), 0.4, colors.HexColor("#cccccc")),
                ("BACKGROUND", (0, 0), (0, -1), colors.HexColor("#f2f4f7")),
                ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
                ("LEFTPADDING", (0, 0), (-1, -1), 5),
            ]
        )
    )
    story.append(Paragraph("<b>Summary</b>", styles["Heading3"]))
    story.append(summary_table)
    story.append(Spacer(1, 6 * mm))

    # --- detail table, highest repair priority first ---
    ordered = sorted(
        defects, key=lambda d: float(d.get("priority_score", 0.0)), reverse=True
    )
    truncated = len(ordered) > max_rows
    ordered = ordered[:max_rows]

    header = [
        "#", "Priority", "Class", "Severity", "Status",
        "Lat", "Lon", "Area m2", "Road", "Detected",
    ]
    rows: list[list[str]] = [header]
    for index, defect in enumerate(ordered, start=1):
        location = defect.get("location") or {}
        size = defect.get("size") or {}
        area = size.get("area_m2")
        # Mark estimates the calibration flagged as unreliable, so a crew
        # never treats a projection artefact as a measurement.
        area_text = "-" if area is None else (
            f"{area:.2f}" + ("" if size.get("reliable", True) else "*")
        )
        rows.append(
            [
                str(index),
                f"{float(defect.get('priority_score', 0)):.0f}",
                defect.get("defect_class", ""),
                defect.get("severity", ""),
                defect.get("status", ""),
                f"{location.get('latitude', 0):.5f}",
                f"{location.get('longitude', 0):.5f}",
                area_text,
                defect.get("road_type", ""),
                _fmt(defect.get("detected_at"))[:16],
            ]
        )

    table = Table(rows, repeatRows=1, colWidths=[
        10 * mm, 18 * mm, 22 * mm, 22 * mm, 22 * mm,
        26 * mm, 26 * mm, 22 * mm, 26 * mm, 34 * mm,
    ])
    style = [
        ("FONTSIZE", (0, 0), (-1, -1), 7.5),
        ("GRID", (0, 0), (-1, -1), 0.3, colors.HexColor("#d0d0d0")),
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#1f2937")),
        ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#f7f8fa")]),
    ]
    # Tint the severity cell so a Severe row is findable at a glance.
    severity_colours = {
        "Severe": colors.HexColor("#fde2e1"),
        "Moderate": colors.HexColor("#fdf0d5"),
        "Minor": colors.HexColor("#e3f0fb"),
    }
    for row_index, defect in enumerate(ordered, start=1):
        colour = severity_colours.get(defect.get("severity", ""))
        if colour is not None:
            style.append(("BACKGROUND", (3, row_index), (3, row_index), colour))
    table.setStyle(TableStyle(style))

    story.append(Paragraph("<b>Defects by repair priority</b>", styles["Heading3"]))
    story.append(table)

    if truncated:
        story.append(Spacer(1, 4 * mm))
        story.append(
            Paragraph(
                f"Showing the top {max_rows} of {total} defects by priority. "
                "Export CSV for the complete set.",
                small,
            )
        )

    story.append(PageBreak())
    story.append(Paragraph("Method and limitations", styles["Heading3"]))
    story.append(
        Paragraph(
            "Defects were detected automatically from road video and scored by a "
            "documented heuristic combining bounding-box size, aspect ratio and "
            "detector confidence. Size figures are ground-plane estimates from a "
            "single camera and describe surface extent only - <b>pothole depth is "
            "not measured</b>, and an entry marked * was flagged by the "
            "calibration as being beyond its reliable range. Repair priority "
            "combines severity, a static per-road-type traffic assumption, defect "
            "class and age; it is a decision aid, not a substitute for inspection. "
            "Coordinates tagged as simulated originate from demo footage. This is "
            "a research-grade prototype: every entry should be confirmed on site "
            "before a work order is issued.",
            small,
        )
    )

    document.build(story)
    return buffer.getvalue()
