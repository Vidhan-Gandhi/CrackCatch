"""Core value types shared across the CrackCatch pipeline.

These are plain dataclasses (no pydantic, no ODM) so that the model package
stays importable from a bare Python environment - a notebook, a training
script or a Raspberry Pi - without pulling in the backend's dependencies.
"""

from __future__ import annotations

import math
import uuid
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from enum import Enum
from typing import Any


class Severity(str, Enum):
    """Severity levels. Names are fixed by the project scope / review paper."""

    MINOR = "Minor"
    MODERATE = "Moderate"
    SEVERE = "Severe"

    @property
    def rank(self) -> int:
        return {"Minor": 1, "Moderate": 2, "Severe": 3}[self.value]


class DefectClass(str, Enum):
    """Detection classes. ``pothole`` and ``crack`` are the mandatory two."""

    POTHOLE = "pothole"
    CRACK = "crack"
    MANHOLE = "manhole"

    @classmethod
    def from_any(cls, value: str) -> "DefectClass":
        """Map loose dataset label names onto our canonical classes.

        RDD2022 uses D00/D10/D20/D40 style labels; other public pothole sets
        use ``Pothole``, ``pot-hole``, ``longitudinal crack`` and so on.
        """
        raw = str(value).strip().lower().replace("-", "_").replace(" ", "_")
        rdd = {
            "d00": cls.CRACK,      # longitudinal crack
            "d10": cls.CRACK,      # transverse crack
            "d20": cls.CRACK,      # alligator crack
            "d40": cls.POTHOLE,    # pothole / rutting
            "d43": cls.MANHOLE,
            "d44": cls.MANHOLE,
        }
        if raw in rdd:
            return rdd[raw]
        if "pot" in raw:
            return cls.POTHOLE
        if "crack" in raw or "fissure" in raw:
            return cls.CRACK
        if "manhole" in raw or "drain" in raw or "cover" in raw:
            return cls.MANHOLE
        raise ValueError(f"Unrecognised defect class label: {value!r}")


class DefectStatus(str, Enum):
    """Repair workflow: New -> Verified -> Scheduled -> Repaired."""

    NEW = "New"
    VERIFIED = "Verified"
    SCHEDULED = "Scheduled"
    REPAIRED = "Repaired"
    REJECTED = "Rejected"  # false positive flagged by an authority user


#: Legal forward transitions in the repair workflow.
STATUS_TRANSITIONS: dict[DefectStatus, tuple[DefectStatus, ...]] = {
    DefectStatus.NEW: (DefectStatus.VERIFIED, DefectStatus.REJECTED),
    DefectStatus.VERIFIED: (DefectStatus.SCHEDULED, DefectStatus.REJECTED),
    DefectStatus.SCHEDULED: (DefectStatus.REPAIRED, DefectStatus.VERIFIED),
    DefectStatus.REPAIRED: (DefectStatus.SCHEDULED,),  # re-open a bad repair
    DefectStatus.REJECTED: (DefectStatus.NEW,),        # undo a wrong rejection
}


@dataclass(frozen=True)
class BoundingBox:
    """Axis-aligned box in absolute pixel coordinates (x1, y1) - (x2, y2)."""

    x1: float
    y1: float
    x2: float
    y2: float

    def __post_init__(self) -> None:
        if self.x2 < self.x1 or self.y2 < self.y1:
            raise ValueError(f"Degenerate bounding box: {self!r}")

    @property
    def width(self) -> float:
        return self.x2 - self.x1

    @property
    def height(self) -> float:
        return self.y2 - self.y1

    @property
    def area(self) -> float:
        return self.width * self.height

    @property
    def centre(self) -> tuple[float, float]:
        return (self.x1 + self.x2) / 2.0, (self.y1 + self.y2) / 2.0

    @property
    def aspect_ratio(self) -> float:
        """Long side / short side, always >= 1. A crack is a high-AR box."""
        w, h = max(self.width, 1e-6), max(self.height, 1e-6)
        return max(w, h) / min(w, h)

    def iou(self, other: "BoundingBox") -> float:
        ix1, iy1 = max(self.x1, other.x1), max(self.y1, other.y1)
        ix2, iy2 = min(self.x2, other.x2), min(self.y2, other.y2)
        inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
        union = self.area + other.area - inter
        return inter / union if union > 0 else 0.0

    def clip(self, width: int, height: int) -> "BoundingBox":
        return BoundingBox(
            max(0.0, min(self.x1, width)),
            max(0.0, min(self.y1, height)),
            max(0.0, min(self.x2, width)),
            max(0.0, min(self.y2, height)),
        )

    def as_xyxy(self) -> tuple[float, float, float, float]:
        return self.x1, self.y1, self.x2, self.y2

    def to_dict(self) -> dict[str, float]:
        return {"x1": self.x1, "y1": self.y1, "x2": self.x2, "y2": self.y2}


@dataclass(frozen=True)
class GeoPoint:
    """A single WGS-84 point tied to a confirmed defect.

    Per the data-governance requirement we only ever persist discrete points
    attached to a detection - never a continuous GPS trail of the vehicle.
    """

    latitude: float
    longitude: float
    accuracy_m: float | None = None
    source: str = "unknown"  # device | exif | gpx | simulated

    def __post_init__(self) -> None:
        if not -90.0 <= self.latitude <= 90.0:
            raise ValueError(f"latitude out of range: {self.latitude}")
        if not -180.0 <= self.longitude <= 180.0:
            raise ValueError(f"longitude out of range: {self.longitude}")

    def distance_m(self, other: "GeoPoint") -> float:
        """Great-circle distance in metres (haversine)."""
        radius = 6_371_000.0
        p1, p2 = math.radians(self.latitude), math.radians(other.latitude)
        dp = p2 - p1
        dl = math.radians(other.longitude - self.longitude)
        a = math.sin(dp / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dl / 2) ** 2
        return 2 * radius * math.asin(math.sqrt(a))

    def to_dict(self) -> dict[str, Any]:
        return {
            "latitude": self.latitude,
            "longitude": self.longitude,
            "accuracy_m": self.accuracy_m,
            "source": self.source,
        }

    def to_geojson(self) -> dict[str, Any]:
        """GeoJSON Point - the format MongoDB's 2dsphere index expects."""
        return {"type": "Point", "coordinates": [self.longitude, self.latitude]}


@dataclass(frozen=True)
class SizeEstimate:
    """Real-world size estimate for a defect.

    Honesty note: these are *estimates* derived from a flat-road perspective
    projection, not lab-grade measurements. Depth is not observable from a
    single monocular frame - see ``docs/SEVERITY.md``.
    """

    area_px: float
    area_frac: float                 # bbox area / frame area, in [0, 1]
    width_m: float | None = None
    length_m: float | None = None
    area_m2: float | None = None
    method: str = "pixel_only"       # pixel_only | ground_plane | reference_object
    #: Ground distance from the camera to the defect, when recoverable.
    range_m: float | None = None
    #: False when the metric numbers exist but should NOT drive a decision -
    #: e.g. the defect is so far away that one pixel spans a large patch of
    #: road, so the estimate carries an error bar wider than the estimate.
    #: Severity scoring falls back to pixel measures when this is False.
    reliable: bool = True
    confidence_note: str = ""

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _round_numeric(mapping: dict[str, Any], digits: int = 4) -> dict[str, Any]:
    """Round float values, pass non-numeric provenance keys through untouched.

    ``severity_breakdown`` carries mostly floats but also a ``_size_basis``
    string naming which measure the size factor came from, which the dashboard
    shows in a tooltip.
    """
    return {
        key: round(value, digits) if isinstance(value, (int, float)) else value
        for key, value in mapping.items()
    }


@dataclass
class Detection:
    """A single surviving detection within one frame (stages 3 + 4)."""

    defect_class: DefectClass
    confidence: float
    bbox: BoundingBox
    severity: Severity = Severity.MINOR
    severity_score: float = 0.0
    severity_breakdown: dict[str, Any] = field(default_factory=dict)
    size: SizeEstimate | None = None
    priority_score: float = 0.0
    track_id: int | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "defect_class": self.defect_class.value,
            "confidence": round(self.confidence, 4),
            "bbox": self.bbox.to_dict(),
            "severity": self.severity.value,
            "severity_score": round(self.severity_score, 4),
            "severity_breakdown": _round_numeric(self.severity_breakdown),
            "size": self.size.to_dict() if self.size else None,
            "priority_score": round(self.priority_score, 2),
            "track_id": self.track_id,
        }


def _utc_now() -> datetime:
    return datetime.now(timezone.utc)


@dataclass
class DefectRecord:
    """One persisted road-damage record - the unit the dashboard displays.

    This is the contract between the model package (producer) and the backend
    storage layer (consumer). The backend re-validates it with pydantic before
    it reaches MongoDB.
    """

    defect_class: DefectClass
    severity: Severity
    confidence: float
    bbox: BoundingBox
    location: GeoPoint
    detected_at: datetime = field(default_factory=_utc_now)
    size: SizeEstimate | None = None
    severity_score: float = 0.0
    severity_breakdown: dict[str, Any] = field(default_factory=dict)
    priority_score: float = 0.0
    status: DefectStatus = DefectStatus.NEW
    source_type: str = "video"       # video | image | camera | crowdsource
    source_ref: str = ""             # filename, device id, upload id
    frame_index: int | None = None
    snapshot_path: str | None = None
    road_type: str = "arterial"
    client_id: str = field(default_factory=lambda: uuid.uuid4().hex)
    notes: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "client_id": self.client_id,
            "defect_class": self.defect_class.value,
            "severity": self.severity.value,
            "severity_score": round(self.severity_score, 4),
            "severity_breakdown": _round_numeric(self.severity_breakdown),
            "priority_score": round(self.priority_score, 2),
            "confidence": round(self.confidence, 4),
            "bbox": self.bbox.to_dict(),
            "location": self.location.to_dict(),
            "detected_at": self.detected_at.isoformat(),
            "size": self.size.to_dict() if self.size else None,
            "status": self.status.value,
            "source_type": self.source_type,
            "source_ref": self.source_ref,
            "frame_index": self.frame_index,
            "snapshot_path": self.snapshot_path,
            "road_type": self.road_type,
            "notes": self.notes,
        }
