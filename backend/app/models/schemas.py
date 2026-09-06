"""API request/response schemas.

These re-validate everything the model package produces before it reaches
MongoDB, and define the exact JSON contract the React dashboard consumes.
"""

from __future__ import annotations

from datetime import datetime
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator

Severity = Literal["Minor", "Moderate", "Severe"]
DefectClassName = Literal["pothole", "crack", "manhole"]
Status = Literal["New", "Verified", "Scheduled", "Repaired", "Rejected"]
SourceType = Literal["video", "image", "camera", "crowdsource"]


class BBox(BaseModel):
    x1: float
    y1: float
    x2: float
    y2: float

    @field_validator("x2")
    @classmethod
    def _x_ordered(cls, v, info):
        if "x1" in info.data and v < info.data["x1"]:
            raise ValueError("x2 must be >= x1")
        return v

    @field_validator("y2")
    @classmethod
    def _y_ordered(cls, v, info):
        if "y1" in info.data and v < info.data["y1"]:
            raise ValueError("y2 must be >= y1")
        return v


class Location(BaseModel):
    latitude: float = Field(ge=-90, le=90)
    longitude: float = Field(ge=-180, le=180)
    accuracy_m: float | None = None
    source: str = "unknown"


class SizeInfo(BaseModel):
    area_px: float
    area_frac: float
    width_m: float | None = None
    length_m: float | None = None
    area_m2: float | None = None
    method: str = "pixel_only"
    range_m: float | None = None
    reliable: bool = True
    confidence_note: str = ""


class DefectBase(BaseModel):
    defect_class: DefectClassName
    severity: Severity
    confidence: float = Field(ge=0.0, le=1.0)
    bbox: BBox
    location: Location
    detected_at: datetime
    size: SizeInfo | None = None
    severity_score: float = 0.0
    severity_breakdown: dict[str, Any] = Field(default_factory=dict)
    priority_score: float = 0.0
    status: Status = "New"
    source_type: SourceType = "video"
    source_ref: str = ""
    frame_index: int | None = None
    snapshot_path: str | None = None
    road_type: str = "arterial"
    notes: str = ""


class DefectCreate(DefectBase):
    client_id: str | None = None


class StatusEvent(BaseModel):
    """One entry in a defect's audit trail."""

    status: Status
    at: datetime
    by: str = "system"
    note: str = ""


class Defect(DefectBase):
    model_config = ConfigDict(populate_by_name=True)

    id: str = Field(alias="_id")
    client_id: str | None = None
    created_at: datetime
    updated_at: datetime
    priority_band: str = "Low"
    history: list[StatusEvent] = Field(default_factory=list)
    #: Photo attached when the defect was marked Repaired (before/after view).
    repair_photo_path: str | None = None
    repaired_at: datetime | None = None
    #: Set when an authority user corrects the model - feeds stage 8.
    review_label: str | None = None
    review_note: str = ""


class DefectUpdate(BaseModel):
    """Authority action on a defect."""

    status: Status | None = None
    note: str = ""
    by: str = "authority"
    road_type: str | None = None
    review_label: DefectClassName | None = None
    severity: Severity | None = None


class DefectPage(BaseModel):
    items: list[Defect]
    total: int
    page: int
    page_size: int
    pages: int


class SeverityBucket(BaseModel):
    severity: Severity
    count: int


class StatusBucket(BaseModel):
    status: Status
    count: int


class ClassBucket(BaseModel):
    defect_class: DefectClassName
    count: int


class TimeBucket(BaseModel):
    date: str
    count: int
    severe: int = 0


class Hotspot(BaseModel):
    latitude: float
    longitude: float
    count: int
    severe_count: int
    mean_priority: float
    intensity: float = Field(ge=0.0, le=1.0)


class AnalyticsSummary(BaseModel):
    total_defects: int
    open_defects: int
    repaired_defects: int
    severe_open: int
    mean_priority: float
    by_severity: list[SeverityBucket]
    by_status: list[StatusBucket]
    by_class: list[ClassBucket]
    over_time: list[TimeBucket]
    top_priority: list[Defect]
    generated_at: datetime


class HeatmapResponse(BaseModel):
    cells: list[Hotspot]
    cell_size_deg: float
    max_count: int


class IngestRequest(BaseModel):
    """Kick off a pipeline run over a server-side path (stage 1 -> 6)."""

    source: str = Field(description="Video file, image, directory, or camera index")
    target_fps: float = Field(default=2.0, gt=0, le=30)
    max_frames: int | None = Field(default=None, gt=0)
    road_type: str = "arterial"
    gps_method: Literal["simulated", "static", "gpx", "exif"] = "simulated"
    gps_start_lat: float | None = None
    gps_start_lon: float | None = None
    gpx_path: str | None = None
    confidence_threshold: float | None = Field(default=None, ge=0.0, le=1.0)


class IngestJob(BaseModel):
    job_id: str
    status: Literal["queued", "running", "completed", "failed"]
    source: str
    created_at: datetime
    started_at: datetime | None = None
    finished_at: datetime | None = None
    records_stored: int = 0
    stats: dict[str, Any] = Field(default_factory=dict)
    error: str | None = None
    detector: str = "unknown"


class CrowdsourceResponse(BaseModel):
    """Reply to a citizen photo submission."""

    accepted: bool
    defects_found: int
    defects: list[Defect] = Field(default_factory=list)
    message: str
    upload_id: str


class HealthResponse(BaseModel):
    status: Literal["ok", "degraded"]
    version: str
    database: str
    database_backend: str
    detector: str
    detector_is_trained_model: bool
    defect_count: int
    warnings: list[str] = Field(default_factory=list)
