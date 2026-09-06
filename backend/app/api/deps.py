"""Shared FastAPI dependencies."""

from __future__ import annotations

from datetime import datetime, timezone

from fastapi import Depends, Query
from motor.motor_asyncio import AsyncIOMotorDatabase

from app.db.mongo import get_database
from app.db.repository import DefectRepository


async def get_repository(
    db: AsyncIOMotorDatabase = Depends(get_database),
) -> DefectRepository:
    return DefectRepository(db)


class DefectFilters:
    """The dashboard's filter bar, as a reusable query-parameter dependency."""

    def __init__(
        self,
        severity: list[str] | None = Query(
            default=None, description="Minor | Moderate | Severe (repeatable)"
        ),
        status: list[str] | None = Query(
            default=None, description="New | Verified | Scheduled | Repaired | Rejected"
        ),
        defect_class: list[str] | None = Query(
            default=None, description="pothole | crack | manhole (repeatable)"
        ),
        date_from: datetime | None = Query(default=None),
        date_to: datetime | None = Query(default=None),
        min_priority: float | None = Query(default=None, ge=0, le=100),
        source_type: str | None = Query(default=None),
        bbox: str | None = Query(
            default=None,
            description="Map viewport as 'min_lon,min_lat,max_lon,max_lat'",
        ),
        search: str | None = Query(default=None, max_length=120),
    ) -> None:
        self.severity = severity
        self.status = status
        self.defect_class = defect_class
        self.date_from = _aware(date_from)
        self.date_to = _aware(date_to)
        self.min_priority = min_priority
        self.source_type = source_type
        self.search = search
        self.bbox = _parse_bbox(bbox)

    def to_query(self) -> dict:
        return DefectRepository.build_filter(
            severity=self.severity,
            status=self.status,
            defect_class=self.defect_class,
            date_from=self.date_from,
            date_to=self.date_to,
            min_priority=self.min_priority,
            source_type=self.source_type,
            bbox=self.bbox,
            search=self.search,
        )


def _aware(value: datetime | None) -> datetime | None:
    if value is not None and value.tzinfo is None:
        return value.replace(tzinfo=timezone.utc)
    return value


def _parse_bbox(raw: str | None) -> tuple[float, float, float, float] | None:
    if not raw:
        return None
    try:
        parts = [float(p) for p in raw.split(",")]
    except ValueError:
        return None
    if len(parts) != 4:
        return None
    min_lon, min_lat, max_lon, max_lat = parts
    return (min(min_lon, max_lon), min(min_lat, max_lat),
            max(min_lon, max_lon), max(min_lat, max_lat))
