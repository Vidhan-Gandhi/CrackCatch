"""Exportable municipal reports (CSV / PDF) for a date range or ward."""

from __future__ import annotations

from datetime import datetime

from fastapi import APIRouter, Depends, Query
from fastapi.responses import Response

from app.api.deps import DefectFilters, get_repository
from app.core.security import require_read
from app.db.repository import DefectRepository
from app.services.report_service import build_csv, build_pdf

router = APIRouter(prefix="/api/reports", tags=["reports"])


def _filename(prefix: str, extension: str) -> str:
    return f"{prefix}_{datetime.now().strftime('%Y%m%d_%H%M')}.{extension}"


@router.get("/csv", summary="Export the filtered defects as CSV")
async def export_csv(
    filters: DefectFilters = Depends(),
    limit: int = Query(10_000, ge=1, le=50_000),
    repository: DefectRepository = Depends(get_repository),
    _role: str = Depends(require_read),
) -> Response:
    defects = await repository.all_matching(filters.to_query(), limit=limit)
    return Response(
        content=build_csv(defects),
        media_type="text/csv",
        headers={
            "Content-Disposition": f'attachment; filename="{_filename("crackcatch", "csv")}"'
        },
    )


@router.get("/pdf", summary="Export a municipal work-order PDF")
async def export_pdf(
    filters: DefectFilters = Depends(),
    ward: str | None = Query(default=None, max_length=80),
    limit: int = Query(10_000, ge=1, le=50_000),
    repository: DefectRepository = Depends(get_repository),
    _role: str = Depends(require_read),
) -> Response:
    defects = await repository.all_matching(filters.to_query(), limit=limit)
    pdf = build_pdf(
        defects,
        ward=ward,
        date_from=filters.date_from,
        date_to=filters.date_to,
    )
    return Response(
        content=pdf,
        media_type="application/pdf",
        headers={
            "Content-Disposition": f'attachment; filename="{_filename("crackcatch_report", "pdf")}"'
        },
    )
