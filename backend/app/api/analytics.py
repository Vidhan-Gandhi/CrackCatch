"""Aggregate analytics: counts over time, severity mix, and the map heatmap.

Serves both the municipal dashboard and the smart-city planner view - the
scope notes these consume the same analytics, differing only in intent
(repair scheduling vs. long-term planning).
"""

from __future__ import annotations

from fastapi import APIRouter, Depends, Query

from app.api.deps import DefectFilters, get_repository
from app.core.security import require_read
from app.db.repository import DefectRepository
from app.models.schemas import AnalyticsSummary, HeatmapResponse

router = APIRouter(prefix="/api/analytics", tags=["analytics"])


@router.get("/summary", response_model=AnalyticsSummary, summary="Dashboard KPIs")
async def summary(
    filters: DefectFilters = Depends(),
    days: int = Query(30, ge=1, le=365, description="Window for the time series"),
    repository: DefectRepository = Depends(get_repository),
    _role: str = Depends(require_read),
) -> AnalyticsSummary:
    return await repository.analytics(filters.to_query(), days=days)


@router.get("/heatmap", response_model=HeatmapResponse, summary="Road-health heatmap")
async def heatmap(
    filters: DefectFilters = Depends(),
    cell_size_deg: float = Query(
        0.002,
        gt=0.0001,
        le=0.05,
        description="Grid cell size in degrees (~0.002 deg = ~200 m)",
    ),
    repository: DefectRepository = Depends(get_repository),
    _role: str = Depends(require_read),
) -> HeatmapResponse:
    """Severity-weighted defect density per grid cell.

    Weights are Minor 1, Moderate 2, Severe 4, so a cell with a few severe
    potholes outranks one with many hairline cracks. ``intensity`` is
    normalised against the busiest cell in the current filter, which is what
    the Leaflet heat layer consumes directly.
    """
    return await repository.heatmap(filters.to_query(), cell_size_deg=cell_size_deg)
