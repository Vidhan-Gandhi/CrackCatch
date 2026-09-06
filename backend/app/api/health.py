"""Health and system-status endpoint.

Deliberately verbose: it reports whether the *real* database and the *trained*
model are in play, so a demo can never silently run on the in-memory fallback
or the classical-CV baseline without that being visible on screen.
"""

from __future__ import annotations

from pathlib import Path

from fastapi import APIRouter, Depends

from app.api.deps import get_repository
from app.core.config import Settings, get_settings
from app.db.mongo import database
from app.db.repository import DefectRepository
from app.models.schemas import HealthResponse
from app.services.pipeline_service import pipeline_service

router = APIRouter(tags=["system"])


@router.get("/api/health", response_model=HealthResponse, summary="System status")
async def health(
    repository: DefectRepository = Depends(get_repository),
    settings: Settings = Depends(get_settings),
) -> HealthResponse:
    db_ok = await database.ping()
    warnings = list(database.warnings)

    try:
        defect_count = await repository.count()
    except Exception:
        defect_count = 0
        db_ok = False
        warnings.append("Defect collection is not readable.")

    detector = pipeline_service.detector_name
    is_trained = pipeline_service.detector_is_trained_model
    if not is_trained:
        warnings.append(
            "Running the classical-CV baseline, not a trained YOLOv8 model. "
            f"No checkpoint at {Path(settings.model_weights)}. Detections are "
            "indicative only - train one with `python model/scripts/train.py`."
        )

    # The in-memory double answers a ping perfectly well, so a successful ping
    # is NOT evidence of real persistence. Anything other than a real MongoDB
    # is degraded - this endpoint exists precisely to surface that.
    persistent = database.backend == "mongodb"

    return HealthResponse(
        status="ok" if (db_ok and persistent and is_trained) else "degraded",
        version=settings.version,
        database="connected" if db_ok else "unavailable",
        database_backend=database.backend,
        detector=detector,
        detector_is_trained_model=is_trained,
        defect_count=defect_count,
        warnings=warnings,
    )
