"""Ingestion: drive the pipeline from a server-side path, an uploaded file, or
a citizen's crowdsourced photo.

All three land in the same stage 1 -> 6 path, which is the point the scope
makes about the crowdsourcing channel: a driver's phone photo "enters the same
detection pipeline as dashcam footage".
"""

from __future__ import annotations

import logging
import uuid
from datetime import datetime, timezone
from pathlib import Path

from fastapi import (
    APIRouter,
    BackgroundTasks,
    Depends,
    File,
    Form,
    HTTPException,
    Query,
    UploadFile,
)

from app.api.deps import get_repository
from app.core.config import Settings, get_settings
from app.core.security import require_any, require_write
from app.db.repository import DefectRepository
from app.models.schemas import CrowdsourceResponse, IngestJob, IngestRequest
from app.services.pipeline_service import pipeline_service

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/api/ingest", tags=["ingest"])

VIDEO_SUFFIXES = {".mp4", ".mov", ".avi", ".mkv", ".webm"}
IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}


def _safe_source_path(raw: str, settings: Settings) -> str:
    """Reject paths outside the repository.

    ``source`` names a file on the *server*, so without this an operator could
    point ingestion at any readable path. Camera indices and stream URLs are
    passed through untouched.
    """
    if raw.isdigit() or raw.startswith(("rtsp://", "http://", "https://")):
        return raw

    repo_root = Path(settings.storage_dir).resolve().parent
    candidate = Path(raw)
    resolved = (candidate if candidate.is_absolute() else repo_root / candidate).resolve()
    if not str(resolved).startswith(str(repo_root)):
        raise HTTPException(
            status_code=400,
            detail=(
                "Source path must live inside the project directory. "
                f"Got {raw!r}, which resolves outside {repo_root}."
            ),
        )
    if not resolved.exists():
        raise HTTPException(status_code=404, detail=f"Source not found: {raw}")
    return str(resolved)


@router.post("/run", response_model=IngestJob, summary="Run the pipeline on a server-side source")
async def run_ingest(
    request: IngestRequest,
    background: BackgroundTasks,
    wait: bool = Query(
        False,
        description="Block until the run completes. Handy for scripted demos; "
        "leave false so the dashboard can watch progress over WebSocket.",
    ),
    repository: DefectRepository = Depends(get_repository),
    settings: Settings = Depends(get_settings),
    _role: str = Depends(require_write),
) -> IngestJob:
    request.source = _safe_source_path(request.source, settings)
    job_id = uuid.uuid4().hex

    if wait:
        return await pipeline_service.run_ingest(request, repository, job_id)

    job = {
        "job_id": job_id,
        "status": "queued",
        "source": request.source,
        "created_at": datetime.now(timezone.utc),
        "started_at": None,
        "finished_at": None,
        "records_stored": 0,
        "stats": {},
        "error": None,
        "detector": pipeline_service.detector_name,
    }
    await repository.save_job(job)
    background.add_task(pipeline_service.run_ingest, request, repository, job_id)
    return IngestJob(**job)


@router.get("/jobs", response_model=list[IngestJob], summary="Recent ingestion jobs")
async def list_jobs(
    limit: int = Query(20, ge=1, le=100),
    repository: DefectRepository = Depends(get_repository),
    _role: str = Depends(require_write),
) -> list[IngestJob]:
    return await repository.list_jobs(limit)


@router.get("/jobs/{job_id}", response_model=IngestJob, summary="Ingestion job status")
async def get_job(
    job_id: str,
    repository: DefectRepository = Depends(get_repository),
    _role: str = Depends(require_write),
) -> IngestJob:
    job = await repository.get_job(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail=f"No job with id {job_id}")
    return IngestJob(**job)


async def _store_upload(file: UploadFile, settings: Settings, allowed: set[str]) -> Path:
    suffix = Path(file.filename or "upload").suffix.lower()
    if suffix not in allowed:
        raise HTTPException(
            status_code=415,
            detail=f"Unsupported file type {suffix!r}. Allowed: {sorted(allowed)}",
        )

    payload = await file.read()
    limit = settings.max_upload_mb * 1024 * 1024
    if len(payload) > limit:
        raise HTTPException(
            status_code=413,
            detail=f"File exceeds the {settings.max_upload_mb} MB upload limit",
        )
    if not payload:
        raise HTTPException(status_code=400, detail="Uploaded file is empty")

    settings.ensure_directories()
    destination = settings.upload_dir / f"{uuid.uuid4().hex}{suffix}"
    destination.write_bytes(payload)
    return destination


@router.post(
    "/upload",
    response_model=CrowdsourceResponse,
    summary="Upload a video or image and run it through the pipeline",
)
async def upload_and_process(
    file: UploadFile = File(...),
    latitude: float | None = Form(default=None),
    longitude: float | None = Form(default=None),
    road_type: str = Form(default="arterial"),
    repository: DefectRepository = Depends(get_repository),
    settings: Settings = Depends(get_settings),
    _role: str = Depends(require_write),
) -> CrowdsourceResponse:
    destination = await _store_upload(
        file, settings, VIDEO_SUFFIXES | IMAGE_SUFFIXES
    )
    source_type = "video" if destination.suffix.lower() in VIDEO_SUFFIXES else "image"

    defects = await pipeline_service.process_upload(
        destination,
        repository,
        latitude=latitude,
        longitude=longitude,
        road_type=road_type,
        source_type=source_type,
    )
    return CrowdsourceResponse(
        accepted=True,
        defects_found=len(defects),
        defects=defects,
        message=(
            f"Processed {destination.name}: {len(defects)} defect(s) detected."
            if defects
            else "No road damage detected in this file."
        ),
        upload_id=destination.stem,
    )


@router.post(
    "/crowdsource",
    response_model=CrowdsourceResponse,
    summary="Citizen photo submission (driver-facing PWA)",
)
async def crowdsource_submit(
    file: UploadFile = File(...),
    latitude: float | None = Form(default=None),
    longitude: float | None = Form(default=None),
    accuracy_m: float | None = Form(default=None),
    note: str = Form(default=""),
    road_type: str = Form(default="arterial"),
    repository: DefectRepository = Depends(get_repository),
    settings: Settings = Depends(get_settings),
    _role: str = Depends(require_any),
) -> CrowdsourceResponse:
    """Accept one photo from a driver and run it through the same pipeline.

    The submitter's coordinates come from the browser Geolocation API. Only
    the single point attached to a confirmed defect is persisted - no trail -
    per the data-governance requirement.
    """
    destination = await _store_upload(file, settings, IMAGE_SUFFIXES)

    defects = await pipeline_service.process_upload(
        destination,
        repository,
        latitude=latitude,
        longitude=longitude,
        accuracy_m=accuracy_m,
        road_type=road_type,
        source_type="crowdsource",
    )

    if note:
        for defect in defects:
            await repository.update_status(
                defect["_id"], new_status=None, note=f"Citizen note: {note}", by="citizen"
            )

    if not defects:
        return CrowdsourceResponse(
            accepted=True,
            defects_found=0,
            defects=[],
            message=(
                "Thanks - your photo was received, but no road damage was "
                "detected in it. Try a clearer shot taken closer to the defect."
            ),
            upload_id=destination.stem,
        )

    worst = max(defects, key=lambda d: d.get("priority_score", 0))
    return CrowdsourceResponse(
        accepted=True,
        defects_found=len(defects),
        defects=defects,
        message=(
            f"Thanks! {len(defects)} defect(s) reported to the road authority. "
            f"Highest severity: {worst['severity']} "
            f"(repair priority {worst['priority_score']:.0f}/100)."
        ),
        upload_id=destination.stem,
    )
