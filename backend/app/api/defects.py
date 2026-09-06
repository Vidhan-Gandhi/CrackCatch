"""Defect CRUD, the repair-status workflow, and explainability."""

from __future__ import annotations

import logging
import uuid
from pathlib import Path

from fastapi import APIRouter, Depends, File, HTTPException, Query, UploadFile
from fastapi.responses import Response

from app.api.deps import DefectFilters, get_repository
from app.core.config import Settings, get_settings
from app.core.security import require_read, require_write
from app.db.repository import DefectRepository
from app.models.schemas import Defect, DefectCreate, DefectPage, DefectUpdate
from app.services.events import broadcaster, publish_defect_created

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/api/defects", tags=["defects"])


@router.get("", response_model=DefectPage, summary="List defects (filterable)")
async def list_defects(
    filters: DefectFilters = Depends(),
    page: int = Query(1, ge=1),
    page_size: int = Query(50, ge=1, le=500),
    sort_by: str = Query("detected_at", pattern="^(detected_at|priority_score|severity_score|created_at)$"),
    sort_dir: str = Query("desc", pattern="^(asc|desc)$"),
    repository: DefectRepository = Depends(get_repository),
    _role: str = Depends(require_read),
) -> DefectPage:
    items, total = await repository.list_defects(
        query=filters.to_query(),
        page=page,
        page_size=page_size,
        sort_by=sort_by,
        sort_dir=-1 if sort_dir == "desc" else 1,
    )
    return DefectPage(
        items=items,
        total=total,
        page=page,
        page_size=page_size,
        pages=max(1, -(-total // page_size)),
    )


@router.get("/near", response_model=list[Defect], summary="Defects near a point")
async def defects_near(
    lat: float = Query(..., ge=-90, le=90),
    lon: float = Query(..., ge=-180, le=180),
    radius_m: float = Query(150.0, gt=0, le=10_000),
    limit: int = Query(20, ge=1, le=200),
    repository: DefectRepository = Depends(get_repository),
    _role: str = Depends(require_read),
) -> list[Defect]:
    return await repository.near(lat, lon, radius_m, limit)


@router.get("/{defect_id}", response_model=Defect, summary="Defect detail")
async def get_defect(
    defect_id: str,
    repository: DefectRepository = Depends(get_repository),
    _role: str = Depends(require_read),
) -> Defect:
    defect = await repository.get(defect_id)
    if defect is None:
        raise HTTPException(status_code=404, detail=f"No defect with id {defect_id}")
    return defect


@router.post("", response_model=Defect, status_code=201, summary="Create a defect")
async def create_defect(
    payload: DefectCreate,
    repository: DefectRepository = Depends(get_repository),
    settings: Settings = Depends(get_settings),
    _role: str = Depends(require_write),
) -> Defect:
    document = payload.model_dump()
    # model_dump() emits `client_id: None` rather than omitting the key, so
    # setdefault would not fire. The repository generates one when it is
    # falsy; this keeps the intent explicit at the call site too.
    if not document.get("client_id"):
        document["client_id"] = uuid.uuid4().hex
    created = await repository.create(document)
    await publish_defect_created(created, settings.alert_priority_threshold)
    return created


@router.patch("/{defect_id}", response_model=Defect, summary="Advance the repair workflow")
async def update_defect(
    defect_id: str,
    payload: DefectUpdate,
    repository: DefectRepository = Depends(get_repository),
    _role: str = Depends(require_write),
) -> Defect:
    """Apply an authority action.

    Status changes are validated against the workflow
    ``New -> Verified -> Scheduled -> Repaired``; an illegal jump is a 409
    rather than a silent no-op, so a mis-wired UI is caught immediately.
    """
    try:
        updated = await repository.update_status(
            defect_id,
            new_status=payload.status,
            note=payload.note,
            by=payload.by,
            road_type=payload.road_type,
            review_label=payload.review_label,
            severity=payload.severity,
        )
    except ValueError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc

    if updated is None:
        raise HTTPException(status_code=404, detail=f"No defect with id {defect_id}")

    await broadcaster.broadcast("defect.updated", updated)
    return updated


@router.post(
    "/{defect_id}/repair-photo",
    response_model=Defect,
    summary="Attach a post-repair photo (before/after verification)",
)
async def upload_repair_photo(
    defect_id: str,
    file: UploadFile = File(...),
    repository: DefectRepository = Depends(get_repository),
    settings: Settings = Depends(get_settings),
    _role: str = Depends(require_write),
) -> Defect:
    defect = await repository.get(defect_id)
    if defect is None:
        raise HTTPException(status_code=404, detail=f"No defect with id {defect_id}")

    suffix = Path(file.filename or "repair.jpg").suffix.lower() or ".jpg"
    if suffix not in {".jpg", ".jpeg", ".png", ".webp"}:
        raise HTTPException(status_code=415, detail=f"Unsupported image type: {suffix}")

    settings.ensure_directories()
    filename = f"{defect_id}_{uuid.uuid4().hex[:8]}{suffix}"
    destination = settings.repair_photo_dir / filename

    payload = await file.read()
    limit = settings.max_upload_mb * 1024 * 1024
    if len(payload) > limit:
        raise HTTPException(
            status_code=413, detail=f"File exceeds the {settings.max_upload_mb} MB limit"
        )
    destination.write_bytes(payload)

    updated = await repository.attach_repair_photo(defect_id, filename)
    if updated is None:
        raise HTTPException(status_code=404, detail=f"No defect with id {defect_id}")
    await broadcaster.broadcast("defect.updated", updated)
    return updated


@router.get(
    "/{defect_id}/explain",
    summary="Explainability overlay for a detection",
    response_class=Response,
    responses={200: {"content": {"image/jpeg": {}}}},
)
async def explain_defect(
    defect_id: str,
    repository: DefectRepository = Depends(get_repository),
    settings: Settings = Depends(get_settings),
    _role: str = Depends(require_read),
) -> Response:
    """Return a heatmap over the defect's snapshot showing what drove it.

    Uses true Grad-CAM when a YOLOv8 checkpoint is loaded, and the
    gradient-free saliency overlay when the classical-CV baseline is running.
    The ``X-Explain-Method`` response header states which was used, so the UI
    can label it honestly rather than implying Grad-CAM in both cases.
    """
    import cv2

    from crackcatch_model.explain import gradcam, saliency_overlay
    from crackcatch_model.types import BoundingBox

    defect = await repository.get(defect_id)
    if defect is None:
        raise HTTPException(status_code=404, detail=f"No defect with id {defect_id}")
    if not defect.get("snapshot_path"):
        raise HTTPException(status_code=404, detail="This defect has no stored snapshot")

    snapshot = settings.snapshot_dir / defect["snapshot_path"]
    if not snapshot.exists():
        raise HTTPException(status_code=404, detail="Snapshot file is missing on disk")

    frame = cv2.imread(str(snapshot))
    if frame is None:
        raise HTTPException(status_code=500, detail="Snapshot could not be decoded")

    box = defect.get("bbox") or {}
    bbox = None
    if box:
        try:
            bbox = BoundingBox(box["x1"], box["y1"], box["x2"], box["y2"])
        except (KeyError, ValueError):
            bbox = None

    if Path(settings.model_weights).exists():
        # gradcam() reports what it actually produced. A checkpoint existing is
        # not evidence that a CAM could be computed, so the header must come
        # from the return value, never from the file check.
        overlay, method = gradcam(frame, str(settings.model_weights), bbox=bbox)
    else:
        overlay = saliency_overlay(frame, bbox)
        method = "saliency (no trained checkpoint; NOT Grad-CAM)"

    ok, encoded = cv2.imencode(".jpg", overlay, [int(cv2.IMWRITE_JPEG_QUALITY), 88])
    if not ok:
        raise HTTPException(status_code=500, detail="Failed to encode the overlay")

    return Response(
        content=encoded.tobytes(),
        media_type="image/jpeg",
        headers={"X-Explain-Method": method, "Cache-Control": "no-store"},
    )


@router.delete("/{defect_id}", status_code=204, summary="Delete a defect")
async def delete_defect(
    defect_id: str,
    repository: DefectRepository = Depends(get_repository),
    _role: str = Depends(require_write),
) -> Response:
    if not await repository.delete(defect_id):
        raise HTTPException(status_code=404, detail=f"No defect with id {defect_id}")
    await broadcaster.broadcast("defect.deleted", {"_id": defect_id})
    return Response(status_code=204)
