"""CrackCatch FastAPI application - pipeline stage 6 (storage) and the API
that stage 7 (the React dashboard) consumes.

Run locally:
    uvicorn app.main:app --reload --app-dir backend

Interactive API docs: http://localhost:8000/docs
"""

from __future__ import annotations

import logging
import sys
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles

# The model package lives in /model and is imported by the service layer. Add
# it to sys.path before any app module pulls it in, so the backend runs from a
# checkout without an editable install (the Docker image installs it properly).
_REPO_ROOT = Path(__file__).resolve().parents[2]
_MODEL_DIR = _REPO_ROOT / "model"
if _MODEL_DIR.exists() and str(_MODEL_DIR) not in sys.path:
    sys.path.insert(0, str(_MODEL_DIR))

from app.api import analytics, defects, health, ingest, reports, ws  # noqa: E402
from app.core.config import get_settings  # noqa: E402
from app.db.mongo import database  # noqa: E402

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)-8s %(name)s: %(message)s",
)
logger = logging.getLogger("crackcatch")


@asynccontextmanager
async def lifespan(app: FastAPI):
    settings = get_settings()
    settings.ensure_directories()
    await database.connect(settings)

    logger.info(
        "CrackCatch %s ready | db=%s | storage=%s",
        settings.version,
        database.backend,
        settings.storage_dir,
    )
    for warning in database.warnings:
        logger.warning(warning)
    if not Path(settings.model_weights).exists():
        logger.warning(
            "No trained checkpoint at %s - the classical-CV baseline will be "
            "used. See docs/MODEL_CARD.md.",
            settings.model_weights,
        )

    yield
    await database.close()
    logger.info("CrackCatch shut down")


settings = get_settings()

app = FastAPI(
    title=settings.app_name,
    version=settings.version,
    lifespan=lifespan,
    description=(
        "Real-time pothole and road-damage detection with severity estimation, "
        "GPS geo-tagging and a municipal repair workflow.\n\n"
        "**Prototype notice:** severity and size are documented estimates from a "
        "single monocular camera, not lab-grade measurements. Pothole *depth* is "
        "not measured. See `/docs/SEVERITY.md` in the repository."
    ),
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
    expose_headers=["Content-Disposition", "X-Explain-Method"],
)


@app.exception_handler(Exception)
async def unhandled_exception_handler(request: Request, exc: Exception) -> JSONResponse:
    """Log the traceback server-side; return a terse message to the client."""
    logger.exception("Unhandled error on %s %s", request.method, request.url.path)
    return JSONResponse(
        status_code=500,
        content={
            "detail": "Internal server error",
            "type": type(exc).__name__,
            "path": request.url.path,
        },
    )


app.include_router(health.router)
app.include_router(defects.router)
app.include_router(analytics.router)
app.include_router(ingest.router)
app.include_router(reports.router)
app.include_router(ws.router)

# Snapshots, uploaded media and repair photos are served straight off disk.
# A production deployment would put these in object storage (S3/GCS) behind a
# CDN with signed URLs - noted in docs/ARCHITECTURE.md.
settings.ensure_directories()
app.mount("/media/snapshots", StaticFiles(directory=settings.snapshot_dir), name="snapshots")
app.mount("/media/repairs", StaticFiles(directory=settings.repair_photo_dir), name="repairs")
app.mount("/media/uploads", StaticFiles(directory=settings.upload_dir), name="uploads")


@app.get("/", tags=["system"], summary="Service banner")
async def root() -> dict:
    return {
        "name": settings.app_name,
        "version": settings.version,
        "docs": "/docs",
        "health": "/api/health",
        "live_feed": "/ws/defects",
        "pipeline": [
            "1. Video/Image Capture",
            "2. Frame Preprocessing",
            "3. AI Detection Engine (YOLOv8)",
            "4. Severity & Size Estimation",
            "5. GPS Geo-Tagging",
            "6. Road Damage Database",
            "7. Authority Dashboard",
            "8. Continuous Model Improvement",
        ],
    }
