"""Bridges the model package (stages 1-5) to storage (stage 6) and the live
dashboard (stage 7).

The heavy work - video decode and CNN inference - is synchronous, CPU-bound
and would block the event loop, so each run executes in a worker thread while
the API stays responsive. Records are handed back to the loop and written to
MongoDB in order, then broadcast to connected dashboards.
"""

from __future__ import annotations

import asyncio
import logging
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from app.core.config import Settings, get_settings
from app.db.repository import DefectRepository
from app.services.events import broadcaster, publish_defect_created

logger = logging.getLogger(__name__)


def _utc_now() -> datetime:
    return datetime.now(timezone.utc)


class PipelineService:
    """Owns detector construction and ingestion-job lifecycle."""

    def __init__(self, settings: Settings | None = None) -> None:
        self.settings = settings or get_settings()
        self._detector = None
        self._detector_lock = asyncio.Lock()

    # ----------------------------------------------------------- detector

    def build_detector(self):
        """Construct the detector once and reuse it across requests.

        Loading YOLO weights costs a second or two; doing it per request would
        dominate every ingest.
        """
        from crackcatch_model.detector import build_detector

        if self._detector is None:
            self._detector = build_detector(
                weights=self.settings.model_weights,
                confidence_threshold=self.settings.confidence_threshold,
                device=self.settings.inference_device,
                allow_fallback=self.settings.detector_fallback,
            )
        return self._detector

    @property
    def detector_name(self) -> str:
        try:
            return self.build_detector().name
        except Exception as exc:
            logger.error("Detector unavailable: %s", exc)
            return "unavailable"

    @property
    def detector_is_trained_model(self) -> bool:
        """False when the classical-CV baseline is standing in for YOLO."""
        return self.detector_name == "yolov8"

    # ------------------------------------------------------------- config

    def _calibration_spec(self) -> dict[str, Any]:
        return {
            "method": "ground_plane",
            "camera_height_m": self.settings.camera_height_m,
            "pitch_deg": self.settings.camera_pitch_deg,
            "hfov_deg": self.settings.camera_hfov_deg,
        }

    def _gps_spec(self, request: Any) -> dict[str, Any]:
        method = getattr(request, "gps_method", "simulated")
        if method == "gpx" and getattr(request, "gpx_path", None):
            return {"method": "gpx", "path": request.gpx_path}
        if method == "static":
            return {
                "method": "static",
                "latitude": request.gps_start_lat or self.settings.demo_origin_lat,
                "longitude": request.gps_start_lon or self.settings.demo_origin_lon,
            }
        return {
            "method": "simulated",
            "start": (
                request.gps_start_lat or self.settings.demo_origin_lat,
                request.gps_start_lon or self.settings.demo_origin_lon,
            ),
            "speed_kmph": self.settings.demo_speed_kmph,
        }

    def _pipeline_config(self, request: Any):
        from crackcatch_model.pipeline import PipelineConfig

        return PipelineConfig(
            weights=self.settings.model_weights,
            confidence_threshold=(
                getattr(request, "confidence_threshold", None)
                or self.settings.confidence_threshold
            ),
            device=self.settings.inference_device,
            allow_detector_fallback=self.settings.detector_fallback,
            calibration=self._calibration_spec(),
            road_type=getattr(request, "road_type", "arterial"),
            snapshot_dir=self.settings.snapshot_dir,
        )

    # ------------------------------------------------------------ running

    def _run_blocking(self, request: Any) -> tuple[list[dict[str, Any]], dict[str, Any], str]:
        """Synchronous pipeline execution - always called in a worker thread."""
        from crackcatch_model.geotag import build_provider
        from crackcatch_model.pipeline import CrackCatchPipeline
        from crackcatch_model.sources import open_source

        pipeline = CrackCatchPipeline(
            config=self._pipeline_config(request),
            detector=self.build_detector(),
            gps=build_provider(self._gps_spec(request)),
        )
        source = open_source(
            request.source,
            target_fps=request.target_fps,
            max_frames=request.max_frames,
        )
        records = [record.to_dict() for record in pipeline.run(source)]
        return records, pipeline.stats.to_dict(), pipeline.detector.name

    async def run_ingest(
        self, request: Any, repository: DefectRepository, job_id: str | None = None
    ) -> dict[str, Any]:
        """Execute one ingestion job end-to-end and persist the results."""
        job_id = job_id or uuid.uuid4().hex
        job: dict[str, Any] = {
            "job_id": job_id,
            "status": "running",
            "source": request.source,
            "created_at": _utc_now(),
            "started_at": _utc_now(),
            "finished_at": None,
            "records_stored": 0,
            "stats": {},
            "error": None,
            "detector": "unknown",
        }
        await repository.save_job(job)
        await broadcaster.broadcast("job.started", {"job_id": job_id, "source": request.source})

        try:
            records, stats, detector_name = await asyncio.to_thread(
                self._run_blocking, request
            )
        except Exception as exc:
            logger.exception("Ingest job %s failed", job_id)
            job.update(
                status="failed",
                error=f"{type(exc).__name__}: {exc}",
                finished_at=_utc_now(),
            )
            await repository.save_job(job)
            await broadcaster.broadcast("job.failed", {"job_id": job_id, "error": job["error"]})
            return job

        stored = 0
        for record in records:
            try:
                saved = await repository.create(record)
                stored += 1
                await publish_defect_created(
                    saved, self.settings.alert_priority_threshold
                )
            except Exception:
                logger.exception("Failed to store a detection from job %s", job_id)

        job.update(
            status="completed",
            records_stored=stored,
            stats=stats,
            detector=detector_name,
            finished_at=_utc_now(),
        )
        await repository.save_job(job)
        await broadcaster.broadcast(
            "job.completed",
            {"job_id": job_id, "records_stored": stored, "stats": stats},
        )
        logger.info("Ingest job %s stored %d defects (%s)", job_id, stored, stats)
        return job

    async def process_upload(
        self,
        file_path: Path,
        repository: DefectRepository,
        latitude: float | None = None,
        longitude: float | None = None,
        accuracy_m: float | None = None,
        road_type: str = "arterial",
        source_type: str = "crowdsource",
    ) -> list[dict[str, Any]]:
        """Run one uploaded photo/video through the pipeline (stage 1 entry
        point for the citizen PWA).

        Location precedence: the browser's Geolocation reading if supplied,
        then the photo's EXIF, then the configured demo origin - each tagged
        with its own ``source`` so the dashboard shows where the pin came from.
        """
        from crackcatch_model.geotag import (
            DeviceGpsProvider,
            ExifGpsProvider,
            StaticGpsProvider,
        )
        from crackcatch_model.pipeline import CrackCatchPipeline, PipelineConfig
        from crackcatch_model.sources import SingleImageSource, VideoFileSource

        if latitude is not None and longitude is not None:
            gps = DeviceGpsProvider(latitude, longitude, accuracy_m)
        else:
            gps = ExifGpsProvider(
                file_path,
                fallback=StaticGpsProvider(
                    self.settings.demo_origin_lat, self.settings.demo_origin_lon
                ),
            )

        config = PipelineConfig(
            weights=self.settings.model_weights,
            confidence_threshold=self.settings.confidence_threshold,
            device=self.settings.inference_device,
            allow_detector_fallback=self.settings.detector_fallback,
            calibration=self._calibration_spec(),
            road_type=road_type,
            snapshot_dir=self.settings.snapshot_dir,
            # One photo of one pothole should yield one record; there is no
            # cross-frame repetition to collapse.
            dedupe=file_path.suffix.lower() not in {".jpg", ".jpeg", ".png", ".webp"},
        )

        def _work() -> list[dict[str, Any]]:
            pipeline = CrackCatchPipeline(
                config=config, detector=self.build_detector(), gps=gps
            )
            if file_path.suffix.lower() in {".mp4", ".mov", ".avi", ".mkv", ".webm"}:
                source = VideoFileSource(file_path, target_fps=2.0)
            else:
                source = SingleImageSource(
                    path=file_path,
                    source_ref=file_path.name,
                    source_type=source_type,
                )
            return [r.to_dict() for r in pipeline.run(source)]

        records = await asyncio.to_thread(_work)

        saved_records = []
        for record in records:
            record["source_type"] = source_type
            saved = await repository.create(record)
            saved_records.append(saved)
            await publish_defect_created(saved, self.settings.alert_priority_threshold)
        return saved_records


#: Process-wide service instance.
pipeline_service = PipelineService()
