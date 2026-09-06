"""Pipeline orchestration - wires stages 1 through 5 together.

``CrackCatchPipeline`` is deliberately storage-agnostic: it *yields*
``DefectRecord`` objects and knows nothing about MongoDB, FastAPI or the
dashboard. The backend consumes the stream and handles stage 6; the frontend
handles stage 7. That is what makes the modularity requirement real rather
than decorative - this module can be driven from a CLI, a notebook or an edge
device with no web stack installed.

Cross-frame deduplication
-------------------------
A pothole filmed at 2 fps from a moving car appears in several consecutive
frames. Writing one database row per frame would inflate every count on the
dashboard and make the heatmap meaningless. ``DefectDeduplicator`` merges
detections of the same class that land within ``dedupe_radius_m`` of each
other inside a ``dedupe_window_s`` time window, keeping the highest-severity
observation as the canonical record.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Callable, Iterator

import cv2
import numpy as np

from crackcatch_model import priority as priority_mod
from crackcatch_model import severity as severity_mod
from crackcatch_model.calibration import (
    Calibration,
    GroundPlaneCalibration,
    NoCalibration,
    build_calibration,
)
from crackcatch_model.detector import Detector, build_detector
from crackcatch_model.geotag import GpsProvider, SimulatedRouteProvider, build_provider
from crackcatch_model.preprocess import (
    DEFAULT_PREPROCESS_CONFIG,
    PreprocessConfig,
    preprocess,
)
from crackcatch_model.sources import CapturedFrame, FrameSource
from crackcatch_model.types import (
    BoundingBox,
    DefectRecord,
    Detection,
    GeoPoint,
    Severity,
)

logger = logging.getLogger(__name__)


@dataclass
class PipelineStats:
    """Throughput accounting. Reported rather than claimed."""

    frames_read: int = 0
    frames_processed: int = 0
    frames_skipped_blur: int = 0
    raw_detections: int = 0
    records_emitted: int = 0
    records_deduped: int = 0
    inference_time_s: float = 0.0
    total_time_s: float = 0.0

    @property
    def inference_fps(self) -> float:
        """Detector-only throughput."""
        if self.inference_time_s <= 0:
            return 0.0
        return self.frames_processed / self.inference_time_s

    @property
    def end_to_end_fps(self) -> float:
        """Capture + preprocess + detect + score + geo-tag + snapshot."""
        if self.total_time_s <= 0:
            return 0.0
        return self.frames_processed / self.total_time_s

    def to_dict(self) -> dict[str, Any]:
        return {
            "frames_read": self.frames_read,
            "frames_processed": self.frames_processed,
            "frames_skipped_blur": self.frames_skipped_blur,
            "raw_detections": self.raw_detections,
            "records_emitted": self.records_emitted,
            "records_deduped": self.records_deduped,
            "inference_time_s": round(self.inference_time_s, 3),
            "total_time_s": round(self.total_time_s, 3),
            "inference_fps": round(self.inference_fps, 2),
            "end_to_end_fps": round(self.end_to_end_fps, 2),
        }


@dataclass
class DefectDeduplicator:
    """Merge repeat sightings of one physical defect across frames."""

    radius_m: float = 12.0
    window_s: float = 20.0

    def __post_init__(self) -> None:
        self._seen: list[DefectRecord] = []

    def _is_duplicate(self, record: DefectRecord) -> DefectRecord | None:
        cutoff = record.detected_at - timedelta(seconds=self.window_s)
        # Walk backwards: the match, if any, is almost always recent.
        for existing in reversed(self._seen):
            if existing.detected_at < cutoff:
                break
            if existing.defect_class is not record.defect_class:
                continue
            if existing.location.distance_m(record.location) <= self.radius_m:
                return existing
        return None

    def add(self, record: DefectRecord) -> tuple[DefectRecord | None, bool]:
        """Return ``(record_to_emit, was_duplicate)``.

        On a duplicate we return ``None`` to emit, but upgrade the stored
        record in place if the new sighting is worse - a pothole seen more
        clearly from closer up should win.
        """
        match = self._is_duplicate(record)
        if match is None:
            self._seen.append(record)
            # Keep the buffer bounded on long runs.
            if len(self._seen) > 500:
                self._seen = self._seen[-250:]
            return record, False

        if record.severity_score > match.severity_score:
            match.severity = record.severity
            match.severity_score = record.severity_score
            match.severity_breakdown = record.severity_breakdown
            match.confidence = max(match.confidence, record.confidence)
            match.size = record.size
            match.priority_score = record.priority_score
            match.snapshot_path = record.snapshot_path or match.snapshot_path
        return None, True

    def reset(self) -> None:
        self._seen.clear()


@dataclass
class PipelineConfig:
    """Everything tunable about a pipeline run, in one place."""

    # Stage 2
    preprocess: PreprocessConfig = field(default_factory=lambda: DEFAULT_PREPROCESS_CONFIG)
    skip_blurred_frames: bool = True

    # Stage 3
    weights: str | Path | None = None
    confidence_threshold: float = 0.25
    device: str = "auto"
    allow_detector_fallback: bool = True

    # Stage 4
    severity: severity_mod.SeverityConfig = field(
        default_factory=lambda: severity_mod.DEFAULT_SEVERITY_CONFIG
    )
    calibration: dict | None = None

    # Priority
    road_type: str = "arterial"

    # Deduplication
    dedupe: bool = True
    dedupe_radius_m: float = 12.0
    dedupe_window_s: float = 20.0

    # Snapshots (stage 6 input)
    snapshot_dir: str | Path | None = "storage/snapshots"
    save_snapshots: bool = True
    snapshot_quality: int = 85


class CrackCatchPipeline:
    """Stages 1-5, as a stream of ``DefectRecord`` objects."""

    def __init__(
        self,
        config: PipelineConfig | None = None,
        detector: Detector | None = None,
        gps: GpsProvider | None = None,
        calibration: Calibration | None = None,
    ) -> None:
        self.config = config or PipelineConfig()
        self.detector = detector or build_detector(
            weights=self.config.weights,
            confidence_threshold=self.config.confidence_threshold,
            device=self.config.device,
            allow_fallback=self.config.allow_detector_fallback,
        )
        self.gps = gps or SimulatedRouteProvider()
        self.calibration = calibration or build_calibration(self.config.calibration)
        self.stats = PipelineStats()
        self._dedupe = (
            DefectDeduplicator(
                radius_m=self.config.dedupe_radius_m,
                window_s=self.config.dedupe_window_s,
            )
            if self.config.dedupe
            else None
        )
        self._snapshot_dir: Path | None = None
        if self.config.save_snapshots and self.config.snapshot_dir:
            self._snapshot_dir = Path(self.config.snapshot_dir)
            self._snapshot_dir.mkdir(parents=True, exist_ok=True)

    # ---------------------------------------------------------------- frames

    def process_frame(self, captured: CapturedFrame) -> list[DefectRecord]:
        """Run stages 2-5 on one captured frame."""
        config = self.config
        frame_started = time.perf_counter()

        prepared = preprocess(captured.image, config.preprocess)
        if config.skip_blurred_frames and prepared.blur < config.preprocess.blur_rejection_threshold:
            self.stats.frames_skipped_blur += 1
            return []

        # --- stage 3: detection ---
        inference_started = time.perf_counter()
        raw = self.detector.predict(prepared.image)
        self.stats.inference_time_s += time.perf_counter() - inference_started
        self.stats.frames_processed += 1
        self.stats.raw_detections += len(raw)

        if not raw:
            self.stats.total_time_s += time.perf_counter() - frame_started
            return []

        width, height = prepared.original_size
        records: list[DefectRecord] = []

        # --- stage 5: geo-tag once per frame, not once per detection ---
        location = self.gps.locate(
            video_time_s=captured.video_time_s, timestamp=captured.timestamp
        )
        if location is None:
            logger.debug(
                "No GPS fix for frame %s of %s; skipping (a defect without a "
                "location cannot be dispatched to a repair crew).",
                captured.index,
                captured.source_ref,
            )
            self.stats.total_time_s += time.perf_counter() - frame_started
            return []

        detected_at = self._frame_time(captured)

        for item in raw:
            # Map the detection back onto original-frame pixel coordinates.
            x1, y1, x2, y2 = prepared.to_original_bbox(*item.bbox.as_xyxy())
            bbox = BoundingBox(x1, y1, x2, y2).clip(width, height)
            if bbox.area < 1.0:
                continue

            # --- stage 4: size + severity ---
            size = self.calibration.estimate(bbox, width, height)
            sev, score, breakdown = severity_mod.classify(
                item.defect_class, bbox, item.confidence, size, config.severity
            )
            priority_score, _ = priority_mod.compute(
                sev, item.defect_class, config.road_type, detected_at
            )

            record = DefectRecord(
                defect_class=item.defect_class,
                severity=sev,
                confidence=item.confidence,
                bbox=bbox,
                location=location,
                detected_at=detected_at,
                size=size,
                severity_score=score,
                severity_breakdown=breakdown,
                priority_score=priority_score,
                source_type=captured.source_type,
                source_ref=captured.source_ref,
                frame_index=captured.index,
                road_type=config.road_type,
                notes=f"detector={item.detector}; label={item.raw_label}",
            )
            records.append(record)

        # --- deduplicate, then snapshot only what survives ---
        emitted: list[DefectRecord] = []
        for record in records:
            if self._dedupe is not None:
                keep, was_dup = self._dedupe.add(record)
                if was_dup:
                    self.stats.records_deduped += 1
                if keep is None:
                    continue
            if self._snapshot_dir is not None:
                record.snapshot_path = self._write_snapshot(
                    prepared.original, record, captured
                )
            emitted.append(record)
            self.stats.records_emitted += 1

        self.stats.total_time_s += time.perf_counter() - frame_started
        return emitted

    @staticmethod
    def _frame_time(captured: CapturedFrame) -> datetime:
        """Wall-clock time of a frame: clip start + offset into the clip."""
        base = captured.timestamp or datetime.now(timezone.utc)
        if base.tzinfo is None:
            base = base.replace(tzinfo=timezone.utc)
        if captured.video_time_s:
            return base + timedelta(seconds=captured.video_time_s)
        return base

    def _write_snapshot(
        self, frame: np.ndarray, record: DefectRecord, captured: CapturedFrame
    ) -> str | None:
        """Persist an annotated crop-in-context for the dashboard detail view."""
        assert self._snapshot_dir is not None
        from crackcatch_model.visualize import annotate

        try:
            annotated = annotate(frame, [record])
            filename = f"{record.client_id}.jpg"
            path = self._snapshot_dir / filename
            ok = cv2.imwrite(
                str(path),
                annotated,
                [int(cv2.IMWRITE_JPEG_QUALITY), self.config.snapshot_quality],
            )
            if not ok:
                logger.warning("cv2.imwrite failed for %s", path)
                return None
            return filename
        except Exception:
            logger.exception("Could not write snapshot for %s", record.client_id)
            return None

    # ------------------------------------------------------------------ runs

    def run(
        self,
        source: FrameSource,
        on_record: Callable[[DefectRecord], None] | None = None,
    ) -> Iterator[DefectRecord]:
        """Stream defect records from a capture source.

        ``on_record`` fires for each emitted record, which is how the backend
        pushes live updates to the dashboard over WebSocket while the run is
        still in progress.
        """
        started = time.perf_counter()
        try:
            for captured in source:
                self.stats.frames_read += 1
                for record in self.process_frame(captured):
                    if on_record is not None:
                        on_record(record)
                    yield record
        finally:
            source.close()
            wall = time.perf_counter() - started
            # total_time_s accumulates per-frame work; wall clock is the
            # honest end-to-end figure including I/O and decode.
            self.stats.total_time_s = max(self.stats.total_time_s, wall)
            logger.info("Pipeline finished: %s", self.stats.to_dict())

    def run_to_list(self, source: FrameSource) -> list[DefectRecord]:
        return list(self.run(source))
