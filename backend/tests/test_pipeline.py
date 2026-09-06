"""Tests for pipeline stages 1, 2, 3 and 5, and the orchestrator's
deduplication and throughput accounting."""

from __future__ import annotations

from datetime import datetime, timezone

import cv2
import numpy as np
import pytest

from crackcatch_model.detector import (
    HeuristicDetector,
    RawDetection,
    build_detector,
    non_max_suppression,
)
from crackcatch_model.geotag import (
    DeviceGpsProvider,
    GpxTrackProvider,
    SimulatedRouteProvider,
    build_provider,
)
from crackcatch_model.pipeline import (
    CrackCatchPipeline,
    DefectDeduplicator,
    PipelineConfig,
)
from crackcatch_model.preprocess import (
    PreprocessConfig,
    blur_score,
    letterbox_resize,
    normalise_lighting,
    preprocess,
    undo_letterbox,
)
from crackcatch_model.sources import (
    ImageDirectorySource,
    SingleImageSource,
    open_source,
)
from crackcatch_model.types import (
    BoundingBox,
    DefectClass,
    DefectRecord,
    GeoPoint,
    Severity,
)


def road_frame(width: int = 1280, height: int = 720, potholes: int = 1) -> np.ndarray:
    """A synthetic road frame with dark blobs in the lower (road) region."""
    rng = np.random.default_rng(11)
    frame = np.full((height, width, 3), 108, dtype=np.uint8)
    noise = rng.integers(-10, 10, frame.shape, dtype=np.int16)
    frame = np.clip(frame.astype(np.int16) + noise, 0, 255).astype(np.uint8)
    for i in range(potholes):
        cx = width // 2 + (i - potholes // 2) * 220
        cv2.ellipse(frame, (cx, int(height * 0.82)), (70, 46), 0, 0, 360, (22, 24, 28), -1)
    return frame


class TestPreprocess:
    def test_output_is_detector_sized(self):
        result = preprocess(road_frame())
        assert result.image.shape[:2] == (640, 640)
        assert result.original_size == (1280, 720)

    def test_letterbox_round_trip_is_exact(self):
        _, scale, pad = letterbox_resize(road_frame(), (640, 640))
        original = (100.0, 200.0, 300.0, 400.0)
        boxed = (
            original[0] * scale + pad[0],
            original[1] * scale + pad[1],
            original[2] * scale + pad[0],
            original[3] * scale + pad[1],
        )
        assert undo_letterbox(*boxed, scale, pad) == pytest.approx(original)

    def test_letterbox_preserves_aspect_ratio(self):
        image, scale, pad = letterbox_resize(np.zeros((360, 1280, 3), np.uint8), (640, 640))
        assert image.shape[:2] == (640, 640)
        assert pad[1] > 0 and pad[0] == 0  # padded top/bottom, not left/right

    def test_clahe_increases_contrast_in_a_shadowed_frame(self):
        """Stage 2's reason for existing: recover detail in shade."""
        dark = np.full((240, 320, 3), 40, dtype=np.uint8)
        dark[:, 160:] = 205  # half deep shadow, half bright
        improved = normalise_lighting(dark)
        assert improved.shape == dark.shape
        # The shadowed half must not stay uniformly crushed.
        assert improved[:, :160].std() >= dark[:, :160].std()

    def test_greyscale_input_is_promoted_to_colour(self):
        result = preprocess(np.full((200, 200), 120, dtype=np.uint8))
        assert result.image.ndim == 3

    def test_empty_frame_is_rejected(self):
        with pytest.raises(ValueError, match="empty frame"):
            preprocess(np.zeros((0, 0, 3), dtype=np.uint8))

    def test_blur_score_separates_sharp_from_blurred(self):
        sharp = road_frame()
        blurred = cv2.GaussianBlur(sharp, (31, 31), 0)
        assert blur_score(sharp) > blur_score(blurred)

    def test_stretch_resize_mode(self):
        config = PreprocessConfig(letterbox=False, target_size=(320, 320))
        result = preprocess(road_frame(), config)
        assert result.image.shape[:2] == (320, 320)


class TestSources:
    def test_single_image_source(self, tmp_path):
        path = tmp_path / "road.png"
        cv2.imwrite(str(path), road_frame(320, 240))
        frames = list(SingleImageSource(path=path, source_ref="road.png"))
        assert len(frames) == 1
        assert frames[0].source_type == "crowdsource"
        assert frames[0].size == (320, 240)

    def test_single_image_from_array(self):
        frames = list(SingleImageSource(image=road_frame(200, 200)))
        assert len(frames) == 1

    def test_single_image_needs_a_source(self):
        with pytest.raises(ValueError, match="needs either"):
            list(SingleImageSource())

    def test_directory_source_walks_images(self, tmp_path):
        for i in range(3):
            cv2.imwrite(str(tmp_path / f"img{i}.png"), road_frame(160, 120))
        (tmp_path / "notes.txt").write_text("ignored")
        source = ImageDirectorySource(tmp_path)
        assert source.count() == 3
        assert len(list(source)) == 3

    def test_directory_source_honours_max_frames(self, tmp_path):
        for i in range(5):
            cv2.imwrite(str(tmp_path / f"img{i}.png"), road_frame(160, 120))
        assert len(list(ImageDirectorySource(tmp_path, max_frames=2))) == 2

    def test_directory_source_skips_corrupt_files(self, tmp_path):
        cv2.imwrite(str(tmp_path / "good.png"), road_frame(160, 120))
        (tmp_path / "broken.png").write_bytes(b"not a png")
        assert len(list(ImageDirectorySource(tmp_path))) == 1

    def test_open_source_dispatch(self, tmp_path):
        image = tmp_path / "a.png"
        cv2.imwrite(str(image), road_frame(64, 64))
        assert isinstance(open_source(str(image)), SingleImageSource)
        assert isinstance(open_source(str(tmp_path)), ImageDirectorySource)

    def test_open_source_rejects_unknown(self, tmp_path):
        bad = tmp_path / "data.bin"
        bad.write_bytes(b"x")
        with pytest.raises(ValueError, match="Unsupported input type"):
            open_source(str(bad))

    def test_open_source_missing_file(self):
        with pytest.raises(FileNotFoundError):
            open_source("/nonexistent/clip.mp4")


class TestDetector:
    @pytest.fixture
    def no_checkpoints(self, tmp_path, monkeypatch):
        """Guarantee that no trained checkpoint is discoverable.

        ``build_detector`` also probes the default weights path, so without
        this these tests would pass or fail depending on whether the developer
        happens to have trained a model - the outcome must not depend on that.
        """
        monkeypatch.setattr(
            "crackcatch_model.detector.DEFAULT_WEIGHTS", tmp_path / "no_such.pt"
        )
        return tmp_path

    def test_fallback_is_used_when_no_weights_exist(self, no_checkpoints):
        detector = build_detector(
            weights=no_checkpoints / "absent.pt", allow_fallback=True
        )
        assert detector.name == "heuristic"

    def test_fallback_can_be_refused(self, no_checkpoints):
        with pytest.raises(FileNotFoundError, match="No YOLO weights found"):
            build_detector(
                weights=no_checkpoints / "absent.pt", allow_fallback=False
            )

    def test_heuristic_detector_finds_a_dark_blob(self):
        detections = HeuristicDetector().predict(road_frame(potholes=1))
        assert len(detections) >= 1
        assert all(d.detector == "heuristic" for d in detections)
        assert all(0.0 <= d.confidence <= 1.0 for d in detections)

    def test_heuristic_detector_ignores_the_sky_region(self):
        """Detections must sit in the road ROI, not the upper frame."""
        frame = road_frame(potholes=1)
        cv2.ellipse(frame, (200, 60), (60, 40), 0, 0, 360, (20, 20, 20), -1)
        detector = HeuristicDetector()
        for detection in detector.predict(frame):
            assert detection.bbox.y1 >= frame.shape[0] * detector.road_roi - 1

    def test_featureless_frame_yields_nothing(self):
        assert HeuristicDetector().predict(np.full((480, 640, 3), 128, np.uint8)) == []

    def test_empty_input_is_safe(self):
        assert HeuristicDetector().predict(np.zeros((0, 0, 3), np.uint8)) == []

    def test_nms_drops_overlapping_boxes(self):
        detections = [
            RawDetection(DefectClass.POTHOLE, 0.9, BoundingBox(0, 0, 100, 100)),
            RawDetection(DefectClass.POTHOLE, 0.7, BoundingBox(5, 5, 105, 105)),
            RawDetection(DefectClass.POTHOLE, 0.8, BoundingBox(400, 400, 500, 500)),
        ]
        kept = non_max_suppression(detections, 0.45)
        assert len(kept) == 2
        assert kept[0].confidence == 0.9  # the strongest of the overlapping pair

    def test_benchmark_reports_fps(self):
        stats = HeuristicDetector().benchmark(road_frame(320, 240), runs=3)
        assert stats["fps"] > 0 and stats["ms_per_frame"] > 0


class TestGeotagging:
    def test_device_provider_returns_its_point(self):
        point = DeviceGpsProvider(19.07, 72.87, 5.0).locate()
        assert point.latitude == 19.07 and point.source == "device"

    def test_simulated_route_advances_with_time(self):
        provider = SimulatedRouteProvider(speed_kmph=36.0, jitter_m=0.0)
        start, later = provider.locate(0.0), provider.locate(10.0)
        # 36 km/h = 10 m/s, so 10 s should be ~100 m.
        assert start.distance_m(later) == pytest.approx(100.0, rel=0.02)

    def test_simulated_points_are_tagged_simulated(self):
        assert SimulatedRouteProvider().locate(1.0).source == "simulated"

    def test_simulated_route_is_reproducible(self):
        a = SimulatedRouteProvider(seed=5).locate(3.0)
        b = SimulatedRouteProvider(seed=5).locate(3.0)
        assert (a.latitude, a.longitude) == (b.latitude, b.longitude)

    def test_gpx_interpolates_between_track_points(self, tmp_path):
        gpx = tmp_path / "drive.gpx"
        gpx.write_text(
            '<?xml version="1.0"?>'
            '<gpx xmlns="http://www.topografix.com/GPX/1/1"><trk><trkseg>'
            '<trkpt lat="19.00" lon="72.00"><time>2024-01-01T00:00:00Z</time></trkpt>'
            '<trkpt lat="19.02" lon="72.02"><time>2024-01-01T00:00:10Z</time></trkpt>'
            "</trkseg></trk></gpx>"
        )
        provider = GpxTrackProvider.from_file(gpx)
        midpoint = provider.locate(5.0)
        assert midpoint.latitude == pytest.approx(19.01)
        assert midpoint.longitude == pytest.approx(72.01)

    def test_gpx_clamps_outside_the_track(self, tmp_path):
        gpx = tmp_path / "d.gpx"
        gpx.write_text(
            '<?xml version="1.0"?>'
            '<gpx xmlns="http://www.topografix.com/GPX/1/1"><trk><trkseg>'
            '<trkpt lat="19.00" lon="72.00"><time>2024-01-01T00:00:00Z</time></trkpt>'
            '<trkpt lat="19.02" lon="72.02"><time>2024-01-01T00:00:10Z</time></trkpt>'
            "</trkseg></trk></gpx>"
        )
        provider = GpxTrackProvider.from_file(gpx)
        assert provider.locate(-5).latitude == pytest.approx(19.00)
        assert provider.locate(999).latitude == pytest.approx(19.02)

    def test_gpx_without_trackpoints_raises(self, tmp_path):
        gpx = tmp_path / "empty.gpx"
        gpx.write_text('<?xml version="1.0"?><gpx></gpx>')
        with pytest.raises(ValueError, match="No <trkpt>"):
            GpxTrackProvider.from_file(gpx)

    def test_provider_factory(self):
        assert isinstance(build_provider(None), SimulatedRouteProvider)
        assert isinstance(
            build_provider({"method": "device", "latitude": 1.0, "longitude": 2.0}),
            DeviceGpsProvider,
        )
        with pytest.raises(ValueError, match="Unknown GPS provider"):
            build_provider({"method": "telepathy"})

    def test_coordinates_are_validated(self):
        with pytest.raises(ValueError, match="latitude out of range"):
            GeoPoint(latitude=91.0, longitude=0.0)


class TestDeduplicator:
    def _record(self, lat: float, lon: float, seconds: int = 0, score: float = 0.5):
        return DefectRecord(
            defect_class=DefectClass.POTHOLE,
            severity=Severity.MODERATE,
            confidence=0.8,
            bbox=BoundingBox(0, 0, 10, 10),
            location=GeoPoint(lat, lon),
            detected_at=datetime(2026, 1, 1, 0, 0, seconds, tzinfo=timezone.utc),
            severity_score=score,
        )

    def test_nearby_sightings_collapse_to_one(self):
        dedupe = DefectDeduplicator(radius_m=12.0, window_s=20.0)
        first, was_dup = dedupe.add(self._record(19.0, 72.0, 0))
        assert first is not None and was_dup is False
        second, was_dup = dedupe.add(self._record(19.00001, 72.00001, 2))
        assert second is None and was_dup is True

    def test_distant_defects_are_distinct(self):
        dedupe = DefectDeduplicator(radius_m=12.0)
        dedupe.add(self._record(19.0, 72.0, 0))
        emitted, was_dup = dedupe.add(self._record(19.01, 72.0, 2))
        assert emitted is not None and was_dup is False

    def test_different_classes_are_not_merged(self):
        dedupe = DefectDeduplicator()
        dedupe.add(self._record(19.0, 72.0, 0))
        crack = self._record(19.0, 72.0, 1)
        crack.defect_class = DefectClass.CRACK
        emitted, was_dup = dedupe.add(crack)
        assert emitted is not None and was_dup is False

    def test_time_window_expires(self):
        dedupe = DefectDeduplicator(radius_m=12.0, window_s=5.0)
        dedupe.add(self._record(19.0, 72.0, 0))
        emitted, was_dup = dedupe.add(self._record(19.0, 72.0, 30))
        assert emitted is not None and was_dup is False

    def test_worse_sighting_upgrades_the_kept_record(self):
        """Seen again from closer up and scored worse - keep the worse one."""
        dedupe = DefectDeduplicator()
        kept, _ = dedupe.add(self._record(19.0, 72.0, 0, score=0.30))
        worse = self._record(19.0, 72.0, 2, score=0.90)
        worse.severity = Severity.SEVERE
        emitted, was_dup = dedupe.add(worse)
        assert emitted is None and was_dup is True
        assert kept.severity_score == 0.90
        assert kept.severity is Severity.SEVERE


class TestPipelineEndToEnd:
    def test_stages_1_to_5_produce_records(self, tmp_path):
        for i in range(3):
            cv2.imwrite(str(tmp_path / f"frame{i}.png"), road_frame(potholes=2))

        pipeline = CrackCatchPipeline(
            PipelineConfig(
                calibration={"method": "ground_plane"},
                snapshot_dir=tmp_path / "snaps",
                dedupe=False,
            )
        )
        records = pipeline.run_to_list(ImageDirectorySource(tmp_path))

        assert records, "the CV baseline should find the synthetic potholes"
        for record in records:
            assert record.severity in set(Severity)
            assert record.location is not None
            assert 0.0 <= record.severity_score <= 1.0
            assert 0.0 <= record.priority_score <= 100.0
            assert record.snapshot_path is not None
            assert (tmp_path / "snaps" / record.snapshot_path).exists()

    def test_records_serialise_for_the_api(self, tmp_path):
        cv2.imwrite(str(tmp_path / "f.png"), road_frame(potholes=1))
        pipeline = CrackCatchPipeline(
            PipelineConfig(snapshot_dir=tmp_path / "s", calibration={"method": "ground_plane"})
        )
        records = pipeline.run_to_list(ImageDirectorySource(tmp_path))
        payload = records[0].to_dict()
        # Non-numeric provenance in the breakdown must not break serialisation.
        assert payload["severity_breakdown"]["_size_basis"]
        assert isinstance(payload["detected_at"], str)
        assert set(payload["bbox"]) == {"x1", "y1", "x2", "y2"}

    def test_stats_are_measured_not_claimed(self, tmp_path):
        for i in range(2):
            cv2.imwrite(str(tmp_path / f"f{i}.png"), road_frame(potholes=1))
        pipeline = CrackCatchPipeline(PipelineConfig(save_snapshots=False))
        pipeline.run_to_list(ImageDirectorySource(tmp_path))
        stats = pipeline.stats.to_dict()
        assert stats["frames_read"] == 2
        assert stats["frames_processed"] == 2
        assert stats["inference_fps"] > 0
        assert stats["end_to_end_fps"] > 0

    def test_pipeline_handles_a_frame_with_no_detections(self, tmp_path):
        cv2.imwrite(str(tmp_path / "flat.png"), np.full((480, 640, 3), 128, np.uint8))
        pipeline = CrackCatchPipeline(PipelineConfig(save_snapshots=False))
        assert pipeline.run_to_list(ImageDirectorySource(tmp_path)) == []


class TestExplainability:
    """The explainer must report what it actually produced.

    An earlier version set the method label from "does a checkpoint exist",
    so a Grad-CAM that silently fell back still claimed to be Grad-CAM. The
    label is the only thing telling a viewer whether they are looking at real
    gradient evidence, so it has to be derived from the return value.
    """

    def test_gradcam_reports_its_method(self, tmp_path):
        from crackcatch_model.explain import gradcam

        frame = road_frame(640, 480, potholes=1)
        # A path with no checkpoint must fall back AND say so.
        _, method = gradcam(frame, str(tmp_path / "missing.pt"))
        assert "NOT Grad-CAM" in method

    def test_gradcam_returns_an_image_and_a_label(self, tmp_path):
        from crackcatch_model.explain import gradcam

        frame = road_frame(640, 480, potholes=1)
        overlay, method = gradcam(frame, str(tmp_path / "missing.pt"))
        assert overlay.shape == frame.shape
        assert isinstance(method, str) and method

    def test_saliency_overlay_preserves_shape(self):
        from crackcatch_model.explain import saliency_overlay
        from crackcatch_model.types import BoundingBox

        frame = road_frame(640, 480, potholes=1)
        assert saliency_overlay(frame, BoundingBox(100, 300, 260, 420)).shape == frame.shape
        assert saliency_overlay(frame, None).shape == frame.shape
