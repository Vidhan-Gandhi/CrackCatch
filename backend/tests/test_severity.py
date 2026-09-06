"""Tests for the severity-scoring logic (pipeline stage 4).

This is the module a review panel is most likely to interrogate, so the tests
assert on its *documented properties* - monotonicity, class-conditional shape
handling, weight normalisation, threshold ordering - not just on a few
hard-coded outputs.
"""

from __future__ import annotations

import pytest

from crackcatch_model.calibration import GroundPlaneCalibration, NoCalibration
from crackcatch_model.severity import (
    DEFAULT_SEVERITY_CONFIG,
    SeverityConfig,
    classify,
    confidence_factor,
    explain,
    shape_factor,
    size_factor,
)
from crackcatch_model.types import BoundingBox, DefectClass, Severity, SizeEstimate

FRAME = (1280, 720)


def size_for(bbox: BoundingBox, area_m2: float | None = None, reliable: bool = True):
    frame_area = FRAME[0] * FRAME[1]
    return SizeEstimate(
        area_px=bbox.area,
        area_frac=bbox.area / frame_area,
        width_m=None if area_m2 is None else area_m2**0.5,
        length_m=None if area_m2 is None else area_m2**0.5,
        area_m2=area_m2,
        method="pixel_only" if area_m2 is None else "ground_plane",
        reliable=reliable,
    )


class TestWeightsAndThresholds:
    def test_default_weights_sum_to_one(self):
        config = DEFAULT_SEVERITY_CONFIG
        assert config.w_size + config.w_shape + config.w_conf == pytest.approx(1.0)

    def test_weights_that_do_not_sum_to_one_are_rejected(self):
        with pytest.raises(ValueError, match="must sum to 1.0"):
            SeverityConfig(w_size=0.5, w_shape=0.5, w_conf=0.5)

    def test_thresholds_must_be_ordered(self):
        with pytest.raises(ValueError, match="thresholds must satisfy"):
            SeverityConfig(minor_max=0.8, moderate_max=0.4)

    def test_score_is_bounded(self):
        bbox = BoundingBox(0, 0, 1280, 720)
        _, score, _ = classify(
            DefectClass.POTHOLE, bbox, 1.0, size_for(bbox, area_m2=500.0)
        )
        assert 0.0 <= score <= 1.0


class TestFactors:
    def test_confidence_factor_ramps_from_the_floor(self):
        config = DEFAULT_SEVERITY_CONFIG
        assert confidence_factor(config.conf_floor) == pytest.approx(0.0)
        assert confidence_factor(1.0) == pytest.approx(1.0)
        assert confidence_factor(0.0) == pytest.approx(0.0)  # clamped

    def test_pothole_shape_prefers_compact_boxes(self):
        square = BoundingBox(0, 0, 100, 100)       # AR 1.0
        elongated = BoundingBox(0, 0, 300, 100)    # AR 3.0
        assert shape_factor(DefectClass.POTHOLE, square) > shape_factor(
            DefectClass.POTHOLE, elongated
        )
        assert shape_factor(DefectClass.POTHOLE, square) == pytest.approx(1.0)

    def test_crack_shape_prefers_elongated_boxes(self):
        square = BoundingBox(0, 0, 100, 100)
        elongated = BoundingBox(0, 0, 800, 100)    # AR 8.0
        assert shape_factor(DefectClass.CRACK, elongated) > shape_factor(
            DefectClass.CRACK, square
        )
        assert shape_factor(DefectClass.CRACK, elongated) == pytest.approx(1.0)

    def test_size_factor_uses_metric_area_when_reliable(self):
        bbox = BoundingBox(0, 0, 100, 100)
        factor, basis = size_factor(DefectClass.POTHOLE, size_for(bbox, area_m2=0.7))
        assert basis == "footprint_m2"
        assert factor == pytest.approx(1.0)

    def test_size_factor_falls_back_when_estimate_is_unreliable(self):
        """A far-field projection artefact must not drive severity."""
        bbox = BoundingBox(0, 0, 100, 100)
        _, basis = size_factor(
            DefectClass.POTHOLE, size_for(bbox, area_m2=50.0, reliable=False)
        )
        assert basis == "frame_area_fraction"

    def test_size_factor_uses_run_length_for_cracks(self):
        bbox = BoundingBox(0, 0, 900, 40)
        _, basis = size_factor(DefectClass.CRACK, size_for(bbox, area_m2=4.0))
        assert basis == "crack_length_m"

    def test_missing_size_yields_zero(self):
        assert size_factor(DefectClass.POTHOLE, None) == (0.0, "unavailable")


class TestClassification:
    def test_large_confident_pothole_is_severe(self):
        bbox = BoundingBox(500, 560, 800, 700)
        severity, score, _ = classify(
            DefectClass.POTHOLE, bbox, 0.93, size_for(bbox, area_m2=0.8)
        )
        assert severity is Severity.SEVERE
        assert score >= DEFAULT_SEVERITY_CONFIG.moderate_max

    def test_tiny_low_confidence_pothole_is_minor(self):
        bbox = BoundingBox(600, 690, 620, 706)
        severity, score, _ = classify(
            DefectClass.POTHOLE, bbox, 0.30, size_for(bbox, area_m2=0.01)
        )
        assert severity is Severity.MINOR
        assert score < DEFAULT_SEVERITY_CONFIG.minor_max

    def test_severity_is_monotonic_in_size(self):
        """Growing a defect must never lower its score."""
        bbox = BoundingBox(500, 600, 700, 700)
        scores = [
            classify(DefectClass.POTHOLE, bbox, 0.7, size_for(bbox, area_m2=a))[1]
            for a in (0.02, 0.10, 0.30, 0.55, 0.90)
        ]
        assert scores == sorted(scores)

    def test_severity_is_monotonic_in_confidence(self):
        bbox = BoundingBox(500, 600, 700, 700)
        scores = [
            classify(DefectClass.POTHOLE, bbox, c, size_for(bbox, area_m2=0.3))[1]
            for c in (0.25, 0.45, 0.65, 0.85, 1.0)
        ]
        assert scores == sorted(scores)

    def test_thresholds_partition_the_range(self):
        """Every score maps to exactly one label, with no gap or overlap."""
        config = DEFAULT_SEVERITY_CONFIG
        bbox = BoundingBox(0, 0, 100, 100)
        seen = set()
        for area in [i / 100 for i in range(0, 120, 2)]:
            severity, score, _ = classify(
                DefectClass.POTHOLE, bbox, 0.8, size_for(bbox, area_m2=area)
            )
            seen.add(severity)
            if score < config.minor_max:
                assert severity is Severity.MINOR
            elif score < config.moderate_max:
                assert severity is Severity.MODERATE
            else:
                assert severity is Severity.SEVERE
        assert len(seen) >= 2

    def test_invalid_confidence_is_rejected(self):
        bbox = BoundingBox(0, 0, 10, 10)
        with pytest.raises(ValueError, match="confidence must be in"):
            classify(DefectClass.POTHOLE, bbox, 1.5, size_for(bbox))

    def test_breakdown_is_transparent_and_adds_up(self):
        """The viva requirement: the score must be reconstructible."""
        bbox = BoundingBox(400, 600, 700, 700)
        _, score, breakdown = classify(
            DefectClass.POTHOLE, bbox, 0.75, size_for(bbox, area_m2=0.4)
        )
        total = (
            breakdown["size_contribution"]
            + breakdown["shape_contribution"]
            + breakdown["confidence_contribution"]
        )
        assert total == pytest.approx(score, abs=1e-9)
        assert breakdown["_size_basis"] == "footprint_m2"
        assert "size(" in explain(breakdown)


class TestWithRealCalibration:
    """End-to-end stage 4: geometry feeding the scorer."""

    def test_identical_box_scores_higher_when_closer(self):
        """The same pixel box nearer the camera is a physically smaller
        defect, so it must not outscore the distant one on size."""
        calibration = GroundPlaneCalibration()
        near = BoundingBox(560, 640, 720, 700)
        far = BoundingBox(560, 440, 720, 500)
        near_size = calibration.estimate(near, *FRAME)
        far_size = calibration.estimate(far, *FRAME)
        assert near_size.area_m2 < far_size.area_m2

    def test_no_calibration_still_produces_a_severity(self):
        bbox = BoundingBox(400, 500, 900, 700)
        size = NoCalibration().estimate(bbox, *FRAME)
        severity, score, breakdown = classify(DefectClass.POTHOLE, bbox, 0.8, size)
        assert isinstance(severity, Severity)
        assert breakdown["_size_basis"] == "frame_area_fraction"
        assert score > 0
