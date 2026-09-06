"""Tests for pixel -> real-world size estimation (pipeline stage 4a)."""

from __future__ import annotations

import math

import pytest

from crackcatch_model.calibration import (
    GroundPlaneCalibration,
    NoCalibration,
    ReferenceObjectCalibration,
    build_calibration,
)
from crackcatch_model.types import BoundingBox

FRAME_W, FRAME_H = 1280, 720


class TestGroundPlaneGeometry:
    def test_rejects_impossible_parameters(self):
        with pytest.raises(ValueError):
            GroundPlaneCalibration(camera_height_m=0)
        with pytest.raises(ValueError):
            GroundPlaneCalibration(hfov_deg=0)
        with pytest.raises(ValueError):
            GroundPlaneCalibration(pitch_deg=95)

    def test_pixels_above_the_horizon_do_not_project(self):
        calibration = GroundPlaneCalibration()
        horizon = calibration.horizon_v(FRAME_W, FRAME_H)
        assert calibration.pixel_to_ground(640, horizon - 5, FRAME_W, FRAME_H) is None
        assert calibration.pixel_to_ground(640, 0, FRAME_W, FRAME_H) is None

    def test_lower_pixels_are_nearer_the_camera(self):
        calibration = GroundPlaneCalibration()
        near = calibration.pixel_to_ground(640, 700, FRAME_W, FRAME_H)
        far = calibration.pixel_to_ground(640, 450, FRAME_W, FRAME_H)
        assert near is not None and far is not None
        assert near[1] < far[1]

    def test_image_centre_column_has_no_lateral_offset(self):
        calibration = GroundPlaneCalibration()
        point = calibration.pixel_to_ground(FRAME_W / 2, 650, FRAME_W, FRAME_H)
        assert point is not None
        assert point[0] == pytest.approx(0.0, abs=1e-9)

    def test_lateral_offset_is_symmetric(self):
        calibration = GroundPlaneCalibration()
        left = calibration.pixel_to_ground(440, 650, FRAME_W, FRAME_H)
        right = calibration.pixel_to_ground(840, 650, FRAME_W, FRAME_H)
        assert left is not None and right is not None
        assert left[0] == pytest.approx(-right[0], abs=1e-9)

    def test_range_matches_closed_form_for_the_centre_column(self):
        """Cross-check the projection against the textbook formula.

        For a pixel on the centre column, the depression angle below the
        optical axis is ``atan((v - cy) / f)``; total depression from
        horizontal is that plus the camera pitch, and the ground range is
        ``h / tan(total)``.
        """
        calibration = GroundPlaneCalibration(camera_height_m=1.5, pitch_deg=10.0, hfov_deg=80.0)
        v = 620.0
        focal = (FRAME_W / 2) / math.tan(math.radians(80.0) / 2)
        depression = math.atan((v - FRAME_H / 2) / focal) + math.radians(10.0)
        expected = 1.5 / math.tan(depression)

        point = calibration.pixel_to_ground(FRAME_W / 2, v, FRAME_W, FRAME_H)
        assert point is not None
        assert point[1] == pytest.approx(expected, rel=1e-6)

    def test_taller_camera_sees_further_for_the_same_pixel(self):
        low = GroundPlaneCalibration(camera_height_m=1.0)
        high = GroundPlaneCalibration(camera_height_m=2.5)
        low_point = low.pixel_to_ground(640, 600, FRAME_W, FRAME_H)
        high_point = high.pixel_to_ground(640, 600, FRAME_W, FRAME_H)
        assert low_point is not None and high_point is not None
        assert high_point[1] > low_point[1]


class TestGroundPlaneEstimates:
    def test_estimate_produces_metric_size_for_a_near_box(self):
        calibration = GroundPlaneCalibration()
        estimate = calibration.estimate(BoundingBox(560, 630, 720, 700), FRAME_W, FRAME_H)
        assert estimate.method == "ground_plane"
        assert estimate.area_m2 is not None and estimate.area_m2 > 0
        assert estimate.reliable is True
        assert estimate.range_m is not None

    def test_box_above_the_horizon_degrades_to_pixels(self):
        calibration = GroundPlaneCalibration()
        estimate = calibration.estimate(BoundingBox(600, 10, 700, 60), FRAME_W, FRAME_H)
        assert estimate.area_m2 is None
        assert "horizon" in estimate.confidence_note

    def test_implausible_footprint_is_flagged_unreliable(self):
        """A box spanning most of the frame projects to an absurd ground
        area; that must be flagged, not reported as a measurement."""
        calibration = GroundPlaneCalibration()
        estimate = calibration.estimate(BoundingBox(60, 300, 1220, 700), FRAME_W, FRAME_H)
        if estimate.area_m2 is not None and estimate.area_m2 > calibration.max_plausible_area_m2:
            assert estimate.reliable is False
            assert "plausibility cap" in estimate.confidence_note

    def test_distant_box_is_flagged_beyond_reliable_range(self):
        calibration = GroundPlaneCalibration(max_reliable_range_m=8.0)
        # Bottom edge at row 340 sits ~11.8 m ahead for the default geometry.
        estimate = calibration.estimate(BoundingBox(600, 320, 680, 340), FRAME_W, FRAME_H)
        assert estimate.range_m is not None and estimate.range_m > 8.0
        assert estimate.reliable is False

    def test_area_frac_is_always_present(self):
        calibration = GroundPlaneCalibration()
        estimate = calibration.estimate(BoundingBox(0, 0, 128, 72), FRAME_W, FRAME_H)
        assert estimate.area_frac == pytest.approx(
            (128 * 72) / (FRAME_W * FRAME_H)
        )


class TestOtherStrategies:
    def test_no_calibration_reports_pixels_only(self):
        estimate = NoCalibration().estimate(BoundingBox(0, 0, 100, 50), FRAME_W, FRAME_H)
        assert estimate.method == "pixel_only"
        assert estimate.area_m2 is None
        assert estimate.area_px == 5000

    def test_reference_object_scale_is_linear(self):
        calibration = ReferenceObjectCalibration(
            reference_width_m=3.5, reference_width_px=700
        )
        assert calibration.metres_per_pixel == pytest.approx(0.005)
        estimate = calibration.estimate(BoundingBox(0, 0, 200, 100), FRAME_W, FRAME_H)
        assert estimate.width_m == pytest.approx(1.0)
        assert estimate.length_m == pytest.approx(0.5)
        assert estimate.area_m2 == pytest.approx(0.5)

    def test_reference_object_rejects_bad_input(self):
        with pytest.raises(ValueError):
            ReferenceObjectCalibration(reference_width_px=0)


class TestFactory:
    def test_builds_each_strategy(self):
        assert isinstance(build_calibration(None), NoCalibration)
        assert isinstance(
            build_calibration({"method": "ground_plane", "camera_height_m": 1.2}),
            GroundPlaneCalibration,
        )
        assert isinstance(
            build_calibration({"method": "reference_object"}), ReferenceObjectCalibration
        )

    def test_unknown_method_raises(self):
        with pytest.raises(ValueError, match="Unknown calibration method"):
            build_calibration({"method": "lidar"})
