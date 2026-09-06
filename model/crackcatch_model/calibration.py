"""Pixel -> real-world size estimation (first half of pipeline stage 4).

Two interchangeable strategies are provided, both deliberately simple and
fully documented so they can be defended in a viva:

``GroundPlaneCalibration``
    A flat-road pinhole-camera projection. Given the camera's mounting height
    and downward pitch, every pixel below the horizon is intersected with the
    road plane, which yields metric coordinates. This is the default for
    dashcam video.

``ReferenceObjectCalibration``
    A scale factor recovered from an object of known real width appearing in
    the frame (a lane marking is 3.5 m wide on an Indian arterial road). Used
    for single crowdsourced photos where camera geometry is unknown.

HONESTY NOTE (repeated in docs/SEVERITY.md and the README): both methods
recover *ground-plane extent*, i.e. how much road surface the defect covers.
Neither recovers **depth**. A 5 cm and a 25 cm deep pothole with the same
opening are indistinguishable to a single monocular camera. True depth needs
stereo, structured light or LiDAR. Everything below is an estimate.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

from crackcatch_model.types import BoundingBox, SizeEstimate


class Calibration:
    """Interface implemented by every calibration strategy."""

    method: str = "pixel_only"

    def estimate(self, bbox: BoundingBox, frame_w: int, frame_h: int) -> SizeEstimate:
        raise NotImplementedError


@dataclass
class NoCalibration(Calibration):
    """Fallback when camera geometry is unknown: pixel measures only."""

    method: str = "pixel_only"

    def estimate(self, bbox: BoundingBox, frame_w: int, frame_h: int) -> SizeEstimate:
        frame_area = max(float(frame_w * frame_h), 1.0)
        return SizeEstimate(
            area_px=bbox.area,
            area_frac=bbox.area / frame_area,
            method=self.method,
            confidence_note=(
                "No camera calibration supplied; severity falls back to the "
                "frame-relative bounding-box area."
            ),
        )


@dataclass
class GroundPlaneCalibration(Calibration):
    """Flat-road monocular projection.

    Coordinate frames
    -----------------
    Camera frame: ``x`` right, ``y`` down, ``z`` along the optical axis.
    World frame:  ``X`` right, ``Y`` up, ``Z`` horizontally forward, origin on
    the road surface directly below the camera.

    The camera sits at ``Y = camera_height_m`` and is pitched ``pitch_deg``
    downwards, so its basis vectors expressed in world coordinates are::

        x_c = ( 1,        0,        0       )
        y_c = ( 0, -cos(t),  -sin(t)        )
        z_c = ( 0, -sin(t),   cos(t)        )

    A pixel ``(u, v)`` back-projects to the camera ray ``(a, b, 1)`` with
    ``a = (u - cx) / fx`` and ``b = (v - cy) / fy``. Written in world
    coordinates that ray is ``d = a*x_c + b*y_c + z_c``, i.e.::

        d_X = a
        d_Y = -(b*cos(t) + sin(t))
        d_Z = cos(t) - b*sin(t)

    Intersecting ``camera + s*d`` with the road plane ``Y = 0`` gives
    ``s = camera_height / (b*cos(t) + sin(t))``. The denominator is the
    downward component of the ray: when it is <= 0 the pixel is at or above
    the horizon and no finite intersection exists.
    """

    camera_height_m: float = 1.35   # typical dashcam on a car windscreen
    pitch_deg: float = 8.0          # downward tilt from horizontal
    hfov_deg: float = 78.0          # horizontal field of view of the lens
    method: str = "ground_plane"

    #: Rays closer to the horizon than this are treated as non-intersecting.
    _MIN_DOWNWARD = 1e-3
    #: Ground points beyond this range are too foreshortened to project at all.
    max_range_m: float = 40.0
    #: Beyond this range the metric numbers are still computed but flagged
    #: unreliable. Rationale: ground sampling distance grows roughly with the
    #: square of range, so at 25 m a single pixel already spans several
    #: centimetres of road and a 2-3 px box error swings the area estimate by
    #: more than the defect's own size.
    max_reliable_range_m: float = 25.0
    #: A single pothole or crack footprint larger than this is physically
    #: implausible on a city road; treat it as a projection artefact.
    max_plausible_area_m2: float = 12.0

    def __post_init__(self) -> None:
        if self.camera_height_m <= 0:
            raise ValueError("camera_height_m must be positive")
        if not 0.0 < self.hfov_deg < 180.0:
            raise ValueError("hfov_deg must be in (0, 180)")
        if not -89.0 < self.pitch_deg < 89.0:
            raise ValueError("pitch_deg must be in (-89, 89)")

    def _intrinsics(self, frame_w: int, frame_h: int) -> tuple[float, float, float]:
        """Return ``(focal_px, cx, cy)`` assuming square pixels."""
        focal = (frame_w / 2.0) / math.tan(math.radians(self.hfov_deg) / 2.0)
        return focal, frame_w / 2.0, frame_h / 2.0

    def horizon_v(self, frame_w: int, frame_h: int) -> float:
        """Image row of the horizon. Pixels above it never meet the road."""
        focal, _, cy = self._intrinsics(frame_w, frame_h)
        # b*cos(t) + sin(t) = 0  ->  b = -tan(t)
        return cy - focal * math.tan(math.radians(self.pitch_deg))

    def pixel_to_ground(
        self, u: float, v: float, frame_w: int, frame_h: int
    ) -> tuple[float, float] | None:
        """Project pixel ``(u, v)`` onto the road plane.

        Returns ``(lateral_m, forward_m)`` or ``None`` if the pixel lies on or
        above the horizon, or beyond ``max_range_m``.
        """
        focal, cx, cy = self._intrinsics(frame_w, frame_h)
        a = (u - cx) / focal
        b = (v - cy) / focal
        t = math.radians(self.pitch_deg)

        downward = b * math.cos(t) + math.sin(t)
        if downward <= self._MIN_DOWNWARD:
            return None

        s = self.camera_height_m / downward
        forward = s * (math.cos(t) - b * math.sin(t))
        lateral = s * a
        if forward <= 0.0 or forward > self.max_range_m:
            return None
        return lateral, forward

    def estimate(self, bbox: BoundingBox, frame_w: int, frame_h: int) -> SizeEstimate:
        frame_area = max(float(frame_w * frame_h), 1.0)
        base = SizeEstimate(
            area_px=bbox.area,
            area_frac=bbox.area / frame_area,
            method=self.method,
        )

        # The bottom edge of the box is where the defect meets the road, so it
        # is the most reliable row to measure lateral width on.
        left = self.pixel_to_ground(bbox.x1, bbox.y2, frame_w, frame_h)
        right = self.pixel_to_ground(bbox.x2, bbox.y2, frame_w, frame_h)
        cu = bbox.centre[0]
        near = self.pixel_to_ground(cu, bbox.y2, frame_w, frame_h)
        far = self.pixel_to_ground(cu, bbox.y1, frame_w, frame_h)

        if left is None or right is None or near is None:
            return SizeEstimate(
                area_px=base.area_px,
                area_frac=base.area_frac,
                method=self.method,
                confidence_note=(
                    "Bounding box sits at or above the horizon, or beyond the "
                    f"{self.max_range_m:.0f} m trust range; metric size not "
                    "recoverable. Falling back to pixel-area severity."
                ),
            )

        width_m = abs(right[0] - left[0])
        range_m = near[1]

        if far is None:
            # Top edge above the horizon - the defect is heavily foreshortened.
            # Fall back to assuming a roughly circular/square footprint.
            length_m = width_m
            note = (
                "Top edge of the box is above the horizon; length assumed "
                "equal to width (near-circular footprint)."
            )
        else:
            length_m = abs(far[1] - near[1])
            note = (
                f"Flat-road projection at ~{range_m:.1f} m ahead. Ground-plane "
                "extent only - depth is NOT measured."
            )

        area_m2 = width_m * length_m
        reliable = True
        if range_m > self.max_reliable_range_m:
            reliable = False
            note += (
                f" Range {range_m:.1f} m exceeds the {self.max_reliable_range_m:.0f} m "
                "reliability limit; metric size shown for reference only and is "
                "not used for severity."
            )
        if area_m2 > self.max_plausible_area_m2:
            reliable = False
            note += (
                f" Footprint {area_m2:.1f} m2 exceeds the "
                f"{self.max_plausible_area_m2:.0f} m2 plausibility cap - most "
                "likely a perspective artefact from a box near the horizon; "
                "severity falls back to pixel area."
            )

        return SizeEstimate(
            area_px=base.area_px,
            area_frac=base.area_frac,
            width_m=round(width_m, 3),
            length_m=round(length_m, 3),
            area_m2=round(area_m2, 3),
            method=self.method,
            range_m=round(range_m, 2),
            reliable=reliable,
            confidence_note=note,
        )


@dataclass
class ReferenceObjectCalibration(Calibration):
    """Scale recovered from an object of known width in the image.

    ``reference_width_m`` is the true width of the reference feature and
    ``reference_width_px`` is how wide it appears. The resulting
    metres-per-pixel factor is only valid at the reference object's depth, so
    this is best for near-field crowdsourced photos taken looking down at a
    single pothole.
    """

    reference_width_m: float = 3.5     # Indian arterial lane width
    reference_width_px: float = 640.0
    method: str = "reference_object"

    def __post_init__(self) -> None:
        if self.reference_width_m <= 0 or self.reference_width_px <= 0:
            raise ValueError("reference dimensions must be positive")

    @property
    def metres_per_pixel(self) -> float:
        return self.reference_width_m / self.reference_width_px

    def estimate(self, bbox: BoundingBox, frame_w: int, frame_h: int) -> SizeEstimate:
        frame_area = max(float(frame_w * frame_h), 1.0)
        mpp = self.metres_per_pixel
        width_m = bbox.width * mpp
        length_m = bbox.height * mpp
        return SizeEstimate(
            area_px=bbox.area,
            area_frac=bbox.area / frame_area,
            width_m=round(width_m, 3),
            length_m=round(length_m, 3),
            area_m2=round(width_m * length_m, 3),
            method=self.method,
            confidence_note=(
                f"Uniform scale {mpp * 100:.2f} cm/px from a "
                f"{self.reference_width_m} m reference. Accurate only at the "
                "reference object's depth; no perspective correction applied."
            ),
        )


def build_calibration(spec: dict | None) -> Calibration:
    """Construct a calibration from a plain config dict.

    ``{"method": "ground_plane", "camera_height_m": 1.35, ...}``
    """
    if not spec:
        return NoCalibration()
    spec = dict(spec)
    method = spec.pop("method", "ground_plane")
    if method == "ground_plane":
        return GroundPlaneCalibration(**spec)
    if method == "reference_object":
        return ReferenceObjectCalibration(**spec)
    if method == "pixel_only":
        return NoCalibration()
    raise ValueError(f"Unknown calibration method: {method!r}")
