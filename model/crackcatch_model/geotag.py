"""GPS geo-tagging - pipeline stage 5.

Attaches a coordinate + timestamp to every confirmed defect. Four providers,
matching the four capture sources:

``DeviceGpsProvider``    live coordinates pushed by the mobile app / dashcam
``ExifGpsProvider``      coordinates embedded in a crowdsourced photo
``GpxTrackProvider``     a recorded drive track, interpolated by video time
``SimulatedRouteProvider`` a synthetic route for demo videos with no GPS

DATA GOVERNANCE (non-functional requirement): a provider is only ever asked
for a coordinate *at the moment a defect is confirmed*. The pipeline never
samples or persists a continuous position trail, so what reaches the database
is a set of discrete defect points, not a reconstructable journey. A
production deployment would additionally need spatial aggregation or
k-anonymity before exposing points publicly - see docs/DATA_GOVERNANCE.md.
"""

from __future__ import annotations

import logging
import math
import xml.etree.ElementTree as ET
from bisect import bisect_left
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

from crackcatch_model.types import GeoPoint

logger = logging.getLogger(__name__)

#: Metres per degree of latitude (spherical approximation).
_M_PER_DEG_LAT = 111_320.0


class GpsProvider:
    """Interface: given a frame's time offset, return a coordinate."""

    source: str = "unknown"

    def locate(
        self, video_time_s: float | None = None, timestamp: datetime | None = None
    ) -> GeoPoint | None:
        raise NotImplementedError


@dataclass
class DeviceGpsProvider(GpsProvider):
    """Coordinates supplied by the capturing device (phone or dashcam).

    Used by the crowdsourcing PWA, which sends the browser Geolocation
    reading alongside the photo.
    """

    latitude: float
    longitude: float
    accuracy_m: float | None = None
    source: str = "device"

    def locate(
        self, video_time_s: float | None = None, timestamp: datetime | None = None
    ) -> GeoPoint | None:
        return GeoPoint(
            latitude=self.latitude,
            longitude=self.longitude,
            accuracy_m=self.accuracy_m,
            source=self.source,
        )


@dataclass
class StaticGpsProvider(DeviceGpsProvider):
    """A fixed coordinate, e.g. for a batch of images from one location."""

    source: str = "static"


def _dms_to_degrees(dms: tuple, ref: str) -> float:
    """Convert EXIF degrees/minutes/seconds rationals to signed degrees."""

    def _rational(value) -> float:
        if isinstance(value, (tuple, list)) and len(value) == 2:
            numerator, denominator = value
            return float(numerator) / float(denominator or 1)
        return float(value)

    degrees, minutes, seconds = (_rational(v) for v in dms)
    result = degrees + minutes / 60.0 + seconds / 3600.0
    if str(ref).upper() in {"S", "W"}:
        result = -result
    return result


@dataclass
class ExifGpsProvider(GpsProvider):
    """Read GPS out of a photo's EXIF header.

    ``fallback`` is used when the image carries no GPS - most Android/iOS
    photos do when location permission was granted, but a WhatsApp-forwarded
    image has been stripped.
    """

    path: str | Path
    fallback: GpsProvider | None = None
    source: str = "exif"

    def __post_init__(self) -> None:
        self._point: GeoPoint | None = None
        self._loaded = False

    @staticmethod
    def read(path: str | Path) -> GeoPoint | None:
        """Extract a ``GeoPoint`` from an image's EXIF, or ``None``."""
        try:
            from PIL import Image, ExifTags
        except ImportError:  # pragma: no cover
            logger.warning("Pillow not installed; cannot read EXIF GPS")
            return None

        try:
            with Image.open(str(path)) as image:
                exif = image.getexif()
                if not exif:
                    return None
                gps_ifd = exif.get_ifd(ExifTags.IFD.GPSInfo)
        except Exception:
            logger.debug("EXIF read failed for %s", path, exc_info=True)
            return None

        if not gps_ifd:
            return None

        tags = {ExifTags.GPSTAGS.get(k, k): v for k, v in gps_ifd.items()}
        lat, lat_ref = tags.get("GPSLatitude"), tags.get("GPSLatitudeRef")
        lon, lon_ref = tags.get("GPSLongitude"), tags.get("GPSLongitudeRef")
        if not (lat and lon and lat_ref and lon_ref):
            return None

        try:
            return GeoPoint(
                latitude=_dms_to_degrees(lat, lat_ref),
                longitude=_dms_to_degrees(lon, lon_ref),
                accuracy_m=None,
                source="exif",
            )
        except (ValueError, TypeError, ZeroDivisionError):
            logger.debug("Malformed EXIF GPS in %s", path, exc_info=True)
            return None

    def locate(
        self, video_time_s: float | None = None, timestamp: datetime | None = None
    ) -> GeoPoint | None:
        if not self._loaded:
            self._point = self.read(self.path)
            self._loaded = True
        if self._point is not None:
            return self._point
        if self.fallback is not None:
            return self.fallback.locate(video_time_s, timestamp)
        return None


@dataclass
class GpxTrackProvider(GpsProvider):
    """A real recorded drive track, interpolated to the frame's time offset.

    This is the honest path for a demo video: record a GPX on your phone while
    filming, and every detection lands on the road you actually drove.
    """

    points: list[tuple[float, float, float]] = field(default_factory=list)
    source: str = "gpx"

    def __post_init__(self) -> None:
        # (elapsed_s, lat, lon), sorted by elapsed time.
        self.points = sorted(self.points, key=lambda p: p[0])
        self._times = [p[0] for p in self.points]

    @classmethod
    def from_file(cls, path: str | Path) -> "GpxTrackProvider":
        """Parse a standard GPX 1.1 ``<trkpt>`` sequence."""
        tree = ET.parse(str(path))
        namespace = {"gpx": "http://www.topografix.com/GPX/1/1"}
        track_points = tree.getroot().findall(".//gpx:trkpt", namespace)
        if not track_points:
            # Some exporters omit the namespace entirely.
            track_points = tree.getroot().findall(".//trkpt")
        if not track_points:
            raise ValueError(f"No <trkpt> elements found in {path}")

        parsed: list[tuple[datetime, float, float]] = []
        for point in track_points:
            lat = float(point.get("lat"))
            lon = float(point.get("lon"))
            time_el = point.find("gpx:time", namespace)
            if time_el is None:
                time_el = point.find("time")
            when = (
                datetime.fromisoformat(time_el.text.replace("Z", "+00:00"))
                if time_el is not None and time_el.text
                else None
            )
            parsed.append((when, lat, lon))

        stamped = [p for p in parsed if p[0] is not None]
        if stamped:
            origin = min(p[0] for p in stamped)
            points = [((p[0] - origin).total_seconds(), p[1], p[2]) for p in stamped]
        else:
            # No timestamps: assume 1 Hz sampling, which is the GPX default.
            points = [(float(i), p[1], p[2]) for i, p in enumerate(parsed)]
        return cls(points=points)

    def locate(
        self, video_time_s: float | None = None, timestamp: datetime | None = None
    ) -> GeoPoint | None:
        if not self.points:
            return None
        target = float(video_time_s or 0.0)

        if target <= self._times[0]:
            _, lat, lon = self.points[0]
            return GeoPoint(lat, lon, source=self.source)
        if target >= self._times[-1]:
            _, lat, lon = self.points[-1]
            return GeoPoint(lat, lon, source=self.source)

        index = bisect_left(self._times, target)
        t0, lat0, lon0 = self.points[index - 1]
        t1, lat1, lon1 = self.points[index]
        span = t1 - t0
        ratio = (target - t0) / span if span > 0 else 0.0
        return GeoPoint(
            latitude=lat0 + (lat1 - lat0) * ratio,
            longitude=lon0 + (lon1 - lon0) * ratio,
            source=self.source,
        )


@dataclass
class SimulatedRouteProvider(GpsProvider):
    """Synthesise a plausible drive for demo footage that has no GPS.

    Walks from ``start`` along ``bearing_deg`` at ``speed_kmph``, so
    consecutive frames of a clip produce consecutive points along a straight
    road rather than all piling onto one pin.

    Coordinates produced here are SIMULATED. Every ``GeoPoint`` it returns is
    stamped ``source="simulated"``, the API surfaces that field, and the
    dashboard badges such defects, so a demo can never be mistaken for a real
    survey.
    """

    start: tuple[float, float] = (19.0760, 72.8777)  # Mumbai, Dadar
    bearing_deg: float = 45.0
    speed_kmph: float = 30.0
    jitter_m: float = 4.0
    seed: int = 42
    source: str = "simulated"

    def __post_init__(self) -> None:
        import random

        self._rng = random.Random(self.seed)

    def locate(
        self, video_time_s: float | None = None, timestamp: datetime | None = None
    ) -> GeoPoint | None:
        elapsed = float(video_time_s or 0.0)
        distance_m = self.speed_kmph * 1000.0 / 3600.0 * elapsed
        bearing = math.radians(self.bearing_deg)

        north_m = distance_m * math.cos(bearing)
        east_m = distance_m * math.sin(bearing)
        if self.jitter_m:
            north_m += self._rng.uniform(-self.jitter_m, self.jitter_m)
            east_m += self._rng.uniform(-self.jitter_m, self.jitter_m)

        lat0, lon0 = self.start
        lat = lat0 + north_m / _M_PER_DEG_LAT
        metres_per_deg_lon = _M_PER_DEG_LAT * math.cos(math.radians(lat0))
        lon = lon0 + east_m / max(metres_per_deg_lon, 1e-6)
        return GeoPoint(latitude=lat, longitude=lon, accuracy_m=8.0, source=self.source)


def build_provider(spec: dict | None) -> GpsProvider:
    """Construct a provider from a plain config dict.

    ``{"method": "simulated", "start": [19.07, 72.87], "speed_kmph": 30}``
    """
    if not spec:
        return SimulatedRouteProvider()
    spec = dict(spec)
    method = spec.pop("method", "simulated")

    if method == "simulated":
        if "start" in spec and isinstance(spec["start"], (list, tuple)):
            spec["start"] = tuple(spec["start"])
        return SimulatedRouteProvider(**spec)
    if method == "device":
        return DeviceGpsProvider(**spec)
    if method == "static":
        return StaticGpsProvider(**spec)
    if method == "gpx":
        return GpxTrackProvider.from_file(spec["path"])
    if method == "exif":
        return ExifGpsProvider(**spec)
    raise ValueError(f"Unknown GPS provider method: {method!r}")
