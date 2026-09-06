"""Video/image capture - pipeline stage 1.

All four input sources named in the scope implement one interface, so any of
them can be dropped into the pipeline unchanged:

* ``VideoFileSource``    - an uploaded or pre-recorded dashcam video
* ``ImageDirectorySource`` - a directory of test images/videos
* ``CameraSource``       - a live dashcam / webcam / mobile camera stream
* ``SingleImageSource``  - one crowdsourced photo from the PWA

Every source yields ``CapturedFrame`` objects and honours ``sample_every`` /
``target_fps`` so that stage 2 receives frames at a configurable interval
rather than every frame of a 30 fps video.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterator

import cv2
import numpy as np

#: Extensions we treat as still images vs. video containers.
IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}
VIDEO_SUFFIXES = {".mp4", ".avi", ".mov", ".mkv", ".m4v", ".webm"}


@dataclass
class CapturedFrame:
    """One frame handed from stage 1 to stage 2."""

    image: np.ndarray
    index: int
    timestamp: datetime
    source_type: str            # video | image | camera | crowdsource
    source_ref: str             # filename / device id / upload id
    #: Seconds into the source clip, when meaningful.
    video_time_s: float | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def size(self) -> tuple[int, int]:
        height, width = self.image.shape[:2]
        return width, height


class FrameSource:
    """Interface implemented by every capture source."""

    source_type: str = "unknown"

    def frames(self) -> Iterator[CapturedFrame]:
        raise NotImplementedError

    def __iter__(self) -> Iterator[CapturedFrame]:
        return self.frames()

    def close(self) -> None:
        """Release any underlying handle. Safe to call more than once."""

    def __enter__(self) -> "FrameSource":
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()


@dataclass
class VideoFileSource(FrameSource):
    """Read a video file, sampling frames at a configurable interval.

    Exactly one of ``sample_every`` (take every Nth frame) or ``target_fps``
    (derive N from the clip's native fps) controls the sampling rate.
    """

    path: str | Path
    sample_every: int | None = None
    target_fps: float | None = 2.0
    max_frames: int | None = None
    start_time: datetime | None = None
    source_type: str = "video"

    def __post_init__(self) -> None:
        self.path = Path(self.path)
        if not self.path.exists():
            raise FileNotFoundError(f"Video not found: {self.path}")
        self._capture: cv2.VideoCapture | None = None

    def _open(self) -> cv2.VideoCapture:
        capture = cv2.VideoCapture(str(self.path))
        if not capture.isOpened():
            raise RuntimeError(f"OpenCV could not open video: {self.path}")
        self._capture = capture
        return capture

    def probe(self) -> dict[str, Any]:
        """Native properties of the clip, without consuming it."""
        capture = cv2.VideoCapture(str(self.path))
        if not capture.isOpened():
            raise RuntimeError(f"OpenCV could not open video: {self.path}")
        info = {
            "fps": capture.get(cv2.CAP_PROP_FPS) or 0.0,
            "frame_count": int(capture.get(cv2.CAP_PROP_FRAME_COUNT) or 0),
            "width": int(capture.get(cv2.CAP_PROP_FRAME_WIDTH) or 0),
            "height": int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0),
        }
        capture.release()
        info["duration_s"] = (
            info["frame_count"] / info["fps"] if info["fps"] > 0 else 0.0
        )
        return info

    def _stride(self, native_fps: float) -> int:
        if self.sample_every is not None:
            return max(1, int(self.sample_every))
        if self.target_fps and native_fps > 0:
            return max(1, int(round(native_fps / self.target_fps)))
        return 1

    def frames(self) -> Iterator[CapturedFrame]:
        capture = self._open()
        native_fps = capture.get(cv2.CAP_PROP_FPS) or 30.0
        stride = self._stride(native_fps)
        base_time = self.start_time or datetime.now(timezone.utc)

        index = 0
        emitted = 0
        try:
            while True:
                ok, frame = capture.read()
                if not ok:
                    break
                if index % stride == 0:
                    video_time = index / native_fps if native_fps > 0 else 0.0
                    yield CapturedFrame(
                        image=frame,
                        index=index,
                        timestamp=base_time,
                        source_type=self.source_type,
                        source_ref=self.path.name,
                        video_time_s=video_time,
                        metadata={"native_fps": native_fps, "stride": stride},
                    )
                    emitted += 1
                    if self.max_frames is not None and emitted >= self.max_frames:
                        break
                index += 1
        finally:
            self.close()

    def close(self) -> None:
        if self._capture is not None:
            self._capture.release()
            self._capture = None


@dataclass
class ImageDirectorySource(FrameSource):
    """Walk a directory of test images (and, optionally, videos).

    This is the source used by ``scripts/run_demo.py --input data/samples``
    and by the batch evaluation harness.
    """

    directory: str | Path
    recursive: bool = True
    include_videos: bool = False
    max_frames: int | None = None
    video_target_fps: float = 1.0
    source_type: str = "image"

    def __post_init__(self) -> None:
        self.directory = Path(self.directory)
        if not self.directory.is_dir():
            raise NotADirectoryError(f"Not a directory: {self.directory}")

    def _listing(self) -> list[Path]:
        pattern = "**/*" if self.recursive else "*"
        allowed = set(IMAGE_SUFFIXES)
        if self.include_videos:
            allowed |= VIDEO_SUFFIXES
        return sorted(
            p
            for p in self.directory.glob(pattern)
            if p.is_file() and p.suffix.lower() in allowed
        )

    def frames(self) -> Iterator[CapturedFrame]:
        emitted = 0
        for index, path in enumerate(self._listing()):
            if self.max_frames is not None and emitted >= self.max_frames:
                return

            if path.suffix.lower() in VIDEO_SUFFIXES:
                nested = VideoFileSource(path, target_fps=self.video_target_fps)
                for frame in nested.frames():
                    yield frame
                    emitted += 1
                    if self.max_frames is not None and emitted >= self.max_frames:
                        return
                continue

            image = cv2.imread(str(path))
            if image is None:
                continue  # unreadable/corrupt file - skip rather than crash
            stat = path.stat()
            yield CapturedFrame(
                image=image,
                index=index,
                timestamp=datetime.fromtimestamp(stat.st_mtime, tz=timezone.utc),
                source_type=self.source_type,
                source_ref=str(path.relative_to(self.directory)),
                metadata={"absolute_path": str(path)},
            )
            emitted += 1

    def count(self) -> int:
        return len(self._listing())


@dataclass
class SingleImageSource(FrameSource):
    """One image - a crowdsourced upload from the driver-facing PWA."""

    path: str | Path | None = None
    image: np.ndarray | None = None
    source_ref: str = "upload"
    source_type: str = "crowdsource"
    timestamp: datetime | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def frames(self) -> Iterator[CapturedFrame]:
        image = self.image
        if image is None:
            if self.path is None:
                raise ValueError("SingleImageSource needs either `path` or `image`")
            image = cv2.imread(str(self.path))
            if image is None:
                raise ValueError(f"Could not decode image: {self.path}")
        yield CapturedFrame(
            image=image,
            index=0,
            timestamp=self.timestamp or datetime.now(timezone.utc),
            source_type=self.source_type,
            source_ref=self.source_ref,
            metadata=dict(self.metadata),
        )


@dataclass
class CameraSource(FrameSource):
    """Live camera stream - a USB dashcam, a laptop webcam, or an RTSP/HTTP
    feed published by the mobile app.

    ``device`` is an integer index for a local camera or a URL string for a
    network stream. Sampling is time-based rather than frame-based because a
    live stream has no fixed frame budget.
    """

    device: int | str = 0
    target_fps: float = 2.0
    max_frames: int | None = None
    warmup_frames: int = 5
    source_type: str = "camera"

    def __post_init__(self) -> None:
        self._capture: cv2.VideoCapture | None = None

    def _open(self) -> cv2.VideoCapture:
        capture = cv2.VideoCapture(self.device)
        if not capture.isOpened():
            raise RuntimeError(
                f"Could not open camera device {self.device!r}. On macOS, grant "
                "camera permission to your terminal in System Settings > "
                "Privacy & Security > Camera."
            )
        self._capture = capture
        return capture

    def frames(self) -> Iterator[CapturedFrame]:
        capture = self._open()
        interval = 1.0 / self.target_fps if self.target_fps > 0 else 0.0

        # Discard the first few frames; most sensors need a moment to expose.
        for _ in range(self.warmup_frames):
            capture.read()

        index = 0
        emitted = 0
        next_due = time.monotonic()
        try:
            while True:
                ok, frame = capture.read()
                if not ok:
                    break
                now = time.monotonic()
                if now >= next_due:
                    next_due = now + interval
                    yield CapturedFrame(
                        image=frame,
                        index=index,
                        timestamp=datetime.now(timezone.utc),
                        source_type=self.source_type,
                        source_ref=f"camera:{self.device}",
                        metadata={"live": True},
                    )
                    emitted += 1
                    if self.max_frames is not None and emitted >= self.max_frames:
                        break
                index += 1
        finally:
            self.close()

    def close(self) -> None:
        if self._capture is not None:
            self._capture.release()
            self._capture = None


def open_source(spec: str, **kwargs: Any) -> FrameSource:
    """Build the right source for a path/device string.

    ``open_source("clip.mp4")``, ``open_source("data/samples")``,
    ``open_source("photo.jpg")``, ``open_source("0")`` (camera index) or
    ``open_source("rtsp://...")`` all do the expected thing, which is what
    lets the CLI take a single ``--input`` argument.
    """
    if spec.isdigit():
        return CameraSource(device=int(spec), **kwargs)
    if spec.startswith(("rtsp://", "http://", "https://")):
        return CameraSource(device=spec, **kwargs)

    path = Path(spec)
    if path.is_dir():
        allowed = {
            k: v
            for k, v in kwargs.items()
            if k in {"recursive", "include_videos", "max_frames", "video_target_fps"}
        }
        return ImageDirectorySource(path, **allowed)
    if not path.exists():
        raise FileNotFoundError(f"No such input: {spec}")
    if path.suffix.lower() in VIDEO_SUFFIXES:
        allowed = {
            k: v
            for k, v in kwargs.items()
            if k in {"sample_every", "target_fps", "max_frames", "start_time"}
        }
        return VideoFileSource(path, **allowed)
    if path.suffix.lower() in IMAGE_SUFFIXES:
        return SingleImageSource(path=path, source_ref=path.name, source_type="image")
    raise ValueError(f"Unsupported input type: {spec}")
