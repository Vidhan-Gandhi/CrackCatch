"""Frame preprocessing - pipeline stage 2.

Sits between capture (stage 1) and detection (stage 3) and performs the four
operations named in the project scope: frame extraction at a configurable
interval, resize, denoise, and lighting normalisation.

Lighting normalisation uses CLAHE (Contrast Limited Adaptive Histogram
Equalisation) on the L channel of LAB rather than a global ``equalizeHist``.
On Indian road footage the frame is routinely half sunlit and half in deep
shadow from buildings or trees; a global equalisation is dominated by the
bright half and leaves the shaded half - where potholes hide - still crushed.
CLAHE equalises locally and clips the contrast gain, which keeps shadowed road
texture visible without amplifying sensor noise into false crack edges.
"""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np


@dataclass(frozen=True)
class PreprocessConfig:
    """Configuration for stage 2."""

    #: Target size fed to the detector. YOLOv8 wants a multiple of 32.
    target_size: tuple[int, int] = (640, 640)
    #: Preserve aspect ratio by letterboxing instead of stretching.
    letterbox: bool = True
    #: CLAHE clip limit. Higher = more local contrast, more noise.
    clahe_clip_limit: float = 2.0
    clahe_grid: tuple[int, int] = (8, 8)
    #: Enable bilateral denoising. Edge-preserving, so cracks survive it.
    denoise: bool = True
    bilateral_d: int = 5
    bilateral_sigma_colour: float = 50.0
    bilateral_sigma_space: float = 50.0
    #: Skip frames whose Laplacian variance is below this (motion blur).
    blur_rejection_threshold: float = 25.0


DEFAULT_PREPROCESS_CONFIG = PreprocessConfig()


def normalise_lighting(
    frame: np.ndarray, config: PreprocessConfig = DEFAULT_PREPROCESS_CONFIG
) -> np.ndarray:
    """CLAHE on the LAB luminance channel; colour is left untouched."""
    if frame.ndim == 2:
        clahe = cv2.createCLAHE(
            clipLimit=config.clahe_clip_limit, tileGridSize=config.clahe_grid
        )
        return clahe.apply(frame)

    lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
    lightness, a_chan, b_chan = cv2.split(lab)
    clahe = cv2.createCLAHE(
        clipLimit=config.clahe_clip_limit, tileGridSize=config.clahe_grid
    )
    lightness = clahe.apply(lightness)
    return cv2.cvtColor(cv2.merge((lightness, a_chan, b_chan)), cv2.COLOR_LAB2BGR)


def denoise(
    frame: np.ndarray, config: PreprocessConfig = DEFAULT_PREPROCESS_CONFIG
) -> np.ndarray:
    """Edge-preserving bilateral filter.

    A Gaussian blur would smear the thin high-frequency edges that *are* the
    crack signal; a bilateral filter suppresses sensor noise while keeping
    those edges intact.
    """
    if not config.denoise:
        return frame
    return cv2.bilateralFilter(
        frame,
        config.bilateral_d,
        config.bilateral_sigma_colour,
        config.bilateral_sigma_space,
    )


def letterbox_resize(
    frame: np.ndarray, target: tuple[int, int]
) -> tuple[np.ndarray, float, tuple[int, int]]:
    """Resize preserving aspect ratio, padding with grey.

    Returns ``(image, scale, (pad_x, pad_y))`` so detections made on the
    letterboxed image can be mapped back to original-frame coordinates.
    """
    target_w, target_h = target
    height, width = frame.shape[:2]
    scale = min(target_w / width, target_h / height)
    new_w, new_h = int(round(width * scale)), int(round(height * scale))
    resized = cv2.resize(frame, (new_w, new_h), interpolation=cv2.INTER_LINEAR)

    pad_x = (target_w - new_w) // 2
    pad_y = (target_h - new_h) // 2
    canvas = np.full((target_h, target_w, frame.shape[2]), 114, dtype=frame.dtype)
    canvas[pad_y : pad_y + new_h, pad_x : pad_x + new_w] = resized
    return canvas, scale, (pad_x, pad_y)


def undo_letterbox(
    x1: float,
    y1: float,
    x2: float,
    y2: float,
    scale: float,
    pad: tuple[int, int],
) -> tuple[float, float, float, float]:
    """Map letterboxed-image coordinates back to the original frame."""
    pad_x, pad_y = pad
    return (
        (x1 - pad_x) / scale,
        (y1 - pad_y) / scale,
        (x2 - pad_x) / scale,
        (y2 - pad_y) / scale,
    )


def blur_score(frame: np.ndarray) -> float:
    """Variance of the Laplacian. Low values indicate motion blur."""
    grey = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) if frame.ndim == 3 else frame
    return float(cv2.Laplacian(grey, cv2.CV_64F).var())


def is_usable(
    frame: np.ndarray, config: PreprocessConfig = DEFAULT_PREPROCESS_CONFIG
) -> bool:
    """Reject frames too motion-blurred to detect reliably on."""
    return blur_score(frame) >= config.blur_rejection_threshold


@dataclass
class PreprocessedFrame:
    """A frame ready for the detector, plus what is needed to invert it."""

    image: np.ndarray                # preprocessed, detector-sized
    original: np.ndarray             # untouched original frame (for snapshots)
    scale: float
    pad: tuple[int, int]
    original_size: tuple[int, int]   # (width, height)
    blur: float

    def to_original_bbox(
        self, x1: float, y1: float, x2: float, y2: float
    ) -> tuple[float, float, float, float]:
        return undo_letterbox(x1, y1, x2, y2, self.scale, self.pad)


def preprocess(
    frame: np.ndarray, config: PreprocessConfig = DEFAULT_PREPROCESS_CONFIG
) -> PreprocessedFrame:
    """Run the full stage-2 chain on one frame."""
    if frame is None or frame.size == 0:
        raise ValueError("preprocess() received an empty frame")
    if frame.ndim == 2:
        frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)

    original = frame.copy()
    height, width = frame.shape[:2]
    blur = blur_score(frame)

    work = normalise_lighting(frame, config)
    work = denoise(work, config)

    if config.letterbox:
        image, scale, pad = letterbox_resize(work, config.target_size)
    else:
        image = cv2.resize(work, config.target_size, interpolation=cv2.INTER_LINEAR)
        scale = 1.0
        pad = (0, 0)
        # A stretched resize needs an anisotropic inverse, which
        # to_original_bbox cannot express, so record the identity and let the
        # caller work in target-image coordinates.

    return PreprocessedFrame(
        image=image,
        original=original,
        scale=scale,
        pad=pad,
        original_size=(width, height),
        blur=blur,
    )
