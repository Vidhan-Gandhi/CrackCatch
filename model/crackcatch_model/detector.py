"""AI detection engine - pipeline stage 3.

The primary detector is YOLOv8 (Ultralytics). Because a fine-tuned road-damage
checkpoint is a *trained artefact* rather than something that can ship in a
git repository, this module also provides a classical computer-vision fallback
so that the end-to-end pipeline - capture, preprocess, detect, score, geo-tag,
store, display - is genuinely runnable on a fresh clone before any training
has happened.

Selection order used by :func:`build_detector`:

1. an explicit ``weights`` path, if given and present;
2. ``model/weights/crackcatch.pt`` - the fine-tuned checkpoint produced by
   ``model/scripts/train.py``;
3. ``HeuristicDetector`` - classical CV, clearly labelled as such.

HONESTY NOTE: the heuristic detector is **not** a substitute for the trained
model. It finds dark, road-coloured blobs and thin linear structures, which
correlates with potholes and cracks but also fires on oil stains, shadows and
tar patches. Its ``confidence`` is a shape/contrast plausibility score, not a
calibrated probability. Every detection it produces is tagged
``detector="heuristic"`` so the dashboard and any metric report can exclude
it. Reported mAP figures in ``docs/MODEL_CARD.md`` come from the YOLOv8
checkpoint only.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Sequence

import cv2
import numpy as np

from crackcatch_model.types import BoundingBox, DefectClass

logger = logging.getLogger(__name__)

#: Where train.py writes the fine-tuned checkpoint.
DEFAULT_WEIGHTS = Path(__file__).resolve().parents[1] / "weights" / "crackcatch.pt"


@dataclass
class RawDetection:
    """A detector output before severity/size scoring."""

    defect_class: DefectClass
    confidence: float
    bbox: BoundingBox
    detector: str = "unknown"
    raw_label: str = ""


class Detector:
    """Interface implemented by every detection backend."""

    name: str = "base"

    def predict(self, image: np.ndarray) -> list[RawDetection]:
        raise NotImplementedError

    def warmup(self, size: tuple[int, int] = (640, 640)) -> None:
        """Run one throwaway inference so the first timed call is honest."""
        blank = np.zeros((size[1], size[0], 3), dtype=np.uint8)
        try:
            self.predict(blank)
        except Exception:  # pragma: no cover - warmup must never break a run
            logger.debug("warmup inference failed", exc_info=True)

    def benchmark(self, image: np.ndarray, runs: int = 20) -> dict[str, float]:
        """Measure achieved inference FPS. Used by scripts/benchmark_fps.py."""
        self.warmup((image.shape[1], image.shape[0]))
        started = time.perf_counter()
        for _ in range(runs):
            self.predict(image)
        elapsed = time.perf_counter() - started
        return {
            "runs": runs,
            "total_s": elapsed,
            "ms_per_frame": 1000.0 * elapsed / runs,
            "fps": runs / elapsed if elapsed > 0 else 0.0,
        }


@dataclass
class YOLOv8Detector(Detector):
    """Ultralytics YOLOv8 wrapper.

    ``device`` accepts ``"cpu"``, ``"cuda"``, ``"mps"`` (Apple Silicon) or
    ``"auto"``, which picks the best available. Given the project's stated
    constraint of no dedicated GPU cluster, CPU is a fully supported path -
    ``yolov8n`` at 640px runs at a usable rate on a modern laptop CPU and the
    achieved figure is reported by ``scripts/benchmark_fps.py`` rather than
    assumed.
    """

    weights: str | Path = DEFAULT_WEIGHTS
    confidence_threshold: float = 0.25
    iou_threshold: float = 0.45
    device: str = "auto"
    imgsz: int = 640
    max_detections: int = 50
    name: str = "yolov8"

    def __post_init__(self) -> None:
        try:
            from ultralytics import YOLO
        except ImportError as exc:  # pragma: no cover
            raise RuntimeError(
                "ultralytics is not installed. `pip install -r requirements.txt`"
            ) from exc

        weights_path = Path(self.weights)
        if not weights_path.exists():
            raise FileNotFoundError(f"YOLO weights not found: {weights_path}")

        self.device = self._resolve_device(self.device)
        self._model = YOLO(str(weights_path))
        self._names: dict[int, str] = dict(self._model.names)
        # Precompute label -> canonical class, skipping classes we don't model
        # (a COCO-pretrained checkpoint has 80 irrelevant classes).
        self._class_map: dict[int, DefectClass] = {}
        for idx, label in self._names.items():
            try:
                self._class_map[idx] = DefectClass.from_any(label)
            except ValueError:
                continue
        if not self._class_map:
            logger.warning(
                "Checkpoint %s exposes no road-damage classes (%s). It is "
                "probably a generic COCO checkpoint, not a CrackCatch model.",
                weights_path,
                list(self._names.values())[:5],
            )

    @staticmethod
    def _resolve_device(requested: str) -> str:
        if requested != "auto":
            return requested
        try:
            import torch

            if torch.cuda.is_available():
                return "cuda"
            if torch.backends.mps.is_available():
                return "mps"
        except Exception:  # pragma: no cover
            pass
        return "cpu"

    @property
    def class_names(self) -> dict[int, str]:
        return dict(self._names)

    def predict(self, image: np.ndarray) -> list[RawDetection]:
        results = self._model.predict(
            image,
            conf=self.confidence_threshold,
            iou=self.iou_threshold,
            imgsz=self.imgsz,
            device=self.device,
            max_det=self.max_detections,
            verbose=False,
        )
        detections: list[RawDetection] = []
        height, width = image.shape[:2]
        for result in results:
            boxes = getattr(result, "boxes", None)
            if boxes is None:
                continue
            for box in boxes:
                cls_idx = int(box.cls.item())
                defect_class = self._class_map.get(cls_idx)
                if defect_class is None:
                    continue  # not a road-damage class
                x1, y1, x2, y2 = (float(v) for v in box.xyxy[0].tolist())
                detections.append(
                    RawDetection(
                        defect_class=defect_class,
                        confidence=float(box.conf.item()),
                        bbox=BoundingBox(x1, y1, x2, y2).clip(width, height),
                        detector=self.name,
                        raw_label=self._names.get(cls_idx, str(cls_idx)),
                    )
                )
        return detections


@dataclass
class HeuristicDetector(Detector):
    """Classical-CV fallback so a fresh clone has a runnable pipeline.

    Method
    ------
    1. Restrict attention to the lower ``road_roi`` fraction of the frame -
       the road surface - which removes sky, buildings and most vehicles.
    2. Blackhat morphology (closing minus original) isolates structures that
       are *darker than their surroundings* at a chosen scale. Both potholes
       and cracks are dark against asphalt; blackhat is scale-selective so it
       responds to them and not to a broad lighting gradient. Several kernel
       sizes are combined by per-pixel maximum so that both a distant hairline
       crack and a near pothole are covered.
    3. Otsu threshold the blackhat response, then take connected components.
    4. Classify each component by shape: elongated (high aspect ratio, low
       fill) -> crack; compact and solid -> pothole.
    5. Score confidence from local contrast and shape plausibility.

    This is a *baseline*, not a model. See the module docstring.
    """

    confidence_threshold: float = 0.25
    road_roi: float = 0.45          # use the bottom 55% of the frame
    min_area_px: int = 220
    max_area_frac: float = 0.28
    #: Blackhat responds to dark structures *smaller than* its kernel, so a
    #: single kernel can only see defects at one scale: a 17 px kernel finds
    #: distant potholes but returns only a thin rim for a large near one.
    #: Running several kernels and keeping the per-pixel maximum covers the
    #: whole range a dashcam sees, from far-field cracks to a pothole filling
    #: a third of the frame.
    blackhat_kernels: tuple[int, ...] = (11, 25, 51)
    #: Box-blur size used to estimate local road brightness for the
    #: scale-free darkness cue. Must be much larger than any single defect.
    background_kernel: int = 151
    #: Minimum contrast spread (p99.5 - median) for a cue to be trusted.
    #: Below this the cue is noise, not structure.
    min_response: int = 12
    #: Aspect ratio above which an elongated component is read as a crack
    #: rather than a pothole. Measured on demo footage, genuine crack
    #: components cluster at 3-8 while pothole blobs sit below 2.5.
    crack_min_aspect: float = 3.0
    max_detections: int = 12
    name: str = "heuristic"

    def _road_mask_offset(self, height: int) -> int:
        return int(height * self.road_roi)

    def predict(self, image: np.ndarray) -> list[RawDetection]:
        if image is None or image.size == 0:
            return []
        height, width = image.shape[:2]
        top = self._road_mask_offset(height)
        roi = image[top:, :]
        if roi.size == 0:
            return []

        grey = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY) if roi.ndim == 3 else roi
        grey = cv2.GaussianBlur(grey, (5, 5), 0)

        blackhat = np.zeros_like(grey)
        for size in self.blackhat_kernels:
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (size, size))
            blackhat = np.maximum(
                blackhat, cv2.morphologyEx(grey, cv2.MORPH_BLACKHAT, kernel)
            )

        # Blackhat saturates once a defect grows past its largest kernel: a
        # pothole filling a third of the frame returns only a thin rim. This
        # second, scale-free cue compares each pixel against a heavily blurred
        # estimate of the local road brightness, so an arbitrarily large dark
        # region still responds at full strength.
        background = cv2.blur(grey, (self.background_kernel, self.background_kernel))
        darkness = cv2.subtract(background, grey)

        # Each cue is thresholded on its OWN Otsu level and the masks are
        # OR-ed. Thresholding their combined maximum instead would let the
        # strong, broad darkness response raise the level above the much
        # fainter blackhat response of a thin crack, so cracks would vanish
        # whenever a large pothole shared the frame.
        masks = []
        for cue in (blackhat, darkness):
            # Admit a cue only if it has real structure. The test is its
            # contrast SPREAD (99.5th percentile minus median), not its peak:
            # sensor noise alone produces isolated peaks of 15-20 while its
            # spread stays near 3, whereas a genuine defect drives the spread
            # to 30+. Thresholding a noise-only cue with Otsu would speckle
            # the mask and, once OR-ed in, merge every component into one
            # frame-sized blob.
            spread = float(np.percentile(cue, 99.5)) - float(np.median(cue))
            if spread < self.min_response:
                continue
            _, mask = cv2.threshold(cue, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            masks.append(mask)

        if not masks:
            return []
        binary = masks[0]
        for mask in masks[1:]:
            binary = cv2.bitwise_or(binary, mask)
        binary = cv2.morphologyEx(
            binary,
            cv2.MORPH_CLOSE,
            cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)),
        )

        count, labels, stats, _ = cv2.connectedComponentsWithStats(binary, 8)
        frame_area = float(width * height)
        candidates: list[RawDetection] = []

        for label in range(1, count):
            x, y, w, h, area = (
                int(stats[label, cv2.CC_STAT_LEFT]),
                int(stats[label, cv2.CC_STAT_TOP]),
                int(stats[label, cv2.CC_STAT_WIDTH]),
                int(stats[label, cv2.CC_STAT_HEIGHT]),
                int(stats[label, cv2.CC_STAT_AREA]),
            )
            if area < self.min_area_px:
                continue
            box_area = float(max(w * h, 1))
            if box_area / frame_area > self.max_area_frac:
                continue

            fill = area / box_area                      # solidity within the box
            aspect = max(w, h) / max(min(w, h), 1)

            if aspect >= self.crack_min_aspect and fill < 0.75:
                defect_class = DefectClass.CRACK
                shape_plausibility = min(1.0, aspect / 8.0)
            elif fill >= 0.45 and aspect < self.crack_min_aspect:
                defect_class = DefectClass.POTHOLE
                shape_plausibility = fill
            else:
                continue

            # Local contrast: how much darker the component is than the road
            # immediately around it. This is the single strongest cue.
            component = labels[y : y + h, x : x + w] == label
            patch = grey[y : y + h, x : x + w]
            if patch.size == 0 or not component.any():
                continue
            inside = float(patch[component].mean())
            surround = float(np.median(grey))
            contrast = max(0.0, (surround - inside) / max(surround, 1.0))

            confidence = float(
                np.clip(0.30 + 0.55 * contrast + 0.15 * shape_plausibility, 0.0, 0.95)
            )
            if confidence < self.confidence_threshold:
                continue

            candidates.append(
                RawDetection(
                    defect_class=defect_class,
                    confidence=confidence,
                    bbox=BoundingBox(
                        float(x), float(y + top), float(x + w), float(y + h + top)
                    ).clip(width, height),
                    detector=self.name,
                    raw_label=f"{defect_class.value}(cv)",
                )
            )

        candidates.sort(key=lambda d: d.confidence, reverse=True)
        return non_max_suppression(candidates, 0.35)[: self.max_detections]


def non_max_suppression(
    detections: Sequence[RawDetection],
    iou_threshold: float = 0.45,
    class_agnostic: bool = False,
) -> list[RawDetection]:
    """Greedy NMS, per class by default.

    Suppression is applied *within* a class, not across classes: a crack
    running past the rim of a pothole is a real, separate defect, and a
    class-agnostic pass would delete whichever of the two scored lower. Pass
    ``class_agnostic=True`` only when duplicate boxes of different classes on
    the same object are the problem you are solving.
    """
    ordered = sorted(detections, key=lambda d: d.confidence, reverse=True)
    kept: list[RawDetection] = []
    for candidate in ordered:
        rivals = (
            kept
            if class_agnostic
            else [k for k in kept if k.defect_class is candidate.defect_class]
        )
        if all(candidate.bbox.iou(rival.bbox) < iou_threshold for rival in rivals):
            kept.append(candidate)
    return kept


def build_detector(
    weights: str | Path | None = None,
    confidence_threshold: float = 0.25,
    device: str = "auto",
    allow_fallback: bool = True,
    **kwargs: Any,
) -> Detector:
    """Return the best available detector, falling back loudly, never silently."""
    candidates: list[Path] = []
    if weights:
        candidates.append(Path(weights))
    candidates.append(DEFAULT_WEIGHTS)

    for path in candidates:
        if not path.exists():
            continue
        try:
            detector = YOLOv8Detector(
                weights=path,
                confidence_threshold=confidence_threshold,
                device=device,
                **kwargs,
            )
            logger.info("Using YOLOv8 detector: %s (device=%s)", path, detector.device)
            return detector
        except Exception:
            logger.exception("Failed to load YOLO weights at %s", path)
            if not allow_fallback:
                raise

    if not allow_fallback:
        raise FileNotFoundError(
            f"No YOLO weights found. Looked in: {[str(p) for p in candidates]}. "
            "Run `python model/scripts/train.py` first, or pass "
            "allow_fallback=True to use the classical-CV baseline."
        )

    logger.warning(
        "No trained YOLOv8 checkpoint found (looked in %s). Falling back to the "
        "classical-CV HeuristicDetector so the pipeline stays runnable. This is "
        "a BASELINE, not the trained model - see docs/MODEL_CARD.md.",
        [str(p) for p in candidates],
    )
    return HeuristicDetector(confidence_threshold=confidence_threshold)
