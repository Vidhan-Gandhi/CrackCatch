"""Severity classification (second half of pipeline stage 4).

The scope requires the scoring logic to be *exposed, not a black box*, so that
it can be explained to a review panel. Accordingly:

* the score is a plain weighted sum of three named factors - bounding-box
  size, aspect ratio and detection confidence - exactly the three inputs
  named in the project scope;
* every factor is normalised to ``[0, 1]`` by an explicit, documented ramp;
* every call returns a ``breakdown`` dict showing each factor's value and its
  weighted contribution, which the dashboard renders on the defect detail
  page and which ``tests/test_severity.py`` asserts on.

    severity_score = w_size * size_factor
                   + w_shape * shape_factor
                   + w_conf * confidence_factor

    score < minor_max      -> Minor
    score < moderate_max   -> Moderate
    otherwise              -> Severe

Interpretation of each factor
-----------------------------
size_factor
    How much road surface the defect occupies. Uses the metric footprint
    (m^2 for potholes, length in m for cracks) when a calibration recovered
    one, and falls back to the frame-relative pixel area otherwise. This is
    the dominant term because a large defect is unambiguously worse.

shape_factor
    Encodes what a *bad* instance of each class looks like. A pothole is worse
    when it is compact (aspect ratio near 1) - an elongated blob on the road
    is more often a patch, a shadow or a rut. A crack is worse when it is
    elongated, because linear extent is what a crack's damage actually is.

confidence_factor
    Detector confidence, re-ramped from the acceptance threshold to 1.0 so
    that a barely-accepted detection contributes ~0 rather than ~0.4.
"""

from __future__ import annotations

from dataclasses import dataclass, field, asdict

from crackcatch_model.types import (
    BoundingBox,
    DefectClass,
    Detection,
    Severity,
    SizeEstimate,
)


def _ramp(value: float, low: float, high: float) -> float:
    """Linear ramp: 0 at or below ``low``, 1 at or above ``high``."""
    if high <= low:
        raise ValueError(f"ramp needs high > low, got {low} -> {high}")
    return max(0.0, min(1.0, (value - low) / (high - low)))


@dataclass(frozen=True)
class SeverityConfig:
    """Tunable, fully documented severity parameters.

    Defaults are calibrated against Indian urban arterial roads: a pothole of
    roughly 0.6 m x 0.6 m is the point at which a two-wheeler is at real risk,
    and a 4 m crack run is the point at which a segment needs resurfacing
    rather than spot-filling.
    """

    # --- factor weights (must sum to 1.0) ---
    w_size: float = 0.55
    w_shape: float = 0.20
    w_conf: float = 0.25

    # --- decision thresholds on the final score ---
    minor_max: float = 0.35
    moderate_max: float = 0.62

    # --- metric size ramps ---
    pothole_area_m2_low: float = 0.05    # ~22 cm across: cosmetic
    pothole_area_m2_high: float = 0.60   # ~78 cm across: hazardous
    crack_length_m_low: float = 0.50
    crack_length_m_high: float = 4.00

    # --- fallback ramp when no calibration is available ---
    area_frac_low: float = 0.002         # 0.2% of the frame
    area_frac_high: float = 0.080        # 8% of the frame

    # --- shape ramps ---
    pothole_ar_compact: float = 1.0      # AR at which a pothole scores 1.0
    pothole_ar_elongated: float = 2.5    # AR at which a pothole scores 0.0
    crack_ar_low: float = 1.5
    crack_ar_high: float = 6.0

    # --- confidence ramp ---
    conf_floor: float = 0.25             # acceptance threshold

    def __post_init__(self) -> None:
        total = self.w_size + self.w_shape + self.w_conf
        if abs(total - 1.0) > 1e-6:
            raise ValueError(f"severity weights must sum to 1.0, got {total}")
        if not 0.0 < self.minor_max < self.moderate_max < 1.0:
            raise ValueError(
                "thresholds must satisfy 0 < minor_max < moderate_max < 1"
            )

    def to_dict(self) -> dict[str, float]:
        return asdict(self)


DEFAULT_SEVERITY_CONFIG = SeverityConfig()


def size_factor(
    defect_class: DefectClass,
    size: SizeEstimate | None,
    config: SeverityConfig = DEFAULT_SEVERITY_CONFIG,
) -> tuple[float, str]:
    """Return ``(factor, basis)`` where ``basis`` names the measure used."""
    if size is None:
        return 0.0, "unavailable"

    # Metric numbers drive severity only when the calibration vouched for
    # them; an unreliable far-field projection falls back to pixel measures.
    metric_ok = size.method != "pixel_only" and size.reliable

    if defect_class is DefectClass.CRACK:
        # A crack's severity is its run length, not its area.
        if metric_ok and size.length_m is not None and size.width_m is not None:
            run = max(size.length_m, size.width_m)
            return (
                _ramp(run, config.crack_length_m_low, config.crack_length_m_high),
                "crack_length_m",
            )
    elif metric_ok and size.area_m2 is not None:
        return (
            _ramp(size.area_m2, config.pothole_area_m2_low, config.pothole_area_m2_high),
            "footprint_m2",
        )

    return (
        _ramp(size.area_frac, config.area_frac_low, config.area_frac_high),
        "frame_area_fraction",
    )


def shape_factor(
    defect_class: DefectClass,
    bbox: BoundingBox,
    config: SeverityConfig = DEFAULT_SEVERITY_CONFIG,
) -> float:
    """Class-conditional aspect-ratio score. See module docstring."""
    ar = bbox.aspect_ratio
    if defect_class is DefectClass.CRACK:
        # Elongated is worse.
        return _ramp(ar, config.crack_ar_low, config.crack_ar_high)
    # Potholes and manholes: compact is worse, so invert the ramp.
    return 1.0 - _ramp(ar, config.pothole_ar_compact, config.pothole_ar_elongated)


def confidence_factor(
    confidence: float, config: SeverityConfig = DEFAULT_SEVERITY_CONFIG
) -> float:
    return _ramp(confidence, config.conf_floor, 1.0)


def classify(
    defect_class: DefectClass,
    bbox: BoundingBox,
    confidence: float,
    size: SizeEstimate | None = None,
    config: SeverityConfig = DEFAULT_SEVERITY_CONFIG,
) -> tuple[Severity, float, dict[str, float]]:
    """Score one detection.

    Returns ``(severity, score, breakdown)``. The breakdown carries both the
    raw factor values and their weighted contributions so the dashboard can
    show *why* a defect was rated the way it was.
    """
    if not 0.0 <= confidence <= 1.0:
        raise ValueError(f"confidence must be in [0, 1], got {confidence}")

    f_size, basis = size_factor(defect_class, size, config)
    f_shape = shape_factor(defect_class, bbox, config)
    f_conf = confidence_factor(confidence, config)

    score = (
        config.w_size * f_size
        + config.w_shape * f_shape
        + config.w_conf * f_conf
    )
    score = max(0.0, min(1.0, score))

    if score < config.minor_max:
        severity = Severity.MINOR
    elif score < config.moderate_max:
        severity = Severity.MODERATE
    else:
        severity = Severity.SEVERE

    breakdown = {
        "size_factor": f_size,
        "shape_factor": f_shape,
        "confidence_factor": f_conf,
        "size_contribution": config.w_size * f_size,
        "shape_contribution": config.w_shape * f_shape,
        "confidence_contribution": config.w_conf * f_conf,
        "aspect_ratio": bbox.aspect_ratio,
        "total": score,
    }
    # Non-numeric provenance, kept alongside for the UI tooltip.
    breakdown["_size_basis"] = basis  # type: ignore[assignment]
    return severity, score, breakdown


def explain(breakdown: dict[str, float]) -> str:
    """Render a breakdown as a one-line human-readable justification."""
    basis = breakdown.get("_size_basis", "unknown")
    return (
        f"size({basis})={breakdown['size_factor']:.2f}*w -> "
        f"{breakdown['size_contribution']:.3f}; "
        f"shape(AR={breakdown['aspect_ratio']:.2f})={breakdown['shape_factor']:.2f}*w -> "
        f"{breakdown['shape_contribution']:.3f}; "
        f"conf={breakdown['confidence_factor']:.2f}*w -> "
        f"{breakdown['confidence_contribution']:.3f}; "
        f"total={breakdown['total']:.3f}"
    )


def apply(
    detection: Detection, config: SeverityConfig = DEFAULT_SEVERITY_CONFIG
) -> Detection:
    """Populate a ``Detection``'s severity fields in place and return it."""
    severity, score, breakdown = classify(
        detection.defect_class,
        detection.bbox,
        detection.confidence,
        detection.size,
        config,
    )
    detection.severity = severity
    detection.severity_score = score
    detection.severity_breakdown = breakdown
    return detection
