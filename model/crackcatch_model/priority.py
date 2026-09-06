"""Repair-priority scoring.

The review paper's stated gap is that *detection-only systems do not support
repair prioritisation*: they tell an authority that 400 defects exist but not
which one to fix on Monday morning. This module closes that gap with a
transparent 0-100 score.

    priority = 100 * (w_sev * severity_norm
                    + w_traffic * traffic_weight
                    + w_class * class_weight
                    + w_age * age_factor)

Every input is a documented constant rather than a learned parameter, because
an authority has to be able to argue for a work order in a council meeting.
``traffic_weight`` is a static per-road-type assumption, not a measurement -
the honest position given no traffic-count feed. Swapping in a real AADT
(annual average daily traffic) figure per road segment is a drop-in change to
``ROAD_TYPE_WEIGHTS`` and is noted as future work.
"""

from __future__ import annotations

from dataclasses import dataclass, asdict
from datetime import datetime, timezone

from crackcatch_model.types import DefectClass, Severity

#: Static traffic-exposure assumption per road class, 0-1.
#: Rationale: a defect on a highway is hit by orders of magnitude more
#: vehicles - and at higher speed, so with worse consequences - than the same
#: defect on a residential lane.
ROAD_TYPE_WEIGHTS: dict[str, float] = {
    "highway": 1.00,
    "arterial": 0.80,     # main city roads - the default
    "collector": 0.55,
    "residential": 0.35,
    "service": 0.20,
    "unknown": 0.50,
}

#: How dangerous each defect class is per unit of severity. A pothole is an
#: immediate impact hazard, especially for two-wheelers, which dominate Indian
#: road traffic. A crack is a progressive structural problem: serious, but it
#: does not throw a rider.
CLASS_WEIGHTS: dict[DefectClass, float] = {
    DefectClass.POTHOLE: 1.00,
    DefectClass.CRACK: 0.60,
    DefectClass.MANHOLE: 0.85,
}

_SEVERITY_NORM: dict[Severity, float] = {
    Severity.MINOR: 0.25,
    Severity.MODERATE: 0.60,
    Severity.SEVERE: 1.00,
}


@dataclass(frozen=True)
class PriorityConfig:
    w_severity: float = 0.45
    w_traffic: float = 0.25
    w_class: float = 0.20
    w_age: float = 0.10
    #: Days after which an unrepaired defect reaches its maximum age boost.
    age_saturation_days: float = 30.0

    def __post_init__(self) -> None:
        total = self.w_severity + self.w_traffic + self.w_class + self.w_age
        if abs(total - 1.0) > 1e-6:
            raise ValueError(f"priority weights must sum to 1.0, got {total}")

    def to_dict(self) -> dict[str, float]:
        return asdict(self)


DEFAULT_PRIORITY_CONFIG = PriorityConfig()


def age_factor(
    detected_at: datetime,
    now: datetime | None = None,
    config: PriorityConfig = DEFAULT_PRIORITY_CONFIG,
) -> float:
    """Older unrepaired defects drift up the queue so nothing starves."""
    now = now or datetime.now(timezone.utc)
    if detected_at.tzinfo is None:
        detected_at = detected_at.replace(tzinfo=timezone.utc)
    if now.tzinfo is None:
        now = now.replace(tzinfo=timezone.utc)
    days = max(0.0, (now - detected_at).total_seconds() / 86_400.0)
    return min(1.0, days / config.age_saturation_days)


def compute(
    severity: Severity,
    defect_class: DefectClass,
    road_type: str = "arterial",
    detected_at: datetime | None = None,
    now: datetime | None = None,
    config: PriorityConfig = DEFAULT_PRIORITY_CONFIG,
) -> tuple[float, dict[str, float]]:
    """Return ``(score_0_to_100, breakdown)``."""
    severity_norm = _SEVERITY_NORM[severity]
    traffic = ROAD_TYPE_WEIGHTS.get(str(road_type).lower(), ROAD_TYPE_WEIGHTS["unknown"])
    class_weight = CLASS_WEIGHTS.get(defect_class, 0.5)
    age = age_factor(detected_at, now, config) if detected_at else 0.0

    score = 100.0 * (
        config.w_severity * severity_norm
        + config.w_traffic * traffic
        + config.w_class * class_weight
        + config.w_age * age
    )
    breakdown = {
        "severity_norm": severity_norm,
        "traffic_weight": traffic,
        "class_weight": class_weight,
        "age_factor": age,
        "severity_contribution": 100.0 * config.w_severity * severity_norm,
        "traffic_contribution": 100.0 * config.w_traffic * traffic,
        "class_contribution": 100.0 * config.w_class * class_weight,
        "age_contribution": 100.0 * config.w_age * age,
        "total": score,
    }
    return round(score, 2), breakdown


def band(score: float) -> str:
    """Map a numeric priority to the label the dashboard shows."""
    if score >= 75.0:
        return "Critical"
    if score >= 55.0:
        return "High"
    if score >= 35.0:
        return "Medium"
    return "Low"
