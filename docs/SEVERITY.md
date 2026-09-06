# Severity, size estimation and repair priority

This document derives every number CrackCatch puts on a defect. The project scope requires the
scoring logic to be *exposed, not a black box*, so that it can be explained in a viva. Nothing here
is learned; every constant is a documented choice you can argue with.

---

## 1. Pixels to metres — the ground-plane projection

**Implementation:** `model/crackcatch_model/calibration.py`

### The honest limitation, first

A single camera cannot measure depth. Projecting a bounding box onto an assumed road plane recovers
**how much road surface a defect covers**, not **how deep it is**. A 5 cm-deep and a 25 cm-deep
pothole with the same opening produce identical estimates. Depth requires stereo, structured light,
or LiDAR. Everything below is a surface-extent estimate.

### Coordinate frames

Camera frame: `x` right, `y` down, `z` along the optical axis.
World frame: `X` right, `Y` up, `Z` horizontally forward, origin on the road directly below the camera.

The camera sits at `Y = h` (its mounting height) and is pitched `θ` downwards, so its basis vectors
expressed in world coordinates are:

```
x_c = ( 1,       0,       0      )
y_c = ( 0,  -cos θ,  -sin θ      )
z_c = ( 0,  -sin θ,   cos θ      )
```

### Back-projection

A pixel `(u, v)` corresponds to the camera ray `(a, b, 1)` where

```
a = (u - cx) / f      b = (v - cy) / f      f = (W/2) / tan(HFOV/2)
```

In world coordinates that ray is `d = a·x_c + b·y_c + z_c`:

```
d_X = a
d_Y = −(b·cos θ + sin θ)
d_Z =   cos θ − b·sin θ
```

Intersecting `camera + s·d` with the road plane `Y = 0`:

```
s = h / (b·cos θ + sin θ)
```

The denominator is the ray's downward component. When it is `≤ 0` the pixel is at or above the
horizon and **no finite intersection exists** — the code returns `None` rather than a nonsense
number. The horizon sits at image row `cy − f·tan θ`.

Ground coordinates are then `X = s·d_X` (lateral) and `Z = s·d_Z` (forward range).

### Measuring a bounding box

- **Width** — project the box's bottom-left and bottom-right corners. The bottom edge is where the
  defect meets the road, so it is the only edge whose projection is physically meaningful.
- **Length** — project bottom-centre and top-centre and take the difference in forward range. If the
  top edge is above the horizon, length falls back to width (a near-circular footprint assumption),
  and the note says so.

### Reliability gating

Ground sampling distance grows roughly with the square of range. At 25 m a single pixel already
spans several centimetres of road, so a 2–3 px box error swings the area estimate by more than the
defect's own size. Two guards:

| Guard | Default | Effect |
| ----- | ------- | ------ |
| `max_range_m` | 40 m | Beyond this, no projection at all |
| `max_reliable_range_m` | 25 m | Metric numbers computed but `reliable = false` |
| `max_plausible_area_m2` | 12 m² | A larger footprint is a projection artefact; `reliable = false` |

When `reliable` is false, severity **ignores the metric numbers** and falls back to the
frame-relative pixel area. The dashboard marks such sizes, and the PDF report annotates them with
`*`. This matters: before the guard existed, a box near the horizon produced a "63 m² pothole".

### Alternative: reference-object calibration

For a single crowdsourced photo, camera geometry is unknown. `ReferenceObjectCalibration` recovers a
uniform metres-per-pixel scale from a feature of known width (an Indian arterial lane marking is
3.5 m). Valid only at the reference object's depth — no perspective correction — which is acceptable
for a near-field photo taken looking down at one pothole.

---

## 2. Severity score

**Implementation:** `model/crackcatch_model/severity.py`

```
severity_score = 0.55 · size_factor
               + 0.20 · shape_factor
               + 0.25 · confidence_factor
```

The three inputs are exactly those named in the project scope: bounding-box area, aspect ratio, and
detection confidence. Weights sum to 1.0 — enforced by a constructor check that raises.

### size_factor — how much road is affected

Dominant term, because a large defect is unambiguously worse. Uses the metric footprint when the
calibration vouched for it, otherwise the frame-relative pixel area.

| Class | Measure | 0.0 at | 1.0 at |
| ----- | ------- | ------ | ------ |
| Pothole / manhole | footprint m² | 0.05 m² | 0.60 m² |
| Crack | run length m | 0.50 m | 4.00 m |
| *fallback (any)* | frame area fraction | 0.002 | 0.080 |

A crack's severity is its **run length**, not its area: a 4 m crack 2 cm wide is a resurfacing
problem, while its area is trivial.

Rationale for the pothole endpoints: ~0.6 m across is roughly where a two-wheeler — which dominates
Indian road traffic — is at real risk rather than merely inconvenienced.

### shape_factor — does this look like a bad instance of its class?

Class-conditional, because "bad" means opposite things for the two classes:

- **Pothole:** compact is worse. `1.0` at aspect ratio 1.0, falling to `0.0` at 2.5. An elongated
  dark blob on the road is more often a patch, a rut, or a shadow than a hole.
- **Crack:** elongated is worse. `0.0` at aspect ratio 1.5, rising to `1.0` at 6.0. Linear extent
  *is* the damage.

### confidence_factor

Detector confidence re-ramped from the acceptance threshold (0.25) to 1.0, so a barely-accepted
detection contributes ≈0 rather than ≈0.4.

### Thresholds

```
score < 0.35  →  Minor
score < 0.62  →  Moderate
otherwise     →  Severe
```

Enforced ordering: `0 < minor_max < moderate_max < 1`.

### Transparency

Every call returns a breakdown carrying each factor, each weighted contribution, the aspect ratio,
the total, and `_size_basis` naming which measure the size factor used. The contributions sum to the
score exactly — asserted in `test_severity.py::test_breakdown_is_transparent_and_adds_up`. The
dashboard renders this on the defect detail page.

Worked example (a 0.3 m² pothole, aspect ratio 1.6, confidence 0.75):

```
size(footprint_m2)=0.45 ×0.55 → 0.248
shape(AR=1.60)    =0.60 ×0.20 → 0.120
confidence        =0.67 ×0.25 → 0.167
                                -----
                          total  0.535  → Moderate
```

---

## 3. Repair priority

**Implementation:** `model/crackcatch_model/priority.py`

The review paper's stated gap: detection-only systems do not support repair prioritisation. A
0–100 score, again a transparent weighted sum:

```
priority = 100 · (0.45·severity_norm + 0.25·traffic + 0.20·class + 0.10·age)
```

| Term | Values |
| ---- | ------ |
| `severity_norm` | Minor 0.25, Moderate 0.60, Severe 1.00 |
| `traffic` | highway 1.00, arterial 0.80, collector 0.55, residential 0.35, service 0.20 |
| `class` | pothole 1.00, manhole 0.85, crack 0.60 |
| `age` | ramps 0 → 1 over 30 days since detection |

**`traffic` is a static assumption, not a measurement.** There is no traffic-count feed, and saying
so is more honest than implying the number is observed. Substituting real AADT per road segment is a
one-line change to `ROAD_TYPE_WEIGHTS`.

`age` exists so that a low-severity defect on a quiet street eventually rises rather than starving
at the bottom of the queue forever.

Bands: `≥75 Critical`, `≥55 High`, `≥35 Medium`, else `Low`. Defects at or above
`ALERT_PRIORITY_THRESHOLD` (default 75) raise a dashboard alert the moment they are detected.

---

## 4. Tuning

Both scorers take a frozen config dataclass, so a municipality with different standards changes the
constants rather than the code:

```python
from crackcatch_model.severity import SeverityConfig, classify

strict = SeverityConfig(minor_max=0.25, moderate_max=0.50,
                        pothole_area_m2_high=0.40)
severity, score, breakdown = classify(cls, bbox, conf, size, config=strict)
```

The property-based tests (monotonicity in size and confidence, threshold partitioning, weight
normalisation) hold for any valid config, so re-tuning cannot silently break the ordering guarantees.
