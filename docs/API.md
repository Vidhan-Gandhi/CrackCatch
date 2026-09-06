# API reference

Base URL `http://localhost:8000`. Interactive docs at `/docs` (Swagger) and `/redoc`.
All responses are JSON unless noted. Timestamps are ISO-8601 UTC.

---

## System

### `GET /api/health`

Deliberately verbose: it reports whether the **real** database and the **trained** model are in
play, so a demo cannot silently run on the in-memory fallback or the CV baseline.

```json
{
  "status": "degraded",
  "version": "1.0.0",
  "database": "connected",
  "database_backend": "in-memory",
  "detector": "heuristic",
  "detector_is_trained_model": false,
  "defect_count": 19,
  "warnings": ["Running the classical-CV baseline, not a trained YOLOv8 model. ..."]
}
```

`status` is `ok` only when a real MongoDB **and** a trained checkpoint are both in use.

---

## Defects

### `GET /api/defects`

Filterable, sortable, paginated list. Every filter below also applies to
`/api/analytics/*` and `/api/reports/*`, so the table, map, charts and exports always agree.

| Parameter | Type | Notes |
|---|---|---|
| `severity` | repeatable | `Minor` \| `Moderate` \| `Severe` |
| `status` | repeatable | `New` \| `Verified` \| `Scheduled` \| `Repaired` \| `Rejected` |
| `defect_class` | repeatable | `pothole` \| `crack` \| `manhole` |
| `date_from`, `date_to` | ISO datetime | |
| `min_priority` | 0–100 | |
| `source_type` | string | `video` \| `image` \| `camera` \| `crowdsource` |
| `bbox` | `min_lon,min_lat,max_lon,max_lat` | map viewport |
| `search` | string | matches source ref, notes, road type |
| `page`, `page_size` | int | page_size ≤ 500 |
| `sort_by` | enum | `detected_at` \| `priority_score` \| `severity_score` \| `created_at` |
| `sort_dir` | `asc` \| `desc` | |

```bash
curl "localhost:8000/api/defects?severity=Severe&severity=Moderate\
&status=New&min_priority=60&sort_by=priority_score&sort_dir=desc"
```

```json
{ "items": [ /* Defect */ ], "total": 42, "page": 1, "page_size": 50, "pages": 1 }
```

### `GET /api/defects/{id}`

One defect. The full object:

```json
{
  "_id": "66f1a2...",
  "client_id": "9f2c...",
  "defect_class": "pothole",
  "severity": "Severe",
  "severity_score": 0.7398,
  "severity_breakdown": {
    "size_factor": 1.0, "shape_factor": 0.5673, "confidence_factor": 0.3052,
    "size_contribution": 0.55, "shape_contribution": 0.1135,
    "confidence_contribution": 0.0763, "aspect_ratio": 1.6491,
    "total": 0.7398, "_size_basis": "footprint_m2"
  },
  "priority_score": 85.0,
  "priority_band": "Critical",
  "confidence": 0.93,
  "bbox": { "x1": 452.0, "y1": 540.0, "x2": 851.0, "y2": 699.0 },
  "location": { "latitude": 19.07607, "longitude": 72.87766,
                "accuracy_m": 8.0, "source": "simulated" },
  "size": { "area_px": 63441.0, "area_frac": 0.0688,
            "width_m": 0.78, "length_m": 0.70, "area_m2": 0.546,
            "method": "ground_plane", "range_m": 5.6, "reliable": true,
            "confidence_note": "Flat-road projection at ~5.6 m ahead. ..." },
  "status": "New",
  "detected_at": "2026-09-06T08:12:03.881Z",
  "source_type": "video", "source_ref": "demo_drive.mp4", "frame_index": 21,
  "snapshot_path": "9f2c....jpg",
  "road_type": "arterial",
  "history": [ { "status": "New", "at": "...", "by": "pipeline", "note": "..." } ],
  "repair_photo_path": null, "review_label": null
}
```

`location.source` is `device` \| `exif` \| `gpx` \| `simulated` \| `static`. **Always check it** —
demo coordinates are simulated and the dashboard badges them accordingly.

`size.reliable: false` means the metric numbers exist but were rejected as a projection artefact and
did **not** drive severity.

### `PATCH /api/defects/{id}`

Advance the repair workflow, correct the model, or annotate.

```bash
curl -X PATCH localhost:8000/api/defects/$ID \
  -H 'Content-Type: application/json' \
  -d '{"status":"Verified","note":"confirmed on site","by":"ward-officer-12"}'
```

| Field | Notes |
|---|---|
| `status` | Must be a legal transition, else **409** |
| `note` | Stored on the audit trail and as `review_note` (feeds stage 8) |
| `by` | Actor recorded in the audit trail |
| `review_label` | Corrected class — the strongest retraining signal |
| `road_type` | Recomputes traffic weighting context |

Legal transitions:

```
New       → Verified | Rejected
Verified  → Scheduled | Rejected
Scheduled → Repaired | Verified
Repaired  → Scheduled          (re-open a bad repair)
Rejected  → New                (undo a wrong rejection)
```

An illegal jump returns 409 with the allowed set, rather than failing silently.

### `POST /api/defects`

Create a defect directly (used by tests and integrations). Body is `DefectCreate`. Returns 201.
Like every creation path, it broadcasts `defect.created` and, above the threshold,
`alert.high_priority`.

### `DELETE /api/defects/{id}` → 204

### `GET /api/defects/near?lat=&lon=&radius_m=`

Proximity query, `2dsphere`-backed with a haversine fallback.

### `POST /api/defects/{id}/repair-photo`

Multipart `file`. Attaches the after-repair photo for before/after verification.

### `GET /api/defects/{id}/explain`

Returns `image/jpeg` — an explainability overlay on the defect's snapshot.
The **`X-Explain-Method`** response header states which method *actually* produced it — it is taken
from the explainer's return value, not from whether a checkpoint happens to exist. Values are
`grad-cam`, or a saliency fallback clearly labelled *not* Grad-CAM (used when no checkpoint is
loaded, or when a CAM could not be computed). Read the header rather than assuming.

---

## Analytics

### `GET /api/analytics/summary`

Accepts every defect filter, plus `days` (time-series window, default 30).

```json
{
  "total_defects": 19, "open_defects": 17, "repaired_defects": 1, "severe_open": 15,
  "mean_priority": 74.2,
  "by_severity": [ {"severity":"Minor","count":1}, ... ],
  "by_status":   [ {"status":"New","count":17}, ... ],
  "by_class":    [ {"defect_class":"pothole","count":9}, ... ],
  "over_time":   [ {"date":"2026-09-06","count":19,"severe":16} ],
  "top_priority": [ /* 10 highest-priority OPEN defects */ ],
  "generated_at": "..."
}
```

`top_priority` excludes `Repaired` and `Rejected` — it is a worklist, not a history.

### `GET /api/analytics/heatmap?cell_size_deg=0.002`

Road-health heatmap. Snaps defects to a lat/lon grid (0.002° ≈ 200 m) and scores each cell by a
**severity-weighted** count (Minor 1, Moderate 2, Severe 4), so damage density is measured rather
than record density.

```json
{ "cells": [ { "latitude": 19.077, "longitude": 72.879, "count": 7,
               "severe_count": 5, "mean_priority": 81.4, "intensity": 1.0 } ],
  "cell_size_deg": 0.002, "max_count": 7 }
```

`intensity` is normalised against the busiest cell **in the current filter** and is what the Leaflet
heat layer consumes. Coordinates are cell **centres**.

---

## Ingestion

### `POST /api/ingest/run?wait=false`

Run the pipeline over a server-side source.

```bash
curl -X POST "localhost:8000/api/ingest/run?wait=true" \
  -H 'Content-Type: application/json' \
  -d '{"source":"data/samples/demo_drive.mp4","target_fps":2,"road_type":"arterial",
       "gps_method":"simulated","gps_start_lat":19.076,"gps_start_lon":72.8777}'
```

`source` is **sandboxed to the project directory** — an absolute path outside it returns 400.
`gps_method` is `simulated` \| `static` \| `gpx` \| `exif`.

With `wait=false` (the default) the job is queued and the response returns immediately; watch
progress on the WebSocket. `wait=true` blocks and returns the finished job with measured stats:

```json
{ "job_id":"...", "status":"completed", "records_stored":19, "detector":"yolov8",
  "stats": { "frames_read":38, "frames_processed":38, "raw_detections":82,
             "records_emitted":19, "records_deduped":63,
             "inference_fps":16.74, "end_to_end_fps":11.64 } }
```

### `GET /api/ingest/jobs` · `GET /api/ingest/jobs/{job_id}`

### `POST /api/ingest/upload`

Multipart: `file` (video or image), optional `latitude`, `longitude`, `road_type`.

### `POST /api/ingest/crowdsource`

The citizen PWA endpoint. Multipart: `file` (image only), optional `latitude`, `longitude`,
`accuracy_m`, `note`, `road_type`.

Location precedence: supplied coordinates → photo EXIF → configured demo origin, each tagged with
its own `source`. Returns a friendly acknowledgement:

```json
{ "accepted": true, "defects_found": 1, "defects": [ /* Defect */ ],
  "message": "Thanks! 1 defect(s) reported... Highest severity: Severe (repair priority 85/100).",
  "upload_id": "..." }
```

---

## Reports

### `GET /api/reports/csv` · `GET /api/reports/pdf?ward=Ward%2012`

Both accept every defect filter. CSV is a flat export; the PDF is a municipal work-order — summary
block, then defects **ordered by repair priority**, severity-tinted, with a method-and-limitations
page. Sizes flagged unreliable are marked `*`.

---

## Live feed

### `WS /ws/defects`

| Event | Payload |
|---|---|
| `defect.created` | the full Defect |
| `defect.updated` | the full Defect |
| `defect.deleted` | `{ "_id": "..." }` |
| `alert.high_priority` | Defect with `priority_score ≥ ALERT_PRIORITY_THRESHOLD` |
| `job.started` / `job.completed` / `job.failed` | job info |

Send `"ping"` to receive `{"type":"pong"}`. Recent events are replayed on connect so a dashboard
opened mid-run is not blank.

```js
const ws = new WebSocket('ws://localhost:8000/ws/defects')
ws.onmessage = (e) => {
  const { type, payload } = JSON.parse(e.data)
  if (type === 'alert.high_priority') console.warn('urgent', payload)
}
```

---

## Media

`GET /media/snapshots/{file}` · `/media/repairs/{file}` · `/media/uploads/{file}`

Served from local disk. In production these belong in object storage behind signed URLs.

---

## Authentication

Disabled by default (`AUTH_ENABLED=false`). When enabled, send `Authorization: Bearer <token>`.

| Role | Permissions |
|---|---|
| `authority` | Full — read, workflow, ingestion |
| `transport` | Read + analytics |
| `planner` | Read + analytics |
| `citizen` | Crowdsource submission only |

401 for a missing/invalid token, 403 for an insufficient role. This is demo-grade — see
[SECURITY.md](SECURITY.md).

---

## Errors

| Code | Meaning |
|---|---|
| 400 | Bad request (e.g. a source path outside the project) |
| 404 | Not found |
| 409 | Illegal workflow transition |
| 413 | Upload exceeds `MAX_UPLOAD_MB` |
| 415 | Unsupported media type |
| 422 | Validation error (FastAPI/Pydantic detail) |
| 500 | Unhandled error — traceback logged server-side, terse message returned |
