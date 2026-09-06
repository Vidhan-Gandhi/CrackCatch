# Architecture

CrackCatch implements the mandatory eight-stage pipeline, with each stage depending only on the one
before it. Any stage can be replaced without touching the rest — the detector is the obvious
example, but the same is true of the severity heuristic, the storage layer and the dashboard.

```
                        ┌──────────────────────────────────────────┐
                        │  model/crackcatch_model  (no web, no DB) │
                        └──────────────────────────────────────────┘
 dashcam ─┐
 video   ─┤   ┌─────────┐   ┌──────────────┐   ┌───────────┐   ┌────────────┐   ┌──────────┐
 camera  ─┼──▶│1 sources│──▶│2 preprocess  │──▶│3 detector │──▶│4 severity  │──▶│5 geotag  │
 images  ─┘   │         │   │ CLAHE,resize │   │  YOLOv8   │   │ calibration│   │ GPS/EXIF │
              └─────────┘   └──────────────┘   └───────────┘   └────────────┘   └────┬─────┘
                                                                                     │
                                              DefectRecord stream ◀──────────────────┘
                                                       │
              ┌────────────────────────────────────────┼─────────────────────────────┐
              ▼                                        ▼                             ▼
      ┌───────────────┐                    ┌────────────────────┐          ┌──────────────────┐
      │6 MongoDB      │◀──────────────────▶│  FastAPI backend   │─────────▶│ WebSocket feed   │
      │  defects      │                    │  /api/*            │          │ /ws/defects      │
      │  ingest_jobs  │                    └────────────────────┘          └────────┬─────────┘
      └───────┬───────┘                                                             │
              │                                    ┌────────────────────────────────▼───────┐
              │  reviewed detections               │7 React dashboard                       │
              └───────────────────────────────────▶│  map + heatmap, filters, workflow,     │
                                   ▲                │  analytics, PDF/CSV, citizen PWA      │
                                   │                └────────────────────┬───────────────────┘
                        ┌──────────┴──────────┐                          │
                        │8 retrain_from_      │◀─────────────────────────┘
                        │  feedback.py        │   verify / reject / correct
                        │ export→train→eval→  │
                        │ promote             │
                        └─────────────────────┘
```

---

## Module boundaries

The single most important structural rule: **`model/crackcatch_model` imports no web framework and
no database driver.** It depends only on NumPy, OpenCV and (for stage 3) Ultralytics. Consequences:

- the detection stack runs unchanged in a notebook, a CLI, a batch job, or on a Raspberry Pi;
- the backend can be replaced without touching the ML code, and vice versa;
- tests for severity and geometry need no database and no HTTP server, so they run in milliseconds.

The contract between the layers is `DefectRecord` — a plain dataclass with `to_dict()`. The backend
re-validates it with Pydantic before it reaches MongoDB, so a change on either side of the boundary
fails loudly rather than corrupting data.

| Layer | Path | Depends on |
|---|---|---|
| Detection & scoring | `model/crackcatch_model/` | numpy, opencv, ultralytics |
| Training & evaluation | `model/scripts/` | + the backend (only stage 8, to read feedback) |
| API & storage | `backend/app/` | fastapi, motor, + `crackcatch_model` |
| Dashboard | `frontend/src/` | the HTTP + WebSocket API only |

---

## Stage-by-stage

### 1. Capture — `sources.py`

One `FrameSource` interface, four implementations: `VideoFileSource`, `ImageDirectorySource`,
`CameraSource` (USB / RTSP / HTTP), `SingleImageSource` (a crowdsourced photo). `open_source()`
dispatches on the argument, which is what lets the CLI, the API and the demo runner all take a
single `--input`. Sampling is frame-strided for files (derived from the clip's native FPS) and
time-based for live streams, which have no fixed frame budget.

### 2. Preprocessing — `preprocess.py`

Sample → letterbox resize → bilateral denoise → CLAHE.

Two deliberate choices worth defending:

- **CLAHE, not global histogram equalisation.** Indian road footage is routinely half sunlit and
  half in deep building shadow. A global equalisation is dominated by the bright half and leaves the
  shaded half — where potholes hide — still crushed. CLAHE equalises locally and clips the contrast
  gain, so shadowed road texture becomes visible without amplifying sensor noise into false crack
  edges.
- **Bilateral filter, not Gaussian blur.** A Gaussian would smear the thin high-frequency edges that
  *are* the crack signal. The bilateral filter is edge-preserving.

Letterboxing records `(scale, pad)` so detections map back to original-frame pixel coordinates
exactly — asserted by a round-trip test.

### 3. Detection — `detector.py`

`YOLOv8Detector` wraps Ultralytics, resolving device automatically (CUDA → MPS → CPU) and mapping
checkpoint class names onto canonical classes, ignoring anything irrelevant (a COCO checkpoint's 80
classes, for example).

`build_detector()` prefers an explicit weights path, then `model/weights/crackcatch.pt`, then falls
back to `HeuristicDetector` — **loudly**, never silently. NMS is per class, not class-agnostic: a
crack running past the rim of a pothole is a separate real defect, and suppressing across classes
would delete whichever scored lower.

### 4. Severity & size — `calibration.py`, `severity.py`, `priority.py`

Fully derived in [SEVERITY.md](SEVERITY.md). The key architectural point is that both scorers take a
frozen config dataclass and return a factor-by-factor breakdown, so the score is reconstructible and
tunable without code changes.

### 5. Geo-tagging — `geotag.py`

Four providers behind one interface: device GPS (the PWA's browser geolocation), photo EXIF,
recorded GPX track (interpolated by video time), and a simulated route for demo footage. A provider
is consulted **once per frame, only when a defect is confirmed** — never sampled continuously —
which is what makes the data-governance guarantee structural rather than a policy promise.

### 6. Storage — `backend/app/db/`

MongoDB via Motor. `DefectRepository` holds every query, so the routers stay thin and the storage
layer is swappable.

- `client_id` (generated by the pipeline) is unique, making ingestion **idempotent** — re-running the
  same clip updates rather than duplicates.
- `location_geo` is a GeoJSON Point backing a `2dsphere` index for proximity queries; it is stripped
  from API responses as an internal field.
- Indexes on `detected_at`, `(status, severity)`, `priority_score` and `defect_class` back the
  dashboard's filters and sorts.
- Status transitions are validated against a state machine; an illegal jump raises, and the API
  turns it into a 409.

**In-memory fallback.** If MongoDB is unreachable and `ALLOW_MEMORY_DB_FALLBACK` is on, the service
starts on an in-process double so a demo survives a forgotten `docker compose up`. It logs a
warning, reports `database_backend: "in-memory"` on `/api/health`, and the dashboard renders a
banner. In Docker the flag is `false` — a missing database there is a real failure.

### 7. Dashboard — `frontend/`

React 18 + Vite. Leaflet drives the map imperatively (the heat layer is a vanilla plugin, and mixing
declarative and imperative layer management causes flicker on every update). Recharts draws the
analytics. The live feed is a WebSocket with exponential-backoff reconnection, so a backend restart
mid-demo heals itself.

Bursts are debounced: an ingestion run emitting 50 detections triggers one refresh, not 50.

### 8. Continuous improvement — `model/scripts/retrain_from_feedback.py`

`export → retrain → evaluate → promote`. Only human-reviewed detections are exported; untouched
`New` rows carry no signal, and training on the model's own unreviewed output would amplify its bias
rather than correct it. Rejected detections become hard negatives, which is what actually teaches the
model to stop firing on tar patches.

Promotion is gated on a measured mAP@0.5 improvement over the incumbent. A candidate that is not
better is kept on disk and not deployed.

---

## Request flow: a live ingestion run

```
POST /api/ingest/run
  → path is sandboxed to the project directory  (rejects /etc/passwd with a 400)
  → job persisted as "queued", returned immediately
  → BackgroundTask: PipelineService.run_ingest
      → asyncio.to_thread(...)          CPU-bound decode + inference off the event loop
      → for each DefectRecord:
          repository.create()           idempotent on client_id
          publish_defect_created()      → WebSocket "defect.created"
                                        → "alert.high_priority" if priority ≥ threshold
      → job marked "completed" with measured stats
```

Every creation path — pipeline, crowdsourced photo, direct API POST — funnels through
`publish_defect_created()`, so the alert banner behaves identically regardless of how a defect
arrived. (An earlier version broadcast alerts only from the pipeline path; a test caught it.)

---

## Scaling path

Deliberate simplifications, and what each would become in production:

| Now | At scale |
|---|---|
| Analytics aggregated in Python over the filtered set | MongoDB aggregation pipeline (`$group`/`$bucket`); the Python path exists so the same code works against the in-memory fallback, whose `$group` support is partial |
| Heatmap gridded per request | Pre-aggregated tiles, refreshed on write |
| Snapshots on local disk via `StaticFiles` | S3/GCS behind a CDN with signed URLs |
| `BackgroundTasks` for ingestion | Celery/RQ with a broker, so jobs survive a restart and scale horizontally |
| Single backend process | Multiple workers behind a load balancer; the WebSocket broadcaster moves to Redis pub/sub so all workers see every event |
| Static bearer tokens | OIDC against the municipality's identity provider |
| Inference in the API process | A dedicated inference service, or on-vehicle edge inference (Jetson) uploading only detections |

---

## Design decisions worth knowing

**Why deduplicate in the pipeline rather than the database?** A pothole filmed at 2 fps from a moving
car appears in several consecutive frames. Writing one row per frame would inflate every dashboard
count and make the heatmap meaningless. Deduplication needs the frame sequence and the GPS track,
both of which exist in the pipeline and not in the database.

**Why is the WebSocket send wrapped in a timeout?** A wedged client — a paused browser tab, a
half-open TCP connection — would otherwise block the broadcast loop, and with it the ingestion
request that triggered it. Sends that exceed 2 s drop the connection instead.

**Why is severity computed at detection time and stored, rather than derived on read?** So that a
later change to the heuristic cannot silently rewrite the history an authority already acted on. The
stored breakdown is an audit record of the decision that was actually made.
