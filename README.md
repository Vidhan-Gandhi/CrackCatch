# CrackCatch

**Real-time pothole and road-damage detection, severity estimation, GPS geo-tagging, and a municipal repair workflow.**

CrackCatch detects potholes and cracks in road video, estimates how bad each one is, pins it to a
map, stores it in a cloud database, and surfaces it on a live dashboard where a road authority can
verify it, schedule it, and mark it repaired — closing the loop from *"damage exists"* to
*"someone with authority can see it and act."*

```
┌────────────┐  ┌─────────────┐  ┌────────────┐  ┌──────────────┐
│ 1. Capture │─▶│ 2. Prepro-  │─▶│ 3. YOLOv8  │─▶│ 4. Severity  │
│ dashcam /  │  │    cessing  │  │  detection │  │  & size      │
│ video /    │  │ sample,     │  │ pothole,   │  │ estimation   │
│ camera /   │  │ resize,     │  │ crack      │  │              │
│ phone      │  │ CLAHE       │  │            │  │              │
└────────────┘  └─────────────┘  └────────────┘  └──────┬───────┘
                                                        ▼
┌──────────────┐  ┌────────────────┐  ┌──────────────┐  ┌─────────────┐
│ 8. Retrain   │◀─│ 7. Authority   │◀─│ 6. Road      │◀─│ 5. GPS geo- │
│  from        │  │    dashboard   │  │  damage DB   │  │   tagging   │
│  feedback    │  │ map, filters,  │  │  (MongoDB)   │  │             │
│              │─▶│ repair status  │  │              │  │             │
└──────────────┘  └────────────────┘  └──────────────┘  └─────────────┘
```

---

## Quick start

### Option A — Docker (everything, one command)

```bash
git clone <your-repo-url> CrackCatch && cd CrackCatch
python3 data/scripts/make_demo_video.py     # generate the sample clip (~17 MB)
docker compose up --build
```

| Service         | URL                             |
| --------------- | ------------------------------- |
| Dashboard       | <http://localhost:5173>         |
| API docs        | <http://localhost:8000/docs>    |
| Health / status | <http://localhost:8000/api/health> |
| MongoDB         | `localhost:27017`               |

Then push the sample clip through the pipeline and watch the dashboard fill up live:

```bash
python3 scripts/run_demo.py
```

### Option B — Local development

```bash
make setup          # venv + Python deps + npm deps + sample clip
make backend        # terminal 1 — API on :8000
make dashboard      # terminal 2 — dashboard on :5173
make demo           # terminal 3 — replay the clip through the pipeline
```

`make help` lists every task.

> **No MongoDB running?** The backend falls back to an in-process database so a demo never dies on
> a missing service. It says so loudly in the logs, reports `database_backend: "in-memory"` on
> `/api/health`, and the dashboard shows a banner. Data is lost on restart. Set
> `ALLOW_MEMORY_DB_FALLBACK=false` to make a missing database a hard failure instead.

---

## Repository layout

```
model/                    Stages 1-5 + 8: detection, scoring, geo-tagging, training
  crackcatch_model/       Importable package - no web or database dependencies
  scripts/                train.py, evaluate.py, benchmark_fps.py, retrain_from_feedback.py
  weights/                Checkpoints (git-ignored; see "Training")
backend/                  Stage 6-7: FastAPI service, MongoDB layer, tests
  app/{api,core,db,models,services}/
  tests/                  152 tests
frontend/                 Stage 7: React + Vite dashboard, Leaflet map, Recharts
data/                     Dataset prep scripts (never raw data) and the demo clip
  scripts/prepare_rdd2022.py         RDD2022 (VOC) -> YOLO, seeded split
  scripts/make_synthetic_dataset.py  Offline synthetic training set
  scripts/make_demo_video.py         Synthetic dashcam clip for demos
docs/                     Architecture, severity method, model card, API, governance
scripts/run_demo.py       End-to-end demo driver
```

The four layers are genuinely decoupled: `crackcatch_model` imports no web framework and no
database driver, so the detection stack runs unchanged in a notebook, a CLI, or on an edge device.
Swapping the detector, the severity heuristic, the storage layer, or the dashboard touches exactly
one of them.

---

## The eight pipeline stages

| # | Stage | Where it lives | Notes |
|---|-------|----------------|-------|
| 1 | Video/image capture | `model/crackcatch_model/sources.py` | Dashcam file, uploaded video, live camera/RTSP, or a directory of test images — one interface, four implementations |
| 2 | Frame preprocessing | `preprocess.py` | Configurable sampling FPS, letterbox resize, bilateral denoise, CLAHE lighting normalisation, motion-blur rejection |
| 3 | AI detection engine | `detector.py` | YOLOv8 (Ultralytics). Classical-CV fallback so a fresh clone runs before training |
| 4 | Severity & size | `calibration.py`, `severity.py` | Ground-plane projection for metric size; transparent weighted severity score |
| 5 | GPS geo-tagging | `geotag.py` | Device GPS, photo EXIF, recorded GPX track, or a simulated route for demos |
| 6 | Road damage database | `backend/app/db/` | MongoDB with 2dsphere index; also accumulates reviewed detections for retraining |
| 7 | Authority dashboard | `frontend/` | Live map + heatmap, filters, repair workflow, analytics, PDF/CSV export |
| 8 | Continuous improvement | `model/scripts/retrain_from_feedback.py` | export → retrain → evaluate → promote, gated on measured improvement |

---

## What the dashboard does

- **Live map** — every defect as a pin, coloured by severity and sized by repair priority.
- **Road-health heatmap** — severity-weighted damage density on a ~200 m grid. A cell with three
  severe potholes burns hotter than one with six hairline cracks, because it weights *damage*, not
  record count.
- **Filterable table** — severity, status, type, date range, minimum priority, free text; sortable;
  every filter applies to the map, the charts, and the exports simultaneously.
- **Defect detail** — snapshot with bounding box, GPS, estimated size, and a full breakdown of *why*
  the severity came out the way it did.
- **Repair workflow** — `New → Verified → Scheduled → Repaired`, with an audit trail. Illegal
  transitions are rejected by the API with a 409, not silently ignored.
- **Before/after verification** — attach a photo when a defect is marked Repaired.
- **Explainability** — Grad-CAM overlay showing what drove a detection.
- **Alerts** — high-priority defects raise a banner the moment they are detected.
- **Reports** — CSV, and a municipal work-order PDF ordered by repair priority.
- **Citizen reporting** (`/report`) — an installable PWA where a driver photographs a pothole; the
  phone's geolocation is attached and the photo enters the *same* pipeline as dashcam footage.

---

## Severity and repair priority

Both scores are plain weighted sums of documented factors, not learned black boxes, so they can be
explained and defended. Full derivation in **[docs/SEVERITY.md](docs/SEVERITY.md)**.

```
severity = 0.55·size + 0.20·shape(aspect ratio) + 0.25·confidence
           < 0.35 Minor      < 0.62 Moderate      else Severe

priority = 100 · (0.45·severity + 0.25·traffic(road type)
                + 0.20·class + 0.10·age)
```

Repair priority answers the gap named in the review paper — detection-only systems tell an authority
that 400 defects exist but not which one to fix on Monday morning.

Every API response carries the full factor breakdown, and the dashboard renders it.

---

## Training

### On RDD2022 (the real dataset)

RDD2022 is the dataset named in the project scope and includes an India subset. It is **not**
redistributed here.

```bash
# 1. Download from https://github.com/sekilab/RoadDamageDetector
# 2. Convert to YOLO format with a seed-controlled split
python data/scripts/prepare_rdd2022.py \
    --source ~/datasets/RDD2022 --output data/rdd2022_yolo \
    --countries India Japan --val-split 0.2 --seed 42

# 3. Fine-tune
python model/scripts/train.py --data data/rdd2022_yolo/data.yaml --epochs 60

# 4. Report mAP@0.5, per-class P/R, confusion matrix, achieved FPS
python model/scripts/evaluate.py --data data/rdd2022_yolo/data.yaml --also-cpu-fps

# 5. yolov8n vs yolov8s accuracy/speed trade-off
python model/scripts/benchmark_fps.py \
    --weights model/weights/crackcatch.pt model/weights/yolov8s.pt
```

### Offline (no dataset download)

```bash
make dataset      # render a synthetic YOLO dataset
make train        # fine-tune yolov8n on it
make evaluate
```

This proves the training pipeline runs end to end and gives the demo real two-class YOLOv8 weights.
**It does not measure real-world accuracy** — see [docs/MODEL_CARD.md](docs/MODEL_CARD.md).

### Continuous improvement (stage 8)

```bash
python model/scripts/retrain_from_feedback.py all --epochs 30
```

Exports the detections an authority verified or rejected on the dashboard, fine-tunes on them,
evaluates the candidate against the deployed checkpoint, and **promotes only if mAP@0.5 actually
improves**. Rejected detections become hard negatives, which is what teaches the model to stop
firing on tar patches and shadows.

---

## Testing

```bash
make test     # 152 tests
```

Covers the severity scoring (monotonicity, weight normalisation, threshold partitioning,
reconstructible breakdowns), the projection geometry (cross-checked against the closed-form
formula), the repository (workflow transitions, filters, analytics, heatmap weighting, retraining
export), every API endpoint (including path-traversal rejection, role enforcement, and the live
WebSocket feed), and the pipeline (preprocessing round-trips, deduplication, source dispatch).

Tests run against an in-process database — no Docker, no network.

---

## Known limitations

Stated plainly, because a research prototype that overclaims is worse than one that does not.

**Severity and size are estimates, not measurements.** Size comes from projecting the bounding box
onto an assumed flat road plane using the camera's height and pitch. It recovers *surface extent*
only. **Depth is not measured and cannot be** from a single monocular camera — a 5 cm and a 25 cm
deep pothole with the same opening are indistinguishable. Real depth needs stereo, structured light,
or LiDAR. Estimates beyond ~25 m, or that imply an implausible footprint, are automatically flagged
`reliable: false` and excluded from severity scoring rather than quietly reported.

**Flat-road assumption.** Camber, speed bumps, and slopes bias the projection. Camera height and
pitch are configured, not calibrated per vehicle.

**Generalisation to Indian roads is unproven here.** RDD2022 and GAPs are not India-specific. The
converter supports RDD2022's India subset and the README documents its use, but no India-specific
evaluation has been run in this repository. Treat any accuracy figure as dataset-specific until it
is reproduced on Indian road imagery.

**The shipped checkpoint is trained on synthetic data.** It exists so the demo has a working
two-class detector. Synthetic asphalt has none of the confounders that make this problem hard — wet
patches, tar repairs, cable shadows, painted markings, debris. Its metrics measure the pipeline, not
the detector's fitness for deployment.

**The classical-CV fallback is a baseline, not a model.** When no checkpoint is present the system
runs a blackhat/darkness detector so the pipeline stays runnable. It finds dark road-coloured blobs,
which correlates with potholes but also with oil stains and shadows, and its crack recall is weak.
It is tagged `detector="heuristic"` on every detection and flagged on `/api/health`.

**Authentication is demo-grade.** Static bearer tokens, no user store, no expiry, off by default.
It demonstrates that the system is *designed* around the four stakeholder roles. A real deployment
needs an identity provider — see [docs/SECURITY.md](docs/SECURITY.md).

**Traffic weighting is an assumption, not a measurement.** Repair priority uses a static per-road-type
weight because there is no traffic-count feed. Substituting real AADT data is a one-line change to
`ROAD_TYPE_WEIGHTS`.

**Analytics are computed in Python, not as MongoDB aggregation pipelines.** Correct and fast at
dashboard scale (thousands of defects); the aggregation rewrite is the documented scaling path.

**Simulated GPS in demos.** Demo footage has no GPS, so a synthetic route is generated. Every such
point is tagged `source: "simulated"`, exposed by the API, and badged in the UI.

---

## Documentation

| Document | Contents |
| -------- | -------- |
| [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) | Stage-by-stage design, data flow, module boundaries, scaling path |
| [docs/SEVERITY.md](docs/SEVERITY.md) | Full derivation of the size projection and severity/priority scoring |
| [docs/MODEL_CARD.md](docs/MODEL_CARD.md) | Intended use, training data, metrics, failure modes, ethics |
| [docs/API.md](docs/API.md) | Endpoint reference with examples |
| [docs/DATA_GOVERNANCE.md](docs/DATA_GOVERNANCE.md) | Location privacy, retention, what a production system would need |
| [docs/SECURITY.md](docs/SECURITY.md) | Auth model, threat notes, hardening checklist |
| [docs/DEMO.md](docs/DEMO.md) | Presentation runbook and likely panel questions |

---

## Tech stack

| Layer | Technology |
| ----- | ---------- |
| Detection | YOLOv8 (Ultralytics), PyTorch |
| Video/image processing | OpenCV |
| Backend API | FastAPI (async), Uvicorn |
| Database | MongoDB (Motor), 2dsphere geospatial index |
| Dashboard | React 18 (Vite), Leaflet + leaflet.heat, Recharts |
| Reports | ReportLab |
| Mobile capture | PWA (phone camera + Geolocation API) |
| Deployment | Docker + Docker Compose |

---

## License and attribution

Research prototype built as a final-year B.Tech major project. RDD2022 is © its authors and is
subject to its own licence; it is referenced, never redistributed. Map tiles © OpenStreetMap
contributors, © CARTO.
