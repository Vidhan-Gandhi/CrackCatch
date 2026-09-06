# CrackCatch — start here

Real-time pothole and road-damage detection with severity estimation, GPS geo-tagging and a
municipal repair workflow. Full documentation is in [README.md](README.md).

---

## What is in this archive

Source, tests, documentation, the full git history, and **the trained YOLOv8 model**
(`model/weights/crackcatch.pt`, 6 MB) — that one is included because it takes ~15 minutes of
training to reproduce.

## What was left out, and how to restore it

Anything a package manager or a single command can produce was stripped, to keep the archive
around 8 MB instead of 1.5 GB.

| Not included | Why | Restore with |
|---|---|---|
| `.venv/` (1.3 GB) | Python packages | `make deps` |
| `frontend/node_modules/` (105 MB) | npm packages | `make frontend-deps` |
| `data/samples/demo_drive.mp4` (16 MB) | Generated sample clip | `make sample` |
| `data/synthetic_yolo/` (53 MB) | Generated training set | `make dataset` |
| `model/weights/yolov8n.pt`, `yolov8s.pt` (28 MB) | Public COCO checkpoints | downloaded automatically on first use |
| `model/runs/`, `storage/` | Training output and runtime snapshots | recreated on use |

`make setup` does the first three in one go.

---

## Fastest way to see it working

You need **Docker** and **Python 3.11+**.

```bash
# 1. Local Python env + dashboard deps + the sample clip  (~5 min, mostly PyTorch)
make setup

# 2. Start MongoDB + API + dashboard
docker compose up --build -d

# 3. Push the sample clip through the whole pipeline and watch it land live
python3 scripts/run_demo.py
```

Then open **<http://localhost:5173>**.

| | |
|---|---|
| Dashboard | <http://localhost:5173> |
| Analytics / heatmap | <http://localhost:5173/analytics> |
| Citizen reporting (PWA) | <http://localhost:5173/report> |
| API docs | <http://localhost:8000/docs> |
| System status | <http://localhost:8000/api/health> |

`/api/health` should report `status: ok`, `database_backend: mongodb`, `detector: yolov8`. If it
says `heuristic` or `in-memory`, something above did not finish — the warnings field explains which.

### No Docker?

The ML pipeline runs standalone, with no database and no backend:

```bash
make setup
python3 scripts/run_demo.py --local
```

---

## Verifying it

```bash
make test      # 172 unit and integration tests, no Docker or network needed
make ui-test   # headless browser check of the dashboard (needs the stack running)
```

---

## Where to look

| Question | File |
|---|---|
| How does the whole thing fit together? | [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) |
| How is severity actually calculated? | [docs/SEVERITY.md](docs/SEVERITY.md) |
| How good is the model, really? | [docs/MODEL_CARD.md](docs/MODEL_CARD.md) |
| What are the endpoints? | [docs/API.md](docs/API.md) |
| How do I demo this? | [docs/DEMO.md](docs/DEMO.md) |
| What are the limitations? | [README.md](README.md#known-limitations) |

**Please read the limitations section before quoting any accuracy figure.** The bundled checkpoint
is trained on synthetic data; its ~0.99 mAP measures that the training pipeline is correct, not that
the detector is deployable. Severity and size are documented estimates from a single camera —
pothole *depth* is not measured and cannot be. Training on RDD2022 for real numbers is a documented
one-command path.

---

## Commit history

`git log --oneline` — 20+ commits organised by pipeline stage rather than one bulk import.
