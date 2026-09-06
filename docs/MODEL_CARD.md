# Model card — CrackCatch road-damage detector

## Model details

| | |
|---|---|
| **Architecture** | YOLOv8n (Ultralytics), single-stage anchor-free detector |
| **Parameters** | 3.01 M |
| **Input** | 640×640 RGB, letterboxed |
| **Classes** | `pothole` (0), `crack` (1) |
| **Training** | Transfer learning from the COCO-pretrained `yolov8n.pt` checkpoint |
| **Framework** | PyTorch 2.9, Ultralytics 8.4 |
| **Served checkpoint** | `model/weights/crackcatch.pt` |

## Intended use

Assist municipal road-maintenance teams in **locating and triaging** road damage from survey video
or citizen photographs. Output is a prioritised worklist for human inspection.

**Not intended for:** structural or load-bearing assessment, automated dispatch without human
review, legal or insurance evidence, or any safety-critical decision made without a person in the
loop. Every prediction is a candidate for inspection, not a finding.

---

## The checkpoint shipped in this repository

> **The shipped checkpoint is trained on synthetic data, and its metrics are not evidence of
> real-world accuracy.**

The evaluation below was produced by `model/scripts/evaluate.py` on the held-out split of the
procedurally generated dataset from `data/scripts/make_synthetic_dataset.py`.

### Training configuration

| | |
|---|---|
| Dataset | 500 train / 120 val synthetic images (1093 pothole, 872 crack boxes) |
| Epochs | 30 |
| Image size | 640 |
| Batch | 16 |
| Optimiser | AdamW (auto-selected), lr 0.001667 |
| Seed | 42, `deterministic=True` |
| Device | Apple M-series GPU (MPS) |
| Augmentation | horizontal flip 0.5, rotation ±5°, translate 0.1, scale 0.4, HSV jitter, mosaic 1.0. Vertical flip disabled — it would place road surface in the sky |

### Results (synthetic held-out split, 120 images / 394 instances)

| Metric | Value |
|---|---|
| mAP@0.5 | **0.9945** |
| mAP@0.5:0.95 | **0.8937** |
| Precision | 0.9902 |
| Recall | 0.9973 |

| Class | Precision | Recall | mAP@0.5 | mAP@0.5:0.95 |
|---|---|---|---|---|
| pothole | 0.9977 | 1.0000 | 0.9950 | 0.9655 |
| crack | 0.9827 | 0.9946 | 0.9941 | 0.8218 |

Confusion matrix and PR curves: `model/runs/val_crackcatch/`. Raw numbers: `docs/metrics.json`.

**Read these numbers correctly.** ~0.99 mAP does not mean the detector is solved. It means the
*synthetic task* is easy: procedural asphalt contains none of the confounders that make real
road-damage detection hard — wet patches, tar repairs and previous fills, shadows from cables and
trees, painted markings, manhole covers, debris, leaves, oil stains, and the enormous appearance
variance of real potholes. These figures demonstrate that the **training pipeline is correct and
reproducible**, nothing more.

### Speed / size ablation

Mirrors Table II of the review paper — the accuracy/FPS trade-off that decides whether inference can
run in the vehicle or must sit on a server. Measured by `model/scripts/benchmark_fps.py` at 640×640,
batch 1, on an Apple M-series laptop.

| Checkpoint | Params | CPU FPS | GPU (MPS) FPS |
|---|---|---|---|
| `crackcatch.pt` (YOLOv8n) | 3.01 M | **25.6** | **113.2** |
| `yolov8s.pt` (YOLOv8s) | 11.17 M | 15.9 | 56.5 |

YOLOv8n is the right default here. At ~26 FPS on CPU alone it comfortably exceeds the 2–5 FPS a
dashcam pipeline actually samples at, so the extra capacity of YOLOv8s buys nothing that matters
while roughly halving throughput and nearly quadrupling parameter count. On a Raspberry Pi or Jetson
this gap decides whether the system runs at all.

End-to-end pipeline throughput (capture → preprocess → detect → score → geo-tag → snapshot) on the
sample clip is ~12 FPS wall-clock, reported by `scripts/run_demo.py`. That is the honest figure for
the whole system, not the detector in isolation.

---

## Training on RDD2022 (the real dataset)

RDD2022 is the dataset named in the project scope and the one to use for reportable metrics. It
includes an India subset, which matters because the review paper explicitly notes that RDD2022 and
GAPs are not India-specific.

```bash
python data/scripts/prepare_rdd2022.py \
    --source ~/datasets/RDD2022 --output data/rdd2022_yolo \
    --countries India Japan --val-split 0.2 --seed 42
python model/scripts/train.py --data data/rdd2022_yolo/data.yaml --epochs 60
python model/scripts/evaluate.py --data data/rdd2022_yolo/data.yaml --also-cpu-fps
```

### Label mapping

| RDD2022 | CrackCatch | Reason |
|---|---|---|
| D00 longitudinal crack | `crack` | |
| D10 transverse crack | `crack` | |
| D20 alligator crack | `crack` | |
| D40 pothole | `pothole` | |
| D43 / D44 (markings, crosswalk blur) | *ignored* | Marking defects, not structural damage; including them costs precision on the two classes the workflow acts on |

The split is seeded and deterministic given the same file listing, satisfying the reproducibility
requirement. RDD2022 is **not redistributed** with this repository.

---

## The classical-CV fallback

When no checkpoint is present, `HeuristicDetector` runs so that a fresh clone has a working
pipeline. It combines a multi-scale blackhat morphological response (dark structures smaller than
each kernel) with a scale-free darkness residual against a blurred background estimate, thresholds
each cue on its own Otsu level, and classifies connected components by aspect ratio and fill.

**It is a baseline, not a model.** It finds dark road-coloured regions, which correlates with
potholes but also fires on oil stains, shadows and tar patches; its crack recall is weak. Its
"confidence" is a contrast/shape plausibility score, not a calibrated probability. Every detection
is tagged `detector="heuristic"`, and `/api/health` reports
`detector_is_trained_model: false` with an explicit warning. No metric in this document comes from it.

---

## Failure modes

| Failure | Cause | Mitigation in the system |
|---|---|---|
| Shadows and tar patches read as potholes | Both are dark regions on asphalt | Hard negatives from dashboard rejections feed stage 8 retraining |
| Wet road / puddles | Specular reflection hides or mimics damage | Not handled; documented limitation |
| Night and low light | Training data is daylight | CLAHE helps marginally; a night dataset is required |
| Distant defects mis-sized | Ground sampling distance grows with range² | Estimates beyond 25 m flagged `reliable: false` and excluded from severity |
| Depth invisible | Monocular geometry | Explicitly not claimed anywhere; severity uses surface extent |
| Motion blur | Vehicle speed vs. shutter | Frames below a Laplacian-variance threshold are skipped |
| Same pothole counted many times | 2 fps sampling from a moving vehicle | Spatial-temporal deduplication (12 m / 20 s) merges repeat sightings |
| Non-flat road | Camber, slopes, speed bumps | Documented bias; ground-plane assumption stated |

---

## Ethical and operational considerations

- **Human in the loop.** Nothing is dispatched automatically. Every defect requires an authority
  user to verify it before it enters the repair workflow.
- **Prioritisation encodes a value judgement.** Weighting a highway above a residential street is a
  policy choice, not a fact. It is written as an editable constant
  (`ROAD_TYPE_WEIGHTS`) and documented so a municipality can contest and change it, rather than
  buried in a learned model. Left unexamined, it would systematically deprioritise repairs in
  low-traffic and lower-income areas.
- **Location data.** Only discrete points attached to confirmed defects are stored — never a
  continuous vehicle trail. See [DATA_GOVERNANCE.md](DATA_GOVERNANCE.md).
- **Crowdsourced photos** may incidentally capture people, vehicles or number plates. This prototype
  does not blur faces or plates; a production deployment must.
- **Do not present these metrics as deployment-ready.** They are synthetic-data figures for a
  research prototype.

---

## Reproducibility

- Pinned dependency versions (`requirements.txt`, `backend/requirements.txt`, `model/requirements.txt`)
- Seeded train/val split, fixed at dataset preparation and not redrawn at training time
- `seed=42`, `deterministic=True` passed to Ultralytics (note: a few MPS kernels have no
  deterministic implementation and warn accordingly — bit-exact reproduction requires CPU or CUDA)
- Every run writes `crackcatch_summary.json` recording dataset, base checkpoint, hyperparameters,
  device and seed
