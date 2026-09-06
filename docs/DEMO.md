# Demo runbook

A tested sequence for presenting CrackCatch to a review panel, plus the questions most likely to be
asked and honest answers to them.

---

## Before the session

```bash
# 1. Bring up the stack (do this early; the first build pulls torch)
docker compose up --build -d
docker compose ps            # all three healthy?

# 2. Confirm the system is in its best state
curl -s localhost:8000/api/health | python3 -m json.tool
```

You want `"detector_is_trained_model": true` and `"database_backend": "mongodb"`. If either is
wrong, fix it *before* the panel sees it:

| Symptom | Fix |
|---|---|
| `detector: "heuristic"` | `make train` (or copy a checkpoint to `model/weights/crackcatch.pt`) and restart the backend |
| `database_backend: "in-memory"` | MongoDB is not up — `docker compose up -d mongo` |
| Sample clip missing | `python3 data/scripts/make_demo_video.py` |

```bash
# 3. Start from a clean board so the live fill-in is visible
make reset-db

# 4. Open two browser tabs
#    http://localhost:5173/dashboard
#    http://localhost:8000/docs
```

Keep one terminal ready with the demo command already typed.

---

## The 8-minute walkthrough

### 1. The problem and the architecture (1 min)

Open the dashboard on an empty board. State the gap: detection-only systems tell an authority that
400 defects exist but not which one to fix on Monday morning. CrackCatch closes the loop from
"damage exists" to "someone with authority can see it and act".

### 2. Live end-to-end run (2 min) — the centrepiece

With the dashboard visible, run:

```bash
python3 scripts/run_demo.py
```

Narrate while defects appear on the map in real time:

> "That's a dashcam clip going through all seven stages live — frames sampled, lighting normalised,
> YOLOv8 detecting potholes and cracks, each one sized against the road plane, scored for severity,
> geo-tagged, written to MongoDB, and pushed to this dashboard over a WebSocket."

The terminal prints measured throughput. **Read the real number aloud** — "about 12 FPS end to end,
17 FPS for the detector on this laptop" — rather than saying "real-time".

Point out the high-priority alert banner firing on the severe potholes.

### 3. Map and heatmap (1 min)

Toggle **Pins → Heatmap**. Explain what makes it meaningful:

> "This is severity-weighted damage density, not a count of records. A cell with three severe
> potholes burns hotter than one with six hairline cracks. That's what a planner needs — where the
> road is failing, not where the camera happened to look."

### 4. Explainability and the severity breakdown (2 min) — the viva material

Click the highest-priority defect. Walk the drawer top to bottom:

- Snapshot with the bounding box.
- **"Explain detection"** → the Grad-CAM overlay. Note the method label under the button.
- **"Why this severity?"** → the three factors, their weighted contributions, and the total.

> "Severity is 0.55 × size, 0.20 × aspect ratio, 0.25 × confidence. Nothing learned, nothing hidden
> — every factor is on screen and the contributions sum to the score."

Then pre-empt the hardest question yourself:

> "Size is a ground-plane projection: we know the camera height and pitch, so we can intersect the
> bounding box with the road plane. It gives surface extent — **not depth**. A single camera cannot
> see depth; a 5 cm and a 25 cm pothole with the same opening look identical. That needs stereo or
> LiDAR. Estimates beyond 25 metres are automatically flagged unreliable and excluded from scoring."

Saying this first is far stronger than being caught by it.

### 5. The repair workflow (1 min)

Mark the defect **Verified → Scheduled → Repaired**, adding a note. Show the audit trail. Attach an
after-repair photo. Then try an illegal jump in `/docs` (New → Repaired) and show the **409**:

> "The workflow is a state machine enforced at the API, not a dropdown you can put in any order."

### 6. Citizen reporting (1 min)

Open `http://localhost:5173/report` — ideally on a phone on the same network.

> "Drivers and commuters are a named stakeholder group. A citizen photographs a pothole, the browser
> supplies the coordinates, and it enters exactly the same pipeline as dashcam footage. It's a PWA,
> so it installs to the home screen."

Submit a photo; show it landing on the authority dashboard.

### 7. Reports and the feedback loop (1 min)

Export the PDF. Show that it is ordered by repair priority, with a method-and-limitations page.

> "And the loop closes: everything the authority verified or rejected becomes training data.
> `retrain_from_feedback.py` exports it, fine-tunes, evaluates against the deployed model, and
> promotes only if mAP actually improves. Rejections become hard negatives — that's what stops the
> model firing on tar patches."

---

## Backup plan

**If Docker fails**, everything still runs without a database or a backend:

```bash
python3 scripts/run_demo.py --local
```

This prints the full pipeline output and the achieved FPS. It proves the ML works even if the
infrastructure does not.

**If the network fails**, the map tiles will not load but every pin, chart, table and export still
works. Say so and move on.

**Have `docs/metrics.json` and `docs/benchmark.json` open in a tab** as a fallback for the numbers.

---

## Likely panel questions

**"Is this real-time?"**
Near-real-time, and here are the measured numbers: ~113 FPS detector on GPU, ~26 FPS on CPU, ~12 FPS
end-to-end including decode, scoring and snapshot writing. A dashcam pipeline samples at 2–5 FPS, so
there is comfortable headroom on CPU alone. Those are measured on this machine, not quoted from a paper.

**"How do you measure pothole depth?"**
We do not, and we do not claim to. Severity uses surface extent from a ground-plane projection.
Depth needs stereo, structured light or LiDAR — that is the honest limitation, and it is documented
in the README, the model card and the exported PDF.

**"Why is your mAP 0.99? That seems too good."**
Because the shipped checkpoint is trained on synthetic data, and the synthetic task is easy. That
number demonstrates the training pipeline is correct, not that the detector is deployable. Real
numbers require RDD2022, and the converter and training scripts for it are in the repository.
**Do not let this number stand unqualified** — volunteering the caveat is the strongest answer available.

**"Will it work on Indian roads?"**
Unproven here, and the review paper's own point is that RDD2022 and GAPs are not India-specific.
`prepare_rdd2022.py` supports the India subset with `--countries India`. Until that evaluation is
run, any accuracy claim is dataset-specific.

**"What if the model is wrong?"**
Nothing is dispatched automatically. Every defect requires human verification, and a rejection is
not discarded — it becomes a hard negative in the retraining set. The system is designed so being
wrong makes it better.

**"Why YOLOv8n rather than something larger?"**
The ablation is in the repo: YOLOv8n is 3.0 M parameters at ~26 FPS on CPU; YOLOv8s is 11.2 M at
~16 FPS. Since the pipeline samples at 2–5 FPS, the extra capacity buys nothing that matters while
halving throughput. On a Jetson or Raspberry Pi that gap decides whether it runs at all.

**"How do you avoid counting one pothole ten times?"**
Spatial-temporal deduplication: detections of the same class within 12 m and 20 s are merged, keeping
the worst observation. On the sample clip that collapses 82 raw detections into 19 records, and the
run reports both figures.

**"Is the privacy claim real or just a policy?"**
Structural. The GPS provider is only consulted after a detection is confirmed — `process_frame`
returns before reaching `gps.locate()` when a frame yields nothing. There is no code path that
writes a position without a defect attached, so a journey cannot be reconstructed.

**"What would you do next?"**
Train on RDD2022's India subset and report real metrics; add stereo or a depth sensor to make
severity a measurement rather than an estimate; replace the static traffic weight with real AADT
data; move inference onto the vehicle so only detections are uploaded.

---

## What not to do

- Do not say "real-time" without a number.
- Do not present the 0.99 mAP without immediately saying it is synthetic.
- Do not demo on an empty database — run `make reset-db` then the live run, so the panel sees it fill.
- Do not let the health banner show `in-memory` or `heuristic` on screen; fix it beforehand.
