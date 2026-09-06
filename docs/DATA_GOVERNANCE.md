# Data governance

The project's non-functional requirements state: *don't log raw continuous GPS trails — only store
point coordinates tied to a confirmed defect*, and note what a production system would need before
any public-facing dashboard. This document records what CrackCatch actually does and what it does
not.

---

## What is stored

| Data | Where | Why |
|---|---|---|
| Defect coordinate (one point per defect) | `defects.location`, `defects.location_geo` | Dispatch a crew to the right place |
| Detection timestamp | `defects.detected_at` | Prioritisation and trend analytics |
| Frame snapshot with the defect | `storage/snapshots/` | Human verification and retraining |
| Bounding box, class, confidence, size, severity | `defects` | The finding itself |
| Repair audit trail (status, actor, note, time) | `defects.history` | Accountability |
| After-repair photo (optional) | `storage/repairs/` | Before/after verification |
| Uploaded source media | `storage/uploads/` | Re-processing and dispute resolution |

## What is deliberately **not** stored

- **No continuous GPS trail.** This is structural, not a policy promise. The GPS provider is
  consulted **once per frame and only when a detection has been confirmed** —
  `CrackCatchPipeline.process_frame` returns before ever calling `gps.locate()` if the frame yields
  no detections. There is no code path that writes a position sample without a defect attached, so
  a journey cannot be reconstructed from the database.
- **No vehicle or device identifier.** `source_ref` holds a filename or a camera index, not an
  IMEI, a registration number, or a driver identity.
- **No citizen identity.** The crowdsourcing endpoint takes a photo, a coordinate and an optional
  note. No account, no email, no phone number, no device fingerprint.
- **No continuous video.** Only the frames in which a defect was found are retained as snapshots.

---

## Residual risks in this prototype

Stated plainly, because "we only store points" is not by itself a privacy guarantee.

**Sparse points can still be re-identifying.** A defect reported at 02:10 outside a single house on
a quiet lane implicates whoever drove there. Deduplication reduces density but does not anonymise.

**Snapshots may contain people, vehicles and number plates.** This prototype does **not** blur faces
or plates. A public deployment must.

**Citizen submissions carry a coordinate at the moment of submission**, which is often the
submitter's own location.

**Photo EXIF may contain more than GPS** — device model, serial numbers, timestamps. CrackCatch reads
only the GPS tags, but the original upload is retained in `storage/uploads/` with its metadata intact.

**Storage is unencrypted local disk** with no per-object access control. Anyone who can read
`/media/snapshots/{file}` can read any snapshot; filenames are UUIDs, which is obscurity, not
access control.

---

## What a production deployment must add

1. **Spatial aggregation before any public view.** Publish grid cells, not points. The heatmap
   endpoint already aggregates to a configurable grid and is the right basis for a public map; the
   per-defect endpoint is not.
2. **k-anonymity threshold.** Suppress any cell containing fewer than *k* defects (k ≥ 5 is a common
   starting point) so a single report cannot be isolated.
3. **Temporal coarsening in public views.** Publish the day, not the second.
4. **Automatic face and number-plate blurring** on ingestion, before a snapshot is written.
5. **Retention policy with enforced deletion.** Suggested: raw uploads 30 days, snapshots of
   repaired defects 1 year, aggregate statistics indefinitely. Implement as a scheduled job with an
   audit log, not a manual cleanup.
6. **Encryption at rest and in transit**; object storage with signed, expiring URLs instead of
   `StaticFiles` on local disk.
7. **Access logging and role separation.** Who viewed which defect, when. Read access to snapshots
   should be authorised per request.
8. **A published privacy notice** for the citizen reporting channel, stating what is collected, how
   long it is kept, and how to request deletion.
9. **DPDP Act 2023 compliance** (India), including a lawful basis for processing, purpose
   limitation, and a grievance mechanism.

---

## Simulated coordinates

Demo footage has no GPS, so a synthetic route is generated. Every such point is tagged
`location.source = "simulated"`, that field is returned by the API, and the dashboard badges it in
amber. A demo therefore cannot be mistaken for a real survey — by a viewer or by a downstream
consumer of the API.

---

## Retention in this repository

`storage/`, prepared datasets and trained checkpoints are all git-ignored. RDD2022 is referenced and
converted by script but **never redistributed** — it remains subject to its authors' licence.
