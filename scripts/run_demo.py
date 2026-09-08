#!/usr/bin/env python3
"""End-to-end demo driver - replays a clip through the whole pipeline.

Runs stages 1-6 and (if the backend is up) pushes every detection to the live
dashboard as it is found, so a presentation can show the map filling in real
time rather than a pre-populated database.

Two modes:

``--via-api`` (default when the backend is reachable)
    POSTs an ingestion job to the running FastAPI service. Detections are
    stored in MongoDB and broadcast over WebSocket, so the dashboard updates
    live. This is the mode to use in front of a panel.

``--local``
    Runs the pipeline in this process and prints the results. No backend or
    database needed - useful to prove the ML pipeline works on its own, and to
    report achieved FPS.

Examples:
    python scripts/run_demo.py                       # auto-detect the backend
    python scripts/run_demo.py --local --fps 3
    python scripts/run_demo.py --input data/samples --local
"""

from __future__ import annotations

import argparse
import json
import sys
import time
import urllib.error
import urllib.request
from collections import Counter
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "model"))

DEFAULT_INPUT = "data/samples/demo_drive.mp4"


def backend_alive(base_url: str, timeout: float = 3.0) -> dict | None:
    try:
        with urllib.request.urlopen(f"{base_url}/api/health", timeout=timeout) as response:
            return json.loads(response.read())
    except Exception:
        return None


def post_json(url: str, payload: dict, timeout: float = 900.0) -> dict:
    request = urllib.request.Request(
        url,
        data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(request, timeout=timeout) as response:
        return json.loads(response.read())


def run_via_api(args, health: dict) -> int:
    print(f"Backend is up: detector={health['detector']}, db={health['database_backend']}")
    if not health.get("detector_is_trained_model"):
        print("  NOTE: running the classical-CV baseline, not a trained YOLOv8 model.")

    print(f"\nSubmitting {args.input} to the pipeline...")
    print("Open http://localhost:5173 to watch defects appear live.\n")

    started = time.perf_counter()
    try:
        job = post_json(
            f"{args.api}/api/ingest/run?wait=true",
            {
                "source": args.input,
                "target_fps": args.fps,
                "max_frames": args.max_frames,
                "road_type": args.road_type,
                "gps_method": "simulated",
                "gps_start_lat": args.lat,
                "gps_start_lon": args.lon,
            },
        )
    except urllib.error.HTTPError as exc:
        print(f"error: the API rejected the job ({exc.code}): {exc.read().decode()[:300]}",
              file=sys.stderr)
        return 1
    except Exception as exc:
        print(f"error: could not reach the API: {exc}", file=sys.stderr)
        return 1

    elapsed = time.perf_counter() - started
    if job.get("status") == "failed":
        print(f"Job failed: {job.get('error')}", file=sys.stderr)
        return 1

    stats = job.get("stats") or {}
    print("-" * 62)
    print(f"  job            : {job['job_id']}")
    print(f"  detector       : {job.get('detector')}")
    print(f"  defects stored : {job.get('records_stored')}")
    print(f"  frames read    : {stats.get('frames_read')}")
    print(f"  deduplicated   : {stats.get('records_deduped')} repeat sightings merged")
    print(f"  inference FPS  : {stats.get('inference_fps')}")
    print(f"  end-to-end FPS : {stats.get('end_to_end_fps')}")
    print(f"  wall clock     : {elapsed:.1f}s")
    print("-" * 62)
    print("\nDashboard: http://localhost:5173   API docs: http://localhost:8000/docs")
    return 0


def run_local(args) -> int:
    import logging

    logging.basicConfig(level=logging.WARNING, format="%(levelname)s %(message)s")

    from crackcatch_model.pipeline import CrackCatchPipeline, PipelineConfig
    from crackcatch_model.sources import open_source

    print("Running the pipeline locally (no backend, no database).\n")

    config = PipelineConfig(
        calibration={
            "method": "ground_plane",
            "camera_height_m": args.camera_height,
            "pitch_deg": args.camera_pitch,
        },
        road_type=args.road_type,
        snapshot_dir=REPO_ROOT / "storage" / "snapshots",
    )
    pipeline = CrackCatchPipeline(config)
    print(f"Detector: {pipeline.detector.name}")
    if pipeline.detector.name != "yolov8":
        print("  NOTE: classical-CV baseline in use (no trained checkpoint found).")

    source = open_source(args.input, target_fps=args.fps, max_frames=args.max_frames)
    print(f"Source  : {args.input}\n")

    records = []
    for record in pipeline.run(source):
        records.append(record)
        size = record.size.area_m2 if record.size else None
        print(
            f"  [{len(records):3d}] {record.defect_class.value:<8} "
            f"{record.severity.value:<8} priority={record.priority_score:5.1f} "
            f"conf={record.confidence:.2f} "
            f"area={'n/a' if size is None else f'{size:.2f}m2'} "
            f"@ {record.location.latitude:.5f},{record.location.longitude:.5f}"
        )

    stats = pipeline.stats.to_dict()
    print("\n" + "-" * 62)
    print(f"  defects found  : {len(records)}")
    print(f"  by class       : {dict(Counter(r.defect_class.value for r in records))}")
    print(f"  by severity    : {dict(Counter(r.severity.value for r in records))}")
    print(f"  frames read    : {stats['frames_read']}  processed: {stats['frames_processed']}")
    print(f"  deduplicated   : {stats['records_deduped']} repeat sightings merged")
    print(f"  inference FPS  : {stats['inference_fps']}")
    print(f"  end-to-end FPS : {stats['end_to_end_fps']}")
    print("-" * 62)
    print("\nAchieved FPS above is measured on this machine, not a claim.")
    print("Snapshots written to storage/snapshots/")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--input", default=DEFAULT_INPUT,
                        help="Video file, image, or directory (default: the demo clip)")
    parser.add_argument("--fps", type=float, default=2.0, help="Frames sampled per second")
    parser.add_argument("--max-frames", type=int, default=None)
    parser.add_argument("--road-type", default="arterial")
    parser.add_argument("--api", default="http://localhost:8000")
    parser.add_argument("--local", action="store_true", help="Force local mode")
    parser.add_argument("--via-api", action="store_true", help="Force API mode")
    parser.add_argument("--lat", type=float, default=19.0760)
    parser.add_argument("--lon", type=float, default=72.8777)
    parser.add_argument("--camera-height", type=float, default=1.35)
    parser.add_argument("--camera-pitch", type=float, default=8.0)
    args = parser.parse_args()

    source_path = REPO_ROOT / args.input
    if not source_path.exists() and not args.input.isdigit():
        print(f"error: input not found: {source_path}", file=sys.stderr)
        if args.input == DEFAULT_INPUT:
            print("Generate the demo clip first:\n"
                  "  python data/scripts/make_demo_video.py", file=sys.stderr)
        return 2

    print("=" * 62)
    print("CrackCatch - end-to-end demo")
    print("=" * 62)

    if args.local:
        return run_local(args)

    health = backend_alive(args.api)
    if health:
        return run_via_api(args, health)

    if args.via_api:
        print(f"error: no backend at {args.api}. Start it with:\n"
              "  docker compose up -d\n"
              "  # or: uvicorn app.main:app --app-dir backend", file=sys.stderr)
        return 1

    print(f"No backend at {args.api} - falling back to local mode.")
    print("(Start the stack with `docker compose up -d` for the live dashboard.)\n")
    return run_local(args)


if __name__ == "__main__":
    raise SystemExit(main())
