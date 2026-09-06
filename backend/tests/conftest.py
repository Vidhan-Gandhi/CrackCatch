"""Shared pytest fixtures.

Tests always run against the in-process MongoDB double, never a real server,
so `pytest` needs no Docker and no network.
"""

from __future__ import annotations

import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
for extra in (REPO_ROOT / "backend", REPO_ROOT / "model"):
    if str(extra) not in sys.path:
        sys.path.insert(0, str(extra))

from app.db.repository import DefectRepository  # noqa: E402


@pytest.fixture
def anyio_backend() -> str:
    return "asyncio"


@pytest.fixture
def memory_db():
    """A fresh in-process MongoDB double per test."""
    from mongomock_motor import AsyncMongoMockClient

    client = AsyncMongoMockClient()
    return client["crackcatch_test"]


@pytest.fixture
def repository(memory_db) -> DefectRepository:
    return DefectRepository(memory_db)


def make_defect(
    defect_class: str = "pothole",
    severity: str = "Severe",
    priority: float = 80.0,
    status: str = "New",
    lat: float = 19.0760,
    lon: float = 72.8777,
    days_ago: int = 0,
    client_id: str | None = None,
    **overrides,
) -> dict:
    """Build a valid defect document for tests."""
    detected = datetime.now(timezone.utc) - timedelta(days=days_ago)
    document = {
        "client_id": client_id or f"test-{defect_class}-{lat}-{lon}-{days_ago}-{severity}",
        "defect_class": defect_class,
        "severity": severity,
        "confidence": 0.82,
        "bbox": {"x1": 100.0, "y1": 300.0, "x2": 260.0, "y2": 420.0},
        "location": {
            "latitude": lat,
            "longitude": lon,
            "accuracy_m": 8.0,
            "source": "simulated",
        },
        "detected_at": detected,
        "size": {
            "area_px": 19200.0,
            "area_frac": 0.02,
            "width_m": 0.6,
            "length_m": 0.5,
            "area_m2": 0.3,
            "method": "ground_plane",
            "range_m": 6.0,
            "reliable": True,
            "confidence_note": "",
        },
        "severity_score": 0.71,
        "severity_breakdown": {"size_factor": 0.5, "_size_basis": "footprint_m2"},
        "priority_score": priority,
        "status": status,
        "source_type": "video",
        "source_ref": "demo_drive.mp4",
        "frame_index": 12,
        "snapshot_path": None,
        "road_type": "arterial",
        "notes": "",
    }
    document.update(overrides)
    return document


@pytest.fixture
def sample_defects() -> list[dict]:
    """A spread across severity, status, class and location."""
    return [
        make_defect("pothole", "Severe", 88.0, "New", 19.0760, 72.8777, 0, "d1"),
        make_defect("pothole", "Moderate", 62.0, "Verified", 19.0768, 72.8781, 1, "d2"),
        make_defect("crack", "Minor", 31.0, "New", 19.0772, 72.8790, 2, "d3"),
        make_defect("crack", "Severe", 79.0, "Scheduled", 19.0761, 72.8778, 3, "d4"),
        make_defect("pothole", "Moderate", 55.0, "Repaired", 19.0900, 72.8900, 5, "d5"),
        make_defect("pothole", "Severe", 91.0, "Rejected", 19.0901, 72.8901, 6, "d6"),
    ]


@pytest.fixture
async def populated(repository, sample_defects) -> DefectRepository:
    for defect in sample_defects:
        await repository.create(defect)
    return repository


@pytest.fixture
def client(tmp_path, monkeypatch):
    """A TestClient wired to a clean in-memory DB and a temp storage dir."""
    from fastapi.testclient import TestClient

    from app.core.config import get_settings
    from app.db import mongo as mongo_module

    get_settings.cache_clear()
    monkeypatch.setenv("STORAGE_DIR", str(tmp_path / "storage"))
    monkeypatch.setenv("MONGO_URI", "mongodb://127.0.0.1:1")  # guaranteed refused
    monkeypatch.setenv("ALLOW_MEMORY_DB_FALLBACK", "true")
    monkeypatch.setenv("MONGO_TIMEOUT_MS", "150")
    monkeypatch.setenv("AUTH_ENABLED", "false")

    from app.main import app
    from app.services.events import broadcaster

    # The broadcaster is a process-wide singleton that replays recent events
    # to newly connected clients. Without clearing it, events from an earlier
    # test are replayed into this test's socket.
    broadcaster.clear_history()
    broadcaster._connections.clear()

    with TestClient(app) as test_client:
        yield test_client

    broadcaster.clear_history()
    broadcaster._connections.clear()
    mongo_module.database.warnings = []
    get_settings.cache_clear()
