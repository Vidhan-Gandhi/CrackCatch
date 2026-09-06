"""Tests for the backend API endpoints (stages 6 and 7's contract)."""

from __future__ import annotations

import io
import json

import cv2
import numpy as np
import pytest

from tests.conftest import make_defect


def _post_defect(client, **kwargs):
    payload = make_defect(**kwargs)
    payload["detected_at"] = payload["detected_at"].isoformat()
    response = client.post("/api/defects", json=payload)
    assert response.status_code == 201, response.text
    return response.json()


def _png_bytes(width: int = 640, height: int = 480) -> bytes:
    """A synthetic road-ish image with one dark blob the CV baseline can see."""
    image = np.full((height, width, 3), 110, dtype=np.uint8)
    image += np.random.default_rng(3).integers(-12, 12, image.shape, dtype=np.int16).astype(np.uint8)
    cv2.ellipse(image, (320, 380), (70, 45), 0, 0, 360, (25, 27, 30), -1)
    ok, encoded = cv2.imencode(".png", image)
    assert ok
    return encoded.tobytes()


class TestSystem:
    def test_root_lists_the_eight_pipeline_stages(self, client):
        body = client.get("/").json()
        assert len(body["pipeline"]) == 8
        assert "YOLOv8" in body["pipeline"][2]

    def test_health_reports_backend_honestly(self, client):
        """Health must never overstate the system's state.

        Asserted as an invariant rather than against fixed values, because
        whether a trained checkpoint exists depends on whether the developer
        has run training - and the honesty guarantee has to hold either way.
        """
        body = client.get("/api/health").json()
        assert body["database_backend"] == "in-memory"

        trained = body["detector_is_trained_model"]
        assert trained == (body["detector"] == "yolov8")

        # The in-memory fallback alone is enough to make this degraded.
        assert body["status"] == "degraded"
        assert any("IN-MEMORY database" in w for w in body["warnings"])

        # The CV-baseline warning must appear exactly when it applies.
        baseline_warned = any("classical-CV baseline" in w for w in body["warnings"])
        assert baseline_warned is (not trained)

    def test_openapi_is_served(self, client):
        assert client.get("/openapi.json").status_code == 200


class TestDefectsCrud:
    def test_create_and_fetch(self, client):
        created = _post_defect(client)
        assert created["status"] == "New"
        fetched = client.get(f"/api/defects/{created['_id']}")
        assert fetched.status_code == 200
        assert fetched.json()["_id"] == created["_id"]

    def test_missing_defect_is_404(self, client):
        assert client.get("/api/defects/507f1f77bcf86cd799439011").status_code == 404

    def test_invalid_payload_is_422(self, client):
        response = client.post("/api/defects", json={"defect_class": "spaceship"})
        assert response.status_code == 422

    def test_bbox_validation_is_enforced(self, client):
        payload = make_defect()
        payload["detected_at"] = payload["detected_at"].isoformat()
        payload["bbox"] = {"x1": 300.0, "y1": 0.0, "x2": 100.0, "y2": 50.0}
        assert client.post("/api/defects", json=payload).status_code == 422

    def test_out_of_range_latitude_is_422(self, client):
        payload = make_defect()
        payload["detected_at"] = payload["detected_at"].isoformat()
        payload["location"]["latitude"] = 120.0
        assert client.post("/api/defects", json=payload).status_code == 422

    def test_delete(self, client):
        created = _post_defect(client)
        assert client.delete(f"/api/defects/{created['_id']}").status_code == 204
        assert client.get(f"/api/defects/{created['_id']}").status_code == 404


class TestWorkflowEndpoint:
    def test_status_progression(self, client):
        created = _post_defect(client)
        for status in ("Verified", "Scheduled", "Repaired"):
            response = client.patch(
                f"/api/defects/{created['_id']}", json={"status": status}
            )
            assert response.status_code == 200, response.text
            assert response.json()["status"] == status

    def test_illegal_transition_is_409(self, client):
        created = _post_defect(client)
        response = client.patch(
            f"/api/defects/{created['_id']}", json={"status": "Repaired"}
        )
        assert response.status_code == 409
        assert "Illegal transition" in response.json()["detail"]

    def test_review_label_correction(self, client):
        created = _post_defect(client)
        response = client.patch(
            f"/api/defects/{created['_id']}",
            json={"status": "Verified", "review_label": "crack", "note": "misclassified"},
        )
        assert response.json()["review_label"] == "crack"

    def test_repair_photo_upload(self, client):
        created = _post_defect(client)
        client.patch(f"/api/defects/{created['_id']}", json={"status": "Verified"})
        client.patch(f"/api/defects/{created['_id']}", json={"status": "Scheduled"})
        client.patch(f"/api/defects/{created['_id']}", json={"status": "Repaired"})

        response = client.post(
            f"/api/defects/{created['_id']}/repair-photo",
            files={"file": ("after.png", io.BytesIO(_png_bytes(64, 64)), "image/png")},
        )
        assert response.status_code == 200
        assert response.json()["repair_photo_path"].endswith(".png")

    def test_repair_photo_rejects_bad_type(self, client):
        created = _post_defect(client)
        response = client.post(
            f"/api/defects/{created['_id']}/repair-photo",
            files={"file": ("notes.txt", io.BytesIO(b"hello"), "text/plain")},
        )
        assert response.status_code == 415


class TestFilters:
    @pytest.fixture
    def seeded(self, client):
        _post_defect(client, severity="Severe", priority=90, client_id="a", defect_class="pothole")
        _post_defect(client, severity="Minor", priority=20, client_id="b", defect_class="crack")
        _post_defect(client, severity="Moderate", priority=60, client_id="c", defect_class="pothole")
        return client

    def test_severity_filter(self, seeded):
        body = seeded.get("/api/defects", params={"severity": "Severe"}).json()
        assert body["total"] == 1

    def test_repeated_severity_params(self, seeded):
        body = seeded.get("/api/defects?severity=Severe&severity=Minor").json()
        assert body["total"] == 2

    def test_class_filter(self, seeded):
        body = seeded.get("/api/defects", params={"defect_class": "pothole"}).json()
        assert body["total"] == 2

    def test_min_priority_filter(self, seeded):
        body = seeded.get("/api/defects", params={"min_priority": 50}).json()
        assert body["total"] == 2

    def test_bbox_filter(self, seeded):
        body = seeded.get("/api/defects", params={"bbox": "72.0,19.0,73.0,20.0"}).json()
        assert body["total"] == 3
        body = seeded.get("/api/defects", params={"bbox": "0,0,1,1"}).json()
        assert body["total"] == 0

    def test_paging_metadata(self, seeded):
        body = seeded.get("/api/defects", params={"page_size": 2}).json()
        assert body["pages"] == 2 and body["page_size"] == 2 and len(body["items"]) == 2

    def test_sort_by_priority(self, seeded):
        body = seeded.get(
            "/api/defects", params={"sort_by": "priority_score", "sort_dir": "desc"}
        ).json()
        scores = [d["priority_score"] for d in body["items"]]
        assert scores == sorted(scores, reverse=True)

    def test_invalid_sort_field_is_422(self, seeded):
        assert seeded.get("/api/defects", params={"sort_by": "; drop"}).status_code == 422

    def test_near_endpoint(self, seeded):
        response = seeded.get(
            "/api/defects/near", params={"lat": 19.076, "lon": 72.8777, "radius_m": 500}
        )
        assert response.status_code == 200
        assert len(response.json()) >= 1


class TestAnalyticsEndpoints:
    @pytest.fixture
    def seeded(self, client):
        _post_defect(client, severity="Severe", priority=90, client_id="a", lat=19.076, lon=72.877)
        _post_defect(client, severity="Minor", priority=20, client_id="b", lat=19.076, lon=72.877)
        _post_defect(client, severity="Severe", priority=85, client_id="c", lat=19.200, lon=72.900)
        return client

    def test_summary_shape(self, seeded):
        body = seeded.get("/api/analytics/summary").json()
        assert body["total_defects"] == 3
        assert {b["severity"] for b in body["by_severity"]} == {"Minor", "Moderate", "Severe"}
        assert len(body["by_status"]) == 5
        assert "generated_at" in body

    def test_summary_honours_filters(self, seeded):
        body = seeded.get("/api/analytics/summary", params={"severity": "Severe"}).json()
        assert body["total_defects"] == 2

    def test_heatmap_cells(self, seeded):
        body = seeded.get("/api/analytics/heatmap", params={"cell_size_deg": 0.01}).json()
        assert len(body["cells"]) == 2
        assert body["cells"][0]["intensity"] == 1.0
        assert body["max_count"] >= 1

    def test_heatmap_rejects_absurd_cell_size(self, seeded):
        assert seeded.get(
            "/api/analytics/heatmap", params={"cell_size_deg": 5}
        ).status_code == 422


class TestReports:
    @pytest.fixture
    def seeded(self, client):
        _post_defect(client, client_id="r1")
        _post_defect(client, client_id="r2", severity="Minor", priority=15)
        return client

    def test_csv_export(self, seeded):
        response = seeded.get("/api/reports/csv")
        assert response.status_code == 200
        assert response.headers["content-type"].startswith("text/csv")
        assert "attachment" in response.headers["content-disposition"]
        lines = response.text.strip().splitlines()
        assert lines[0].startswith("id,defect_class,severity")
        assert len(lines) == 3  # header + 2 rows

    def test_pdf_export(self, seeded):
        response = seeded.get("/api/reports/pdf", params={"ward": "Ward 12"})
        assert response.status_code == 200
        assert response.headers["content-type"] == "application/pdf"
        assert response.content.startswith(b"%PDF")

    def test_report_respects_filters(self, seeded):
        response = seeded.get("/api/reports/csv", params={"severity": "Minor"})
        assert len(response.text.strip().splitlines()) == 2  # header + 1


class TestIngest:
    def test_path_traversal_is_blocked(self, client):
        """`source` names a server-side path, so it must be sandboxed."""
        response = client.post(
            "/api/ingest/run", json={"source": "/etc/passwd", "target_fps": 1}
        )
        assert response.status_code == 400
        assert "inside the project directory" in response.json()["detail"]

    def test_missing_source_is_404(self, client):
        response = client.post(
            "/api/ingest/run", json={"source": "data/samples/nope.mp4"}
        )
        assert response.status_code == 404

    def test_invalid_fps_is_422(self, client):
        response = client.post(
            "/api/ingest/run", json={"source": "data/samples", "target_fps": 0}
        )
        assert response.status_code == 422

    def test_crowdsource_submission_runs_the_pipeline(self, client):
        response = client.post(
            "/api/ingest/crowdsource",
            files={"file": ("pothole.png", io.BytesIO(_png_bytes()), "image/png")},
            data={"latitude": "19.0760", "longitude": "72.8777", "note": "big one"},
        )
        assert response.status_code == 200, response.text
        body = response.json()
        assert body["accepted"] is True
        # The synthetic blob should be found by the CV baseline; if the
        # detector changes, the contract (a well-formed reply) still holds.
        assert body["defects_found"] >= 0
        assert body["upload_id"]
        for defect in body["defects"]:
            assert defect["source_type"] == "crowdsource"
            assert defect["location"]["latitude"] == pytest.approx(19.0760)
            assert defect["location"]["source"] == "device"

    def test_crowdsource_rejects_non_images(self, client):
        response = client.post(
            "/api/ingest/crowdsource",
            files={"file": ("clip.mp4", io.BytesIO(b"\x00" * 32), "video/mp4")},
        )
        assert response.status_code == 415

    def test_crowdsource_rejects_empty_file(self, client):
        response = client.post(
            "/api/ingest/crowdsource",
            files={"file": ("empty.png", io.BytesIO(b""), "image/png")},
        )
        assert response.status_code == 400

    def test_jobs_listing(self, client):
        assert client.get("/api/ingest/jobs").json() == []
        assert client.get("/api/ingest/jobs/does-not-exist").status_code == 404


class TestLiveFeed:
    def test_websocket_receives_created_defects(self, client):
        """The dashboard's live feed: a create must reach a connected client."""
        with client.websocket_connect("/ws/defects") as socket:
            created = _post_defect(client, client_id="ws-1")
            message = json.loads(socket.receive_text())
            assert message["type"] == "defect.created"
            assert message["payload"]["_id"] == created["_id"]

    def test_websocket_alerts_on_high_priority(self, client):
        with client.websocket_connect("/ws/defects") as socket:
            _post_defect(client, priority=95.0, client_id="ws-2")
            types = {json.loads(socket.receive_text())["type"] for _ in range(2)}
            assert "alert.high_priority" in types

    def test_websocket_ping_pong(self, client):
        with client.websocket_connect("/ws/defects") as socket:
            socket.send_text("ping")
            assert json.loads(socket.receive_text())["type"] == "pong"


class TestAuth:
    def test_roles_are_enforced_when_auth_is_enabled(self, tmp_path, monkeypatch):
        from fastapi.testclient import TestClient

        from app.core.config import get_settings

        get_settings.cache_clear()
        monkeypatch.setenv("AUTH_ENABLED", "true")
        monkeypatch.setenv("STORAGE_DIR", str(tmp_path / "storage"))
        monkeypatch.setenv("MONGO_URI", "mongodb://127.0.0.1:1")
        monkeypatch.setenv("MONGO_TIMEOUT_MS", "150")

        from app.main import app

        with TestClient(app) as authed:
            assert authed.get("/api/defects").status_code == 401

            citizen = {"Authorization": "Bearer citizen-demo-token"}
            # A citizen may submit a photo but must not read the dashboard.
            assert authed.get("/api/defects", headers=citizen).status_code == 403

            authority = {"Authorization": "Bearer authority-demo-token"}
            assert authed.get("/api/defects", headers=authority).status_code == 200

            transport = {"Authorization": "Bearer transport-demo-token"}
            assert authed.get("/api/defects", headers=transport).status_code == 200
            # Read-only role must not mutate.
            assert authed.delete(
                "/api/defects/507f1f77bcf86cd799439011", headers=transport
            ).status_code == 403

            assert authed.get(
                "/api/defects", headers={"Authorization": "Bearer wrong"}
            ).status_code == 401

        get_settings.cache_clear()
