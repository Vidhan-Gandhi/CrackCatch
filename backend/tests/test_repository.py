"""Tests for the storage layer (pipeline stage 6): workflow, filters,
analytics, heatmap and the stage 8 retraining export."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pytest

from tests.conftest import make_defect

pytestmark = pytest.mark.asyncio


class TestCreate:
    async def test_create_sets_defaults_and_audit_trail(self, repository):
        created = await repository.create(make_defect())
        assert created["status"] == "New"
        assert created["priority_band"] == "Critical"  # priority 80
        assert len(created["history"]) == 1
        assert created["history"][0]["by"] == "pipeline"
        assert "created_at" in created and "updated_at" in created

    async def test_geojson_index_field_is_written_but_hidden(self, repository, memory_db):
        created = await repository.create(make_defect(lat=19.5, lon=72.5))
        raw = await memory_db["defects"].find_one({"client_id": created["client_id"]})
        assert raw["location_geo"] == {"type": "Point", "coordinates": [72.5, 19.5]}
        # The internal field must not leak into the API payload.
        assert "location_geo" not in created

    async def test_client_id_is_never_null(self, repository):
        """A null client_id collides on the unique sparse index.

        A sparse index skips ABSENT fields but still indexes explicit nulls,
        so two documents carrying `client_id: None` raise DuplicateKeyError on
        real MongoDB. Pydantic models emit the key as None rather than omitting
        it, so the repository must always assign one.
        """
        without = await repository.create(make_defect(client_id=None))
        assert without["client_id"]

        explicit_none = make_defect()
        explicit_none["client_id"] = None
        second = await repository.create(explicit_none)
        assert second["client_id"]
        assert second["client_id"] != without["client_id"]

    async def test_missing_client_id_key_is_generated(self, repository):
        payload = make_defect()
        payload.pop("client_id")
        created = await repository.create(payload)
        assert created["client_id"]

    async def test_client_id_makes_ingestion_idempotent(self, repository):
        """Re-running the same clip must not duplicate rows."""
        payload = make_defect(client_id="same-defect")
        first = await repository.create(payload)
        second = await repository.create(dict(payload))
        assert first["_id"] == second["_id"]
        assert await repository.count() == 1

    async def test_priority_band_boundaries(self, repository):
        for priority, expected in [(90, "Critical"), (75, "Critical"), (60, "High"),
                                   (40, "Medium"), (10, "Low")]:
            created = await repository.create(
                make_defect(priority=priority, client_id=f"band-{priority}")
            )
            assert created["priority_band"] == expected


class TestStatusWorkflow:
    async def test_full_happy_path(self, repository):
        defect = await repository.create(make_defect())
        for status in ("Verified", "Scheduled", "Repaired"):
            defect = await repository.update_status(defect["_id"], status, by="tester")
            assert defect["status"] == status
        assert defect["repaired_at"] is not None
        # New + 3 transitions.
        assert len(defect["history"]) == 4

    async def test_illegal_transition_is_rejected(self, repository):
        defect = await repository.create(make_defect())
        with pytest.raises(ValueError, match="Illegal transition"):
            await repository.update_status(defect["_id"], "Repaired")

    async def test_rejection_and_undo(self, repository):
        defect = await repository.create(make_defect())
        defect = await repository.update_status(defect["_id"], "Rejected", note="not a pothole")
        assert defect["status"] == "Rejected"
        defect = await repository.update_status(defect["_id"], "New")
        assert defect["status"] == "New"

    async def test_review_label_is_recorded_for_retraining(self, repository):
        defect = await repository.create(make_defect())
        updated = await repository.update_status(
            defect["_id"], "Verified", review_label="crack", note="actually a crack"
        )
        assert updated["review_label"] == "crack"
        assert updated["review_note"] == "actually a crack"

    async def test_update_missing_defect_returns_none(self, repository):
        assert await repository.update_status("507f1f77bcf86cd799439011", "Verified") is None

    async def test_attach_repair_photo(self, repository):
        defect = await repository.create(make_defect())
        updated = await repository.attach_repair_photo(defect["_id"], "after.jpg")
        assert updated["repair_photo_path"] == "after.jpg"

    async def test_delete(self, repository):
        defect = await repository.create(make_defect())
        assert await repository.delete(defect["_id"]) is True
        assert await repository.get(defect["_id"]) is None


class TestFilteringAndPaging:
    async def test_filter_by_severity(self, populated):
        query = populated.build_filter(severity=["Severe"])
        items, total = await populated.list_defects(query)
        assert total == 3
        assert {d["severity"] for d in items} == {"Severe"}

    async def test_filter_by_status_and_class(self, populated):
        query = populated.build_filter(status=["New"], defect_class=["crack"])
        _, total = await populated.list_defects(query)
        assert total == 1

    async def test_filter_by_min_priority(self, populated):
        query = populated.build_filter(min_priority=80.0)
        _, total = await populated.list_defects(query)
        assert total == 2  # 88 and 91

    async def test_filter_by_date_range(self, populated):
        now = datetime.now(timezone.utc)
        query = populated.build_filter(date_from=now - timedelta(days=2, hours=1))
        _, total = await populated.list_defects(query)
        assert total == 3

    async def test_filter_by_map_viewport(self, populated):
        # A box around the tight cluster, excluding the two outliers at 19.09.
        query = populated.build_filter(bbox=(72.870, 19.070, 72.880, 19.080))
        _, total = await populated.list_defects(query)
        assert total == 4

    async def test_paging_is_consistent(self, populated):
        page1, total = await populated.list_defects({}, page=1, page_size=2)
        page2, _ = await populated.list_defects({}, page=2, page_size=2)
        assert total == 6
        assert len(page1) == 2 and len(page2) == 2
        assert {d["_id"] for d in page1}.isdisjoint({d["_id"] for d in page2})

    async def test_sort_by_priority(self, populated):
        items, _ = await populated.list_defects({}, sort_by="priority_score", sort_dir=-1)
        scores = [d["priority_score"] for d in items]
        assert scores == sorted(scores, reverse=True)

    async def test_near_finds_the_local_cluster(self, populated):
        nearby = await populated.near(19.0760, 72.8777, radius_m=300)
        assert len(nearby) >= 2
        assert all(abs(d["location"]["latitude"] - 19.076) < 0.01 for d in nearby)


class TestAnalytics:
    async def test_summary_counts(self, populated):
        summary = await populated.analytics()
        assert summary["total_defects"] == 6
        # New, Verified, Scheduled are open; Repaired and Rejected are not.
        assert summary["open_defects"] == 4
        assert summary["repaired_defects"] == 1
        assert summary["severe_open"] == 2  # d6 is Severe but Rejected

    async def test_severity_buckets_are_complete(self, populated):
        summary = await populated.analytics()
        by_severity = {b["severity"]: b["count"] for b in summary["by_severity"]}
        assert by_severity == {"Minor": 1, "Moderate": 2, "Severe": 3}

    async def test_top_priority_excludes_closed_defects(self, populated):
        summary = await populated.analytics()
        statuses = {d["status"] for d in summary["top_priority"]}
        assert "Repaired" not in statuses and "Rejected" not in statuses
        scores = [d["priority_score"] for d in summary["top_priority"]]
        assert scores == sorted(scores, reverse=True)

    async def test_time_series_is_chronological(self, populated):
        summary = await populated.analytics(days=30)
        dates = [b["date"] for b in summary["over_time"]]
        assert dates == sorted(dates)
        assert sum(b["count"] for b in summary["over_time"]) == 6

    async def test_empty_database_does_not_divide_by_zero(self, repository):
        summary = await repository.analytics()
        assert summary["total_defects"] == 0
        assert summary["mean_priority"] == 0.0


class TestHeatmap:
    async def test_clusters_snap_to_grid_cells(self, populated):
        heatmap = await populated.heatmap(cell_size_deg=0.01)
        # Four defects near 19.076 and two near 19.090 -> two cells.
        assert len(heatmap["cells"]) == 2
        assert heatmap["cells"][0]["intensity"] == pytest.approx(1.0)

    async def test_severity_weighting_beats_raw_count(self, repository):
        """Three Severe defects must outrank four Minor ones."""
        for i in range(3):
            await repository.create(
                make_defect("pothole", "Severe", 90, lat=19.10, lon=72.90,
                            client_id=f"sev{i}")
            )
        for i in range(4):
            await repository.create(
                make_defect("crack", "Minor", 20, lat=19.20, lon=72.90,
                            client_id=f"min{i}")
            )
        heatmap = await repository.heatmap(cell_size_deg=0.01)
        top = heatmap["cells"][0]
        assert top["severe_count"] == 3
        assert top["count"] == 3          # fewer defects...
        assert top["intensity"] == pytest.approx(1.0)  # ...but higher intensity

    async def test_cell_centre_is_reported(self, repository):
        await repository.create(make_defect(lat=19.0005, lon=72.0005))
        heatmap = await repository.heatmap(cell_size_deg=0.01)
        cell = heatmap["cells"][0]
        assert cell["latitude"] == pytest.approx(19.005)
        assert cell["longitude"] == pytest.approx(72.005)


class TestRetrainingExport:
    async def test_only_human_reviewed_defects_are_exported(self, populated):
        """Untouched `New` rows carry no human signal and must be excluded."""
        exported = await populated.export_for_retraining()
        statuses = {d["status"] for d in exported}
        assert "New" not in statuses
        assert statuses == {"Verified", "Scheduled", "Repaired", "Rejected"}
        assert len(exported) == 4

    async def test_rejected_can_be_excluded(self, populated):
        exported = await populated.export_for_retraining(include_rejected=False)
        assert "Rejected" not in {d["status"] for d in exported}
        assert len(exported) == 3


class TestJobs:
    async def test_job_round_trip(self, repository):
        job = {
            "job_id": "job-1",
            "status": "running",
            "source": "demo.mp4",
            "created_at": datetime.now(timezone.utc),
        }
        await repository.save_job(job)
        assert (await repository.get_job("job-1"))["status"] == "running"

        job["status"] = "completed"
        await repository.save_job(job)
        assert (await repository.get_job("job-1"))["status"] == "completed"
        assert len(await repository.list_jobs()) == 1

    async def test_missing_job(self, repository):
        assert await repository.get_job("nope") is None
