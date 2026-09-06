"""Tests for settings parsing.

These exist because of a real failure: `CORS_ORIGINS` was declared as
`list[str]`, and pydantic-settings JSON-decodes complex fields from the
environment *before* any validator runs. The comma-separated form used in
docker-compose.yml therefore raised `SettingsError` at import time and the
backend container crash-looped, while local development (which used the
default) was completely unaffected.
"""

from __future__ import annotations

import pytest

from app.core.config import Settings


class TestCorsOrigins:
    def test_comma_separated_env(self, monkeypatch):
        monkeypatch.setenv("CORS_ORIGINS", "http://a.com,http://b.com")
        assert Settings().cors_origins == ["http://a.com", "http://b.com"]

    def test_comma_separated_with_spaces(self, monkeypatch):
        monkeypatch.setenv("CORS_ORIGINS", " http://a.com , http://b.com ")
        assert Settings().cors_origins == ["http://a.com", "http://b.com"]

    def test_json_array_env(self, monkeypatch):
        monkeypatch.setenv("CORS_ORIGINS", '["http://x.com","http://y.com"]')
        assert Settings().cors_origins == ["http://x.com", "http://y.com"]

    def test_empty_env(self, monkeypatch):
        monkeypatch.setenv("CORS_ORIGINS", "")
        assert Settings().cors_origins == []

    def test_default_is_a_usable_list(self, monkeypatch):
        monkeypatch.delenv("CORS_ORIGINS", raising=False)
        origins = Settings().cors_origins
        assert isinstance(origins, list)
        assert "http://localhost:5173" in origins


class TestApiTokens:
    def test_json_object_env(self, monkeypatch):
        monkeypatch.setenv("API_TOKENS", '{"tok":"authority"}')
        assert Settings().api_tokens == {"tok": "authority"}

    def test_empty_env(self, monkeypatch):
        monkeypatch.setenv("API_TOKENS", "")
        assert Settings().api_tokens == {}

    def test_default_covers_every_stakeholder_role(self, monkeypatch):
        monkeypatch.delenv("API_TOKENS", raising=False)
        roles = set(Settings().api_tokens.values())
        assert roles == {"authority", "transport", "planner", "citizen"}


class TestPaths:
    def test_storage_subdirectories_derive_from_storage_dir(self, tmp_path, monkeypatch):
        monkeypatch.setenv("STORAGE_DIR", str(tmp_path))
        settings = Settings()
        assert settings.snapshot_dir == tmp_path / "snapshots"
        assert settings.upload_dir == tmp_path / "uploads"
        assert settings.repair_photo_dir == tmp_path / "repairs"

    def test_ensure_directories_creates_them(self, tmp_path, monkeypatch):
        monkeypatch.setenv("STORAGE_DIR", str(tmp_path / "fresh"))
        settings = Settings()
        settings.ensure_directories()
        assert settings.snapshot_dir.is_dir()
        assert settings.upload_dir.is_dir()
        assert settings.repair_photo_dir.is_dir()


class TestNumericSettings:
    @pytest.mark.parametrize(
        "name,value,attr,expected",
        [
            ("CONFIDENCE_THRESHOLD", "0.4", "confidence_threshold", 0.4),
            ("ALERT_PRIORITY_THRESHOLD", "60", "alert_priority_threshold", 60.0),
            ("CAMERA_HEIGHT_M", "1.8", "camera_height_m", 1.8),
            ("MAX_UPLOAD_MB", "50", "max_upload_mb", 50),
        ],
    )
    def test_numeric_env_overrides(self, monkeypatch, name, value, attr, expected):
        monkeypatch.setenv(name, value)
        assert getattr(Settings(), attr) == expected

    def test_boolean_env_override(self, monkeypatch):
        monkeypatch.setenv("ALLOW_MEMORY_DB_FALLBACK", "false")
        assert Settings().allow_memory_db_fallback is False
