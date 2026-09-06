"""Backend configuration.

All settings are environment-driven (12-factor) with defaults that work for a
local `docker compose up`. Nothing secret is hard-coded; see `.env.example`.
"""

from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path

from typing import Annotated

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, NoDecode, SettingsConfigDict

REPO_ROOT = Path(__file__).resolve().parents[3]


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=(REPO_ROOT / ".env"),
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # --- service ---
    app_name: str = "CrackCatch API"
    version: str = "1.0.0"
    environment: str = "development"
    debug: bool = True

    # --- storage (stage 6) ---
    mongo_uri: str = "mongodb://localhost:27017"
    mongo_db: str = "crackcatch"
    #: Fall back to an in-process MongoDB double when the real server is
    #: unreachable. Keeps a demo alive if Docker is not running; logs loudly
    #: and is reported by /api/health so it can never pass unnoticed.
    allow_memory_db_fallback: bool = True
    mongo_timeout_ms: int = 3000

    # --- object storage for snapshots ---
    storage_dir: Path = REPO_ROOT / "storage"

    @property
    def snapshot_dir(self) -> Path:
        return self.storage_dir / "snapshots"

    @property
    def upload_dir(self) -> Path:
        return self.storage_dir / "uploads"

    @property
    def repair_photo_dir(self) -> Path:
        return self.storage_dir / "repairs"

    @property
    def retraining_dir(self) -> Path:
        return self.storage_dir / "retraining"

    # --- detection (stage 3) ---
    model_weights: Path = REPO_ROOT / "model" / "weights" / "crackcatch.pt"
    confidence_threshold: float = 0.25
    inference_device: str = "auto"
    detector_fallback: bool = True

    # --- stage 4 calibration defaults ---
    camera_height_m: float = 1.35
    camera_pitch_deg: float = 8.0
    camera_hfov_deg: float = 78.0

    # --- stage 5 demo geo defaults ---
    demo_origin_lat: float = 19.0760
    demo_origin_lon: float = 72.8777
    demo_speed_kmph: float = 30.0

    # --- API ---
    # NoDecode hands the raw env string to _split_origins below. Without it,
    # pydantic-settings JSON-decodes complex fields before validators run, so
    # the comma-separated form used in docker-compose.yml would raise at
    # import time rather than being parsed.
    cors_origins: Annotated[list[str], NoDecode] = Field(
        default_factory=lambda: [
            "http://localhost:5173",
            "http://localhost:4173",
            "http://localhost:3000",
        ]
    )
    max_upload_mb: int = 200
    #: Defects at or above this priority raise a dashboard alert banner.
    alert_priority_threshold: float = 75.0

    # --- auth (demo-grade; see docs/SECURITY.md) ---
    auth_enabled: bool = False
    api_tokens: Annotated[dict[str, str], NoDecode] = Field(
        default_factory=lambda: {
            "authority-demo-token": "authority",
            "transport-demo-token": "transport",
            "planner-demo-token": "planner",
            "citizen-demo-token": "citizen",
        }
    )

    @field_validator("cors_origins", mode="before")
    @classmethod
    def _split_origins(cls, value):
        """Accept a JSON array, a comma-separated string, or a real list."""
        if isinstance(value, str):
            value = value.strip()
            if not value:
                return []
            if value.startswith("["):
                return json.loads(value)
            return [v.strip() for v in value.split(",") if v.strip()]
        return value

    @field_validator("api_tokens", mode="before")
    @classmethod
    def _parse_tokens(cls, value):
        """Accept a JSON object string, or a real dict."""
        if isinstance(value, str):
            value = value.strip()
            return json.loads(value) if value else {}
        return value

    def ensure_directories(self) -> None:
        for directory in (
            self.storage_dir,
            self.snapshot_dir,
            self.upload_dir,
            self.repair_photo_dir,
            self.retraining_dir,
        ):
            directory.mkdir(parents=True, exist_ok=True)


@lru_cache
def get_settings() -> Settings:
    return Settings()
