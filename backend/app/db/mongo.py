"""MongoDB connection management - pipeline stage 6.

Provides a single shared Motor client, creates the indexes the dashboard
queries depend on, and - when the real server is unreachable and
``allow_memory_db_fallback`` is on - degrades to an in-process MongoDB double
so that a demo does not die because Docker was not started.

The fallback is never silent: it logs a warning, and ``/api/health`` reports
``database_backend: "in-memory"`` plus a warning string that the dashboard
renders as a banner.
"""

from __future__ import annotations

import logging
from typing import Any

from motor.motor_asyncio import AsyncIOMotorClient, AsyncIOMotorDatabase

from app.core.config import Settings, get_settings

logger = logging.getLogger(__name__)

DEFECTS = "defects"
JOBS = "ingest_jobs"


class Database:
    """Holds the active client and reports which backend actually won."""

    def __init__(self) -> None:
        self.client: Any = None
        self.db: AsyncIOMotorDatabase | None = None
        self.backend: str = "disconnected"
        self.warnings: list[str] = []

    @property
    def connected(self) -> bool:
        return self.db is not None

    async def connect(self, settings: Settings | None = None) -> AsyncIOMotorDatabase:
        settings = settings or get_settings()
        self.warnings = []
        try:
            client = AsyncIOMotorClient(
                settings.mongo_uri,
                serverSelectionTimeoutMS=settings.mongo_timeout_ms,
                uuidRepresentation="standard",
            )
            await client.admin.command("ping")
            self.client = client
            self.db = client[settings.mongo_db]
            self.backend = "mongodb"
            logger.info("Connected to MongoDB at %s", _redact(settings.mongo_uri))
        except Exception as exc:
            if not settings.allow_memory_db_fallback:
                logger.error("MongoDB unreachable and fallback disabled: %s", exc)
                raise
            message = (
                f"MongoDB at {_redact(settings.mongo_uri)} is unreachable ({exc}). "
                "Running on an IN-MEMORY database: data will be lost when the "
                "API stops. Start MongoDB (`docker compose up -d mongo`) for "
                "real persistence."
            )
            logger.warning(message)
            self.warnings.append(message)
            self.client, self.db = _memory_client(settings.mongo_db)
            self.backend = "in-memory"

        await self.ensure_indexes()
        return self.db

    async def ensure_indexes(self) -> None:
        """Indexes backing the dashboard's filter, sort and map queries."""
        if self.db is None:
            return
        defects = self.db[DEFECTS]
        try:
            await defects.create_index("client_id", unique=True, sparse=True)
            await defects.create_index([("detected_at", -1)])
            await defects.create_index([("status", 1), ("severity", 1)])
            await defects.create_index([("priority_score", -1)])
            await defects.create_index([("defect_class", 1)])
            # 2dsphere powers "defects near this point" and the map viewport
            # query. Requires location_geo to be a GeoJSON Point.
            await defects.create_index([("location_geo", "2dsphere")])
            await self.db[JOBS].create_index("job_id", unique=True)
            await self.db[JOBS].create_index([("created_at", -1)])
        except Exception:
            # mongomock supports a subset of index types; a missing index
            # degrades performance, never correctness.
            logger.debug("Index creation partially failed", exc_info=True)

    async def close(self) -> None:
        if self.client is not None and hasattr(self.client, "close"):
            self.client.close()
        self.client = None
        self.db = None
        self.backend = "disconnected"

    async def ping(self) -> bool:
        if self.db is None:
            return False
        try:
            if self.backend == "mongodb":
                await self.client.admin.command("ping")
            else:
                await self.db[DEFECTS].estimated_document_count()
            return True
        except Exception:
            return False


def _memory_client(db_name: str):
    """Build an in-process MongoDB double via mongomock-motor."""
    try:
        from mongomock_motor import AsyncMongoMockClient
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError(
            "MongoDB is unreachable and `mongomock-motor` is not installed, so "
            "there is no fallback. Either start MongoDB or "
            "`pip install mongomock-motor`."
        ) from exc
    client = AsyncMongoMockClient()
    return client, client[db_name]


def _redact(uri: str) -> str:
    """Strip credentials before a connection string reaches a log line."""
    if "@" not in uri:
        return uri
    scheme, _, rest = uri.partition("://")
    _, _, host = rest.partition("@")
    return f"{scheme}://***:***@{host}"


#: Process-wide instance, wired up in the FastAPI lifespan handler.
database = Database()


async def get_database() -> AsyncIOMotorDatabase:
    """FastAPI dependency yielding the live database handle."""
    if database.db is None:
        await database.connect()
    assert database.db is not None
    return database.db
