"""Live event fan-out to dashboard WebSocket clients.

Stage 7 requires a "live/near-live feed of newly detected defects". Rather
than have the browser poll, the ingestion pipeline publishes each stored
defect here and every connected dashboard receives it immediately.
"""

from __future__ import annotations

import asyncio
import json
import logging
from datetime import datetime
from typing import Any

logger = logging.getLogger(__name__)


def _encode(value: Any) -> Any:
    if isinstance(value, datetime):
        return value.isoformat()
    return str(value)


class EventBroadcaster:
    """Tracks connected sockets and pushes JSON messages to all of them."""

    #: A send that takes longer than this means the client is wedged (a
    #: paused browser tab, a half-open TCP connection, or - in tests - a
    #: socket whose event loop is gone). We drop it rather than let one dead
    #: dashboard stall the ingestion request that triggered the broadcast.
    SEND_TIMEOUT_S = 2.0

    def __init__(self, history_size: int = 50) -> None:
        self._connections: set[Any] = set()
        self._lock = asyncio.Lock()
        self._history: list[dict[str, Any]] = []
        self._history_size = history_size

    @property
    def connection_count(self) -> int:
        return len(self._connections)

    async def connect(self, websocket: Any) -> None:
        await websocket.accept()
        async with self._lock:
            self._connections.add(websocket)
        logger.info("Dashboard connected (%d live)", len(self._connections))
        # Replay recent events so a dashboard opened mid-run is not blank.
        for event in list(self._history):
            try:
                await asyncio.wait_for(
                    websocket.send_text(json.dumps(event, default=_encode)),
                    timeout=self.SEND_TIMEOUT_S,
                )
            except (Exception, asyncio.TimeoutError):
                break

    async def disconnect(self, websocket: Any) -> None:
        async with self._lock:
            self._connections.discard(websocket)
        logger.info("Dashboard disconnected (%d live)", len(self._connections))

    async def broadcast(self, event_type: str, payload: Any) -> None:
        """Send an event to every connected client, dropping dead sockets."""
        event = {
            "type": event_type,
            "payload": payload,
            "at": datetime.now().astimezone().isoformat(),
        }
        self._history.append(event)
        if len(self._history) > self._history_size:
            self._history = self._history[-self._history_size :]

        if not self._connections:
            return

        message = json.dumps(event, default=_encode)
        async with self._lock:
            targets = list(self._connections)

        stale: list[Any] = []
        for websocket in targets:
            try:
                await asyncio.wait_for(
                    websocket.send_text(message), timeout=self.SEND_TIMEOUT_S
                )
            except asyncio.TimeoutError:
                logger.warning("Dashboard send timed out; dropping the connection")
                stale.append(websocket)
            except Exception:
                stale.append(websocket)

        if stale:
            async with self._lock:
                for websocket in stale:
                    self._connections.discard(websocket)
            logger.debug("Dropped %d stale dashboard connections", len(stale))

    def clear_history(self) -> None:
        self._history.clear()

    async def reset(self) -> None:
        """Drop all state. Used between tests and on shutdown."""
        async with self._lock:
            self._connections.clear()
        self._history.clear()


#: Process-wide broadcaster shared by the ingestion service and the WS router.
broadcaster = EventBroadcaster()


async def publish_defect_created(defect: dict[str, Any], alert_threshold: float) -> None:
    """Announce a newly stored defect, raising an alert when it is urgent.

    Every path that creates a defect - the ingestion pipeline, a crowdsourced
    photo, or a direct API POST - goes through here, so the dashboard's alert
    banner behaves identically no matter how the defect arrived.
    """
    await broadcaster.broadcast("defect.created", defect)
    if float(defect.get("priority_score", 0.0)) >= alert_threshold:
        await broadcaster.broadcast("alert.high_priority", defect)
