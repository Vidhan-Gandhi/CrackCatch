"""WebSocket endpoint delivering the live defect feed to the dashboard."""

from __future__ import annotations

import logging

from fastapi import APIRouter, WebSocket, WebSocketDisconnect

from app.services.events import broadcaster

logger = logging.getLogger(__name__)
router = APIRouter(tags=["live"])


@router.websocket("/ws/defects")
async def defect_feed(websocket: WebSocket) -> None:
    """Push ``defect.created``, ``defect.updated``, ``alert.high_priority``
    and job lifecycle events as they happen.

    The client sends nothing meaningful; we read only to notice a disconnect
    promptly and to answer keepalive pings.
    """
    await broadcaster.connect(websocket)
    try:
        while True:
            message = await websocket.receive_text()
            if message == "ping":
                await websocket.send_text('{"type":"pong"}')
    except WebSocketDisconnect:
        pass
    except Exception:
        logger.debug("WebSocket closed unexpectedly", exc_info=True)
    finally:
        await broadcaster.disconnect(websocket)
