from __future__ import annotations

import asyncio
import contextlib
import json
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, AsyncIterator

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from .interceptor import LatentBeamInterceptor
from .lifecycle_logger import ContextGraphLifecycleLogger
from .memory import EpigeneticMemoryController
from .router import ConfluenceRouter
from .transaction_log import TokenTransactionLog


BASE_DIR = Path(__file__).resolve().parent.parent
STATIC_DIR = BASE_DIR / "static"

app = FastAPI(title="AuraTrack", version="0.1.0")
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")

router_layer = ConfluenceRouter()
interceptor = LatentBeamInterceptor()
memory_controller = EpigeneticMemoryController()
lifecycle_logger = ContextGraphLifecycleLogger()


@dataclass
class RuntimeSession:
    session_id: str
    prompt: str = ""
    system_context: str = ""
    sub_context: dict[str, Any] = field(default_factory=dict)
    token_log: TokenTransactionLog = field(default_factory=TokenTransactionLog)
    packets: dict[int, dict[str, Any]] = field(default_factory=dict)
    stream_task: asyncio.Task[None] | None = None
    cancellation_token: StreamCancellationToken | None = None
    stream_buffer: list[dict[str, Any]] = field(default_factory=list)
    run_id: str = field(default_factory=lambda: uuid.uuid4().hex)

    def begin_stream(self) -> "StreamCancellationToken":
        self.cancellation_token = StreamCancellationToken(self.run_id)
        self.stream_buffer.clear()
        return self.cancellation_token

    def request_cancel(self, reason: str) -> None:
        if self.cancellation_token is not None:
            self.cancellation_token.cancel(reason)

    def flush_stream_buffer(self) -> int:
        flushed = len(self.stream_buffer)
        self.stream_buffer.clear()
        return flushed


class StreamCancellationToken:
    """Per-run cancellation handle used to break the streaming loop before a rewrite."""

    def __init__(self, run_id: str) -> None:
        self.run_id = run_id
        self.reason: str | None = None
        self._event = asyncio.Event()

    def cancel(self, reason: str = "cancelled") -> None:
        self.reason = reason
        self._event.set()

    @property
    def cancelled(self) -> bool:
        return self._event.is_set()


@app.get("/")
async def index() -> FileResponse:
    return FileResponse(STATIC_DIR / "index.html")


@app.get("/health")
async def health() -> dict[str, str]:
    return {"status": "ok", "service": "AuraTrack"}


@app.websocket("/ws/auratrack/{session_id}")
async def auratrack_socket(websocket: WebSocket, session_id: str) -> None:
    await websocket.accept()
    session = RuntimeSession(session_id=session_id)
    await websocket.send_json({"type": "session_open", "session_id": session_id})

    try:
        while True:
            raw_message = await websocket.receive_text()
            message = json.loads(raw_message)
            message_type = message.get("type")

            if message_type == "start":
                await start_session(websocket, session, message.get("prompt", ""))
            elif message_type == "steer":
                await steer_session(websocket, session, message)
            elif message_type == "evaporate":
                await cancel_stream(session, websocket, notify=True)
                await websocket.send_json({"type": "mesh_evaporated", "mesh": evaporate_session_mesh(session.session_id)})
            elif message_type == "ping":
                await websocket.send_json({"type": "pong"})
            else:
                await websocket.send_json({"type": "error", "message": f"Unknown message type: {message_type}"})
    except WebSocketDisconnect:
        await cancel_stream(session)
        evaporate_session_mesh(session.session_id)


async def start_session(websocket: WebSocket, session: RuntimeSession, prompt: str) -> None:
    await cancel_stream(session)
    evaporate_session_mesh(session.session_id)
    session.prompt = prompt.strip() or "Build AuraTrack and expose steering hooks."
    session.token_log.reset()
    session.packets = {}
    session.stream_buffer.clear()
    session.run_id = uuid.uuid4().hex

    await websocket.send_json({"type": "router_started", "session_id": session.session_id})
    session.sub_context = await router_layer.analyze_sub_perceptions(session.prompt)
    await websocket.send_json({"type": "router_complete", "sub_context": session.sub_context})

    mesh = memory_controller.manifest_temporary_mesh(session.session_id, session.sub_context)
    lifecycle_logger.manifested(session.session_id, mesh)
    await websocket.send_json({"type": "mesh_manifested", "mesh": mesh})

    session.system_context = f"You are guided by these hidden sub-perceptions: {session.sub_context}"
    session.stream_task = asyncio.create_task(stream_run(websocket, session))


async def steer_session(websocket: WebSocket, session: RuntimeSession, message: dict[str, Any]) -> None:
    token_id = int(message["token_id"])
    alternative = message["alternative"]

    await cancel_stream(session, websocket, notify=True)
    session.run_id = uuid.uuid4().hex
    try:
        rewrite = session.token_log.rewrite_at(token_id, alternative, session.run_id)
    except IndexError as exc:
        await websocket.send_json({"type": "error", "message": str(exc)})
        return

    session.packets = session.token_log.packets()

    if memory_controller.get_mesh(session.session_id) is None:
        mesh = memory_controller.manifest_temporary_mesh(session.session_id, session.sub_context)
        lifecycle_logger.manifested(session.session_id, mesh)
        await websocket.send_json({"type": "mesh_manifested", "mesh": mesh})

    await websocket.send_json(
        {
            "type": "history_rewritten",
            "token_id": token_id,
            "replacement": rewrite.replacement,
            "alternative": alternative,
            "history": rewrite.history,
            "text": rewrite.text,
            "run_id": session.run_id,
        }
    )
    session.stream_task = asyncio.create_task(stream_run(websocket, session))


async def stream_run(websocket: WebSocket, session: RuntimeSession) -> None:
    run_id = session.run_id
    cancellation_token = session.begin_stream()
    packet_source = interceptor.stream_with_steering_hooks(
        prompt=session.prompt,
        system_context=session.system_context,
        committed_history=session.token_log.history(),
        run_id=run_id,
    )
    try:
        async for packet in stream_with_cancellation(packet_source, cancellation_token):
            if run_id != session.run_id or cancellation_token.cancelled:
                return
            session.stream_buffer.append(packet)
            session.token_log.append_packet(packet)
            packet["committed_history"] = session.token_log.history()
            packet["text"] = session.token_log.text()
            session.packets[packet["token_id"]] = packet
            await websocket.send_json({"type": "token", "packet": packet})
            session.stream_buffer.clear()

        if run_id == session.run_id:
            await websocket.send_json({"type": "stream_complete", "run_id": run_id, "text": session.token_log.text()})
            lifecycle_logger.stream_completion_successful(session.session_id)
            await asyncio.sleep(0.2)
            await websocket.send_json({"type": "mesh_evaporated", "mesh": evaporate_session_mesh(session.session_id)})
    except asyncio.CancelledError:
        raise
    except WebSocketDisconnect:
        session.request_cancel("disconnect")
        return


async def cancel_stream(session: RuntimeSession, websocket: WebSocket | None = None, notify: bool = False) -> None:
    task = session.stream_task
    if task is None or task.done():
        session.stream_task = None
        session.flush_stream_buffer()
        return

    run_id = session.run_id
    session.request_cancel("steer")
    task.cancel()
    try:
        await task
    except asyncio.CancelledError:
        pass
    finally:
        session.stream_task = None

    flushed_packets = session.flush_stream_buffer()
    if notify and websocket is not None:
        await websocket.send_json({"type": "stream_cancelled", "run_id": run_id, "flushed_packets": flushed_packets})


async def stream_with_cancellation(
    packet_source: AsyncIterator[dict[str, Any]],
    cancellation_token: StreamCancellationToken,
) -> AsyncIterator[dict[str, Any]]:
    try:
        async for packet in packet_source:
            if cancellation_token.cancelled:
                break
            yield packet
            if cancellation_token.cancelled:
                break
    finally:
        aclose = getattr(packet_source, "aclose", None)
        if aclose is not None:
            with contextlib.suppress(Exception):
                await aclose()


def evaporate_session_mesh(session_id: str) -> dict[str, Any]:
    result = memory_controller.evaporate_mesh(session_id)
    lifecycle_logger.evaporated(session_id, result)
    return result


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("auratrack.server:app", host="127.0.0.1", port=8000, reload=True)
